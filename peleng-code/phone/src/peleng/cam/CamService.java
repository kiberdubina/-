package peleng.cam;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.os.BatteryManager;
import android.os.IBinder;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.List;

public class CamService extends Service implements Camera.PreviewCallback {
    static final int PORT = 8080;
    private static final String CHANNEL = "peleng-cam";

    private Camera camera;
    private SurfaceTexture texture;
    private ServerSocket server;
    private int width = 640;
    private int height = 480;
    private long lastFrameMs;
    private volatile byte[] latestJpeg;
    private final Object clientsLock = new Object();
    private int clients;
    private long lastClientMs;
    private final long startedMs = System.currentTimeMillis();
    private volatile boolean running = true;
    private boolean watchingPlug;
    private final BroadcastReceiver unplug = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            if (intent.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1) == 0) {
                stopSelf();
            }
        }
    };

    @Override
    public void onCreate() {
        super.onCreate();
        NotificationManager manager = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
        NotificationChannel channel = new NotificationChannel(
                CHANNEL, "Камера Peleng", NotificationManager.IMPORTANCE_LOW);
        manager.createNotificationChannel(channel);
        Intent open = new Intent(this, MainActivity.class);
        PendingIntent pending = PendingIntent.getActivity(this, 0, open, PendingIntent.FLAG_UPDATE_CURRENT);
        Notification notification = new Notification.Builder(this, CHANNEL)
                .setSmallIcon(android.R.drawable.ic_menu_camera)
                .setContentTitle("Камера Peleng включена")
                .setContentText("Снимает, пока кабель в управляющем ноутбуке")
                .setContentIntent(pending)
                .setOngoing(true)
                .build();
        startForeground(1, notification);
        Intent plugged = registerReceiver(unplug, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
        watchingPlug = true;
        if (plugged != null && plugged.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1) == 0) {
            stopSelf();
            return;
        }
        openCamera();
        Thread serverThread = new Thread(this::serve, "peleng-mjpeg");
        serverThread.setDaemon(true);
        serverThread.start();
        Thread watchdog = new Thread(this::watchClients, "peleng-watch");
        watchdog.setDaemon(true);
        watchdog.start();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        return START_NOT_STICKY;
    }

    @Override
    public void onDestroy() {
        running = false;
        if (watchingPlug) {
            unregisterReceiver(unplug);
            watchingPlug = false;
        }
        if (server != null) {
            try {
                server.close();
            } catch (IOException ignored) {
            }
        }
        closeCamera();
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    private void openCamera() {
        try {
            camera = Camera.open();
            Camera.Parameters parameters = camera.getParameters();
            Camera.Size size = pick(parameters.getSupportedPreviewSizes());
            if (size != null) {
                width = size.width;
                height = size.height;
                parameters.setPreviewSize(width, height);
            }
            parameters.setPreviewFormat(ImageFormat.NV21);
            camera.setParameters(parameters);
            texture = new SurfaceTexture(1);
            camera.setPreviewTexture(texture);
            camera.setPreviewCallback(this);
            camera.startPreview();
        } catch (RuntimeException | IOException error) {
            closeCamera();
        }
    }

    private Camera.Size pick(List<Camera.Size> sizes) {
        if (sizes == null || sizes.isEmpty()) {
            return null;
        }
        Camera.Size best = sizes.get(0);
        int bestScore = Integer.MAX_VALUE;
        for (Camera.Size size : sizes) {
            int score = Math.abs(size.width - 640) + Math.abs(size.height - 480);
            if (score < bestScore) {
                best = size;
                bestScore = score;
            }
        }
        return best;
    }

    @Override
    public void onPreviewFrame(byte[] data, Camera camera) {
        long now = System.currentTimeMillis();
        if (data == null || now - lastFrameMs < 80) {
            return;
        }
        lastFrameMs = now;
        YuvImage image = new YuvImage(data, ImageFormat.NV21, width, height, null);
        ByteArrayOutputStream jpeg = new ByteArrayOutputStream();
        if (!image.compressToJpeg(new Rect(0, 0, width, height), 70, jpeg)) {
            return;
        }
        latestJpeg = jpeg.toByteArray();
    }

    private void closeCamera() {
        if (camera == null) {
            return;
        }
        camera.setPreviewCallback(null);
        camera.stopPreview();
        camera.release();
        camera = null;
        if (texture != null) {
            texture.release();
            texture = null;
        }
    }

    private void watchClients() {
        while (running) {
            try {
                Thread.sleep(400);
            } catch (InterruptedException ignored) {
                return;
            }
            synchronized (clientsLock) {
                long now = System.currentTimeMillis();
                boolean idleTooLong = clients == 0 && lastClientMs == 0 && now - startedMs > 20000;
                boolean clientLeft = clients == 0 && lastClientMs > 0 && now - lastClientMs > 2000;
                if (idleTooLong || clientLeft) {
                    stopSelf();
                    return;
                }
            }
        }
    }

    private void serve() {
        try {
            server = new ServerSocket(PORT, 4, InetAddress.getByName("127.0.0.1"));
            while (running) {
                handle(server.accept());
            }
        } catch (IOException ignored) {
        }
    }

    private void handle(Socket client) {
        boolean counted = false;
        try {
            client.setTcpNoDelay(true);
            synchronized (clientsLock) {
                clients++;
                lastClientMs = System.currentTimeMillis();
                counted = true;
            }
            OutputStream out = client.getOutputStream();
            byte[] request = new byte[512];
            client.getInputStream().read(request);
            if (!new String(request).contains("/video")) {
                byte[] body = "Peleng Cam\n".getBytes();
                out.write(("HTTP/1.0 200 OK\r\nContent-Length: " + body.length + "\r\n\r\n").getBytes());
                out.write(body);
                client.close();
                return;
            }
            out.write((
                    "HTTP/1.0 200 OK\r\n"
                            + "Connection: close\r\n"
                            + "Cache-Control: no-cache\r\n"
                            + "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n"
            ).getBytes());
            while (!client.isClosed()) {
                byte[] jpeg = latestJpeg;
                if (jpeg == null) {
                    Thread.sleep(40);
                    continue;
                }
                out.write(("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + jpeg.length + "\r\n\r\n").getBytes());
                out.write(jpeg);
                out.write("\r\n".getBytes());
                out.flush();
                Thread.sleep(40);
            }
        } catch (Exception ignored) {
        } finally {
            if (counted) {
                synchronized (clientsLock) {
                    clients = Math.max(0, clients - 1);
                    lastClientMs = System.currentTimeMillis();
                }
            }
            try {
                client.close();
            } catch (IOException ignored) {
            }
        }
    }
}
