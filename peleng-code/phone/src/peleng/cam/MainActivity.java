package peleng.cam;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

public class MainActivity extends Activity {
    private TextView status;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(48, 48, 48, 48);
        status = new TextView(this);
        status.setTextSize(20);
        status.setText("Нажмите «Разрешить», когда телефон спросит про камеру.");
        root.addView(status);

        Button stop = new Button(this);
        stop.setText("Выключить камеру");
        stop.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                stopService(new Intent(MainActivity.this, CamService.class));
                status.setText("Камера выключена.");
            }
        });
        LinearLayout.LayoutParams gap = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        gap.topMargin = 48;
        gap.gravity = Gravity.CENTER_HORIZONTAL;
        root.addView(stop, gap);
        setContentView(root);

        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 1);
            return;
        }
        begin();
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] permissions, int[] results) {
        if (results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) {
            begin();
            return;
        }
        status.setText("Без разрешения камера не включится. Это нормальная защита телефона.");
    }

    private void begin() {
        startForegroundService(new Intent(this, CamService.class));
        status.setText(
                "Камерой управляет ноутбук.\n\n"
                        + "Съёмка идёт, пока кабель воткнут в него.\n"
                        + "Наверху горит уведомление. Кнопка ниже гасит камеру сразу."
        );
    }
}
