import cv2
import numpy as np

d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
sx, sy, sq, mk = 5, 7, 0.04, 0.03
board = cv2.aruco.CharucoBoard((sx, sy), sq, mk, d)
det = cv2.aruco.CharucoDetector(board)
W, H = sx * sq, sy * sq
bpx = 280
bimg = board.generateImage((sx * bpx, sy * bpx))
if bimg.ndim == 2:
    bimg = cv2.cvtColor(bimg, cv2.COLOR_GRAY2BGR)
bh, bw = bimg.shape[:2]
src = np.float32([[0, 0], [bw - 1, 0], [bw - 1, bh - 1], [0, bh - 1]])
obj4 = np.float32([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]])
iw, ih = 960, 540
K = np.array([[900.0, 0, iw / 2], [0, 900, ih / 2], [0, 0, 1]])
dist = np.zeros(5)


def look_at(center, target):
    center = np.asarray(center, np.float64)
    target = np.asarray(target, np.float64)
    forward = target - center
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, up))) > 0.92:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    R = np.stack([right, down, forward], axis=0)
    t = (-R @ center).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    return rvec, t


def render(rvec, tvec):
    imgpts, _ = cv2.projectPoints(obj4, rvec, tvec, K, dist)
    dst = imgpts.reshape(-1, 2).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    canvas = np.full((ih, iw, 3), 48, np.uint8)
    warped = cv2.warpPerspective(bimg, M, (iw, ih), flags=cv2.INTER_CUBIC)
    mask = cv2.warpPerspective(np.full(bimg.shape[:2], 255, np.uint8), M, (iw, ih))
    canvas[mask > 10] = warped[mask > 10]
    return canvas


center_board = np.array([W / 2, H / 2, 0.0])
cams = {
    "A": np.array([W / 2, -0.22, 0.62]),
    "B": np.array([-0.28, H / 2, 0.58]),
    "C": np.array([W + 0.25, H + 0.12, 0.64]),
}
poses = {k: look_at(c, center_board + np.array([0.0, 0.0, 0.02])) for k, c in cams.items()}

adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
adet = cv2.aruco.ArucoDetector(adict)
# Control: camera on the -Z side, axes aligned with the board image.
r_front, _ = cv2.Rodrigues(np.eye(3))
t_front = np.array([[-W / 2], [-H / 2], [0.55]], np.float64)
img_front = render(r_front, t_front)
cv2.imwrite("_probe_front.png", img_front)
cc, ids, _, _ = det.detectBoard(img_front)
corners, mids2, _ = adet.detectMarkers(img_front)
print(
    "front charuco",
    0 if ids is None else len(ids),
    "markers",
    0 if mids2 is None else len(mids2),
)
for name, (rv, tv) in poses.items():
    img = render(rv, tv)
    cv2.imwrite(f"_probe_{name}.png", img)
    cc, ids, mc, mids = det.detectBoard(img)
    n = 0 if ids is None else len(ids)
    corners, mids2, _ = adet.detectMarkers(img)
    pts, _ = cv2.projectPoints(obj4, rv, tv, K, dist)
    print(name, "charuco", n, "markers", 0 if mids2 is None else len(mids2), "quad", np.round(pts.reshape(-1, 2), 1))

rng = np.random.default_rng(2)
views = []
rv, tv = poses["A"]
Rc, _ = cv2.Rodrigues(rv)
for i in range(12):
    ang = np.array(
        [
            rng.uniform(-0.4, 0.4),
            rng.uniform(-0.4, 0.4),
            rng.uniform(-0.25, 0.25),
        ]
    )
    Rb, _ = cv2.Rodrigues(ang)
    tb = np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), 0.0])
    Rrel = Rc @ Rb
    trel = (Rc @ tb).reshape(3, 1) + tv
    rrel, _ = cv2.Rodrigues(Rrel)
    img = render(rrel, trel)
    cc, ids, _, _ = det.detectBoard(img)
    n = 0 if ids is None else len(ids)
    ok = n >= 8 and ids is not None and not board.checkCharucoCornersCollinear(ids)
    print("view", i, n, "keep", ok)
    if ok:
        obj, imgp = board.matchImagePoints(cc, ids)
        views.append((obj.astype(np.float32), imgp.astype(np.float32)))

print("views", len(views))
rms, K2, dist2, rvecs, tvecs = cv2.calibrateCamera(
    [v[0] for v in views],
    [v[1] for v in views],
    (iw, ih),
    None,
    None,
    flags=cv2.CALIB_FIX_K3,
)
print("rms", round(float(rms), 3))
print("K", np.round(K2, 1))
print("dist", np.round(dist2.ravel(), 4))

img = render(rv, tv)
cc, ids, _, _ = det.detectBoard(img)
obj, imgp = board.matchImagePoints(cc, ids)
ok, r_est, t_est = cv2.solvePnP(obj, imgp, K2, dist2)
print("pnp", ok, "dt_cm", round(float(np.linalg.norm(t_est.ravel() - tv.ravel()) * 100), 2))
