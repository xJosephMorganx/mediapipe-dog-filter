import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Wayland config
os.environ.pop("QT_QPA_PLATFORM", None)
os.environ["GDK_BACKEND"] = "wayland"

import cv2
import math
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ANCHO_INICIAL = 960
ALTO_INICIAL = 720

# =========================
# Modelo Mediapipe
# =========================
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

# =========================
# Cargar imágenes
# =========================
def cargar_filtro(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] < 4:
        print(f"Error cargando {path}")
        exit()
    return img

ears = cargar_filtro("dog_ears.png")
nose = cargar_filtro("dog_nose.png")
tongue = cargar_filtro("dog_tongue.png")

# =========================
# Overlay robusto
# =========================
def overlay_png(bg, overlay, x, y, w, h):
    if w <= 0 or h <= 0:
        return bg

    overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)

    bh, bw = bg.shape[:2]

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, bw), min(y + h, bh)

    if x1 >= x2 or y1 >= y2:
        return bg

    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    overlay_crop = overlay[oy1:oy2, ox1:ox2]
    alpha = overlay_crop[:, :, 3] / 255.0

    for c in range(3):
        bg[y1:y2, x1:x2, c] = (
            alpha * overlay_crop[:, :, c] +
            (1 - alpha) * bg[y1:y2, x1:x2, c]
        )

    return bg

# =========================
# Cámara
# =========================
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Filtro Perro CachyOS", cv2.WINDOW_NORMAL)

# Primer resize (puede que Wayland lo ignore)
cv2.resizeWindow("Filtro Perro CachyOS", ANCHO_INICIAL, ALTO_INICIAL)

timestamp = 0
frame_count = 0

print("Listo. ESC para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    #  Solo forzar tamaño en los primeros frames
    frame_count += 1
    if frame_count <= 15:
        cv2.resizeWindow("Filtro Perro CachyOS", ANCHO_INICIAL, ALTO_INICIAL)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect_for_video(mp_image, timestamp)
    timestamp += 1

    if result.face_landmarks:
        lm = result.face_landmarks[0]

        def pt(i):
            return int(lm[i].x * w), int(lm[i].y * h)

        left, right = pt(234), pt(454)
        forehead = pt(10)
        nose_tip = pt(1)

        m_top, m_bot = pt(13), pt(14)
        m_l, m_r = pt(78), pt(308)

        face_width = math.dist(left, right)
        mouth_open = math.dist(m_top, m_bot)
        mouth_width = max(math.dist(m_l, m_r), 1)

        # Orejas
        ew = int(face_width * 1.5)
        eh = int(ew * ears.shape[0] / ears.shape[1])
        frame = overlay_png(frame, ears,
                            int(forehead[0] - ew/2),
                            int(forehead[1] - eh*0.8),
                            ew, eh)

        # Nariz
        nw = int(face_width * 0.6)
        nh = int(nw * nose.shape[0] / nose.shape[1])
        frame = overlay_png(frame, nose,
                            int(nose_tip[0] - nw/2),
                            int(nose_tip[1] - face_width*0.14),
                            nw, nh)

        # Lengua
        if (mouth_open / mouth_width) > 0.35:
            tw = int(mouth_width * 1.6)
            th = int(tw * tongue.shape[0] / tongue.shape[1])
            frame = overlay_png(frame, tongue,
                                int((m_l[0]+m_r[0])/2 - tw/2),
                                int(m_bot[1] - face_width*0.1),
                                tw, th)

            cv2.putText(frame, "Lengua", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Filtro Perro CachyOS", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
time.sleep(0.1)
