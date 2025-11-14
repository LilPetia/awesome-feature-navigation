import cv2
import numpy as np

# путь к видео
video_path = "Left_cam.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Не удалось открыть видео: {video_path}")

# окно для ползунков
cv2.namedWindow("controls")


def nothing(_):
    pass


# создаём ползунки для каждого канала BGR
cv2.createTrackbar("LB_thresh", "controls", 0, 255, nothing)
cv2.createTrackbar("LG_thresh", "controls", 0, 255, nothing)
cv2.createTrackbar("LR_thresh", "controls", 215, 255, nothing)
cv2.createTrackbar("HB_thresh", "controls", 148, 255, nothing)
cv2.createTrackbar("HG_thresh", "controls", 234, 255, nothing)
cv2.createTrackbar("HR_thresh", "controls", 255, 255, nothing)

prev_centers = None      # центры по строкам с прошлого кадра
prev_centerline = None   # картинка линии с прошлого кадра

alpha = 0.8              # насколько сильно сглаживаем по времени (0..1)
min_pixels = 10          # если линия почти исчезла — берём прошлую

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # уменьшаем картинку
    h0, w0, _ = frame.shape
    frame = cv2.resize(frame, (w0 // 2, h0 // 2))

    # читаем текущие значения ползунков
    lb = cv2.getTrackbarPos("LB_thresh", "controls")
    lg = cv2.getTrackbarPos("LG_thresh", "controls")
    lr = cv2.getTrackbarPos("LR_thresh", "controls")
    hb = cv2.getTrackbarPos("HB_thresh", "controls")
    hg = cv2.getTrackbarPos("HG_thresh", "controls")
    hr = cv2.getTrackbarPos("HR_thresh", "controls")

    # бинарная маска по цвету
    mask = cv2.inRange(frame, (lb, lg, lr), (hb, hg, hr))
    mask = cv2.medianBlur(mask, 5)  # чуть сгладили шум

    # ---------- оставляем только самую большую связную компоненту ----------
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]  # площади компонент (кроме фона)
        max_label = 1 + areas.argmax()
        biggest_mask = (labels == max_label).astype("uint8") * 255
    else:
        biggest_mask = (mask > 0).astype("uint8") * 255
    # ----------------------------------------------------------------------

    h, w = biggest_mask.shape

    # --- обрезаем по высоте толстую полосу ДО построения линии ---
    top_cut = h // 2       # верхняя 1/2
    bot_cut = h // 10      # нижняя 1/10

    biggest_mask[:top_cut, :] = 0
    if bot_cut > 0:
        biggest_mask[h - bot_cut:, :] = 0

    # слегка закрываем дырки, чтобы полоса была сплошной
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    biggest_mask = cv2.morphologyEx(biggest_mask, cv2.MORPH_CLOSE, close_kernel)

    # --- строим центральную линию по каждой строке ---
    centerline = np.zeros_like(biggest_mask, dtype=np.uint8)
    current_centers = -np.ones(h, dtype=np.int32)

    for y in range(top_cut, h - bot_cut):
        row = biggest_mask[y]
        xs = np.flatnonzero(row)
        if xs.size == 0:
            continue

        # центр между левым и правым краем полосы
        cx = int((int(xs[0]) + int(xs[-1])) / 2)

        # сглаживание по времени
        if prev_centers is not None and prev_centers[y] >= 0:
            cx = int(round(alpha * prev_centers[y] + (1.0 - alpha) * cx))

        # ставим один белый пиксель
        centerline[y, cx] = 255
        current_centers[y] = cx

    # если вдруг линия почти пропала — используем прошлый кадр
    if np.count_nonzero(centerline) < min_pixels and prev_centerline is not None:
        centerline = prev_centerline.copy()
        current_centers = prev_centers.copy()
    else:
        prev_centerline = centerline.copy()
        prev_centers = current_centers.copy()

    # показываем
    cv2.imshow("original", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("best mask", centerline)

    # выход по 'q'
    if cv2.waitKey(80) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()