import cv2
import numpy as np


class LineDetector:
    def __init__(self, alpha=0.8, min_pixels=10):
        self.prev_centers = None
        self.prev_centerline = None
        self.alpha = alpha
        self.min_pixels = min_pixels

    def process(self, frame, thresholds):
        """
        Возвращает: (ресайз кадра, маска, линия)
        """
        # --- ИЗМЕНЕНИЕ 1: Увеличили размер ---
        # Раньше делили на 2 (//2), теперь умножаем на 0.8 (80% от оригинала)
        # Если нужно еще больше — поставь 1.0
        scale_percent = 0.8
        width = int(frame.shape[1] * scale_percent)
        height = int(frame.shape[0] * scale_percent)
        dim = (width, height)

        frame_resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # 2. Формирование маски по цвету
        lower = (thresholds["LB"], thresholds["LG"], thresholds["LR"])
        upper = (thresholds["HB"], thresholds["HG"], thresholds["HR"])

        mask = cv2.inRange(frame_resized, lower, upper)
        mask = cv2.medianBlur(mask, 5)

        # 3. Оставляем самую большую область
        clean_mask = self._keep_largest_component(mask)

        # 4. Обрезка (ROI)
        h, w = clean_mask.shape
        top_cut = h // 2
        bot_cut = h // 10

        clean_mask[:top_cut, :] = 0
        if bot_cut > 0:
            clean_mask[h - bot_cut:, :] = 0

        # 5. Закрываем дырки
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, close_kernel)

        # 6. Строим линию
        centerline = self._calculate_centerline(clean_mask, top_cut, h - bot_cut)

        return frame_resized, mask, centerline

    def _keep_largest_component(self, mask):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_label = 1 + areas.argmax()
            biggest_mask = (labels == max_label).astype("uint8") * 255
            return biggest_mask
        else:
            return (mask > 0).astype("uint8") * 255

    def _calculate_centerline(self, mask, start_y, end_y):
        h, w = mask.shape
        centerline = np.zeros_like(mask, dtype=np.uint8)
        current_centers = -np.ones(h, dtype=np.int32)

        points_found = False

        for y in range(start_y, end_y):
            row = mask[y]
            xs = np.flatnonzero(row)
            if xs.size == 0:
                continue

            points_found = True
            cx = int((int(xs[0]) + int(xs[-1])) / 2)

            if self.prev_centers is not None and self.prev_centers[y] >= 0:
                cx = int(round(self.alpha * self.prev_centers[y] + (1.0 - self.alpha) * cx))

            centerline[y, cx] = 255
            current_centers[y] = cx

        if np.count_nonzero(centerline) < self.min_pixels and self.prev_centerline is not None:
            return self.prev_centerline

        if points_found:
            self.prev_centerline = centerline.copy()
            self.prev_centers = current_centers.copy()

        return centerline
