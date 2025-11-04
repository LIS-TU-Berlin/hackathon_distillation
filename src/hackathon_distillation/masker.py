import cv2
import numpy as np

class Masker:
    def __init__(
        self,
        # known_rgb=(0.2, 0.5, 0.6),  # blue
        # hue_tol_deg=10,
        # sat_range=(0.2, 1.0),
        # val_range=(0.1, 1.0),
        known_rgb=(1.0, 0.419, 0.),  # orange
        hue_tol_deg=15,
        sat_range=(0.5, 1.0),
        val_range=(0.5, 1.0),
    ):
        # ----- blob detector init (unchanged, keep if you still want blob()) -----
        params = cv2.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.minCircularity = 0.4
        params.filterByArea = True
        params.minArea = 400
        params.maxArea = 2e5
        params.minThreshold = 1
        params.maxThreshold = 150
        params.thresholdStep = 10
        self.detector = cv2.SimpleBlobDetector_create(params)

        # ----- store thresholds for color segmentation -----
        self.hue_tol_deg = hue_tol_deg
        self.sat_min, self.sat_max = sat_range
        self.val_min, self.val_max = val_range

        # convert known_rgb (float tuple 0..1) into HSV once
        rgb_arr = np.array([[list(known_rgb)]], dtype=np.float32)  # shape (1,1,3)
        rgb_arr_255 = (rgb_arr * 255.0).astype(np.uint8)  # cv2 expects 0..255 uint8
        # cv2 assumes BGR, so we need to swap channels: RGB -> BGR
        bgr_arr_255 = rgb_arr_255[..., ::-1]
        hsv_arr = cv2.cvtColor(bgr_arr_255, cv2.COLOR_BGR2HSV).astype(np.float32)
        # hsv: H in [0,179], S in [0,255], V in [0,255] for uint8 OpenCV HSV
        self.target_h = hsv_arr[0,0,0]  # hue
        self.target_s = hsv_arr[0,0,1]
        self.target_v = hsv_arr[0,0,2]

    def _to_uint8_rgb(self, img):
        if img.dtype == np.float32 or img.dtype == np.float64:
            vmin = float(img.min())
            vmax = float(img.max())
            if vmax <= 1.0 and vmin >= 0.0:
                img_u8 = (img * 255.0).clip(0,255).astype(np.uint8)
            else:
                img_u8 = img.clip(0,255).astype(np.uint8)
        else:
            img_u8 = img.astype(np.uint8)
        return img_u8

    def color_mask(self, img_rgb: np.ndarray, use_clahe: bool = True, keep_largest: bool = True):
        """
        Returns a clean uint8 mask (H,W) with 255 on the blue ball.
        Steps: RGB->HSV, (optional) CLAHE on V, hue gate around known color,
            S/V range check, open+close morphology, keep largest CC.
        """
        # 1) ensure uint8 and convert RGB->BGR for OpenCV
        img_u8 = self._to_uint8_rgb(img_rgb)
        img_bgr = img_u8[..., ::-1]  # RGB -> BGR
        # 2) HSV (+ optional CLAHE on V to stabilize exposure)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        if use_clahe:
            H, S, V = cv2.split(hsv)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            V = clahe.apply(V)
            hsv = cv2.merge([H, S, V])
        # float arrays for math
        hsv_f = hsv.astype(np.float32)
        H = hsv_f[..., 0]  # [0,179]
        S = hsv_f[..., 1]  # [0,255]
        V = hsv_f[..., 2]  # [0,255]
        # 3) shortest circular hue distance (OpenCV hue units: 0..179, i.e., 2°/unit)
        dh = np.abs(H - self.target_h)
        dh = np.minimum(dh, 180.0 - dh)
        # hue tolerance in OpenCV units (deg/2)
        allowed_hue_diff_cv2 = float(self.hue_tol_deg) / 2.0
        hue_ok = dh <= allowed_hue_diff_cv2
        # 4) S/V gates
        sat_ok = (S >= self.sat_min * 255.0) & (S <= self.sat_max * 255.0)
        val_ok = (V >= self.val_min * 255.0) & (V <= self.val_max * 255.0)
        mask_bool = hue_ok & sat_ok & val_ok
        # 5) morphology (ellipse kernel works well on balls)
        mask = np.zeros(H.shape, dtype=np.uint8)
        mask[mask_bool] = 255
        k = 5 if max(mask.shape) > 400 else 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        # 6) keep largest connected component (drop small blue clutter)
        if keep_largest:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num_labels > 1:
                # label 0 is background
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                keep = (labels == largest_label)
                mask[...] = 0
                mask[keep] = 255
        return mask

    def blob(self, img: np.ndarray):
        img_u8 = self._to_uint8_rgb(img)
        gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
        keypoints = self.detector.detect(gray)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        detected = False
        for kp in keypoints:
            detected = True
            x, y = int(kp.pt[0]), int(kp.pt[1])
            r = int(kp.size / 2)
            cv2.circle(mask, (x, y), r, 255, -1)

        return mask, detected
