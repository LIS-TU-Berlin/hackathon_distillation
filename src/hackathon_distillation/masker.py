import cv2
import numpy as np

class Masker:
    def __init__(
        self,
        known_rgb=(0.2, 0.5, 0.6),  # normalized [0..1]
        hue_tol_deg=10,            # how far hue can drift
        sat_range=(0.2, 1.0),      # allowed saturation range
        val_range=(0.1, 1.0),      # allowed value/brightness range
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

    def color_mask(self, img_rgb: np.ndarray):
        """
        Returns a binary mask (H,W) uint8 where 255 = pixel whose color
        matches the known object color (robust-ish to lighting).
        """

        # 1. ensure we have uint8 and convert RGB->BGR for OpenCV
        img_u8 = self._to_uint8_rgb(img_rgb)
        img_bgr = img_u8[..., ::-1]  # RGB -> BGR

        # 2. convert frame to HSV
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

        H = hsv[..., 0]  # [0,179]
        S = hsv[..., 1]  # [0,255]
        V = hsv[..., 2]  # [0,255]

        # 3. compute hue distance with wrap-around
        # hue is circular (0 ~ 180). We'll do shortest distance on a circle.
        dh = np.abs(H - self.target_h)
        dh = np.minimum(dh, 180.0 - dh)

        # 4. threshold by hue closeness, saturation range, value range
        hue_ok = dh <= self.hue_tol_deg * (180.0/360.0) * 360.0/2.0
        # Wait, that looks scary. Let's explain & simplify:

        # cv2 hue range is 0..179 which corresponds to 0..360 degrees
        # So 1 "hue unit" in cv2 HSV = 2 degrees.
        # If we want hue_tol_deg in real degrees:
        # allowed_hue_diff_in_cv2_units = hue_tol_deg / 2
        allowed_hue_diff_cv2 = self.hue_tol_deg / 2.0
        hue_ok = dh <= allowed_hue_diff_cv2

        sat_ok = (S >= self.sat_min*255.0) & (S <= self.sat_max*255.0)
        val_ok = (V >= self.val_min*255.0) & (V <= self.val_max*255.0)

        mask_bool = hue_ok & sat_ok & val_ok

        # 5. convert to uint8 [0,255]
        mask = np.zeros_like(H, dtype=np.uint8)
        mask[mask_bool] = 255
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

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
