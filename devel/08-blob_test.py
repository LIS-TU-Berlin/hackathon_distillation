import torch
import robotic as ry
import cv2
import numpy as np
from ultralytics import YOLO, SAM, FastSAM
from typing import List, Tuple, Any
from einops import rearrange
import time

import hackathon_distillation as hack
from pathlib import Path

def blob(img:np.ndarray):
        
        # Setup SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.minCircularity = 0.4
        params.filterByArea = True
        params.minArea = 400
        
        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(img)

        mask = np.zeros_like(img, dtype=np.uint8)

        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            r = int(kp.size / 2)
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

        return mask

if __name__ == "__main__":
    
    # Path to folder with rgb images
    img_path = "/home/sayantan/Downloads/imgs/output/"

    pngs = sorted(Path(img_path).glob("*.png"))
    if not pngs:
        print(f"No PNG files found in {img_path}")
        img = None
    else:
        for png in pngs:
            img = cv2.imread(str(png), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Failed to load image: {img_path}")
            else:
                mask = blob(img)
                cv2.imshow(f"{png}: image", img)
                cv2.imshow(f"{png}:mask", mask)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
