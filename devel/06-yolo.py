import torch
import robotic as ry
import cv2
import numpy as np
from ultralytics import YOLO, SAM, FastSAM
from typing import List, Tuple, Any
from einops import rearrange

import hackathon_distillation as hack

class Robot:
    def __init__(self, real=False):
        self.S = hack.Scene()
        self.q0 = self.S.C.getJointState()
        self.bot = ry.BotOp(C=self.S.C, useRealRobot=real)
        
    def run(self):

        model = YOLO("yolo11n.pt", task='detect')

        rgb, depth = self.bot.getImageAndDepth('cameraWrist')
        D = hack.DataPlayer(rgb, depth)

        while True:
            key = self.bot.sync(self.S.C, .1)
            if key==ord('q'):
                break

            rgb, depth = self.bot.getImageAndDepth('cameraWrist')

            depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)

            cv2.imwrite('depth.png', depth_vis)

            # Dummy
            imgs = ['depth.png']
            results = model(imgs)

            results[0].plot()

            D.update(rgb, depth)

            break

class Masker:

    def __init__(self, detect_model_pth:str="yolo11m.pt", segment_model_pth:str="yolo11m-seg.pt", bot:ry.BotOp=None):
        self.bot = bot
        self.detect_model = YOLO(detect_model_pth)
        self.segment_model = FastSAM(segment_model_pth)

    def detect(self, img:np.ndarray):
        """
        
        """
        assert img.ndim == 3 and img.shape[2] == 3

        results = self.detect_model([img])
        for r in results:
            r.show() 

    def segment(self, img:np.ndarray):
        """
        
        """
        results = self.segment_model([img])
        for r in results:
            r.show() 

            for m in r.masks:
                d:np.ndarray = m.data.detach().cpu().numpy()
                print(type(d), d.shape, d.dtype)
                d = d[0]*255
                print(np.max(d), np.min(d))
                cv2.imshow("depth", d)
                cv2.waitKey(0)



if __name__ == "__main__":

    m = Masker()

    img = cv2.imread('/home/sayantan/Pictures/test/objects.jpg', cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("image not found")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #m.detect_yolo(rgb)
    m.segment(rgb)

