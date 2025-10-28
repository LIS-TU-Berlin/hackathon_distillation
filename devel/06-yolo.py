import torch
import robotic as ry
import hackathon_distillation as hack
import cv2
import numpy as np
from ultralytics import YOLO

class Robot:
    def __init__(self, real=False):
        self.S = hack.Scene()
        self.q0 = self.S.C.getJointState()
        self.bot = ry.BotOp(C=self.S.C, useRealRobot=real)
        
    def run(self):

        #model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
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

if __name__ == "__main__":

    R = Robot(real=True)
    R.run()
