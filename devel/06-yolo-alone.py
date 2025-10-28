import torch
import robotic as ry
import hackathon_distillation as hack
import cv2
import numpy as np

# This works, YOLO detects objects in depth images saved from the robot camera
# Running together with botop does not work, botop waits foreoever for the camera image

class Robot:
    def __init__(self, real=False):
        self.S = hack.Scene()
        self.q0 = self.S.C.getJointState()
        self.bot = ry.BotOp(C=self.S.C, useRealRobot=real)
        
    def run(self):
        
        rgb, depth = self.bot.getImageAndDepth('cameraWrist')
        D = hack.DataPlayer(rgb, depth)

        while True:
            key = self.bot.sync(self.S.C, .1)
            if key==ord('q'):
                break

            rgb, depth = self.bot.getImageAndDepth('cameraWrist')

            cv2.imwrite('rgb.png', rgb)
            
            depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)

            cv2.imwrite('depth.png', depth_vis)

            D.update(rgb, depth)

            print(np.max(depth), np.min(depth))
            break

    def yolo(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        imgs = ['depth.png']
        results = model(imgs)

        results.print()
        results.show()

if __name__ == "__main__":

    R = Robot(real=True)
    R.run()
    R.yolo()
