import torch
import robotic as ry
import hackathon_distillation as hack
import cv2

class Robot:
    def __init__(self, real=False):
        self.S = hack.Scene()
        self.q0 = self.S.C.getJointState()
        self.bot = ry.BotOp(C=self.S.C, useRealRobot=real)
        
    def run(self):
        
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        rgb, depth = self.bot.getImageAndDepth('cameraWrist')
        D = hack.DataPlayer(rgb, depth)

        while True:
            key = self.bot.sync(self.S.C, .1)
            if key==ord('q'):
                break

            rgb, depth = self.bot.getImageAndDepth('cameraWrist')
            D.update(rgb, depth)

if __name__ == "__main__":

    R = Robot(real=True)
    R.run()
