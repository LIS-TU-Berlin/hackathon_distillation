import torch
import robotic as ry
import cv2
import numpy as np
from ultralytics import YOLO, SAM, FastSAM
from typing import List, Tuple, Any
from einops import rearrange
import time

import hackathon_distillation as hack

class Masker:

    def __init__(
        self, 
        detect_model_pth:str="yolo11m.pt", 
        segment_model_pth:str="yolo11l-seg.pt", 
        bot:ry.BotOp=None
        ):
        self.bot = bot
        self.detect_model = YOLO(detect_model_pth)
        self.segment_model = FastSAM(segment_model_pth)

    def IK(self, target_pos):
        komo = ry.KOMO(self.S.C, 1, 1, 0, False)
        komo.addControlObjective([], 0, 1e-1)
        komo.addObjective([], ry.FS.position, ['ref'], ry.OT.sos, [1e2], target_pos)
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
        komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
        komo.addObjective([], ry.FS.negDistance, ["l_panda_coll3", "wall"], ry.OT.ineq)

        sol = ry.NLP_Solver(komo.nlp(), verbose=0)
        sol.setOptions(stopInners=10, damping=1e-4) ##very low cost
        ret = sol.solve()

        if not ret.feasible:
            print(f"KOMO report: {komo.report()}")

        return [komo.getPath()[0], ret]

    def segment(self, img:np.ndarray):
        """
        
        """
        results = self.segment_model([img])

        masks = []
        for r in results:
            if r.masks:
                for m in r.masks:
                    _m = m.data.detach().cpu().numpy()[0]*255
                    masks.append(_m.astype(np.uint8))
        
        # Union of all masks
        if len(masks) > 0:
            mask = np.zeros_like(masks[0], dtype=np.uint8)
            for m in masks:
                mask = self.img_or(mask, m)
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        return mask

    def blob(self, img:np.ndarray):
        
        # Setup SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.minCircularity = 0.6
        params.filterByArea = True
        params.minArea = 500
        
        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(img)

        mask = np.zeros_like(img, dtype=np.uint8)

        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            r = int(kp.size / 2)
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

        return mask
    
    def img_or(self, img1:np.ndarray, img2:np.ndarray):
        return cv2.bitwise_or(img1, img2)

    def img_and(self, img1:np.ndarray, img2:np.ndarray):
        return cv2.bitwise_and(img1, img2)
    
if __name__ == "__main__":

    S = hack.Scene()
    q0 = S.C.getJointState()
    ee0 = S.C.getFrame('l_gripper').getPosition()
    bot = ry.BotOp(C=S.C, useRealRobot=True)

    m = Masker(bot=bot)

    rgb, depth = bot.getImageAndDepth('cameraWrist')

    mask_seg = m.segment(rgb)
    mask_blob = m.blob(rgb)

    D = hack.DataPlayer(mask_blob, depth)

    t0 = bot.get_t()
    target_pos = ee0.copy()

    seg_ms = 0.0
    blob_ms = 0.0   
    cnt = 0

    while bot.get_t() - t0 < 30:
        rgb, depth = bot.getImageAndDepth('cameraWrist')

        start = time.perf_counter()
        mask_seg = m.segment(rgb)
        seg_ms += (time.perf_counter() - start) * 1000.0
        
        start = time.perf_counter()
        mask_blob = m.blob(rgb)
        blob_ms += (time.perf_counter() - start) * 1000.0
        
        D.update(mask_blob, depth)

        target_pos *= 1.001
        #q_target, ret = m.IK(target_pos)

        if True:
            bot.moveTo(q0, timeCost=1.0, overwrite=True)
        else:
            print("Not feasible!")
            break

        bot.sync(S.C, .1)
        
        time.sleep(0.1)

        cnt += 1

        key = bot.sync(S.C, .1)
        if key==ord('q'):
            break

        print(f"Average segmentation time: {seg_ms/cnt:.2f} ms")
        print(f"Average blob detection time: {blob_ms/cnt:.2f} ms")


