import argparse
import robotic as ry
import numpy as np
from typing import Tuple, List, Any
import time
import torch

import hackathon_distillation as hack

# Input depth, 3D EE position, output position, keep orientation free
# out: seq of actions: [prediction_horizon, 3D points]

class Robot:

    def __init__(self, args):
        self.args = args
        self.S = hack.Scene()
        self.q0 = self.S.C.getJointState()
        self.bot = ry.BotOp(C=self.S.C, useRealRobot=self.args.real)
        self.modelpth = args.modelpth

        # Load the model
        self.model = None  # TODO

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
    
    def predict(self, depth:np.ndarray, ee_pos:np.ndarray):

        depth_torch = torch.from_numpy(depth)
        ee_pos_torch = torch.from_numpy(ee_pos)

        # tgt_ee_pos_torch = self.model(depth_torch, ee_pos_torch)
        # tgt_ee_pos = tgt_ee_pos_torch.cpu().detach().numpy()
        # return tgt_ee_pos
        return ee_pos*1.01 # TODO: remove this and uncomment above
    
    def view(self):
        rgb, depth = self.bot.getImageAndDepth('cameraWrist')
        
        D = hack.DataPlayer(rgb, depth)
        while True:
            key = self.bot.sync(self.S.C, .1)
            if key==ord('q'):
                break

            rgb, depth = self.bot.getImageAndDepth('cameraWrist')
            D.update(rgb, depth)

    def validate(self):
        """
        Run predictions from model
        """        

        rgb, depth = self.bot.getImageAndDepth('cameraWrist')
        
        D = hack.DataPlayer(rgb, depth)

        # Initialize motion
        self.bot.moveTo(self.q0)
        self.bot.wait(self.S.C, forKeyPressed=False, forTimeToEnd=True)
        self.bot.sync(self.S.C, .1)

        t0 = self.bot.get_t()

        while self.bot.get_t() - t0 < args.T_ep:

            # Inputs for model: rgb, depth
            rgb, depth = self.bot.getImageAndDepth('cameraWrist')
            D.update(rgb, depth)

            # Inputs for model: ee position
            ee_pos = self.S.C.getFrame('l_gripper').getPosition()

            # Get the target position from model
            target_pos = self.predict(depth, ee_pos)

            q_target, ret = self.IK(target_pos)
            if ret.feasible:
                self.bot.moveTo(q_target, timeCost=self.args.tc, overwrite=True)
            else:
                print("Not feasible!")
                return

            self.bot.sync(self.S.C, .1)
           
            time.sleep(args.sleep)

            key = self.bot.sync(self.S.C, .1)
            if key==ord('q'):
                break

if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("--T_ep", type=int, default=10, help="Time to move")
    p.add_argument("--tc", type=float, default=1.0, help="Arg for bot.moveTo (lower is slower)")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep time")
    p.add_argument("--modelpth", type=str, default="", help="Path to model")
    p.add_argument("--real", action="store_true", default=False, help="Use this arg if real robot is used")  # Use this arg to run on the real robot 
    args = p.parse_args()

    print(args)

    Robot(args).validate()





    