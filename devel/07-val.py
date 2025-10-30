import argparse
from pathlib import Path

import robotic as ry
import numpy as np
from typing import Tuple, List, Any
import time
import torch

import hackathon_distillation as hack
from collections import deque

from hackathon_distillation.policy.policy import Policy
from hackathon_distillation.policy.trainer import DATA_PATH


# Input depth, 3D EE position, output position, keep orientation free
# out: seq of actions: [prediction_horizon, 3D points]

class Robot:

    def __init__(self, args):
        self.args = args
        self.S = hack.Scene()
        self.q0 = self.S.C.getJointState()
        self.ball_pos0 = self.S.ball.getPosition()
        self.bot = ry.BotOp(C=self.S.C, useRealRobot=self.args.real)
        self.modelpth = args.modelpth
        self.device_id = args.device_id
        self.generate_ball_motion()

        # Load the model
        stats_file = Path(DATA_PATH / "data_stats.pt")
        self.policy = Policy(Path(self.modelpth), stats_file, map_location=f"cpu")  # todo: fix for cpu

    def IK(self, camera_relative_ball_pos):
        self.S.ref_target.setRelativePosition(camera_relative_ball_pos)
        ball_pos = self.S.ref_target.getPosition()

        komo = ry.KOMO(self.S.C, 1, 1, 0, False)
        komo.addControlObjective([], 0, 1e-1)
        komo.addObjective([], ry.FS.position, ['ref'], ry.OT.sos, [1e2], ball_pos)
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
        komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
        komo.addObjective([], ry.FS.negDistance, ["l_panda_coll3", "wall"], ry.OT.ineq)

        sol = ry.NLP_Solver(komo.nlp(), verbose=0)
        sol.setOptions(stopInners=10, damping=1e-4) ##very low cost
        ret = sol.solve()

        if not ret.feasible:
            print(f"KOMO report: {komo.report()}")

        return [komo.getPath()[0], ret]
    
    def predict(self, depth:np.ndarray):
        batch = {"obs.img": torch.Tensor(depth)[None]}
        actions = self.policy.select_action(batch)
        return actions
    
    def view(self):
        rgb, depth = self.bot.getImageAndDepth('cameraWrist')
        
        D = hack.DataPlayer(rgb, depth)
        while True:
            key = self.bot.sync(self.S.C, .1)
            if key==ord('q'):
                break

            rgb, depth = self.bot.getImageAndDepth('cameraWrist')
            D.update(rgb, depth)

    def generate_ball_motion(self): 
        num_ctrl_points = 5 #IMPORTANT: more control points make the ball move faster
        points = np.array([.2, .2, .2]) * np.random.randn(num_ctrl_points, 3)
        points[0] = 0.
        points[-1] = 0.
        points += self.ball_pos0
        times = np.linspace(0., self.args.T_ep, num_ctrl_points)
        self.spline = ry.BSpline()
        self.spline.set(2, points, times)

    def validate(self):
        """
        Run predictions from model
        """        

        if self.args.real:
            rgb, depth = self.bot.getImageAndDepth('cameraWrist')
        else:
            rgb, depth = self.S.get_rgb_and_depth()
        #
        D = hack.DataPlayer(rgb, depth)

        # Initialize motion
        self.bot.moveTo(self.q0)
        self.bot.wait(self.S.C, forKeyPressed=False, forTimeToEnd=True)
        self.bot.sync(self.S.C, .1)

        t0 = self.bot.get_t()

        while self.bot.get_t() - t0 < self.args.T_ep:
            t = self.bot.get_t()
            ball_target_pos = self.spline.eval([t]). reshape(3)
            ball_target_pos = np.clip(ball_target_pos, a_min=self.ball_pos0+[-.2,-.2,-.2], a_max=self.ball_pos0+[.2,.2,.2])
            self.S.ball.setPosition(ball_target_pos) #for display only

            if self.args.real:
                rgb, depth = self.bot.getImageAndDepth('cameraWrist')
            else:
                rgb, depth = self.S.get_rgb_and_depth()
            D.update(rgb, depth)

            # Prediction is wrt to wrist camera
            target_pos = self.predict(depth)
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
    p.add_argument("--hist", type=int, default=2, help="Len of input history")
    p.add_argument("--maxhist", type=int, default=2, help="Max. len of input history")
    p.add_argument("--tc", type=float, default=1.0, help="Arg for bot.moveTo (lower is slower)")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep time")
    p.add_argument("--modelpth", type=str, default="/home/data/hackathon/ckpts/mlp/run_02/last.pt", help="Path to model")
    # p.add_argument("--modelpth", type=str, default="/home/braun/hackathon_distillation/ckpts/ddpm_unet_test/last.pt", help="Path to model")
    p.add_argument("--real", action="store_true", default=False, help="Use this arg if real robot is used")  # Use this arg to run on the real robot
    p.add_argument("--device_id", type=int, default=1, help="for cuda")  # Use this arg to run on the real robot
    args = p.parse_args()

    print(args)

    Robot(args).validate()

    # stats_file = Path(DATA_PATH / "data_stats.pt")
    # policy = Policy(Path("/home/data/hackathon/ckpts/mlp/run_02/last.pt"), stats_file, map_location="cuda:1")





    