import argparse
import robotic as ry
import numpy as np
from typing import Tuple, List, Any
import time

import hackathon_distillation as hack

# Input depth, output position, keep orientation free

class Robot:

    def __init__(self, args):
        self.args = args
        self.S = hack.Scene()
        self.q0 = self.S.C.getJointState()
        self.bot = ry.BotOp(C=self.S.C, useRealRobot=self.args.real)

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
    
    def get_target_pose(self):
        # get target pose from model
        pass

    def replay(self):
        """
        Replay h5 data
        """
        
        # Load h5 data
        reader = hack.H5Reader(self.args.data)
        
        for episode in reader.fil.keys():

            print(f"Testing episode {episode}")
            data_ee = reader.read(f"{episode}/ee_pos")

            self.bot.moveTo(self.q0)
            self.bot.wait(self.S.C, forKeyPressed=False, forTimeToEnd=True)

            for p in data_ee:

                self.bot.sync(self.S.C, .1)

                t = self.bot.get_t()

                # Get the target position from model
                target_pos = p.copy()
                target_pos = np.clip(target_pos, a_min=[-10., .1, .6], a_max=[10., 10., 10.])

                q_target, ret = self.IK(target_pos)
                if ret.feasible:
                    self.bot.moveTo(q_target, timeCost=5., overwrite=True)
                else:
                    print("Not feasible!")
                    return

    def run(self):
        """
        Run on robot
        """

        while self.bot.get_t() < self.args.T_episode:
            pass

            
if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("--T_episode", type=int, default=10, help="Time to move (with sine motion profile)")
    p.add_argument("--data", type=str, default="", help="Path to h5 file")
    p.add_argument("--real", action="store_true", default=False, help="Use this arg if real robot is used")  # Use this arg to run on the real robot 
    args = p.parse_args()

    print(args)

    Robot(args).replay()





    