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

        # Fix top down orientation
        komo.addObjective([], ry.FS.scalarProductXX, ["cameraWrist", "table"], ry.OT.eq, [-1.0])
        komo.addObjective([], ry.FS.scalarProductXX, ["cameraWrist", "ball"], ry.OT.eq, [-1.0])


        sol = ry.NLP_Solver(komo.nlp(), verbose=0)
        sol.setOptions(stopInners=10, damping=1e-4) ##very low cost
        ret = sol.solve()

        if not ret.feasible:
            print(f"KOMO report: {komo.report()}")

        return [komo.getPath()[0], ret]
    
    def get_target_pose(self):
        # get target pose from model
        pass

    def view(self):
        rgb, depth = self.bot.getImageAndDepth('cameraWrist')
        
        D = hack.DataPlayer(rgb, depth)
        while True:
            key = self.bot.sync(self.S.C, .1)
            if key==ord('q'):
                break

            rgb, depth = self.bot.getImageAndDepth('cameraWrist')
            D.update(rgb, depth)

    def replay(self):
        """
        Replay h5 data
        """
        
        # Load h5 data
        reader = hack.H5Reader(self.args.data)
        rgb, depth = self.bot.getImageAndDepth('cameraWrist')
        
        D = hack.DataPlayer(rgb, depth)

        for i, episode in enumerate(reader.fil.keys()):

            print(f"Testing episode {episode}")
            data_ee = reader.read(f"{episode}/ee_pos")

            self.bot.moveTo(self.q0)
            self.bot.wait(self.S.C, forKeyPressed=False, forTimeToEnd=True)

            for p in data_ee:

                rgb, depth = self.bot.getImageAndDepth('cameraWrist')
                D.update(rgb, depth)

                self.bot.sync(self.S.C, .1)

                t = self.bot.get_t()

                # Get the target position from model
                target_pos = p.copy()
                target_pos = np.clip(target_pos, a_min=[-10., .1, .6], a_max=[10., 10., 10.])

                self.S.ball.setPosition(target_pos)
                self.bot.sync(self.S.C, .1)

                q_target, ret = self.IK(target_pos)
                if ret.feasible:
                    self.bot.moveTo(q_target, timeCost=self.args.tc, overwrite=True)
                else:
                    print("Not feasible!")
                    return
                
                time.sleep(args.sleep)


                key = self.bot.sync(self.S.C, .1)
                if key==ord('q'):
                    break

            if i == self.args.ep - 1:
                break  # only run 2 episodes for demo

    def view_img(self):
        """
        View h5 data - loop through and display images
        """
        
        # Load h5 data
        reader = hack.H5Reader(self.args.data)
        #rgb, depth = self.bot.getImageAndDepth('cameraWrist')
        
        for i, episode in enumerate(reader.fil.keys()):

            print(f"Testing episode {episode}")
            rgb = reader.read(f"{episode}/rgb")
            depth = reader.read(f"{episode}/depth")

            D = hack.DataPlayer(rgb[0], depth[0])

            for i in range(rgb.shape[0]):
                D.update(rgb[i], depth[i])
                time.sleep(0.1)

    def test_constraints(self):

        frames = self.S.C.getFrames()
        for f in frames:
            print(f.name)

        f = self.S.C.addFrame(name='cam')
        f.setShape(type=ry.ST.marker, size=[.3])
        f.setPosition(self.S.C.getFrame('cameraWrist').getPosition())
        f.setQuaternion(self.S.C.getFrame('cameraWrist').getQuaternion())

        f = self.S.C.addFrame(name='tab')
        f.setShape(type=ry.ST.marker, size=[.3])
        f.setPosition(self.S.C.getFrame('table').getPosition())
        f.setQuaternion(self.S.C.getFrame('table').getQuaternion())
        self.S.C.view()

        print(self.S.C.getFrame('cameraWrist').getAttributes())
        
        time.sleep(10.0)

        poss_ball = self.S.C.getFrame('ball').getPosition()
        pos_camera = self.S.C.getFrame('cameraWrist').getPosition()




    def run(self):
        """
        Run on robot
        """

        while self.bot.get_t() < self.args.T_ep:
            pass

if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("--T_ep", type=int, default=10, help="Time to move")
    p.add_argument("--ep", type=int, default=1, help="Number of episodes to replay")
    p.add_argument("--tc", type=float, default=1.0, help="Arg for bot.moveTo (lower is slower)")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep time")
    p.add_argument("--data", type=str, default="new_data.h5", help="Path to h5 file")
    p.add_argument("--real", action="store_true", default=False, help="Use this arg if real robot is used")  # Use this arg to run on the real robot 
    args = p.parse_args()

    print(args)

    #Robot(args).replay()
    Robot(args).view_img()
    #Robot(args).test_constraints()




    