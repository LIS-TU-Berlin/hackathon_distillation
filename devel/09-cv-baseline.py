import argparse
import robotic as ry
import numpy as np
from typing import Tuple, List, Any
import time
import open3d as o3d
import hackathon_distillation as hack
from hackathon_distillation.masker import Masker

# TODO: Check fxycxy - is it same for sim and real?

class Robot:

    def __init__(self, args):
        self.args = args
        self.S = hack.Scene()

        # High contrast between ball and background helps detection
        self.S.C.getFrame("table").setColor([1., 1., 1.])
        self.S.C.getFrame("wall").setColor([1., 1., 1.])
        #self.S.C.getFrame("ball").setColor([0., 0., 1.])

        self.bot = ry.BotOp(C=self.S.C, useRealRobot=self.args.real)
        self.q0 = self.S.C.getJointState()
        self.ball_pos0 = self.S.ball.getPosition()
        self.fxycxy = [322.1999816894531, 322.1999816894531, 320.0, 180.0]
        self.generate_ball_motion()

        self.masker = Masker()

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

    def generate_ball_motion(self): 
        num_ctrl_points = 3 #IMPORTANT: more control points make the ball move faster
        points = np.array([.2, .2, .2]) * np.random.randn(num_ctrl_points, 3)
        points[0] = 0.
        points[-1] = 0.
        points += self.ball_pos0
        times = np.linspace(0., self.args.T_ep, num_ctrl_points)
        self.spline = ry.BSpline()
        self.spline.set(2, points, times)

    def predict(self, rgb:np.ndarray, depth:np.ndarray):

        camera_relative_ball_pos = None
        
        mask, detected = self.masker.blob(rgb)

        # Ensure all mask channels are identical
        if mask.ndim == 3 and mask.shape[2] == 3:
            assert np.array_equal(mask[:, :, 0], mask[:, :, 1]) and np.array_equal(mask[:, :, 1], mask[:, :, 2]), "Mask channels are not identical"
            mask = mask[:, :, 0]

        pts = ry.depthImage2PointCloud(depth, self.fxycxy)

        if detected:
            print("Detected")

            # Select points where mask == 1.0
            sel = (mask == 255)
            ball_pts = pts[sel]  # shape (N,3)

            assert ball_pts.size > 0

            # Remove outliers from selected points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(ball_pts)
            pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            ball_pts = np.asarray(pcd_clean.points)

            # Display selected point
            self.S.C.getFrame("cameraWrist").setPointCloud(points=ball_pts) .setColor([0.,1.,0.])

            # Compute centroid
            ball_centroid = np.mean(ball_pts, axis=0)
            print(ball_centroid)

            camera_relative_ball_pos = ball_centroid.copy()
        else:
            print("Not detected")

        return camera_relative_ball_pos, mask
    
    def main_sim(self):

        # Initialize motion
        self.bot.home(self.S.C)
        self.bot.sync(self.S.C, .1)

        rgb, depth = self.S.get_rgb_and_depth()
        D = hack.DataPlayer(rgb, depth)

        t0 = self.bot.get_t()

        while self.bot.get_t() - t0 < self.args.T_ep:

            t = self.bot.get_t()       

            ball_target_pos = self.spline.eval([t]). reshape(3)
            ball_target_pos = np.clip(ball_target_pos, a_min=self.ball_pos0+[-.2,-.2,-.2], a_max=self.ball_pos0+[.2,.2,.2])
            self.S.ball.setPosition(ball_target_pos) #for display only

            rgb, depth = self.S.get_rgb_and_depth()

            # Prediction is wrt to wrist camera
            target_pos, mask = self.predict(rgb, depth)
            D.update(rgb, mask)

            if target_pos is not None:
                q_target, ret = self.IK(target_pos)

                if ret.feasible:
                    self.bot.moveTo(q_target, timeCost=self.args.tc, overwrite=True)
                    pass
                else:
                    print("Not feasible!")
                    return

                self.bot.sync(self.S.C, .1)
            
                time.sleep(self.args.sleep)
            else:
                time.sleep(self.args.sleep)

            key = self.bot.sync(self.S.C, .1)
            if key==ord('q'):
                break


    def main_real(self):

        # Initialize motion
        self.bot.home(self.S.C)
        self.bot.sync(self.S.C, .1)

        rgb, depth = self.bot.getImageAndDepth('cameraWrist')
        target_pos, mask = self.predict(rgb, depth)

        D = hack.DataPlayer(rgb, depth)

        t0 = self.bot.get_t()

        while self.bot.get_t() - t0 < self.args.T_ep:

            t = self.bot.get_t()       

            rgb, depth = self.bot.getImageAndDepth('cameraWrist')

            # Prediction is wrt to wrist camera
            target_pos, mask = self.predict(rgb, depth)

            D.update(rgb, mask)       

            if target_pos is not None:
                q_target, ret = self.IK(target_pos)

                if ret.feasible:
                    self.bot.moveTo(q_target, timeCost=self.args.tc, overwrite=True)
                    pass
                else:
                    print("Not feasible!")
                    return

                self.bot.sync(self.S.C, .1)
            
                time.sleep(self.args.sleep)
            else:
                time.sleep(self.args.sleep)

            key = self.bot.sync(self.S.C, .1)
            if key==ord('q'):
                del self.bot
                break

        del self.bot


if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("--T_ep", type=int, default=10, help="Time to move")
    p.add_argument("--tc", type=float, default=0.5, help="Arg for bot.moveTo (lower is slower)")
    p.add_argument("--sleep", type=float, default=0.1, help="Sleep time")
    p.add_argument("--real", action="store_true", default=False, help="Use this arg if real robot is used")  # Use this arg to run on the real robot 
    args = p.parse_args()

    if args.real:
        Robot(args).main_real()
    else:
        Robot(args).main_sim()


