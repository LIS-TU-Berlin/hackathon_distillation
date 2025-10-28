import robotic as ry
import hackathon_distillation as hack
import time
import numpy as np


class Behavior:
    def __init__(self):
        self.S = hack.Scene()
        self.T_episode = 5.
        self.q0 = self.S.C.getJointState()

        self.reset()

    def reset(self):
        self.S.C.setJointState(self.q0)

        points = .2 * np.random.randn(20,3)
        points += self.S.ball.getPosition()
        times = np.linspace(0., self.T_episode, 20)
        self.spline = ry.BSpline()
        self.spline.set(2, points, times)

    def IK(self, target_pos):
        komo = ry.KOMO(self.S.C, 1, 1, 0, False)
        komo.addControlObjective([], 0, 1e-1)
        komo.addObjective([], ry.FS.position, ['l_gripper'], ry.OT.sos, [1e2], target_pos)

        sol = ry.NLP_Solver(komo.nlp(), verbose=0)
        # sol.setOptions(stopInners=4, damping=1e-4, verbose=0)
        ret = sol.solve()

        return [komo.getPath()[0], ret]

    def run(self):

        bot = ry.BotOp(self.S.C, useRealRobot=False)

        while bot.get_t() < self.T_episode:
            bot.sync(self.S.C, .1)

            t = bot.get_t()
            target_pos = self.spline.eval([t]). reshape(3)
            target_pos = np.clip(target_pos, a_min=[-10., .1, .6], a_max=[10., 10., 10.])

            self.S.ball.setPosition(target_pos)

            
            q_target, ret = self.IK(target_pos)
            if ret.feasible:
                bot.moveTo(q_target, timeCost=5., overwrite=True)



if __name__ == "__main__":
    B = Behavior()
    B.run()
