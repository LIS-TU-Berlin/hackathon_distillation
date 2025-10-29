import robotic as ry
import hackathon_distillation as hack
import time
import numpy as np

class ExpertBehavior:
    T_episode = 5.
    tau_sim = .01
    tau_step = .1
    episode_count = 0

    def __init__(self):
        self.S = hack.Scene()
        self.q0 = self.S.C.getJointState()
        self.pos0 = self.S.ball.getPosition()

        self.reset()

    def reset(self):
        q = self.q0.copy()
        q += .2*np.random.randn(7)
        self.S.C.setJointState(q)

        num_ctrl_points = 5 #reduce to make smoother

        points = np.array([.2, .2, .1]) * np.random.randn(num_ctrl_points, 3)
        points += self.pos0
        times = np.linspace(0., self.T_episode, num_ctrl_points)
        self.spline = ry.BSpline()
        self.spline.set(2, points, times)

    def IK(self, target_pos):
        komo = ry.KOMO(self.S.C, 1, 1, 0, False)
        komo.addControlObjective([], 0, 1e-1)
        komo.addObjective([], ry.FS.position, ['ref'], ry.OT.sos, [1e2], target_pos)

        # Collisions and joint limits
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
        komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
        komo.addObjective([], ry.FS.negDistance, ["l_panda_coll3", "wall"], ry.OT.ineq)

        # Try to keep the ball in view
        komo.addObjective([], ry.FS.scalarProductXX, ["cameraWrist", "table"], ry.OT.eq, [-1.0])
        komo.addObjective([], ry.FS.scalarProductXX, ["cameraWrist", "ball"], ry.OT.eq, [-1.0])

        sol = ry.NLP_Solver(komo.nlp(), verbose=0)
        sol.setOptions(stopInners=10, damping=1e-4) ##very low cost
        ret = sol.solve()

        return [komo.getPath()[0], ret]

    def run_with_BotOp(self):

        bot = ry.BotOp(self.S.C, useRealRobot=False)

        while bot.get_t() < self.T_episode:
            bot.sync(self.S.C, .1)

            t = bot.get_t()
            target_pos = self.spline.eval([t]). reshape(3)
            target_pos = np.clip(target_pos, a_min=[-10., .1, .6], a_max=[10., 10., 10.])

            self.S.ball.setPosition(target_pos) #for display only

            ref_pos = self.S.ref.getPosition()

            action = target_pos

            q_target, ret = self.IK(action)
            if ret.feasible:
                bot.moveTo(q_target, timeCost=5., overwrite=True)

    def run_with_Sim(self, h5=None, verbose=1):

        print(f'=== run and episode {self.episode_count}')
        sim = ry.Simulation(self.S.C, engine=ry.SimulationEngine.physx, verbose=0)
        sim.resetSplineRef()
        sim.selectSensor('cameraWrist')
        sim.setSimulateDepthNoise(True)

        num_steps = int(self.T_episode / self.tau_step) + 1
        data_ee_pos = np.empty((num_steps, 3))
        data_q = np.empty((num_steps, 7))
        data_rgb = np.empty((num_steps, 360, 640, 3))
        data_depth = np.empty((num_steps, 360, 640))
        data_ee_action = np.empty((num_steps, 3))
        step = 0

        #self.S.C.getFrame('cameraWrist').setAttributes({'zRange': [0.1, 10.0]})

        t = 0
        for step in range(num_steps):
            for k in range(int(self.tau_step / self.tau_sim)):
                sim.step([], self.tau_sim, ry.ControlMode.spline)
            t += self.tau_step

            if verbose>0:
                self.S.C.view(False, f'time: {t}')
                time.sleep(self.tau_step)

            target_pos = self.spline.eval([t]). reshape(3)
            target_pos = np.clip(target_pos, a_min=[-10., .1, .6], a_max=[10., 10., 10.])
            self.S.ball.setPosition(target_pos) #for display only

            action = target_pos

            if h5 is not None:
                data_ee_action[step] = action
                data_ee_pos[step] = self.S.ref.getPosition()
                data_q[step] = self.S.C.getJointState()
                data_rgb[step], data_depth[step] = sim.getImageAndDepth()

            q_target, ret = self.IK(action)
            if ret.feasible:
                sim.setSplineRef(q_target, [.1], False)
        print('=== done')

        if h5 is not None:
            tag = f'epi{self.episode_count:04}/'
            h5.write(tag+'ee_action', data_ee_action)
            h5.write(tag+'ee_pos', data_ee_pos)
            h5.write(tag+'q', data_q)
            h5.write(tag+'rgb', data_rgb, dtype='uint8')
            h5.write(tag+'depth', data_depth, dtype='float32')

        self.episode_count += 1
