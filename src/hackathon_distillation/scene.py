import robotic as ry
import matplotlib.pyplot as plt
from matplotlib import animation
from .scene import *

class Scene:
    def __init__(self):
        self.C = ry.Config()
        self.C.addFile('$RAI_PATH/scenarios/pandaSingle.g')
        self.ball = self.C.addFrame('ball')
        self.ball.setShape(ry.ST.sphere, [.0315]) .setColor([.2, .5, .6]) .setPosition([-.05, .2, 1.])

        self.wall = self.C.addFrame('wall')
        self.wall.setShape(ry.ST.ssBox, size=[2.0, .05, 2.0, 0.005]) .setPosition([0.0, -0.4, 1.5])

        self.ref = self.C.addFrame('ref', 'l_gripper')
        self.ref.setRelativePosition([0,0,-.2]) .setShape(ry.ST.marker, [.2])

        self.camview = ry.CameraView(self.C)
        self.camview.setCamera(self.C.getFrame('cameraWrist'))

    def get_rgb_and_depth(self):
        rgb, depth = self.camview.computeImageAndDepth(self.C, simulateDepthNoise=True)
        # pcl = ry.depthImage2PointCloud(depth, self.camview.getFxycxy())
        return rgb, depth

class DataPlayer:

    def __init__(self, rgb, depth):
        self.rgb = rgb
        self.depth = depth
        self.fig = plt.figure()
        self.fig.add_subplot(1,2,1)
        self.im1 = plt.imshow(rgb)
        self.fig.add_subplot(1,2,2)
        self.im2 = plt.imshow(depth)
        plt.pause(.01)

    def update(self, rgb, depth):
        self.im1.set_data(rgb)
        self.im2.set_data(depth)
        plt.pause(.01)

