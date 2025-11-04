import robotic as ry
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from .scene import *

class Scene:
    def __init__(self):
        self.C = ry.Config()
        self.C.addFile('$RAI_PATH/scenarios/pandaSingle.g')

        table = self.C.getFrame("table")  # explicitly set the color here
        table.setColor([1., 1., 1.])

        self.ball = self.C.addFrame('ball')
        self.ball.setShape(ry.ST.sphere, [.0315]) .setColor([.2, .5, .6]) .setPosition([-.05, .45, 1.])

        self.wall = self.C.addFrame('wall')
        self.wall.setShape(ry.ST.ssBox, size=[2.0, .05, 2.0, 0.005]) .setPosition([0.0, -0.4, 1.5])

        self.ref = self.C.addFrame('ref', 'l_gripper')
        self.ref.setRelativePosition([0,0,-.2]) .setShape(ry.ST.marker, [.2])

        self.ref_target = self.C.addFrame('ref_target', 'cameraWrist')

        self.camview = ry.CameraView(self.C)
        self.camview.setCamera(self.C.getFrame('cameraWrist'))

    def get_rgb_and_depth(self, get_noise: bool = True):
        rgb, depth = self.camview.computeImageAndDepth(self.C, simulateDepthNoise=get_noise)
        # pcl = ry.depthImage2PointCloud(depth, self.camview.getFxycxy())
        return rgb, depth * 1000.

class DataPlayer:

    def __init__(self, rgb, depth):
        self.rgb = rgb
        self.depth = depth
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        self.im1 = self.ax1.imshow(rgb)
        self.im2 = self.ax2.imshow(depth, cmap="viridis")
        self.cbar = self.fig.colorbar(self.im2, ax=self.ax2, fraction=0.046, pad=0.04)
        plt.pause(.01)
    
    # def __del__(self):

    def update(self, rgb, depth):
        self.im1.set_data(rgb)
        self.im1.set_clim(vmin=np.min(depth), vmax=np.max(depth))
        self.im2.set_data(depth)
        self.im2.set_clim(vmin=np.min(depth), vmax=np.max(depth))
        plt.pause(.01)
