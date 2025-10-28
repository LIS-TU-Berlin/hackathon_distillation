import robotic as ry
import matplotlib.pyplot as plt

ry.params_add({'DepthNoise/binocular_baseline': .05,
  'DepthNoise/depth_smoothing': 1,
  'DepthNoise/noise_all': .05,
  'DepthNoise/noise_wide': 4.,
  'DepthNoise/noise_local': .4,
  'DepthNoise/noise_pixel': .04})

C = ry.Config()
C.addFile('$RAI_PATH/scenarios/pandaSingle.g')
C.view(False)

cam = ry.CameraView(C)
cam.setCamera(C.getFrame('cameraWrist'))
rgb, depth = cam.computeImageAndDepth(C, simulateDepthNoise=True)
pcl = ry.depthImage2PointCloud(depth, cam.getFxycxy())

print(rgb.shape, depth.shape, pcl.shape)
print(C.viewer().getCamera_fxycxy())

fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(rgb)
fig.add_subplot(1,2,2)
plt.imshow(depth)
plt.show()