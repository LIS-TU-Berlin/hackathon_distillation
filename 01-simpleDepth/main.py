import robotic as ry
import matplotlib.pyplot as plt
import hackathon_distillation as hack

ry.params_add({'DepthNoise/binocular_baseline': .05,
  'DepthNoise/depth_smoothing': 1,
  'DepthNoise/noise_all': .05,
  'DepthNoise/noise_wide': 4.,
  'DepthNoise/noise_local': .4,
  'DepthNoise/noise_pixel': .04})

class Scene:
    def __init__(self):
        self.C = ry.Config()
        self.C.addFile('$RAI_PATH/scenarios/pandaSingle.g')
        self.C.addFrame('ball') .setShape(ry.ST.sphere, [.03]) .setColor([.2, .7, .8]) .setPosition([-.05, .2, 1.])
        self.C.view(False)

        self.camview = ry.CameraView(self.C)
        self.camview.setCamera(self.C.getFrame('cameraWrist'))

    def get_rgb_and_depth(self):
        rgb, depth = self.camview.computeImageAndDepth(self.C, simulateDepthNoise=True)
        # pcl = ry.depthImage2PointCloud(depth, self.camview.getFxycxy())
        return rgb, depth

    def plot_rgb_and_depth(self, rgb, depth):
        fig = plt.figure()
        fig.add_subplot(1,2,1)
        plt.imshow(rgb)
        fig.add_subplot(1,2,2)
        plt.imshow(depth)
        plt.show()

def mini_demo():
    S = Scene()
    rgb, depth = S.get_rgb_and_depth()
    S.plot_rgb_and_depth(rgb, depth)

def store_h5():
    h5 = hack.H5Writer('data.h5')

    # write a manifest into the data file
    N = 10
    manifest = { 'info': 'This file contains just rendered rgb and depth images', 'num_data': N }
    h5.write_dict('manifest', manifest)

    # generate N renders and write into data file
    S = Scene()
    for i in range(N):
        rgb, depth = S.get_rgb_and_depth()
        h5.write(f'dat{i:04}/rgb', rgb, dtype='uint8')
        h5.write(f'dat{i:04}/depth', depth, dtype='float32')

def print_file_info():
    h5 = hack.H5Reader('data.h5')
    print('=== this is the same as calling ry_h5Info from command line ===')
    h5.print_info()
    print('===')

def howto_load_from_file():
    h5 = hack.H5Reader('data.h5')
    manifest = h5.read_dict('manifest')

    S = Scene()
    N = manifest['num_data']
    for i in range(N):
        rgb = h5.read(f'dat{i:04}/rgb')
        depth = h5.read(f'dat{i:04}/depth')
        S.plot_rgb_and_depth(rgb, depth)

if __name__ == "__main__":
    store_h5()
    print_file_info()
    howto_load_from_file()
