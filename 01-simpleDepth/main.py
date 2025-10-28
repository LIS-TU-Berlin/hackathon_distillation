import robotic as ry
import matplotlib.pyplot as plt
import hackathon_distillation as hack

ry.params_add({'DepthNoise/binocular_baseline': .05,
  'DepthNoise/depth_smoothing': 1,
  'DepthNoise/noise_all': .05,
  'DepthNoise/noise_wide': 4.,
  'DepthNoise/noise_local': .4,
  'DepthNoise/noise_pixel': .04})

def mini_demo():
    S = hack.Scene()
    rgb, depth = S.get_rgb_and_depth()
    S.plot_rgb_and_depth(rgb, depth)

def store_h5():
    h5 = hack.H5Writer('data.h5')

    S = hack.Scene()

    # write a manifest into the data file
    N = 10
    manifest = { 'info': 'This file contains just rendered rgb and depth images',
                 'num_data': N,
                 'fxycxy': list(S.camview.getFxycxy())q
                }
    h5.write_dict('manifest', manifest)

    # generate N renders and write into data file
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
    print('manifest:', manifest)

    S = hack.Scene()
    N = manifest['num_data']
    for i in range(N):
        rgb = h5.read(f'dat{i:04}/rgb')
        depth = h5.read(f'dat{i:04}/depth')
        S.plot_rgb_and_depth(rgb, depth)

if __name__ == "__main__":
    store_h5()
    print_file_info()
    howto_load_from_file()
