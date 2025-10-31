#!/usr/bin/env python3

import hackathon_distillation as hack
import time
import robotic as ry

default_noise_setting = {'DepthNoise/binocular_baseline': .03, #zero -> (almost no shadows)
  'DepthNoise/depth_smoothing': 1, # zero -> no smoothed edges (for zero baseline also no shadows at all)
  'DepthNoise/noise_all': .02, # zero -> no noise
  'DepthNoise/noise_wide': 4.,
  'DepthNoise/noise_local': .4,
  'DepthNoise/noise_pixel': .04}

my_noise_setting = {'DepthNoise/binocular_baseline': .01, #zero -> (almost no shadows)
  'DepthNoise/depth_smoothing': 0, # zero -> no smoothed edges (for zero baseline also no shadows at all)
  'DepthNoise/noise_all': .05, # zero -> no noise
  'DepthNoise/noise_wide': 1.,
  'DepthNoise/noise_local': .4,
  'DepthNoise/noise_pixel': .04}

ry.params_add(my_noise_setting)

def data_generation(file='new_data.h5', num_episodes=10):
    B = hack.ExpertBehavior()
    h5 = hack.H5Writer(file)
    S = hack.Scene()

    manifest = { 'info': 'several episodes of ball following expert behavior',
                 'num_episodes': num_episodes,
                 'tau_step': B.tau_step,
                 'fxycxy': list(S.camview.getFxycxy())
                }
    h5.write_dict('manifest', manifest)

    for e in range(num_episodes):
        B.reset()
        B.run_with_Sim(h5=h5, verbose=0)

def data_checker(file='data.h5'):
    h5 = hack.H5Reader(file)
    manifest = h5.read_dict('manifest')
    print('manifest:', manifest)
    fxycxy = manifest['fxycxy']
    num_episodes = manifest['num_episodes']

    S = hack.Scene()
    f_pcl = S.C.addFrame('pcl', 'cameraWrist')

    for i in range(num_episodes):
        ee_action = h5.read(f'epi{i:04}/ee_action')
        q = h5.read(f'epi{i:04}/q')
        rgb = h5.read(f'epi{i:04}/rgb')
        depth = h5.read(f'epi{i:04}/depth')

        D = hack.DataPlayer(rgb[0], depth[0])
        for t in range(rgb.shape[0]):
            S.C.setJointState(q[t])
            S.ball.setPosition(ee_action[t])
            pts = ry.depthImage2PointCloud(depth[t], fxycxy)
            f_pcl.setPointCloud(pts, rgb[t]) #.setColor([1.,0.,0.])

            key = S.C.view(False, f'episode {i} step {t}')
            if key == ord('q'):
                break

            D.update(rgb[t], depth[t])

            time.sleep(.1)

if __name__ == "__main__":
    data_generation(file='test_data.h5', num_episodes=100)
    # data_checker(file='test_data.h5')

    # hack.DataHelper().push_to_HAL('data.h5')
    # hack.DataHelper().pull_from_HAL('data.h5')
