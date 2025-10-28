#!/usr/bin/env python3

import hackathon_distillation as hack
import time

def data_generation(num_episodes=10):
    B = hack.ExpertBehavior()
    h5 = hack.H5Writer('data.h5')

    manifest = { 'info': 'several episodes of ball following expert behavior',
                 'num_episodes': num_episodes,
                 'tau_step': B.tau_step
                }
    h5.write_dict('manifest', manifest)

    for e in range(num_episodes):
        B.reset()
        B.run_with_Sim(h5=h5, verbose=0)

def data_checker(file='data.h5'):
    h5 = hack.H5Reader(file)
    manifest = h5.read_dict('manifest')
    print('manifest:', manifest)

    S = hack.Scene()
    N = manifest['num_episodes']
    for i in range(N):
        q = h5.read(f'epi{i:04}/q')
        rgb = h5.read(f'epi{i:04}/rgb')
        depth = h5.read(f'epi{i:04}/depth')

        D = hack.DataPlayer(rgb[0], depth[0])
        for t in range(rgb.shape[0]):
            S.C.setJointState(q[t])
            S.C.view(False, f'episode {i} step {t}')
            D.update(rgb[t], depth[t])
            time.sleep(.01)


if __name__ == "__main__":
    # data_generation(10)
    data_checker()

    # hack.DataHelper().push_to_HAL('data.h5')
    # hack.DataHelper().pull_from_HAL('data.h5')
