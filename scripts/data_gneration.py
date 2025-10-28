#!/usr/bin/env python3

import hackathon_distillation as hack

def data_generation(num_episodes=10):
    B = hack.ExpertBehavior()
    h5 = hack.H5Writer('data.h5')

    for e in range(num_episodes):
        B.reset()
        B.run_with_Sim(h5=h5, verbose=0)


if __name__ == "__main__":
    # data_generation()

    dh = hack.DataHelper()
    # dh.push_to_HAL('data.h5')
    dh.pull_from_HAL('data.h5')
