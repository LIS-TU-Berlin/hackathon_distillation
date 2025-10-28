import subprocess
import os
from pathlib import Path

class DataHelper:
    hal_path = '$USER@hal-9000.lis.tu-berlin.de:/home/data/hackathon/'

    def __init__(self):
        self.hal_path = os.path.expandvars(self.hal_path)
        self.rsync = 'rsync -vrlptzP --update --mkpath'.split()

    def push_to_HAL(self, file):
        cmd = self.rsync + [file, self.hal_path]
        print('== sync command:', cmd)
        subprocess.run(cmd)

    def pull_from_HAL(self, file):
        Path('hal/').mkdir(parents=True, exist_ok=True)
        cmd = self.rsync + [self.hal_path+file, 'hal/']
        print('== sync command:', cmd)
        subprocess.run(cmd)
