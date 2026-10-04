import subprocess,sys
from pathlib import Path
from train81 import NAMES
if __name__=='__main__':
    for name in NAMES:
        subprocess.run([sys.executable,str(Path(__file__).with_name('train81.py')),'--model',name,*sys.argv[1:]],check=True)
