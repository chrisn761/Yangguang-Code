"""Train only the four corrected, structurally distinct final models."""
import subprocess
import sys
from pathlib import Path

MODELS=['GEL-unidirectional','GEL-no-attention-final','GEL-no-residual-final','LSTM-final']
if __name__=='__main__':
    for model in MODELS:
        subprocess.run([sys.executable,str(Path(__file__).with_name('train81.py')),
                        '--model',model,'--force-retrain',*sys.argv[1:]],check=True)
