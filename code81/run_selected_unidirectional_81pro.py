"""Retrain only seven requested models; originals remain in timestamp folders."""
import subprocess
import sys
from pathlib import Path

SELECTED=['GEL-no-attention-simple','GEL-no-residual-simple','CNN-LSTM','LSTM','GRU','MSCNN-LSTM','Transformer']
if __name__=='__main__':
    for model in SELECTED:
        subprocess.run([sys.executable,str(Path(__file__).with_name('train81.py')),
                        '--model',model,'--force-retrain',*sys.argv[1:]],check=True)
