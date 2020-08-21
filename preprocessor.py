#%%
import numpy as np
import pandas as pd
import config as cfg
from configparser import ConfigParser as cfg
#%%
cfg=cfg()
cfg.read('./PSA.conf')
WINDOW=int(cfg['MAIN']['WINDOW'])
STRIDE=int(cfg['MAIN']['STRIDE'])
SAMPLING_RATE=int(cfg['MAIN']['SAMPLING_RATE'])

#%%
INPUT = np.load('./DATA/RAW/input_ie.npy')
OUTPUT_ie = np.load('F:/ie_diagnosis/output_ie.npy')
OUTPUT_MSLB = np.load('./DATA/RAW/output_MSLB.npy')
OUTPUT_SGTR = np.load('./DATA/RAW/output_SGTR.npy')
