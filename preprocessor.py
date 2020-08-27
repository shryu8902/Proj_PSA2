#%%
import numpy as np
import pandas as pd
import config as cfg
from configparser import ConfigParser as cfg
import tqdm, glob, pickle, datetime, re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#%%
cfg=cfg()
cfg.read('./PSA.conf')
WINDOW=int(cfg['MAIN']['WINDOW'])
STRIDE=int(cfg['MAIN']['STRIDE'])
SAMPLING_RATE=int(cfg['MAIN']['SAMPLING_RATE'])
SEED=0
#%%
INPUT = np.load('D:/ie_diagnosis/input_ie.npy')
OUTPUT_ie = np.load('D:/ie_diagnosis/output_ie.npy')
OUTPUT_MSLB = np.load('./DATA/RAW/output_MSLB.npy')
OUTPUT_SGTR = np.load('./DATA/RAW/output_SGTR.npy')

#%%
Input = pd.DataFrame(INPUT)
Input['Class']= Input[0]<=99999
# %%
# MSLB_list = sorted([os.path.basename(x) for x in glob.glob('D:/ie_diagnosis/MSLB_csv/*.csv')])
# SGTR_list = sorted([os.path.basename(x) for x in glob.glob('D:/ie_diagnosis/SGTR_csv/*.csv')])
MSLB_list = sorted(glob.glob('D:/ie_diagnosis/MSLB_csv/*.csv'))
SGTR_list = sorted(glob.glob('D:/ie_diagnosis/SGTR_csv/*.csv'))
All_list = MSLB_list + SGTR_list
ClassLabel = np.concatenate([np.ones(np.shape(MSLB_list)),np.zeros(np.shape(SGTR_list))]) # 1 : MSLB, 0 : SGTR

Train_list, Test_list = train_test_split(All_list, test_size = 10000, random_state = SEED, stratify = ClassLabel)

with open('D:/ie_diagnosis/train_test.pickle','wb') as f:
    train_test_list = {'Train':Train_list, 'Test':Test_list}
    pickle.dump(train_test_split,f)

SCALERS = {'standard':StandardScaler(), 'minmax':MinMaxScaler()}
for file in tqdm.tqdm(All_list):
    temp = pd.read_csv(file)
    temp2 = np.float32(temp.values[...,1:])
    SCALERS['standard'].partial_fit(temp2)
    SCALERS['minmax'].partial_fit(temp2)
    if file in Train_list:
        if 'MSLB' in file:
            save_root = 'D:/ie_diagnosis/DATA/Train/MSLB/'
        else:
            save_root = 'D:/ie_diagnosis/DATA/Train/SGTR/'
    elif file in Test_list:
        if 'MSLB' in file:
            save_root = 'D:/ie_diagnosis/DATA/Test/MSLB/'
        else:
            save_root = 'D:/ie_diagnosis/DATA/Test/SGTR/'
    filename = os.path.basename(file).replace('.csv','')
    np.save(save_root+filename, temp2)

with open('D:/ie_diagnosis/SCALERS.pickle','wb') as f:
    pickle.dump(SCALERS,f)

#%%
TRAIN_MSLB = sorted(glob.glob('D:/ie_diagnosis/DATA/TRAIN/MSLB/*'))
TRAIN_SGTR = sorted(glob.glob('D:/ie_diagnosis/DATA/TRAIN/SGTR/*'))
SGTR = []
for i, file in tqdm.tqdm(enumerate(TRAIN_SGTR)):
    temp = np.load(file)[::5,:]
    SGTR.append(temp)