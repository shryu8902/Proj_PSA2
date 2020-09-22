#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import config as cfg
from configparser import ConfigParser as cfg
import tqdm, glob, pickle, datetime, re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
#%%
# Read configurations
cfg=cfg()
cfg.read('./PSA.conf')
WINDOW=int(cfg['MAIN']['WINDOW'])
STRIDE=int(cfg['MAIN']['STRIDE'])
SAMPLING_RATE=int(cfg['MAIN']['SAMPLING_RATE'])
SEED=0

#%% 
# Code for converting to pickle files
# Data are already saved in external HDD
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
            save_root = 'F:/ie_diagnosis/DATA/Train/MSLB/'
        else:
            save_root = 'F:/ie_diagnosis/DATA/Train/SGTR/'
    elif file in Test_list:
        if 'MSLB' in file:
            save_root = 'F:/ie_diagnosis/DATA/Test/MSLB/'
        else:
            save_root = 'F:/ie_diagnosis/DATA/Test/SGTR/'
    filename = os.path.basename(file).replace('.csv','')
    np.save(save_root+filename, temp2)

with open('F:/ie_diagnosis/SCALERS.pickle','wb') as f:
    pickle.dump(SCALERS,f)

#%% 
# Below codes are used for creating train/test data
TRAIN_MSLB = sorted(glob.glob('F:/ie_diagnosis/DATA/TRAIN/MSLB/*'))
TRAIN_SGTR = sorted(glob.glob('F:/ie_diagnosis/DATA/TRAIN/SGTR/*'))
CODE_inputs = np.load('F:/ie_diagnosis/input_ie.npy')
# break size = CODE_inputs[:,1]
SGTR = []
for i,file in tqdm.tqdm(enumerate(TRAIN_SGTR)):
    file_index = int(re.search('Run-(.*?).npy', file).group(1))-1 
    temp = np.load(file)[:3600,:]
    start = np.where(temp[...,0]<=50.0)[0][0]
    last_act = np.max(CODE_inputs[file_index,2:16:2])
    start_ = np.int((np.ceil(start+last_act*60)))
    temp_after_trip = temp[start_:start_+900,...]
    SGTR.append(temp_after_trip)
SGTR = np.array(SGTR)
MSLB = []
for i,file in tqdm.tqdm(enumerate(TRAIN_MSLB)):
    file_index = int(re.search('Run-(.*?).npy', file).group(1))-1+42687 
    temp = np.load(file)[:3600,:]
    start = np.where(temp[...,0]<=50.0)[0][0]
    last_act = np.max(CODE_inputs[file_index,2:16:2])
    start_ = np.int((np.ceil(start+last_act*60)))
    temp_after_trip = temp[start_:start_+900,...]
    MSLB.append(temp_after_trip)
MSLB=np.array(MSLB)
np.savez('./DATA/Train_v3',MSLB=MSLB,SGTR=SGTR)

#%%
TEST_MSLB = sorted(glob.glob('F:/ie_diagnosis/DATA/TEST/MSLB/*'))
TEST_SGTR = sorted(glob.glob('F:/ie_diagnosis/DATA/TEST/SGTR/*'))
CODE_inputs = np.load('F:/ie_diagnosis/input_ie.npy')

SGTR = []
for i, file in tqdm.tqdm(enumerate(TEST_SGTR)):
    file_index = int(re.search('Run-(.*?).npy', file).group(1))-1 
    temp = np.load(file)[:3600,:]
    start = np.where(temp[...,0]<=50.0)[0][0]
    last_act = np.max(CODE_inputs[file_index,2:16:2])
    start_ = np.int((np.ceil(start+last_act*60)))
    temp_after_trip = temp[start_:start_+900,...]
    SGTR.append(temp_after_trip)
SGTR = np.array(SGTR)

MSLB = []
for i, file in tqdm.tqdm(enumerate(TEST_MSLB)):
    file_index = int(re.search('Run-(.*?).npy', file).group(1))-1+42687 
    temp = np.load(file)[:3600,:]
    start = np.where(temp[...,0]<=50.0)[0][0]
    last_act = np.max(CODE_inputs[file_index,2:16:2])
    start_ = np.int((np.ceil(start+last_act*60)))
    temp_after_trip = temp[start_:start_+900,...]
    MSLB.append(temp_after_trip)
MSLB = np.array(MSLB)
np.savez('./DATA/Test_v3',MSLB=MSLB,SGTR=SGTR)
#%%
TRAIN=np.load('./DATA/Train.npz')
TEST = np.load('./DATA/Test.npz')

#%%
MSLB_add_list = sorted(glob.glob('D:/ie_diagnosis/MSLB_Additional_csv/*.csv'))
SGTR_add_list = sorted(glob.glob('D:/ie_diagnosis/SGTR_Additional_csv/*.csv'))
All_add_list = MSLB_add_list + SGTR_add_list
ClassLabel = np.concatenate([np.ones(np.shape(MSLB_add_list)),np.zeros(np.shape(SGTR_add_list))]) # 1 : MSLB, 0 : SGTR
Train_list, Test_list = train_test_split(All_add_list, test_size = 10000, random_state = SEED, stratify = ClassLabel)

with open('D:/ie_diagnosis_add/train_test.pickle','wb') as f:
    train_test_list = {'Train':Train_list, 'Test':Test_list}
    pickle.dump(train_test_split,f)
#%%
ensure_dir('D:/ie_diagnosis_add/DATA/Train/MSLB/')
ensure_dir('D:/ie_diagnosis_add/DATA/Test/MSLB/')
ensure_dir('D:/ie_diagnosis_add/DATA/Train/SGTR/')
ensure_dir('D:/ie_diagnosis_add/DATA/Test/SGTR/')
#%%
for file in tqdm.tqdm(All_add_list):
    temp = pd.read_csv(file)
    temp2 = np.float32(temp.values[...,1:])
    # SCALERS['standard'].partial_fit(temp2)
    # SCALERS['minmax'].partial_fit(temp2)
    if file in Train_list:
        if 'MSLB' in file:
            save_root = 'D:/ie_diagnosis_add/DATA/Train/MSLB/'
        else:
            save_root = 'D:/ie_diagnosis_add/DATA/Train/SGTR/'
    elif file in Test_list:
        if 'MSLB' in file:
            save_root = 'D:/ie_diagnosis_add/DATA/Test/MSLB/'
        else:
            save_root = 'D:/ie_diagnosis_add/DATA/Test/SGTR/'
    filename = os.path.basename(file).replace('.csv','')
    np.save(save_root+filename, temp2)

#%%
# Below codes are used for creating train/test data
TRAIN_MSLB = sorted(glob.glob('F:/ie_diagnosis_add/DATA/TRAIN/MSLB/*'))
TRAIN_SGTR = sorted(glob.glob('F:/ie_diagnosis_add/DATA/TRAIN/SGTR/*'))
MSLB_inputs = np.array(pd.read_csv('F:/ie_diagnosis_add/ie_input_addmslb.csv',header=None))
SGTR_inputs = np.array(pd.read_csv('F:/ie_diagnosis_add/ie_input_addsgtr.csv',header=None))

SGTR = []
for i,file in tqdm.tqdm(enumerate(TRAIN_SGTR)):
    file_index = int(re.search('Run-(.*?).npy', file).group(1))-1 
    temp = np.load(file)[:3600,:]
    start = np.where(temp[...,0]<=50.0)[0][0]
    last_act = np.max(SGTR_inputs[file_index,2:16:2])
    start_ = np.int((np.ceil(start+last_act*60)))
    temp_after_trip = temp[start_:start_+900,...]
    SGTR.append(temp_after_trip)
SGTR = np.array(SGTR)
MSLB = []
for i,file in tqdm.tqdm(enumerate(TRAIN_MSLB)):
    file_index = int(re.search('Run-(.*?).npy', file).group(1))-1 
    temp = np.load(file)[:3600,:]
    start = np.where(temp[...,0]<=50.0)[0][0]
    last_act = np.max(MSLB_inputs[file_index,2:16:2])
    start_ = np.int((np.ceil(start+last_act*60)))
    temp_after_trip = temp[start_:start_+900,...]
    MSLB.append(temp_after_trip)
MSLB=np.array(MSLB)
np.savez('./DATA/Train_add_v3',MSLB=MSLB,SGTR=SGTR)
#%%
TEST_MSLB = sorted(glob.glob('F:/ie_diagnosis_add/DATA/TEST/MSLB/*'))
TEST_SGTR = sorted(glob.glob('F:/ie_diagnosis_add/DATA/TEST/SGTR/*'))
MSLB_inputs = np.array(pd.read_csv('F:/ie_diagnosis_add/ie_input_addmslb.csv',header=None))
SGTR_inputs = np.array(pd.read_csv('F:/ie_diagnosis_add/ie_input_addsgtr.csv',header=None))

SGTR = []
for i, file in tqdm.tqdm(enumerate(TEST_SGTR)):
    file_index = int(re.search('Run-(.*?).npy', file).group(1))-1 
    temp = np.load(file)[:3600,:]
    start = np.where(temp[...,0]<=50.0)[0][0]
    last_act = np.max(SGTR_inputs[file_index,2:16:2])
    start_ = np.int((np.ceil(start+last_act*60)))
    temp_after_trip = temp[start_:start_+900,...]
    SGTR.append(temp_after_trip)
SGTR = np.array(SGTR)

MSLB = []
for i, file in tqdm.tqdm(enumerate(TEST_MSLB)):
    file_index = int(re.search('Run-(.*?).npy', file).group(1))-1 
    temp = np.load(file)[:3600,:]
    start = np.where(temp[...,0]<=50.0)[0][0]
    last_act = np.max(MSLB_inputs[file_index,2:16:2])
    start_ = np.int((np.ceil(start+last_act*60)))
    temp_after_trip = temp[start_:start_+900,...]
    MSLB.append(temp_after_trip)
MSLB = np.array(MSLB)
np.savez('./DATA/Test_add_v3',MSLB=MSLB,SGTR=SGTR)
#%%
TRAIN = np.load('./DATA/Train_v2.npz')
TEST = np.load('./DATA/Test_v2.npz')



#%% 
############       Create label data by pair of event type & LOCA size         ##################
## event type : MSLB or SGTR
## Break size : 1, 2, 4 
## Previously : label was 1 for MSLB and 0 for SGTR.
## Class labels = MSLB,1 = 0, MSLB,2 =1, SGTR,1 = 2, SGTR,2=3, SGTR,4 = 4

CODE_inputs = np.load('D:/ie_diagnosis/input_ie.npy')
# break size = CODE_inputs[:,1]

TRAIN_MSLB = sorted(glob.glob('D:/ie_diagnosis/DATA/TRAIN/MSLB/*'))
file_index = [int(re.search('Run-(.*?).npy', x).group(1))-1+42687 for x in TRAIN_MSLB]
TRAIN_MSLB_Class = tf.one_hot([0 if x == 1 else 1 if x==2 else 3 for x in CODE_inputs[file_index,1]],5).numpy()

TRAIN_SGTR = sorted(glob.glob('D:/ie_diagnosis/DATA/TRAIN/SGTR/*'))
file_index = [int(re.search('Run-(.*?).npy', x).group(1))-1 for x in TRAIN_SGTR]
TRAIN_SGTR_Class = tf.one_hot([2 if x == 1 else 3 if x==2 else 4 for x in CODE_inputs[file_index,1]],5).numpy()

TEST_MSLB = sorted(glob.glob('D:/ie_diagnosis/DATA/TEST/MSLB/*'))
file_index = [int(re.search('Run-(.*?).npy', x).group(1))-1+42687 for x in TEST_MSLB]
TEST_MSLB_Class = tf.one_hot([0 if x == 1 else 1 if x==2 else 3 for x in CODE_inputs[file_index,1]],5).numpy()

TEST_SGTR = sorted(glob.glob('D:/ie_diagnosis/DATA/TEST/SGTR/*'))
file_index = [int(re.search('Run-(.*?).npy', x).group(1))-1 for x in TEST_SGTR]
TEST_SGTR_Class = tf.one_hot([2 if x == 1 else 3 if x==2 else 4 for x in CODE_inputs[file_index,1]],5).numpy()

np.savez('./DATA/TRAIN_class',MSLB=TRAIN_MSLB_Class,SGTR=TRAIN_SGTR_Class)
np.savez('./DATA/TEST_class',MSLB=TEST_MSLB_Class,SGTR=TEST_SGTR_Class)

#%%
############        F O R       A D I T I O N A L       D A T A             ####################
############       Create label data by pair of event type & LOCA size      ####################

MSLB_inputs = np.array(pd.read_csv('D:/ie_diagnosis_add/ie_input_addmslb.csv',header=None))
SGTR_inputs = np.array(pd.read_csv('D:/ie_diagnosis_add/ie_input_addsgtr.csv',header=None))

TRAIN_MSLB = sorted(glob.glob('D:/ie_diagnosis_add/DATA/TRAIN/MSLB/*'))
file_index = [int(re.search('Run-(.*?).npy', x).group(1))-1 for x in TRAIN_MSLB]
TRAIN_MSLB_Class = tf.one_hot([0 if x == 1 else 1 if x==2 else 3 for x in MSLB_inputs[file_index,1]],5).numpy()

TRAIN_SGTR = sorted(glob.glob('D:/ie_diagnosis_add/DATA/TRAIN/SGTR/*'))
file_index = [int(re.search('Run-(.*?).npy', x).group(1))-1 for x in TRAIN_SGTR]
TRAIN_SGTR_Class = tf.one_hot([2 if x == 1 else 3 if x==2 else 4 for x in SGTR_inputs[file_index,1]],5).numpy()

TEST_MSLB = sorted(glob.glob('D:/ie_diagnosis_add/DATA/TEST/MSLB/*'))
file_index = [int(re.search('Run-(.*?).npy', x).group(1))-1 for x in TEST_MSLB]
TEST_MSLB_Class = tf.one_hot([0 if x == 1 else 1 if x==2 else 3 for x in MSLB_inputs[file_index,1]],5).numpy()

TEST_SGTR = sorted(glob.glob('D:/ie_diagnosis_add/DATA/TEST/SGTR/*'))
file_index = [int(re.search('Run-(.*?).npy', x).group(1))-1 for x in TEST_SGTR]
TEST_SGTR_Class = tf.one_hot([2 if x == 1 else 3 if x==2 else 4 for x in SGTR_inputs[file_index,1]],5).numpy()

np.savez('./DATA/TRAIN_add_class',MSLB=TRAIN_MSLB_Class,SGTR=TRAIN_SGTR_Class)
np.savez('./DATA/TEST_add_class',MSLB=TEST_MSLB_Class,SGTR=TEST_SGTR_Class)


