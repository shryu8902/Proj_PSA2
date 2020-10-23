#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import tqdm, glob, pickle, datetime, re
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
#%%
# Read configurations
SEED=0

#%%
input_ie = pd.DataFrame(np.load('D:/ie_diagnosis/input_ie.npy'))
input_addmslb = pd.read_csv('D:/ie_diagnosis_add/ie_input_addmslb.csv',header=None)
input_addsgtr = pd.read_csv('D:/ie_diagnosis_add/ie_input_addsgtr.csv',header=None)
input_all=input_ie.append(input_addmslb).append(input_addsgtr)
input_all_sorted = input_all.sort_values(input_all.columns[0], ascending = True)
input_all_sorted = input_all_sorted.reset_index(drop=True)
input_all_sorted[0]=input_all_sorted[0].astype('int64')
input_all_sorted.set_index(input_all_sorted.columns[0],copy=True)
input_all_sorted[1]=input_all_sorted[1].astype('int64')
input_all_sorted[16]=input_all_sorted[16].astype('int64')

with open('D:/ie_201008/input_all.pickle','wb') as f:
    pickle.dump(input_all_sorted,f)
#%%
#%%
TRAIN_files = sorted(glob.glob('D:/ie_201008/train/*.csv'))
TRAIN_index = sorted([int(os.path.basename(x).split('.csv')[0]) for x in TRAIN_files])

TEST_files = sorted(glob.glob('D:/ie_201008/test/*.csv'))
TEST_index = sorted([int(os.path.basename(x).split('.csv')[0]) for x in TEST_files])

SCALERS = {'standard':StandardScaler(), 'minmax':MinMaxScaler()}
for file in tqdm.tqdm(TRAIN_files + TEST_files):
    temp = pd.read_csv(file,header=None)
    temp2 = np.float32(temp.values[...,1:])
    assert temp2.shape==(360,19)
    SCALERS['standard'].partial_fit(temp2)
    SCALERS['minmax'].partial_fit(temp2)
    save_root = 'D:/ie_201008/TT/'
    filename = os.path.basename(file).replace('.csv','')
    np.save(save_root+filename, temp2) 
with open('D:/ie_201008/TT_SCALERS.pickle','wb') as f:
    pickle.dump(SCALERS,f)

#%%
# with open('D:/ie_201008/input_all.pickle','rb') as f:
#     input_ie = pickle.load(f)
# TEST_ie = input_ie.set_index(input_ie[0]).loc[TEST_index]
# TRAIN_ie = input_ie.set_index(input_ie[0]).loc[TRAIN_index]
# with open('D:/ie_201008/TRAIN_ie.pickle','wb') as f:
#     pickle.dump(TRAIN_ie,f)
# with open('D:/ie_201008/TEST_ie.pickle','wb') as f:
#     pickle.dump(TEST_ie,f)
with open('F:/ie_201008/TRAIN_ie.pickle','rb') as f:
    TRAIN_ie = pickle.load(f)
with open('F:/ie_201008/TEST_ie.pickle','rb') as f:
    TEST_ie = pickle.load(f)

#%%
name = 'TEST'
if name == 'TRAIN':
    ie = TRAIN_ie
elif name == 'TEST':
    ie = TEST_ie
act = []
trip = []

label = []
for file_index in tqdm.tqdm(ie.index):
    save_root = 'F:/ie_201008/TT/'
    file_root = save_root + str(file_index)+'.npy'
    temp = np.load(file_root)
    start = np.where(temp[...,0]<=50.0)[0][0]
    last_act = np.max(ie.loc[file_index][2:16:2])
    start_ = np.int((np.ceil(start+last_act*60/10)))

    temp_after_action = temp[start_:start_+90,...]
    temp_after_trip = temp[start:start+90,...]
    act.append(temp_after_action) 
    trip.append(temp_after_trip)

    loca_size, mslb = ie.loc[file_index][[1,16]]
    if mslb:
        if loca_size == 1:
            label.append(0)
        elif loca_size == 2:
            label.append(1)
        else :
            raise ValueError
    elif mslb == 0:
        if loca_size ==1:
            label.append(2)
        elif loca_size ==2:
            label.append(3)
        elif loca_size == 4:
            label.append(4)
        else :
            raise ValueError
    else:
        raise ValueError
act = np.array(act)
trip = np.array(trip)
label = np.array(label)
ensure_dir('./DATA/')
np.save('./DATA/{}_after_action.npy'.format(name),act)
np.save('./DATA/{}_after_trip.npy'.format(name),trip)
np.save('./DATA/{}_label.npy'.format(name),label)
## Previously : label was 1 for MSLB and 0 for SGTR.
## Class labels = MSLB,1 = 0, MSLB,2 =1, SGTR,1 = 2, SGTR,2=3, SGTR,4 = 4


