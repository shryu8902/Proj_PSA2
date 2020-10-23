#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
import numpy as np
import pandas as pd
import tqdm, glob, pickle, datetime, re, time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import gc
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

#%%
train_input_ = np.load('F:/gen_data/train_input.npy')[:,1:]
val_input_ = np.load('F:/gen_data/test_input.npy')[:,1:]
test_input_ = np.load('F:/gen_data/untrain_input.npy')[:,1:]
train_output_ = np.load('F:/gen_data/train_output.npy')
val_output_ = np.load('F:/gen_data/test_output.npy')
test_output_ = np.load('F:/gen_data/untrain_output.npy')
#%%
OUT_SCALER = {'standard':StandardScaler(),'minmax':MinMaxScaler()}
IN_SCALER = {'standard':StandardScaler(),'minmax':MinMaxScaler()}
for dat in [train_input,val_input,test_input]:
    IN_SCALER['standard'].partial_fit(dat)
    IN_SCALER['minmax'].partial_fit(dat)

for dat in [train_output,val_output,test_output]:
    OUT_SCALER['standard'].partial_fit(dat.reshape(-1,1))
    OUT_SCALER['minmax'].partial_fit(dat.reshape(-1,1))
#  with open('./IN_SCALER.pickle','wb') as f:
#      pickle.dump(IN_SCALER,f)
#  with open('./OUT_SCALER.pickle','wb') as f:
#      pickle.dump(OUT_SCALER,f)
with open('./IN_SCALER.pickle','rb') as f:
    IN_SCALER = pickle.load(f)
with open('./OUT_SCALER.pickle','rb') as f:
    OUT_SCALER = pickle.load(f)

norm_train_input = IN_SCALER['minmax'].transform(train_input)
with open('F:/ie_diagnosis/SCALERS.pickle','rb') as f:
    SCALERS=pickle.load(f)
SCALE = 'minmax'
train_input = IN_SCALER[SCALE].transform(train_input_)
val_input = IN_SCALER[SCALE].transform(val_input_)
test_input = IN_SCALER[SCALE].transform(test_input_)
train_output = OUT_SCALER[SCALE].transform(train_output_.reshape(-1,1)).reshape(-1,360)
val_output = OUT_SCALER[SCALE].transform(val_output_.reshape(-1,1)).reshape(-1,360)
test_output = OUT_SCALER[SCALE].transform(test_output_.reshape(-1,1)).reshape(-1,360)

#%%
from tensorflow.keras.layers import Input, RepeatVector,Bidirectional,LSTM,TimeDistributed,Dense
from tensorflow.keras.models import Model
def RNN_model(num_cell=128, dr_rates = 0.3):
    inputs = Input(shape =(16) ,name='input')
    num_rows = tf.shape(inputs,name='num_rows')[0]
    pos_enc_tile = tf.tile(positional_encoding(360,16), [num_rows, 1,1],name='pos_enc_tile')
    inputs_extend = RepeatVector(360,name='extend_inputs')(inputs)
    input_with_PE = pos_enc_tile+inputs_extend
    layer_1 = Bidirectional(LSTM(num_cell, return_sequences=True,dropout=dr_rates))(input_with_PE)
    layer_2 = Bidirectional(LSTM(num_cell, return_sequences=True,dropout=dr_rates))(layer_1)
    outputs = TimeDistributed(Dense(1))(layer_2)
    model= Model(inputs, outputs)
    return model
#%%

base_model = RNN_model(128,0.3)
base_model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mean_absolute_error','mean_squared_error']) 
base_model.fit(train_input,train_output,validation_data=(val_input,val_output),
                batch_size = 512, epochs=10)
#%% 
        history = model.fit(X_train, Y_train.reshape(-1,520,1), 
                        validation_data=(X_val,Y_val),
                        batch_size = 512, epochs=100,
                        callbacks=[early_stopping,model_save]))
 
#%%


# Add positional embedding a. channelwise, b. just add to original data
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.convert_to_tensor(pos_encoding, dtype=tf.float32)