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
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import gc
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

#%%
# Read training/test data
with open('F:/ie_201008/TRAIN_ie.pickle','rb') as f:
    TRAIN_ie = pickle.load(f)
with open('F:/ie_201008/TEST_ie.pickle','rb') as f:
    TEST_ie = pickle.load(f)
#%%############        MODEL  FOR  ATER    ACTION      ################ 
TRAIN = np.load('./DATA/TRAIN_after_trip.npy')
TRAIN_y= tf.one_hot(np.load('./DATA/TRAIN_label.npy'),depth=5).numpy()
TEST = np.load('./DATA/TEST_after_trip.npy')
TEST_y= tf.one_hot(np.load('./DATA/TEST_label.npy'),depth=5).numpy()

#%% ############    NORMALIZING     ################ 

with open('F:/ie_diagnosis/SCALERS.pickle','rb') as f:
    SCALERS=pickle.load(f)
for index, values in tqdm.tqdm(enumerate(TRAIN)):
    TRAIN[index,...] = SCALERS['minmax'].transform(values)
for index, values in tqdm.tqdm(enumerate(TEST)):
    TEST[index,...] = SCALERS['minmax'].transform(values)
#%%
def base_model(inp_shape):         
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(128,3,strides=3,input_shape=inp_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv1D(64,3,strides=3))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv1D(64,3,strides=3))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64,activation='selu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(5,activation='softmax'))
    model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics=['accuracy'])
    return model
#%%
def base_RNN_model(inp_shape):         
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,return_sequences=True)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,return_sequences=True)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=False)))
    model.add(tf.keras.layers.Dense(64,activation='selu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(5,activation='softmax'))
    model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics=['accuracy'])
    return model
    
#%%
# Case 1 : 300
# Case 2 : 600
# Case 3 : 900
# TRAIN2 = np.delete(TRAIN,6,2)
# TEST2 =np.delete(TEST,6,2)

TR_X, VAL_X, TR_Y, VAL_Y = train_test_split(TRAIN, TRAIN_y, test_size=0.1, random_state=0)
# TR_X,VAL_X, TR_Y, VAL_Y = train_test_split(TRAIN, TRAIN_y, test_size=0.2, random_state=0)
losses = pd.DataFrame()
#%%
for i in [30,60,90]:
    model = base_RNN_model((i,19))
    ensure_dir('./Model/RNN2/base_at{}'.format(i))
    path = './Model/RNN2/base_at{}'.format(i) + '/base{epoch:04d}.ckpt'
    checkpoint = ModelCheckpoint(path, monitor = 'val_loss',verbose = 1,
                save_best_only = True,
                mode = 'auto',
                save_weights_only = True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-5)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    hist=model.fit(TR_X[:,:i,:],TR_Y,
               validation_data=(VAL_X[:,:i,:],VAL_Y),
               callbacks=[checkpoint,reduce_lr,early_stopping],epochs=100,batch_size=256)
    with open('./Model/RNN2/base_at{}/hist.pkl'.format(i), 'wb') as f:
          pickle.dump(hist.history,f)
    del(hist)
    TR_l, TR_a = model.evaluate(TR_X[:,:i,:],TR_Y, batch_size = 256)
    VA_l, VA_a = model.evaluate(VAL_X[:,:i,:],VAL_Y, batch_size = 256)
    TE_l, TE_a = model.evaluate(TEST[:,:i,:],TEST_y,batch_size=256)
    K.clear_session()
    del(model)
    losses = losses.append({'SEC': i*10, 'TR_loss': TR_l, 'TR_acc': TR_a, 'VAL_loss': VA_l,
                    'VAL_acc':VA_a,'TEST_loss':TE_l,'TEST_acc':TE_a}, ignore_index=True)
    gc.collect()
    gc.enable()

#%%
with open('./Model/RNN2/base_at.pkl','wb') as f:
    pickle.dump(losses,f)
#%%
with open('./Model/RNN2/base_aa60/hist.pkl','rb') as f:
    hist = pickle.load(f)

plt.plot(hist['loss'],label = 'train_loss')
plt.plot(hist['val_loss'],label ='val_loss')
plt.legend()
plt.plot(hist['accuracy'],label = 'train_loss')
plt.plot(hist['val_accuracy'],label ='val_loss')
plt.legend()
#%%
with open('./Model/base_act3.pkl','rb') as f:
    losses = pickle.load(f)
#%%
with open('./Model/1D_CNN/base_aa.pkl','rb') as f:
    aa_CNN_losses = pickle.load(f)
with open('./Model/1D_CNN/base_at.pkl','rb') as f:
    at_CNN_losses = pickle.load(f)
with open('./Model/RNN2/base_aa.pkl','rb') as f:
    aa_RNN_losses = pickle.load(f)
with open('./Model/RNN2/base_at.pkl','rb') as f:
    at_RNN_losses = pickle.load(f)


#%%

# model2 = base_model((600,19))
# model3 = base_model((900,19))

history1 = model1.fit(TR_X[:,:300,...],TR_Y, validation_data=(VAL_X[:,:300,...],VAL_Y), epochs = 10, batch_size = 64)
# history2 = model2.fit(TR_X[:,:600,...],TR_Y, validation_data=(VAL_X[:,:600,...],VAL_Y), epochs = 10, batch_size = 64)
# history3 = model3.fit(TR_X[:,:900,...],TR_Y, validation_data=(VAL_X[:,:900,...],VAL_Y), epochs = 10, batch_size = 64)

# model2.evaluate(TEST[:,:600,...],TEST_y)
# model3.evaluate(TEST[:,:900,...],TEST_y)

#%%
plt.pcolor(TRAIN[60000,...].T)

#%%
fpr_1, tpr_1, threshold_1 = roc_curve(TEST_y,model1.predict(TEST[:,:300,:]))
fpr_2, tpr_2, threshold_2 = roc_curve(TEST_y,model2.predict(TEST[:,:600,:]))
fpr_3, tpr_3, threshold_3 = roc_curve(TEST_y,model3.predict(TEST[:,:900,:]))

roc_auc_1 = auc(fpr_1,tpr_1)
roc_auc_2 = auc(fpr_2,tpr_2)
roc_auc_3 = auc(fpr_3,tpr_3)

plt.plot(fpr_1, tpr_1, color='darkorange',
         lw=2, label='AUC = %0.4f' % roc_auc_1)
plt.plot(fpr_2, tpr_2, color='darkgreen',
         lw=2, label='AUC = %0.4f' % roc_auc_2)
plt.plot(fpr_3, tpr_3, color='darkblue',
         lw=2, label='AUC = %0.4f' % roc_auc_3)

# plt.plot(fpr_gan, tpr_gan, color='darkgreen',
#          lw=2, label='GANomaly (AUC = %0.4f)' % roc_auc_gan)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([-0.05, 1.0])
# plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('./ROCURVE.png')