#%%
import tensorflow as tf
import numpy as np
import pandas as pd
import tqdm, glob, pickle, datetime, re, time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split

#%%
# Read training/test data
past = time.time()
RAW_TRAIN = np.load('./DATA/Train_v2.npz')
RAW_TEST = np.load('./DATA/Test_v2.npz')
TRAIN = np.concatenate([RAW_TRAIN['MSLB'],RAW_TRAIN['SGTR']])
TRAIN_y = np.concatenate([np.ones(len(RAW_TRAIN['MSLB'])),np.zeros(len(RAW_TRAIN['SGTR']))]) # 1 : MSLB, 0 : SGTR
TEST = np.concatenate([RAW_TEST['MSLB'],RAW_TEST['SGTR']])
TEST_y = np.concatenate([np.ones(len(RAW_TEST['MSLB'])),np.zeros(len(RAW_TEST['SGTR']))]) # 1 : MSLB, 0 : SGTR
print('Read in {}'.format(time.time()-past))
#%% Data Normalization
with open('F:/ie_diagnosis/SCALERS.pickle','rb') as f:
    SCALERS=pickle.load(f)
for index, values in tqdm.tqdm(enumerate(TRAIN)):
    TRAIN[index,...] = SCALERS['minmax'].transform(values)
for index, values in tqdm.tqdm(enumerate(TEST)):
    TEST[index,...] = SCALERS['minmax'].transform(values)
#%%
def base_model(inp_shape):         
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(64,3,strides=2,activation='selu',input_shape=inp_shape))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv1D(64,3,strides=2,activation='selu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv1D(128,3,strides=2,activation='selu'))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam',loss = 'binary_crossentropy', metrics=['accuracy'])
    return model

#%%
# Case 1 : 300
# TRAIN2 = np.delete(TRAIN,6,2)
# TEST2 =np.delete(TEST,6,2)
TR_X,VAL_X, TR_Y, VAL_Y = train_test_split(TRAIN, TRAIN_y, test_size=0.2, random_state=0)

model1 = base_model((300,19))
model2 = base_model((600,19))
model3 = base_model((900,19))
history1 = model1.fit(TR_X[:,:300,...],TR_Y, validation_data=(VAL_X[:,:300,...],VAL_Y), epochs = 10, batch_size = 64)
history2 = model2.fit(TR_X[:,:600,...],TR_Y, validation_data=(VAL_X[:,:600,...],VAL_Y), epochs = 10, batch_size = 64)
history3 = model3.fit(TR_X[:,:900,...],TR_Y, validation_data=(VAL_X[:,:900,...],VAL_Y), epochs = 10, batch_size = 64)

model1.evaluate(TEST[:,:300,...],TEST_y)
model2.evaluate(TEST[:,:600,...],TEST_y)
model3.evaluate(TEST[:,:900,...],TEST_y)

#%%
plt.pcolor(TRAIN[60000,...].T)

#%%

fpr, tpr, threshold = roc_curve(TEST_y,TEST_y_hat)
roc_auc = auc(fpr,tpr)
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='AUC = %0.4f' % roc_auc)
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