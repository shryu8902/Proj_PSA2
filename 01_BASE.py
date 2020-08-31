#%%
import tensorflow as tf
import numpy as np
import pandas as pd
import tqdm, glob, pickle, datetime, re, time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

#%%
# Read training/test data
past = time.time()
RAW_TRAIN = np.load('./DATA/Train.npz')
RAW_TEST = np.load('./DATA/Test.npz')
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
def reformulator(X):
    X_reform = []
    for index, value in X:
        x_mean = np.mean(value,axis=0)
        x_std = np.mean(value,axis=0)
        X_reform.append(np.concatenate([x_mean,x_std]))
    return(np.array(X_reform))    

TRAIN_x_reform = reformulator(TRAIN)
#%%
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D(64,3,strides=2,activation='selu',input_shape=(720,19)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv1D(64,3,strides=2,activation='selu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv1D(128,3,strides=2,activation='selu'))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss = 'binary_crossentropy', metrics=['accuracy'])
#%%
model.fit(TRAIN,TRAIN_y,epochs=10, batch_size = 64)

#%%
TRAIN_y_hat = model.predict(TRAIN)
#%%
model.evaluate(TEST,TEST_y)
TEST_y_hat = model.predict(TEST)
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