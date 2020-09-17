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

#%%
# Read training/test data
past = time.time()
RAW_TRAIN = np.load('./DATA/Train_v2.npz')
RAW_TEST = np.load('./DATA/Test_v2.npz')
RAW_TRAIN_label = np.load('./DATA/TRAIN_class.npz')
RAW_TEST_label = np.load('./DATA/TEST_class.npz')

RAW_TRAIN_add = np.load('./DATA/Train_add_v2.npz')
RAW_TEST_add = np.load('./DATA/Test_add_v2.npz')
RAW_TRAIN_add_label = np.load('./DATA/TRAIN_add_class.npz')
RAW_TEST_add_label = np.load('./DATA/TEST_add_class.npz')

# 0 : MSLB, 1 
# 1 : MSLB, 2
# 2 : SGTR, 1
# 3 : SGTR, 2
# 4 : SGTR, 4
 
TRAIN = np.concatenate([RAW_TRAIN['MSLB'],RAW_TRAIN_add['MSLB'],RAW_TRAIN['SGTR'],RAW_TRAIN_add['SGTR']])
TRAIN_y = np.concatenate([RAW_TRAIN_label['MSLB'],RAW_TRAIN_add_label['MSLB'],RAW_TRAIN_label['SGTR'],RAW_TRAIN_add_label['SGTR']])

TEST = np.concatenate([RAW_TEST['MSLB'],RAW_TEST_add['MSLB'],RAW_TEST['SGTR'],RAW_TEST_add['SGTR']])
TEST_y = np.concatenate([RAW_TEST_label['MSLB'],RAW_TEST_add_label['MSLB'],RAW_TEST_label['SGTR'],RAW_TEST_add_label['SGTR']])

del(RAW_TRAIN,RAW_TRAIN_add,RAW_TRAIN_label, RAW_TRAIN_add_label,RAW_TEST,RAW_TEST_add,RAW_TEST_add_label,RAW_TEST_label)
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
    model.add(tf.keras.layers.Dense(5,activation='softmax'))
    model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics=['accuracy'])
    return model

#%%
# Case 1 : 300
# Case 2 : 600
# Case 3 : 900
# TRAIN2 = np.delete(TRAIN,6,2)
# TEST2 =np.delete(TEST,6,2)
index=[x for x in range(len(TRAIN))]
TR_ind, VAL_ind = train_test_split(index, test_size=0.2, random_state=0)
# TR_X,VAL_X, TR_Y, VAL_Y = train_test_split(TRAIN, TRAIN_y, test_size=0.2, random_state=0)

model1 = base_model((300,19))
# model2 = base_model((600,19))
# model3 = base_model((900,19))

history1 = model1.fit(TRAIN[TR_ind,:300,...],TRAIN_y[TR_ind,...], validation_data=(TRAIN[VAL_ind,:300,...],TRAIN_y[VAL_ind,...]), epochs=10, batch_size=64)



history1 = model1.fit(TR_X[:,:300,...],TR_Y, validation_data=(VAL_X[:,:300,...],VAL_Y), epochs = 10, batch_size = 64)
# history2 = model2.fit(TR_X[:,:600,...],TR_Y, validation_data=(VAL_X[:,:600,...],VAL_Y), epochs = 10, batch_size = 64)
# history3 = model3.fit(TR_X[:,:900,...],TR_Y, validation_data=(VAL_X[:,:900,...],VAL_Y), epochs = 10, batch_size = 64)

model1.evaluate(TEST[:,:300,...],TEST_y)
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