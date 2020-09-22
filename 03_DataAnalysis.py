#%%
past = time.time()
RAW_TRAIN = np.load('./DATA/Train_v3.npz')
RAW_TEST = np.load('./DATA/Test_v3.npz')
RAW_TRAIN_label = np.load('./DATA/TRAIN_class.npz')
RAW_TEST_label = np.load('./DATA/TEST_class.npz')

RAW_TRAIN_add = np.load('./DATA/Train_add_v3.npz')
RAW_TEST_add = np.load('./DATA/Test_add_v3.npz')
RAW_TRAIN_add_label = np.load('./DATA/TRAIN_add_class.npz')
RAW_TEST_add_label = np.load('./DATA/TEST_add_class.npz')

int_train_label_mslb = np.argmax(RAW_TRAIN_label['MSLB'], axis=1)
int_train_label_sgtr = np.argmax(RAW_TRAIN_label['SGTR'], axis=1)
int_test_label_mslb = np.argmax(RAW_TEST_label['MSLB'], axis=1)
int_test_label_sgtr = np.argmax(RAW_TEST_label['SGTR'], axis=1)
int_train_add_label_mslb = np.argmax(RAW_TRAIN_add_label['MSLB'], axis=1)
int_train_add_label_sgtr = np.argmax(RAW_TRAIN_add_label['SGTR'], axis=1)
int_test_add_label_mslb = np.argmax(RAW_TEST_add_label['MSLB'], axis=1)
int_test_add_label_sgtr = np.argmax(RAW_TEST_add_label['SGTR'], axis=1)

TRAIN = np.concatenate([RAW_TRAIN['MSLB'],RAW_TRAIN_add['MSLB'],RAW_TRAIN['SGTR'],RAW_TRAIN_add['SGTR']])
TRAIN = np.float16(TRAIN[:,::10,:])
TRAIN_y = np.concatenate([int_train_label_mslb,int_train_add_label_mslb,int_train_label_sgtr,int_train_add_label_sgtr])

TEST = np.concatenate([RAW_TEST['MSLB'],RAW_TEST_add['MSLB'],RAW_TEST['SGTR'],RAW_TEST_add['SGTR']])
TEST = np.float16(TEST[:,::10,:])
TEST_y = np.concatenate([int_test_label_mslb,int_test_add_label_mslb,int_test_label_sgtr,int_test_add_label_sgtr])

del(RAW_TRAIN,RAW_TRAIN_add,RAW_TRAIN_label, RAW_TRAIN_add_label,RAW_TEST,RAW_TEST_add,RAW_TEST_add_label,RAW_TEST_label)

#%%
for ch in range(19):
    ch= 0
    for cls in range(5):
        TR_index = np.where(TRAIN_y==cls)[0]
        TE_index = np.where(TEST_y==cls)[0]
        plt.subplot(5,2,2*cls+1)
        for i in TR_index:
            plt.plot(TRAIN[TR_index,::5,ch])
        plt.subplot(5,2,2*cls+2)
        for i in TE_index:
            plt.plot(TEST[TE_index,::5,ch])

#%%
    for i in 
        plt.plot(TRAIN)
#%%
RAW_TEST = np.load('./DATA/Test_v2.npz')
#%%
parity = []
for i in range(4000):
    parity.append(np.sum(RAW_TEST['MSLB'][i,...]))