

# Read training/test data
past = time.time()
RAW_TRAIN = np.load('./DATA/Train_v2.npz')
RAW_TEST = np.load('./DATA/Test_v2.npz')
TRAIN = np.concatenate([RAW_TRAIN['MSLB'],RAW_TRAIN['SGTR']])
TRAIN_y = np.concatenate([np.ones(len(RAW_TRAIN['MSLB'])),np.zeros(len(RAW_TRAIN['SGTR']))]) # 1 : MSLB, 0 : SGTR
TEST = np.concatenate([RAW_TEST['MSLB'],RAW_TEST['SGTR']])
TEST_y = np.concatenate([np.ones(len(RAW_TEST['MSLB'])),np.zeros(len(RAW_TEST['SGTR']))]) # 1 : MSLB, 0 : SGTR
print('Read in {}'.format(time.time()-past))