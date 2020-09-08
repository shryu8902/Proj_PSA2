#%%
RAW_TEST = np.load('./DATA/Test_v2.npz')
#%%
parity = []
for i in range(4000):
    parity.append(np.sum(RAW_TEST['MSLB'][i,...]))