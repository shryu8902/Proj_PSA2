#%%
#import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

def reformulator(X):
    X_reform = []
    for index, value in enumerate(X):
        x_mean = np.mean(value,axis=0)
        x_std = np.mean(value,axis=0)
        X_reform.append(np.concatenate([x_mean,x_std]))
    return(np.array(X_reform))    

#%%
history1 = model1.fit(TRAIN[TR_ind,:300,...],TRAIN_y[TR_ind,...], validation_data=(TRAIN[VAL_ind,:300,...],TRAIN_y[VAL_ind,...]), epochs=10, batch_size=64)

TR_X_feat1 = reformulator(TRAIN[:,:300,:])
# TR_X_feat2 = reformulator(TR_X[:,:600,:])
# TR_X_feat3 = reformulator(TR_X[:,:900,:])
#%%
#Create Decision Tree
DcsTree_1 = RandomForestRegressor(max_depth=10, random_state=0,n_estimators=1, bootstrap=False)
DcsTree_2 = RandomForestRegressor(max_depth=10, random_state=0,n_estimators=1, bootstrap=False)
DcsTree_3 = RandomForestRegressor(max_depth=10, random_state=0,n_estimators=1, bootstrap=False)

#%%
DcsTree_1.fit(TR_X_feat1[TR_ind,...],TRAIN_y[TR_ind,...])
DcsTree_2.fit(TR_X_feat2,TR_Y)
DcsTree_3.fit(TR_X_feat3,TR_Y)

#%%
# ALL TREE IS 100% ACCURATE!!
DcsTree_1.score(reformulator(TEST[:,:300,:]),TEST_y)
DcsTree_2.score(reformulator(TEST[:,:600,:]),TEST_y)
DcsTree_3.score(reformulator(TEST[:,:900,:]),TEST_y)
#%%
from sklearn import tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (8,6), dpi=300)
tree.plot_tree(DcsTree_1.estimators_[1],
               filled = True);
# plt.savefig('./Figs/Decisiontree.png')
#%%
for i in tqdm.tqdm(range(19)):
    past = time.time()
    for f in MSLB_TRAIN:
        plt.subplot(2,2,1)
        plt.plot(f[:,i])
        plt.title('MSLB_TRAIN #{}'.format(i))
    for f in MSLB_TEST:
        plt.subplot(2,2,2)
        plt.plot(f[:,i])
        plt.title('MSLB_TEST #{}'.format(i))
    for f in SGTR_TRAIN:
        plt.subplot(2,2,3)
        plt.plot(f[:,i])
        plt.title('SGTR_TRAIN #{}'.format(i))
    for f in SGTR_TEST:
        plt.subplot(2,2,4)
        plt.plot(f[:,i])
        plt.title('SGTR_TEST #{}'.format(i))
    plt.tight_layout()
    print(time.time()-past)
    plt.savefig('./Figs/Ch{}.png'.format(i))
    plt.close()

#%%
MSLB_TRAIN = RAW_TRAIN['MSLB'][:,::10,:].astype(np.float16)
SGTR_TRAIN = RAW_TRAIN['SGTR'][:,::10,:].astype(np.float16)
MSLB_TEST = RAW_TEST['MSLB'][:,::10,:].astype(np.float16)
SGTR_TEST = RAW_TEST['SGTR'][:,::10,:].astype(np.float16)



[RAW_TRAIN['MSLB'],RAW_TRAIN['SGTR']
#%%
TEST_simple=np.where(TEST_feat[:,31]<=0.06,0,1)

#%%
plt.plot(TEST_y)
plt.plot(TEST_simple)

#%%
fig, axes = plt.subplots(nrows=1,ncols=2, figsize=(8,6),dpi=300)
axes[0].plot(TRAIN_feat[:,6])
axes[1].plot(TEST_feat[:,6])
plt.savefig('./Figs/Ch7_value.png')