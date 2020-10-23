#%%
#import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

def reformulator(X,point=300):
    X_reform = []
    for index, value in enumerate(X[:,:point,:]):
        x_mean = np.mean(value,axis=0)
        x_std = np.mean(value,axis=0)
        X_reform.append(np.concatenate([x_mean,x_std]))
    return(np.array(X_reform))
def Point(X,point=300):
    return(X[:,point-1,:])
def Slope(X,point=300):
    return(X[:,point-1,:]-X[:,point-6,:])
def PointSlope(X,point=300):
    p = Point(X,point)
    s = Slope(X,point)
    return(np.concatenate([p,s],axis=1))
def sampler(X,point=300):
    temp = X[:,:point:10,...]
    X_shape = temp.shape    
    return(temp.reshape((X_shape[0],-1)))

#%%
losses = pd.DataFrame()
#%%
for i in tqdm.tqdm([300,600,900]):    
    model = RandomForestClassifier(max_depth=10, random_state=0, n_estimators=1)
    model.fit(Point(TRAIN,i),np.argmax(TRAIN_y,axis=1))
    TR_a = model.score(Point(TRAIN,i),np.argmax(TRAIN_y,axis=1))
    TE_a = model.score(Point(TEST,i),np.argmax(TEST_y,axis=1))
    losses=losses.append({'SEC':i,'Feature':'point','TR_acc':TR_a, 'TE_acc':TE_a},ignore_index=True)
    del model
for i in tqdm.tqdm([300,600,900]):    
    model = RandomForestClassifier(max_depth=10, random_state=0, n_estimators=1)
    model.fit(Slope(TRAIN,i),np.argmax(TRAIN_y,axis=1))
    TR_a = model.score(Slope(TRAIN,i),np.argmax(TRAIN_y,axis=1))
    TE_a = model.score(Slope(TEST,i),np.argmax(TEST_y,axis=1))
    losses=losses.append({'SEC':i,'Feature':'slope','TR_acc':TR_a, 'TE_acc':TE_a},ignore_index=True)
    del model
for i in tqdm.tqdm([300,600,900]):    
    model = RandomForestClassifier(max_depth=10, random_state=0, n_estimators=1)
    model.fit(PointSlope(TRAIN,i),np.argmax(TRAIN_y,axis=1))
    TR_a = model.score(PointSlope(TRAIN,i),np.argmax(TRAIN_y,axis=1))
    TE_a = model.score(PointSlope(TEST,i),np.argmax(TEST_y,axis=1))
    losses=losses.append({'SEC':i,'Feature':'point_slope','TR_acc':TR_a, 'TE_acc':TE_a},ignore_index=True)
    del model
for i in tqdm.tqdm([300,600,900]):    
    model = RandomForestClassifier(max_depth=10, random_state=0, n_estimators=1)
    model.fit(reformulator(TRAIN,i),np.argmax(TRAIN_y,axis=1))
    TR_a = model.score(reformulator(TRAIN,i),np.argmax(TRAIN_y,axis=1))
    TE_a = model.score(reformulator(TEST,i),np.argmax(TEST_y,axis=1))
    losses=losses.append({'SEC':i,'Feature':'mean_var','TR_acc':TR_a, 'TE_acc':TE_a},ignore_index=True)
    del model
for i in tqdm.tqdm([300,600,900]):    
    model = RandomForestClassifier(max_depth=10, random_state=0, n_estimators=1)
    model.fit(sampler(TRAIN,i),np.argmax(TRAIN_y,axis=1))
    TR_a = model.score(sampler(TRAIN,i),np.argmax(TRAIN_y,axis=1))
    TE_a = model.score(sampler(TEST,i),np.argmax(TEST_y,axis=1))
    losses=losses.append({'SEC':i,'Feature':'sampler','TR_acc':TR_a, 'TE_acc':TE_a},ignore_index=True)
    del model
#%%

TR_X_feat1 = reformulator(TRAIN[:,:300,:])
TR_X_feat2 = reformulator(TRAIN[:,:600,:])
TR_X_feat3 = reformulator(TRAIN[:,:900,:])
TR_X_feat1 = sampler(TRAIN[:,:300,:])
TR_X_feat2 = sampler(TRAIN[:,:600,:])
TR_X_feat3 = sampler(TRAIN[:,:900,:])

#%%
#Create Decision Tree
DcsTree_1 = RandomForestClassifier(max_depth=10, random_state=0,n_estimators=10, bootstrap=True)
DcsTree_2 = RandomForestClassifier(max_depth=10, random_state=0,n_estimators=10, bootstrap=True)
DcsTree_3 = RandomForestClassifier(max_depth=10, random_state=0,n_estimators=10, bootstrap=True)

#%%
DcsTree_1.fit(TRAIN[:,300,...]-TRAIN[:,295,...],np.argmax(TRAIN_y,axis=1))
DcsTree_1.fit(TRAIN[:,600,...],np.argmax(TRAIN_y,axis=1))
DcsTree_1.fit(TRAIN[:,899,...],np.argmax(TRAIN_y,axis=1))

DcsTree_2.fit(TR_X_feat2,np.argmax(TRAIN_y,axis=1))
DcsTree_3.fit(TR_X_feat3,np.argmax(TRAIN_y,axis=1))

#%%
# ALL TREE IS 100% ACCURATE!!
DcsTree_1.score(TEST[:,300,:]-TEST[:,295,:],np.argmax(TEST_y,axis=1))
DcsTree_1.score(TEST[:,600,:],np.argmax(TEST_y,axis=1))
DcsTree_1.score(TEST[:,899,:],np.argmax(TEST_y,axis=1))

DcsTree_2.score(sampler(TEST[:,:600,:]),np.argmax(TEST_y,axis=1))
DcsTree_3.score(sampler(TEST[:,:900,:]),np.argmax(TEST_y,axis=1))
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