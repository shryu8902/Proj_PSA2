#%%

import lightgbm as lgb
def reformulator(X):
    X_reform = []
    for index, value in enumerate(X):
        x_mean = np.mean(value,axis=0)
        x_std = np.mean(value,axis=0)
        X_reform.append(np.concatenate([x_mean,x_std]))
    return(np.array(X_reform))    
#%%
TRAIN_feat = reformulator(TRAIN2)
TEST_feat = reformulator(TEST2)

train_ds = lgb.Dataset(TRAIN_feat, label = TRAIN_y) 
test_ds = lgb.Dataset(TEST_feat, label = TEST_y) 
#%%
params = {'learning_rate': 0.01, 
          'max_depth': 10, 
          'boosting': 'gbdt', 
          'objective': 'binary', 
          'metric': 'binary_logloss',  
          'seed':0}
#%%
lgb_model = lgb.train(params, train_ds, 100)
TRAIN_feat_y_pred = lgb_model.predict_proba(TRAIN_feat)
y_pred=lgb_model.predict(TEST_feat)
#%%
m = tf.keras.metrics.BinaryAccuracy()
m.update_state(TEST_y,TEST_simple)
m.result().numpy()

#%%
fpr, tpr, threshold = roc_curve(TEST_y,y_pred)
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

#%%
lgb.plot_tree(lgb_model,tree_index=30)
lgb.plot_importance(lgb_model)
# %%
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=10, random_state=0,n_estimators=1, bootstrap=False)
regr.fit(TRAIN_feat, TRAIN_y)
#%%
regr.score(TRAIN_feat,TRAIN_y)
regr.score(TEST_feat, TEST_y)

regr.feature_importances_
#%%
from sklearn import tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (8,6), dpi=300)
tree.plot_tree(regr.estimators_[0],
               filled = True);
# plt.savefig('./Figs/Decisiontree.png')


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