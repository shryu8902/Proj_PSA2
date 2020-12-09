import 

#%%

def data_transformer(normed_input,input_data,seqlen):
    BRK = np.repeat(normed_input[:,0,np.newaxis,np.newaxis],seqlen,axis=1)    
    TYPE = np.repeat(normed_input[:,15,np.newaxis,np.newaxis],seqlen,axis=1)    
    REDATA = np.concatenate([BRK,TYPE],axis=2)
    for i in range(1,15,2):
        time = np.ceil(input_data[:,i]*60/10).astype(np.int32)
        action = normed_input[:,i+1]
        act_seq = np.zeros((len(time),seqlen,1))
        for j in range(len(time)):
            act_seq[j,...][time[j]]=action[j]
        REDATA=np.concatenate([REDATA,act_seq],axis=2)
    return REDATA
#%%
from sklearn.preprocessing import KBinsDiscretizer
discretizer = KBinsDiscretizer(n_bins=12, encode='onehot-dense', strategy='quantile')
train_output_.shape
train_out_prob = discretizer.fit_transform(train_output_.reshape(-1,1)).reshape(-1,360,128)
train_out_hat_ = discretizer.inverse_transform(train_out_prob.reshape(-1,128)).reshape(-1,360)
x=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
x_t = discretizer.fit_transform(x.reshape(-1,1)).reshape(-1,3,12)
class prob_maker():
    def __init__(self, base_data):
        super(prob_maker,self).__init__()
        self.base_data
#%%
def simpleRNN(num_cell=128, dr_rates = 0.3):
    inputs = Input(shape =(360,9) ,name='input')
    layer_1 = Bidirectional(LSTM(num_cell, return_sequences=True,dropout=dr_rates))(inputs)   
    layer_2 = Bidirectional(LSTM(num_cell, return_sequences=True,dropout=dr_rates))(layer_1)
    layer_3 = TimeDistributed(Dense(32,activation='selu'))(layer_2)
    outputs_ = TimeDistributed(Dense(1))(layer_3)
    outputs = Flatten()(outputs_)
    model= Model(inputs, outputs)
    return model
#%%
t_train = data_transformer(train_input,train_input_,360)
t_val = data_transformer(val_input,val_input_,360)
t_test = data_transformer(test_input,test_input_,360)
#%%
vername = 'verX'
base_model = simpleRNN(128,0.2)
base_model.compile(optimizer='adam',
                # loss='mean_squared_error',
                loss = 'mean_squared_error',
                metrics=['mean_absolute_error','mean_squared_error']) 

ensure_dir('./Model/FR_RNN/{}'.format(vername))
path = './Model/FR_RNN/{}'.format(vername)+'/e{epoch:04d}.ckpt'
checkpoint = ModelCheckpoint(path, monitor = 'val_loss',verbose = 1,
            save_best_only = True,
            mode = 'auto',
            save_weights_only = True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
hist=base_model.fit(t_train,train_output,
               validation_data=(t_val,val_output),
               callbacks=[checkpoint,reduce_lr,early_stopping],epochs=100,batch_size=512)
with open('./Model/FR_RNN/{}/hist.pkl'.format(vername), 'wb') as f:
        pickle.dump(hist.history,f)
#%%
test_output_hat = base_model.predict(t_test)
test_output_hat_ =OUT_SCALER[SCALE].inverse_transform(test_output_hat.reshape(-1,1)).reshape(-1,360)
val_output_hat = base_model.predict(t_val)
val_output_hat_ = OUT_SCALER[SCALE].inverse_transform(val_output_hat.reshape(-1,1)).reshape(-1,360)

def MAPE(y_true,y_pred):
    ape = abs(y_true-y_pred)/y_true*100
    mape = np.mean(ape,axis=-1)
    return mape
def MSE(y_true,y_pred):
    se = (y_true-y_pred)**2
    mse = np.mean(se,axis=-1)
    return mse

np.mean(MAPE(test_output_, test_output_hat_))
np.mean(MSE(test_output_, test_output_hat_))
np.mean(MAPE(val_output_, val_output_hat_))
np.mean(MSE(val_output_, val_output_hat_))

#%%
# untrain : SGTR: 273(0),274(1),141825(17998),141826(17999)
index = 273
df_test = pd.DataFrame(test_)
iloc = df_test.loc[df_test[0]==index].index[0]
if df_test.iloc[iloc][16] == 0:
    ttype = 'SGTR'
else :
    ttype = 'MSLB'
plt.plot(test_output_[iloc,:],label='True')
plt.plot(test_output_hat_[iloc,:],label='Fast running')
plt.legend()
plt.grid()
plt.title('Index : {}, Type: {}'.format(str(index),ttype))
# plt.savefig('./test_diff_{}.png'.format(str(index)))
#%%
# val : SGTR : 26848(0),18902(1), MSLB 113865(7), 139096(8)
index = 139096
df_test = pd.DataFrame(val_)
iloc = df_test.loc[df_test[0]==index].index[0]
if df_test.iloc[iloc][16] == 0:
    ttype = 'SGTR'
else :
    ttype = 'MSLB'
plt.plot(val_output_[iloc,:],label='True')
plt.plot(val_output_hat_[iloc,:],label='Fast running')
plt.legend()
plt.grid()
plt.title('Index : {}, Type: {}'.format(str(index),ttype))
