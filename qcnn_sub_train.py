
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import glob
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
import joblib

from keras.layers import LeakyReLU

#initail_quantum_bit_function

def initail_Qcw(master, sub, filter_i, filter_j, filter_depth, filter_num, bit_length):
    Qcw = np.full( (master, sub, filter_i, filter_j, filter_depth, filter_num, bit_length), 1/np.sqrt(2),dtype=np.float32)
    return Qcw

def initail_Qcb(master, sub, filter_num, bit_length):
    Qcb = np.full( (master, sub, filter_num, bit_length ), 1/np.sqrt(2),dtype=np.float32)
    return Qcb

def initail_Qfw(master, sub, i, j, bit_length):
    Qfw = np.full( (master, sub, i, j, bit_length), 1/np.sqrt(2),dtype=np.float32)
    return Qfw

def initail_Qfb(master, sub, j, bit_length):
    Qfb = np.full( (master, sub, j, bit_length), 1/np.sqrt(2),dtype=np.float32)
    return Qfb

def initail_Qcov(master, sub, out_i, out_j, filter_num):
    Qcov = np.full( (master, sub, out_i, out_j, filter_num), 1/np.sqrt(2),dtype=np.float32)
    return Qcov


# In[70]:


#quantum_bit_to_binary_bit_function

def Qcw_to_bcw(Qcw,bcw):
    filter_i, filter_j, filter_depth, filter_num, bit_length = Qcw.shape
    for num in range (filter_num):
        for i in range (filter_i):
            for j in range (filter_j):
                for d in range (filter_depth):
                    for bit in range (bit_length):
                        bcw[i,j,d,num,bit]=np.where(np.square(Qcw[i,j,d,num,bit])<=np.random.rand() , 1, 0)
    return bcw

def Qcb_to_bcb(Qcb,bcb):
    filter_num, bit_length = Qcb.shape
    for num in range (filter_num):
        for bit in range (bit_length):
            bcb[num,bit]=np.where(np.square(Qcb[num,bit])<=np.random.rand() , 1, 0)
    return bcb

def Qfw_to_bfw(Qfw,bfw):
    fw_i, fw_j, bit_length = Qfw.shape
    for i in range (fw_i):
        for j in range (fw_j):
            for bit in range (bit_length):
                bfw[i,j,bit]=np.where(np.square(Qfw[i,j,bit])<=np.random.rand() , 1, 0)
    return bfw

def Qfb_to_bfb(Qfb,bfb):
    fb_j, bit_length = Qfb.shape
    for j in range (fb_j):
        for bit in range (bit_length):
            bfb[j,bit]=np.where(np.square(Qfb[j,bit])<=np.random.rand() , 1, 0)
    return bfb

def Qcov_to_bcov(Qcov,bcov):
    out_i, out_j, filter_num = Qcov.shape
    for num in range (filter_num):
        for i in range (out_i):
            for j in range (out_j):
                bcov[i,j,num]=np.where(np.square(Qcov[i,j,num])<=np.random.rand() , 1, 0)
    return np.int32(bcov)


# In[71]:


#binary_bit_to_real_value_function

def bcw_to_real_cw(bcw,tau_cw,tau_value,sigma,real_cw,_max,_min):
    filter_i, filter_j, filter_depth, filter_num, bit_length = bcw.shape
    _range = (_max-_min)/np.power(2,bit_length)
    _range_helf = _range/2
    for num in range (filter_num):
        for i in range (filter_i):
            for j in range (filter_j):
                for d in range (filter_depth):
                    dec = np.int32(np.polyval(bcw[i,j,d,num,:],2))
                    mu = _min+(_range*(dec+1))-_range_helf
                    real_cw[i,j,d,num]=np.random.normal( mu
                                                         ,sigma*np.power(tau_value,tau_cw[i,j,d,num,dec]))
        
                    tau_cw[i,j,d,num,dec]=tau_cw[i,j,d,num,dec]+1
    return real_cw

def bcb_to_real_cb(bcb,tau_cb,tau_value,sigma,real_cb,_max,_min):
    filter_num, bit_length = bcb.shape
    _range = (_max-_min)/np.power(2,bit_length)
    _range_helf = _range/2
    for num in range (filter_num):
        dec = np.int32(np.polyval(bcb[num,:],2))
        mu = _min+(_range*(dec+1))-_range_helf
        real_cb[num,0]=np.random.normal( mu
                                         ,sigma*np.power(tau_value,tau_cb[num,dec]))
       
        tau_cb[num,dec]=tau_cb[num,dec]+1
    return real_cb

def bfw_to_real_fw(bfw,tau_fw,tau_value,sigma,real_fw,_max,_min):
    fw_i, fw_j, bit_length = bfw.shape
    _range = (_max-_min)/np.power(2,bit_length)
    _range_helf = _range/2
    for i in range (fw_i):
        for j in range (fw_j):
            dec = np.int32(np.polyval(bfw[i,j,:],2))
            mu = _min+(_range*(dec+1))-_range_helf
            real_fw[i,j]=np.random.normal( mu
                                           ,sigma*np.power(tau_value,tau_fw[i, j, dec]))
            
            tau_fw[i, j, dec]=tau_fw[i, j, dec]+1
    return real_fw

def bfb_to_real_fb(bfb,tau_fb,tau_value,sigma,real_fb,_max,_min):
    fb_j, bit_length = bfb.shape
    _range = (_max-_min)/np.power(2,bit_length)
    _range_helf = _range/2
    for j in range (fb_j):
        dec = np.int32(np.polyval(bfb[j,:],2))
        mu = _min+(_range*(dec+1))-_range_helf                                  
        real_fb[j]=np.random.normal( mu
                                    ,sigma*np.power(tau_value,tau_fb[j,dec]))
       
        tau_fb[j,dec]=tau_fb[j,dec]+1
    return real_fb


# In[72]:


# qconv_function

def qconv(input_feature,_filter_np,decide_conv_array,biases_np,conv_output,model_cov1):
    filter_i ,filter_j, feature_depth, filter_num= _filter_np.shape
    
    batch = input_feature.shape[0]
      
    out_i = (input_feature.shape[2]-filter_i+1)
    out_j = (input_feature.shape[2]-filter_j+1)
    
    #ref
    ref_np = np.zeros([batch,out_i,out_j,1])
    

    for filter_n in range(filter_num):
        for i in range( out_i ) :
            for j in range( out_j ) :
                if decide_conv_array[i,j,filter_n]==1:
                    
                    model_cov1.get_layer('cov').set_weights([np.reshape(_filter_np[:,:,:,filter_n],[filter_i,filter_j,feature_depth,1]),
                                                             biases_np[filter_n,:]])
                    
                    _conv1_num = model_cov1.predict(input_feature[:,i:i+filter_i, j:j+filter_j, :])
                    ref_np[:,i,j,:] = np.reshape(_conv1_num,[batch,1])
                else:
                    ref_np[:,i,j,:]=0
        conv_output[:,:,:,filter_n] = ref_np[:,:,:,0]
    return conv_output


# In[73]:


# quantum_gate_funciton

def Q_update(Q, b, Sub_best_b, rotate_angle):
    
    address1 = np.where((b==0) & (Sub_best_b==1))
    address2 = np.where((b==1) & (Sub_best_b==0))
    
    Q[address1] = np.dot(Q[address1] , np.cos(-rotate_angle)) - np.dot( np.sqrt((1-np.square(Q[address1]))) , np.sin(-rotate_angle) )
    Q[address2] = np.dot(Q[address2] , np.cos(rotate_angle)) - np.dot( np.sqrt((1-np.square(Q[address2]))) , np.sin(rotate_angle) )
    
    return Q


# In[74]:


#mean_squared_error
def mse(y_test,y_preditc):
    error = y_preditc-y_test
    #error[np.where(error > 0.15)] = error[np.where(error > 0.15)]**3
    #error[np.where((error < 0.15) & (error > 0.07))] = error[np.where((error < 0.15) & (error > 0.07))]**2
    #error[np.where(error < 0.07)] = error[np.where(error < 0.07)]
    mse=np.sum(error**2)/len(y_test)
    #if y_preditc-y_test>0.2:
        #mse=np.sum((y_preditc-y_test)**4)/len(y_test)
    #else:
        #mse=np.sum((y_preditc-y_test)**2)/len(y_test)
    return mse


# In[75]:


def fix(y_test,y_preditc):
    error = y_preditc-y_test
    #預測值小於實際值
    error[np.where((error > -0.1) & (error < 0))] = error[np.where((error > -0.1) & (error < 0))]*-1.1
    error[np.where((error > -0.2) & (error < -0.1))] = error[np.where((error > -0.2) & (error < -0.1))]*-1.2
    error[np.where((error > -0.3) & (error < -0.2))] = error[np.where((error > -0.3) & (error < -0.2))]*-1.35
    error[np.where((error > -0.4) & (error < -0.3))] = error[np.where((error > -0.4) & (error < -0.3))]*-1.55
    error[np.where((error > -0.5) & (error < -0.4))] = error[np.where((error > -0.5) & (error < -0.4))]*-2
    error[np.where(error < -0.5)] = error[np.where(error < -0.5)]*-2.3
    
    #預測值超過實際值
    error[np.where((error < 0.1) & (error > 0))] = error[np.where((error < 0.1) & (error > 0))]*0.9 
    error[np.where((error < 0.2) & (error > 0.1))] = error[np.where((error < 0.2) & (error > 0.1))]*0.85
    error[np.where((error < 0.3) & (error > 0.2))] = error[np.where((error < 0.3) & (error > 0.2))]*0.8
    error[np.where((error < 0.4) & (error > 0.3))] = error[np.where((error < 0.4) & (error > 0.3))]*0.75
    error[np.where((error < 0.5) & (error > 0.4))] = error[np.where((error < 0.5) & (error > 0.4))]*0.7
    error[np.where(error > 0.5)] = error[np.where(error > 0.5)]*0.6
    
    return y_preditc+error


# In[76]:


#data_input

#load_data
path1=r'./data/' #訓練 讀檔的路徑

file1=glob.glob(os.path.join(path1, "2017*.csv"))
df_train=pd.DataFrame([])

for f in file1:
    out = pd.read_csv(f,sep=',')
    index_names_1=out.loc[:,'time']      #把 'time'那行的數據 讀出 => index_names_1
    out.index=index_names_1             #把index的label 以'time'這行的時間命名
    out=out.loc['09:00:00':'17:00:00',:] #用index的label 去保留 要的時間
    df_train=pd.concat([df_train,out],axis=0,ignore_index=True)

x = df_train[['tep','li']]  # # # #
y = df_train[['kw']]         # 
y = y/2   #三台inverter 總共6台MPPT

x_2 = x.values[:,0:2]  
x_Amplification = np.column_stack([x_2,x_2,x_2,x_2,x_2,x_2,x_2,x_2,x_2,x_2,x_2,x_2,x_2[:,0:1]])  # 5*5

# 0.75 #
train_x_disorder, test_x_disorder, train_y_disorder, test_y_disorder = train_test_split(x_Amplification, y,train_size=0.75, random_state=1)

#x_Amplification_log = np.log(x_Amplification)         #取log
#train_x_disorder_log = np.log(train_x_disorder)       #取log
#test_x_disorder_log = np.log(test_x_disorder)         #取log

#標準化
x_min_max_scaler = MinMaxScaler().fit(x_Amplification) #以 全部資料 為最大最小標準化 的標準
y_min_max_scaler = MinMaxScaler().fit(train_y_disorder)
joblib.dump(x_min_max_scaler, 'QCNNscaler201804260428x.pkl') #將x_min_max_scaler縮放標準儲存
joblib.dump(y_min_max_scaler, 'QCNNscaler201804260428y.pkl') #將y_min_max_scaler縮放標準儲存

x_train=x_min_max_scaler.transform(train_x_disorder) #以行為單位 做最大最小標準化
y_train=y_min_max_scaler.transform(train_y_disorder)

x_test=x_min_max_scaler.transform(test_x_disorder)#以x_train_data 為最大最小標準化 的標準 去對x_test_data 進行最大最小標準化
y_test=y_min_max_scaler.transform(test_y_disorder)



# In[77]:


x_train = np.float32(np.reshape(x_train, [-1, 5, 5, 1]))
y_train = np.float32(y_train)

x_test = np.float32(np.reshape(x_test, [-1, 5, 5, 1]))
y_test = np.float32(y_test)


print(x_train.shape)
print(x_train.dtype)

print(x_test.shape)
print(x_test.dtype)

print(y_train.shape)
print(y_train.dtype)

print(y_test.shape)
print(y_test.dtype)


# In[78]:


#parameters
epoach = 200
master = 1
sub = 10
tau_value=0.8
sigma = 0.0125
rotate_angle = 0.2*3.14
bit_length = 4
_max = 1
_min = -1.2
mutation = 15

fitness = np.zeros([master,sub],dtype=np.float32)
Sub_best_fitness = np.float32(10000)
loss_fitness = np.full( (epoach), 10000,dtype=np.float32)
#initail_conv1_parameters
Qcw1 = initail_Qcw(master,sub,2,2,1,16,bit_length)
Qcb1 = initail_Qcb(master,sub,16,bit_length)

bcw1 = np.zeros([master,sub,2,2,1,16,bit_length],dtype=np.float32)
bcb1 = np.zeros([master,sub,16,bit_length],dtype=np.float32)

Qcov1 = initail_Qcov(master,sub,4,4,16)
bcov1=np.zeros([master,sub,4,4,16],dtype=np.float32)

tau_cw1 = np.ones([master,sub,2,2,1,16,np.power(2,bit_length)],dtype=np.float32)
cw1 = np.ones([master,sub,2,2,1,16],dtype=np.float32)

tau_cb1 = np.ones([master,sub,16,np.power(2,bit_length)],dtype=np.float32)
cb1 = np.ones([master,sub,16,1],dtype=np.float32)

Sub_best_bcw1 = np.zeros([2,2,1,16,bit_length],dtype=np.float32)
Sub_best_bcb1 = np.zeros([16,bit_length],dtype=np.float32)
Sub_best_bcov1 = np.zeros([4,4,16],dtype=np.float32)
Sub_best_cw1 = np.zeros([2,2,1,16],dtype=np.float32)
Sub_best_cb1 = np.zeros([16,1],dtype=np.float32)

#initail_conv2_parameters
Qcw2 = initail_Qcw(master,sub,2,2,16,24,bit_length)
Qcb2 = initail_Qcb(master,sub,24,bit_length)

bcw2 = np.zeros([master,sub,2,2,16,24,bit_length],dtype=np.float32)

bcb2 = np.zeros([master,sub,24,bit_length],dtype=np.float32)

Qcov2 = initail_Qcov(master,sub,1,1,24)
bcov2=np.zeros([master,sub,1,1,24],dtype=np.float32)

tau_cw2 = np.ones([master,sub,2,2,16,24,np.power(2,bit_length)],dtype=np.float32)
cw2 = np.ones([master,sub,2,2,16,24],dtype=np.float32)

tau_cb2 = np.ones([master,sub,24,np.power(2,bit_length)],dtype=np.float32)
cb2 = np.ones([master,sub,24,1],dtype=np.float32)

Sub_best_bcw2 = np.zeros([2,2,16,24,bit_length],dtype=np.float32)
Sub_best_bcb2 = np.zeros([24,bit_length],dtype=np.float32)
Sub_best_bcov2 = np.zeros([1,1,24],dtype=np.float32)
Sub_best_cw2 = np.ones([2,2,16,24],dtype=np.float32)
Sub_best_cb2 = np.ones([24,1],dtype=np.float32)

#initail_full_connect1_parameters
Qfw1 = initail_Qfw(master,sub,1*1*24,128,bit_length)
Qfb1 = initail_Qfb(master,sub,128,bit_length)

bfw1 = np.ones([master,sub,1*1*24,128,bit_length],dtype=np.float32)
bfb1 = np.ones([master,sub,128,bit_length],dtype=np.float32)

tau_fw1 = np.ones([master,sub,1*1*24,128,np.power(2,bit_length)],dtype=np.float32)
tau_fb1 = np.ones([master,sub,128,np.power(2,bit_length)],dtype=np.float32)

fw1 = np.ones([master,sub,1*1*24,128],dtype=np.float32)
fb1 = np.ones([master,sub,128],dtype=np.float32)

Sub_best_bfw1 = np.ones([1*1*24,128,bit_length],dtype=np.float32)
Sub_best_bfb1 = np.ones([128,bit_length],dtype=np.float32)
Sub_best_fw1 = np.ones([1*1*24,128],dtype=np.float32)
Sub_best_fb1 = np.ones([128],dtype=np.float32)

#initail_full_connect2_parameters
Qfw2 = initail_Qfw(master,sub,128,1,bit_length)
Qfb2 = initail_Qfb(master,sub,1,bit_length)

bfw2 = np.ones([master,sub,128,1,bit_length],dtype=np.float32)
bfb2 = np.ones([master,sub,1,bit_length],dtype=np.float32)

tau_fw2 = np.ones([master,sub,128,1,np.power(2,bit_length)],dtype=np.float32)
tau_fb2 = np.ones([master,sub,1,np.power(2,bit_length)],dtype=np.float32)

fw2 = np.ones([master,sub,128,1],dtype=np.float32)
fb2 = np.ones([master,sub,1],dtype=np.float32)

Sub_best_bfw2 = np.ones([128,1,bit_length],dtype=np.float32)
Sub_best_bfb2 = np.ones([1,bit_length],dtype=np.float32)
Sub_best_fw2 = np.ones([128,1],dtype=np.float32)
Sub_best_fb2 = np.ones([1],dtype=np.float32)


# In[79]:


#卷積層1 tanh sigmoid
model_cov1 = Sequential()

model_cov1.add(Convolution2D(
                  filters = 1,
                  kernel_size=(2, 2),
                  padding = 'same', # padding method
                  strides=(2, 2),
                  name="cov"
                ))
model_cov1.build(input_shape=[None, 2, 2, 1])

#池化層
model_pool = Sequential()

model_pool.add(Activation('tanh'))

model_pool.add(MaxPooling2D(
           pool_size = (2,2), #每2x2產生一個特徵
           strides = (2,2),  #每次滑動2x2大小去尋找特徵
           padding = 'same' # padding method  
))


#卷積層2
model_cov2 = Sequential()

model_cov2.add(Convolution2D(
                  filters = 1,
                  kernel_size=(2, 2),
                  padding = 'same', # padding method
                  strides=(2, 2),
                  name="cov"
                ))
model_cov2.build(input_shape=[None, 2, 2, 16])

#全連接層1
model_dense = Sequential()

model_dense.add(Activation('tanh'))

model_dense.add(Flatten())
model_dense.add(Dense(128,name="Dense_1"))

model_dense.add(LeakyReLU(alpha=0.01))
#model_dense.add(Activation('tanh'))

model_dense.add(Dense(1,name="output_layer"))
model_dense.add(Activation('sigmoid'))
model_dense.build(input_shape=[None, 1,1,24])


# In[80]:



gtime1 = time.time()
for k in range (epoach):
    for i in range (master):
        for j in range (sub):

            
            #conv_output
            conv1_output = np.zeros(shape=[20898, 4, 4, 16], dtype=np.float32)
            conv2_output = np.zeros(shape=[20898, 1, 1, 24], dtype=np.float32)

            #conv1_parameters_to_real_number
            bcw1[i,j] = Qcw_to_bcw(Qcw1[i,j],bcw1[i,j])
            bcb1[i,j] = Qcb_to_bcb(Qcb1[i,j],bcb1[i,j])
            bcov1[i,j] = Qcov_to_bcov(Qcov1[i,j],bcov1[i,j])
            cw1[i,j] = bcw_to_real_cw(bcw1[i,j],tau_cw1[i,j],tau_value,sigma,cw1[i,j],_max,_min)
            cb1[i,j] = bcb_to_real_cb(bcb1[i,j],tau_cb1[i,j],tau_value,sigma,cb1[i,j],_max,_min)

            #conv2_parameters_to_real_number
            bcw2[i,j] = Qcw_to_bcw(Qcw2[i,j],bcw2[i,j])
            bcb2[i,j] = Qcb_to_bcb(Qcb2[i,j],bcb2[i,j])
            bcov2[i,j] = Qcov_to_bcov(Qcov2[i,j],bcov2[i,j])
            cw2[i,j] = bcw_to_real_cw(bcw2[i,j],tau_cw2[i,j],tau_value,sigma,cw2[i,j],_max,_min)
            cb2[i,j] = bcb_to_real_cb(bcb2[i,j],tau_cb2[i,j],tau_value,sigma,cb2[i,j],_max,_min)


            #full_connect1_parameters_to_real
            bfw1[i,j] = Qfw_to_bfw(Qfw1[i,j],bfw1[i,j])
            bfb1[i,j] = Qfb_to_bfb(Qfb1[i,j],bfb1[i,j])

            fw1[i,j] = bfw_to_real_fw(bfw1[i,j],tau_fw1[i,j],tau_value,sigma,fw1[i,j],_max,_min)
            fb1[i,j] = bfb_to_real_fb(bfb1[i,j],tau_fb1[i,j],tau_value,sigma,fb1[i,j],_max,_min)
            
            #full_connect2_parameters_to_real
            bfw2[i,j] = Qfw_to_bfw(Qfw2[i,j],bfw2[i,j])
            bfb2[i,j] = Qfb_to_bfb(Qfb2[i,j],bfb2[i,j])

            fw2[i,j] = bfw_to_real_fw(bfw2[i,j],tau_fw2[i,j],tau_value,sigma,fw2[i,j],_max,_min)
            fb2[i,j] = bfb_to_real_fb(bfb2[i,j],tau_fb2[i,j],tau_value,sigma,fb2[i,j],_max,_min)
            
            
            
            conv1_output=qconv(x_train,cw1[i,j],bcov1[i,j],cb1[i,j],conv1_output,model_cov1)
            pooling = model_pool.predict(conv1_output)
            #print(pooling.shape)
            conv2_output=qconv(pooling,cw2[i,j],bcov2[i,j],cb2[i,j],conv2_output,model_cov2)
            
            #print(conv1_output)
            #print(conv2_output.shape)
            
            model_dense.get_layer('Dense_1').set_weights([fw1[i,j],fb1[i,j]])
            model_dense.get_layer('output_layer').set_weights([fw2[i,j],fb2[i,j]])
            preds = model_dense.predict(conv2_output)
            
            #preds = fix(y_train,preds)
            #print(preds.shape)
            #preds[np.where(preds > 1.0)] = np.float32(1.0)
            #preds[np.where(preds < 0.0)] = np.float32(0.0)
            
            #fitness[i,j] = mean_squared_error(y_train,preds)
            #print(y_train)
            #print(preds)
            fitness[i,j] = mse(y_train,preds)
            
            if fitness[i,j]<Sub_best_fitness:
                
                Qcw1[i,j] = Q_update(Qcw1[i,j], bcw1[i,j], Sub_best_bcw1 , rotate_angle)
                Qcb1[i,j] = Q_update(Qcb1[i,j], bcb1[i,j], Sub_best_bcb1 , rotate_angle)
                Qcov1[i,j] = Q_update(Qcov1[i,j], bcov1[i,j], Sub_best_bcov1 , rotate_angle)

                Qcw2[i,j] = Q_update(Qcw2[i,j], bcw2[i,j], Sub_best_bcw2 , rotate_angle)
                Qcb2[i,j] = Q_update(Qcb2[i,j], bcb2[i,j], Sub_best_bcb2 , rotate_angle)
                Qcov2[i,j] = Q_update(Qcov2[i,j], bcov2[i,j], Sub_best_bcov2 , rotate_angle)

                Qfw1[i,j] = Q_update(Qfw1[i,j], bfw1[i,j], Sub_best_bfw1 , rotate_angle)
                Qfb1[i,j] = Q_update(Qfb1[i,j], bfb1[i,j], Sub_best_bfb1 , rotate_angle)

                Qfw2[i,j] = Q_update(Qfw2[i,j], bfw2[i,j], Sub_best_bfw2 , rotate_angle)
                Qfb2[i,j] = Q_update(Qfb2[i,j], bfb2[i,j], Sub_best_bfb2 , rotate_angle)
                
                Sub_best_fitness = fitness[i,j]
                
          
                Sub_best_fitness = fitness[i,j]
                Sub_best_Qcw1 = Qcw1[i,j]
                Sub_best_Qcb1 = Qcb1[i,j]
                Sub_best_Qcov1 = Qcov1[i,j]
                Sub_best_bcw1 = bcw1[i,j]
                Sub_best_bcb1 = bcb1[i,j]
                Sub_best_bcov1 = bcov1[i,j]
                Sub_best_cw1 = cw1[i,j]
                Sub_best_cb1 = cb1[i,j]

                Sub_best_Qcw2 = Qcw2[i,j]
                Sub_best_Qcb2 = Qcb2[i,j]
                Sub_best_Qcov2 = Qcov2[i,j]
                Sub_best_bcw2 = bcw2[i,j]
                Sub_best_bcb2 = bcb2[i,j]
                Sub_best_bcov2 = bcov2[i,j]
                Sub_best_cw2 = cw2[i,j]
                Sub_best_cb2 = cb2[i,j]

                Sub_best_Qfw1 = Qfw1[i,j]
                Sub_best_Qfb1 = Qfb1[i,j]
                Sub_best_bfw1 = bfw1[i,j]
                Sub_best_bfb1 = bfb1[i,j]
                Sub_best_fw1 = fw1[i,j]
                Sub_best_fb1 = fb1[i,j]

                Sub_best_Qfw2 = Qfw2[i,j]
                Sub_best_Qfb2 = Qfb2[i,j]
                Sub_best_bfw2 = bfw2[i,j]
                Sub_best_bfb2 = bfb2[i,j]
                Sub_best_fw2 = fw2[i,j]
                Sub_best_fb2 = fb2[i,j]
                
                
                # Prediction    
                conv1_output_pre = np.zeros(shape=[6966, 4, 4, 16], dtype=np.float32)
                conv2_output_pre = np.zeros(shape=[6966, 1, 1, 24], dtype=np.float32)   

                conv1_output_pre = qconv(x_test,Sub_best_cw1,Sub_best_bcov1,Sub_best_cb1,conv1_output_pre,model_cov1)
                pooling_pre = model_pool.predict(conv1_output_pre)

                conv2_output_pre = qconv(pooling_pre,Sub_best_cw2,Sub_best_bcov2,Sub_best_cb2,conv2_output_pre,model_cov2)

                model_dense.get_layer('Dense_1').set_weights([Sub_best_fw1,Sub_best_fb1])
                model_dense.get_layer('output_layer').set_weights([Sub_best_fw2,Sub_best_fb2])
                prediction_value = model_dense.predict(conv2_output_pre)
                
                #prediction_value = fix(y_test,prediction_value)
                #prediction_value[np.where(prediction_value > 1.0)] = np.float32(1.0)
                #prediction_value[np.where(prediction_value < 0.0)] = np.float32(0.0)
                
                preds_val=y_min_max_scaler.inverse_transform(prediction_value)
                np.save('mid_preds_val.npy',preds_val )
                
                Pred_fitness = mean_squared_error(test_y_disorder.values,preds_val)

                ###

                # plt.plot(preds_val/1000, color = 'g', label = 'Qcnn', linestyle = 'solid')
                # plt.xlabel('data')
                # plt.ylabel('kW')
                # plt.ylim((0,2))
                # plt.legend(loc = 'upper left')
                # plt.savefig('QCNN_test_curve.png')
                # plt.show()

                ###
                print('Pred_test_loss_fitness',Pred_fitness)
                                         
                np.save('mid_Sub_best_bcov1.npy',Sub_best_bcov1 )
                np.save('mid_Sub_best_cw1.npy',Sub_best_cw1 )
                np.save('mid_Sub_best_cb1.npy',Sub_best_cb1 )

                np.save('mid_Sub_best_bcov2.npy',Sub_best_bcov2 )
                np.save('mid_Sub_best_cw2.npy',Sub_best_cw2 )
                np.save('mid_Sub_best_cb2.npy',Sub_best_cb2 )

                np.save('mid_Sub_best_fw1.npy',Sub_best_fw1 )
                np.save('mid_Sub_best_fb1.npy',Sub_best_fb1 )

                np.save('mid_Sub_best_fw2.npy',Sub_best_fw2 )
                np.save('mid_Sub_best_fb2.npy',Sub_best_fb2 )
         
    print('loss_train_fitness',Sub_best_fitness)
    if k%mutation==0:
            for j in range (sub):
                Qcw1[i,j] = Sub_best_Qcw1
                Qcb1[i,j] = Sub_best_Qcb1
                Qcov1[i,j] = Sub_best_Qcov1
                
                Qcw2[i,j] = Sub_best_Qcw2
                Qcb2[i,j] = Sub_best_Qcb2
                Qcov2[i,j] = Sub_best_Qcov2
                
                Qfw1[i,j] = Sub_best_Qfw1
                Qfb1[i,j] = Sub_best_Qfb1
                
                Qfw2[i,j] = Sub_best_Qfw2
                Qfb2[i,j] = Sub_best_Qfb2
    loss_fitness[k] = Sub_best_fitness
    
gtime2 = time.time()
gtotal = gtime2 - gtime1
print('計算時間：', gtotal)    

#================Picture=================

plt.plot(test_y_disorder.values/1000, color = 'r', label = 'real', linestyle = 'solid')
plt.xlabel('data')
plt.ylabel('kW')
plt.ylim((0,2))
plt.legend(loc = 'upper left')
plt.savefig('mid__Real_1_curve.png')
plt.show()

plt.plot(test_y_disorder.values/1000, color = 'r', label = 'real', linestyle = 'solid')
plt.plot(preds_val/1000, color = 'g', label = 'Qcnn', linestyle = 'solid')
plt.xlabel('data')
plt.ylabel('kW')
plt.ylim((0,2))
plt.legend(loc = 'upper left')
plt.savefig('mid__All_1_curve.png')
plt.show()

plt.plot(loss_fitness, color = 'r', label = '60 day', linestyle = 'solid')
plt.xlabel('epoch')
plt.ylabel('loss')
#ticks_x = np.arange(0, 10 ,1)
#plt.xticks(ticks_x)
plt.ylim((0,0.5))
plt.xlim((0,epoach))
plt.legend()
plt.savefig('mid_QCNN_loss_curve_60day.png')
plt.show()


# In[41]:


fit = mean_squared_error(y_test,prediction_value)
print(fit)


plt.plot(preds_val/1000, color = 'g', label = 'Qcnn', linestyle = 'solid')
plt.plot(test_y_disorder.values/1000, color = 'r', label = 'real', linestyle = 'solid')
plt.xlabel('data')
plt.ylabel('kW')
plt.ylim((0,2))
plt.legend(loc = 'upper left')
plt.savefig('g_mid__All_1_curve.png')
plt.show()
#print(model_cov1.summary())
#print(model_pool.summary())
#print(model_cov2.summary())
#print(model_dense.summary())


plt.plot((preds_val/1000), color = 'g', label = 'Qcnn', linestyle = 'solid')
plt.xlabel('data')
plt.ylabel('kW')
plt.ylim((0,2))
plt.legend(loc = 'upper left')
plt.savefig('QCNN_curve.png')
plt.show()
print('Pred_test_loss_fitness',Pred_fitness)

plt.plot(test_y_disorder.values/1000, color = 'r', label = 'real', linestyle = 'solid')
plt.xlabel('data')
plt.ylabel('kW')
plt.ylim((0,2))
plt.legend(loc = 'upper left')
plt.savefig('__Real_1_curve.png')
plt.show()

plt.plot(test_y_disorder.values/1000, color = 'r', label = 'real', linestyle = 'solid')
plt.plot(preds_val/1000, color = 'g', label = 'Qcnn', linestyle = 'solid')
plt.xlabel('data')
plt.ylabel('kW')
plt.ylim((0,2))
plt.legend(loc = 'upper left')
plt.savefig('__All_1_curve.png')
plt.show()
