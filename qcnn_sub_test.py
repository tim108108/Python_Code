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
                    
                    model_cov1.get_layer('cov').set_weights([np.reshape(_filter_np[:,:,:,filter_n],
                                                            [filter_i,filter_j,feature_depth,1]),
                                                             biases_np[filter_n,:]])
                    
                    _conv1_num = model_cov1.predict(input_feature[:,i:i+filter_i, j:j+filter_j, :])
                    ref_np[:,i,j,:] = np.reshape(_conv1_num,[batch,1])
                else:
                    ref_np[:,i,j,:]=0
        conv_output[:,:,:,filter_n] = ref_np[:,:,:,0]
    return conv_output


# In[11]:


#data_input

#load_data
path1=r'./data/' #訓練 讀檔的路徑
file1=glob.glob(os.path.join(path1, "2017*.csv"))
df_train=pd.DataFrame([])

for f in file1:
    out = pd.read_csv(f,sep=',')
    index_names_1=out.loc[:,'time'] #把 'time'那行的數據 讀出 => index_names_1
    out.index=index_names_1 #把index的label 以'time'這行的時間命名
    out=out.loc['09:00:00':'17:00:00',:] #用index的label 去保留 要的時間
    df_train=pd.concat([df_train,out],axis=0,ignore_index=True)

x = df_train[['tep', 'li']]  # # # #
y = df_train[['kw']]         # 
y = y/2

x_2 = x.iloc[:,0:2].values  
x_Amplification = np.column_stack([x_2,x_2,x_2,x_2,x_2,x_2,x_2,x_2,x_2,x_2,x_2,x_2,x_2[:,0:1]])  # 5*5

# 0.75 #
train_x_disorder, test_x_disorder, train_y_disorder, test_y_disorder = train_test_split(x_Amplification, y,train_size=0.75, random_state=1)

#標準化
x_min_max_scaler = MinMaxScaler().fit(train_x_disorder) #以x_train_data 為最大最小標準化 的標準
y_min_max_scaler = MinMaxScaler().fit(train_y_disorder)
#x_train=x_min_max_scaler.transform(train_x_disorder) #以行為單位 做最大最小標準化
#y_train=y_min_max_scaler.transform(train_y_disorder)

#x_test=x_min_max_scaler.transform(test_x_disorder)#以x_train_data 為最大最小標準化 的標準 去對x_test_data 進行最大最小標準化
#y_test=y_min_max_scaler.transform(test_y_disorder)

#scalerX = joblib.load('QCNNscaler201804260428x.pkl') #載入之前模型標準的縮放器
#scalerY = joblib.load('QCNNscaler201804260428y.pkl') #載入之前模型標準的縮放器
x_test = x_min_max_scaler.transform(test_x_disorder)
y_test = y_min_max_scaler.transform(test_y_disorder)


# In[12]:


#x_train = np.float32(np.reshape(x_train, [-1, 5, 5, 1]))
#y_train = np.float32(y_train)

x_test = np.float32(np.reshape(x_test, [-1, 5, 5, 1]))
y_test = np.float32(y_test)


#print(x_train.shape)
#print(x_train.dtype)

print(x_test.shape)
print(x_test.dtype)

#print(y_train.shape)
#print(y_train.dtype)

print(y_test.shape)
print(y_test.dtype)


# In[13]:


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


# In[14]:


mid_Sub_best_bcov1 = np.load('mid_Sub_best_bcov1.npy')
mid_Sub_best_cw1 = np.load('mid_Sub_best_cw1.npy')
mid_Sub_best_cb1 = np.load('mid_Sub_best_cb1.npy')

mid_Sub_best_bcov2 = np.load('mid_Sub_best_bcov2.npy')
mid_Sub_best_cw2 = np.load('mid_Sub_best_cw2.npy')
mid_Sub_best_cb2 = np.load('mid_Sub_best_cb2.npy')

mid_Sub_best_fw1 = np.load('mid_Sub_best_fw1.npy')
mid_Sub_best_fb1 = np.load('mid_Sub_best_fb1.npy')

mid_Sub_best_fw2 = np.load('mid_Sub_best_fw2.npy')
mid_Sub_best_fb2 = np.load('mid_Sub_best_fb2.npy')


# In[17]:


# mid Prediction    
mid_conv1_output_pre = np.zeros(shape=[6966, 4, 4, 16], dtype=np.float32)
mid_conv2_output_pre = np.zeros(shape=[6966, 1, 1, 24], dtype=np.float32)

mid_conv1_output_pre = qconv(x_test,mid_Sub_best_cw1,mid_Sub_best_bcov1,mid_Sub_best_cb1,mid_conv1_output_pre,model_cov1)
mid_pooling_pre = model_pool.predict(mid_conv1_output_pre)

mid_conv2_output_pre = qconv(mid_pooling_pre,mid_Sub_best_cw2,mid_Sub_best_bcov2,mid_Sub_best_cb2,mid_conv2_output_pre,model_cov2)

model_dense.get_layer('Dense_1').set_weights([mid_Sub_best_fw1,mid_Sub_best_fb1])
model_dense.get_layer('output_layer').set_weights([mid_Sub_best_fw2,mid_Sub_best_fb2])
mid_prediction_value = model_dense.predict(mid_conv2_output_pre)
                
#prediction_value[np.where(prediction_value > 1.0)] = np.float32(1.0)
#prediction_value[np.where(prediction_value < 0.0)] = np.float32(0.0)
                
mid_preds_val=y_min_max_scaler.inverse_transform(mid_prediction_value)


# In[18]:


Pred_fitness = mean_squared_error(y_test,mid_prediction_value)
                                         
plt.plot(test_y_disorder.values/1000, color = 'r', label = 'real', linestyle = 'solid')
plt.xlabel('data')
plt.ylabel('kW')
plt.ylim((0,2))
plt.legend(loc = 'upper left')
plt.savefig('__Real_1_curve.png')
plt.show()

plt.plot(mid_preds_val/1000, color = 'g', label = 'Qcnn', linestyle = 'solid')
plt.xlabel('data')
plt.ylabel('kW')
plt.ylim((0,2))
plt.legend(loc = 'upper left')
plt.savefig('QCNN_test_curve_pre.png')
plt.show()
print('Pred_test_loss_fitness',Pred_fitness)

plt.plot(test_y_disorder.values/1000, color = 'r', label = 'real', linestyle = 'solid')
plt.plot(mid_preds_val/1000, color = 'g', label = 'Qcnn', linestyle = 'solid')
plt.xlabel('data')
plt.ylabel('kW')
plt.ylim((0,2))
plt.legend(loc = 'upper left')
plt.savefig('__All_1_curve.png')
plt.show()


#
# final = combine(up_preds_val,mid_preds_val,down_preds_val)
# plt.plot(test_y_disorder.values/1000, color = 'r', label = 'real', linestyle = 'solid')
# plt.plot(final/1000, color = 'g', label = 'Qcnn', linestyle = 'solid')
# plt.xlabel('data')
# plt.ylabel('kW')
# plt.ylim((0,2))
# plt.legend(loc = 'upper left')
# plt.savefig('__All_combine_1_curve.png')
# plt.show()
#
# Pred_fitness = mean_squared_error(test_y_disorder.values,final)
# print('Pred_test_loss_fitness',Pred_fitness)
#
# acc = final/test_y_disorder.values
# #print(final)
# #print(test_y_disorder.values)
# #print(acc)
# sum = 0
# for i in range(len(acc)):
#     sum = sum + acc[i]
# accurate = (sum/len(acc))*100
# print(accurate)
#
#
# a = final - test_y_disorder.values
# for i in range(len(a)):
#     if a[i] < 0:
#         a[i]=a[i]*-1
# b = a/test_y_disorder.values
# sum1 = 0
# for i in range(len(b)):
#     sum1 = sum1 + b[i]
#
# accurate1 = ( 1-(sum1/len(b)))*100
# print(accurate1)





