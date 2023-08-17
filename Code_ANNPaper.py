# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 05:22:18 2023

@author: HOSSEIN
"""

#####################Import Libraries################
import scipy.io as spio
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.layers import Activation
from keras.layers import Dense
from keras import Input
from keras.models import Model
from keras.models import load_model

######################################################
#####################Defined Functions################
######################################################

######################Plotting training history
def plot_hist(net_hist):    
    Hist=net_hist.history
    type(Hist)
    Hist.keys()
    losses=Hist['loss']
    val_losses=Hist['val_loss']
    plt.figure()
    plt.plot(losses)
    plt.plot(val_losses)
    plt.xlabel('epochs')
    plt.ylabel('val')
    plt.legend(['loss','val_loss'])

######################Loading training data
def load_data(n_files):
    Ref = np.array ([ [ 0 for y in range(88 ) ] for x in range( 1 ) ])
    Chop = np.array ([ [ 0 for y in range(88 ) ] for x in range( 1 ) ])
    for n_files in range (n_files):
        filename="Data"+str(n_files+1)+".mat"
        mat = spio.loadmat(filename, squeeze_me=True)
        Chop1=mat['Chop']
        Ref1=mat['Ref']
        Chop=np.concatenate([Chop,Chop1])
        Ref=np.concatenate([Ref,Ref1])
    Chop = np.delete(Chop, 0, 0)
    Ref = np.delete(Ref, 0, 0)
    maxRef=np.amax(Ref,1)
    return maxRef,Chop

######################Generate the model
def model_generator():
    
    input_signal = Input(shape=(88,))
    d=Dense(10,kernel_initializer='uniform')(input_signal)
    d = Activation('relu')(d)
    d=Dense(5,kernel_initializer='uniform')(d)
    d = Activation('relu')(d)
    d=Dense(1,kernel_initializer='uniform')(d)
    output = Activation('linear')(d)
    model=Model(input_signal,output)
    model.summary()
    return model

######################Training the model with K-fold corss validation    
def K_Fold_training(model,optimizer,Ref,Chop,kfolds,ndata):
    Batch=np.floor(ndata/kfolds).astype('int')
    Indx_kfold=list(np.random.choice(ndata, size=ndata,replace=False))
    Selected_net=np.random.choice(kfolds, size=1)
    for kk in range(kfolds):
        val_indx=Indx_kfold[kk*Batch:(kk+1)*Batch]
        train_indx=list(set(Indx_kfold)-set(val_indx))
        X_train=Chop[train_indx,:]
        Y_train=Ref[train_indx]
        X_val=Chop[val_indx,:]
        Y_val=Ref[val_indx]
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        net_hist=model.fit(X_train,Y_train,batch_size=512,epochs=120,validation_data=(X_val, Y_val))
        plot_hist(net_hist)
        if kk==0:
            model.save('network.h5')

######################Loading test data   

def Load_test_data(name):    
    mat1 = spio.loadmat(name+'.mat', squeeze_me=True)
    Ref=mat1['RefImg']
    Chopped=mat1['ChopImg']
    Signal_trace=mat1['SelectedPhase']
    return Ref,Chopped,Signal_trace

######################Prediction on the test data with trained model
def Evaluate_net(model,Ref_Img,Chopped_Img,Signal_trace):
            HH=len(Signal_trace)
            ZZ=len(Signal_trace[0])
            WW=len(Signal_trace[0,0])
            RawData=np.zeros(shape=[HH*WW,ZZ])
            nn=0
            for ww in range(WW):
                for hh in range(HH):
                    AA=np.array(Signal_trace[hh,:,ww])
                    RawData[nn,:]=AA-np.mean(AA)
                    nn=nn+1
            Predict=mymodel.predict([RawData])        
            Predict=Predict.reshape(WW,HH)
            Predict=Predict.transpose() 
            return Predict

######################################################        
############## Code's Main Body ######################
######################################################

n_selected_data=96000   #Number of training datapoints
Ref,Chop=load_data(8)   #Number of opened files containing training datasets
kfolds=10               #Number of folds in K-fold method

#Shuffling training dataset and selecting n data
Indx=np.random.choice(len(Ref), size=n_selected_data,replace=False)
Selected_Ref=Ref[Indx]
Selected_Chop=Chop[Indx,:]

#Plot a sample signal 
Rand_INDX=np.random.choice(len(Chop), size=1)
plt.figure()
plt.plot(Chop[Rand_INDX[0],:])
plt.title('sample of Short-trace signal')
plt.ylabel('Amp(rad)')
plt.xlabel('datapoints')
#Generating the model
mymodel=model_generator()
opt = Adam(learning_rate=0.4e-3, decay=1e-3 / 200)

#Training the model
K_Fold_training(mymodel,opt,Selected_Ref,Selected_Chop,kfolds,n_selected_data)
model = load_model('network.h5')

#Evaluation on the trained model
Ref_Img,Chopped_Img,Signal_trace=Load_test_data("Val1") #Select between Val1 to Val5 
        
Gen_Img=Evaluate_net(model,Ref_Img,Chopped_Img,Signal_trace)

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(Ref_Img, vmin=0, vmax=0.1)
plt.title('Ground_truth')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(Chopped_Img, vmin=0, vmax=0.1)
plt.title('Initial Image')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(Gen_Img, vmin=0, vmax=0.1)
plt.title('Network')
plt.axis('off')

