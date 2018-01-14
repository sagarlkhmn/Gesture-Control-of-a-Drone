# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:25:45 2017

@author: Sagar Lakhmani
"""
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM,GRU
from keras.utils import plot_model
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils
from keras.models import load_model
from keras import optimizers
from keras import regularizers

import tensorflow
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing

name = input("Enter model name:")
# image specification
img_rows,img_cols,img_depth=32,32,150

K.set_image_dim_ordering('th')
# Training data

X_tr=[]           # variable to store entire dataset
X_test = []
#Reading boxing action class
i = 0

listing = os.listdir('kth dataset/boxing')

for vid in listing:
    vid = 'kth dataset/boxing/'+vid
    frames = []
    cap = cv2.VideoCapture(vid)
    fps = cap.get(5)
    print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
  

    for k in range(150):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    input=np.array(frames)

    print (input.shape)
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print (ipt.shape)

    
    if i>=80:
        X_test.append(ipt)
    else:
        X_tr.append(ipt)
    i = i+1
#Reading hand clapping action class
i = 0
listing2 = os.listdir('kth dataset/handclapping')

for vid2 in listing2:
    vid2 = 'kth dataset/handclapping/'+vid2
    frames = []
    cap = cv2.VideoCapture(vid2)
    fps = cap.get(5)
    print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))

    for k in range(150):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    print (input.shape)
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print (ipt.shape)

    if i>=80:
        X_test.append(ipt)
    else:
        X_tr.append(ipt)
    i = i+1

#Reading hand waving action class
i = 0
listing3 = os.listdir('kth dataset/handwaving')

for vid3 in listing3:
    vid3 = 'kth dataset/handwaving/'+vid3
    frames = []
    cap = cv2.VideoCapture(vid3)
    fps = cap.get(5)
    print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))

    for k in range(150):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    print (input.shape)
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print (ipt.shape)

    if i>=80:
        X_test.append(ipt)
    else:
        X_tr.append(ipt)
    i = i+1

#Reading jogging action class
i = 0
listing4 = os.listdir('kth dataset/jogging')

for vid4 in listing4:
    vid4 = 'kth dataset/jogging/'+vid4
    frames = []
    cap = cv2.VideoCapture(vid4)
    fps = cap.get(5)
    print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))

    for k in range(150):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    print (input.shape)
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print (ipt.shape)

    if i>=80:
        X_test.append(ipt)
    else:
        X_tr.append(ipt)
    i = i+1

#Reading running action class
i = 0
listing5 = os.listdir('kth dataset/running')

for vid5 in listing5:
    vid5 = 'kth dataset/running/'+vid5
    frames = []
    cap = cv2.VideoCapture(vid5)
    fps = cap.get(5)
    print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))

    for k in range(150):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    print (input.shape)
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print (ipt.shape)

    if i>=80:
        X_test.append(ipt)
    else:
        X_tr.append(ipt)
    i = i+1
  

#Reading walking action class  
i = 0
listing6 = os.listdir('kth dataset/walking')

for vid6 in listing6:
    vid6 = 'kth dataset/walking/'+vid6
    frames = []
    cap = cv2.VideoCapture(vid6)
    fps = cap.get(5)
    print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))

    for k in range(150):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    print (input.shape)
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print (ipt.shape)

    if i>=80:
        X_test.append(ipt)
    else:
        X_tr.append(ipt)
    i = i+1



X_tr_array = np.array(X_tr)   # convert the frames read into array
X_test_array = np.array(X_test)
num_samples = len(X_tr_array) 
print (num_samples)

num_testSamples = len(X_test_array) 
#Assign Label to each class

label=np.ones((num_samples,),dtype = int)
label[0:80]= 0
label[80:160] = 1
label[160:240] = 2
label[240:320] = 3
label[320:400]= 4
label[400:] = 5

testlabel=np.ones((num_testSamples,),dtype = int)
testlabel[0:20]= 0
testlabel[20:39] = 1
testlabel[39:59] = 2
testlabel[59:79] = 3
testlabel[79:99]= 4
testlabel[99:] = 5

train_data = [X_tr_array,label]

(X_train, y_train) = (train_data[0],train_data[1])
print('X_Train shape:', X_train.shape)

train_set = np.zeros((num_samples, 1, img_rows,img_cols,img_depth))

for h in range(num_samples):
    train_set[h][0][:][:][:]=X_train[h,:,:,:]
  

patch_size = 150    # img_depth or number of frames used for each video

print(train_set.shape, 'train samples')


test_set = np.zeros((num_testSamples, 1, img_rows,img_cols,img_depth))
test_data = [X_test_array,testlabel]
(X_test, y_test) = (test_data[0],test_data[1])
for h in range(num_testSamples):
    test_set[h][0][:][:][:]=X_test[h,:,:,:]
    
# CNN Training parameters

batch_size = 2
nb_classes = 6
nb_epoch = 50

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)


# number of convolutional filters to use at each layer
nb_filters = [32, 32]

# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [3, 3]

# level of convolution to perform at each layer (CONV x CONV)
nb_conv = [5,5]

# Pre-processing

train_set = train_set.astype('float32')

train_set -= np.mean(train_set)

train_set /=np.max(train_set)

test_set = test_set.astype('float32')

test_set -= np.mean(test_set)

test_set /=np.max(test_set)

model = Sequential()

# Define model
################################################## 3DCNN  ###########################

## Not needed
#model.add(Convolution3D(nb_filters[0],nb_depth=nb_conv[0], nb_row=nb_conv[0], nb_col=nb_conv[0], input_shape=(1, img_rows, img_cols, patch_size), activation='relu'))


#model.add(Convolution3D(nb_filters[0], kernel_dim1=nb_conv[0], kernel_dim2=nb_conv[0], kernel_dim3=nb_conv[0], input_shape=(1, img_rows, img_cols, img_depth), activation='relu'))
#
#model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))
#
#model.add(Dropout(0.5))
#
#model.add(Flatten())
#
#model.add(Dense(128, init='normal', activation='relu'))
#
#model.add(Dropout(0.5))
#
#model.add(Dense(nb_classes,init='normal'))
#
#model.add(Activation('softmax'))


####################################  LRCN   ##########################################################

#model = Sequential()

model.add(TimeDistributed(Conv2D(32, (5, 5), strides=(1, 1),
            activation='relu', padding='same'), input_shape=(1, img_rows, img_cols, patch_size)))
model.add(TimeDistributed(Conv2D(64, (5,5),
            kernel_initializer="he_normal", activation='relu')))
model.add(TimeDistributed(MaxPooling2D((5, 5), strides=(1, 1))))

#model.add(TimeDistributed(Flatten()))
###########################################################################################
##                                       Dont need this part
#model.add(TimeDistributed(Conv2D(128, (5,5),
#            padding='same', activation='relu')))
#model.add(TimeDistributed(Conv2D(128, (5,5),
#            padding='same', activation='relu')))
#model.add(TimeDistributed(MaxPooling2D((5, 5), strides=(1, 1))))
#
#model.add(TimeDistributed(Conv2D(128, (3,3),
#            padding='same', activation='relu')))
#model.add(TimeDistributed(Conv2D(128, (3,3),
#            padding='same', activation='relu')))
#model.add(TimeDistributed(MaxPooling2D((3, 3), strides=(1, 1))))

#model.add(TimeDistributed(Conv2D(256, (3,3),
#            padding='same', activation='relu')))
#model.add(TimeDistributed(Conv2D(256, (3,3),
#            padding='same', activation='relu')))
#model.add(TimeDistributed(MaxPooling2D((3, 3), strides=(1, 1))))
#        
#model.add(TimeDistributed(Conv2D(512, (3,3),
#            padding='same', activation='relu')))
#model.add(TimeDistributed(Conv2D(512, (3,3),
#            padding='same', activation='relu')))
#model.add(TimeDistributed(MaxPooling2D((3, 3), strides=(1, 1))))
##########################################################################################
#                                       LRCN Cont

model.add(TimeDistributed(Flatten()))
#
#model.add(Dropout(0.5))
#model.add(LSTM(64, return_sequences=False, dropout=0.5))
#model.add(Dense(nb_classes, activation='softmax'))

##############################################################################
#                                        LSTM Only
#model.add(LSTM(128, return_sequences=False, dropout=0.5,input_shape=(img_rows, img_cols, patch_size)))
#model.add(Dense(nb_classes, activation='softmax'))

#                                        GRU 
model.add(GRU(128, dropout=0.5))
model.add(Dense(nb_classes, activation='softmax'))

###########################################################################
#                                   Conlstm2d
#model.add(ConvLSTM2D(64, (3,3),input_shape=(360, patch_size,1, img_rows, img_cols),
#            padding='same',return_sequences=True,data_format='channels_first'))
#model.add(BatchNormalization())
#
#model.add(ConvLSTM2D(64, (3,3),
#            padding='same',return_sequences=True))
#model.add(BatchNormalization())
#model.add(ConvLSTM2D(64, (3,3),
#            padding='same',return_sequences=True))
#model.add(BatchNormalization())
#model.add(Conv3D(filters=1,kernel_size = (3,3,3),activation='relu',padding='same',data_format='channels_first'))


#
sgd = optimizers.sgd(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics = ['accuracy'])

# Split the data

X_train_new, X_val_new, y_train_new,y_val_new =  train_test_split(train_set, Y_train, test_size=0.25, random_state=4)


# Train the model

hist = model.fit(X_train_new, y_train_new, validation_data=(X_val_new,y_val_new),
          batch_size=batch_size,nb_epoch = nb_epoch,shuffle=True)


#hist = model.fit(train_set, Y_train, batch_size=batch_size,
#         nb_epoch=nb_epoch,validation_split=0.2, show_accuracy=True,
#           shuffle=True)


 # Evaluate the model
Lscore = model.evaluate(X_val_new, y_val_new, batch_size=batch_size)
print('Test score:', score)
#print('Test accuracy:', score) 



# Plot the results
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(50)

ypredict = model.predict(test_set)

Y_test = np_utils.to_categorical(y_test, ypredict.shape[1])

maxy = np.max(ypredict,axis=1)
for i in range(len(ypredict)):
    for j in range(ypredict.shape[1]):
        if maxy[i] == ypredict[i][j]:
            ypredict[i][j] = 1
        else:
            ypredict[i][j] = 0
count = 0
for i in range(len(ypredict)):
    if np.array_equal(ypredict[i], Y_test[i]):
        count = count + 1;
acc = (count*100)/len(ypredict)
print('Accuracy:', (count*100)/len(ypredict))
plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss(acc:'+str(acc)+')')
plt.grid(True)
plt.legend(['train','val'])
print (plt.style.available) # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.savefig(name +'.jpg')
model.save(name + '.h5')
#plot_model(model,to_file = 'model.png')

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.savefig(name +'_acc.jpg')