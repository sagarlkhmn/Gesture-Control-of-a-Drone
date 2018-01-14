# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:02:39 2017

@author: Sagar Lakhmani
"""

from keras.models import load_model
from keras.utils import plot_model
import numpy as np
import os
model = load_model(path+'\Models\3DCNN.h5')
plot_model(model,to_file = 'model1.png',show_layer_names=True, show_shapes=True )

ypredict = model.predict(test_set)

Y_test = np_utils.to_categorical(y_test, 6)

#maxy = np.max(ypredict,axis=1)
#for i in range(len(ypredict)):
#    for j in range(ypredict.shape[1]):
#        if maxy[i] == ypredict[i][j]:
#            ypredict[i][j] = 1
#        else:
#            ypredict[i][j] = 0
#count = 0
#for i in range(len(ypredict)):
#    if np.array_equal(ypredict[i], Y_test[i]):
#        count = count + 1;
#
#print('Accuracy:', (count*100)/119)