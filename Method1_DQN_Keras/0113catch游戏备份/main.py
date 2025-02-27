# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 11:09:52 2019

@author: heyon
"""

#First we import some libraries
#Json for loading and saving the model (optional)
#import json
import matplotlib.pyplot as plt
#Python image libarary for rendering
#from PIL import Image
#iPython display for making sure we can render the frames
#from IPython import display
#numpy for handeling matrix operations
#import numpy as np
#time, to, well... keep track of time
#import time

#seaborn for rendering
import seaborn
#Keras is a deep learning libarary
#from keras.models import model_from_json
#from keras.models import Sequential
#from keras.layers.core import Dense
#from keras.optimizers import sgd


import config as c
#environment describe
#import envidesc as ed
#resultdisplay
import rsltdisp as rd
#train and test
import tandt as tt



############################################################/*mainprocess*/

#Setup matplotlib so that it runs nicely in iPython
#%matplotlib inline
#setting up seaborn
seaborn.set()
#Define model
model = tt.baseline_model(c.grid_size,c.num_actions,c.hidden_size)
model.summary()
    
# Train the model
# For simplicity of the noteb
hist = tt.train(model, c.epoch,verbose=0)#记录了5000代中，有多少代是positive的数据。
print("Training done")

tt.test(model)

plt.plot(rd.moving_average_diff(hist))
plt.ylabel('Average of victories per game')
plt.show()