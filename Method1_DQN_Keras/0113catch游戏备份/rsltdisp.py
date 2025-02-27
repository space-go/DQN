# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 11:09:52 2019

@author: heyon
"""

#matplotlib for rendering
import matplotlib.pyplot as plt
#Python image libarary for rendering
#from PIL import Image
#iPython display for making sure we can render the frames
from IPython import display
import time
import numpy as np

import config as c

def display_screen(action,points,input_t):
    #Function used to render the game screen
    #Get the last rendered frame
    #global last_frame_time
    print("Action %s, Points: %d" % (c.translate_action[action],points))
    #Only display the game screen if the game is not over
    if("End" not in c.translate_action[action]):
        #Render the game with matplotlib
        plt.imshow(input_t.reshape((c.grid_size,)*2),
               interpolation='none', cmap='gray')
        #Clear whatever we rendered before
        display.clear_output(wait=True)
        #And display the rendering
        display.display(plt.gcf())
    #Update the last frame time
    c.last_frame_time = set_max_fps(c.last_frame_time)
    
def set_max_fps(last_frame_time,FPS = 1):
    current_milli_time = lambda: int(round(time.time() * 1000))
    sleep_time = 1./FPS - (current_milli_time() - last_frame_time)
    if sleep_time > 0:
        time.sleep(sleep_time)
    return current_milli_time()

def moving_average_diff(a, n=100):
    diff = np.diff(a)
    ret = np.cumsum(diff, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
