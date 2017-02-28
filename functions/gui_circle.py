# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 23:11:38 2016

Pick circular ROIs on an image using 4 mouse clicks
saves resulting ROI to csv file
load existing data if already exists by default
run by calling get_circle_rois()

img_file: this would typically be an average image from a video
force_input=0: can set to one to ask user for input even if csv file already exists


@author: johannes
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import wx
import os
import warnings

#disable deprecation warning caused by self.fig.canvas.stop_event_loop()
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)

#this function returns the ROI list as pandas df and the file name to stored csv
def get_circle_rois(img_file,output_file_suffix='',force_input=0):
    out_file=img_file[:-4]+output_file_suffix+'.csv'
    rois=gui_circles(img_file=img_file,out_file=out_file,force_input=force_input).rois
    return (rois,out_file)
    #wx dialog box

#create a GUI that shows the image, asks how many circles to draw and what the diameter of an arena is   
class gui_circles(object):
    def __init__(self,img_file,out_file,force_input=0):

        self.frame = cv2.imread(img_file)
        self.fig = plt.figure(figsize=(15,9))
        self.ax = self.fig.add_subplot(111)
        self.out_file=out_file
        xaxis = self.frame.shape[1]
        yaxis = self.frame.shape[0]
        
        #use WX to query user for number of arenas and arena diameter
        #ask() is a utility function to query user
        def ask(parent=None, message='', default_value=''):
            dlg = wx.TextEntryDialog(parent, message, defaultValue=default_value)
            dlg.ShowModal()
            result = dlg.GetValue()
            dlg.Destroy()
            return result
            
        #get user input only if output file doesn't exist already or if forced to regenerate
        if ~np.equal(~os.path.isfile(out_file),-2) or force_input:
            
            
            self.im = self.ax.imshow(self.frame[::-1,:], cmap='gray', extent=(0,xaxis,0,yaxis), picker=5)
            self.fig.canvas.draw()
            # Initialize wx App
            app = wx.App()
            app.MainLoop()
    
            # Call Dialog
            self.numCircles = np.int(ask(message = 'How Many Arenas?'))
            self.ArenaDiameter = np.int(ask(message = 'Arena diameter in mm ?'))
            
            self.ClickCount=0 #counter for points that belong to one circle
            
            #set onpick1() as callback for mouse click events
            self.fig.canvas.mpl_connect('button_press_event', self.onpick1)
            self.CirclesDone=0 #counter for circles defined already
            self.x = np.zeros([self.numCircles,4])
            self.y = np.zeros([self.numCircles,4])
            self.roiAll=[] #circular roi for each dish arena
            self.roiSq=[] #rectangular roi around each dish
            
            #block the main script until all points are collected
            self.fig.canvas.start_event_loop(timeout=-1)
            
            
        #if output file exists, read ROIs from file
        else:
            self.rois=pd.read_csv(out_file,header=0,index_col=0,sep=',')
            self.roiAll=self.rois.ix[:,0:3].values
            self.roiSq=self.rois.ix[:,3:7].values
            self.ArenaDiameter=self.rois.ix[:,7].values
            plt.close()

    #for each pick, check if still need points (want a total of 4 points)        
    def onpick1(self,event):
        
        if np.less(self.CirclesDone,self.numCircles):
            self.x[self.CirclesDone,self.ClickCount]=event.xdata
            self.y[self.CirclesDone,self.ClickCount]=event.ydata
            
            if np.less(self.ClickCount,3):
                self.ClickCount += 1
            else: #4 points defined, calculate circle
                self.ClickCount=0
                A=np.array([self.x[self.CirclesDone,:],self.y[self.CirclesDone,:],np.ones(4)])
                A=A.transpose()
                b=-self.x[self.CirclesDone,:]**2-self.y[self.CirclesDone,:]**2
                coefs=np.linalg.lstsq(A,b)
                roi=np.zeros(3)
                roi[0]=-coefs[0][0]/2 #center x
                roi[1]=-coefs[0][1]/2 #center y
                roi[2]=np.sqrt(roi[0]**2+roi[1]**2-coefs[0][2]) #radius
                circle=plt.Circle((roi[0],roi[1]),roi[2],color='b',fill=False)
                event.canvas.figure.gca().add_artist(circle)
                event.canvas.draw()
                self.CirclesDone +=1
                #print self.CirclesDone
                self.roiAll.append(roi)
                largest_square=np.min([self.frame.shape[0:1]-roi[0:1]-4,roi[0:1]])*2
                largest_square=largest_square-4+np.mod(largest_square,4)
                wh=np.min([largest_square,roi[2]*2+roi[2]*.1]) #width and height of roi around circular roi
                wh=np.min([largest_square,wh+4-np.mod(wh,4)]) #expand to multiple of 4 for videoCodecs
                self.roiSq.append([wh,wh,roi[0]-wh/2,roi[1]-wh/2,self.ArenaDiameter])
                
        else:
            print 'all arenas defined:'
            
            self.roiSq=np.array(list(self.roiSq)).astype('int')
            head, tail = os.path.split(self.out_file)
            headers=['circle center x','circle center y','circle radius','square width','square height','square top left x','square top left y','arena size']
            roi_both=np.hstack((self.roiAll,self.roiSq))
            self.rois=pd.DataFrame(data=roi_both,columns=headers,index=None)
            self.rois.to_csv(self.out_file,sep=',')
            print self.rois
            plt.close()
            
            #release the main script
            self.fig.canvas.stop_event_loop()
    
