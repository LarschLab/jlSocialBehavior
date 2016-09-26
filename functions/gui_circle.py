# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 23:11:38 2016

@author: johannes
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import tkFileDialog
import wx
import os
import video_functions as vf


    #wx dialog box
        
class gui_circles(object):
    def __init__(self,img_file,out_file,force_input=0):

        self.frame = cv2.imread(img_file)
        self.fig = plt.figure(figsize=(6,9))
        self.ax = self.fig.add_subplot(111)
        xaxis = self.frame.shape[1]
        yaxis = self.frame.shape[0]
        
        if ~np.equal(~os.path.isfile(out_file),-2) or force_input:
            self.im = self.ax.imshow(self.frame[::-1,:], cmap='gray', extent=(0,xaxis,0,yaxis), picker=5)
            self.fig.canvas.mpl_connect('button_press_event', self.onpick1)
            self.fig.canvas.draw()
    
            self.ClickCount=0
            # Initialize wx App
            app = wx.App()
            app.MainLoop()
    
            # Call Dialog
            self.numPicks = np.int(ask(message = 'How Many Arenas?'))
            self.PicksDone=0
            self.x = np.zeros([self.numPicks,4])
            self.y = np.zeros([self.numPicks,4])
            self.roiAll=[] #circular roi for each dish arena
            self.roiSq=[] #rectangular roi around each dish
        else:
            #self.roiSq = pickle.load( open(out_file, "rb" ) )
            rois=pd.read_csv(out_file,header=0,index_col=0,sep=',')
            self.roiAll=rois.ix[:,0:2]
            self.roiSq=rois.ix[:,3:6]
    
    def ask(parent=None, message='', default_value=''):
        dlg = wx.TextEntryDialog(parent, message, defaultValue=default_value)
        dlg.ShowModal()
        result = dlg.GetValue()
        dlg.Destroy()
        return result
            
    def onpick1(self,event):
        if np.less(self.PicksDone,self.numPicks):
            self.x[self.PicksDone,self.ClickCount]=event.xdata
            self.y[self.PicksDone,self.ClickCount]=event.ydata
            
            if np.less(self.ClickCount,3):
                self.ClickCount += 1
            else: #4 points defined, calculate circle
                self.ClickCount=0
                A=np.array([self.x[self.PicksDone,:],self.y[self.PicksDone,:],np.ones(4)])
                A=A.transpose()
                b=-self.x[self.PicksDone,:]**2-self.y[self.PicksDone,:]**2
                coefs=np.linalg.lstsq(A,b)
                roi=np.zeros(3)
                roi[0]=-coefs[0][0]/2 #center x
                roi[1]=-coefs[0][1]/2 #center y
                roi[2]=np.sqrt(roi[0]**2+roi[1]**2-coefs[0][2]) #radius
                circle=plt.Circle((roi[0],roi[1]),roi[2],color='b',fill=False)
                event.canvas.figure.gca().add_artist(circle)
                event.canvas.draw()
                self.PicksDone +=1
                #print self.PicksDone
                self.roiAll.append(roi)
                wh=roi[2]*2+roi[2]*.1 #width and height of roi around circular roi
                wh=wh+16-np.mod(wh,16) #expand to multiple of 16 for videoCodecs
                self.roiSq.append([wh,wh,roi[0]-wh/2,roi[1]-wh/2])
                
        else:
            print 'all arenas defined'
            plt.close()
            self.roiSq=np.array(list(self.roiSq)).astype('int')
            head, tail = os.path.split(self.avi_path)
            headers=['c1','c2','c3','s1','s2','s3','s4']
            rois=pd.DataFrame([self.roiAll,self.roiSq])
            rois.to_csv(out_file,sep=',')
    
