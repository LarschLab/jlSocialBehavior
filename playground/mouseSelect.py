# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:27:53 2016

@author: jlarsch
"""



#import matplotlib.pyplot as plt
#from matplotlib.patches import Rectangle
#
#class Annotate(object):
#    def __init__(self):
#        self.ax = plt.gca()
#        self.rect = Rectangle((0,0), 1, 1)
#        self.x0 = None
#        self.y0 = None
#        self.x1 = None
#        self.y1 = None
#        self.ax.add_patch(self.rect)
#        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
#        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
#        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
#
#    def on_press(self, event):
#        print('press')
#        self.x0 = event.xdata
#        self.y0 = event.ydata
#
#    def on_release(self, event):
#        print('release')
#        self.x1 = event.xdata
#        self.y1 = event.ydata
#        self.rect.set_width(self.x1 - self.x0)
#        self.rect.set_height(self.y1 - self.y0)
#        self.rect.set_xy((self.x0, self.y0))
#        self.ax.figure.canvas.draw()
#        
#    def on_motion(self, event):
##        'on motion we will move the rect if the mouse is over us'
##        if self.press is None: return
##        if event.inaxes != self.rect.axes: return
##        x0, y0, xpress, ypress = self.press
##        dx = event.xdata - xpress
##        dy = event.ydata - ypress
##        #print('x0=%f, xpress=%f, event.xdata=%f, dx=%f, x0+dx=%f' %
##        #      (x0, xpress, event.xdata, dx, x0+dx))
##        self.rect.set_x(x0+dx)
##        self.rect.set_y(y0+dy)
##
##        self.rect.figure.canvas.draw()
#        self.x1 = event.xdata
#        self.y1 = event.ydata
#        self.rect.set_xy((self.x1, self.y1))
#        self.ax.figure.canvas.draw()
#        
#
#a = Annotate()
#plt.show()

from matplotlib.widgets import Lasso
from matplotlib.colors import colorConverter
from matplotlib.collections import RegularPolyCollection
from matplotlib import path

import matplotlib.pyplot as plt
from numpy import nonzero
from numpy.random import rand


class Datum(object):
    colorin = colorConverter.to_rgba('red')
    colorout = colorConverter.to_rgba('blue')

    def __init__(self, x, y, include=False):
        self.x = x
        self.y = y
        if include:
            self.color = self.colorin
        else:
            self.color = self.colorout


class LassoManager(object):
    def __init__(self, ax, data):
        self.axes = ax
        self.canvas = ax.figure.canvas
        self.data = data

        self.Nxy = len(data)

        facecolors = [d.color for d in data]
        self.xys = [(d.x, d.y) for d in data]
        fig = ax.figure
        self.collection = RegularPolyCollection(
            fig.dpi, 6, sizes=(100,),
            facecolors=facecolors,
            offsets=self.xys,
            transOffset=ax.transData)

        ax.add_collection(self.collection)

        self.cid = self.canvas.mpl_connect('button_press_event', self.onpress)

    def callback(self, verts):
        facecolors = self.collection.get_facecolors()
        p = path.Path(verts)
        ind = p.contains_points(self.xys)
        for i in range(len(self.xys)):
            if ind[i]:
                facecolors[i] = Datum.colorin
            else:
                facecolors[i] = Datum.colorout

        self.canvas.draw_idle()
        self.canvas.widgetlock.release(self.lasso)
        del self.lasso

    def onpress(self, event):
        if self.canvas.widgetlock.locked():
            return
        if event.inaxes is None:
            return
        self.lasso = Lasso(event.inaxes,
                           (event.xdata, event.ydata),
                           self.callback)
        # acquire a lock on the widget drawing
        self.canvas.widgetlock(self.lasso)

if __name__ == '__main__':

    data = [Datum(*xy) for xy in rand(100, 2)]

    ax = plt.axes(xlim=(0, 1), ylim=(0, 1), autoscale_on=False)
    lman = LassoManager(ax, data)

    plt.show()