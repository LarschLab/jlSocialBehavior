__author__ = 'jlarsch'

#import Tkinter
import tkFileDialog
import joFishHelper
import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

batchmode=0

if batchmode:
    df=pd.read_csv('d:/data/b/GRvsHet_pairs_10cmDish.csv',sep=',')
    #df=pd.read_csv('d:/data/b/GRvsHet_pairs_10cmDish_20160107.csv',sep=',')
    #df=pd.read_csv('d:/data/b/TL_isolatedVsGroup.csv',sep=',')
    experiment=[];
    
    with PdfPages('d:/data/b/GR2015.pdf') as pdf:
    #with PdfPages('d:/data/b/TL_isolatedVsGroup.pdf') as pdf:
        
        for index, row in df.iterrows():
            print 'processing: ', row['aviPath']
            currAvi=row['aviPath']
            #avi_path = tkFileDialog.askopenfilename(initialdir=os.path.normpath('d:/data/b'))
            #Tkinter.Tk().withdraw() # Close the root window - not working?
            experiment.append(joFishHelper.experiment(currAvi))
            experiment[index].plotOverview()
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
            
            
        df['ShoalIndex']=pd.DataFrame([f.ShoalIndex for f in experiment])
        df['totalTravel']=pd.DataFrame([f.totalPairTravel for f in experiment])
        
        fig, axes = plt.subplots(nrows=1, ncols=2)
        
        sns.boxplot(ax=axes[0],x='condition',y='ShoalIndex',data=df,width=0.2)
        sns.stripplot(ax=axes[0],x='condition',y='ShoalIndex',data=df, size=4,jitter=True,edgecolor='gray')

        sns.boxplot(ax=axes[1],x='condition',y='totalTravel',data=df,width=0.2)
        sns.stripplot(ax=axes[1],x='condition',y='totalTravel',data=df, size=4,jitter=True,edgecolor='gray')
        
        axes[0].set_ylim(0,1)
        axes[1].set_ylim(0,)
        
        pdf.savefig()
else:    
    avi_path = tkFileDialog.askopenfilename(initialdir=os.path.normpath('d:/data/b'))
    experiment=joFishHelper.experiment(avi_path)
    experiment.plotOverview()