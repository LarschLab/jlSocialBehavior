import numpy as np
import os
import matrixUtilities_joh as mu
import random
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as sta
import plotFunctions_joh as johPlt
import randSpacing
from matplotlib.backends.backend_pdf import PdfPages
import video_functions as vf

class ExperimentMeta(object):
    #This class collects paths, arena and video parameters
    def __init__(self,path,arenaDiameter_mm=100):
        self.arenaDiameter_mm = arenaDiameter_mm
        self.arenaCenterPx = [0,0] #assign later from class 'Pair'
        self.pxPmm = 0 #assign later from class 'Pair'
        
        #If a video file name is passed, collect video parameters
        if path.endswith('.avi') or path.endswith('.mp4'):
            self.aviPath = path
            #get video meta data
            vp=vf.getVideoProperties(path) #video properties
            self.ffmpeginfo = vp
            self.videoDims = [vp['width'] , vp['height']]
            self.numFrames=vp['nb_frames']
            self.fps=vp['fps']
            #self.date=vp['TAG:date']
        else:
            self.fps=30
            
        #concatenate dependent file paths (trajectories, pre-analysis)
        head, tail = os.path.split(path)
        head=os.path.normpath(head)
        
        trajectoryPath = os.path.join(head,'trajectories_nogaps.mat')
        if np.equal(~os.path.isfile(trajectoryPath),-2):
            self.trajectoryPath = trajectoryPath
        else:
            trajectoryPath = os.path.join(head,'trajectories.mat')
            if np.equal(~os.path.isfile(trajectoryPath),-2):
                self.trajectoryPath = trajectoryPath
                
        AnSizeFilePath = os.path.join(head,'animalSize.txt')
        if np.equal(~os.path.isfile(AnSizeFilePath),-2):
            self.AnSizeFilePath = AnSizeFilePath
        
        self.dataPath = os.path.join(head,'analysisData.mat')

class Trajectory(object):
    def __init__(self,xy=np.array([])):
        self.xy=xy
        
    def x(self):
        return self.xy[:,0]
    
    def y(self):
        return self.xy[:,1]
        
       
class Animal(object):
    def __init__(self,ID=0,trajectory=[],expInfo=[]):
        self.expInfo=expInfo
        self.ID=ID
        self.position=Trajectory()
        self.positionPol=Trajectory() #x= theta, y=rho
        self.d_position=Trajectory()
        self.dd_position=Trajectory()
        
        #position of neighbor
        self.N_relPos=Trajectory()
        self.N_relPosPolRotCart=Trajectory()
        
        self.travel=np.array([])
        self.totalTravel=np.array([])
        self.speed=np.array([])
        self.accel=np.array([])
        self.heading=np.array([])
        self.d_heading=np.array([])
        
        self.position.xy=trajectory
        
        self.update_trajectory_transforms()
        
        
        #how far did the animal go out towards center?
        #this can be used to determine upper/lower animal in stacks
    
    def update_trajectory_transforms(self):
        posPol=[mu.cart2pol(*self.position.xy.T)]
        self.positionPol.xy=np.squeeze(np.array(posPol).T)
        
        #change in position for x and y
        self.d_position.xy =np.diff(self.position.xy,axis=0)
        
        self.dd_position.xy =np.diff(self.d_position.xy,axis=0)
        
        #travel distance (cartesian displacement)
        self.travel=mu.distance(*self.d_position.xy.T)
        
        #tavel speed
        self.speed=self.travel*self.expInfo.fps
        self.totalTravel=np.nansum(np.abs(self.travel))
        
        #travel acceleration
        self.accel=np.diff(self.speed)
        self.heading=mu.cart2pol(*self.d_position.xy.T)[0] #heading[0] = heading, heading[1] = speed
        #heading: pointing right = 0, pointung up = pi/2      
        self.d_heading=np.diff(self.heading)
    
        
    def get_neighbor_position(self, neighbor_animal):
        #position of neighbor relative to current animal. (where current animal has a neighbor)
        
        self.N_relPos.xy=neighbor_animal.position.xy-self.position.xy
        relPosPol=[mu.cart2pol(*self.N_relPos.xy.T)]
        relPosPolRot=np.squeeze(np.array(relPosPol).T)
        relPosPolRot=relPosPolRot[1:,:]
        relPosPolRot[:,0]=relPosPolRot[:,0]-self.heading
        tmp=[mu.pol2cart(relPosPolRot[:,0],relPosPolRot[:,1])]
        self.N_relPosPolRotCart.xy=np.squeeze(np.array(tmp).T)
        
        mapBins=np.arange(-31,32)
        self.neighborMat=np.zeros([62,62])
        #creates the neighbormat for current animal (where neighbor was)
        #this map seems flipped both horizontally and vertically! vertical is corrected at plotting.
        self.neighborMat=np.histogramdd(self.N_relPosPolRotCart.xy,bins=[mapBins,mapBins])[0]
    
    def getNeighborForce(self):
        #position here should be transformed (rotated, relative). [:,0,:] is the position of 
        mapBins=np.arange(-31,32)
        self.ForceMat=sta.binned_statistic_2d(self.N_relPosPolRotCart.x()[1:],self.N_relPosPolRotCart.y()[1:],self.accel,bins=[mapBins,mapBins])[0]
        
    def analyze_neighbor_interactions(self,neighbor_animal):
        self.get_neighbor_position(neighbor_animal)
        self.getNeighborForce()

        
        #self.dwellH,self.dwellL,self.dwellHTL,self.dwellLTH=getShoalDwellTimes(self.IAD)

    def get_outward_venture(self,maxCenterDistance=50):
        #distance from center
        histData=self.positionPol.y()
        histBins=np.linspace(0,maxCenterDistance,100)   
        self.PolhistBins=histBins
        
        self.Pol_n =np.histogram(histData[~np.isnan(histData)],bins=histBins,normed=1)[0]        
    
class Pair(object):
    #Class for trajectories of one pair of animals.
    #Calculation and storage of dependent time series: speed, heading, etc.
    def __init__(self, trajectories, expInfo):
        self.animals=[]
        self.animals.append(Animal(1,trajectories[:,0,:],expInfo))
        self.animals.append(Animal(2,trajectories[:,1,:],expInfo))
        self.get_stack_order()
        self.get_IAD()
        
        self.animals[0].analyze_neighbor_interactions(self.animals[1])
        self.animals[1].analyze_neighbor_interactions(self.animals[0])

        
    def get_stack_order(self):
        for x in self.animals:
            x.get_outward_venture(self.max_out_venture())
        
        out_venture_all=[x.Pol_n for x in self.animals]
        #figure out who is on top. Animal in top arena can go outwards further
        if out_venture_all[0][-2]==0: #second last bin of animal0 is empty, meaning animal1 went out further ->was on top
            self.StackTopAnimal=np.array([0,1])
        elif out_venture_all[1][-2]==0:
            self.StackTopAnimal=np.array([1,0])
        else: #no animal went out more than one bin further than the other -> likely no stack experiment
            self.StackTopAnimal=np.array([0,0])   
            

            
            
    def get_IAD(self):
        dist=Trajectory()
        dist.xy=self.animals[0].position.xy-self.animals[1].position.xy
        self.IAD = np.sqrt(dist.x()**2 + dist.y()**2) 
    
        #absolute inter animal distance IAD
        
        self.IAD_m=np.nanmean(self.IAD)
        histBins=np.arange(100)
        self.IADhist =np.histogram(self.IAD,bins=histBins,normed=1)[0]
        


    def getShoalDwellTimes(IAD):
        IADsm=mu.runningMean(IAD,30)
        lowThreshold=np.nanmax(IADsm)/1.5
        
        lowIAD=(np.less(IAD,lowThreshold)).astype(int)
        hiIAD=(np.greater(IAD,lowThreshold)).astype(int)
        
        #transitions from low to high inter animal distance and vice versa
        LowToHigh=np.where(np.equal((hiIAD[:-1]-hiIAD[1:]),-1))[0]
        HighToLow=np.where(np.equal((lowIAD[:-1]-lowIAD[1:]),-1))[0]
        
        #number of transitions to use below
        maxL=np.min([np.shape(HighToLow)[0],np.shape(LowToHigh)[0]])-2
        
        #How long are High and low dwell times?
        #calculate from transition times. Order depends on starting state of data
        
        if HighToLow[0]>LowToHigh[0]: #meaning starting low
            HiDwell=HighToLow[0:maxL]-LowToHigh[0:maxL]
            LowDwell=LowToHigh[1:maxL]-HighToLow[0:maxL-1]
        else: #meaning starting high
            HiDwell=HighToLow[1:maxL]-LowToHigh[0:maxL-1]
            LowDwell=LowToHigh[0:maxL]-HighToLow[0:maxL]
    
        return HiDwell,LowDwell,HighToLow,LowToHigh

        
    def avgSpeed(self):
        a1=np.nanmean(self.animals[0].speed)
        a2=np.nanmean(self.animals[1].speed)
        return np.array([a1,a2])
        
    def max_out_venture(self):
        mov=[]
        mov.append([x.positionPol.y() for x in self.animals])
        return np.nanmax(mov)

    def get_var_from_all_animals(self, var):
        #this function returns a specified variable from all animals as a matrix.
        #animal number will be second dimension
        
        tmp=self.animals
        for i in range(len(var)):
            tmp=[getattr(x, var[i]) for x in tmp]
        return np.stack(tmp,axis=1)

class shiftedPair(object):
    #Class to calculate null hypothesis time series using shifted pairs of animals
    def __init__(self, pair,expInfo):
        self.nRuns=10
        self.minShift=5*60*expInfo.fps
        self.sPair=[]
        #generate nRuns instances of Pair class with one animal time shifted against the other
        for i in range(self.nRuns):
            traShift=pair.get_var_from_all_animals(['position','xy'])
            
            shiftIndex=int(random.uniform(self.minShift,traShift.shape[0]-self.minShift))
            #time-rotate animal 0, keep animal 1 as is
            traShift[:,0,:]=np.roll(traShift[:,0,:],shiftIndex,axis=0)
            self.sPair.append(Pair(traShift,expInfo))
            
        #calculate mean and std IAD for the goup of shifted instances
        self.spIAD_m = np.nanmean([x.IAD_m for x in self.sPair])
        self.spIAD_std = np.nanstd([x.IAD_m for x in self.sPair])

def shiftedPairSystematic(tra,expInfo,nRuns):
    #function to test systematically the effect of the extent of the time shift
    shiftInterval=expInfo.fps
    sIADList=[]
    #generate nRuns instances of Pair class with one animal time shifted against the other
    for i in range(nRuns):
        traShift=tra.positionPx.copy()
        shiftIndex=i*shiftInterval
        #time-rotate animal 0, keep animal 1 as is
        traShift[:,0,:]=np.roll(traShift[:,0,:],shiftIndex,axis=0)
        tmpPair=Pair(traShift,expInfo)
        sIADList.append(tmpPair.IAD_m)
    return sIADList
    
    

            
    
class xy_point(object):
    def __init__(self,coordinates):
        self.x=coordinates[0]
        self.y=coordinates[1]

def leadership(PosMat):
    front=np.sum(np.sum(PosMat[:31,:,:],axis=1),axis=0)
    back =np.sum(np.sum(PosMat[32:,:,:],axis=1),axis=0)
    LeadershipIndex=(front-back)/(front+back)
    return LeadershipIndex
        
class experiment(object):
    #Class to collect, store and plot data belonging to one experiment
    def __init__(self,path):
        self.expInfo=ExperimentMeta(path)
        self.AnSize=np.array(np.loadtxt(self.expInfo.AnSizeFilePath, skiprows=1,dtype=int))
        mat=scipy.io.loadmat(self.expInfo.trajectoryPath)
        self.rawTra=mat['trajectories']
        #take out nan in the beginnin DONT do this for now, this would shift trace averages or require correction!
        nanInd=np.where(np.isnan(self.rawTra))
        if np.equal(np.shape(nanInd)[1],0) or np.greater(np.max(nanInd),1000):
            LastNan=0
        else:
            LastNan=np.max(nanInd)+1
        
        #rawTra=rawTra[LastNan:,:,:]
        self.skipNanInd=LastNan

        maxPixel=np.nanmax(self.rawTra,0)
        minPixel=np.nanmin(self.rawTra,0)
        self.expInfo.trajectoryDiameterPx=np.mean(maxPixel-minPixel)
        #expInfo.pxPmm=expInfo.trajectoryDiameterPx/expInfo.arenaDiameter_mm
        self.expInfo.pxPmm=8.6
        self.expInfo.arenaCenterPx=np.mean(maxPixel-(self.expInfo.trajectoryDiameterPx/2),axis=0)
        
        position=(self.rawTra-self.expInfo.arenaCenterPx) / self.expInfo.pxPmm
        
        #proceed with animal-pair analysis if there is more than one trajectory
        if np.shape(self.rawTra)[1]>1:
            self.Pair=Pair(position.copy(),self.expInfo)
        
            #self.AnShape=AnimalShapeParameters()
            
            #generate shifted control 'mock' pairs
            self.sPair=shiftedPair(self.Pair,self.expInfo)
            
            #calculate mean IAD histogram for shifted pairs
            IADHistall=[]
            IADHistall.append([x.IADhist[0:30*60*90] for x in self.sPair.sPair])
            self.spIADhist_m=np.nanmean(IADHistall,axis=1)            
            
            self.ShoalIndex=(self.sPair.spIAD_m-self.Pair.IAD_m)/self.sPair.spIAD_m
            #self.totalPairTravel=sum(self.Pair.totalTravel)
            self.avgSpeed=self.Pair.avgSpeed
            probTra=mat['probtrajectories']
            self.idQuality=np.mean(probTra[probTra>=0])*100
            
    
    def plotOverview(self,condition='notDefined'):
        outer_grid = gridspec.GridSpec(4, 4)        
        plt.figure(figsize=(8, 8))   
        #plt.text(0,0.95,path)
        plt.figtext(0,.01,self.expInfo.aviPath)
        plt.figtext(0,.03,condition)
        plt.figtext(0,.05,self.idQuality)
        
        plt.subplot(4,4,1,rasterized=True)
        plt.cla()
        plt.plot(self.Pair.animals[0].position.x(),self.Pair.animals[0].position.y(),'b.',markersize=1,alpha=0.1)
        plt.plot(self.Pair.animals[1].position.x(),self.Pair.animals[1].position.y(),'r.',markersize=1,alpha=0.1)
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        plt.title('raw trajectories')  
        
        plt.subplot(4,4,3)
        plt.cla()
        PolhistBins=self.Pair.animals[0].PolhistBins
        plt.step(PolhistBins[:-1], self.Pair.animals[0].Pol_n,'b',lw=2,where='mid')
        plt.step(PolhistBins[:-1], self.Pair.animals[1].Pol_n,'r',lw=2,where='mid')
        plt.ylim([0, .1])

        
        
        #plot IAD time series
        x=np.arange(float(np.shape(self.Pair.IAD)[0]))/(self.expInfo.fps*60)
        plt.subplot(4,1,2,rasterized=True)
        plt.cla()
        plt.plot(x,self.Pair.IAD,'b.',markersize=1)
        plt.xlim([0, 90]) 
        plt.xlabel('time [minutes]')
        plt.ylabel('IAD [mm]')
        plt.title('Inter Animal Distance over time')
        
        #IAD histogram for raw and shifted data and simulated random dots
        plt.subplot(4,4,2)
        plt.cla()
        #get rid of nan
        IAD=self.Pair.IAD
        IAD=IAD[~np.isnan(IAD)]
        histBins=np.arange(100)
        n, bins, patches = plt.hist(IAD, bins=histBins, normed=1, histtype='stepfilled')
        plt.step(histBins[:-1],self.spIADhist_m.T,'k',lw=1)
        #simulate random uniform spacing
        rad=(self.expInfo.trajectoryDiameterPx/self.expInfo.pxPmm)/2
        num=1000
        dotList,dotND = randSpacing.randomDotsOnCircle(rad,num)
        n, bins, patches = plt.hist(dotND, bins=histBins, normed=1, histtype='step')
        plt.xlabel('IAD [mm]')
        plt.ylabel('p')
        plt.title('IAD')
        plt.ylim([0, .05])
        
        plt.subplot(4,4,4)
        plt.cla()
        x=[1,2]
        y=[self.Pair.IAD_m,self.sPair.spIAD_m]
        yerr=[0,self.sPair.spIAD_std]
        plt.bar(x, y, yerr=yerr, width=0.5,color='k')
        lims = plt.ylim()
        plt.ylim([0, lims[1]]) 
        plt.xlim([0.5, 3]) 
        plt.ylabel('[mm]')
        plt.xticks([1.25,2.25],['raw','shift'])
        plt.title('mean IAD')
        
        plt.subplot(4,4,9)
        plt.cla()
        a1=self.Pair.animals[0].neighborMat
        a2=self.Pair.animals[1].neighborMat
        
        meanPosMat=np.nanmean(np.stack([a1,a2],-1),axis=2)
        plt.imshow(meanPosMat,interpolation='none', extent=[-31,31,-31,31],clim=[0,100],origin='lower')
        plt.title('mean neighbor position')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        
        #neighborhood matrix for animal0
        #the vertial orientation was confirmed correct in march 2016
        plt.subplot(4,4,13)
        plt.cla()
        PosMat=self.Pair.animals[0].neighborMat
        plt.imshow(PosMat,interpolation='none', extent=[-31,31,-31,31],clim=[0,200],origin='lower')
        plt.title('mean neighbor position')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        
        plt.subplot(4,4,14)
        plt.cla()
        PosMat=self.Pair.animals[1].neighborMat
        plt.imshow(PosMat,interpolation='none', extent=[-31,31,-31,31],clim=[0,200],origin='lower')
        plt.title('mean neighbor position')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        
        #LEADERSHIP
        plt.subplot(4,4,15)
        plt.cla()
        self.LeadershipIndex=leadership(np.stack([a1,a2],-1))
        x=[1,2]
        barlist=plt.bar(x,self.LeadershipIndex, width=0.5,color='b')
        barlist[1].set_color('r')
        plt.title('Leadership')
        plt.ylabel('index')
        plt.ylim([-.5, .5])
        plt.xlim([0.5, 3])
        
        if np.sum(self.Pair.StackTopAnimal)>0:
            xtickList=[int(np.where(self.Pair.StackTopAnimal==1)[0])+1.25,int(np.where(self.Pair.StackTopAnimal==0)[0])+1.25]
            plt.xticks(xtickList,['top','bottom'])
        else:
            plt.xticks([1.25,2.25],['same','dish'])
            
        plt.subplot(4,4,16)
        plt.cla()
        x=[1,2]
        barlist=plt.bar(x,self.AnSize[:,1], width=0.5,color='b')
        barlist[1].set_color('r')
        plt.xlim([0.5, 3])
        plt.title('Body size')
        plt.ylabel('area [px]')
        


        plt.subplot(4,4,10)
        plt.title('accel=f(pos_n)')
        a1=self.Pair.animals[0].ForceMat
        a2=self.Pair.animals[1].ForceMat
        meanForceMat=np.nanmean(np.stack([a1,a2],-1),axis=2)
        johPlt.plotMapWithXYprojections(meanForceMat,3,outer_grid[9],31,0.01)
        
        plt.subplot(4,4,11)
        plt.cla()
        plt.bar([1,2],self.Pair.avgSpeed(),width=0.5)
        plt.title('avgSpeed')
   
        plt.tight_layout()
        plt.show()
        
        pdfPath=self.expInfo.aviPath+'.pdf'
        with PdfPages(pdfPath) as pdf:
            pdf.savefig()