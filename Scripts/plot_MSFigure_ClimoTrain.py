"""
Train ANN to predict the year for ToE in the CONUS

Reference   : Barnes et al. [2020, JAMES] & Labe and Barnes [2021, JAMES]
Author      : Zachary M. Labe
Date        : 6 April 2023
Environment : conda activate env-tf27
Tensorflow  : 2.7 (XAI for v2.0.1)
Version     : 5 (shuffle maps of SPEAR_MED)
"""

###############################################################################
###############################################################################
###############################################################################
### Import packages
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import calc_Utilities as UT
import sys
import scipy.stats as sts
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import cmocean
import cmasher

###############################################################################
###############################################################################
###############################################################################
### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')  
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
            ax.xaxis.set_ticks([]) 

letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]

###############################################################################
###############################################################################
###############################################################################
### Testing data for ANN --------------------------------------
resolution = 'MEDS'
datasetsingleall = ['SPEAR_MED']
datasetnamelist = ['SPEAR_MED_SSP585']
dataobsname = 'NClimGrid'
dataset_obs = 'NClimGrid_%s' % resolution
yearsall = [np.arange(1921,2100+1,1)]
monthlychoiceall = ['JJA']
monthlychoice = monthlychoiceall[0]
variqall = ['T2M']
variqname = ['TAVG']
reg_name = 'US'
segment_data_factorq = [0.8]
lat_bounds,lon_bounds = UT.regions(reg_name)

###############################################################################
###############################################################################
###############################################################################
### Select whether to standardize over a baseline
baselineSTD = True
yrminb = 1981
yrmaxb = 2010
baselineSTDyrs = np.arange(yrminb,yrmaxb+1,1)

###############################################################################
###############################################################################
###############################################################################
### Masking and preprocessing arguments
rm_annual_mean = False
rm_merid_mean = False
rm_ensemble_mean = False
land_only = False
ocean_only = False
CONUS_only = True

### Parameters
if reg_name == 'US':
    debug = True
    NNType = 'ANN'
    annType = 'class'
    classChunkHalf = 5
    classChunk = 10
    biasBool = True
    if resolution == 'MEDS':
        hiddensList = [[10,10,10]]
    elif resolution == 'LOWS':
        hiddensList = [[20,20,20]]
    elif resolution == 'HIGHS':
        hiddensList = [[20,20,20]]
    else:
        print(ValueError('This is the wrong resolution!!!'))
        sys.exit()
    ridge_penalty = [0.001]
    actFun = 'relu'
    iterations = [500]
elif any([reg_name=='W_US',reg_name=='Ce_US',reg_name=='E_US']):
    debug = True
    NNType = 'ANN'
    annType = 'class'
    classChunkHalf = 5
    classChunk = 10
    biasBool = True
    hiddensList = [[100,100]]
    ridge_penalty = [0.001]
    actFun = 'relu'
    iterations = [500]
else:
    print(ValueError('This is the wrong region name selected!!!'))
    sys.exit()
    
lr_here = .01
batch_size = 32 
random_network_seed = 87750
random_segment_seed = 71541

### Years
yearsmodel = yearsall[0]
yearsobs = np.arange(1921,2022+1,1)
 
###############################################################################
###############################################################################
###############################################################################
### Read in data
directorypredictions = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/Predictions/Regions/'
directorymodel = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/savedModels/'
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/MS_Figures/'
dataset = datasetsingleall[0]
variq = variqall[0]
modelType = 'TrainedOn' + dataset
savename = modelType+'_Vari-'+variq+'_Obs-' + dataset_obs + '_Region-' + reg_name + '_Months-' + monthlychoice + '_L2-'+ str(ridge_penalty[0])+ '_LR-' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(iterations[0]) + '_' + NNType + str(hiddensList[0][0]) + 'x' + str(hiddensList[0][-1]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
savenameModelTestTrain = modelType+'_'+variq+'_modelTrainTest_SegSeed'+str(random_segment_seed)+'_NetSeed'+str(random_network_seed)

trainIndices = np.genfromtxt(directorypredictions + savename + 'TrainIndices_ENSEMBLES.txt')
lenens = len(trainIndices)
    
parameters = np.load(directorymodel + savenameModelTestTrain + '.npz')
lats = parameters['lats'][:]
lons = parameters['lons'][:]
Xmean = parameters['Xmean'][:].reshape(lats.shape[0],lons.shape[0])
Xstd = parameters['Xstd'][:].reshape(lats.shape[0],lons.shape[0])

###############################################################################
###############################################################################
###############################################################################
### Plot climo figures
def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')  
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.xaxis.set_ticks([])
def setcolor(x, color):
      for m in x:
          for t in x[m][1]:
              t.set_color(color)

fig = plt.figure(figsize=(10,4))

label = r'\textbf{TAVG Mean [$^{\circ}$C]}'
limit = np.arange(10,30.01,0.1)
barlim = np.round(np.arange(10,31,5),2)

ax = plt.subplot(1,2,1)
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='h',area_thresh=1000)
m.drawcoastlines(color='k',linewidth=1)
m.drawstates(color='k',linewidth=0.5)
m.drawcountries(color='k',linewidth=1,zorder=12)

circle = m.drawmapboundary(fill_color='darkgrey',color='darkgrey',
                  linewidth=0.7)
circle.set_clip_on(False)
    
lon2,lat2 = np.meshgrid(lons,lats)

cs1 = m.pcolormesh(lon2,lat2,Xmean,vmin=10,vmax=30,latlon=True)
cs1.set_cmap('twilight')

ax.annotate(r'\textbf{Western}',xy=(0,0),xytext=(0.19,0.62),
          textcoords='axes fraction',color='gold',fontsize=14,
          rotation=350,ha='center',va='center')
ax.annotate(r'\textbf{Central}',xy=(0,0),xytext=(0.54,0.57),
          textcoords='axes fraction',color='gold',fontsize=14,
          rotation=0,ha='center',va='center')
ax.annotate(r'\textbf{Eastern}',xy=(0,0),xytext=(0.85,0.62),
          textcoords='axes fraction',color='gold',fontsize=14,
          rotation=10,ha='center',va='center')

ax.annotate(r'\textbf{W.}',xy=(0,0),xytext=(0.325,0.55),
          textcoords='axes fraction',color='gold',fontsize=10,
          rotation=353,ha='center',va='center')
ax.annotate(r'\textbf{CO}',xy=(0,0),xytext=(0.3235,0.51),
          textcoords='axes fraction',color='gold',fontsize=10,
          rotation=353,ha='center',va='center')

ax.annotate(r'\textbf{[%s]}' % (letters[0]),xy=(0,0),xytext=(0.0,1.05),
          textcoords='axes fraction',color='k',fontsize=9,
          rotation=0,ha='center',va='center')

lat_bounds,lon_bounds = UT.regions('W_US')
la1 = lat_bounds[0]+9.67
la2 = lat_bounds[1]-6.1
lo1 = lon_bounds[0]
lo2 = lon_bounds[1]
m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=4,color='gold',zorder=20,clip_on=False)
lat_bounds,lon_bounds = UT.regions('Ce_US')
la1 = lat_bounds[0]+9.7
la2 = lat_bounds[1]-8.2
lo1 = lon_bounds[0]
lo2 = lon_bounds[1]
m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=4,color='gold',zorder=20,clip_on=False)

### CO box
la1 = 37
la2 = 41
lo1 = 360-108
lo2 = 360-105
lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
latsslice = np.ones(len(lonsslice))*la2
m.plot(lonsslice, latsslice, color='gold', linewidth=1.5, latlon=True,zorder=4)
latsslice = np.ones(len(lonsslice))*la1
m.plot(lonsslice, latsslice, color='gold', linewidth=1.5, latlon=True,zorder=4)
m.drawgreatcircle(lo1, la1, lo1, la2,linewidth=1.5,color='gold',zorder=4)
m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=1.5,color='gold',zorder=4)

parallels = np.arange(-90,91,5)
meridians = np.arange(-180,180,15)
par=m.drawparallels(parallels,labels=[True,False,False,True],linewidth=0.5,
                color='w',fontsize=6,zorder=40)
mer=m.drawmeridians(meridians,labels=[False,False,True,False],linewidth=0.5,
                    fontsize=6,color='w',zorder=40)
setcolor(mer,'dimgrey')
setcolor(par,'dimgrey')

cbar_ax1 = fig.add_axes([0.16,0.08,0.2,0.025])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=10,color='k',labelpad=3)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=7)
cbar1.outline.set_edgecolor('dimgrey')

###############################################################################
label = r'\textbf{TAVG Std. Dev. [$^{\circ}$C]}'
limit = np.arange(0,2.01,0.05)
barlim = np.round(np.arange(0,3,1),2)

ax = plt.subplot(1,2,2)
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='h',area_thresh=1000)
m.drawcoastlines(color='k',linewidth=1)
m.drawstates(color='k',linewidth=0.5)
m.drawcountries(color='k',linewidth=1,zorder=12)

circle = m.drawmapboundary(fill_color='darkgrey',color='darkgrey',
                  linewidth=0.7)
circle.set_clip_on(False)
    
lon2,lat2 = np.meshgrid(lons,lats)

cs1 = m.pcolormesh(lon2,lat2,Xstd,vmin=0,vmax=2,latlon=True)
cs1.set_cmap('bone')

ax.annotate(r'\textbf{[%s]}' % (letters[1]),xy=(0,0),xytext=(0.0,1.05),
          textcoords='axes fraction',color='k',fontsize=9,
          rotation=0,ha='center',va='center')
ax.annotate(r'\textbf{Western}',xy=(0,0),xytext=(0.19,0.62),
          textcoords='axes fraction',color='gold',fontsize=14,
          rotation=350,ha='center',va='center')
ax.annotate(r'\textbf{Central}',xy=(0,0),xytext=(0.54,0.57),
          textcoords='axes fraction',color='gold',fontsize=14,
          rotation=0,ha='center',va='center')
ax.annotate(r'\textbf{Eastern}',xy=(0,0),xytext=(0.85,0.62),
          textcoords='axes fraction',color='gold',fontsize=14,
          rotation=10,ha='center',va='center')

ax.annotate(r'\textbf{W.}',xy=(0,0),xytext=(0.325,0.55),
          textcoords='axes fraction',color='gold',fontsize=10,
          rotation=353,ha='center',va='center')
ax.annotate(r'\textbf{CO}',xy=(0,0),xytext=(0.3235,0.51),
          textcoords='axes fraction',color='gold',fontsize=10,
          rotation=353,ha='center',va='center')

lat_bounds,lon_bounds = UT.regions('W_US')
la1 = lat_bounds[0]+9.67
la2 = lat_bounds[1]-6.1
lo1 = lon_bounds[0]
lo2 = lon_bounds[1]
m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=4,color='gold',zorder=20,clip_on=False)
lat_bounds,lon_bounds = UT.regions('Ce_US')
la1 = lat_bounds[0]+9.7
la2 = lat_bounds[1]-8.2
lo1 = lon_bounds[0]
lo2 = lon_bounds[1]
m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=4,color='gold',zorder=20,clip_on=False)

### CO box
la1 = 37
la2 = 41
lo1 = 360-108
lo2 = 360-105
lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
latsslice = np.ones(len(lonsslice))*la2
m.plot(lonsslice, latsslice, color='gold', linewidth=1.5, latlon=True,zorder=4)
latsslice = np.ones(len(lonsslice))*la1
m.plot(lonsslice, latsslice, color='gold', linewidth=1.5, latlon=True,zorder=4)
m.drawgreatcircle(lo1, la1, lo1, la2,linewidth=1.5,color='gold',zorder=4)
m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=1.5,color='gold',zorder=4)

parallels = np.arange(-90,91,5)
meridians = np.arange(-180,180,15)
par=m.drawparallels(parallels,labels=[False,True,False,False],linewidth=0.5,
                color='w',fontsize=6,zorder=40)
mer=m.drawmeridians(meridians,labels=[False,False,True,False],linewidth=0.5,
                    fontsize=6,color='w',zorder=40)
setcolor(mer,'dimgrey')
setcolor(par,'dimgrey')
    
cbar_ax1 = fig.add_axes([0.64,0.08,0.2,0.025])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=10,color='k',labelpad=3)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=7)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
        
plt.savefig(directoryfigure + 'TrainingClimo_Map.png',dpi=600)
