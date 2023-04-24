"""
Calculate TOE for observations using the predict the year approach

Author     : Zachary M. Labe
Date       : 26 September 2022
Version    : 1
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import cmocean as cmocean
import cmasher as cmr
import calc_Utilities as UT
import calc_dataFunctions as df
import scipy.stats as sts

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

### Parameters
directorydata = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/Predictions/Regions/'
dataset_obs = 'NClimGrid_LOWS'
years = np.arange(1921,2021+1,1)
variqq = ['T2M','TMAX','TMIN']
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/Figures/TOE_ANN/'
land_only = False
ocean_only = False
rm_annual_mean = False
rm_merid_mean = False
rm_ensemble_mean = False
random_network_seed = 87750
random_segment_seed = 71541
hiddensList = [[20,20]]
ridge_penalty = [0.01]
monthlychoiceq = ['JJA']
reg_name = 'US'
classChunk = 10
lr_here = 0.01
batch_size = 32
iterations = [500]
NNType = 'ANN'
dataset = 'LENS2_LOWS'

def read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice,scenario)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons  

toeall = []
corrall = []
for vv in range(len(variqq)):
    variq = variqq[vv]
    
    toem = []
    corrm = []
    for mm in range(len(monthlychoiceq)):
        monthlychoice = monthlychoiceq[mm]
        ###############################################################################
        ### Read in LRP after training on SPEAR_MED
        modelType = 'TrainedOn%s' % dataset
        savename = modelType+'_Vari-'+variq+'_Obs-' + dataset_obs + '_Region-' + reg_name + '_Months-' + monthlychoice + '_L2-'+ str(ridge_penalty[0])+ '_LR-' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(iterations[0]) + '_' + NNType + str(hiddensList[0][0]) + 'x' + str(hiddensList[0][-1]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed)
        if rm_annual_mean == True:
            savename = savename + '_AnnualMeanRemoved' 
        if rm_ensemble_mean == True:
            savename = savename + '_EnsembleMeanRemoved' 
        if rm_merid_mean == True:
            savename = savename + '_MeridionalMeanRemoved' 
        if land_only == True: 
            savename = savename + '_LANDONLY'
        if ocean_only == True:
            savename = savename + '_OCEANONLY'
            
        ypred = np.genfromtxt(directorydata + savename + 'OBSERVATION_PREDICTIONS.txt',unpack=True)
        yactual = years
        
        ### Calculate baseline
        yrmin = 1921
        yrmax = 1950
        yearq = np.where((years >= 1921) & (years <= 1950))[0]
        baseline = ypred[yearq]
        baseMAX = np.max(baseline)
        
        ### Calculate ToE
        ToE = np.array([np.nan])
        for yr in range(len(ypred)):
            currentpred = ypred[yr]
            if currentpred > baseMAX:
                if np.min(ypred[yr:]) > baseMAX:
                    if np.isnan(ToE):
                        ToE[:] = yr
                  
        if np.isnan(ToE):
            actual_TOE = np.nan
        else:
            actual_TOE = years[int(ToE[0])]
        
        corr = sts.spearmanr(yactual,ypred)[0]
        
        corrm.append(corr)
        toem.append(actual_TOE)
    corrall.append(corrm)
    toeall.append(toem)
    
print(variqq)
print(toeall)
print(corrall)
    
### Make sample plot
plt.figure()
plt.scatter(years,ypred)
plt.scatter(years[int(ToE[0])],ypred[int(ToE[0])],color='k')
plt.scatter(years[np.argmax(baseline)],ypred[np.argmax(baseline)],color='r')
plt.axvline(x=actual_TOE,color='k')
plt.axhline(y=baseMAX,color='r')
plt.title(r'%s - %s' % (monthlychoice,variq))
plt.xlabel(r'Actual Years')
plt.ylabel(r'Predicted Years')
plt.savefig(directoryfigure + 'SAMPLE_TOE.png',dpi=300)
    
toeready = np.asarray(toeall)
corrready = np.asarray(corrall)
                
