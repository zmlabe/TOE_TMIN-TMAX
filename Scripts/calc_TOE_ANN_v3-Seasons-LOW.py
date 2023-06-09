"""
Calculate TOE for observations using the predict the year approach for LOW

Author     : Zachary M. Labe
Date       : 13 February 2023
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
directorydata = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/Predictions/LoopFinal/Seasons/'
variqq = ['T2M']
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/Figures/TOE_ANN/Seasons/'
land_only = False
ocean_only = False
rm_annual_mean = False
rm_merid_mean = False
rm_ensemble_mean = False
monthlychoiceq = ['DJF','MAM','JJA','SON','annual']
reg_name = 'E_US'
NNType = 'ANN'
dataset = 'SPEAR_LOW'
resolution = 'LOWS'
dataset_obs = 'NClimGrid_LOWS'
SAMPLES = 100

if reg_name == 'US':
    debug = True
    NNType = 'ANN'
    annType = 'class'
    classChunkHalf = 5
    classChunk = 10
    biasBool = False
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
    biasBool = False
    hiddensList = [[100,100]]
    ridge_penalty = [0.001]
    actFun = 'relu'
    iterations = [500]
else:
    print(ValueError('This is the wrong region name selected!!!'))
    sys.exit()

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
            
        ypredq = np.genfromtxt(directorydata + 'ANN_PredictTheYear_v3-LoopFinal-ObsPred_%s_%s_%s-%s_%s_SAMPLES-%s.txt' % (variq,monthlychoice,dataset,dataset_obs,reg_name,SAMPLES),unpack=True)
        
        toeN = []
        corrN = []
        for ss in range(SAMPLES):
            if monthlychoice == 'DJF':
                yactual = np.arange(1921,2021+1,1)
            else:
                yactual = np.arange(1921,2022+1,1)
            years = yactual
            ypred = ypredq[:,ss]
            
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
            
            ### Calculate skill
            corr = sts.spearmanr(yactual,ypred)[0]
            
            toeN.append(actual_TOE)
            corrN.append(corr)
        corrm.append(corrN)
        toem.append(toeN)
    corrall.append(corrm)
    toeall.append(toem)
    
### Sort the TOE calculations
toe_annall = np.asarray(toeall).squeeze()
cor_annall = np.asarray(corrall).squeeze()

toe_plot = toe_annall.copy()
toe_plot[np.where(np.isnan(toe_plot))] = 2022

### Save TOE calculations
directorytoe = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/Predictions/ToeFinal/'
np.savez(directorytoe + 'TOE_v3-LoopFinal-ObsPred_Seasons_%s-%s_%s_SAMPLES-%s.npz' % (dataset,dataset_obs,reg_name,SAMPLES),
         toe_annall=toe_annall,toe_plot=toe_plot)
np.savez(directorytoe + 'SpearmanCorr_v3-LoopFinal-ObsPred_Seasons_%s-%s_%s_SAMPLES-%s.npz' % (dataset,dataset_obs,reg_name,SAMPLES),
         cor_annall=cor_annall)
                
