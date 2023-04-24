"""
Calculate albedo for SPEAR_MED and find relationship with LRP

Author    : Zachary M. Labe
Date      : 23 January 2023
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import numpy as np
import cmocean
import cmasher as cmr
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import sys
import scipy.stats as sts
from scipy.ndimage import gaussian_filter

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

### Parameters
monthq = ['JAN','FEB','MAR','ARP','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
variq = 'TMAX'
variscatter = 'ALBEDO'
slicenan = 'nan'
reg_name = 'E_US'
lat_bounds,lon_bounds = UT.regions(reg_name)
monthlychoiceq = ['JJA']
slicemonthnamen = ['JUN-AUG']
scenario = 'SSP585'

### Select which trend to compare
timeperiodq = ['historical']

### Masking and preprocessing arguments
rm_annual_mean = False
rm_merid_mean = False
rm_ensemble_mean = False
land_only = False
ocean_only = False
CONUS_only = True

### LRP Parameters
directorypredictions = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/Predictions/Regions/'
directorydata = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/LRP/Regions/'
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/Figures/LRP/Scatter/Albedo/'
random_network_seed = 87750
random_segment_seed = 71541
ridge_penalty = [0.001]
classChunk = 10
lr_here = 0.01
batch_size = 32
iterations = [500]
NNType = 'ANN'
dataset = 'SPEAR_MED'
resolution = 'MEDS'
dataset_obs = 'NClimGrid_MEDS'
if reg_name == 'US':
    if resolution == 'MEDS':
        hiddensList = [[10,10,10]]
    elif resolution == 'LOWS':
        hiddensList = [[20,20,20]]
    elif resolution == 'HIGHS':
        hiddensList = [[20,20,20]]
elif any([reg_name=='W_US',reg_name=='Ce_US',reg_name=='E_US']):
    hiddensList = [[100,100]]
    ridge_penalty = [0.001]
testingn = 2 # ensemble members used for testing

### Smooth LRP fields
sigmafactor = 1.5

### Calculate linear trends
def calcTrend(data):
    if data.ndim == 3:
        slopes = np.empty((data.shape[1],data.shape[2]))
        x = np.arange(data.shape[0])
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                mask = np.isfinite(data[:,i,j])
                y = data[:,i,j]
                
                if np.sum(mask) == y.shape[0]:
                    xx = x
                    yy = y
                else:
                    xx = x[mask]
                    yy = y[mask]      
                if np.isfinite(np.nanmean(yy)):
                    slopes[i,j],intercepts, \
                    r_value,p_value,std_err = sts.linregress(xx,yy)
                else:
                    slopes[i,j] = np.nan
    elif data.ndim == 4:
        slopes = np.empty((data.shape[0],data.shape[2],data.shape[3]))
        x = np.arange(data.shape[1])
        for ens in range(data.shape[0]):
            for i in range(data.shape[2]):
                for j in range(data.shape[3]):
                    mask = np.isfinite(data[ens,:,i,j])
                    y = data[ens,:,i,j]
                    
                    if np.sum(mask) == y.shape[0]:
                        xx = x
                        yy = y
                    else:
                        xx = x[mask]
                        yy = y[mask]      
                    if np.isfinite(np.nanmean(yy)):
                        slopes[ens,i,j],intercepts, \
                        r_value,p_value,std_err = sts.linregress(xx,yy)
                    else:
                        slopes[ens,i,j] = np.nan
    
    dectrend = slopes * 10.   
    print('Completed: Finished calculating trends!')      
    return dectrend

###############################################################################
###############################################################################
###############################################################################
### Read in climate models      
def read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice,scenario)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons  


for m in range(len(monthlychoiceq)):
    monthlychoice = monthlychoiceq[m]
    timeperiod = timeperiodq[0]

    ### Years for each simulation
    yearmax = 2020
    years = np.arange(1921,yearmax+1,1)
    years_med = np.arange(1921,2100+1,1)
    years_noaer = np.arange(1921,2020+1,1)
    
    ### Read in test indices
    modelType = 'TrainedOn%s' % dataset
    savename = modelType+'_Vari-'+variq+'_Obs-' + dataset_obs + '_Region-' + reg_name + '_Months-' + monthlychoice + '_L2-'+ str(ridge_penalty[0])+ '_LR-' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(iterations[0]) + '_' + NNType + str(hiddensList[0][0]) + 'x' + str(hiddensList[0][-1]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed)
    testIndices = np.genfromtxt(directorypredictions + savename + 'TestIndices_ENSEMBLES.txt',dtype=np.int32)

    ### Get data
    lat_bounds,lon_bounds = UT.regions(reg_name)
    spearn,lats,lons = read_primary_dataset(variscatter,'SPEAR_MED',monthlychoice,scenario,lat_bounds,lon_bounds)
    lon2,lat2 = np.meshgrid(lons,lats)
    
    ### Mask out CONUS
    data_obs = np.full(spearn[0].shape,np.nan) # just for function to work, ha!
    spearn, data_obs = dSS.mask_CONUS(spearn,data_obs,resolution,lat_bounds,lon_bounds)
    
    ### Select same years
    yearq_med = np.where((years_med >= years.min()) & (years_med <= years.max()))[0]
    
    spear = spearn[:,yearq_med,:,:]
    
    ### Calculate ensemble means
    mean_spear = np.nanmean(spear,axis=0)
    
    ### Calculate trends 
    spear_trendq = calcTrend(spear)
    spear_trend = np.nanmean(spear_trendq[testIndices,:,:],axis=0)
    
    runs = [spear_trend]
    modelname = ['SPEAR_MED']
    modelnameyears = ['1921-2020']
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Read in LRP 
    data = Dataset(directorydata + 'LRPMap_Z-neg_Testing' + '_' + variq + '_' + savename + '.nc')
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]                             
    lrp_SPEAR_zq = data.variables['LRP'][:].reshape(testingn,years_med.shape[0],lat.shape[0],lon.shape[0])
    data.close()
    
    ### Calculate ensemble mean 
    lrp_SPEAR_z = np.nanmean(lrp_SPEAR_zq,axis=0)
    lrp_SPEAR_zmean = np.nanmean(lrp_SPEAR_z,axis=0)
    
    ### Take means across all years
    yeartrendlrpq = years.max() + 1 # index
    timem = np.where(years_med == yeartrendlrpq)[0][0]
    lrp_SPEAR_z1 = np.nanmean(lrp_SPEAR_z[:timem,:,:],axis=0)
    lrp_SPEAR_z2 = np.nanmean(lrp_SPEAR_z[timem:,:,:],axis=0)
    
    # mask = lrp_SPEAR_z1.copy()
    # mask[np.where(mask!=0)] = 1.
    # lrp_SPEAR_z1 = gaussian_filter(lrp_SPEAR_z1,sigma=sigmafactor,order=0)*mask
        
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### LRP scatter
    lrp_SPEAR_z1 = lrp_SPEAR_z1
    lrp_scatter = lrp_SPEAR_z1.ravel()
    trend_scatter = spear_trend.ravel()
    
    # sigq = 0.001
    # significance = np.where(lrp_scatter <= sigq)[0]
    # lrp_scatter[significance] = np.nan
    
    mask = ~np.isnan(trend_scatter) & ~np.isnan(lrp_scatter)
    slope, intercept, r_value, p_value, std_err = sts.linregress(trend_scatter[mask],lrp_scatter[mask])
    
    ###############################################################################
    ###############################################################################
    ###############################################################################               
    ### Plot Figure
    ### Adjust axes in time series plots 
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
            
    fig = plt.figure()
    ax = plt.subplot(111)
    
    adjust_spines(ax, ['left', 'bottom'])            
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('dimgrey')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both',length=4.,width=2,which='major',color='dimgrey')
    ax.tick_params(axis='x',labelsize=6,pad=1.5)
    ax.tick_params(axis='y',labelsize=6,pad=1.5)
    
    plt.scatter(trend_scatter,lrp_scatter,marker='o',s=30,color='maroon',
                alpha=0.4,edgecolors='maroon',linewidth=0,clip_on=False)
    
    plt.xticks(np.arange(-0.5,0.6,0.1),map(str,np.round(np.arange(-0.5,0.6,0.1),2)),fontsize=10)
    plt.yticks(np.arange(-1,1,0.1),map(str,np.round(np.arange(-1,1,0.1),2)),fontsize=10)
    plt.xlim([-0.5,0.5])
    plt.ylim([-0.4,0.4])
    
    plt.title('XAI for %s over %s using %s' % (monthlychoice,reg_name,variq),color='k',fontsize=15)
    
    plt.xlabel(r'\textbf{Albedo Trend for SPEAR-MED}',fontsize=11,color='dimgrey')
    plt.ylabel(r'\textbf{ANN Relevance for SPEAR-MED}',fontsize=11,color='dimgrey')
    
    plt.tight_layout()
    plt.savefig(directoryfigure + savename + '_SCATTER_LRP-AlbedoTrend_v1.png',dpi=300)
