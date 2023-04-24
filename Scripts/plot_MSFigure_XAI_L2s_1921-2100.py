"""
LRP for SPEAR epochs to see if the regions change for 20th century

Author     : Zachary M. Labe
Date       : 30 January 2023
Version    : 2 (positive and negative relevance values)
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
from scipy.ndimage import gaussian_filter

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

### Parameters
directorydata = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/LRP/Regions/'
years = np.arange(1921,2100+1,1)
variq = 'T2M'
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/MS_Figures/' 
land_only = False
ocean_only = False
rm_annual_mean = False
rm_merid_mean = False
rm_ensemble_mean = False
random_network_seed = 87750
random_segment_seed = 71541
ridgePenaltyall = [0.001,0.01,0.1,0.5,1,5]
monthlychoiceq = ['JJA']
reg_name = 'US'
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
testingn = 2 # ensemble members used for testing

### Smooth LRP fields
sigmafactor = 1.5

fig = plt.figure(figsize=(10,5.5))
for rr in range(len(ridgePenaltyall)):
    ridge_penalty = [ridgePenaltyall[rr]]
    
    monthlychoice = monthlychoiceq[0]
    ###############################################################################
    ### Read in LRP after training on SPEAR
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
    
    data = Dataset(directorydata + 'LRPMap_Z-neg_Testing' + '_' + variq + '_' + savename + '.nc')
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]                             
    lrp_SPEAR_zq = data.variables['LRP'][:].reshape(testingn,years.shape[0],lat.shape[0],lon.shape[0])
    data.close()
    
    ### Calculate ensemble mean 
    lrp_SPEAR_z = np.nanmean(lrp_SPEAR_zq,axis=0)
    
    ### Take means across all years
    lrp_SPEAR_zall = np.nanmean(lrp_SPEAR_z[:,:,:],axis=0)
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Plot subplot of observations
    letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
    limit = np.arange(-0.6,0.601,0.001)
    barlim = np.round(np.arange(-0.6,0.601,0.2),2)
    cmap = cmr.fusion_r
    label = r'\textbf{LRP$_{z}$ Rule [Relevance] -- SPEAR_MED}'
    
    ###############################################################################
    ax1 = plt.subplot(2,3,rr+1)
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='l')
    m.drawcoastlines(color='dimgrey',linewidth=1)
    m.drawstates(color='dimgrey',linewidth=0.5)
    m.drawcountries(color='dimgrey',linewidth=0.5)
        
    ### Variable
    lrp_SPEAR_z1 = lrp_SPEAR_zall/np.nanmax(abs(lrp_SPEAR_zall))
    
    x, y = np.meshgrid(lon,lat)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)
    
    cs1 = m.contourf(x,y,lrp_SPEAR_z1,limit,extend='both',latlon=True)
    cs1.set_cmap(cmap) 
       
    if rr == 0:
        plt.title(r'\textbf{L$_{2}$ = %s' % ridge_penalty[0],fontsize=14,color='k')
    else:
        plt.title(r'\textbf{L$_{2}$ = %s' % ridge_penalty[0],fontsize=14,color='dimgrey')
    ax1.annotate(r'\textbf{[%s]}' % letters[rr],xy=(0,0),xytext=(0.98,1.05),
                  textcoords='axes fraction',color='k',fontsize=9,
                  rotation=0,ha='center',va='center')

###############################################################################
cbar_ax1 = fig.add_axes([0.40,0.095,0.2,0.025])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=10,color='dimgrey',labelpad=1.8)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=7)
cbar1.outline.set_edgecolor('dimgrey')
    
plt.tight_layout()
plt.subplots_adjust(hspace=0,bottom=0.14,wspace=0.2)

plt.savefig(directoryfigure + 'PredictTheYear_LRPcomparison-%s_forL2s_%s_%s_%s.png' % (dataset,variq,monthlychoice,reg_name),dpi=600)
