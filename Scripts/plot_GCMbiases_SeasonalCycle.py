"""
Plot biases for SPEAR_MED over CONUS

Author    : Zachary M. Labe
Date      : 14 August 2022
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import numpy as np
import cmocean
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import sys
import scipy.stats as sts

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/Figures/Bias/' 

### Parameters
monthq = ['JAN','FEB','MAR','ARP','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
variq = 'TMIN'
slicenan = 'nan'
reg_name = 'US'
lat_bounds,lon_bounds = UT.regions(reg_name)
monthlychoice = 'none'

dataset = 'SPEAR_MED'
dataset_obs = 'NClimGrid_MEDS'

### Select years of data
if dataset == 'SPEAR_MED':
    scenario = 'SSP585'
    years = np.arange(1921,2100+1,1)
elif dataset == 'SPEAR_MED_NATURAL':
    scenario = 'NATURAL'
    years = np.arange(1921,2100+1,1)
elif dataset == 'SPEAR_MED_Scenario':
    scenario = 'SSP245'
    years = np.arange(1921,2100+1,1)

if any([dataset_obs == 'NClimGrid_MEDS']):
    yearsobs = np.arange(1921,2021+1,1)
    timeexperi = ['1921-1971','1972-2021']
elif dataset_obs == 'ERA5_MEDS':
    yearsobs = np.arange(1979,2021+1,1)
    timeexperi = ['1979-2000','2001-2021']

### Masking and preprocessing arguments
rm_annual_mean = False
rm_merid_mean = False
rm_ensemble_mean = False
land_only = False
ocean_only = False
CONUS_only = True

###############################################################################
###############################################################################
###############################################################################
### Read in climate models      
def read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice,scenario)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons  

### Get data
lat_bounds,lon_bounds = UT.regions(reg_name)
data_all,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds)
data_obs_all,lats_obs,lons_obs = read_primary_dataset(variq,dataset_obs,monthlychoice,scenario,lat_bounds,lon_bounds)

### Prepare data for preprocessing
data, data_obs, = data_all, data_obs_all,
if rm_annual_mean == True:        
    data, data_obs = dSS.remove_annual_mean(data,data_obs,lats,lons,lats_obs,lons_obs)
    print('*Removed annual mean*')
if rm_merid_mean == True:
    data, data_obs = dSS.remove_merid_mean(data,data_obs,lats,lons,lats_obs,lons_obs)
    print('*Removed meridian mean*')  
if land_only == True:
    data, data_obs = dSS.remove_ocean(data,data_obs,lat_bounds,lon_bounds) 
    print('*Removed ocean*')
if ocean_only == True:
    data, data_obs = dSS.remove_land(data,data_obs,lat_bounds,lon_bounds) 
    print('*Removed land*')
if CONUS_only == True:
    data, data_obs = dSS.mask_CONUS(data,data_obs,'MEDS',lat_bounds,lon_bounds)
    print('*Removed everything by CONUS*')

### Select years
if any([dataset_obs == 'NClimGrid_MEDS']):
    yearsq = np.where((years >= 1921) & (years <= 2021))[0]
    model = data[:,yearsq,:,:,:]
elif dataset_obs == 'ERA5_MEDS':
    yearsq = np.where((years >= 1979) & (years <= 2021))[0]
    model = data[:,yearsq,:,:,:]
    
### Calculate CONUS average
lon2,lat2 = np.meshgrid(lons,lats)
ave_model = UT.calc_weightedAve(model,lat2)
ave_obs = UT.calc_weightedAve(data_obs,lat2)

### Calculate yearly means
obs_mean = np.nanmean(ave_obs,axis=0)

### Calculate yearly and ensemble means
model_mean = np.nanmean(ave_model,axis=(0,1))

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
        
fig = plt.figure(figsize=(9,5))
ax = plt.subplot(121)

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

plt.plot(obs_mean,linestyle='-',linewidth=2,marker='o',markersize=5,
         color='maroon',label=r'\textbf{NClimGrid}',clip_on=False)
plt.plot(model_mean,linestyle='-',linewidth=2,marker='o',markersize=5,
         color='teal',label=r'\textbf{SPEAR_MED}',clip_on=False)
         

leg = plt.legend(shadow=False,fontsize=8,loc='upper center',
      bbox_to_anchor=(0.5,0.06),fancybox=True,ncol=4,frameon=False,
      handlelength=1,handletextpad=0.5)

plt.xticks(np.arange(0,12,1),monthq)
plt.yticks(np.round(np.arange(-40,42,2),2),np.round(np.arange(-40,42,2),2))
plt.xlim([-0.5,11.5])

if variq == 'T2M':
    plt.ylim([-2,26])
elif variq == 'TMAX':
    plt.ylim([2,34])
elif variq == 'TMIN':
    plt.ylim([-10,18])

plt.ylabel(r'\textbf{%s [$\bf{^\circ}$C]}' % variq,fontsize=11,
                      color='dimgrey')
plt.title(r'\textbf{MEAN CONUS SEASONAL CYCLE -- 1921-2021}',
                    color='k',fontsize=12)

###############################################################################
###############################################################################
###############################################################################  
diff = model_mean - obs_mean 

ax = plt.subplot(122)
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

recdiff_masked = np.ma.masked_less_equal(diff,0)
plt.bar(np.arange(len(diff)),diff,color='deepskyblue',
        edgecolor='deepskyblue',zorder=9)
plt.bar(np.arange(len(recdiff_masked)),recdiff_masked,color='crimson',
        edgecolor='crimson',zorder=9)

plt.xticks(np.arange(0,12,1),monthq)
plt.yticks(np.round(np.arange(-40,42,1),2),np.round(np.arange(-40,42,1),2))
plt.xlim([-0.5,11.5])

plt.ylim([-4,4])

plt.ylabel(r'\textbf{%s [$\bf{^\circ}$C]}' % variq,fontsize=11,
                      color='dimgrey')
plt.title(r'\textbf{SPEAR_MED minus NClimGrid -- 1921-2021}',
                    color='k',fontsize=12)
plt.tight_layout()        

### Save figure
plt.savefig(directoryfigure+'SeasonalCycleDifferences_CONUS_%s.png' % variq,dpi=300)
