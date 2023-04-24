"""
Compare regional mean temperature anomalies for different SPEAR experiments
 
Author    : Zachary M. Labe
Date      : 2 September 2022
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import numpy as np
import cmocean
import calc_Utilities as UT
import sys
import itertools
import read_NClimGrid_monthlyMEDS as NC
import read_SPEAR_MED as SP
import read_SPEAR_MED_NATURAL as NAT
import scipy.stats as sts
import calc_dataFunctions as df
import calc_Stats as dSS

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/MS_Figures/' 

### Parameters
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
slicemonthn = ['JJA']
slicemonthnamen = ['JUN-AUG']
slicenan = 'nan'
years = np.arange(1921,2022+1,1)
yearsmodels = np.arange(1921,2100+1,1)

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

def read_primary_dataset(variq,datasetname,monthlychoice,scenario,lat_bounds,lon_bounds):
    
    if any([datasetname == 'SPEAR_MED_shuffle_space',datasetname == 'SPEAR_MED_shuffle_time']):
        dataset_pick = 'SPEAR_MED'
    else:
        dataset_pick = datasetname
    
    data,lats,lons = df.readFiles(variq,dataset_pick,monthlychoice,scenario)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',datasetname,' is shaped',data.shape)
    return datar,lats,lons

def readData(dataset_obs,resolution,monthlychoice,reg_name,variq,scenario,yearsd):
    
    if resolution == 'LOWS':
        dataset = 'SPEAR_LOW'
    elif resolution == 'MEDS':
        dataset = 'SPEAR_MED'
        
    lat_bounds,lon_bounds = UT.regions(reg_name)
    data_all,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds)
    data_obs_all,lats_obs,lons_obs = read_primary_dataset(variq,dataset_obs,monthlychoice,scenario,lat_bounds,lon_bounds)
    
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
        data, data_obs = dSS.mask_CONUS(data,data_obs,resolution,lat_bounds,lon_bounds)
        
    ### Calculate anomalies
    yearq = np.where((yearsd >= 1981) & (yearsd <= 2010))[0]
    print(yearsd[yearq])
    climobs = np.nanmean(data_obs[yearq,:,:],axis=0) 
    obsanom = np.asarray(data_obs - climobs)
        
    return obsanom,lats_obs,lons_obs

yearsall = [np.arange(1921,2015+1,1),np.arange(1940,2022+1,1),np.arange(1921,2022+1,1)]
ct,lat20,lon20 = readData('20CRv3_LOWS','LOWS','JJA','US','T2M','SSP585',yearsall[0])
era,latera,lonera = readData('ERA5_MEDS','MEDS','JJA','US','T2M','SSP585',yearsall[1])
nc,latnc,lonnc = readData('NClimGrid_MEDS','MEDS','JJA','US','T2M','SSP585',yearsall[2])

lon20_2,lat20_2 = np.meshgrid(lon20,lat20)
lonera_2,latera_2 = np.meshgrid(lonera,latera)
lonnc_2,latnc_2 = np.meshgrid(lonnc,latnc)

### Means
ave_c = UT.calc_weightedAve(ct,lat20_2)
ave_e = UT.calc_weightedAve(era,latera_2)
ave_n = UT.calc_weightedAve(nc,latnc_2)
  
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
ax.yaxis.grid(color='darkgrey',linestyle='-',linewidth=0.5,clip_on=False,alpha=0.4)

plt.plot(yearsall[0],ave_c,linestyle='--',linewidth=1.5,color='teal',
         clip_on=False,zorder=30,label=r'\textbf{20CRv3}',dashes=(1,0.3))
plt.plot(yearsall[2],ave_n,linestyle='-',linewidth=3,color='k',
         clip_on=False,zorder=30,label=r'\textbf{NClimGrid}')
plt.plot(yearsall[1],ave_e,linestyle='-',linewidth=1.5,color='maroon',
         clip_on=False,zorder=30,label=r'\textbf{ERA5}')

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10),fontsize=11)
plt.yticks(np.round(np.arange(-18,18.1,0.5),2),np.round(np.arange(-18,18.1,0.5),2),fontsize=11)
plt.xlim([1920,2022])
plt.ylim([-2,2])

plt.ylabel(r'\textbf{Baseline of 1981-2010 [$\bf{^\circ}$C]}',fontsize=11,
                      color='k')

leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
      bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=3,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.tight_layout()        

### Save figure
plt.savefig(directoryfigure+'JJA_T-CONUS_1921-2022-ObsComparioson.png',dpi=600)
