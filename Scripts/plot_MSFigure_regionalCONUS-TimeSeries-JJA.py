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

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/MS_Figures/' 

### Parameters
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
slicemonthn = ['JJA']
monthlychoice = slicemonthn[0]
slicemonthnamen = ['JUN-AUG']
slicenan = 'nan'
scenario = 'SSP585'
CONUS_only = True
resolution = 'MEDS'
import calc_Stats as dSS
years = np.arange(1921,2022+1,1)
yearsmodels = np.arange(1921,2100+1,1)

def read_primary_dataset(variq,datasetname,monthlychoice,scenario,lat_bounds,lon_bounds):
    
    if any([datasetname == 'SPEAR_MED_shuffle_space',datasetname == 'SPEAR_MED_shuffle_time']):
        dataset_pick = 'SPEAR_MED'
    else:
        dataset_pick = datasetname
    
    data,lats,lons = df.readFiles(variq,dataset_pick,monthlychoice,scenario)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',datasetname,' is shaped',data.shape)
    return datar,lats,lons 

def readData(variq,reg_name,monthlychoice):
    lat_bounds,lon_bounds = UT.regions(reg_name)
    data_all,lats,lons = read_primary_dataset(variq,'SPEAR_MED',monthlychoice,scenario,lat_bounds,lon_bounds)
    data_obs_all,lats_obs,lons_obs = read_primary_dataset(variq,'NClimGrid_MEDS',monthlychoice,scenario,lat_bounds,lon_bounds)
    
    data, data_obs, = data_all, data_obs_all
    if CONUS_only == True:
        varmask, obs = dSS.mask_CONUS(data,data_obs,resolution,lat_bounds,lon_bounds)
        print('*Removed everything by CONUS*')
    
    ### Calculate anomalies
    yearq = np.where((years >= 1981) & (years <= 2010))[0]
    climobs = np.nanmean(obs[yearq,:,:],axis=0)
    climspe = np.nanmean(varmask[:,yearq,:,:],axis=1)
    
    obsanom = obs - climobs
    speanom = varmask - climspe[:,np.newaxis,:,:]

    ### Calculate CONUS averages
    lon2,lat2 = np.meshgrid(lons,lats)
    obs_ave = UT.calc_weightedAve(obsanom,lat2)
    spear_ave = UT.calc_weightedAve(speanom,lat2)
    
    ### Calculate model statistics
    spear_meanens = np.nanmean(spear_ave,axis=0)
    spear_maxens = np.nanmax(spear_ave,axis=0)
    spear_minens = np.nanmin(spear_ave,axis=0)
    
    return obs_ave,spear_meanens,spear_maxens,spear_minens

### Read in data
slicemonth = slicemonthn[0]
slicemonthname = slicemonthnamen[0]

### Return data
obs_ave_w,spear_meanens_w,spear_maxens_w,spear_minens_w = readData('T2M','W_US',slicemonth)
obs_ave_ce,spear_meanens_ce,spear_maxens_ce,spear_minens_ce = readData('T2M','Ce_US',slicemonth)
obs_ave_e,spear_meanens_e,spear_maxens_e,spear_minens_e = readData('T2M','E_US',slicemonth)

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
        
fig = plt.figure(figsize=(7,9))
ax = plt.subplot(311)

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

plt.fill_between(x=yearsmodels,y1=spear_minens_w,y2=spear_maxens_w,facecolor='teal',zorder=0,
         alpha=0.4,edgecolor='none')

plt.plot(yearsmodels,spear_meanens_w,linestyle='-',linewidth=2,color='teal',
         label=r'\textbf{SPEAR_MED}')

plt.plot(years,obs_ave_w,linestyle='--',linewidth=2,color='maroon',
         dashes=(1,0.5),clip_on=False,zorder=30,label=r'\textbf{NClimGrid}')

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10),fontsize=11)
plt.yticks(np.round(np.arange(-18,18.1,2),2),np.round(np.arange(-18,18.1,2),2),fontsize=11)
plt.xlim([1920,2022])
plt.ylim([-4,4])

plt.text(2021,4,r'\textbf{[%s]}' % letters[0],fontsize=10,color='k')

plt.title(r'\textbf{%s: WESTERN USA}' % slicemonthname,
                    color='dimgrey',fontsize=17)

###############################################################################
###############################################################################
############################################################################### 
ax = plt.subplot(312)
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

plt.fill_between(x=yearsmodels,y1=spear_minens_ce,y2=spear_maxens_ce,facecolor='teal',zorder=0,
         alpha=0.4,edgecolor='none')

plt.plot(yearsmodels,spear_meanens_ce,linestyle='-',linewidth=2,color='teal',
         label=r'\textbf{SPEAR_MED}')

plt.plot(years,obs_ave_ce,linestyle='--',linewidth=2,color='maroon',
         dashes=(1,0.5),clip_on=False,zorder=30,label=r'\textbf{NClimGrid}')

leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
      bbox_to_anchor=(0.5,0.98),fancybox=True,ncol=3,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10),fontsize=11)
plt.yticks(np.round(np.arange(-18,18.1,2),2),np.round(np.arange(-18,18.1,2),2),fontsize=11)
plt.xlim([1920,2022])
plt.ylim([-4,4])

plt.text(2021,4,r'\textbf{[%s]}' % letters[1],fontsize=10,color='k')

plt.ylabel(r'\textbf{Baseline of 1981-2010 [$\bf{^\circ}$C]}',fontsize=11,
                      color='k')
plt.title(r'\textbf{%s: CENTRAL USA}' % slicemonthname,
                    color='dimgrey',fontsize=17)

###############################################################################
###############################################################################
############################################################################### 
ax = plt.subplot(313)
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

plt.fill_between(x=yearsmodels,y1=spear_minens_e,y2=spear_maxens_e,facecolor='teal',zorder=0,
         alpha=0.4,edgecolor='none')

plt.plot(yearsmodels,spear_meanens_e,linestyle='-',linewidth=2,color='teal',
         label=r'\textbf{SPEAR_MED}')

plt.plot(years,obs_ave_e,linestyle='--',linewidth=2,color='maroon',
         dashes=(1,0.5),clip_on=False,zorder=30,label=r'\textbf{NClimGrid}')

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10),fontsize=11)
plt.yticks(np.round(np.arange(-18,18.1,2),2),np.round(np.arange(-18,18.1,2),2),fontsize=11)
plt.xlim([1920,2022])
plt.ylim([-4,4])

plt.title(r'\textbf{%s: EASTERN USA}' % slicemonthname,
                    color='dimgrey',fontsize=17)

plt.text(2021,4,r'\textbf{[%s]}' % letters[2],fontsize=10,color='k')

plt.tight_layout()        

### Save figure
plt.savefig(directoryfigure+'%s_RegionsCONUS_1921-2022.png' % slicemonth,dpi=600)

