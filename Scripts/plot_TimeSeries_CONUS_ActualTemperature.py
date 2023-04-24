"""
Compare regional mean temperatures for different SPEAR experiments
 
Author    : Zachary M. Labe
Date      : 12 September 2022
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

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/Figures/' 

### Parameters
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
slicemonthn = ['JFM','AMJ','JAS','OND','annual','JJA']
slicemonthnamen = ['JAN-MAR','APR-JUN','JUL-SEP','OND','Annual','JUN-AUG']
slicenan = 'nan'
years = np.arange(1921,2021+1,1)

def readData(variq,slicemonth,slicenan,years):
    ### Read in obs
    lat,lon,obs = NC.read_NClimGrid_monthlyMEDS(variq,'/work/Zachary.Labe/Data/',slicemonth,years,3,slicenan)
    
    ### Read in SPEAR_MED
    lat,lon,var = SP.read_SPEAR_MED('/work/Zachary.Labe/Data/SPEAR/SPEAR_MED/monthly/',variq,slicemonth,4,slicenan,30,'historical')

    ### Create mask
    mask = obs.copy()
    mask[np.where(np.isnan(mask))] = 0.
    mask[np.where(mask != 0)] = 1.
    
    ### Only consider CONUS
    varmask = var * mask[-1]
    varmask[np.where(varmask == 0.)] = np.nan
    
    ### Calculate CONUS averages
    lon2,lat2 = np.meshgrid(lon,lat)
    obs_ave = UT.calc_weightedAve(obs,lat2)
    spear_ave = UT.calc_weightedAve(varmask,lat2)
    
    return obs_ave,spear_ave

for i in range(len(slicemonthn)):
    slicemonth = slicemonthn[i]
    slicemonthname = slicemonthnamen[i]
    
    ### Return data
    obs_max, spear_max = readData('TMAX',slicemonth,slicenan,years)
    obs_min, spear_min = readData('TMIN',slicemonth,slicenan,years)
    obs_avg, spear_avg = readData('T2M',slicemonth,slicenan,years)
    
    ### Calculate model statistics
    spear_meanens_max = np.nanmean(spear_max,axis=0)
    spear_maxens_max = np.nanmax(spear_max,axis=0)
    spear_minens_max = np.nanmin(spear_max,axis=0)
    
    ###############################################################################
    ###############################################################################
    ############################################################################### 
    spear_meanens_min = np.nanmean(spear_min,axis=0)
    spear_maxens_min = np.nanmax(spear_min,axis=0)
    spear_minens_min = np.nanmin(spear_min,axis=0)
    
    ###############################################################################
    ###############################################################################
    ############################################################################### 
    spear_meanens_avg = np.nanmean(spear_avg,axis=0)
    spear_maxens_avg = np.nanmax(spear_avg,axis=0)
    spear_minens_avg = np.nanmin(spear_avg,axis=0)
    
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
    
    plt.fill_between(x=years,y1=spear_minens_max,y2=spear_maxens_max,facecolor='maroon',zorder=0,
             alpha=0.4,edgecolor='none')
    plt.plot(years,spear_meanens_max,linestyle='-',linewidth=2,color='maroon',
             label=r'\textbf{SPEAR-MED-SSP5-8.5}')
    
    plt.plot(years,obs_max,linestyle='--',linewidth=2,color='k',
             dashes=(1,0.5),clip_on=False,zorder=30,label=r'\textbf{NClimGrid}')
    
    plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
    plt.yticks(np.round(np.arange(-30,41,5),2),np.round(np.arange(-30,41,5),2))
    plt.xlim([1920,2021])
    plt.ylim([-15,35])
    
    plt.title(r'\textbf{%s: TMAX}' % slicemonthname,
                        color='k',fontsize=17)
    
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
    
    plt.fill_between(x=years,y1=spear_minens_min,y2=spear_maxens_min,facecolor='maroon',zorder=0,
             alpha=0.4,edgecolor='none')
    plt.plot(years,spear_meanens_min,linestyle='-',linewidth=2,color='maroon',
             label=r'\textbf{SPEAR-MED-SSP5-8.5}')
    
    plt.plot(years,obs_min,linestyle='--',linewidth=2,color='k',
             dashes=(1,0.5),clip_on=False,zorder=30,label=r'\textbf{NClimGrid}')
    
    leg = plt.legend(shadow=False,fontsize=8,loc='upper center',
          bbox_to_anchor=(0.5,1.05),fancybox=True,ncol=3,frameon=False,
          handlelength=1,handletextpad=0.5)
    
    plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
    plt.yticks(np.round(np.arange(-30,41,5),2),np.round(np.arange(-30,41,5),2))
    plt.xlim([1920,2021])
    plt.ylim([-15,35])
    
    plt.ylabel(r'\textbf{[$\bf{^\circ}$C]}',fontsize=11,
                          color='dimgrey')
    plt.title(r'\textbf{%s: TMIN}' % slicemonthname,
                        color='k',fontsize=17)
    
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
    
    plt.fill_between(x=years,y1=spear_minens_avg,y2=spear_maxens_avg,facecolor='maroon',zorder=0,
             alpha=0.4,edgecolor='none')
    plt.plot(years,spear_meanens_avg,linestyle='-',linewidth=2,color='maroon',
             label=r'\textbf{SPEAR-MED-SSP5-8.5}')
    
    plt.plot(years,obs_avg,linestyle='--',linewidth=2,color='k',
             dashes=(1,0.5),clip_on=False,zorder=30,label=r'\textbf{NClimGrid}')
    
    plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
    plt.yticks(np.round(np.arange(-30,41,5),2),np.round(np.arange(-30,41,5),2))
    plt.xlim([1920,2021])
    plt.ylim([-20,35])
    
    plt.title(r'\textbf{%s: TAVG}' % slicemonthname,
                        color='k',fontsize=17)
    
    plt.tight_layout()        
    
    ### Save figure
    plt.savefig(directoryfigure+'%s_T-CONUS_1921-2021_ActualTemperature.png' % slicemonth,dpi=1000)
