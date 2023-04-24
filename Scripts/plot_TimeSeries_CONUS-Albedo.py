"""
Compare regional mean albedo anomalies for different SPEAR experiments
 
Author    : Zachary M. Labe
Date      : 23 January 2023
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import numpy as np
import cmocean
import calc_Utilities as UT
import sys
import itertools
import read_ERA5_monthlyMEDS as ER
import read_SPEAR_MED as SP
import read_SPEAR_MED_NATURAL as NAT
import read_NClimGrid_monthlyMEDS as NC
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
years = np.arange(1979,2021+1,1)
yearsmodels = np.arange(1921,2100+1,1)

def readData(variq,slicemonth,slicenan,years):
    ### Read in obs
    lat,lon,lev,obs = ER.read_ERA5_monthlyMEDS(variq,'/work/Zachary.Labe/Data/',slicemonth,years,3,True,slicenan,'surface') 
    lattemp,lontemp,obstemp = NC.read_NClimGrid_monthlyMEDS('T2M','/work/Zachary.Labe/Data/',slicemonth,years,3,slicenan)
    
    ### Read in SPEAR_MED
    lat,lon,var = SP.read_SPEAR_MED('/work/Zachary.Labe/Data/SPEAR/SPEAR_MED/monthly/',variq,slicemonth,4,slicenan,30,'all')
    
    ### Read in SPEAR_MED_NATURAL
    lat,lon,nat = NAT.read_SPEAR_MED_NATURAL('/work/Zachary.Labe/Data/SPEAR/SPEAR_MED_NATURAL/monthly/',variq,slicemonth,4,slicenan,30,'all')

    ### Create mask
    mask = obstemp.copy()
    mask[np.where(np.isnan(mask))] = 0.
    mask[np.where(mask != 0)] = 1.
    
    ### Only consider CONUS
    varmask = var * mask[-1]
    varmask[np.where(varmask == 0.)] = np.nan
    natmask = nat * mask[-1]
    natmask[np.where(natmask == 0.)] = np.nan
    
    ### Calculate anomalies
    yearq = np.where((years >= 1981) & (years <= 2010))[0]
    climobs = np.nanmean(obs[yearq,:,:],axis=0)
    climspe = np.nanmean(varmask[:,yearq,:,:],axis=1)
    climnat = np.nanmean(natmask[:,yearq,:,:],axis=1)
    
    obsanom = obs - climobs
    speanom = varmask - climspe[:,np.newaxis,:,:]
    natanom = natmask - climnat[:,np.newaxis,:,:]
    
    ### Calculate CONUS averages
    lon2,lat2 = np.meshgrid(lon,lat)
    obs_ave = UT.calc_weightedAve(obsanom,lat2)
    spear_ave = UT.calc_weightedAve(speanom,lat2)
    natural_ave = UT.calc_weightedAve(natanom,lat2)
    
    return obs_ave,spear_ave,natural_ave

for i in range(len(slicemonthn)):
    slicemonth = slicemonthn[i]
    slicemonthname = slicemonthnamen[i]
    
    ### Return data
    obs_avg, spear_avg, natural_avg = readData('ALBEDO',slicemonth,slicenan,years)

    ###############################################################################
    ###############################################################################
    ############################################################################### 
    spear_meanens_avg = np.nanmean(spear_avg,axis=0)
    spear_maxens_avg = np.nanmax(spear_avg,axis=0)
    spear_minens_avg = np.nanmin(spear_avg,axis=0)
    
    spear_natural_meanens_avg = np.nanmean(natural_avg,axis=0)
    spear_natural_maxens_avg = np.nanmax(natural_avg,axis=0)
    spear_natural_minens_avg = np.nanmin(natural_avg,axis=0)
    
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
            
    fig = plt.figure(figsize=(11,6))
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
    
    plt.fill_between(x=yearsmodels,y1=spear_natural_minens_avg,y2=spear_natural_maxens_avg,facecolor='teal',zorder=0,
              alpha=0.4,edgecolor='none')
    plt.fill_between(x=yearsmodels,y1=spear_minens_avg,y2=spear_maxens_avg,facecolor='maroon',zorder=0,
             alpha=0.4,edgecolor='none')
    
    plt.plot(yearsmodels,spear_natural_meanens_avg,linestyle='-',linewidth=2,color='teal',
              label=r'\textbf{SPEAR-MED-NATURAL}')
    plt.plot(yearsmodels,spear_meanens_avg,linestyle='-',linewidth=2,color='maroon',
             label=r'\textbf{SPEAR-MED-SSP5-8.5}')
    
    plt.plot(years,obs_avg,linestyle='--',linewidth=2,color='k',
             dashes=(1,0.5),clip_on=False,zorder=30,label=r'\textbf{ERA5}')
    
    plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
    plt.yticks(np.round(np.arange(-25,25.01,1),2),np.round(np.arange(-25,25.01,1),2))
    plt.xlim([1920,2100])
    plt.ylim([-5,5])
    
    plt.title(r'\textbf{%s: ALBEDO}' % slicemonthname,
                        color='k',fontsize=17)
    plt.ylabel(r'\textbf{Anomalies relative to 1981-2010 [Percent]}',fontsize=11,
                          color='dimgrey')
    
    plt.tight_layout()        
    
    ### Save figure
    plt.savefig(directoryfigure+'%s_ALBEDO-CONUS_1921-2100.png' % slicemonth,dpi=300)
