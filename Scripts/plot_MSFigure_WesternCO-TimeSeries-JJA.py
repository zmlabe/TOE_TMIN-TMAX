"""
Compare western CO temperature anomalies for different SPEAR experiments
 
Author    : Zachary M. Labe
Date      : 3 April 2023
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
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/MS_Figures/' 

### Parameters
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
slicemonthn = ['JJA']
slicemonthnamen = ['JUN-AUG']
slicenan = 'nan'
years = np.arange(1921,2022+1,1)
yearsmodels = np.arange(1921,2100+1,1)

def readData(variq,slicemonth,slicenan,years):
    ### Read in obs
    lat,lon,obs = NC.read_NClimGrid_monthlyMEDS(variq,'/work/Zachary.Labe/Data/',slicemonth,years,3,slicenan)
    
    ### Read in SPEAR_MED
    lat,lon,var = SP.read_SPEAR_MED('/work/Zachary.Labe/Data/SPEAR/SPEAR_MED/monthly/',variq,slicemonth,4,slicenan,30,'all')

    ### Create mask
    mask = obs.copy()
    mask[np.where(np.isnan(mask))] = 0.
    mask[np.where(mask != 0)] = 1.
    
    ### Only consider CONUS
    varmask = var * mask[-1]
    varmask[np.where(varmask == 0.)] = np.nan
    
    ### Calculate anomalies
    yearq = np.where((years >= 1981) & (years <= 2010))[0]
    climobs = np.nanmean(obs[yearq,:,:],axis=0)
    climspe = np.nanmean(varmask[:,yearq,:,:],axis=1)
    
    obsanom = obs - climobs
    spearnom = varmask - climspe[:,np.newaxis,:,:]
    
    ### Calculate box over western CO
    latq = np.where((lat >= 37) & (lat <= 41))[0]
    lonq = np.where((lon >= (360-108)) & ((lon <= 360-105)))[0]
    latnew = lat[latq]
    lonnew = lon[lonq]
    
    obsanom1 = obsanom[:,latq,:]
    obsanom_co = obsanom1[:,:,lonq]
    
    spearanom1 = spearnom[:,:,latq,:]
    spearanom_co = spearanom1[:,:,:,lonq]
    
    ### Check region
    plt.figure()
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='l')
    m.drawcoastlines(color='k',linewidth=1)
    m.drawstates(color='k',linewidth=1)
    m.drawcountries(color='k',linewidth=1)
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)
    x, y = np.meshgrid(lonnew,latnew)
    m.contourf(x,y,obsanom_co[0],300,extend='both',latlon=True)

    ### Calculate CONUS averages
    lon2,lat2 = np.meshgrid(lonnew,latnew)
    obs_ave = UT.calc_weightedAve(obsanom_co,lat2)
    spear_ave = UT.calc_weightedAve(spearanom_co,lat2)
    
    return obs_ave,spear_ave

for i in range(len(slicemonthn)):
    slicemonth = slicemonthn[i]
    slicemonthname = slicemonthnamen[i]
    
    ### Return data
    obs_max, spear_max = readData('TMAX',slicemonth,slicenan,years)
    obs_min, spear_min = readData('TMIN',slicemonth,slicenan,years)
    obs_avg, spear_avg= readData('T2M',slicemonth,slicenan,years)
    
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
            
    fig = plt.figure(figsize=(8,9))
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
    
    plt.fill_between(x=yearsmodels,y1=spear_minens_max,y2=spear_maxens_max,facecolor='teal',zorder=0,
             alpha=0.4,edgecolor='none')
    
    plt.plot(yearsmodels,spear_meanens_max,linestyle='-',linewidth=2,color='teal',
             label=r'\textbf{SPEAR_MED}')
    
    plt.plot(years,obs_max,linestyle='--',linewidth=2,color='maroon',
             dashes=(1,0.5),zorder=30,label=r'\textbf{NClimGrid}')
    
    plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10),fontsize=13)
    plt.yticks(np.round(np.arange(-18,18.1,1),2),np.round(np.arange(-18,18.1,1),2),fontsize=13)
    plt.xlim([1920,1990])
    plt.ylim([-5,3])
    
    plt.text(1990,3,r'\textbf{[%s]}' % letters[1],fontsize=10,color='k')
    
    plt.title(r'\textbf{TMAX}',
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
    
    plt.fill_between(x=yearsmodels,y1=spear_minens_min,y2=spear_maxens_min,facecolor='teal',zorder=0,
             alpha=0.4,edgecolor='none')
    
    plt.plot(yearsmodels,spear_meanens_min,linestyle='-',linewidth=2,color='teal',
             label=r'\textbf{SPEAR_MED}')
    
    plt.plot(years,obs_min,linestyle='--',linewidth=2,color='maroon',
             dashes=(1,0.5),zorder=30,label=r'\textbf{NClimGrid}')
    
    leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
          bbox_to_anchor=(0.5,0.98),fancybox=True,ncol=3,frameon=False,
          handlelength=1,handletextpad=0.5)
    for line,text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    
    plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10),fontsize=13)
    plt.yticks(np.round(np.arange(-18,18.1,1),2),np.round(np.arange(-18,18.1,1),2),fontsize=13)
    plt.xlim([1920,1990])
    plt.ylim([-5,3])
    
    plt.text(1990,3,r'\textbf{[%s]}' % letters[2],fontsize=10,color='k')
    
    plt.ylabel(r'\textbf{Baseline of 1981-2010 [$\bf{^\circ}$C]}',fontsize=11,
                          color='k')
    plt.title(r'\textbf{%s - TMIN - Western Colorado}' % slicemonthname,
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
    
    plt.fill_between(x=yearsmodels,y1=spear_minens_avg,y2=spear_maxens_avg,facecolor='teal',zorder=0,
             alpha=0.4,edgecolor='none')
    
    plt.plot(yearsmodels,spear_meanens_avg,linestyle='-',linewidth=2,color='teal',
             label=r'\textbf{SPEAR_MED}')
    
    plt.plot(years,obs_avg,linestyle='--',linewidth=2,color='maroon',
             dashes=(1,0.5),zorder=30,label=r'\textbf{NClimGrid}')
    
    plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10),fontsize=13)
    plt.yticks(np.round(np.arange(-18,18.1,1),2),np.round(np.arange(-18,18.1,1),2),fontsize=13)
    plt.xlim([1920,1990])
    plt.ylim([-5,3])
    
    plt.text(1990,3,r'\textbf{[%s]}' % letters[3],fontsize=10,color='k')
    
    plt.title(r'\textbf{TAVG}',
                        color='dimgrey',fontsize=17)
    
    plt.tight_layout()        
    
    ### Save figure
    plt.savefig(directoryfigure+'%s_T-WesternCO_1921-1990.png' % slicemonth,dpi=600)
