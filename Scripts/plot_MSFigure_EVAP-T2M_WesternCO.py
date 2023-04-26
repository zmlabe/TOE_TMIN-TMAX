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
import read_ERA5_monthlyMEDS as ER
import statsmodels.api as sm

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/MS_Figures/' 

### Parameters
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
slicemonth = ['JJA']
slicemonthname = ['JUN-AUG']
slicenan = 'nan'

def readData(variq,slicemonth,slicenan):
    yearsobs = np.arange(1921,2022+1,1)
    yearsmodels = np.arange(1921,2100+1,1)
    
    ### Read in obs for mask
    lattemp,lontemp,obstemp = NC.read_NClimGrid_monthlyMEDS('T2M','/work/Zachary.Labe/Data/',slicemonth,yearsobs,3,slicenan)
    
    ### Read in SPEAR_MED
    lat,lon,var = SP.read_SPEAR_MED('/work/Zachary.Labe/Data/SPEAR/SPEAR_MED/monthly/',variq,slicemonth,4,slicenan,30,'all')
    
    ### Create mask
    mask = obstemp.copy()
    mask[np.where(np.isnan(mask))] = 0.
    mask[np.where(mask != 0)] = 1.
    
    ### Only consider CONUS
    varmask = var * mask[-1]
    varmask[np.where(varmask == 0.)] = np.nan
    
    ### Calculate anomalies
    yearmq = np.where((yearsmodels >= 1921) & (yearsmodels <= 1950))[0]
    climspe = np.nanmean(varmask[:,yearmq,:,:],axis=1)
    
    speanomc = varmask - climspe[:,np.newaxis,:,:]
    
    ### Calculate box over western CO
    latq = np.where((lat >= 37) & (lat <= 41))[0]
    lonq = np.where((lon >= (360-108)) & ((lon <= 360-105)))[0]
    latnew = lat[latq]
    lonnew = lon[lonq]
    
    spearanom1 = speanomc[:,:,latq,:]
    spearanom = spearanom1[:,:,:,lonq]
    
    ### Check region
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
    m.contourf(x,y,spearanom[0,0],300,extend='both',latlon=True)
    
    ### Calculate CONUS averages
    lon2,lat2 = np.meshgrid(lonnew,latnew)
    spear_avea = UT.calc_weightedAve(spearanom,lat2)
    
    ### Slice years
    yearqmodels = np.where((yearsmodels >= 1921) & (yearsmodels < 1990))[0]
    spear_ave = spear_avea[:,yearqmodels]
    
    return spear_ave

### Return data
spear_avg_evap = readData('EVAP','JJA',slicenan)
spear_avg_t = readData('T2M','JJA',slicenan)

### Prepare for scatter 1
var1_scatter = spear_avg_t.ravel()
var2_scatter = spear_avg_evap.ravel()

slope_f1, intercept_f1, r_value_f1, p_value_f1, std_err_f1 = sts.linregress(var1_scatter,var2_scatter)

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

line1 = slope_f1*np.arange(-3,3.1,0.1) + intercept_f1

plt.scatter(var1_scatter,var2_scatter,marker='o',s=30,color='teal',
            alpha=0.4,edgecolors='teal',linewidth=0,clip_on=False)
plt.plot(np.arange(-3,3.1,0.1),line1,color='k',linestyle='-',linewidth=3)

plt.xticks(np.arange(-5,5.1,0.5),map(str,np.round(np.arange(-5,5.1,0.5),2)),fontsize=11)
plt.yticks(np.arange(-5,5.1,0.5),map(str,np.round(np.arange(-5,5.1,0.5),2)),fontsize=11)
plt.xlim([-3,3])
plt.ylim([-1.5,1.5])

plt.xlabel(r'\textbf{TAVG Anomalies [$^{\circ}$C]}',fontsize=11,color='k')
plt.ylabel(r'\textbf{Evaporation Rate Anomalies [mm/day]}',fontsize=11,color='k')

plt.text(-2.97,-1.47,r'\textbf{R=%s, p $<$ 0.001}' % np.round(r_value_f1,2),fontsize=17,color='k')

plt.text(3,1.5,r'\textbf{[a]}',fontsize=10,color='k')

plt.tight_layout()
plt.savefig(directoryfigure + 'Scatter_T2M-EVAP_SPEAR_MED_1921-1989_WesternCO.png',dpi=600)
