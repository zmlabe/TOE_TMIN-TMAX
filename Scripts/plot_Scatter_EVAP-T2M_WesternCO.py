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
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/Figures/Scatter/' 

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
    yearmq = np.where((yearsmodels >= 1921) & (yearsmodels <= 1950))[0]
    climspe = np.nanmean(varmask[:,yearmq,:,:],axis=1)
    climnat = np.nanmean(natmask[:,yearmq,:,:],axis=1)
    
    speanomc = varmask - climspe[:,np.newaxis,:,:]
    natanomc = natmask - climnat[:,np.newaxis,:,:]
    
    ### Calculate box over western CO
    latq = np.where((lat >= 37) & (lat <= 41))[0]
    lonq = np.where((lon >= (360-108)) & ((lon <= 360-105)))[0]
    latnew = lat[latq]
    lonnew = lon[lonq]
    
    spearanom1 = speanomc[:,:,latq,:]
    spearanom = spearanom1[:,:,:,lonq]
    
    natanom1 = natanomc[:,:,latq,:]
    natanom = natanom1[:,:,:,lonq]
    
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
    m.contourf(x,y,natanom[0,0],300,extend='both',latlon=True)
    
    ### Calculate CONUS averages
    lon2,lat2 = np.meshgrid(lonnew,latnew)
    spear_avea = UT.calc_weightedAve(spearanom,lat2)
    natural_avea = UT.calc_weightedAve(natanom,lat2)
    
    ### Slice years
    yearqmodels = np.where((yearsmodels >= 1921) & (yearsmodels <= 1950))[0]
    spear_ave = spear_avea[:,yearqmodels]
    natural_ave = natural_avea[:,yearqmodels]
    
    return spear_ave,natural_ave

# ### Return data
# spear_avg_evap, natural_avg_evap = readData('EVAP','JJA',slicenan)
# spear_avg_t, natural_avg_t = readData('T2M','JJA',slicenan)
# spear_avg_run, natural_avg_run = readData('RUNOFF','JJA',slicenan)
# spear_avg_snow, natural_avg_snow = readData('SNOW','MAM',slicenan)

# ### Prepare for scatter 1
# var1_scatter = spear_avg_t.ravel()
# var2_scatter = spear_avg_evap.ravel()
# var1n_scatter = natural_avg_t.ravel()
# var2n_scatter = natural_avg_evap.ravel()

# slope_f1, intercept_f1, r_value_f1, p_value_f1, std_err_f1 = sts.linregress(var1_scatter,var2_scatter)
# slope_n1, intercept_n1, r_value_n1, p_value_n1, std_err_n1 = sts.linregress(var1n_scatter,var2n_scatter)

# ### Prepare for scatter 2
# var3_scatter = spear_avg_run.ravel()
# var3n_scatter = natural_avg_run.ravel()

# slope_f2, intercept_f2, r_value_f2, p_value_f2, std_err_f2 = sts.linregress(var1_scatter,var3_scatter)
# slope_n2, intercept_n2, r_value_n2, p_value_n2, std_err_n2 = sts.linregress(var1n_scatter,var3n_scatter)

# ### Prepare snow scatter
# var4_scatter = spear_avg_snow.ravel()
# var4n_scatter = natural_avg_snow.ravel()

# slope_f3, intercept_f3, r_value_f3, p_value_f3, std_err_f3 = sts.linregress(var1_scatter,var4_scatter)
# slope_n3, intercept_n3, r_value_n3, p_value_n3, std_err_n3 = sts.linregress(var1n_scatter,var4n_scatter)

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

line1 = slope_f1*np.arange(-3,3.1,0.1) + intercept_f1

plt.scatter(var1_scatter,var2_scatter,marker='o',s=30,color='maroon',
            alpha=0.4,edgecolors='maroon',linewidth=0,clip_on=False)
plt.plot(np.arange(-3,3.1,0.1),line1,color='teal',linestyle='-',linewidth=3)

plt.xticks(np.arange(-5,5.1,0.5),map(str,np.round(np.arange(-5,5.1,0.5),2)),fontsize=10)
plt.yticks(np.arange(-5,5.1,0.5),map(str,np.round(np.arange(-5,5.1,0.5),2)),fontsize=10)
plt.xlim([-3,3])
plt.ylim([-1.5,1.5])

plt.title(r'\textbf{SPEAR_MED FOR 1921-1950 OVER WESTERN COLORADO}',color='k',fontsize=11)

plt.xlabel(r'\textbf{JJA -- T2M Anomalies [$^{\circ}$C]}',fontsize=11,color='dimgrey')
plt.ylabel(r'\textbf{JJA -- Evaporation Rate Anomalies [mm/day]}',fontsize=11,color='dimgrey')

plt.text(-3,-1.5,r'\textbf{R=%s, p $<$ 0.001}' % np.round(r_value_f1,2),fontsize=11,color='maroon')

plt.tight_layout()
plt.savefig(directoryfigure + 'Scatter_T2M-EVAP_SPEAR_MED_1921-1950_WesternCO.png',dpi=300)

###############################################################################

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

line1 = slope_f2*np.arange(-3,3.1,0.1) + intercept_f2

plt.scatter(var1_scatter,var3_scatter,marker='o',s=30,color='maroon',
            alpha=0.4,edgecolors='maroon',linewidth=0,clip_on=False)
plt.plot(np.arange(-3,3.1,0.1),line1,color='teal',linestyle='-',linewidth=3)

plt.xticks(np.arange(-5,5.1,0.5),map(str,np.round(np.arange(-5,5.1,0.5),2)),fontsize=10)
plt.yticks(np.arange(-5,5.1,0.1),map(str,np.round(np.arange(-5,5.1,0.1),2)),fontsize=10)
plt.xlim([-3,3])
plt.ylim([-0.2,0.2])

plt.title(r'\textbf{SPEAR_MED FOR 1921-1950 OVER WESTERN COLORADO}',color='k',fontsize=11)

plt.xlabel(r'\textbf{JJA -- T2M Anomalies [$^{\circ}$C]}',fontsize=11,color='dimgrey')
plt.ylabel(r'\textbf{JJA -- Runoff Rate Anomalies [mm/day]}',fontsize=11,color='dimgrey')

plt.text(-3,-0.2,r'\textbf{R=%s, p $<$ 0.001}' % np.round(r_value_f2,2),fontsize=11,color='maroon')

plt.tight_layout()
plt.savefig(directoryfigure + 'Scatter_T2M-RUNOFF_SPEAR_MED_1921-1950_WesternCO.png',dpi=300)

###############################################################################

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

line1 = slope_f2*np.arange(-3,3.1,0.1) + intercept_f2

plt.scatter(var1_scatter,var4_scatter,marker='o',s=30,color='maroon',
            alpha=0.4,edgecolors='maroon',linewidth=0,clip_on=False)
plt.plot(np.arange(-3,3.1,0.1),line1,color='teal',linestyle='-',linewidth=3)

plt.xticks(np.arange(-5,5.1,0.5),map(str,np.round(np.arange(-5,5.1,0.5),2)),fontsize=10)
plt.yticks(np.arange(-5,5.1,0.1),map(str,np.round(np.arange(-5,5.1,0.1),2)),fontsize=10)
plt.xlim([-3,3])
plt.ylim([-0.2,0.2])

plt.title(r'\textbf{SPEAR_MED FOR 1921-1950 OVER WESTERN COLORADO}',color='k',fontsize=11)

plt.xlabel(r'\textbf{JJA -- T2M Anomalies [$^{\circ}$C]}',fontsize=11,color='dimgrey')
plt.ylabel(r'\textbf{MAM -- Snow Depth Anomalies [m]}',fontsize=11,color='dimgrey')

plt.text(-3,-0.2,r'\textbf{R=%s}' % np.round(r_value_f3,2),fontsize=11,color='maroon')

plt.tight_layout()
plt.savefig(directoryfigure + 'Scatter_T2M-SNOWDEPTH_SPEAR_MED_1921-1950_WesternCO.png',dpi=300)
