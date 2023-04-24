"""
Calculate SNR for mean trend / standard deviation of trends

Author    : Zachary M. Labe
Date      : 16 January 2023
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

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/Figures/SNR/' 

### Parameters
monthq = ['JAN','FEB','MAR','ARP','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
variq = 'TMIN'
slicenan = 'nan'
reg_name = 'US'
lat_bounds,lon_bounds = UT.regions(reg_name)
monthlychoiceq = ['JFM','AMJ','JAS','OND','annual','JJA']
slicemonthnamen = ['JAN-MAR','APR-JUN','JUL-SEP','OND','Annual','JUN-AUG']
scenario = 'SSP585'

### Select which SNR to compare
timeperiodq = ['historical']

### Masking and preprocessing arguments
rm_annual_mean = False
rm_merid_mean = False
rm_ensemble_mean = False
land_only = False
ocean_only = False
CONUS_only = True

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
    years = np.arange(1921,2020+1,1)
    years_med = np.arange(1921,2100+1,1)
    years_noaer = np.arange(1921,2020+1,1)

    ### Get data
    lat_bounds,lon_bounds = UT.regions(reg_name)
    spearn,lats,lons = read_primary_dataset(variq,'SPEAR_MED',monthlychoice,scenario,lat_bounds,lon_bounds)
    lon2,lat2 = np.meshgrid(lons,lats)
    
    ### Select same years
    yearq_med = np.where((years_med >= years.min()) & (years_med <= years.max()))[0]
    yearq_noaer = np.where((years_noaer >= years.min()) & (years_noaer <= years.max()))[0]
    
    spear = spearn[:,yearq_med,:,:]
    
    ### Calculate ensemble means
    mean_spear = np.nanmean(spear,axis=0)
    
    ### Calculate trends 
    spear_trend = calcTrend(spear)
    
    spear_trendmean = calcTrend(mean_spear)

    ### Calculate 1 standard deviation of trends
    std_spear = np.nanstd(spear_trend[:,:,:],axis=0)
    
    ### Calculate SNR
    SNR_spear = abs(spear_trendmean)/std_spear
    
    runs = [SNR_spear]
    modelname = ['SPEAR_MED']
    modelnameyears = ['1921-2020']
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Plot variable data for trends
    plt.rc('text',usetex=True)
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
    
    ### Set limits for contours and colorbars
    limit = np.arange(0,5.1,0.5)
    barlim = np.arange(0,6,1)
    cmap = cmr.fall
    label = r'\textbf{%s [signal-to-noise] for %s}' % (variq,monthlychoice)
        
    fig = plt.figure(figsize=(9,4))
    for r in range(len(runs)):
        var = runs[r]
        
        ax1 = plt.subplot(1,1,r+1)
        
        m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                    projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='l',
                    area_thresh=10000)
        m.drawcoastlines(color='darkgrey',linewidth=1)
        m.drawstates(color='darkgrey',linewidth=0.5)
        m.drawcountries(color='darkgrey',linewidth=0.5)
    
        circle = m.drawmapboundary(fill_color='dimgrey',color='dimgrey',
                          linewidth=0.7)
        circle.set_clip_on(False)
        
        cs = m.contourf(lon2,lat2,var,limit,extend='max',latlon=True)
                
        cs.set_cmap(cmap) 
        if any([r==0,r==1]):
            ax1.annotate(r'\textbf{%s}' % modelname[r],xy=(0,0),xytext=(0.5,1.1),
                          textcoords='axes fraction',color='dimgrey',fontsize=22,
                          rotation=0,ha='center',va='center')
        if any([r==0]):
            ax1.annotate(r'\textbf{%s}' % modelnameyears[r],xy=(0,0),xytext=(-0.03,0.5),
                          textcoords='axes fraction',color='k',fontsize=9,
                          rotation=90,ha='center',va='center')
        ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.98,1.02),
                      textcoords='axes fraction',color='k',fontsize=8,
                      rotation=0,ha='center',va='center')
        
        m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)
    
    ###########################################################################
    cbar_ax = fig.add_axes([0.32,0.095,0.4,0.03])                
    cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                        extend='max',extendfrac=0.07,drawedges=False)
    
    cbar.set_label(label,fontsize=9,color='dimgrey',labelpad=1.6)  
    
    cbar.set_ticks(barlim)
    cbar.set_ticklabels(list(map(str,barlim)))
    cbar.ax.tick_params(axis='x', size=.01,labelsize=5)
    cbar.outline.set_edgecolor('dimgrey')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85,wspace=0.01,hspace=0,bottom=0.14)
    
    plt.savefig(directoryfigure + 'SNR_%s_%s_%s.png' % (variq,monthlychoice,timeperiod),dpi=300)
    
