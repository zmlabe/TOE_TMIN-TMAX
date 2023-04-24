"""
Calculate trend for different SPEAR_MED periods

Author    : Zachary M. Labe
Date      : 15 August 2022
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
plt.rc('savefig',facecolor='black')
plt.rc('axes',edgecolor='darkgrey')
plt.rc('xtick',color='darkgrey')
plt.rc('ytick',color='darkgrey')
plt.rc('axes',labelcolor='darkgrey')
plt.rc('axes',facecolor='black')
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/Dark_Figures/' 

### Parameters
monthq = ['JAN','FEB','MAR','ARP','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
variables = ['T2M','EVAP','WA','RUNOFF','ALBEDO','SNOW','PRECT','TS']
slicenan = 'nan'
reg_name = 'US'
lat_bounds,lon_bounds = UT.regions(reg_name)
monthlychoiceq = ['JFM','AMJ','JJA']
slicemonthnamen = ['JAN-MAR','APR-JUN','JUN-AUG']
scenario = 'SSP585'

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

for s in range(len(variables)):
    variq = variables[s]
    for m in range(len(monthlychoiceq)):
        monthlychoice = monthlychoiceq[m]
    
        years = np.arange(1921,1970+1,1)
        years_med = np.arange(1921,2100+1,1)
        years_noaer = np.arange(1921,2020+1,1)
        years_natural = np.arange(1921,2100+1,1)
    
        ### Get data
        lat_bounds,lon_bounds = UT.regions(reg_name)
        spearn,lats,lons = read_primary_dataset(variq,'SPEAR_MED',monthlychoice,scenario,lat_bounds,lon_bounds)
        noaern,lats,lons = read_primary_dataset(variq,'SPEAR_MED_NOAER',monthlychoice,scenario,lat_bounds,lon_bounds)
        naturaln,lats,lons = read_primary_dataset(variq,'SPEAR_MED_NATURAL',monthlychoice,scenario,lat_bounds,lon_bounds)
        
        ### Prepare plotting grid
        lon2,lat2 = np.meshgrid(lons,lats)
        
        ### Select same years
        yearq_med = np.where((years_med >= years.min()) & (years_med <= years.max()))[0]
        yearq_noaer = np.where((years_noaer >= years.min()) & (years_noaer <= years.max()))[0]
        yearq_natural = np.where((years_natural >= years.min()) & (years_natural <= years.max()))[0]
        
        spear = spearn[:,yearq_med,:,:]
        noaer = noaern[:,yearq_noaer,:,:]
        natural = naturaln[:,yearq_natural,:,:]
        
        ### Calculate ensemble means
        mean_spear = np.nanmean(spear,axis=0)
        mean_noaer = np.nanmean(noaer,axis=0)
        mean_natural = np.nanmean(natural,axis=0)
        
        ### Calculate trends  
        spear_trendmean = calcTrend(mean_spear)
        noaer_trendmean = calcTrend(mean_noaer)
        natural_trendmean = calcTrend(mean_natural)
        
        runs = [spear_trendmean,noaer_trendmean,natural_trendmean]
        modelname = ['SPEAR_MED','SPEAR_MED_NOAER','SPEAR_MED_NATURAL']
        modelnameyears = ['1921-1960']
        
        ###########################################################################
        ###########################################################################
        ###########################################################################
        ### Plot variable data for trends
        plt.rc('text',usetex=True)
        plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
        
        ### Set limits for contours and colorbars
        if any([variq == 'T2M', variq=='TS']):
            label = r'\textbf{%s Trend [$^{\circ}$C/decade] for %s-%s in %s}' % (variq,years.min(),years.max(),monthlychoice)
            limit = np.arange(-0.8,0.81,0.01)
            barlim = np.round(np.arange(-0.8,0.81,0.4),2)
            cmap = cmocean.cm.balance
        elif variq == 'EVAP':
            label = r'\textbf{%s Trend [mm/decade] for %s-%s in %s}' % (variq,years.min(),years.max(),monthlychoice)
            limit = np.arange(-0.4,0.41,0.01)
            barlim = np.round(np.arange(-0.4,0.41,0.2),2)
            cmap = cmr.waterlily
        elif variq == 'PRECT':
            label = r'\textbf{%s Trend [mm/decade] for %s-%s in %s}' % (variq,years.min(),years.max(),monthlychoice)
            limit = np.arange(-0.4,0.41,0.01)
            barlim = np.round(np.arange(-0.4,0.41,0.2),2)
            cmap = cmr.waterlily
        elif variq == 'RUNOFF':
            label = r'\textbf{%s Trend [mm/decade] for %s-%s in %s}' % (variq,years.min(),years.max(),monthlychoice)
            limit = np.arange(-0.4,0.41,0.01)
            barlim = np.round(np.arange(-0.4,0.41,0.2),2)
            cmap = cmocean.cm.balance
        elif variq == 'WA':
            label = r'\textbf{%s Trend [mm/decade] for %s-%s in %s}' % (variq,years.min(),years.max(),monthlychoice)
            limit = np.arange(-0.4,0.41,0.01)
            barlim = np.round(np.arange(-0.4,0.41,0.2),2)
            cmap = cmr.waterlily
        elif variq == 'ALBEDO':
            label = r'\textbf{%s Trend [percent/decade] for %s-%s in %s}' % (variq,years.min(),years.max(),monthlychoice)
            limit = np.arange(-2,2.01,0.01)
            barlim = np.round(np.arange(-2,3,1),2)
            cmap = cmocean.cm.balance
        elif variq == 'SNOW':
            label = r'\textbf{%s Trend [m/decade] for %s-%s in %s}' % (variq,years.min(),years.max(),monthlychoice)
            limit = np.arange(-0.02,0.021,0.001)
            barlim = np.round(np.arange(-0.02,0.03,0.01),2)
            cmap = cmr.waterlily
        else:
            print(ValueError('VARIABLES DOES NOT HAVE UNITS YET!'))
            sys.exit()
            
        fig = plt.figure(figsize=(9,3.5))
        for r in range(len(runs)):
            var = runs[r]
            
            ax1 = plt.subplot(1,3,r+1)
            
            m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                        projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='l',
                        area_thresh=10000)
            m.drawcoastlines(color='darkgrey',linewidth=1)
            m.drawstates(color='darkgrey',linewidth=0.5)
            m.drawcountries(color='darkgrey',linewidth=0.5)
        
            circle = m.drawmapboundary(fill_color='k',color='k',
                              linewidth=0.7)
            circle.set_clip_on(False)
            
            cs = m.contourf(lon2,lat2,var,limit,extend='both',latlon=True)
                    
            cs.set_cmap(cmap) 
            if any([r==0,r==1,r==2]):
                ax1.annotate(r'\textbf{%s}' % modelname[r],xy=(0,0),xytext=(0.5,1.1),
                              textcoords='axes fraction',color='w',fontsize=17,
                              rotation=0,ha='center',va='center')
            
            m.drawlsmask(land_color=(0,0,0,0),ocean_color='k',lakes=False,zorder=11)
        
        ###########################################################################
        cbar_ax = fig.add_axes([0.32,0.14,0.4,0.03])                
        cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                            extend='both',extendfrac=0.07,drawedges=False)
        
        cbar.set_label(label,fontsize=9,color='darkgrey',labelpad=1.6)  
        
        cbar.set_ticks(barlim)
        cbar.set_ticklabels(list(map(str,barlim)))
        cbar.ax.tick_params(axis='x', size=.01,labelsize=5)
        cbar.outline.set_edgecolor('darkgrey')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85,wspace=0.01,hspace=0,bottom=0.12)
        
        plt.savefig(directoryfigure + 'TrendLinear_%s_%s_%s-%s.png' % (variq,monthlychoice,years.min(),years.max()),dpi=300)
        
