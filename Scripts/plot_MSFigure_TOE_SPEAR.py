"""
Calculate TOE for SPEAR_MED

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
import palettable.cubehelix as cm
import palettable.scientific.sequential as scm
import pandas as pd

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/MS_Figures/' 

### Parameters
monthq = ['JAN','FEB','MAR','ARP','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
slicenan = 'nan'
reg_name = 'US'
lat_bounds,lon_bounds = UT.regions(reg_name)
monthlychoiceq = ['JJA']
slicemonthnamen = ['JUN-AUG']
scenario = 'SSP585'

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

### Calculate ToE
def calcToE(database,datalimit,years):
    """ 
    Calculate ToE from Lehner et al. 2017
    """
    
    toe = np.empty((database.shape[0],database.shape[2],database.shape[3]))
    toe[:,:,:] = np.nan
    for ens in range(database.shape[0]):
        for i in range(database.shape[2]):
            for j in range(database.shape[3]):
                limit = datalimit[ens,i,j]
                for yr in range(database.shape[1]):
                    smooth = database[ens,yr,i,j]
                    if smooth > limit:
                        if np.nanmin(database[ens,yr:,i,j]) > limit:
                            if np.isnan(toe[ens,i,j]):
                                toe[ens,i,j] = years[yr]
                        
        print('Completed: Ensemble #%s ToE!' % (ens+1))
    
    return toe

### Functions for calculating moving averages
def moving_average(data,window):
    """ 
    Calculating rolling mean over set window
    """
    ### Import functions
    import numpy as np
    
    movemean = np.convolve(data,np.ones(window),'valid') / window
    return movemean

def rollingMean(data,w,mp):
    """ 
    Calculating rolling mean over set window
    """
    ### Import functions
    import numpy as np
    import pandas as pd
    
    datadf = pd.Series(data)
    movemean = datadf.rolling(window=w,min_periods=mp).mean().to_numpy()
    return movemean

def calcTOE(variq):
    monthlychoice = monthlychoiceq[0]
    
    ### Years for each simulation
    years = np.arange(1921,2020+1,1)
    years_med = np.arange(1921,2100+1,1)
    years_nonoaer = np.arange(1921,2020+1,1)
    
    ### Get data
    lat_bounds,lon_bounds = UT.regions(reg_name)
    spearn,lats,lons = read_primary_dataset(variq,'SPEAR_MED',monthlychoice,scenario,lat_bounds,lon_bounds)
    lon2,lat2 = np.meshgrid(lons,lats)
    
    ### Select same years
    yearq_med = np.where((years_med >= years.min()) & (years_med <= years.max()))[0]
    
    spear = spearn[:,yearq_med,:,:]
    
    ### Calculate ensemble means
    mean_spear = np.nanmean(spear,axis=0)
    
    ### 10-year running mean        
    window = 10 
    min_periods = 10
    smooth_spear = np.empty((spear.shape[0],spear.shape[1],spear.shape[2],spear.shape[3]))
    for ens in range(spear.shape[0]):
        for i in range(spear.shape[2]):
            for j in range(spear.shape[3]):
                smooth_spear[ens,:,i,j] = rollingMean(spear[ens,:,i,j],window,min_periods)
        print('Completed: SPEAR_MED Ensemble #%s running mean!' % (ens+1))
        
    ### Slice baseline of 1921-1950
    minyr = 1921
    maxyr = 1950
    yearq = np.where((years >= minyr) & ((years <= maxyr)))[0]
    spearbase = smooth_spear[:,yearq,:,:]
    
    ### 2 Sigma of 1920-1949
    spear2 = np.nanstd(spearbase[:,:,:,:],axis=1) * 2.
    
    ### Limit of baseline
    spearbasemean = np.nanmean(spearbase[:,:,:,:],axis=1)
    
    spearlimit = spearbasemean + spear2
    
    ### Calculate ToE
    toe_spear = calcToE(smooth_spear,spearlimit,years)
    
    ### Calculate ensemble mean ToE
    mtoe_spear = np.nanmedian(toe_spear,axis=0)

    return mtoe_spear,lats,lons    

### Calc TOE 
tavg_spear,lats,lons = calcTOE('T2M')
tmax_spear,lats,lons = calcTOE('TMAX')
tmin_spear,lats,lons = calcTOE('TMIN')

### Prepare for plotting
runs = [tavg_spear,tmax_spear,tmin_spear]
modelname = [r'TAVG',r'TMAX',r'TMIN']

###########################################################################
###########################################################################
###########################################################################
### Plot variable data for trends
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

### Set limits for contours and colorbars
barlim = np.round(np.arange(1920,2020+1,10),2)
barlim2 = np.round(np.arange(1925,2030+1,10),2)
barlim3 = [r'1920s',r'1930s',r'1940s',r'1950s',r'1960s',r'1970s',r'1980s',r'1990s',
           r'2000s',r'2010s','']
limit = np.arange(1920,2020+1,10)
cmap = scm.Batlow_17.mpl_colormap   
    
fig = plt.figure(figsize=(9,3))
for r in range(len(runs)):
    var = runs[r]
    lon2,lat2 = np.meshgrid(lons,lats)
    
    ax1 = plt.subplot(1,3,r+1)
    
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

    ax1.annotate(r'\textbf{%s}' % modelname[r],xy=(0,0),xytext=(0.5,1.1),
                  textcoords='axes fraction',color='dimgrey',fontsize=22,
                  rotation=0,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.98,1.05),
                  textcoords='axes fraction',color='k',fontsize=8,
                  rotation=0,ha='center',va='center')
    
    m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)

###########################################################################
cbar_ax = fig.add_axes([0.30,0.095,0.4,0.03])                
cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)

cbar.set_label(r'\textbf{TIMING OF EMERGENCE [Years]}',fontsize=11,color='dimgrey',labelpad=1.4)  
cbar.set_ticks(barlim2)
cbar.set_ticklabels(barlim3)
cbar.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar.outline.set_edgecolor('dimgrey')
cbar.outline.set_linewidth(1)
cbar.dividers.set_color('dimgrey')
cbar.dividers.set_linewidth(1)

plt.tight_layout()
plt.subplots_adjust(top=0.85,wspace=0.01,hspace=0,bottom=0.14)

plt.savefig(directoryfigure + 'ToE_T2MTMAXTMIN_JJA.png',dpi=600)
