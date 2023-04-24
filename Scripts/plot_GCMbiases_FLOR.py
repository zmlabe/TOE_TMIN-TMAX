"""
Plot biases for FLOR over CONUS

Author    : Zachary M. Labe
Date      : 10 October 2022
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import numpy as np
import cmocean
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import sys
import scipy.stats as sts

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/Figures/Bias/FLOR/' 

### Parameters
monthq = ['JAN','FEB','MAR','ARP','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
variq = 'TMIN'
monthlychoiceq = ['JFM','AMJ','JAS','OND','annual','JJA']
slicemonthnamen = ['JAN-MAR','APR-JUN','JUL-SEP','OND','Annual','JUN-AUG']
slicenan = 'nan'
reg_name = 'US'
lat_bounds,lon_bounds = UT.regions(reg_name)

dataset = 'FLOR'
dataset_obs = 'NClimGrid_MEDS'

### Select years of data
if dataset == 'FLOR':
    scenario = 'RCP85'
    years = np.arange(1921,2100+1,1)

if any([dataset_obs == 'NClimGrid_MEDS']):
    yearsobs = np.arange(1921,2021+1,1)
    timeexperi = ['1921-1971','1972-2021']
elif any([dataset_obs == '20CRv3_LOWS']):
    yearsobs = np.arange(1921,2015+1,1)
    timeexperi = ['1921-1968','1969-2015']
elif dataset_obs == 'ERA5_MEDS':
    yearsobs = np.arange(1979,2021+1,1)
    timeexperi = ['1979-2000','2001-2021']

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

### Get data
for m in range(len(monthlychoiceq)):
    monthlychoice = monthlychoiceq[m]

    lat_bounds,lon_bounds = UT.regions(reg_name)
    data_all,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds)
    data_obs_all,lats_obs,lons_obs = read_primary_dataset(variq,dataset_obs,monthlychoice,scenario,lat_bounds,lon_bounds)
    
    ### Prepare data for preprocessing
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
        data, data_obs = dSS.mask_CONUS(data,data_obs,'MEDS',lat_bounds,lon_bounds)
        print('*Removed everything by CONUS*')
        
    ### Select years
    if any([dataset_obs == 'NClimGrid_MEDS']):
        yearsq = np.where((years >= 1921) & (years <= 2021))[0]
        model = data[:,yearsq,:,:]
        
    elif any([dataset_obs == '20CRv3_MEDS']):
        yearsq = np.where((years >= 1921) & (years <= 2015))[0]
        model = data[:,yearsq,:,:]
        
    elif dataset_obs == 'ERA5_MEDS':
        yearsq = np.where((years >= 1979) & (years <= 2021))[0]
        model = data[:,yearsq,:,:]
        
    ### Calculate bias for two periods
    yearSplit = yearsobs.shape[0]//2
    bias1 = model[:,:yearSplit,:,:] - data_obs[:yearSplit,:,:]
    bias2 = model[:,yearSplit:,:,:] - data_obs[yearSplit:,:,:]
    
    ### Calculate ensemble and yearly mean 
    mean1 = np.nanmean(bias1,axis=(0,1))
    mean2 = np.nanmean(bias2,axis=(0,1))
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Plot biases
    fig = plt.figure(figsize=(8,4))
    
    label = r'\textbf{FLOR Bias -- %s minus %s [%s; $^{\circ}$C]}' % (dataset,dataset_obs,variq)
    if any([variq == 'TMIN',variq == 'TMAX']):
        limit = np.arange(-15,15.01,0.05)
        barlim = np.round(np.arange(-15,16,5),2)
    else:
        limit = np.arange(-5,5.01,0.05)
        barlim = np.round(np.arange(-5,5.1,1),2)
    
    plotdata = [mean1,mean2]
    plotlat = [lats,lats]
    plotlon = [lons,lons]
    
    for i in range(len(plotdata)):
        ax = plt.subplot(1,2,i+1)
        
        var = plotdata[i]
        lat1 = plotlat[i]
        lon1 = plotlon[i]
        
        m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                    projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='l',
                    area_thresh=10000)
        m.drawcoastlines(color='darkgrey',linewidth=1)
        m.drawstates(color='darkgrey',linewidth=0.5)
        m.drawcountries(color='darkgrey',linewidth=0.5)
    
        circle = m.drawmapboundary(fill_color='dimgrey',color='dimgrey',
                          linewidth=0.7)
        circle.set_clip_on(False)
            
        lon2,lat2 = np.meshgrid(lon1,lat1)
        
        cs1 = m.contourf(lon2,lat2,var,limit,extend='both',latlon=True)
        
        cs1.set_cmap(cmocean.cm.balance)
        
        plt.title(r'\textbf{%s for %s}' % (timeexperi[i],monthlychoice),fontsize=11,color='dimgrey')
        ax.annotate(r'\textbf{[%s]}' % (letters[i]),xy=(0,0),xytext=(0.0,1.03),
                  textcoords='axes fraction',color='k',fontsize=7,
                  rotation=0,ha='center',va='center')
        
    cbar_ax1 = fig.add_axes([0.305,0.14,0.4,0.03])                
    cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                        extend='max',extendfrac=0.07,drawedges=False)
    cbar1.set_label(label,fontsize=8,color='k',labelpad=5)  
    cbar1.set_ticks(barlim)
    cbar1.set_ticklabels(list(map(str,barlim)))
    cbar1.ax.tick_params(axis='x', size=.01,labelsize=7)
    cbar1.outline.set_edgecolor('dimgrey')
    plt.tight_layout()
            
    plt.savefig(directoryfigure + 'Bias_%s_USA_%s-%s_%s.png' % (variq,dataset,dataset_obs,monthlychoice),dpi=300)
    
