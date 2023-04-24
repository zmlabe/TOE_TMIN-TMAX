"""
Plot biases for SPEAR_MED over CONUS

Author    : Zachary M. Labe
Date      : 14 August 2022
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import numpy as np
import cmocean
import palettable.cubehelix as cm
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import sys
import scipy.stats as sts
import calc_DetrendData as DT

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/Figures/Bias/' 

### Parameters
monthq = ['JAN','FEB','MAR','ARP','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
variq = 'TMIN'
monthlychoiceq = ['JFM','AMJ','JAS','OND','annual','JJA']
slicemonthnamen = ['JAN-MAR','APR-JUN','JUL-SEP','OND','Annual','JUN-AUG']
slicenan = 'nan'
reg_name = 'US'
lat_bounds,lon_bounds = UT.regions(reg_name)

dataset = 'SPEAR_MED'
dataset_obs = 'NClimGrid_MEDS'

### Select years of data
if dataset == 'SPEAR_MED':
    scenario = 'SSP585'
    years = np.arange(1921,2100+1,1)
elif dataset == 'SPEAR_MED_NATURAL':
    scenario = 'NATURAL'
    years = np.arange(1921,2100+1,1)
elif dataset == 'SPEAR_MED_Scenario':
    scenario = 'SSP245'
    years = np.arange(1921,2100+1,1)

if any([dataset_obs == 'NClimGrid_MEDS']):
    yearsobs = np.arange(1921,2021+1,1)
    timeexperi = ['1921-1971','1972-2021']
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
        
    ### Select years
    if any([dataset_obs == 'NClimGrid_MEDS']):
        yearsq = np.where((years >= 1921) & (years <= 2021))[0]
        model = data[:,yearsq,:,:]
        
    elif dataset_obs == 'ERA5_MEDS':
        yearsq = np.where((years >= 1979) & (years <= 2021))[0]
        model = data[:,yearsq,:,:]
        
    ### Calculate bias for two periods
    yearSplit = yearsobs.shape[0]//2
    bias1 = model[:,:yearSplit,:,:] - data_obs[:yearSplit,:,:]
    bias2 = model[:,yearSplit:,:,:] - data_obs[yearSplit:,:,:]
    
    ### Detrend data
    model_dt1 = DT.detrendData(model[:,:yearSplit,:,:],'surface','yearly')
    model_dt2 = DT.detrendData(model[:,yearSplit:,:,:],'surface','yearly')
    
    obs_dt1 = DT.detrendDataR(data_obs[:yearSplit,:,:],'surface','yearly')
    obs_dt2 = DT.detrendDataR(data_obs[yearSplit:,:,:],'surface','yearly')
    
    ### Calculate variance
    var_model1 = np.var(model_dt1[:,:,:,:],axis=1)
    var_model2 = np.var(model_dt2[:,:,:,:],axis=1)
    
    var_obs1 = np.var(obs_dt1[:,:,:],axis=0)
    var_obs2 = np.var(obs_dt2[:,:,:],axis=0)
    
    ### Calculate bias in variance
    bias_var1 = var_model1 - var_obs1
    bias_var2 = var_model2 - var_obs2
    
    ### Calculate ensemble mean bias
    bias_mean1 = np.nanmean(bias_var1,axis=0)
    bias_mean2 = np.nanmean(bias_var2,axis=0)
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Plot biases
    fig = plt.figure(figsize=(9,5))
    
    label1 = r'\textbf{Variance -- %s [%s; $^{\circ}$C$^{2}$]}' % (dataset,variq)
    label2 = r'\textbf{Bias -- %s minus %s [%s; $^{\circ}$C$^{2}$]}' % (dataset,dataset_obs,variq)
    limit1 = np.arange(0,8.1,0.1)
    barlim1 = np.arange(0,9,2)
    limit2 = np.arange(-5,5.01,0.05)
    barlim2 = np.round(np.arange(-5,5.1,1),2)
    
    plotdata = [np.nanmean(var_model1,axis=0),np.nanmean(var_model2,axis=0),bias_mean1,bias_mean2]
    cmaps = [cm.cubehelix2_16.mpl_colormap,cm.cubehelix2_16.mpl_colormap,
             cmocean.cm.balance,cmocean.cm.balance]
    plotlat = [lats,lats,lats,lats]
    plotlon = [lons,lons,lons,lons]
    
    for i in range(len(plotdata)):
        ax = plt.subplot(2,2,i+1)
        
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
        
        if i < 2:
            cs1 = m.contourf(lon2,lat2,var,limit1,extend='max',latlon=True)
            cs1.set_cmap(cmaps[i])
        else:
            cs2 = m.contourf(lon2,lat2,var,limit2,extend='both',latlon=True)
            cs2.set_cmap(cmaps[i])
        
        if i < 2:
            plt.title(r'\textbf{%s for %s}' % (timeexperi[i],monthlychoice),fontsize=11,color='dimgrey')
        ax.annotate(r'\textbf{[%s]}' % (letters[i]),xy=(0,0),xytext=(0.0,1.03),
                  textcoords='axes fraction',color='k',fontsize=7,
                  rotation=0,ha='center',va='center')
        
    cbar_ax1 = fig.add_axes([0.915,0.56,0.02,0.3])                
    cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='vertical',
                        extend='max',extendfrac=0.07,drawedges=False)
    cbar1.set_label(label1,fontsize=5,color='k',labelpad=5)  
    cbar1.set_ticks(barlim1)
    cbar1.set_ticklabels(list(map(str,barlim1)))
    cbar1.ax.tick_params(axis='y', size=.01,labelsize=5)
    cbar1.outline.set_edgecolor('dimgrey')
    
    cbar_ax2 = fig.add_axes([0.912,0.1,0.02,0.3])                
    cbar2 = fig.colorbar(cs2,cax=cbar_ax2,orientation='vertical',
                        extend='both',extendfrac=0.07,drawedges=False)
    cbar2.set_label(label2,fontsize=5,color='k',labelpad=5)  
    cbar2.set_ticks(barlim2)
    cbar2.set_ticklabels(list(map(str,barlim2)))
    cbar2.ax.tick_params(axis='y', size=.01,labelsize=5)
    cbar2.outline.set_edgecolor('dimgrey')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=-0.22)
            
    plt.savefig(directoryfigure + 'BiasVariance_%s_USA_%s-%s_%s.png' % (variq,dataset,dataset_obs,monthlychoice),dpi=300)
    
