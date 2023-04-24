"""
LRP for observations on SPEAR_MED to see if the regions change

Author     : Zachary M. Labe
Date       : 26 September 2022
Version    : 1
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import cmocean as cmocean
import cmasher as cmr
import calc_Utilities as UT
import calc_dataFunctions as df

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

### Parameters
directorydata = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/LRP/'
dataset_obs = 'NClimGrid_MEDS'
years = np.arange(1921,2021+1,1)
variq = 'T2M'
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/Figures/LRP/Obs/'
land_only = False
ocean_only = False
rm_annual_mean = False
rm_merid_mean = False
rm_ensemble_mean = False
random_network_seed = 87750
random_segment_seed = 71541
hiddensList = [[20,20]]
ridge_penalty = [0.01]
monthlychoiceq = ['JFM','AMJ','JAS','OND','annual','JJA']
monthlychoiceq = ['JJA']
reg_name = 'US'
classChunk = 10
lr_here = 0.01
batch_size = 32
iterations = [500]
NNType = 'ANN'
yrcase = 1992

def read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice,scenario)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons  

for mm in range(1):
    monthlychoice = monthlychoiceq[mm]
    ###############################################################################
    ### Read in LRP after training on SPEAR_MED
    modelType = 'TrainedOnSPEAR_MED'
    savename = modelType+'_Vari-'+variq+'_Obs-' + dataset_obs + '_Months-' + monthlychoice + '_L2-'+ str(ridge_penalty[0])+ '_LR-' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(iterations[0]) + '_' + NNType + str(hiddensList[0][0]) + 'x' + str(hiddensList[0][-1]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed)
    if rm_annual_mean == True:
        savename = savename + '_AnnualMeanRemoved' 
    if rm_ensemble_mean == True:
        savename = savename + '_EnsembleMeanRemoved' 
    if rm_merid_mean == True:
        savename = savename + '_MeridionalMeanRemoved' 
    if land_only == True: 
        savename = savename + '_LANDONLY'
    if ocean_only == True:
        savename = savename + '_OCEANONLY'
    
    data = Dataset(directorydata + 'LRPMap_Z_Obs' + '_' + variq + '_' + savename + '.nc')
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]                             
    lrp_SPEAR_zn = data.variables['LRP'][:].reshape(years.shape[0],lat.shape[0],lon.shape[0])
    data.close()
    
    data = Dataset(directorydata + 'LRPMap_E_Obs' + '_' + variq + '_' + savename + '.nc')
    lrp_SPEAR_en = data.variables['LRP'][:].reshape(years.shape[0],lat.shape[0],lon.shape[0])
    data.close()
    
    data = Dataset(directorydata + 'LRPMap_IG_Obs' + '_' + variq + '_' + savename + '.nc')
    lrp_SPEAR_ign = data.variables['LRP'][:].reshape(years.shape[0],lat.shape[0],lon.shape[0])
    data.close()
    
    ### Read in Observations
    lat_bounds,lon_bounds = UT.regions(reg_name)
    obs,lats,lons = read_primary_dataset(variq,'NClimGrid_MEDS',monthlychoice,'SSP585',lat_bounds,lon_bounds)
    
    ### Calculate anomalies
    yearq = np.where((years >= 1981) & (years <= 2010))[0]
    clim = np.nanmean(obs[yearq,:,:],axis=0)
    anom = obs - clim
    
    ### Pick case year
    yrcaseq = np.where((years == yrcase))[0]
    
    anomyr = anom[yrcaseq,:,:].squeeze()
    lrp_SPEAR_z = lrp_SPEAR_zn[yrcaseq,:,:].squeeze()
    lrp_SPEAR_e = lrp_SPEAR_en[yrcaseq,:,:].squeeze()
    lrp_SPEAR_ig = lrp_SPEAR_ign[yrcaseq,:,:].squeeze()
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Plot subplot of observations
    letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
    limit = np.arange(0,0.41,0.005)
    barlim = np.round(np.arange(0,0.41,0.1),2)
    cmap = cm.cubehelix2_16.mpl_colormap
    label = r'\textbf{%s-%s -- LRP-NClimGrid [Relevance]}' % (monthlychoice,variq)
    
    fig = plt.figure(figsize=(10,5))
    ###############################################################################
    ax1 = plt.subplot(231)
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='l')
    m.drawcoastlines(color='dimgrey',linewidth=1)
    m.drawstates(color='dimgrey',linewidth=0.5)
    m.drawcountries(color='dimgrey',linewidth=0.5)
        
    ### Variable
    lrp_SPEAR_z = lrp_SPEAR_z/np.max(lrp_SPEAR_z)
    
    x, y = np.meshgrid(lon,lat)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)
    
    cs1 = m.contourf(x,y,lrp_SPEAR_z,limit,extend='max',latlon=True)
    cs1.set_cmap(cmap) 
       
    plt.title(r'\textbf{LRP$_{z}$ Rule}',fontsize=20,color='dimgrey')
    ax1.annotate(r'\textbf{[%s]}' % letters[0],xy=(0,0),xytext=(0.98,1.05),
                  textcoords='axes fraction',color='k',fontsize=9,
                  rotation=0,ha='center',va='center')
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ax2 = plt.subplot(232)
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='l')
    m.drawcoastlines(color='dimgrey',linewidth=1)
    m.drawstates(color='dimgrey',linewidth=0.5)
    m.drawcountries(color='dimgrey',linewidth=0.5)
    
    ### Variable
    lrp_SPEAR_e = lrp_SPEAR_e/np.max(lrp_SPEAR_e)
    
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)
    
    cs1 = m.contourf(x,y,lrp_SPEAR_e,limit,extend='max',latlon=True)
    cs1.set_cmap(cmap) 
     
    plt.title(r'\textbf{LRP$_{\epsilon}$ Rule}',fontsize=20,color='dimgrey')
    ax2.annotate(r'\textbf{[%s]}' % letters[1],xy=(0,0),xytext=(0.98,1.05),
                  textcoords='axes fraction',color='k',fontsize=9,
                  rotation=0,ha='center',va='center')
    
    ###############################################################################
    ax1 = plt.subplot(233)
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='l')
    m.drawcoastlines(color='dimgrey',linewidth=1)
    m.drawstates(color='dimgrey',linewidth=0.5)
    m.drawcountries(color='dimgrey',linewidth=0.5)
        
    ### Variable
    lrp_SPEAR_ig = lrp_SPEAR_ig/np.max(lrp_SPEAR_ig)
    
    x, y = np.meshgrid(lon,lat)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)
    
    cs1 = m.contourf(x,y,lrp_SPEAR_ig,limit,extend='max',latlon=True)
    cs1.set_cmap(cmap) 
       
    plt.title(r'\textbf{Integrated Gradients}',fontsize=20,color='dimgrey')
    ax1.annotate(r'\textbf{[%s]}' % letters[2],xy=(0,0),xytext=(0.98,1.05),
                  textcoords='axes fraction',color='k',fontsize=9,
                  rotation=0,ha='center',va='center')
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    limita = np.arange(-6,6.01,0.01)
    barlima = np.round(np.arange(-6,7,2),2)
    cmapa = cmocean.cm.balance
    labela = r'\textbf{%s-%s -- Anomaly [$^{\circ}$C]}' % (monthlychoice,variq)
    
    ax2 = plt.subplot(234)
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='l')
    m.drawcoastlines(color='dimgrey',linewidth=1)
    m.drawstates(color='dimgrey',linewidth=0.5)
    m.drawcountries(color='dimgrey',linewidth=0.5)
    
    ### Variable
    
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)
    
    csa = m.contourf(x,y,anomyr,limita,extend='both',latlon=True)
    csa.set_cmap(cmapa) 
         
    ax2.annotate(r'\textbf{[%s]}' % letters[3],xy=(0,0),xytext=(0.98,1.05),
                  textcoords='axes fraction',color='k',fontsize=9,
                  rotation=0,ha='center',va='center')
    ax2.annotate(r'\textbf{%s}' % yrcase,xy=(0,0),xytext=(-0.12,1.1),
                  textcoords='axes fraction',color='k',fontsize=40,
                  rotation=90,ha='center',va='center')
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ax2 = plt.subplot(235)
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='l')
    m.drawcoastlines(color='dimgrey',linewidth=1)
    m.drawstates(color='dimgrey',linewidth=0.5)
    m.drawcountries(color='dimgrey',linewidth=0.5)
    
    ### Variable
    
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)
    
    csa = m.contourf(x,y,anomyr,limita,extend='both',latlon=True)
    csa.set_cmap(cmapa) 
         
    ax2.annotate(r'\textbf{[%s]}' % letters[3],xy=(0,0),xytext=(0.98,1.05),
                  textcoords='axes fraction',color='k',fontsize=9,
                  rotation=0,ha='center',va='center')
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ax2 = plt.subplot(236)
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='l')
    m.drawcoastlines(color='dimgrey',linewidth=1)
    m.drawstates(color='dimgrey',linewidth=0.5)
    m.drawcountries(color='dimgrey',linewidth=0.5)
    
    ### Variable
    
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)
    
    csa = m.contourf(x,y,anomyr,limita,extend='both',latlon=True)
    csa.set_cmap(cmapa) 
         
    ax2.annotate(r'\textbf{[%s]}' % letters[3],xy=(0,0),xytext=(0.98,1.05),
                  textcoords='axes fraction',color='k',fontsize=9,
                  rotation=0,ha='center',va='center')
    
    ###############################################################################
    cbar_ax1 = fig.add_axes([0.93,0.56,0.01,0.2])                
    cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='vertical',
                        extend='max',extendfrac=0.07,drawedges=False)
    cbar1.set_label(label,fontsize=6,color='dimgrey',labelpad=3)  
    cbar1.set_ticks(barlim)
    cbar1.set_ticklabels(list(map(str,barlim)))
    cbar1.ax.tick_params(axis='y', size=.01,labelsize=7)
    cbar1.outline.set_edgecolor('dimgrey')
    
    cbar_axa = fig.add_axes([0.93,0.23,0.01,0.2])                
    cbara = fig.colorbar(csa,cax=cbar_axa,orientation='vertical',
                        extend='both',extendfrac=0.07,drawedges=False)
    cbara.set_label(labela,fontsize=6,color='dimgrey',labelpad=3)  
    cbara.set_ticks(barlima)
    cbara.set_ticklabels(list(map(str,barlima)))
    cbara.ax.tick_params(axis='y', size=.01,labelsize=7)
    cbara.outline.set_edgecolor('dimgrey')
    
    # plt.tight_layout()
    plt.subplots_adjust(hspace=-0.2)
    
    plt.savefig(directoryfigure + 'PredictTheYear_LRPcomparison-NClimGrid_%s_%s_CaseStudy-%s.png' % (variq,monthlychoice,yrcase),dpi=300)
