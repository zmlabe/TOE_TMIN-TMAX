"""
Make figure of LRP maps for manuscript

Author     : Zachary M. Labe
Date       : 3 March 2022
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

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

### Parameters
directorydata = '/Users/zlabe/Documents/Research/SolarIntervention/Data/'
variq = 'SST' 
directoryfigure = '/Users/zlabe/Desktop/SAI/LRP/%s/' % variq

### Read in LRP after training on WACCM
data = Dataset(directorydata + 'LRPMap_Z_Testing_WACCM_%s.nc' % variq)
lat = data.variables['lat'][:]
lon = data.variables['lon'][:]
lrp_waccm_zq = data.variables['LRP'][:]
data.close()

data = Dataset(directorydata + 'LRPMap_E_Testing_WACCM_%s.nc' % variq)
lrp_waccm_eq = data.variables['LRP'][:]
data.close()

data = Dataset(directorydata + 'LRPMap_IG_Testing_WACCM_%s.nc' % variq)
lrp_waccm_ig = data.variables['LRP'][:]
data.close()

###############################################################################
### Read in LRP after training on ARISE
data = Dataset(directorydata + 'LRPMap_Z_Testing_ARISE_%s.nc' % variq)
lrp_arise_zq = data.variables['LRP'][:]
data.close()

data = Dataset(directorydata + 'LRPMap_E_Testing_ARISE_%s.nc' % variq)
lrp_arise_eq = data.variables['LRP'][:]
data.close()

data = Dataset(directorydata + 'LRPMap_IG_Testing_ARISE_%s.nc' % variq)
lrp_arise_ig = data.variables['LRP'][:]
data.close()

### Take means across all years
lrp_waccm_z = np.nanmean(lrp_waccm_zq,axis=0)
lrp_waccm_e = np.nanmean(lrp_waccm_eq,axis=0)
lrp_waccm_ig = np.nanmean(lrp_waccm_ig,axis=0)
lrp_arise_z = np.nanmean(lrp_arise_zq,axis=0)
lrp_arise_e = np.nanmean(lrp_arise_eq,axis=0)
lrp_arise_ig = np.nanmean(lrp_arise_ig,axis=0)

###############################################################################
###############################################################################
###############################################################################
### Plot subplot of observations
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
limit = np.arange(0,0.81,0.005)
barlim = np.round(np.arange(0,0.81,0.1),2)
cmap = cm.cubehelix2_16.mpl_colormap
label = r'\textbf{LRP-testing [Relevance]}'

fig = plt.figure(figsize=(10,5))
###############################################################################
ax1 = plt.subplot(231)
m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.5)
    
### Variable
lrp_waccm_z = lrp_waccm_z/np.max(lrp_waccm_z)

lon = np.where(lon >180,lon-360,lon)
x, y = np.meshgrid(lon,lat)
   
circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.contourf(x,y,lrp_waccm_z,limit,extend='max',latlon=True)
cs1.set_cmap(cmap) 
   
plt.title(r'\textbf{LRP$_{z}$ Rule}',fontsize=20,color='dimgrey')
ax1.annotate(r'\textbf{[%s]}' % letters[0],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')
ax1.annotate(r'\textbf{Trained on WACCM}',xy=(0,0),xytext=(-0.04,0.5),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=90,ha='center',va='center')

###############################################################################
###############################################################################
###############################################################################
ax2 = plt.subplot(232)
m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.5)

### Variable
lrp_waccm_e = lrp_waccm_e/np.max(lrp_waccm_e)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.contourf(x,y,lrp_waccm_e,limit,extend='max',latlon=True)
cs1.set_cmap(cmap) 
 
plt.title(r'\textbf{LRP$_{\epsilon}$ Rule}',fontsize=20,color='dimgrey')
ax2.annotate(r'\textbf{[%s]}' % letters[1],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
ax1 = plt.subplot(233)
m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.5)
    
### Variable
lrp_waccm_ig = lrp_waccm_ig/np.max(lrp_waccm_ig)

lon = np.where(lon >180,lon-360,lon)
x, y = np.meshgrid(lon,lat)
   
circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.contourf(x,y,lrp_waccm_z,limit,extend='max',latlon=True)
cs1.set_cmap(cmap) 
   
plt.title(r'\textbf{Integrated Gradients}',fontsize=20,color='dimgrey')
ax1.annotate(r'\textbf{[%s]}' % letters[2],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
###############################################################################
###############################################################################
ax2 = plt.subplot(234)
m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.5)

### Variable
lrp_arise_z = lrp_arise_z/np.max(lrp_arise_z)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs2 = m.contourf(x,y,lrp_arise_z,limit,extend='max',latlon=True)
cs2.set_cmap(cmap) 
     
ax2.annotate(r'\textbf{[%s]}' % letters[3],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')
ax2.annotate(r'\textbf{Trained on ARISE}',xy=(0,0),xytext=(-0.04,0.5),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=90,ha='center',va='center')

###############################################################################
###############################################################################
###############################################################################
ax2 = plt.subplot(235)
m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.5)

### Variable
lrp_arise_e = lrp_arise_e/np.max(lrp_arise_e)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.contourf(x,y,lrp_arise_e,limit,extend='max',latlon=True)
cs1.set_cmap(cmap) 
     
ax2.annotate(r'\textbf{[%s]}' % letters[4],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
###############################################################################
###############################################################################
ax2 = plt.subplot(236)
m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.5)

### Variable
lrp_arise_ig = lrp_arise_ig/np.max(lrp_arise_ig)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.contourf(x,y,lrp_arise_ig,limit,extend='max',latlon=True)
cs1.set_cmap(cmap) 
     
ax2.annotate(r'\textbf{[%s]}' % letters[5],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
cbar_ax1 = fig.add_axes([0.395,0.1,0.2,0.025])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=10,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=7)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(hspace=-0.4)

plt.savefig(directoryfigure + 'PredictTheYear_LRPcomparison-Testing_%s.png' % variq,dpi=300)