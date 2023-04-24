"""
Calculate example maps for ANN achitecture

Author    : Zachary M. Labe
Date      : 8 February 2022
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
import read_NClimGrid_monthlyMEDS as NC

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
variq = 'T2M'
slicenan = 'nan'
reg_name = 'US'
lat_bounds,lon_bounds = UT.regions(reg_name)
scenario = 'SSP585'
years = np.arange(1921,2021+1,1)

### Masking and preprocessing arguments
rm_annual_mean = False
rm_merid_mean = False
rm_ensemble_mean = False
land_only = False
ocean_only = False
CONUS_only = True

data = Dataset('/work/Zachary.Labe/Data/SPEAR/SPEAR_MED/monthly/T2M/T2M_01_1921-2010.nc')
lats = data.variables['lat'][:]
lons = data.variables['lon'][:]
var = data.variables['T2M'][:]
data.close()

### Read in obs
lat,lon,obs = NC.read_NClimGrid_monthlyMEDS(variq,'/work/Zachary.Labe/Data/','JJA',years,3,slicenan)

### Create mask
mask = obs.copy()
mask[np.where(np.isnan(mask))] = 0.
mask[np.where(mask != 0)] = 1.

### Only consider CONUS
varmask = var * mask[-1]
varmask[np.where(varmask == 0.)] = np.nan

### Plot example
example = varmask[300] - 273.15

###############################################################################
###############################################################################
###############################################################################
### Graphs
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
def setcolor(x, color):
      for m in x:
          for t in x[m][1]:
              t.set_color(color)
        
fig = plt.figure()
ax = plt.subplot(111)

var = example
limit = np.arange(10,36,1)
barlim = np.arange(10,36,10)

m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='l',
            area_thresh=10000)
m.drawcoastlines(color='k',linewidth=0.5)
m.drawstates(color='k',linewidth=1)
m.drawcountries(color='k',linewidth=0.5)

circle = m.drawmapboundary(fill_color='k',color='k',
                  linewidth=0.7)
circle.set_clip_on(False)
    
lon2,lat2 = np.meshgrid(lons,lats)

cs1 = m.pcolormesh(lon2,lat2,var,latlon=True)

cs1.set_cmap('twilight')
# m.drawlsmask(land_color=(0,0,0,0),ocean_color='k',lakes=False,zorder=11)

plt.tight_layout()
plt.savefig(directoryfigure + 'Map_ArchitectureExample_TREFHT_3.png',dpi=1000)
