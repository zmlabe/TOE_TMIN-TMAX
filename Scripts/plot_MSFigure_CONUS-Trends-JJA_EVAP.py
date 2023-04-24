"""
Plot trends for different SPEAR runs

Author    : Zachary M. Labe
Date      : 9 August 2022
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import numpy as np
import cmocean
import calc_Utilities as UT
import sys
import itertools
import read_SPEAR_MED as SP
import read_SPEAR_MED_NATURAL as SPNO
import read_SPEAR_MED_Scenario as S45
import scipy.stats as sts

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/MS_Figures/' 

### Parameters
monthq = ['JAN','FEB','MAR','ARP','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
experi1 = ['SPEAR_MED','SPEAR_MED_NATURAL']
variq = 'EVAP'
sliceperiod = 'JJA'
yearssat = np.arange(1921,2022+1,1)
slicenan = 'nan'
datareader = True

### Trend time frame
yearmin = 1921
yearmax = 1989
trendyears = np.arange(yearmin,yearmax+1,1)

# ### Read data
# if datareader == True:
#     lat1s,lon1s,spear = SP.read_SPEAR_MED('/work/Zachary.Labe/Data/SPEAR/SPEAR_MED/monthly/',variq,
#                                             sliceperiod,4,
#                                             slicenan,30,'historical')
#     lat1s,lon1s,spearno = SPNO.read_SPEAR_MED_NATURAL('/work/Zachary.Labe/Data/SPEAR/SPEAR_MED_NATURAL/monthly/',variq,
#                                             sliceperiod,4,
#                                             slicenan,30,'historical')
   
# ### Calculate anomalies for a 1981-2010 baseline 
# def calc_anomalies(years,data):
#     """ 
#     Calculate anomalies
#     """
    
#     ### Baseline - 1981-2010
#     if data.ndim == 3:
#         yearqold = np.where((years >= 1981) & (years <= 2010))[0]
#         climold = np.nanmean(data[yearqold,:,:],axis=0)
#         anoms = data - climold
#     elif data.ndim == 4:
#         yearqold = np.where((years >= 1981) & (years <= 2010))[0]
#         climold = np.nanmean(data[:,yearqold,:,:],axis=1)
#         anoms = data - climold[:,np.newaxis,:,:]
    
#     return anoms

# spear_anom = calc_anomalies(yearssat,spear)
# spear_natural_anom = calc_anomalies(yearssat,spearno)

# ### Calculate trends
# trend_spear = UT.linearTrend(spear_anom,yearssat,'surface',yearmin,yearmax)*10.
# trend_natural_spear = UT.linearTrend(spear_natural_anom,yearssat,'surface',yearmin,yearmax)*10.

# ### Calculate ensemble means
# trend_spearm = np.nanmean(trend_spear,axis=0)
# trend_natural_spearm = np.nanmean(trend_natural_spear,axis=0)

###############################################################################
###############################################################################
###############################################################################
### Plot subplot of different AMIPS to compare with SPEAR
fig = plt.figure(figsize=(8,4))

label = r'\textbf{Evaporation Trend [mm/day/decade] for %s-%s}' % (yearmin,yearmax)
limit = np.arange(-0.3,0.31,0.01)
barlim = np.round(np.arange(-0.3,0.31,0.1),2)

plotdata = [trend_spearm,trend_natural_spearm]
plotlat = [lat1s,lat1s]
plotlon = [lon1s,lon1s]

for i in range(len(plotdata)):
    ax = plt.subplot(1,2,i+1)
    
    var = plotdata[i]
    lat1 = plotlat[i]
    lon1 = plotlon[i]
    
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='l')
    m.drawcoastlines(color='dimgrey',linewidth=1)
    m.drawstates(color='dimgrey',linewidth=0.5)
    m.drawcountries(color='dimgrey',linewidth=1)
    

    circle = m.drawmapboundary(fill_color='white',color='dimgrey',
                      linewidth=0.7)
    circle.set_clip_on(False)
        
    lon2,lat2 = np.meshgrid(lon1,lat1)
    
    cs1 = m.contourf(lon2,lat2,var,limit,extend='both',latlon=True)
    
    cs1.set_cmap(cmocean.cm.curl_r)
    
    plt.title(r'\textbf{%s}' % experi1[i],fontsize=15,color='dimgrey')
    ax.annotate(r'\textbf{[%s]}' % (letters[i]),xy=(0,0),xytext=(0.0,1.07),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')
    
cbar_ax1 = fig.add_axes([0.305,0.13,0.4,0.03])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=7,color='k',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=7)
cbar1.outline.set_edgecolor('dimgrey')
plt.tight_layout()
        
plt.savefig(directoryfigure + 'Trend_%s_USA_SPEARcompare_%s_%s-%s.png' % (variq,sliceperiod,yearmin,yearmax),dpi=1000)
