"""
Evaluate land properties over western CO
 
Author    : Zachary M. Labe
Date      : 6 March 2023
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
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/Figures/LandProperties/' 

### Parameters
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
slicemonthn = ['JFM','AMJ','JAS','OND','annual','JJA']
slicemonthnamen = ['JAN-MAR','APR-JUN','JUL-SEP','OND','Annual','JUN-AUG']
# slicemonthn = ['JJA']
# slicemonthnamen = ['JUN-AUG']
# slicemonthn = ['AMJ']
# slicemonthnamen = ['APR-JUN']
# slicemonthn = ['JFM']
# slicemonthnamen = ['JAN-MAR']
slicenan = 'nan'
variq1 = 'frac_crop'
variq2 = 'frac_ntrl'
variq3 = 'frac_past'
variq4 = 'frac_scnd'

years = np.arange(1979,2021+1,1)
yearsmodels = np.arange(1921,2100+1,1)

def readData(variq,slicemonth,slicenan,years):
    ### Read in mask
    lattemp,lontemp,obstemp = NC.read_NClimGrid_monthlyMEDS('T2M','/work/Zachary.Labe/Data/',slicemonth,years,3,slicenan)
    
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
    
    ### Calculate box over western CO
    latq = np.where((lat >= 37) & (lat <= 41))[0]
    lonq = np.where((lon >= (360-108)) & ((lon <= 360-105)))[0]
    latnew = lat[latq]
    lonnew = lon[lonq]
    
    spearanom1 = varmask[:,:,latq,:]
    spearanom = spearanom1[:,:,:,lonq]
    
    natanom1 = natmask[:,:,latq,:]
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
    m.contourf(x,y,spearanom[0,0],300,extend='both',latlon=True)
    
    ### Calculate CONUS averages
    lon2,lat2 = np.meshgrid(lonnew,latnew)
    spear_ave = UT.calc_weightedAve(spearanom,lat2)
    natural_ave = UT.calc_weightedAve(natanom,lat2)
    
    return spear_ave,natural_ave

for i in range(len(slicemonthn)):
    slicemonth = slicemonthn[i]
    slicemonthname = slicemonthnamen[i]
    
    ### Return data
    spear_crop, natural_crop = readData(variq1,slicemonth,slicenan,years)
    spear_ntrl, natural_ntrl = readData(variq2,slicemonth,slicenan,years)
    spear_past, natural_past = readData(variq3,slicemonth,slicenan,years)
    spear_scnd, natural_scnd = readData(variq4,slicemonth,slicenan,years)
    
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
    
    plt.plot(yearsmodels,np.nanmean(spear_crop,axis=0),linestyle='-',linewidth=3,color='deepskyblue',
              label=r'\textbf{CROP}')
    plt.plot(yearsmodels,np.nanmean(natural_crop,axis=0),linestyle='--',linewidth=3,color='deepskyblue',
             dashes=(1,0.3))
    
    plt.plot(yearsmodels,np.nanmean(spear_past,axis=0),linestyle='-',linewidth=3,color='darkblue',
              label=r'\textbf{PASTURE}')
    plt.plot(yearsmodels,np.nanmean(natural_past,axis=0),linestyle='--',linewidth=3,color='darkblue',
             dashes=(1,0.3))
    
    plt.plot(yearsmodels,np.nanmean(spear_scnd,axis=0),linestyle='-',linewidth=3,color='crimson',
              label=r'\textbf{SECONDARY}')
    plt.plot(yearsmodels,np.nanmean(natural_scnd,axis=0),linestyle='--',linewidth=3,color='crimson',
             dashes=(1,0.3))
    
    plt.plot(yearsmodels,np.nanmean(spear_ntrl,axis=0),linestyle='-',linewidth=3,color='darkred',
              label=r'\textbf{NATURAL}')
    plt.plot(yearsmodels,np.nanmean(natural_ntrl,axis=0),linestyle='--',linewidth=3,color='darkred',
             dashes=(1,0.3))
    
    leg = plt.legend(shadow=False,fontsize=8,loc='upper center',
          bbox_to_anchor=(0.5,1.01),fancybox=True,ncol=4,frameon=False,
          handlelength=1,handletextpad=0.5)
    
    plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
    plt.yticks(np.round(np.arange(0,1.01,0.1),2),np.round(np.arange(0,1.01,0.1),2))
    plt.xlim([1920,2100])
    plt.ylim([0,0.8])
    
    plt.title(r'\textbf{%s: Western CO}' % slicemonthname,
                        color='k',fontsize=17)
    plt.ylabel(r'\textbf{Mean Fraction of Land}',color='dimgrey',fontsize=10)
    
    plt.tight_layout()        
   
    ### Save figure
    plt.savefig(directoryfigure+'%s_LandProperties-WesternCO_1921-2100.png' % (slicemonth),dpi=300)
    
    ###############################################################################
    ###############################################################################
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
    
    plt.plot(yearsmodels,np.nanmean(spear_crop,axis=0),linestyle='-',linewidth=3,color='deepskyblue',
              label=r'\textbf{CROP}')
    plt.plot(yearsmodels,np.nanmean(natural_crop,axis=0),linestyle='--',linewidth=3,color='deepskyblue',
             dashes=(1,0.3))
    
    leg = plt.legend(shadow=False,fontsize=8,loc='upper center',
          bbox_to_anchor=(0.5,1.01),fancybox=True,ncol=4,frameon=False,
          handlelength=1,handletextpad=0.5)
    
    plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
    plt.yticks(np.round(np.arange(0,1.01,0.1),2),np.round(np.arange(0,1.01,0.1),2))
    plt.xlim([1920,1990])
    plt.ylim([0,0.2])
    
    plt.title(r'\textbf{%s: Western CO}' % slicemonthname,
                        color='k',fontsize=17)
    plt.ylabel(r'\textbf{Mean Fraction of Land}',color='dimgrey',fontsize=10)
    
    plt.tight_layout()        
    
    ### Save figure
    plt.savefig(directoryfigure+'%s_LandPropertiesCROP-WesternCO_1921-1990.png' % (slicemonth),dpi=300)
    
    ###############################################################################
    ###############################################################################
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
    
    plt.plot(yearsmodels,np.nanmean(spear_past,axis=0),linestyle='-',linewidth=3,color='darkblue',
              label=r'\textbf{PASTURE}')
    plt.plot(yearsmodels,np.nanmean(natural_past,axis=0),linestyle='--',linewidth=3,color='darkblue',
             dashes=(1,0.3))
    
    leg = plt.legend(shadow=False,fontsize=8,loc='upper center',
          bbox_to_anchor=(0.5,1.01),fancybox=True,ncol=4,frameon=False,
          handlelength=1,handletextpad=0.5)
    
    plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
    plt.yticks(np.round(np.arange(0,1.01,0.1),2),np.round(np.arange(0,1.01,0.1),2))
    plt.xlim([1920,1990])
    plt.ylim([0.1,0.3])
    
    plt.title(r'\textbf{%s: Western CO}' % slicemonthname,
                        color='k',fontsize=17)
    plt.ylabel(r'\textbf{Mean Fraction of Land}',color='dimgrey',fontsize=10)
    
    plt.tight_layout()        
    
    ### Save figure
    plt.savefig(directoryfigure+'%s_LandPropertiesPASTURE-WesternCO_1921-1990.png' % (slicemonth),dpi=300)
    
    ###############################################################################
    ###############################################################################
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
    
    plt.plot(yearsmodels,np.nanmean(spear_scnd,axis=0),linestyle='-',linewidth=3,color='crimson',
              label=r'\textbf{SECONDARY}')
    plt.plot(yearsmodels,np.nanmean(natural_scnd,axis=0),linestyle='--',linewidth=3,color='crimson',
             dashes=(1,0.3))
    
    leg = plt.legend(shadow=False,fontsize=8,loc='upper center',
          bbox_to_anchor=(0.5,1.01),fancybox=True,ncol=4,frameon=False,
          handlelength=1,handletextpad=0.5)
    
    plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
    plt.yticks(np.round(np.arange(0,1.01,0.1),2),np.round(np.arange(0,1.01,0.1),2))
    plt.xlim([1920,1990])
    plt.ylim([0,0.3])
    
    plt.title(r'\textbf{%s: Western CO}' % slicemonthname,
                        color='k',fontsize=17)
    plt.ylabel(r'\textbf{Mean Fraction of Land}',color='dimgrey',fontsize=10)
    
    plt.tight_layout()        
    
    ### Save figure
    plt.savefig(directoryfigure+'%s_LandPropertiesSECONDARY-WesternCO_1921-1990.png' % (slicemonth),dpi=300)
    
    ###############################################################################
    ###############################################################################
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
    
    plt.plot(yearsmodels,np.nanmean(spear_ntrl,axis=0),linestyle='-',linewidth=3,color='darkred',
              label=r'\textbf{NATURAL}')
    plt.plot(yearsmodels,np.nanmean(natural_ntrl,axis=0),linestyle='--',linewidth=3,color='darkred',
             dashes=(1,0.3))
    
    leg = plt.legend(shadow=False,fontsize=8,loc='upper center',
          bbox_to_anchor=(0.5,1.01),fancybox=True,ncol=4,frameon=False,
          handlelength=1,handletextpad=0.5)
    
    plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
    plt.yticks(np.round(np.arange(0,1.01,0.1),2),np.round(np.arange(0,1.01,0.1),2))
    plt.xlim([1920,1990])
    plt.ylim([0.4,0.75])
    
    plt.title(r'\textbf{%s: Western CO}' % slicemonthname,
                        color='k',fontsize=17)
    plt.ylabel(r'\textbf{Mean Fraction of Land}',color='dimgrey',fontsize=10)
    
    plt.tight_layout()        
    
    ### Save figure
    plt.savefig(directoryfigure+'%s_LandPropertiesNATURAL-WesternCO_1921-1990.png' % (slicemonth),dpi=300)
