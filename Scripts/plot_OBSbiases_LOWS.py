"""
Plot differences in each observational product

Author    : Zachary M. Labe
Date      : 14 August 2022
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
import read_ERA5_monthlyLOWS as ER
import read_NClimGrid_monthlyLOWS as NC
import read_20CRv3_monthlyLOWS as CR

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/Figures/Bias_Obs/' 

### Parameters
monthq = ['JAN','FEB','MAR','ARP','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
variq = 'T2M'
monthlychoiceq = ['JFM','AMJ','JAS','OND','annual','JJA']
slicemonthnamen = ['JAN-MAR','APR-JUN','JUL-SEP','OND','Annual','JUN-AUG']
sliceperiod = 'JJA'
slicenan = 'nan'
sliceshape = 3 
addclimo = True
level = 'surface'
yearsER = np.arange(1979,2021+1,1)
yearsNC = np.arange(1921,2021+1,1)
yearsCR = np.arange(1921,2015+1,1)

for m in range(len(monthlychoiceq)):
    sliceperiod = monthlychoiceq[m]
    
    ### Read in observations
    if variq == 'T2M':
        lat,lon,lev,eran = ER.read_ERA5_monthlyLOWS(variq,'/work/Zachary.Labe/Data/',sliceperiod,
                                                   np.arange(1979,2021+1,1),sliceshape,addclimo,
                                                   slicenan,level)
        lat,lon,nclimn = NC.read_NClimGrid_monthlyLOWS(variq,'/work/Zachary.Labe/Data/',sliceperiod,
                                                    np.arange(1895,2021+1,1),sliceshape,slicenan)
        lat,lon,twentyn = CR.read_20CRv3_monthlyLOWS(variq,'/work/Zachary.Labe/Data/20CRv3_LOWS/',sliceperiod,
                                              np.arange(1836,2015+1,1),sliceshape,slicenan)
        
        ### Compare on same years
        years = np.arange(1979,2015+1,1)
        yearqER = np.where((yearsER >= 1979) & (yearsER <= 2015))[0]
        yearqNC = np.where((yearsNC >= 1979) & (yearsNC <= 2015))[0]
        yearqCR = np.where((yearsCR >= 1979) & (yearsCR <= 2015))[0]
        
        era = eran[yearqER,:,:]
        ncl = nclimn[yearqNC,:,:]
        twe = twentyn[yearqCR,:,:]
    
        ### Calculate differences
        diff1 = np.nanmean(ncl - era,axis=0)
        diff2 = np.nanmean(ncl - twe,axis=0)
        diff3 = np.nanmean(era - twe,axis=0)
        
        ### Create mask
        mask = nclimn[-1]
        mask[np.where(np.isnan(mask))] = 0.
        mask[np.where(mask != 0)] = 1.
        
        ### Mask differences
        diff1m = diff1 * mask
        diff1m[np.where(diff1m == 0.)] = np.nan
        diff2m = diff2 * mask
        diff2[np.where(diff2m == 0.)] = np.nan
        diff3m = diff3 * mask
        diff3m[np.where(diff3m == 0.)] = np.nan
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Plot biases
        fig = plt.figure(figsize=(8,4))
        
        plotdata = [diff1m,diff2m,diff3m]
        diffname = ['NClimGrid minus ERA5','NClimGrid minus 20CRv3','ERA5 minus 20CRv3']
        plotlat = [lat,lat,lat]
        plotlon = [lon,lon,lon]
        
        label = r'\textbf{Difference [%s; $^{\circ}$C]}' % (variq)
        limit = np.arange(-5,5.01,0.05)
        barlim = np.round(np.arange(-5,5.1,1),2)
        
        for i in range(len(plotdata)):
            ax = plt.subplot(1,3,i+1)
            
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
            # cs1 = m.pcolormesh(lon2,lat2,var,vmin=-5,vmax=5,latlon=True)
            
            cs1.set_cmap(cmocean.cm.balance)
            
            plt.title(r'\textbf{%s for %s}' % (diffname[i],sliceperiod),fontsize=11,color='dimgrey')
            ax.annotate(r'\textbf{[%s]}' % (letters[i]),xy=(0,0),xytext=(-0.04,1.03),
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
                
        plt.savefig(directoryfigure + 'ObsBias_ERA5_%s_USA_%s_LOWSres.png' % (variq,sliceperiod),dpi=300)
    
    ###############################################################################
    ###############################################################################
    ############################################################################### 
    ###############################################################################
    ###############################################################################
    ############################################################################### 
    ###############################################################################
    ###############################################################################
    ###############################################################################    
    ### Read in observations
    elif any([variq == 'TMAX',variq == 'TMIN']):
        lat,lon,nclimn = NC.read_NClimGrid_monthlyLOWS(variq,'/work/Zachary.Labe/Data/',sliceperiod,
                                                    np.arange(1895,2021+1,1),sliceshape,slicenan)
        lat,lon,twentyn = CR.read_20CRv3_monthlyLOWS(variq,'/work/Zachary.Labe/Data/20CRv3_LOWS/',sliceperiod,
                                              np.arange(1836,2015+1,1),sliceshape,slicenan)
        
        ### Compare on same years
        years = np.arange(1921,2015+1,1)
        yearqNC = np.where((yearsNC >= 1921) & (yearsNC <= 2015))[0]
        yearqCR = np.where((yearsCR >= 1921) & (yearsCR <= 2015))[0]
        
        ncl = nclimn[yearqNC,:,:]
        twe = twentyn[yearqCR,:,:]
    
        ### Calculate differences
        splitperiod = years.shape[0]//2
        diff1 = np.nanmean(ncl[:splitperiod,:,:] - twe[:splitperiod,:,:],axis=0)
        diff2 = np.nanmean(ncl[splitperiod:,:,:] - twe[splitperiod:,:,:],axis=0)
        
        ### Create mask
        mask = nclimn[-1]
        mask[np.where(np.isnan(mask))] = 0.
        mask[np.where(mask != 0)] = 1.
        
        ### Mask differences
        diff1m = diff1 * mask
        diff1m[np.where(diff1m == 0.)] = np.nan
        diff2m = diff2 * mask
        diff2[np.where(diff2m == 0.)] = np.nan
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Plot biases
        fig = plt.figure(figsize=(8,4))
        
        plotdata = [diff1m,diff2m]
        diffname = ['1921-1968','1969-2015']
        plotlat = [lat,lat]
        plotlon = [lon,lon]
        
        label = r'\textbf{NClimGrid minus 20CRv3 [%s; $^{\circ}$C]}' % (variq)
        limit = np.arange(-5,5.01,0.05)
        barlim = np.round(np.arange(-5,5.1,1),2)
        
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
            
            plt.title(r'\textbf{%s for %s}' % (diffname[i],sliceperiod),fontsize=11,color='dimgrey')
            ax.annotate(r'\textbf{[%s]}' % (letters[i]),xy=(0,0),xytext=(-0.04,1.03),
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
                
        plt.savefig(directoryfigure + 'ObsBias_%s_USA_%s_LOWSres.png' % (variq,sliceperiod),dpi=300)
        
        
