"""
Plot GMST for the different SPEAR emission scenarios over the CONUS

Author     : Zachary M. Labe
Date       : 12 April 2023
Version    : 1 
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import cmocean
import cmasher as cmr
import numpy as np
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import scipy.stats as sts

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

### Directories
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/MS_Figures/' 

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
modelGCMs = ['SPEAR_MED_NATURAL','SPEAR_MED_Scenario','SPEAR_MED_Scenario','SPEAR_MED']
dataset_obs = 'NClimGrid_MEDS'
lenOfPicks = len(modelGCMs)
monthlychoice = 'JJA'
variq = 'T2M'
reg_name = 'US'
level = 'surface'
###############################################################################
###############################################################################
timeper = ['naturalforcing','futureforcing','futureforcing','futureforcing']
scenarioall = ['Natural','SSP119','SSP245','SSP585']
scenarioalln = ['Natural','SSP1-1.9','SSP2-4.5','Historical + SSP5-8.5']
###############################################################################
###############################################################################
rm_annual_mean = False
rm_merid_mean = False
rm_ensemble_mean = False
land_only = False
ocean_only = False
CONUS_only = True
###############################################################################
###############################################################################
baseline = np.arange(1981,2010+1,1)
###############################################################################
###############################################################################
window = 0
yearsall = [np.arange(1921+window,2100+1,1),np.arange(1921+window,2100+1,1),np.arange(1921+window,2100+1,1),
            np.arange(1921+window,2100+1,1)]
yearsobs = np.arange(1921+window,2022+1,1)
###############################################################################
###############################################################################
numOfEns = 30
lentime = len(yearsall)
###############################################################################
###############################################################################
lat_bounds,lon_bounds = UT.regions(reg_name)
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Read in model and observational/reanalysis data
def read_primary_dataset(variq,datasetname,monthlychoice,scenario,lat_bounds,lon_bounds):
    
    if any([datasetname == 'SPEAR_MED_shuffle_space',datasetname == 'SPEAR_MED_shuffle_time']):
        dataset_pick = 'SPEAR_MED'
    else:
        dataset_pick = datasetname
    
    data,lats,lons = df.readFiles(variq,dataset_pick,monthlychoice,scenario)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',datasetname,' is shaped',data.shape)
    return datar,lats,lons

### Loop in all climate models
data_allq = []
for no in range(len(modelGCMs)):
    dataset = modelGCMs[no]
    scenario = scenarioall[no]
    data_all,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds)
    data_obs_all,lats_obs,lons_obs = read_primary_dataset(variq,dataset_obs,monthlychoice,'historical',lat_bounds,lon_bounds)
    
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
    
    data_allq.append(data)
data = np.asarray(data_allq)

### Calculate historical baseline for calculating anomalies (and ensemble mean)
historical = data[1]
historicalyrs = yearsall[1]

yearhq = np.where((historicalyrs >= baseline.min()) & (historicalyrs <= baseline.max()))[0]
historicalc = np.nanmean(np.nanmean(historical[:,yearhq,:,:],axis=1),axis=0)

### Calculate anomalies
data_anom = []
for no in range(len(modelGCMs)):
    anomq = data[no] - historicalc[np.newaxis,np.newaxis,:,:]
    data_anom.append(anomq)

### Calculate CONUS average
lon2,lat2 = np.meshgrid(lons,lats)
aveall = []
maxens = []
minens = []
meanens = []
medianens = []
for no in range(len(modelGCMs)):
    aveallq = UT.calc_weightedAve(data_anom[no],lat2)

    maxensq = np.nanmax(aveallq,axis=0)
    minensq = np.nanmin(aveallq,axis=0)
    meanensq = np.nanmean(aveallq,axis=0)
    medianensq = np.nanmedian(aveallq,axis=0)
    
    aveall.append(aveallq)
    maxens.append(maxensq)
    minens.append(minensq)
    meanens.append(meanensq)
    medianens.append(medianensq)

yearhqo = np.where((yearsobs >= baseline.min()) & (yearsobs <= baseline.max()))[0]
climobs = np.nanmean(data_obs[yearhqo,:,:],axis=0)
anomobs = data_obs - climobs
aveobs = UT.calc_weightedAve(anomobs,lat2)

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

color = cmr.rainforest(np.linspace(0.00,0.8,len(aveall)))
for i,c in zip(range(len(aveall)),color): 
    if i == 0:
        c = 'dimgrey'
    plt.fill_between(x=yearsall[i],y1=minens[i],y2=maxens[i],facecolor=c,zorder=1,
             alpha=0.4,edgecolor='none',clip_on=False)
    plt.plot(yearsall[i],meanens[i],linestyle='-',linewidth=2,color=c,
             label=r'\textbf{%s}' % scenarioalln[i],zorder=2,clip_on=False)

plt.plot(yearsobs,aveobs,linestyle='--',linewidth=1,color='k',
         dashes=(1,0.5),clip_on=False,zorder=30,label=r'\textbf{NClimGrid}')

leg = plt.legend(shadow=False,fontsize=13,loc='upper center',
      bbox_to_anchor=(0.5,1.05),fancybox=True,ncol=2,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(-18,18.1,1),2),np.round(np.arange(-18,18.1,1),2))
plt.xlim([1920,2100])
plt.ylim([-3,8])

plt.ylabel(r'\textbf{TAVG Anomaly [$^{\circ}$C] Relative to 1981-2010}',
           fontsize=10,color='k')


### Save figure
plt.savefig(directoryfigure+'JJA_T-CONUS_1921-2022-FutureProjections.png',dpi=600)
