"""
Plot TOE for observations using the predict the year approach after 
different samples for SPEAR LOW

Author     : Zachary M. Labe
Date       : 13 February 2023
Version    : 1
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import palettable.wesanderson as ww
import palettable.cubehelix as cm
import cmocean as cmocean
import cmasher as cmr
import calc_Utilities as UT
import calc_dataFunctions as df
import scipy.stats as sts

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

### Parameters
directorydata = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/Predictions/LoopFinal/Seasons/'
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/Figures/TOE_ANN/Seasons/'
variqq = ['T2M']
land_only = False
ocean_only = False
rm_annual_mean = False
rm_merid_mean = False
rm_ensemble_mean = False
monthlychoiceq = ['DJF','MAM','JJA','SON']
reg_name = 'US'
resolution = 'LOWS'
NNType = 'ANN'
dataset = 'SPEAR_LOW'
dataset_obs = 'NClimGrid_LOWS'
SAMPLES = 100

if reg_name == 'US':
    debug = True
    NNType = 'ANN'
    annType = 'class'
    classChunkHalf = 5
    classChunk = 10
    biasBool = False
    if resolution == 'MEDS':
        hiddensList = [[10,10,10]]
    elif resolution == 'LOWS':
        hiddensList = [[20,20,20]]
    elif resolution == 'HIGHS':
        hiddensList = [[20,20,20]]
    else:
        print(ValueError('This is the wrong resolution!!!'))
        sys.exit()
    ridge_penalty = [0.001]
    actFun = 'relu'
    iterations = [500]
elif any([reg_name=='W_US',reg_name=='Ce_US',reg_name=='E_US']):
    debug = True
    NNType = 'ANN'
    annType = 'class'
    classChunkHalf = 5
    classChunk = 10
    biasBool = False
    hiddensList = [[100,100]]
    ridge_penalty = [0.001]
    actFun = 'relu'
    iterations = [500]
else:
    print(ValueError('This is the wrong region name selected!!!'))
    sys.exit()

### Save TOE calculations
for sea in range(len(monthlychoiceq)):
    monthlychoice = monthlychoiceq[sea]
    directorytoe = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/Predictions/ToeFinal/'
    toe = np.load(directorytoe + 'TOE_v3-LoopFinal-ObsPred_Seasons_%s-%s_%s_SAMPLES-%s.npz' % (dataset,dataset_obs,reg_name,SAMPLES))
    skill = np.load(directorytoe + 'SpearmanCorr_v3-LoopFinal-ObsPred_Seasons_%s-%s_%s_SAMPLES-%s.npz' % (dataset,dataset_obs,reg_name,SAMPLES))
    
    plot_toe = toe['toe_annall'][sea,:].squeeze()
    plot_corr = skill['cor_annall'][sea,:].squeeze()
    
    ### Percentiles
    minPER = 5
    maxPER = 95
    
    mean_toe = np.nanmean(plot_toe)
    median_toe = np.nanmedian(plot_toe)
    toe_max = np.nanpercentile(plot_toe,minPER)
    toe_min = np.nanpercentile(plot_toe,maxPER)
    
    mean_cor = np.nanmean(plot_corr)
    median_cor = np.nanmedian(plot_corr)
    cor_max = np.nanpercentile(plot_corr,minPER)
    cor_min = np.nanpercentile(plot_corr,maxPER)
    
    ###############################################################################
    ###############################################################################
    ###############################################################################    
    ### Create graph 
    plt.rc('text',usetex=True)
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
    
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
            
    c2=ww.FantasticFox2_5.mpl_colormap
    
    ###############################################################################
    ###############################################################################
    ############################################################################### 
    ### Training figure
    fig = plt.figure(figsize=(8,4))
    ax = plt.subplot(121)
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both',length=4,width=2,which='major',color='dimgrey',
                   labelbottom='off',bottom='off')
    ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
    ax.xaxis.grid(zorder=1,color='darkgrey',alpha=1,clip_on=False)
    
    ccc=ww.Aquatic1_5.mpl_colormap(np.linspace(0,0.7,len(variqq)))
    plt.scatter(0,median_toe,s=100,c=ccc,edgecolor=ccc,zorder=5,clip_on=False)
    plt.scatter(0,mean_toe,s=100,c='crimson',edgecolor=ccc,zorder=5,clip_on=False,
                marker='x')
    
    
    plt.errorbar(0,median_toe,
                  yerr=np.array([[median_toe-toe_min,toe_max-median_toe]]).T,
                  color=ccc,linewidth=1.5,capthick=3,capsize=10,clip_on=False)
    
    plt.ylabel(r'\textbf{Timing of Emergence}',color='k',fontsize=11)    
    plt.xticks(np.arange(0,1,1),variqq,size=20)
    plt.yticks(np.arange(1950,2040,10),map(str,np.round(np.arange(1950,2040,10),2)))
    plt.xlim([-0.3,1])
    plt.ylim([1970,2030])
    
    ############################################################################### 
    ax = plt.subplot(122)
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both',length=4,width=2,which='major',color='dimgrey',
                   labelbottom='off',bottom='off')
    ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
    ax.xaxis.grid(zorder=1,color='darkgrey',alpha=1,clip_on=False)
    
    ccc=ww.Aquatic1_5.mpl_colormap(np.linspace(0,0.7,len(variqq)))
    plt.scatter(0,mean_cor,s=100,c=ccc,edgecolor=ccc,zorder=5,clip_on=False)
    
    plt.errorbar(0,median_cor,
                  yerr=np.array([[median_cor-cor_min,cor_max-median_cor]]).T,
                  color=ccc,linewidth=1.5,capthick=3,capsize=10,clip_on=False)
    
    plt.ylabel(r'\textbf{Rank Correlation Coefficient}',color='k',fontsize=11)    
    plt.xticks(np.arange(0,1,1),variqq,size=20)
    plt.yticks(np.arange(-1,1.1,0.1),map(str,np.round(np.arange(-1,1.1,0.1),2)))
    plt.xlim([-0.3,1])
    plt.ylim([-0.1,1])
    
    fig.suptitle(r'\textbf{Region -- %s -- %s}' % (reg_name,monthlychoice), color='dimgrey',fontsize=25)
    
    plt.tight_layout()
    plt.savefig(directoryfigure + 'TOE_%s_%s-%s_%s_SAMPLES-%s.png' % (monthlychoice,dataset,dataset_obs,reg_name,SAMPLES),dpi=300)
                
