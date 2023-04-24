"""
Plot TOE for observations using the predict the year approach after 
different samples

Author     : Zachary M. Labe
Date       : 20 October 2022
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
years = np.arange(1921,2021+1,1)
variqq = [r'\textbf{TMAX}',r'\textbf{TMIN}']
land_only = False
ocean_only = False
rm_annual_mean = False
rm_merid_mean = False
rm_ensemble_mean = False
random_network_seed = 87750
random_segment_seed = 71541
hiddensList = [[10,10,10]]
ridge_penalty = [0.001]
monthlychoiceq = ['DJF','MAM','JJA','SON']
reg_name = 'E_US'
classChunk = 10
lr_here = 0.01
batch_size = 32
iterations = [500]
NNType = 'ANN'
dataset = 'SPEAR_MED'
dataset_obs = 'NClimGrid_MEDS'
SAMPLES = 100

### Save TOE calculations
for sea in range(len(monthlychoiceq)):
    monthlychoice = monthlychoiceq[sea]
    directorytoe = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/Predictions/ToeFinal/'
    toe = np.load(directorytoe + 'TOE_v3-LoopFinal-ObsPred_Seasons_%s-%s_%s_SAMPLES-%s.npz' % (dataset,dataset_obs,reg_name,SAMPLES))
    skill = np.load(directorytoe + 'SpearmanCorr_v3-LoopFinal-ObsPred_Seasons_%s-%s_%s_SAMPLES-%s.npz' % (dataset,dataset_obs,reg_name,SAMPLES))
    
    plot_toe = toe['toe_annall'][:,sea,:].squeeze()
    plot_corr = skill['cor_annall'][:,sea,:].squeeze()
    
    ### Percentiles
    minPER = 5
    maxPER = 95
    
    mean_toe = np.nanmean(plot_toe,axis=1)
    median_toe = np.nanmedian(plot_toe,axis=1)
    toe_max = np.nanpercentile(plot_toe,minPER,axis=1)
    toe_min = np.nanpercentile(plot_toe,maxPER,axis=1)
    
    mean_cor = np.nanmean(plot_corr,axis=1)
    median_cor = np.nanmedian(plot_corr,axis=1)
    cor_max = np.nanpercentile(plot_corr,minPER,axis=1)
    cor_min = np.nanpercentile(plot_corr,maxPER,axis=1)
    
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
    
    ccc=ww.Aquatic1_5.mpl_colormap(np.linspace(0,0.7,len(plot_toe)))
    for i in range(len(plot_toe)):
        plt.scatter(i,median_toe[i],s=100,c=ccc[i],edgecolor=ccc[i],zorder=5,clip_on=False)
        plt.scatter(i,mean_toe[i],s=100,c='crimson',edgecolor=ccc[i],zorder=5,clip_on=False,
                    marker='x')
        
        
        plt.errorbar(i,median_toe[i],
                      yerr=np.array([[median_toe[i]-toe_min[i],toe_max[i]-median_toe[i]]]).T,
                      color=ccc[i],linewidth=1.5,capthick=3,capsize=10,clip_on=False)
    
    plt.ylabel(r'\textbf{Timing of Emergence}',color='k',fontsize=11)    
    plt.xticks(np.arange(0,2,1),variqq,size=20)
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
    
    ccc=ww.Aquatic1_5.mpl_colormap(np.linspace(0,0.7,len(plot_toe)))
    for i in range(len(plot_corr)):
        plt.scatter(i,mean_cor[i],s=100,c=ccc[i],edgecolor=ccc[i],zorder=5,clip_on=False)
        
        plt.errorbar(i,median_cor[i],
                      yerr=np.array([[median_cor[i]-cor_min[i],cor_max[i]-median_cor[i]]]).T,
                      color=ccc[i],linewidth=1.5,capthick=3,capsize=10,clip_on=False)
    
    plt.ylabel(r'\textbf{Rank Correlation Coefficient}',color='k',fontsize=11)    
    plt.xticks(np.arange(0,2,1),variqq,size=20)
    plt.yticks(np.arange(-1,1.1,0.1),map(str,np.round(np.arange(-1,1.1,0.1),2)))
    plt.xlim([-0.3,1])
    plt.ylim([-0.1,1])
    
    fig.suptitle(r'\textbf{Region -- %s -- %s}' % (reg_name,monthlychoice), color='dimgrey',fontsize=25)
    
    plt.tight_layout()
    plt.savefig(directoryfigure + 'TOE_%s_%s-%s_%s_SAMPLES-%s.png' % (monthlychoice,dataset,dataset_obs,reg_name,SAMPLES),dpi=300)
                
