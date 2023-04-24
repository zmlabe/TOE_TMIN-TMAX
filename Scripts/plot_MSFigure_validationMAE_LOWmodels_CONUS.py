"""
Create plot to show validation scores across models

Reference   : Barnes et al. [2020, JAMES] & Labe and Barnes [2021, JAMES]
Author      : Zachary M. Labe
Date        : 24 January 2023
Environment : conda activate env-tf27
Tensorflow  : 2.7 (XAI for v2.0.1)
Version     : 2 (looping through L2 and architecture)
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as c
import numpy as np

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/MS_Figures/'

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

### Hyperparamters for files of the ANN model
rm_annual_mean = False
rm_merid_mean = False
rm_ensemble_mean = False
land_only = False
ocean_only = False
CONUS_only = True
COUNTER = 20
hiddenall = [[10],[20],[10,10],[20,20],[50,50],[100,100],[10,10,10],[20,20,20]]
ridgePenaltyall = [0.001,0.01,0.1,0.5,1,5]
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
combinations = COUNTER * len(hiddenall) * len(ridgePenaltyall)
directorydata = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/Predictions/Loop/'

### Read in hyperparameters
fig = plt.figure(figsize=(6,8))
maetypen = ['vala','valb','valall']
for mm in range(len(maetypen)):
    maetype = maetypen[mm]
    
    if maetype == 'vala':
        maetypes = '1921-1989'
    elif maetype == 'valb':
        maetypes = '1990-2100'
    elif maetype == 'valall':
        maetypes = '1921-2100'
    modelsall = ['SPEAR_LOW','FLOR_LOWS','LENS1_LOWS','LENS2_LOWS','MIROC6_LE_LOWS']
    
    mae_median = []
    min_l2 = []
    arg_lw = []
    argwhere_ann = []
    seedsmin = []
    for i in range(len(modelsall)):    
        region = 'US'
        variq = 'T2M'
        months = 'JJA'
        dataset = modelsall[i]
        dataset_obs = 'NClimGrid_LOWS'
        scores = np.load(directorydata + 'ANN_PredictTheYear_v2-LoopHyperparameters_%s_%s_%s-%s_%s.npz' % (variq,months,dataset,dataset_obs,region))
        mae = scores['mae_%s' % maetype]
        
        ### Calculate median
        mae_medianq = np.nanmedian(mae,axis=2) # across seeds
        min_l2q = np.nanmin(mae_medianq,axis=1)
        arg_lwq = np.argmin(mae_medianq,axis=1)
        argwhere_annq = np.argmin(min_l2q)
        
        ### Find best score
        l2_bestarc = arg_lwq[argwhere_annq]
        seedsminq = mae[argwhere_annq,l2_bestarc,:]
        
        mae_median.append(mae_medianq)
        min_l2.append(min_l2q)
        arg_lw.append(arg_lwq)
        argwhere_ann.append(argwhere_annq)
        seedsmin.append(seedsminq)
        
    plotmodels = np.asarray(seedsmin)
        
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Graph for accuracy
    labels = [r'\textbf{SPEAR_LOW}',r'\textbf{FLOR (LOW)}',
              r'\textbf{CESM1-LE}', r'\textbf{CESM2-LE}',
              r'\textbf{MIROC6-LE}']
    
    ax = plt.subplot(3,1,mm+1)
    
    plotdata = plotmodels.transpose()
    
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
    ax.tick_params(axis="x",which="both",bottom = False,top=False,
                    labelbottom=False)
    
    ax.yaxis.grid(zorder=1,color='darkgrey',alpha=0.7,clip_on=False,linewidth=2)
    
    def set_box_color(bp, color):
        plt.setp(bp['boxes'],color=color)
        plt.setp(bp['whiskers'], color=color,linewidth=6)
        plt.setp(bp['caps'], color='k',alpha=0)
        plt.setp(bp['medians'], color='k',linewidth=2)
    
    positionsq = np.arange(len(labels))
    bpl = plt.boxplot(plotdata,positions=positionsq,widths=0.6,
                      patch_artist=True,sym='')
    
    # Modify boxes
    cp= 'deepskyblue'
    set_box_color(bpl,cp)
    plt.plot([], c=cp, label=r'\textbf{MAE}',clip_on=False)
        
    for i in range(plotdata.shape[1]):
        y = plotdata[:,i]
        x = np.random.normal(positionsq[i], 0.04, size=len(y))
        plt.plot(x, y,color='crimson', alpha=0.8,zorder=10,marker='.',linewidth=0,markersize=7,markeredgewidth=0)
     
    plt.yticks(np.arange(0,101,5),list(map(str,np.round(np.arange(0,101,5),2))),
                fontsize=10) 
    plt.ylim([0,15])
    
    plt.text(-0.38,-1.4,r'\textbf{%s}' % labels[0],fontsize=11,color='dimgrey',
              ha='left',va='center')
    plt.text(0.63,-1.4,r'\textbf{%s}' % labels[1],fontsize=11,color='dimgrey',
              ha='left',va='center')
    plt.text(1.65,-1.4,r'\textbf{%s}' % labels[2],fontsize=11,color='dimgrey',
              ha='left',va='center')
    plt.text(2.65,-1.4,r'\textbf{%s}' % labels[3],fontsize=11,color='dimgrey',
              ha='left',va='center')
    plt.text(3.65,-1.4,r'\textbf{%s}' % labels[4],fontsize=11,color='dimgrey',
              ha='left',va='center')
    
    plt.text(4.4,13.6,r'\textbf{[%s]}' % letters[mm],fontsize=10,color='k')
    plt.text(1.5,13,r'\textbf{%s}' % maetypes,fontsize=17,color='k')
    
        
    plt.ylabel(r'\textbf{MAE [years]}',color='dimgrey',fontsize=12)

plt.tight_layout()
plt.savefig(directoryfigure + 'validationMAE-%s_BestANNModel_%s_%s_%s_CompareModels-LOWS.png' % (maetype,variq,months,region),dpi=600)

