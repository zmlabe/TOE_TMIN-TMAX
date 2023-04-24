"""
Create plots to show validation MAE for different architectures

Reference   : Barnes et al. [2020, JAMES] & Labe and Barnes [2021, JAMES]
Author      : Zachary M. Labe
Date        : 10 October 2022
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
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/MS_Figures/' 

### Read in hyperparameters
maetype = 'valall'
if maetype == 'vala':
    maetypes = '1921-1989'
elif maetype == 'valb':
    maetypes = '1990-2100'
elif maetype == 'valall':
    maetypes = '1921-2100'
region = 'US'
variq = 'T2M'
months = 'JJA'
dataset = 'SPEAR_MED'
dataset_obs = 'NClimGrid_MEDS'
scores = np.load(directorydata + 'ANN_PredictTheYear_v2-LoopHyperparameters_%s_%s_%s-%s_%s.npz' % (variq,months,dataset,dataset_obs,region))
mae = scores['mae_%s' % maetype]

###############################################################################
###############################################################################
###############################################################################
### Graph for accuracy
labels = [r'\textbf{1-LAYER$_{10}$}',r'\textbf{1-LAYER$_{20}$}',
          r'\textbf{2-LAYERS$_{10}$}', r'\textbf{2-LAYERS$_{30}$}',
          r'\textbf{2-LAYERS$_{50}$}',r'\textbf{2-LAYERS$_{100}$}',
          r'\textbf{3-LAYERS$_{10}$}',r'\textbf{3-LAYERS$_{20}$}']

fig = plt.figure(figsize=(8,5))
for plo in range(len(hiddenall)):
    ax = plt.subplot(2,4,plo+1)
    
    plotdata = mae[plo,:,:].transpose()
    
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
    ax.tick_params(axis="x",which="both",bottom = False,top=False,
                    labelbottom=False)
    
    ax.yaxis.grid(zorder=1,color='darkgrey',alpha=0.7,clip_on=False,linewidth=0.5)
    
    def set_box_color(bp, color):
        plt.setp(bp['boxes'],color=color)
        plt.setp(bp['whiskers'], color=color,linewidth=1.5)
        plt.setp(bp['caps'], color='w',alpha=0)
        plt.setp(bp['medians'], color='w',linewidth=1)
    
    positionsq = np.arange(len(ridgePenaltyall))
    bpl = plt.boxplot(plotdata,positions=positionsq,widths=0.6,
                      patch_artist=True,sym='')
    
    # Modify boxes
    cp= 'maroon'
    set_box_color(bpl,cp)
    plt.plot([], c=cp, label=r'\textbf{MAE}',clip_on=False)
        
    for i in range(plotdata.shape[1]):
        y = plotdata[:,i]
        x = np.random.normal(positionsq[i], 0.04, size=len(y))
        plt.plot(x, y,color='teal', alpha=0.5,zorder=10,marker='.',linewidth=0,markersize=5,markeredgewidth=0)
     
    if any([plo==0,plo==4]):
        plt.yticks(np.arange(0,101,5),list(map(str,np.round(np.arange(0,101,5),2))),
                    fontsize=6) 
        plt.ylim([0,30])
    else:
        plt.yticks(np.arange(0,101,5),list(map(str,np.round(np.arange(0,101,5),2))),
                    fontsize=6) 
        plt.ylim([0,30])
        ax.axes.yaxis.set_ticklabels([])

    if any([plo==4,plo==5,plo==6,plo==7]):
        plt.text(-0.25,-1.4,r'\textbf{%s}' % ridgePenaltyall[0],fontsize=5,color='dimgrey',
                  ha='left',va='center')
        plt.text(0.75,-1.4,r'\textbf{%s}' % ridgePenaltyall[1],fontsize=5,color='dimgrey',
                  ha='left',va='center')
        plt.text(1.8,-1.4,r'\textbf{%s}' % ridgePenaltyall[2],fontsize=5,color='dimgrey',
                  ha='left',va='center')
        plt.text(2.8,-1.4,r'\textbf{%s}' % ridgePenaltyall[3],fontsize=5,color='dimgrey',
                  ha='left',va='center')
        plt.text(3.9,-1.4,r'\textbf{%s}' % ridgePenaltyall[4],fontsize=5,color='dimgrey',
                  ha='left',va='center')
        plt.text(4.9,-1.4,r'\textbf{%s}' % ridgePenaltyall[5],fontsize=5,color='dimgrey',
                  ha='left',va='center')
  
    plt.text(2.8,33,r'%s' % labels[plo],fontsize=11,color='dimgrey',
              ha='center',va='center')
    plt.text(-1,31.5,r'\textbf{[%s]}' % letters[plo],color='k',fontsize=6)
    
    if any([plo==0,plo==4]):
        plt.ylabel(r'\textbf{MAE [years]}',color='k',fontsize=7)
        
    fig.suptitle(r'\textbf{%s [%s]}'  % (dataset,maetypes),fontsize=20,color='dimgrey')

plt.tight_layout()
plt.text(-10.35,-3.5,r'\textbf{Ridge Regularization [L$_{2}$]}',fontsize=8,color='k',
         ha='left',va='center')  
plt.savefig(directoryfigure + 'validationMAE-%s_LoopHyperparameters_%s_%s_%s_%s.png' % (maetype,variq,months,region,dataset),dpi=300)

