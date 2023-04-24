"""
Train ANN to predict the year for ToE in the CONUS

Reference   : Barnes et al. [2020, JAMES] & Labe and Barnes [2021, JAMES]
Author      : Zachary M. Labe
Date        : 28 March 2023
Environment : conda activate env-tf27
Tensorflow  : 2.7 (XAI for v2.0.1)
Version     : 5 (shuffle maps of SPEAR_MED)
"""

###############################################################################
###############################################################################
###############################################################################
### Import packages
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import calc_Utilities as UT
import sys

###############################################################################
###############################################################################
###############################################################################
### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
###############################################################################
### Testing data for ANN --------------------------------------
resolution = 'MEDS'
datasetsingle = ['SPEAR_%s' % resolution[:3]]
datasetname = 'SPEAR_MED_SSP585'
dataobsname = 'NClimGrid'
# datasetsingle = ['FLOR_%s' % resolution]
dataset_obs = 'NClimGrid_%s' % resolution
yearsall = [np.arange(1921,2100+1,1)]
monthlychoiceall = ['JJA']
monthlychoice = monthlychoiceall[0]
variq = 'T2M'
reg_name = 'US'
segment_data_factorq = [0.8]
lat_bounds,lon_bounds = UT.regions(reg_name)

###############################################################################
###############################################################################
###############################################################################
### Select whether to standardize over a baseline
baselineSTD = True
yrminb = 1981
yrmaxb = 2010
baselineSTDyrs = np.arange(yrminb,yrmaxb+1,1)

###############################################################################
###############################################################################
###############################################################################
### Masking and preprocessing arguments
rm_annual_mean = False
rm_merid_mean = False
rm_ensemble_mean = False
land_only = False
ocean_only = False
CONUS_only = True

### Parameters
if reg_name == 'US':
    debug = True
    NNType = 'ANN'
    annType = 'class'
    classChunkHalf = 5
    classChunk = 10
    biasBool = True
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
    biasBool = True
    hiddensList = [[100,100]]
    ridge_penalty = [0.001]
    actFun = 'relu'
    iterations = [500]
else:
    print(ValueError('This is the wrong region name selected!!!'))
    sys.exit()
    
lr_here = .01
batch_size = 32 
random_network_seed = 87750
random_segment_seed = 71541

### Years
yearsmodel = yearsall[0]
yearsobs = np.arange(1921,2022+1,1)

###############################################################################
###############################################################################
###############################################################################
### Read in data
directorypredictions = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/Predictions/Regions/'
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/MS_Figures/'
dataset = datasetsingle[0]
modelType = 'TrainedOn' + dataset
savename = modelType+'_Vari-'+variq+'_Obs-' + dataset_obs + '_Region-' + reg_name + '_Months-' + monthlychoice + '_L2-'+ str(ridge_penalty[0])+ '_LR-' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(iterations[0]) + '_' + NNType + str(hiddensList[0][0]) + 'x' + str(hiddensList[0][-1]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
savenameModelTestTrain = modelType+'_'+variq+'_modelTrainTest_SegSeed'+str(random_segment_seed)+'_NetSeed'+str(random_network_seed)

testIndices = np.genfromtxt(directorypredictions + savename + 'TestIndices_ENSEMBLES.txt')
lenens = len(testIndices)

Ypred = np.genfromtxt(directorypredictions + savename + 'TEST_PREDICTIONS.txt').reshape(lenens,yearsmodel.shape[0])
Yactual = yearsmodel
Ypredobs = np.genfromtxt(directorypredictions + savename + 'OBSERVATION_PREDICTIONS.txt')

### Baseline
baseq = np.where((yearsmodel >= 1921) & (yearsmodel <= 1950))[0]
whereyr = np.argmax(Ypredobs[baseq])
maxyr = np.max(Ypredobs[baseq])

### Calculate ToE
ToE = np.array([np.nan])
for yr in range(len(Ypredobs)):
    currentpred = Ypredobs[yr]
    if currentpred > maxyr:
        if np.min(Ypredobs[yr:]) > maxyr:
            if np.isnan(ToE):
                ToE[:] = yr
if np.isnan(ToE):
    actual_TOE = np.nan
else:
    actual_TOE = yearsobs[int(ToE[0])]

###############################################################################
###############################################################################
###############################################################################
### Plotting functions
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
            
fig = plt.figure(figsize=(8,7))
ax = plt.subplot(111)

adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
# ax.xaxis.grid(color='dimgrey',linestyle='-',linewidth=2,clip_on=False)

plt.fill_between(yearsmodel[baseq],1919,2101,color='darkgrey',edgecolor=None,
                 alpha=0.4)
plt.fill_between(np.arange(yearsobs[int(ToE[0])],yearsobs[-1]+1,1),1919,2101,color='crimson',edgecolor=None,
                 alpha=0.2)

for i in range(Ypred.shape[0]):
    if i == 0:
        plt.plot(Yactual,Ypred[i,:],'o',
                markersize=8,color='teal',clip_on=False,alpha=0.4,
                markeredgecolor='dimgrey',markeredgewidth=0.8,
                label=r'\textbf{%s}' % datasetname)
    else:
        plt.plot(Yactual,Ypred[i,:],'o',
                markersize=8,color='teal',clip_on=False,alpha=0.4,
                markeredgecolor='dimgrey',markeredgewidth=0.4)        

plt.plot(yearsobs,Ypredobs,'X',color='maroon',markersize=8,
          label=r'\textbf{%s}' % dataobsname,clip_on=False,markeredgecolor='dimgrey',markeredgewidth=0.4)
plt.scatter(yearsobs[whereyr],Ypredobs[whereyr],color='crimson',marker='X',s=100,
          clip_on=False,zorder=11)
plt.scatter(yearsobs[int(ToE[0])],Ypredobs[int(ToE[0])],color='crimson',marker='X',s=100,
          clip_on=False,zorder=11)
plt.axhline(Ypredobs[whereyr],xmin=0.065,xmax=0.565,linewidth=1,color='crimson')
plt.axvline(yearsobs[int(ToE[0])],linewidth=1,color='crimson',clip_on=False,
            ymin=0,ymax=0.7)

plt.xlabel(r'\textbf{ACTUAL YEAR}',fontsize=15,color='dimgrey')
plt.ylabel(r'\textbf{PREDICTED YEAR}',fontsize=15,color='dimgrey')
plt.plot(np.arange(yearsall[0].min(),yearsall[0].max()+1,1),np.arange(yearsall[0].min(),yearsall[0].max()+1,1),'-',
          color='dimgrey',linewidth=2,clip_on=False)

plt.text(2022.5,1990,r'\textbf{Latest prediction for 1921-1950}',color='crimson')
plt.text(2004.5,2047,r'\textbf{Actual ToE = 2005}',color='crimson',ha='right')

plt.xticks(np.arange(yearsall[0].min()-1,2101,20),map(str,np.arange(yearsall[0].min()-1,2101,20)),size=10)
plt.yticks(np.arange(yearsall[0].min()-1,2101,20),map(str,np.arange(yearsall[0].min()-1,2101,20)),size=10)
plt.xlim([yearsall[0].min()-1,yearsall[0].max()])   
plt.ylim([yearsall[0].min()-1,yearsall[0].max()])

leg = plt.legend(shadow=False,fontsize=17,loc='upper center',
              bbox_to_anchor=(0.82,0.13),fancybox=True,ncol=1,frameon=False,
              handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

### Save the predict the year plot
plt.tight_layout()
plt.savefig(directoryfigure + 'SPEAR_MED_predction.png',dpi=600)  
