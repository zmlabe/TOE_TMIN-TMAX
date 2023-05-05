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
import scipy.stats as sts
import calc_Stats as dSS

###############################################################################
###############################################################################
###############################################################################
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

letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]

###############################################################################
###############################################################################
###############################################################################
### Testing data for ANN --------------------------------------
resolutionall = ['MEDS','MEDS','MEDS','MEDS','MEDS','MEDS','MEDS','MEDS','MEDS']
datasetsingleall = ['SPEAR_MED','SPEAR_MED','SPEAR_MED','SPEAR_MED','SPEAR_MED','SPEAR_MED','SPEAR_MED','SPEAR_MED','SPEAR_MED']
datasetnamelist = ['SPEAR_MED','SPEAR_MED','SPEAR_MED','SPEAR_MED','SPEAR_MED','SPEAR_MED','SPEAR_MED','SPEAR_MED','SPEAR_MED']
dataobsname = 'NClimGrid'
yearsall = [np.arange(1921,2100+1,1)]
monthlychoiceall = ['JJA']
monthlychoice = monthlychoiceall[0]
variqall = ['TMAX','TMAX','TMAX','TMIN','TMIN','TMIN','T2M','T2M','T2M']
variqname = ['TMAX','TMAX','TMAX','TMIN','TMIN','TMIN','TAVG','TAVG','TAVG']
reg_nameall = ['W_US','Ce_US','E_US','W_US','Ce_US','E_US','W_US','Ce_US','E_US']
regionalname = ['WESTERN USA','CENTRAL USA','EASTERN USA']
segment_data_factorq = [0.8]

### Threshold for min ToE
yearThresh = 5

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

### Years
yearsmodel = yearsall[0]
yearsobs = np.arange(1921,2022+1,1)

fig = plt.figure(figsize=(10,8))
for lll in range(len(datasetsingleall)):
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Read in data
    directorypredictions = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/Predictions/Regions/'
    directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/MS_Figures/'
    dataset = datasetsingleall[lll]
    variq = variqall[lll]
    resolution = resolutionall[lll]
    reg_name = reg_nameall[lll]
    
    lat_bounds,lon_bounds = UT.regions(reg_name)
    
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
    
    dataset_obs = 'NClimGrid_%s' % resolution
    modelType = 'TrainedOn' + dataset
    savename = modelType+'_Vari-'+variq+'_Obs-' + dataset_obs + '_Region-' + reg_name + '_Months-' + monthlychoice + '_L2-'+ str(ridge_penalty[0])+ '_LR-' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(iterations[0]) + '_' + NNType + str(hiddensList[0][0]) + 'x' + str(hiddensList[0][-1]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
    savenameModelTestTrain = modelType+'_'+variq+'_modelTrainTest_SegSeed'+str(random_segment_seed)+'_NetSeed'+str(random_network_seed)
    
    testIndices = np.genfromtxt(directorypredictions + savename + 'TestIndices_ENSEMBLES.txt')
    lenens = len(testIndices)
    
    trainIndices = np.genfromtxt(directorypredictions + savename + 'TrainIndices_ENSEMBLES.txt')
    lenensTrain = len(trainIndices)
    
    Ypred = np.genfromtxt(directorypredictions + savename + 'TEST_PREDICTIONS.txt').reshape(lenens,yearsmodel.shape[0])
    Yactual = yearsmodel
    Ypredobs = np.genfromtxt(directorypredictions + savename + 'OBSERVATION_PREDICTIONS.txt')
    
    ### Calculate statistics on observations
    slope, intercept, r, p, se = sts.linregress(yearsobs,Ypredobs)
    trendline = slope*yearsobs + intercept
    
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
    
    print('\n')
    print(variq,reg_name,actual_TOE,np.round(sts.spearmanr(yearsobs,Ypredobs)[0],2))
    print(r,p,se)
    print('\n\n')
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Plotting functions
    ax = plt.subplot(3,3,lll+1)
    
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
    ax.yaxis.grid(color='darkgrey',linestyle='-',linewidth=0.5,clip_on=False,alpha=0.6)
    
    plt.plot(np.arange(yearsall[0].min(),yearsall[0].max()+1,1),np.arange(yearsall[0].min(),yearsall[0].max()+1,1),'-',
              color='dimgrey',linewidth=2,clip_on=False)
    
    for i in range(Ypred.shape[0]):
        if i == 0:
            plt.plot(Yactual,Ypred[i,:],'o',
                    markersize=4,color='teal',clip_on=False,alpha=0.4,
                    markeredgecolor='dimgrey',markeredgewidth=0.8,
                    label=r'\textbf{%s}' % (datasetnamelist[lll]))
        else:
            plt.plot(Yactual,Ypred[i,:],'o',
                    markersize=4,color='teal',clip_on=False,alpha=0.4,
                    markeredgecolor='dimgrey',markeredgewidth=0.4)        
    
    plt.plot(yearsobs,Ypredobs,'X',color='maroon',markersize=4,
              label=r'\textbf{%s}' % dataobsname,clip_on=False,markeredgecolor='dimgrey',markeredgewidth=0.4)
    
    if np.isnan(ToE) == False:
        plt.scatter(yearsobs[int(ToE[0])],Ypredobs[int(ToE[0])],color='crimson',marker='X',s=50,
                    clip_on=False,zorder=11)
        plt.axvline(yearsobs[int(ToE[0])],linewidth=1,color='crimson',clip_on=False,
                    ymin=0,ymax=0.69)
        if actual_TOE < (yearsobs[-1]-yearThresh):
            plt.text(yearsobs[int(ToE[0])]+0.3,2046.5,r'\textbf{ToE}',color='crimson',ha='center',fontsize=7)
        else:
            plt.text(yearsobs[int(ToE[0])]+0.3,2046.5,r'\textbf{ToE*}',color='crimson',ha='center',fontsize=7)
    # plt.plot(yearsobs,trendline,linewidth=2,color='crimson',clip_on=False)
    # plt.text(yearsobs[-1]+5,trendline[-1]-1,r'\textbf{R$^{2}$=%s}' % np.round(r**2,2),color='crimson',fontsize=9)
    
    if any([lll==3]):
        plt.ylabel(r'\textbf{PREDICTED YEAR}',fontsize=15,color='dimgrey')
    
    if lll == 7:
        plt.xlabel(r'\textbf{ACTUAL YEAR}',fontsize=15,color='dimgrey')
        
    iyears = np.where(Yactual<1990)[0]
    plt.text(1921,2082, r'\textbf{RMSE before 1990} = %s years' % (np.round(dSS.rmse(Ypred[:,iyears],Yactual[iyears]),decimals=1)),
              fontsize=7,ha='left',color='teal')
    
    iyears = np.where(Yactual>=1990)[0]
    plt.text(1921,2074, r'\textbf{RMSE after 1990} = %s years' % (np.round(dSS.rmse(Ypred[:,iyears],Yactual[iyears]),decimals=1)),
              fontsize=7,ha='left',color='teal')
    
    plt.xticks(np.arange(yearsall[0].min()-1,2101,25),map(str,np.arange(yearsall[0].min()-1,2101,25)),size=8)
    plt.yticks(np.arange(yearsall[0].min()-1,2101,25),map(str,np.arange(yearsall[0].min()-1,2101,25)),size=8)
    plt.xlim([yearsall[0].min()-1,yearsall[0].max()])   
    plt.ylim([yearsall[0].min()-1,yearsall[0].max()])
    
    if lll < 3:
        plt.title(r'\textbf{%s}' % regionalname[lll],color='k',fontsize=15)
    
    if lll == 7:
        leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
                      bbox_to_anchor=(0.5,-0.24),fancybox=True,ncol=2,frameon=False,
                      handlelength=1,handletextpad=0.5)
        for line,text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())
        
    plt.text(1920,2096.5,r'\textbf{[%s]}' % letters[lll],fontsize=10,color='k')
    
    if any([lll==2,lll==5,lll==8]):
        plt.text(2101,1985,r'\textbf{%s}' % variqname[lll],fontsize=18,color='dimgrey',rotation=270)

### Save the predict the year plot
plt.tight_layout()
plt.savefig(directoryfigure + 'SPEAR_MED_prediction_Regions.png',dpi=600)  
