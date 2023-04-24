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
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]

### Parameters
directorydata = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/Predictions/LoopFinal/'
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/MS_Figures/' 
years = np.arange(1921,2021+1,1)
variqq = [r'\textbf{TAVG}',r'\textbf{TMAX}',r'\textbf{TMIN}']
land_only = False
ocean_only = False
rm_annual_mean = False
rm_merid_mean = False
rm_ensemble_mean = False
random_network_seed = 87750
random_segment_seed = 71541
hiddensList = [[100,100]]
ridge_penalty = [0.001]
monthlychoice = 'JJA'
reg_name = 'W_US'
classChunk = 10
lr_here = 0.01
batch_size = 32
iterations = [500]
NNType = 'ANN'
dataset = 'SPEAR_MED'
dataset_obs = 'NClimGrid_MEDS'
SAMPLES = 100

### Save TOE calculations
directorytoe = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/Predictions/ToeFinal/'
toe = np.load(directorytoe + 'TOE_v3-LoopFinal-ObsPred_%s_%s-%s_%s_SAMPLES-%s.npz' % (monthlychoice,dataset,dataset_obs,reg_name,SAMPLES))
skill = np.load(directorytoe + 'SpearmanCorr_v3-LoopFinal-ObsPred_%s_%s-%s_%s_SAMPLES-%s.npz' % (monthlychoice,dataset,dataset_obs,reg_name,SAMPLES))

plot_toe = toe['toe_annall']
plot_corr = skill['cor_annall']

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
###############################################################################
###############################################################################
############################################################################### 
### Training figure
fig = plt.figure(figsize=(8,9))
ax = plt.subplot(321)
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

def set_box_color(bp, color):
    plt.setp(bp['boxes'],color=color)
    plt.setp(bp['whiskers'], color=color,linewidth=1.5)
    plt.setp(bp['caps'], color='w',alpha=0)
    plt.setp(bp['medians'], color='r',linewidth=1)
    plt.setp(bp['means'], color='k',linewidth=2)
    
### Mask nans for boxplot
datamask = plot_toe.copy()
mask = ~np.isnan(datamask)
filtered_data = [d[m] for d, m in zip(datamask, mask)]

positionsq = np.arange(len(plot_toe))
bpl = plt.boxplot(filtered_data,positions=positionsq,widths=0.74,
                  patch_artist=True,sym='',showmeans=True,meanline=True)

# Modify boxes
cp= 'teal'
set_box_color(bpl,cp)
plt.plot([], c=cp, label=r'\textbf{MAE}',clip_on=False)
plt.text(2.3,2022,r'\textbf{[%s]}' % letters[0],color='k',fontsize=10)
 
plt.xticks(np.arange(0,3,1),[],size=0)
plt.yticks(np.arange(1950,2040,10),map(str,np.round(np.arange(1950,2040,10),2)))
plt.xlim([-0.5,2.5])
plt.ylim([1990,2022])

############################################################################### 
ax = plt.subplot(322)
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

def set_box_color(bp, color):
    plt.setp(bp['boxes'],color=color)
    plt.setp(bp['whiskers'], color=color,linewidth=1.5)
    plt.setp(bp['caps'], color='w',alpha=0)
    plt.setp(bp['medians'], color='r',linewidth=1)
    plt.setp(bp['means'], color='k',linewidth=2)
    
### Mask nans for boxplot
datamask = plot_corr.copy()
mask = ~np.isnan(datamask)
filtered_data = [d[m] for d, m in zip(datamask, mask)]

positionsq = np.arange(len(plot_corr))
bpl = plt.boxplot(filtered_data,positions=positionsq,widths=0.74,
                  patch_artist=True,sym='',showmeans=True,meanline=True)

# Modify boxes
cp= 'maroon'
set_box_color(bpl,cp)
plt.plot([], c=cp, label=r'\textbf{MAE}',clip_on=False)

plt.text(2.6,0.4,r'\textbf{Western USA}',fontsize=20,rotation=270,color='dimgrey',va='center')
plt.text(2.3,1,r'\textbf{[%s]}' % letters[1],color='k',fontsize=10)
   
plt.xticks(np.arange(0,3,1),[],size=0)
plt.yticks(np.arange(-1,1.1,0.2),map(str,np.round(np.arange(-1,1.1,0.2),2)))
plt.xlim([-0.6,2.6])
plt.ylim([-0.1,1])

###############################################################################
###############################################################################
############################################################################### 
###############################################################################
###############################################################################
############################################################################### 
###############################################################################
###############################################################################
############################################################################### 
### Parameters
years = np.arange(1921,2021+1,1)
variqq = [r'\textbf{TAVG}',r'\textbf{TMAX}',r'\textbf{TMIN}']
land_only = False
ocean_only = False
rm_annual_mean = False
rm_merid_mean = False
rm_ensemble_mean = False
random_network_seed = 87750
random_segment_seed = 71541
hiddensList = [[100,100]]
ridge_penalty = [0.001]
monthlychoice = 'JJA'
reg_name = 'Ce_US'
classChunk = 10
lr_here = 0.01
batch_size = 32
iterations = [500]
NNType = 'ANN'
dataset = 'SPEAR_MED'
dataset_obs = 'NClimGrid_MEDS'
SAMPLES = 100

### Save TOE calculations
directorytoe = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/Predictions/ToeFinal/'
toe = np.load(directorytoe + 'TOE_v3-LoopFinal-ObsPred_%s_%s-%s_%s_SAMPLES-%s.npz' % (monthlychoice,dataset,dataset_obs,reg_name,SAMPLES))
skill = np.load(directorytoe + 'SpearmanCorr_v3-LoopFinal-ObsPred_%s_%s-%s_%s_SAMPLES-%s.npz' % (monthlychoice,dataset,dataset_obs,reg_name,SAMPLES))

plot_toe = toe['toe_annall']
plot_corr = skill['cor_annall']

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
### Training figure
ax = plt.subplot(323)
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

def set_box_color(bp, color):
    plt.setp(bp['boxes'],color=color)
    plt.setp(bp['whiskers'], color=color,linewidth=1.5)
    plt.setp(bp['caps'], color='w',alpha=0)
    plt.setp(bp['medians'], color='r',linewidth=1)
    plt.setp(bp['means'], color='k',linewidth=2)
    
### Mask nans for boxplot
datamask = plot_toe.copy()
mask = ~np.isnan(datamask)
filtered_data = [d[m] for d, m in zip(datamask, mask)]

positionsq = np.arange(len(plot_toe))
bpl = plt.boxplot(filtered_data,positions=positionsq,widths=0.74,
                  patch_artist=True,sym='',showmeans=True,meanline=True)

# Modify boxes
cp= 'teal'
set_box_color(bpl,cp)
plt.plot([], c=cp, label=r'\textbf{MAE}',clip_on=False)
plt.text(2.3,2022,r'\textbf{[%s]}' % letters[2],color='k',fontsize=10)

plt.ylabel(r'\textbf{Timing of Emergence}',color='k',fontsize=11)      
plt.xticks(np.arange(0,3,1),[],size=0)
plt.yticks(np.arange(1950,2040,10),map(str,np.round(np.arange(1950,2040,10),2)))
plt.xlim([-0.5,2.5])
plt.ylim([1990,2022])

############################################################################### 
ax = plt.subplot(324)
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

def set_box_color(bp, color):
    plt.setp(bp['boxes'],color=color)
    plt.setp(bp['whiskers'], color=color,linewidth=1.5)
    plt.setp(bp['caps'], color='w',alpha=0)
    plt.setp(bp['medians'], color='r',linewidth=1)
    plt.setp(bp['means'], color='k',linewidth=2)
    
### Mask nans for boxplot
datamask = plot_corr.copy()
mask = ~np.isnan(datamask)
filtered_data = [d[m] for d, m in zip(datamask, mask)]

positionsq = np.arange(len(plot_corr))
bpl = plt.boxplot(filtered_data,positions=positionsq,widths=0.74,
                  patch_artist=True,sym='',showmeans=True,meanline=True)

# Modify boxes
cp= 'maroon'
set_box_color(bpl,cp)
plt.plot([], c=cp, label=r'\textbf{MAE}',clip_on=False)
plt.text(2.6,0.4,r'\textbf{Central USA}',fontsize=20,rotation=270,color='dimgrey',va='center')
plt.text(2.3,1,r'\textbf{[%s]}' % letters[3],color='k',fontsize=10)

plt.ylabel(r'\textbf{Rank Correlation Coefficient}',color='k',fontsize=11)    
plt.xticks(np.arange(0,3,1),[],size=0)
plt.yticks(np.arange(-1,1.1,0.2),map(str,np.round(np.arange(-1,1.1,0.2),2)))
plt.xlim([-0.6,2.6])
plt.ylim([-0.1,1])

###############################################################################
###############################################################################
############################################################################### 
###############################################################################
###############################################################################
############################################################################### 
###############################################################################
###############################################################################
############################################################################### 
### Parameters
years = np.arange(1921,2021+1,1)
variqq = [r'\textbf{TAVG}',r'\textbf{TMAX}',r'\textbf{TMIN}']
land_only = False
ocean_only = False
rm_annual_mean = False
rm_merid_mean = False
rm_ensemble_mean = False
random_network_seed = 87750
random_segment_seed = 71541
hiddensList = [[100,100]]
ridge_penalty = [0.001]
monthlychoice = 'JJA'
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
directorytoe = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/Predictions/ToeFinal/'
toe = np.load(directorytoe + 'TOE_v3-LoopFinal-ObsPred_%s_%s-%s_%s_SAMPLES-%s.npz' % (monthlychoice,dataset,dataset_obs,reg_name,SAMPLES))
skill = np.load(directorytoe + 'SpearmanCorr_v3-LoopFinal-ObsPred_%s_%s-%s_%s_SAMPLES-%s.npz' % (monthlychoice,dataset,dataset_obs,reg_name,SAMPLES))

plot_toe = toe['toe_annall']
plot_corr = skill['cor_annall']

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
### Training figure
ax = plt.subplot(325)
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

def set_box_color(bp, color):
    plt.setp(bp['boxes'],color=color)
    plt.setp(bp['whiskers'], color=color,linewidth=1.5)
    plt.setp(bp['caps'], color='w',alpha=0)
    plt.setp(bp['medians'], color='r',linewidth=1)
    plt.setp(bp['means'], color='k',linewidth=2)
    
### Mask nans for boxplot
datamask = plot_toe.copy()
mask = ~np.isnan(datamask)
filtered_data = [d[m] for d, m in zip(datamask, mask)]

positionsq = np.arange(len(plot_toe))
bpl = plt.boxplot(filtered_data,positions=positionsq,widths=0.74,
                  patch_artist=True,sym='',showmeans=True,meanline=True)

# Modify boxes
cp= 'teal'
set_box_color(bpl,cp)
plt.plot([], c=cp, label=r'\textbf{MAE}',clip_on=False)
plt.text(2.3,2022,r'\textbf{[%s]}' % letters[4],color='k',fontsize=10)

plt.xticks(np.arange(0,3,1),variqq,size=20)
plt.yticks(np.arange(1950,2040,10),map(str,np.round(np.arange(1950,2040,10),2)))
plt.xlim([-0.5,2.5])
plt.ylim([1990,2022])

############################################################################### 
ax = plt.subplot(326)
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

def set_box_color(bp, color):
    plt.setp(bp['boxes'],color=color)
    plt.setp(bp['whiskers'], color=color,linewidth=1.5)
    plt.setp(bp['caps'], color='w',alpha=0)
    plt.setp(bp['medians'], color='r',linewidth=1)
    plt.setp(bp['means'], color='k',linewidth=2)
    
### Mask nans for boxplot
datamask = plot_corr.copy()
mask = ~np.isnan(datamask)
filtered_data = [d[m] for d, m in zip(datamask, mask)]

positionsq = np.arange(len(plot_corr))
bpl = plt.boxplot(filtered_data,positions=positionsq,widths=0.74,
                  patch_artist=True,sym='',showmeans=True,meanline=True)

# Modify boxes
cp= 'maroon'
set_box_color(bpl,cp)
plt.plot([], c=cp, label=r'\textbf{MAE}',clip_on=False)
plt.text(2.6,0.4,r'\textbf{Eastern USA}',fontsize=20,rotation=270,color='dimgrey',va='center')
plt.text(2.3,1,r'\textbf{[%s]}' % letters[5],color='k',fontsize=10)
  
plt.xticks(np.arange(0,3,1),variqq,size=20)
plt.yticks(np.arange(-1,1.1,0.2),map(str,np.round(np.arange(-1,1.1,0.2),2)))
plt.xlim([-0.6,2.6])
plt.ylim([-0.1,1])


plt.tight_layout()
plt.savefig(directoryfigure + 'TOE_%s_%s-%s_%s_SAMPLES-%s.png' % (monthlychoice,dataset,dataset_obs,'REGIONS',SAMPLES),dpi=300)
                
