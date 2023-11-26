# TOE_TMIN-TMAX [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10005347.svg)](https://doi.org/10.5281/zenodo.10005347)
Timing of emergence of CONUS summertime temperatures using explainable neural networks

###### Under construction... ```[Python 3.9]```

## Contact
Zachary Labe - [Research Website](https://zacklabe.com/) - [@ZLabe](https://twitter.com/ZLabe)

## Description
+ ```Scripts/```: Main [Python](https://www.python.org/) scripts/functions used in data analysis and plotting
+ ```requirements.txt```: List of environments and modules associated with the most recent version of this project. A Python [Anaconda3 Distribution](https://docs.continuum.io/anaconda/) was used for our analysis. Tools including [NCL](https://www.ncl.ucar.edu/), [CDO](https://code.mpimet.mpg.de/projects/cdo), and [NCO](http://nco.sourceforge.net/) were also used for initial data processing.

## Data
+ ERA5 : [[DATA]](https://cds.climate.copernicus.eu/cdsapp#!/home)
    + Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horányi, A., Muñoz‐Sabater, J., ... & Simmons, A. (2020). The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society, doi:10.1002/qj.3803 [[PUBLICATION]](https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.3803)
+ CESM1 Large Ensemble Project (LENS1) : [[DATA]](http://www.cesm.ucar.edu/projects/community-projects/LENS/data-sets.html)
    + Kay, J. E and Coauthors, 2015: The Community Earth System Model (CESM) large ensemble project: A community resource for studying climate change in the presence of internal climate variability. Bull. Amer. Meteor. Soc., 96, 1333–1349, doi:10.1175/BAMS-D-13-00255.1 [[PUBLICATION]](http://journals.ametsoc.org/doi/full/10.1175/BAMS-D-13-00255.1)
+ CESM2 Large Ensemble Project (LENS2) : [[DATA]](https://www.cesm.ucar.edu/projects/community-projects/LENS2/)
    + Rodgers, K. B., Lee, S. S., Rosenbloom, N., Timmermann, A., Danabasoglu, G., Deser, C., ... & Yeager, S. G. (2021). Ubiquity of human-induced changes in climate variability. Earth System Dynamics Discussions, 1-22, doi:10.1175/BAMS-D-13-00255.1 [[PUBLICATION]](https://esd.copernicus.org/preprints/esd-2021-50/)
+ GFDL FLOR: Forecast-oriented Low Ocean Resolution : [[DATA]](https://www.gfdl.noaa.gov/cm2-5-and-flor/)
    + Vecchi, G. A., Delworth, T., Gudgel, R., Kapnick, S., Rosati, A., Wittenberg, A. T., ... & Zhang, S. (2014). On the seasonal forecasting of regional tropical cyclone activity. Journal of Climate, 27(21), 7994-8016. doi:10.1175/JCLI-D-14-00158.1 [[PUBLICATION]](https://journals.ametsoc.org/view/journals/clim/27/21/jcli-d-14-00158.1.xml)
+ GFDL SPEAR: Seamless System for Prediction and EArth System Research : [[DATA]](https://www.gfdl.noaa.gov/spear_large_ensembles/)
    + Delworth, T. L., Cooke, W. F., Adcroft, A., Bushuk, M., Chen, J. H., Dunne, K. A., ... & Zhao, M. (2020). SPEAR: The next generation GFDL modeling system for seasonal to multidecadal prediction and projection. Journal of Advances in Modeling Earth Systems, 12(3), e2019MS001895. doi:10.1029/2019MS001895 [[PUBLICATION]](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019MS001895)
+ Sixth Version of Model for Interdisciplinary Research on Climate (MIROC6) : [[DATA]](https://climexp.knmi.nl/selectfield_cmip6.cgi?id=someone@somewhere)
    + Tatebe, H., Ogura, T., Nitta, T., Komuro, Y., Ogochi, K., Takemura, T., ... & Kimoto, M. (2019). Description and basic evaluation of simulated mean state, internal variability, and climate sensitivity in MIROC6. Geoscientific Model Development, 12(7), 2727-2765. doi:10.5194/gmd-12-2727-2019 [[PUBLICATION]](https://gmd.copernicus.org/articles/12/2727/2019/gmd-12-2727-2019.html)
+ NOAA Monthly U.S. Climate Gridded Dataset (NClimGrid) : [[DATA]](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00332)
    + Vose, R. S., Applequist, S., Squires, M., Durre, I., Menne, M. J., Williams, C. N., Jr., Fenimore, C., Gleason, K., & Arndt, D. (2014). Improved Historical Temperature and Precipitation Time Series for U.S. Climate Divisions, Journal of Applied Meteorology and Climatology, 53(5), 1232-1251. doi:10.1175/JAMC-D-13-0248.1 [[PUBLICATION]](https://journals.ametsoc.org/view/journals/apme/53/5/jamc-d-13-0248.1.xml)
+ NOAA-CIRES-DOE Twentieth Century Reanalysis (20CRv3) : [[DATA]](https://psl.noaa.gov/data/gridded/data.20thC_ReanV3.html)
    + Slivinski, L. C., Compo, G. P., Whitaker, J. S., Sardeshmukh, P. D., Giese, B. S., McColl, C., ... & Wyszyński, P. (2019). Towards a more reliable historical reanalysis: Improvements for version 3 of the Twentieth Century Reanalysis system. Quarterly Journal of the Royal Meteorological Society, 145(724), 2876-2908. doi:10.1002/qj.3598 [[PUBLICATION]](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3598)
    
## Publications
+ **[1]** **Labe, Z.M.**, N.C. Johnson, and T.L. Delworth (2023), Changes in United States summer temperatures revealed by explainable neural networks, *submitted*. [[PREPRINT]](https://doi.org/10.22541/essoar.168987129.98069596/v1)

## Conferences/Presentations
+ **[4]** **Labe, Z.M.**, N.C. Johnson, and T.L. Delworth. Distinguishing the regional emergence of United States summer temperatures between observations and climate model large ensembles, *23rd Conference on Artificial Intelligence for Environmental Science*, Baltimore, MD (Jan 2024). [[Abstract]](https://ams.confex.com/ams/104ANNUAL/meetingapp.cgi/Paper/431288)
+ **[3]** **Labe, Z.M.**, Climate change extremes by season in the United States, *Hershey Horticulture Society*, Hershey, PA (Sep 2023). [[Slides]](https://www.slideshare.net/ZacharyLabe/climate-change-extremes-by-season-in-the-united-states)
+ **[2]** **Labe, Z.M.** Using explainable AI to identify key regions of climate change in GFDL SPEAR large ensembles, *GFDL Lunchtime Seminar Series*, Princeton, NJ, USA (Mar 2023). [[Slides]](https://www.slideshare.net/ZacharyLabe/using-explainable-ai-to-identify-key-regions-of-climate-change-in-gfdl-spear-large-ensembles)
+ **[1]** **Labe, Z.M.** Forced climate signals with explainable AI and large ensembles, Atmospheric and Oceanic Sciences Student/Postdoc Seminar, Princeton University, NJ, USA (Feb 2023). [[Slides]](https://www.slideshare.net/ZacharyLabe/forced-climate-signals-with-explainable-ai-and-large-ensembles)
