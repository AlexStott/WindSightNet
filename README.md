# WindSightNet code repository

Repository for code to plot Figures for "WindSightNet: the inter-annual variability of Martian winds retrieved from InSight's seismic data with machine learning" submitted to JGR: Planets.
Authors: A. E. Stott, R. F. Garcia, N. Murdoch, D. Mimoun, M. Drilleau, C. Newman, A. Spiga, D. Banfield, M. Lemmon, S. Navarro, L. Mora-Sotomayor, C. Charalambous, W. T. Pike, P. Lognonné, W. B .Banerdt

Plot_WindSightNetData.py
Python file plotting each data Figure in the paper. Use as guideline for using WindSightNet wind speed and direction catalogue 

Requirements:
WindSightNet data available for download: 
A. E. Stott et al. https://doi.org/10.5281/zenodo.14267663

File contains value of wind speed and direction for a sample at given times:
Sol - number of sol of InSight mission 
UTC - Coordinated Universal Time of sample
LTST - Local True Solar Time of sample
L_s - Solar longitude value of sample
Time - seconds since UNIX epoch
Data is considered to be sampled at a rate of 0.01 Hz when there are no gaps.

For Figure 11
InSight optical depth data available: Lemmon, Mark (2024), “InSight Mars lander optical depths”, Mendeley Data, V1, doi: 10.17632/7fn53rjdpc.1

To create axes with L_s in use code to convert Mars times from:
https://pss-gitlab.math.univ-paris-diderot.fr/sainton/marsconverter
Note - not necessary for basic plotting of original data which already has converted times

InSight seismic data available: 
InSight Mars SEIS Data Service. (2019). SEIS raw data, Insight Mission. IPGP, JPL, CNES, ETHZ, ICL, MPS, ISAE-Supaero, LPG, MFSC. https://doi.org/10.18715/SEIS.INSIGHT.XB_2016

InSight TWINS wind data available: 
J A Manfredi, Insight Auxiliary Payload Sensor Subsystem (APSS) Temperatures and Wind Sensor for Insight (TWINS) Archive Bundle, (2019), https://doi.org/10.17189/1518950

