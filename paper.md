---
title: 'XLUR: A land use regression wizard for ArcGIS Pro'
tags:
  - Python
  - ArcGIS Pro
  - Land use regression
authors:
  - name: Anna Molter
    orcid: 0000-0002-0824-4478
    affiliation: 1
affiliations:
 - name: Department of Geography, School of Environment, Education and Development, The University of Manchester
   index: 1
date: 4 July 2019
bibliography: paper.bib
---

# Summary

## Introduction

XLUR is a Python toolbox for ArcGIS Pro (v2.2.4 or higher, Environmental Systems Research Institute (ESRI), Redlands, CA)
that enables the development and application of land use regression models via a wizard style interface. 
Land use regression (LUR) is a commonly used technique in environmental sciences to analyse factors influencing pollutant levels 
and to predict pollutant levels at unmeasured locations. LUR is extensively used 
in studies on air pollution exposure [@Molter:2010a;@Molter:2010b], but it is widely applicable and has been used in 
fields ranging from water pollution [@Kelsey:2004] to urban climatology [@Heusinkveld:2014]. 

## Purpose of the software

The traditional approach to the development of a LUR model requires several steps: 
1. The creation of a point dataset of known observation data for a variable of interest (such as nitrogen dioxide concentrations or air temperatures), which will be used as outcome variables.
2. Carrying out various spatial analyses of additional geospatial data with these point locations using geographic information systems (GIS) to extract potential predictor variables.
3. Data wrangling of extracted data from steps 1 and 2 into a format that can be readily used by statistical software packages.
4. Carrying out multiple regression analysis to obtain a best fit parsimonious model.

If done manually, steps 2 to 4 are repetitive and time consuming, making them inefficient and prone to error. 
XLUR provides a wizard style interface that guides a user 
through the development of a LUR model without the need to access and run multiple tools and 
additional software packages. XLUR largely automates steps 2 to 4, which significantly speeds
up the model development process and reduces user error. Depending on the type and amount of data to be processed, 
and the available hardware, models can be developed in less than one hour. Furthermore, 
the output files produced by XLUR ensure that the model development is 
well documented and reproducible. Lastly, XLUR also makes the method available for a wider range of users.  

In addition to developing LUR models, XLUR can also apply a previously developed 
model to a new set of locations within the same study area. New locations can be
defined by the user or can be based on a dataset containing regularly located points or randomly located points. 
Again, XLUR largely automates this process with minimal effort from the user. 

XLUR is aimed at GIS specialists. It uses the ArcGIS Pro software, which is the most widely 
used commercial GIS software worldwide. XLUR is based on the LUR methodology used in the 
European Study of Cohorts for Air Pollution Effects (ESCAPE) [@Beelen:2013;@Eeftens:2012] as set out in the ESCAPE Exposure 
assessment manual [@escape]. Within air pollution research the ESCAPE methodology is used as
the standard for developing LUR models. XLUR also allows hybrid LUR models to be developed, based on an extension 
of the ESCAPE methodology that included the addition of satellite derived data and data from chemical transport 
models [@deHoogh:2016]. 

## Current application

XLUR has been developed through the NERC Newton-DIPI funded Urban hybriD models for AiR pollution exposure Assessment (UDARA)
study, which is a collaboration between the University of Manchester and Institut Teknologi
Bandung. This study aims to develop air pollution prediction models for Indonesian urban areas and to analyse 
the effects of air pollution on health indicators provided by the Indonesian Family Life Survey.

## State of the field

Currently, only a small number of LUR software packages have been developed. RLUR [@Morley:2018] has been developed in the 
R programming language and is aimed at users with a background in statistical analyses, such as exposure scientists 
or epidemiologists. PyLUR [@Ma:2020] is written in Python, but not implemented within the ArcGIS software. Instead, 
it uses GDAL/OGR libraries for spatial analysis. The authors report that PyLUR currently does not have a user-friendly 
graphic user interface (GUI) and at the time of writing it is not available in an open source repository. OpenLUR 
[@Lautenschlager:2020] is designed to develop LUR models exclusively based on OpenStreetMap data. Unlike the XLUR, RLUR, 
and PyLUR software packages, the OpenLUR software package does not use the ESCAPE methodology, but an unsupervised machine 
learning process featuring automated hyper-parameter tuning. 

One major difference between these LUR software packages and XLUR is that they are designed specifically for air 
pollution models, i.e. their spatial analyses only extract potential predictors relevant for air pollution modelling. In 
contrast, XLUR is more widely applicable and its wizard style interface can be used to extract potential predictor variables
for a range of environmental phenomena. Another major difference is that XLUR can develop classic LUR models and hybrid LUR
models, that add a measure of global variability to the measures of local variability modelled in LUR. Furthermore, XLUR is
the only software that is implemented within ArcGIS Pro.   

## Availability and implementation

XLUR is a Python toolbox for use within ArcGIS Pro [@ArcGISPro]. An ArcGIS Pro Project file (XLUR.aprx) containing the XLUR toolbox is available on the GitHub repository (https://github.com/anmolter/XLUR). The repository also provides the source code of the tools, a user manual, an example dataset for the tutorial in the user manual, and example outputs. The user manual contains instructions for installing additional Python packages (wxpython, statsmodels, seaborn, patsy) required by XLUR via ArcGIS Pro's Python Package Manager. 

# Acknowledgments

XLUR was reviewed and tested by Prof S Lindley, Department of Geography, School of Environment, 
Education and Development, The University of Manchester. 
This work is supported via the NERC Newton-DIPI Urban hybriD models for AiR pollution exposure 
Assessment (UDARA) project, PIs: Prof G McFiggans, Faculty of Science and Engineering, 
The University of Manchester, UK, and Dr D Driejana, Faculty of Civil and Environmental Engineering,
Institut Teknologi Bandung, Indonesia, NE/P014631/1. It builds on work carried out in the 
European Union's Seventh Framework 
Programme Theme ENV.2007.1.2.2.2. European cohort on air pollution. 

# References
