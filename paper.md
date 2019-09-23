---
title: 'PyLUR: A land use regression wizard for ArcGIS Pro'
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
bibliography: PyLUR_paper.bib
---

# Summary

## Introduction

PyLUR is a Python toolbox for ArcGIS Pro (v2.2.4 or higher, Environmental Systems Research Institute (ESRI), Redlands, CA) 
that enables the development and application of land use regression models via a wizard style interface. 
Land use regression (LUR) is a commonly used technique in environmental sciences to analyse factors influencing pollutant levels 
and to predict pollutant levels at unmeasured locations. LUR is extensively used 
in studies on air pollution exposure [@Molter:2010a;@Molter:2010b], but it is widely applicable and has been used in 
fields ranging from water pollution [@Kelsey:2004] to urban climatology [@Heusinkveld:2014). 

## Purpose of the software

The traditional approach to the development of a LUR model requires several steps: 
1. The creation of a point dataset of known observation data for a variable of interest (such as nitrogen dioxide concentrations or air temperatures), which will be used as outcome variables.
2. Carrying out various spatial analyses of additional geospatial data with these point locations using geographic information systems (GIS) to extract potential predictor variables.
3. Data wrangling of extracted data from steps 1 and 2 into a format that can be readily used by statistical software packages.
4. Carrying out multiple regression analysis to obtain a best fit parsimonious model.

If done manually, steps 2 to 4 are repetitive and time consuming, making them inefficient and prone to error. 
PyLUR provides a wizard style interface that guides a user 
through the development of a LUR model without the need to access and run multiple tools and 
additional software packages. PyLUR largely automates steps 2 to 4, which significantly speeds
up the model development process and reduces user error. Depending on the type and amount of data to be processed, 
and the available hardware, models can be developed in less than one hour. Furthermore, 
the output files produced by PyLUR ensure that the model development is 
well documented and reproducible. Lastly, PyLUR also makes the method available for a wider range of users.  

In addition to developing LUR models, PyLUR can also apply a previously developed 
model to a new set of locations within the same study area. New locations can be
defined by the user or can be based on a dataset containing regularly located points or randomly located points. 
Again, PyLUR largely automates this process with minimal effort from the user. 

PyLUR is aimed at GIS specialists. It uses the ArcGIS Pro software, which is the most widely 
used commercial GIS software worldwide. PyLUR is based on the LUR methodology used in the 
European Study of Cohorts for Air Pollution Effects (ESCAPE) [@Beelen:2013;@Eeftens:2012] as set out in the ESCAPE Exposure 
assessment manual [@escape]. Within air pollution research the ESCAPE methodology is used as
the standard for developing LUR models. PyLUR also allows hybrid LUR models to be developed, based on an extension 
of the ESCAPE methodology that included the addition of satellite derived data and data from chemical transport 
models [@de Hoogh:2016]. 

## Current application

PyLUR has been developed through the NERC Newton-DIPI funded Urban hybriD models for AiR pollution exposure Assessment (UDARA)
study, which is a collaboration between the University of Manchester and Institut Teknologi
Bandung. This study aims to develop air pollution prediction models for Indonesian urban areas and to analyse 
the effects of air pollution on health indicators provided by 
the Indonesian Family Life Survey.

# Acknowledgments

PyLUR was reviewed and tested by Prof S Lindley, Department of Geography, School of Environment, 
Education and Development, The University of Manchester. 
This work is supported via the NERC Newton-DIPI Urban hybriD models for AiR pollution exposure 
Assessment (UDARA) project, PIs: Prof G McFiggans, Faculty of Science and Engineering, 
The University of Manchester, UK, and Dr D Driejana, Faculty of Civil and Environmental Engineering,
Institut Teknologi Bandung, Indonesia, NE/P014631/1. It builds on work carried out in the 
European Union's Seventh Framework 
Programme Theme ENV.2007.1.2.2.2. European cohort on air pollution. 

# References
