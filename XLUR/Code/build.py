import wx
import wx.html2
import webbrowser
from wx.lib.masked import NumCtrl
from wx.lib.wordwrap import wordwrap
from wx.lib.agw import ultimatelistctrl as ULC
import wx.lib.inspection
import arcpy
import sys
import os
import logging
import sqlite3
import csv
import time
import pandas as pd
try:
    from pandas.tools.plotting import table
except ImportError:
    from pandas.plotting import table
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
import seaborn
from patsy import dmatrices
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.gofplots import ProbPlot
import itertools
import math
import string
import random
import shutil


#------------------------------------------------------------------------------
# User defined global functions
#------------------------------------------------------------------------------
# Function to capture system errors (Pokemon style) in file
def exception_hook(exc_type, exc_value, exc_traceback):
    logging.error( "Uncaught exception",exc_info=(exc_type, exc_value, exc_traceback))
    arcpy.AddMessage('+++ERROR+++ Uncaught exception -> See GOTCHA')
#------------------------------------------------------------------------------
# Function to check that a field name exists
def FieldExist(featureclass, fieldname):
    fieldList = arcpy.ListFields(featureclass, fieldname)
    fieldCount = len(fieldList)
    if (fieldCount == 1):
        return True
    else:
        return False
#-------------------------------------------------------------------------------
# Function to get unique values in field
def unique_values(fc , field):
    with arcpy.da.SearchCursor(fc, [field]) as cursor:
        return {row[0] for row in cursor}
#-------------------------------------------------------------------------------
# Function to create a customised field map
# This can be used to selectively import fields
# Uses output field names in the dictionary
def customFieldMap(in_fc,fldsNamesDict):
    inputFieldNames=list(fldsNamesDict.keys())#list of input field names
    customfieldmappings = arcpy.FieldMappings()# create an empty field mapping object
    for field in inputFieldNames:# for each field, create an individual field map, and add it to the field mapping object
        fmap = arcpy.FieldMap()
        fmap.addInputField(in_fc, field)
        outputFieldName = fmap.outputField
        outputFieldName.name = fldsNamesDict[field]#use output field name from dictionary
        outputFieldName.aliasName = fldsNamesDict[field]#also need to change alias name?
        fmap.outputField = outputFieldName
        customfieldmappings.addFieldMap(fmap)
    return customfieldmappings
#-------------------------------------------------------------------------------
# load feature classes in SQLite DB, adapted from https://tereshenkov.wordpress.com/2016/12/17/load-esri-geodatabase-tables-into-sqlite/
def run(SQL=None):
    return conn.execute(SQL).fetchall()
#------------------------------------------------------------------------------
def fc_to_sql(conn,source_fc):
    #getting info about the source feature class
    source_basename = arcpy.Describe(source_fc).baseName
    arcgis_sqlite_types_mappings = {'Date':'realdate','Double':'float64','Single':'float64',
                                    'Integer':'int32','SmallInteger':'int16','String':'text',
                                    'OID':'int32'}
    geometry_columns = ('shape')
    #use SQL to create a table
    columns = ['{} {}'.format(field.name,arcgis_sqlite_types_mappings[field.type])
               for field in arcpy.ListFields(source_fc) if field.name.lower() not in geometry_columns and field.type!='OID']
    #creating the table (with all columns except the geometry column)
    run('''CREATE TABLE {table} ({columns});'''.format(table=source_basename,
                                                       columns=','.join(columns)))
    #getting a list of column names to store the data
    data_columns = [str(field.name) for field in arcpy.ListFields(source_fc) if field.name.lower() not in geometry_columns and field.type!='OID']
    #creating a list of data rows
    rows = (r for r in arcpy.da.SearchCursor(source_fc,data_columns))
    #insert attribute values into the SQL table
    insert_values = ','.join(['?' for i in range(len(data_columns))])
    sql_insert_rows = '''Insert into {table_name} ({columns}) values ({insert_values})'''
    conn.executemany(sql_insert_rows.format(table_name=source_basename,
                                            columns=','.join(data_columns),
                                            insert_values=insert_values),rows)
    db.commit()
    return conn

#-------------------------------------------------------------------------------
# function to insert title pages into pdf
def title_page(txt):
    firstPage = plt.figure(figsize=(11.69,8.27)) # prints to A4 landscape
    firstPage.clf()
    firstPage.text(0.5,0.5,txt, transform=firstPage.transFigure, size=24, ha="center")
#-------------------------------------------------------------------------------
# function to insert title pages with a note at the bottom into pdf
def title_pagenote(txt,note):
    firstPage = plt.figure(figsize=(11.69,8.27)) # prints to A4 landscape
    firstPage.clf()
    firstPage.text(0.5,0.5,txt, transform=firstPage.transFigure, size=24, ha="center")
    firstPage.text(0.05,0.05,note, transform=firstPage.transFigure, size=16, ha="left",va="bottom")
#------------------------------------------------------------------------------
#function for boxplot and descriptive table
def plot_bxp(df,var):
    plt.figure(figsize=(11.69,8.27))
    plt.subplot(121)
    vals=df[var].dropna().values.tolist() # extract column as list, drop missing values
    xs=[random.gauss(1.0, 0.04) for _ in range(len(vals))] # random jitter for points
    plt.title(var)
    plt.boxplot(vals)
    plt.scatter(xs,vals,alpha=0.5)
    ptab=plt.subplot(122,frame_on=False)
    desc=df[var].describe().round(4)
    ptab.xaxis.set_visible(False)
    ptab.yaxis.set_visible(False)
    table(ptab, desc,loc='upper right',colWidths=[0.7])
#------------------------------------------------------------------------------
# function to plot correlation matrix
def plot_corr(df,width,height):
    corrmat=df.corr()
    fig,ax = plt.subplots()
    fig.set_size_inches(width,height) # prints to A4 landscape
    labels = corrmat.where(np.triu(np.ones(corrmat.shape)).astype(np.bool))
    labels = np.round(labels,decimals=2)
    labels = labels.replace(np.nan,' ', regex=True)
    mask = np.triu(np.ones(corrmat.shape)).astype(np.bool)
    ax = seaborn.heatmap(corrmat, mask=mask, cmap='bwr', fmt='', square=True, vmin=-1, vmax=1,linewidths=0.5,linecolor='black')
    mask = np.ones(corrmat.shape)-mask
    ax = seaborn.heatmap(corrmat, mask=mask, cmap=ListedColormap(['white']),annot=labels,cbar=False, fmt='', linewidths=0.5,linecolor='black')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xticklabels(labels.columns,rotation=90)
    plt.tight_layout()
#------------------------------------------------------------------------------
# function for matrix of pairwise regression plots
def plot_pwreg(df):
    g = seaborn.PairGrid(df)
    g.map_lower(seaborn.regplot,line_kws={'color': 'black'},ci=None)
    for i, j in zip(*np.triu_indices_from(g.axes, 0)):
        g.axes[i, j].set_visible(False)
#------------------------------------------------------------------------------
# function to recode values into positive or negative
def recode_coeff(dict):
    for k,v in dict.items():
        if k=='p_XCOORD' or k=='p_YCOORD':
            dict[k]=0
        elif v>0:
            dict[k]=1
        else:
            dict[k]=-1
#------------------------------------------------------------------------------
# function to check that direction of effect matches a priori assumption
def direction_effect(b,dict_lkup):
    r=b.items()<=dict_lkup.items()
    return r
#------------------------------------------------------------------------------
# function to replace items in a list with values from a dictionary
def replace_matched_items(word_list, dictionary):
   for lst in word_list:
      for ind, item in enumerate(lst):
          lst[ind] = dictionary.get(item, item)
#------------------------------------------------------------------------------
# function to calcuate VIF for each predictor
def var_inf_fac(formula,data):
    y,X=dmatrices(formula,data,return_type='dataframe')
    vif=pd.DataFrame()
    vif["Predictor variable"]=X.columns
    vif["VIF"]=[variance_inflation_factor(X.values, j) for j in range(X.shape[1])]
    vif.VIF = vif.VIF.round(2)
    return vif
#------------------------------------------------------------------------------
# function to create QQ plots
def qq_plot(model,var):
    model_std_res = model.get_influence().resid_studentized_internal # get standardized residuals
    QQ = ProbPlot(model_std_res)
    plt_qq = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
    plt_qq.set_figheight(8)
    plt_qq.set_figwidth(12)
    plt_qq.axes[0].set_title(var+' Normal Q-Q')
    plt_qq.axes[0].set_xlabel('Theoretical Quantiles')
    plt_qq.axes[0].set_ylabel('Standardized Residuals')
#------------------------------------------------------------------------------
# function to create residual plots
def res_plot(model,var):
    model_pred = model.fittedvalues # predicted values
    model_res = model.resid # Residuals
    plt_res = plt.figure(1)
    plt_res.set_figheight(8)
    plt_res.set_figwidth(12)
    plt_res.axes[0] = seaborn.residplot(model_pred,model_res,lowess=True,scatter_kws={'alpha': 0.5},line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plt_res.axes[0].set_title(var+' Residuals vs Predicted')
    plt_res.axes[0].set_xlabel('Predicted values')
    plt_res.axes[0].set_ylabel('Residuals')
#------------------------------------------------------------------------------
# function to create scale location plots to check for heteroskedasticity
def scloc_plot(model,var):
    model_std_res = model.get_influence().resid_studentized_internal # get standardized residuals
    model_std_res_abs_sqrt = np.sqrt(abs(model_std_res)) # square root of absolute standardised residual
    model_pred = model.fittedvalues # in lm fitted values are the same as predicted values
    plt_scloc = plt.figure(3)
    plt_scloc.set_figheight(8)
    plt_scloc.set_figwidth(12)
    plt.scatter(model_pred, model_std_res_abs_sqrt, alpha=0.5)
    seaborn.regplot(model_pred, model_std_res_abs_sqrt,scatter=False, ci=False, lowess=True,line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plt_scloc.axes[0].set_title(var+' Scale-Location')
    plt_scloc.axes[0].set_xlabel('Predicted values')
    plt_scloc.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');
#------------------------------------------------------------------------------
# function for leave one out cross validation
def loocv(dataframe,formula,dependent):
    loopred=pd.DataFrame()
    for i in range(len(dataframe)):
        tempdf=dataframe.drop(dataframe.index[[i]])
        model_cv=smf.ols(formula=formula,data=tempdf,missing='drop').fit()
        lpred=model_cv.predict(dataframe[i:i+1]).to_frame()
        loopred = loopred.append(lpred,ignore_index=True)
    loocv_df=dataframe[[dependent]]
    loocv_df=loocv_df.join(loopred).dropna()
    loocv_df.columns=['observed','predicted']
    loocv_df['squRes']=(loocv_df['observed']-loocv_df['predicted'])**2
    pearson=round(loocv_df['observed'].corr(loocv_df['predicted']),2)
    lm=smf.ols(formula='predicted~observed',data=loocv_df).fit()
    r2=round(lm.rsquared_adj,2)
    mse=loocv_df['squRes'].mean()
    rmse=round(math.sqrt(mse),2)
    axmax=max(loocv_df['observed'].max(),loocv_df['predicted'].max())*1.1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(loocv_df['observed'],loocv_df['predicted'],color='blue')
    ax.plot([0, 1], [0, 1], color='black', linewidth=1.5, linestyle='dashed',transform=ax.transAxes)
    ax.set_ylim(ymin=0,ymax=axmax)
    ax.set_xlim(xmin=0,xmax=axmax)
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')
    ax.set_title('Leave one out cross validation of '+dependent)
    ax.text(0.05,0.95, 'Pearson r= '+str(pearson)+'\n$Adj. R^2= $'+str(r2)+'\nRMSE= '+str(rmse),ha='left', va='top', transform=ax.transAxes)
    ax.grid(True)
#------------------------------------------------------------------------------
# function to save final model description
def final_model_out(dependent,model,formula,dataframe,filepath,sqliteconnection,dataframe_residuals):
    log.write('\n\nFinal model for '+dependent+':\n')
    log.write(model.summary().as_text())
    log.write('\n\nVariance Inflation of model\n')
    vif=var_inf_fac(formula,dataframe)
    log.write(vif.to_string())
    log.write('\n\nCase summaries\n')
    log.write(model.get_influence().summary_table().as_text())
    log.write('\n\nDF Betas\n')
    dfbetas=model.get_influence().summary_frame().filter(regex="dfb")
    log.write(dfbetas.to_string())
    log.write('\n')
    timestamp=time.strftime("%Y%m%d_%H%M%S")
    with PdfPages(filepath+'\\Diagnostic_plots_'+dependent+'_'+timestamp+'.pdf') as pdf:
        qq_plot(model,dependent)
        pdf.savefig()
        plt.close()
        res_plot(model,dependent)
        pdf.savefig()
        plt.close()
        scloc_plot(model,dependent)
        pdf.savefig()
        plt.close()
    loocv(dataframe,formula,dependent)
    plt.savefig(filepath+'\\LOOCV_'+dependent+'_'+timestamp+'.pdf')
    plt.close()
    df_coef=model.params.to_frame() # save model parameters to dataframe
    df_coef.reset_index(inplace=True) # make index column
    df_coef.columns=['X','coefficient'] # rename columns
    df_coef.to_sql('MODEL_'+dependent,sqliteconnection,if_exists='replace') # add to sqlite as table
    df_resid2=model.resid.to_frame() # save residuals to dataframe
    df_resid2.columns=['RES_'+dependent] # rename column to show var name
    df_resid2.reset_index(inplace=True) # make index column
    df_resid2['siteID_INT'] = df_resid2['index']+1 # make siteID_INT column based on index
    del df_resid2['index'] # delete index column
    global df_resid
    df_resid=dataframe_residuals.merge(df_resid2,on='siteID_INT',how='left') # join to dataframe based on siteID_INT


#------------------------------------------------------------------------------
# global variables, lists, dictionaries
#------------------------------------------------------------------------------
# current directory
curPath = os.path.dirname(os.path.abspath(__file__))

# use all cores for parallel processing
arcpy.env.parallelProcessingFactor = "100%"

# set auto cancelling to true
arcpy.env.autoCancelling = True

# tick marks
mark_empty = ' '
mark_done = '\u2713'

# Predictor variable names entered by User
var_list=list()

# Dictionary for buffer Distances
bufDists={}

# Dictionary of extraction methods
extrmet={'Distance':'dist','Inverse distance':'invd',
        'Inverse distance squared':'invsq','Value':'val',
        'Value * Distance':'valdist','Value * Inverse distance':'valinvd',
        'Value * Inverse distance squared':'valinvsq'}

# Dictionary for sources and sinks
sourcesink={'p_XCOORD':'Not applicable','p_YCOORD':'Not applicable'}

# variable names
pA_name = '999'
pB_name = '999'
pC_name = '999'
pD_name = '999'
pE_name = '999'
pF_name = '999'
pG_name = '999'

#list of mandatory Variables
force_vars=list()

#list of ASCII character codes
ascii_lower = list(range(97,122))
ascii_upper = list(range(65,90))
ascii_char = ascii_lower + ascii_upper + [8,9,127,314,316] #add delete etc

#list of ASCII char, number and underscore
ascii_num = list(range(48,58))
ascii_underscore = [95]
ascii_all = ascii_num + ascii_underscore + ascii_char
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Time code
eltime0 = time.clock()
time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
time_strt_sql = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
time_start = time.localtime()
arcpy.AddMessage((time.strftime("Start Time: %A %d %b %Y %H:%M:%S", time_start)))
# ------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Create wizard

class WizardPanel1(wx.Panel):
    """First page"""

    def __init__(self, parent, title=None):
        """Constructor"""
        wx.Panel.__init__(self, parent)

        wx.ToolTip.Enable(True)

        self.sizer = wx.GridBagSizer(0,0)

        title1 = wx.StaticText(self, -1, "Settings")
        title1.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(title1, pos=(0,0), flag=wx.TOP|wx.LEFT|wx.BOTTOM, border=10)

        self.sizer.Add(wx.StaticLine(self), pos=(1, 0), span=(1, 6),flag=wx.EXPAND|wx.BOTTOM, border=5)

        header0 = wx.StaticText(self,-1,label="Set Project Name")
        header0.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header0, pos=(2, 0), flag=wx.ALL, border=10)

        help_btn0 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn0.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn0.SetForegroundColour(wx.Colour(0,0,255))
        help_btn0.Bind(wx.EVT_BUTTON, self.onHlp0)
        self.sizer.Add(help_btn0, pos=(2,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text1 = wx.StaticText(self,-1,label="Project Name")
        self.sizer.Add(text1, pos=(3, 0), flag=wx.ALL, border=10)

        self.tc0 = wx.TextCtrl(self, -1)
        self.tc0.SetMaxLength(10)
        self.tc0.SetToolTip(wx.ToolTip('Enter a name for the LUR project'))
        self.tc0.Bind(wx.EVT_CHAR,self.onChar0)
        self.sizer.Add(self.tc0, pos=(3, 1), span=(1,2), flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=5)

        self.enter_btn0 = wx.Button(self, label="Enter")
        self.enter_btn0.Bind(wx.EVT_BUTTON, self.onEnt0)
        self.sizer.Add(self.enter_btn0, pos=(3,3), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)

        self.mark0 = wx.StaticText(self,-1,label=mark_empty)
        self.mark0.SetForegroundColour((0,255,0))
        self.mark0.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark0, pos=(3,4), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(4, 0), span=(1, 6),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        header1 = wx.StaticText(self,-1,label="Set Directories")
        header1.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header1, pos=(5, 0), flag=wx.ALL, border=10)

        help_btn1 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn1.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn1.SetForegroundColour(wx.Colour(0,0,255))
        help_btn1.Bind(wx.EVT_BUTTON, self.onHlp1)
        self.sizer.Add(help_btn1, pos=(5,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text2 = wx.StaticText(self,-1,label="Input File Geodatabase")
        self.sizer.Add(text2, pos=(6, 0), flag=wx.ALL, border=10)

        self.browse_btn1 = wx.DirPickerCtrl(self)
        self.browse_btn1.Bind(wx.EVT_DIRPICKER_CHANGED, self.onBrw1)
        self.browse_btn1.SetToolTip(wx.ToolTip('Select the File Geodatabase containing the data'))
        self.sizer.Add(self.browse_btn1, pos=(6,1), span=(1, 4), flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.browse_btn1.Disable()

        self.mark1 = wx.StaticText(self,-1,label=mark_empty)
        self.mark1.SetForegroundColour((0,255,0))
        self.mark1.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark1, pos=(6,5), flag=wx.TOP|wx.BOTTOM|wx.RIGHT, border=5)

        text4 = wx.StaticText(self, label="Output Folder")
        self.sizer.Add(text4, pos=(7, 0), flag=wx.ALL, border=10)

        self.browse_btn3 = wx.DirPickerCtrl(self)
        self.browse_btn3.Bind(wx.EVT_DIRPICKER_CHANGED, self.onBrw3)
        self.browse_btn3.SetToolTip(wx.ToolTip('Select a folder where the output files will be saved'))
        self.sizer.Add(self.browse_btn3, pos=(7, 1), span=(1, 4), flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.browse_btn3.Disable()

        self.mark3 = wx.StaticText(self,-1,label=mark_empty)
        self.mark3.SetForegroundColour((0,255,0))
        self.mark3.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark3, pos=(7,5), flag=wx.TOP|wx.BOTTOM|wx.RIGHT, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(8, 0), span=(1, 6),flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=5)

        header2 = wx.StaticText(self,-1,label="Set Coordinate System")
        header2.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header2, pos=(9, 0), flag=wx.ALL, border=10)

        help_btn2 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn2.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn2.SetForegroundColour(wx.Colour(0,0,255))
        help_btn2.Bind(wx.EVT_BUTTON, self.onHlp2)
        self.sizer.Add(help_btn2, pos=(9,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text5 = wx.StaticText(self, label="Enter WKID number")
        self.sizer.Add(text5, pos=(10, 0), flag=wx.ALL, border=10)

        self.tc1 = NumCtrl(self, integerWidth=6, fractionWidth=0, allowNegative=False, allowNone = True)
        self.tc1.SetToolTip(wx.ToolTip('Enter the WKID of the projected coordinate system of the study area'))
        self.sizer.Add(self.tc1, pos=(10, 1), span=(1,2), flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=10)
        self.tc1.Disable()

        enter_btn1 = wx.Button(self, label="OK")
        enter_btn1.Bind(wx.EVT_BUTTON, self.onEnt1)
        self.sizer.Add(enter_btn1, pos=(10,3), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=10)

        self.mark4 = wx.StaticText(self,-1,label=mark_empty)
        self.mark4.SetForegroundColour((0,255,0))
        self.mark4.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark4, pos=(10,4), flag=wx.ALL, border=10)

        self.sizer.Add(wx.StaticLine(self), pos=(11, 0), span=(1, 6),flag=wx.EXPAND|wx.BOTTOM, border=5)

        header3 = wx.StaticText(self,-1,label="Set Study Area")
        header3.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header3, pos=(12, 0), flag=wx.ALL, border=10)

        help_btn3 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn3.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn3.SetForegroundColour(wx.Colour(0,0,255))
        help_btn3.Bind(wx.EVT_BUTTON, self.onHlp3)
        self.sizer.Add(help_btn3, pos=(12,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text5 = wx.StaticText(self, label="Study Area Feature Class")
        self.sizer.Add(text5, pos=(13, 0), flag=wx.ALL, border=10)

        self.cb1 = wx.ComboBox(self,choices=[],style=wx.CB_DROPDOWN)
        self.cb1.SetToolTip(wx.ToolTip('From the dropdown list select the feature class containing a polygon of the study area'))
        self.sizer.Add(self.cb1, pos=(13, 1), span=(1,2),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.cb1.Bind(wx.EVT_COMBOBOX, self.onCb1)
        self.cb1.Disable()

        self.mark5 = wx.StaticText(self,-1,label=mark_empty)
        self.mark5.SetForegroundColour((0,255,0))
        self.mark5.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark5, pos=(13,3), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(14, 0), span=(1, 6),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        # add next button
        self.nextBtn1 = wx.Button(self, label="Next >")
        self.nextBtn1.Bind(wx.EVT_BUTTON, self.onNext1)
        self.sizer.Add(self.nextBtn1, pos=(15,4),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)
        self.nextBtn1.Disable()

        # add cancel button
        self.cancBtn1 = wx.Button(self, label="Cancel")
        self.cancBtn1.Bind(wx.EVT_BUTTON, self.onCanc1)
        self.cancBtn1.SetToolTip(wx.ToolTip('Cancel'))
        self.sizer.Add(self.cancBtn1, pos=(15,5),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        self.sizer.AddGrowableCol(1)
        self.SetSizer(self.sizer)
        self.Layout()
        self.Fit()

    def onHlp0(self,event):
        """Help window for setting project name"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p1_SetProjectName.html")
        htmlViewerInstance.Show()

    def onChar0(self,event):
        keycode = event.GetKeyCode()
        if keycode in ascii_all: #restrict input
            event.Skip()

    def onEnt0(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """enter variable name"""
        inp_prj = self.tc0.GetValue()
        if not inp_prj:
            wx.MessageBox('Please enter a project name','Error',wx.OK|wx.ICON_ERROR)
        elif inp_prj[0].isalpha()==False:
            wx.MessageBox('The project name must start with a letter','Error',wx.OK|wx.ICON_ERROR)
        else: # change this to be based on the database, so that entry can be changed
            global prj_name
            prj_name = inp_prj+"_"+time_str
            self.Parent.statusbar.SetStatusText('Ready')
            del wait
            self.browse_btn1.Enable()
            self.tc0.Disable()
            self.enter_btn0.Disable()
            self.mark0.SetLabel(mark_done) # change tick mark to done

    def onHlp1(self,event):
        """Help window for setting directories"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p1_SetDirectories.html")
        htmlViewerInstance.Show()

    def onBrw1(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Get input fgdb path"""
        global in_fgdb
        in_fgdb = self.browse_btn1.GetPath()
        if in_fgdb[-4:]!='.gdb':
            wx.MessageBox('Invalid selection.\nPlease select a File Geodatabase.','Error',wx.OK|wx.ICON_ERROR)
        else:
            arcpy.env.workspace = in_fgdb
            # lists of feature classes in fgdb
            polys = list()
            lines = list()
            points = list()
            # search through fgdb and datasets in fgdb
            datasets = arcpy.ListDatasets(feature_type='feature')
            datasets = [''] + datasets if datasets is not None else []
            for ds in datasets:
                for fc in arcpy.ListFeatureClasses('','Polygon',feature_dataset=ds):
                    path = os.path.join(ds, fc)
                    polys.append(path)
                for fc in arcpy.ListFeatureClasses('','Polyline',feature_dataset=ds):
                    path = os.path.join(ds, fc)
                    lines.append(path)
                for fc in arcpy.ListFeatureClasses('','Point',feature_dataset=ds):
                    path = os.path.join(ds, fc)
                    points.append(path)
            rasters=arcpy.ListRasters("*",'') # raster can't be in feature dataset, so no subsearch needed
            polys.sort()
            lines.sort()
            points.sort()
            rasters.sort()
            # append polys to cb1 in this panel
            self.cb1.SetValue('')
            self.cb1.Clear()
            self.cb1.Append(polys)
            # append point files to cb1 in panel 2
            self.Parent.panel2.cb1.SetValue('')
            self.Parent.panel2.cb1.Clear()
            self.Parent.panel2.cb1.Append(points)
            # append polygon files to cb1 in panel 3A
            self.Parent.panel3A.cb1.SetValue('')
            self.Parent.panel3A.cb1.Clear()
            self.Parent.panel3A.cb1.Append(polys)
            # append line files to cb1 in panel 3B
            self.Parent.panel3B.cb1.SetValue('')
            self.Parent.panel3B.cb1.Clear()
            self.Parent.panel3B.cb1.Append(lines)
            # append line files to cb1 in panel 3C
            self.Parent.panel3C.cb1.SetValue('')
            self.Parent.panel3C.cb1.Clear()
            self.Parent.panel3C.cb1.Append(points)
            # append polygon files to cb1 in panel 3D
            self.Parent.panel3D.cb1.SetValue('')
            self.Parent.panel3D.cb1.Clear()
            self.Parent.panel3D.cb1.Append(polys)
            # append line files to cb1 in panel 3E
            self.Parent.panel3E.cb1.SetValue('')
            self.Parent.panel3E.cb1.Clear()
            self.Parent.panel3E.cb1.Append(lines)
            # append point files to cb1 in panel 3F
            self.Parent.panel3F.cb1.SetValue('')
            self.Parent.panel3F.cb1.Clear()
            self.Parent.panel3F.cb1.Append(points)
            # append raster files to cb1 in panel 3G
            self.Parent.panel3G.cb1.SetValue('')
            self.Parent.panel3G.cb1.Clear()
            self.Parent.panel3G.cb1.Append(rasters)

            self.browse_btn3.Enable()
            self.mark1.SetLabel(mark_done) # change tick mark to done
            self.Parent.statusbar.SetStatusText('Ready') # change status bar
            del wait

    def onBrw3(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Get output path """
        out_folder = self.browse_btn3.GetPath()
        if out_folder[-4:]=='.gdb':
            wx.MessageBox('Invalid selection.\nThe output folder must not be a File Geodatabase.','Error',wx.OK|wx.ICON_ERROR)
        else:
            global out_path
            out_path = os.path.join(out_folder,prj_name)
            os.mkdir(out_path) # create project folder in output
            arcpy.CreateFileGDB_management(out_path, prj_name)
            global out_fgdb
            out_fgdb = out_path+"\\"+prj_name+".gdb"
            global log
            log = open(out_path+'\\LOG_'+time_str+'.txt', 'w')# create log file
            log.write(time.strftime("Start Time: %A %d %b %Y %H:%M:%S", time_start))
            log.write(time.strftime("\n\nSettings - Start Time: %A %d %b %Y %H:%M:%S", time_start))
            log.write('\n\nInput File Geodatabase: '+in_fgdb)
            log.write('\nOutput Folder: '+out_path)
            #create error file
            logging.basicConfig(filename=out_path+'\\GOTCHA.log',filemode='w',level=logging.DEBUG,format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',datefmt="%Y-%m-%d %H:%M:%S")
            sys.excepthook = exception_hook
            # Create SQLite database
            global db
            db = sqlite3.connect(out_path+"\\LurSqlDB.sqlite")
            # cursor object
            global conn
            conn = db.cursor()
            #turn journal off, i.e. no rollback
            conn.execute('''PRAGMA journal_mode=OFF;''')
            conn.execute('''PRAGMA auto_vacuum=FULL;''')
            conn.execute('''PRAGMA SQLITE_DEFAULT_CACHE_SIZE=-8000;''')
            conn.execute('''PRAGMA SQLITE_DEFAULT_SYNCHRONOUS=0''') # will lead to corruption during power loss or OS crash, but not app crash!
            db.commit()
            #create timings table
            conn.execute('''CREATE TABLE timings (Panel Int, Step Varchar, vTime DateTime)''')
            conn.execute('''INSERT INTO timings VALUES('panel1','start',datetime())''')
            db.commit()

            self.tc1.Enable()
            self.mark3.SetLabel(mark_done)# change tick mark to done
            self.Parent.statusbar.SetStatusText('Ready') # change status bar
            del wait
            self.browse_btn1.Disable()
            self.browse_btn3.Disable()
            log.flush()

    def onHlp2(self,event):
        """Help window for setting coordinate system"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p1_SetCoordinateSystem.html")
        htmlViewerInstance.Show()

    def onEnt1(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Enter WKID"""
        sr_inp = self.tc1.GetValue()
        try:
            global sr
            sr = arcpy.SpatialReference(sr_inp)
            if sr.type=="Geographic":
                wx.MessageBox('This is a geographic coordinate system','Error',wx.OK|wx.ICON_ERROR)
            elif sr.type=="Projected":
                global out_fds
                out_fds = out_fgdb+"\\LURdata"
                arcpy.CreateFeatureDataset_management(out_fgdb, 'LURdata', sr) # create feature dataset
                desc = arcpy.Describe(out_fds)
                wx.MessageBox("The coordinate system is set to: {0}".format(desc.spatialReference.name),'Information',wx.OK|wx.ICON_INFORMATION)
                log.write("\nThe coordinate system is set to: {0}".format(desc.spatialReference.name))
                log.write("\nUnit of measurement: {0}".format(desc.spatialReference.linearUnitName))
                self.cb1.Enable()
                self.mark4.SetLabel(mark_done) # change tick mark to done
                self.Parent.statusbar.SetStatusText('Ready') # change status bar
                del wait
                log.flush()
        except:
            wx.MessageBox('Invalid input','Error',wx.OK|wx.ICON_ERROR)

    def onHlp3(self,event):
        """Help window for setting study area"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p1_SetStudyArea.html")
        htmlViewerInstance.Show()

    def onCb1(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Select study area polygon"""
        global fc_starea
        fc_starea = self.cb1.GetValue() #input file
        num_row=int(arcpy.GetCount_management(fc_starea).getOutput(0))
        if num_row==1:
            arcpy.FeatureClassToFeatureClass_conversion(fc_starea,out_fds,'studyArea')
            log.write('\nStudy Area Feature Class: '+fc_starea+'\n')
            self.nextBtn1.Enable()
            self.mark5.SetLabel(mark_done) # change tick mark to done
            self.Parent.statusbar.SetStatusText('Ready') # change status bar
            self.tc1.Disable()
            del wait
        elif num_row==0:#check that it contains features
            wx.MessageBox('The selected feature class is empty','Error',wx.OK|wx.ICON_ERROR)
        else:# more than 1 feature
            wx.MessageBox('The selected feature class contains more than one polygon.\nPlease use a feature class containing a single polygon.','Error',wx.OK|wx.ICON_ERROR)

    def onNext1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Next Page"""
        log.write(time.strftime("\nSettings - End Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        log.write(time.strftime("\nOutcomes - Start Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel1','stop',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel2','start',datetime())''')
        db.commit()
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel2,1,wx.EXPAND)
        newsize = self.Parent.panel2.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel2.Show()
        self.Parent.statusbar.SetStatusText('Ready') # change status bar
        del wait
        log.flush()

    def onCanc1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Cancel Wizard"""
        dlg = wx.MessageDialog(None, "Cancelling the wizard will delete all progress. Do you wish to cancel the wizard?",'Cancel',wx.YES_NO | wx.ICON_WARNING)
        result = dlg.ShowModal()
        if result == wx.ID_YES:
            arcpy.AddMessage((time.strftime("Cancelled by user: %A %d %b %Y %H:%M:%S", time.localtime())))
            try:
                log.close()
                logging.shutdown()
                db.close()
                shutil.rmtree(out_path)
            except NameError as e:
                arcpy.AddMessage(e)
            except OSError as e:
                arcpy.AddMessage(e)
            del wait
            self.Parent.Close()

        else:
            self.Parent.statusbar.SetStatusText('Ready')
            del wait


#-------------------------------------------------------------------------------
class WizardPanel2(wx.Panel):
    """Second page"""

    def __init__(self, parent, title=None):
        """Constructor"""
        wx.Panel.__init__(self, parent)

        wx.ToolTip.Enable(True)

        self.sizer = wx.GridBagSizer(0,0)

        title2 = wx.StaticText(self, -1, "Outcomes")
        title2.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(title2, pos=(0,0), flag=wx.TOP|wx.LEFT|wx.BOTTOM, border=10)

        self.sizer.Add(wx.StaticLine(self), pos=(1, 0), span=(1, 5),flag=wx.EXPAND|wx.BOTTOM, border=5)

        header1 = wx.StaticText(self,-1,label="Set Dependent Variable")
        header1.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header1, pos=(2, 0), flag=wx.ALL, border=10)

        help_btn1 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn1.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn1.SetForegroundColour(wx.Colour(0,0,255))
        help_btn1.Bind(wx.EVT_BUTTON, self.onHlp1)
        self.sizer.Add(help_btn1, pos=(2,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text2 = wx.StaticText(self,-1,label="Monitoring Sites")
        self.sizer.Add(text2, pos=(3, 0), flag=wx.ALL, border=10)

        self.cb1 = wx.ComboBox(self,choices=[],style=wx.CB_DROPDOWN)
        self.cb1.SetToolTip(wx.ToolTip('From the dropdown list select the feature class containing the monitoring sites and measured values'))
        self.sizer.Add(self.cb1, pos=(3, 1), span=(1,2),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.cb1.Bind(wx.EVT_COMBOBOX, self.onCb1)

        self.mark1 = wx.StaticText(self,-1,label=mark_empty)
        self.mark1.SetForegroundColour((0,255,0))
        self.mark1.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark1, pos=(3,3), flag=wx.ALL, border=5)

        text3 = wx.StaticText(self, label="Select Site ID")
        self.sizer.Add(text3, pos=(4, 0), flag=wx.ALL, border=10)

        self.cb2 = wx.ComboBox(self,choices=[],style=wx.CB_DROPDOWN)
        self.cb2.SetToolTip(wx.ToolTip('From the dropdown list select the field containing unique IDs for each site'))
        self.sizer.Add(self.cb2, pos=(4, 1), span=(1,2),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.cb2.Bind(wx.EVT_COMBOBOX, self.onCb2)
        self.cb2.Disable()

        self.mark2 = wx.StaticText(self,-1,label=mark_empty)
        self.mark2.SetForegroundColour((0,255,0))
        self.mark2.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark2, pos=(4,3), flag=wx.ALL, border=5)

        text4 = wx.StaticText(self, label="Dependent Variables")
        self.sizer.Add(text4, pos=(5, 0), flag=wx.ALL, border=10)

        self.chkbx1 = wx.CheckListBox(self,choices=[])
        self.chkbx1.SetToolTip(wx.ToolTip('Tick all fields that contain measured values that will be modelled'))
        self.sizer.Add(self.chkbx1, pos=(5, 1), span=(5,2),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.chkbx1.Disable()

        sel_btn1 = wx.Button(self, label="Select")
        sel_btn1.Bind(wx.EVT_BUTTON, self.onSel1)
        self.sizer.Add(sel_btn1, pos=(5,3), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)

        self.mark3 = wx.StaticText(self,-1,label=mark_empty)
        self.mark3.SetForegroundColour((0,255,0))
        self.mark3.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark3, pos=(5,4), flag=wx.ALL, border=5)

        self.sizer.Add( wx.StaticLine(self), pos=(10, 0), span=(1, 5),flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=5)

        header2 = wx.StaticText(self,-1,label="Outcomes Added")
        header2.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header2, pos=(11, 0), flag=wx.ALL, border=10)

        self.list_ctrl = wx.ListCtrl(self,-1,style=wx.LC_REPORT|wx.LC_NO_HEADER)
        self.list_ctrl.InsertColumn(0,"",width=wx.LIST_AUTOSIZE)
        self.sizer.Add(self.list_ctrl, pos=(12, 0), span=(1, 5),flag=wx.EXPAND|wx.LEFT|wx.RIGHT|wx.BOTTOM, border=10)

        self.sizer.Add( wx.StaticLine(self), pos=(13, 0), span=(1, 5),flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=5)

        # add next button
        self.nextBtn = wx.Button(self, label="Next >")
        self.nextBtn.Bind(wx.EVT_BUTTON, self.onNext)
        self.sizer.Add(self.nextBtn, pos=(14,3),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)
        self.nextBtn.Disable()

        # add cancel button
        self.cancBtn1 = wx.Button(self, label="Cancel")
        self.cancBtn1.Bind(wx.EVT_BUTTON, self.onCanc1)
        self.cancBtn1.SetToolTip(wx.ToolTip('Cancel'))
        self.sizer.Add(self.cancBtn1, pos=(14,4),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        self.sizer.AddGrowableCol(1)
        self.sizer.AddGrowableRow(6)
        self.SetSizer(self.sizer)
        self.Layout()
        self.Fit()


    def onHlp1(self,event):
        """Help window for setting directories"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p2_SetDependentVariable.html")
        htmlViewerInstance.Show()

    def onCb1(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Select monitoring sites featureclass"""
        global fc_dep
        fc_dep = self.cb1.GetValue()
        global fldsNamesDict
        # check for spatial duplicates
        stop = 0
        statDict = {} # create an empty dictionary
        searchRows = arcpy.da.SearchCursor(fc_dep, ["SHAPE@WKT","OID@"]) # use data access search cursor to get OID and geometry
        for searchRow in searchRows:
            geomValue,oidValue = searchRow
            if geomValue in statDict:
                wx.MessageBox('Spatial duplicates found','Error',wx.OK|wx.ICON_ERROR)
                stop = 1
                break
            else:
                statDict[geomValue] = [oidValue]
        # check fc contains right fields
        num_row=int(arcpy.GetCount_management(fc_dep).getOutput(0))
        if num_row>0 and stop==0:
            str_fields = [f.name for f in arcpy.ListFields(fc_dep,"","String")] # get text fields
            str_fields.sort()
            if not str_fields:
                wx.MessageBox('The selected feature class contains no text fields','Error',wx.OK|wx.ICON_ERROR)
            else:
                self.cb2.Clear()
                self.cb2.Append(str_fields)

            num_fields = [f.name for f in arcpy.ListFields(fc_dep,"",'Double') if not f.required] #get numeric fields
            num_fields.extend([f.name for f in arcpy.ListFields(fc_dep,"",'Integer')])
            num_fields.extend([f.name for f in arcpy.ListFields(fc_dep,"",'Single')])
            num_fields.extend([f.name for f in arcpy.ListFields(fc_dep,"",'SmallInteger')])
            num_fields.sort()
            if not num_fields:
                wx.MessageBox('The selected feature class contains no numeric fields','Error',wx.OK|wx.ICON_ERROR)
            else:
                self.chkbx1.Clear()
                self.chkbx1.Append(num_fields)
                fldsNamesDict={}
                log.write('\nMonitoring Sites Feature Class: '+fc_dep)
                self.cb2.Enable()
                self.mark1.SetLabel(mark_done) # change tick mark to done
        elif num_row==0:#check that it contains features
            wx.MessageBox('The selected feature class is empty','Error',wx.OK|wx.ICON_ERROR)

        self.Parent.statusbar.SetStatusText('Ready')
        del wait


    def onCb2(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Select monitoring sites"""
        global inp_uid
        inp_uid = self.cb2.GetValue()
        # strip any endspaces from strings and replace whitespace with underscore
        with arcpy.da.UpdateCursor(fc_dep, inp_uid) as cursor:
            for row in cursor:
                row=[i.strip() if i is not None else None for i in row]
                row=[i.replace(" ","_") if i is not None else None for i in row]
                cursor.updateRow(row)
        # Check for duplicates in unique ID
        statDict = {} # create an empty dictionary
        searchRows = arcpy.da.SearchCursor(fc_dep, [str(inp_uid),"OID@"])
        for searchRow in searchRows:
            uidValue,oidValue = searchRow
            if uidValue in statDict:
                wx.MessageBox('Duplicate IDs found','Error',wx.OK|wx.ICON_ERROR)
                break
            else:
                statDict[uidValue] = [oidValue]
        # Check for NULL values in unique ID
        with arcpy.da.SearchCursor(fc_dep,str(inp_uid)) as cursor: #create a cursor that looks at all fields in siteID
            for row in cursor:
                if row[0] is None:
                    wx.MessageBox('Missing values found','Error',wx.OK|wx.ICON_ERROR)
                    break
        # add uid to dictionary
        new_entry={str(inp_uid):'siteID'}
        fldsNamesDict.update(new_entry)
        log.write('\nSite ID variable: '+inp_uid)
        self.chkbx1.Enable()
        self.mark2.SetLabel(mark_done) # change tick mark to done
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onSel1(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """select pollutants"""
        global poll_inp
        poll_inp = self.chkbx1.GetCheckedStrings()
        for field in poll_inp:
            new_entry={field:"dep"+field}
            fldsNamesDict.update(new_entry)
        # create fieldmap
        fieldmappings=customFieldMap(fc_dep,fldsNamesDict)
        # copy the feature class
        if arcpy.Exists(out_fds+"\\depVar")==True:
            arcpy.Delete_management(out_fds+"\\depVar")
        arcpy.FeatureClassToFeatureClass_conversion(fc_dep,out_fds,"depVar", "", fieldmappings)
        # set depVar
        global depVar
        depVar=out_fds+"\\depVar"
        # add integer ID
        arcpy.AddField_management(depVar,"siteID_INT", "SHORT")
        with arcpy.da.UpdateCursor(depVar,"siteID_INT") as cursor:
            i=0
            for row in cursor:
                i=i+1
                row[0]=i
                cursor.updateRow(row)
        # Add XY to dependent variable
        arcpy.AddXY_management(depVar)
        arcpy.AlterField_management(depVar, 'POINT_X', 'p_XCOORD', 'p_XCOORD')
        arcpy.AlterField_management(depVar, 'POINT_Y', 'p_YCOORD', 'p_YCOORD')
        # Check that all points are within study area
        arcpy.MakeFeatureLayer_management(depVar, out_fds+"\\depVar_lyr")
        arcpy.SelectLayerByLocation_management(out_fds+"\\depVar_lyr", 'WITHIN', out_fds+"\\studyArea")
        #count features
        origcount=int(arcpy.GetCount_management(depVar).getOutput(0))
        selcount=int(arcpy.GetCount_management(out_fds+"\\depVar_lyr").getOutput(0))
        if origcount>selcount:
            wx.MessageBox('One or more sites are located outside of the study area.','Error',wx.OK|wx.ICON_ERROR)
            arcpy.Delete_management(depVar)
            arcpy.Delete_management(out_fds+"\\depVar_lyr")
        else:
            # Quick check of values => NULL, Zero, Negative
            fields = dict((f.name, []) for f in arcpy.ListFields(depVar) if not f.required and f.name!='siteID')
            # create cursor and add values to keys in dictionary
            cursor = arcpy.SearchCursor(depVar)
            for row in cursor:
                for f in list(fields.keys()):
                    fields[f].append(row.getValue(f))
            for field, values in fields.items():
                boo=[s is None for s in values]
                zer=[s==0 for s in values]
                neg=[s is not None and s<0 for s in values]
                if sum(boo)>0 or sum(zer)>0 or sum(neg)>0:
                    wx.MessageBox('One or more of the dependent variables contains missing, zero or negative values','Warning',wx.OK|wx.ICON_WARNING)
                    break
            # dependent names to listctrl
            index=0
            for field in poll_inp:
                i = "dep"+field
                self.list_ctrl.InsertItem(index,i)
                if index % 2:
                    self.list_ctrl.SetItemBackgroundColour(index, col="white")
                else:
                    self.list_ctrl.SetItemBackgroundColour(index, col=(222,234,246))
                index+=1
            self.list_ctrl.SetColumnWidth(0,wx.LIST_AUTOSIZE)
            self.nextBtn.Enable()
        # delete layer
        arcpy.Delete_management(out_fds+"\\depVar_lyr")
        log.write('\nDependent variables: '+str(poll_inp))
        self.mark3.SetLabel(mark_done) # change tick mark to done
        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        self.cb1.Disable()
        self.cb2.Disable()
        self.chkbx1.Disable()

    def onNext(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """"""
        # create feature class with sites only
        fldsNamesDict={}
        new_entry={"siteID_INT":"siteID_INT"}
        fldsNamesDict.update(new_entry)
        fieldmappings=customFieldMap(depVar,fldsNamesDict) # create fieldmap
        if arcpy.Exists(out_fds+"\\sites")==True:
            arcpy.Delete_management(out_fds+"\\sites")
        arcpy.FeatureClassToFeatureClass_conversion(depVar,out_fds,"sites", "", fieldmappings)
        global sites
        sites = out_fds+"\\sites"
        # copy to sql db
        fc_to_sql(conn,depVar)
        conn.execute('''CREATE UNIQUE INDEX depVar_idx on depVar(siteID_INT);''')
        conn.execute('''CREATE TABLE dat4stats AS
                        SELECT *
                        FROM depVar''')
        conn.execute('''CREATE UNIQUE INDEX dat4stats_idx on dat4stats(siteID_INT);''')
        db.commit()
        # predictor names to panel 3
        cursor = conn.execute('''SELECT * FROM dat4stats''')
        varnames = list([x[0] for x in cursor.description])
        prednames = [name for name in varnames if (name[0] =='p')]
        global WizardPanel3
        WizardPanel3 = self.Parent.panel3
        index=0
        for i in prednames:
            WizardPanel3.list_ctrl.InsertItem(index,i)
            WizardPanel3.list_ctrl.SetItem(index,1,sourcesink[i])
            if index % 2:
                WizardPanel3.list_ctrl.SetItemBackgroundColour(index, col="white")
            else:
                WizardPanel3.list_ctrl.SetItemBackgroundColour(index, col=(222,234,246))
            index+=1
        WizardPanel3.list_ctrl.SetColumnWidth(0,wx.LIST_AUTOSIZE)
        WizardPanel3.list_ctrl.SetColumnWidth(1,wx.LIST_AUTOSIZE)

        log.write(time.strftime("\n\nOutcomes - End Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        log.write(time.strftime("\nPredictors - Start Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel2','stop',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel3','start',datetime())''')
        db.commit()
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel3,1,wx.EXPAND)
        newsize = self.Parent.panel3.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel3.Show()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onCanc1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Cancel Wizard"""
        dlg = wx.MessageDialog(None, "Cancelling the wizard will delete all progress. Do you wish to cancel the wizard?",'Cancel',wx.YES_NO | wx.ICON_WARNING)
        result = dlg.ShowModal()
        if result == wx.ID_YES:
            arcpy.AddMessage((time.strftime("Cancelled by user: %A %d %b %Y %H:%M:%S", time.localtime())))
            try:
                log.close()
                logging.shutdown()
                db.close()
                shutil.rmtree(out_path)
            except NameError as e:
                arcpy.AddMessage(e)
            except OSError as e:
                arcpy.AddMessage(e)
            del wait
            self.Parent.Close()
        else:
            self.Parent.statusbar.SetStatusText('Ready')
            del wait

#-------------------------------------------------------------------------------
class WizardPanel3(wx.Panel):
    """Third page"""

    def __init__(self, parent, title=None):
        """Constructor"""
        wx.Panel.__init__(self, parent)
        wx.ToolTip.Enable(True)
        self.sizer = wx.GridBagSizer(0,0)

        title2 = wx.StaticText(self, -1, "Predictors")
        title2.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(title2, pos=(0,0),flag=wx.TOP|wx.LEFT|wx.BOTTOM, border=10)

        line1 = wx.StaticLine(self)
        self.sizer.Add(line1, pos=(1, 0), span=(1, 5),flag=wx.EXPAND|wx.BOTTOM, border=5)

        header1 = wx.StaticText(self,-1,label="Which Type of Predictor Variable would you like to add?")
        header1.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header1, pos=(2, 0), span=(1,2),flag=wx.ALL, border=10)

        help_btn1 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn1.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn1.SetForegroundColour(wx.Colour(0,0,255))
        help_btn1.Bind(wx.EVT_BUTTON, self.onHlp1)
        self.sizer.Add(help_btn1, pos=(2,2), flag=wx.TOP|wx.BOTTOM, border=5)

        box = wx.StaticBox(self, label='')
        boxSizer = wx.StaticBoxSizer(box, wx.VERTICAL)
        gbs = wx.GridBagSizer(0, 0)

        text2 = wx.StaticText(self,-1,label="A. Polygon Area or Value within Buffer")
        gbs.Add(text2, pos=(0, 0), flag=wx.ALL, border=10)
        add_btnA = wx.Button(self, label="Add")
        add_btnA.Bind(wx.EVT_BUTTON, self.onAddA)
        gbs.Add(add_btnA, pos=(0,1), flag=wx.ALL, border=5)

        text3 = wx.StaticText(self, label="B. Line Length or Value within Buffer")
        gbs.Add(text3, pos=(1, 0), flag=wx.ALL, border=10)
        add_btnB = wx.Button(self, label="Add")
        add_btnB.Bind(wx.EVT_BUTTON, self.onAddB)
        gbs.Add(add_btnB, pos=(1,1), flag=wx.ALL, border=5)

        text4 = wx.StaticText(self, label="C. Point Count or Value within Buffer")
        gbs.Add(text4, pos=(2, 0), flag=wx.ALL, border=10)
        add_btnC = wx.Button(self, label="Add")
        add_btnC.Bind(wx.EVT_BUTTON, self.onAddC)
        gbs.Add(add_btnC, pos=(2,1), flag=wx.ALL, border=5)

        gbs.AddGrowableCol(0)
        boxSizer.Add(gbs, flag=wx.EXPAND)
        self.sizer.Add(boxSizer, pos=(3, 0), span=(3,3), flag=wx.EXPAND|wx.ALL, border=5)

        box2 = wx.StaticBox(self, label='')
        boxSizer2 = wx.StaticBoxSizer(box2, wx.VERTICAL)
        gbs2 = wx.GridBagSizer(0, 0)

        text5 = wx.StaticText(self, label="D. Distance to and/or Value of nearest Polygon")
        gbs2.Add(text5, pos=(0, 0), flag=wx.ALL, border=10)
        add_btnD = wx.Button(self, label="Add")
        add_btnD.Bind(wx.EVT_BUTTON, self.onAddD)
        gbs2.Add(add_btnD, pos=(0,1), flag=wx.ALL, border=5)

        text6 = wx.StaticText(self, label="E. Distance to and/or Value of nearest Line")
        gbs2.Add(text6, pos=(1, 0), flag=wx.ALL, border=10)
        add_btnE = wx.Button(self, label="Add")
        add_btnE.Bind(wx.EVT_BUTTON, self.onAddE)
        gbs2.Add(add_btnE, pos=(1,1), flag=wx.ALL, border=5)

        text7 = wx.StaticText(self, label="F. Distance to and/or Value of nearest Point")
        gbs2.Add(text7, pos=(2, 0), flag=wx.ALL, border=10)
        add_btnF = wx.Button(self, label="Add")
        add_btnF.Bind(wx.EVT_BUTTON, self.onAddF)
        gbs2.Add(add_btnF, pos=(2,1), flag=wx.ALL, border=5)

        gbs2.AddGrowableCol(0)
        boxSizer2.Add(gbs2, flag=wx.EXPAND)
        self.sizer.Add(boxSizer2, pos=(6, 0), span=(3,3), flag=wx.EXPAND|wx.LEFT|wx.RIGHT|wx.BOTTOM, border=5)

        box3 = wx.StaticBox(self, label='')
        boxSizer3 = wx.StaticBoxSizer(box3, wx.VERTICAL)
        gbs3 = wx.GridBagSizer(0, 0)

        text8 = wx.StaticText(self, label="G. Value of Raster cell")
        gbs3.Add(text8, pos=(0, 0), flag=wx.ALL, border=10)
        add_btnG = wx.Button(self, label="Add")
        add_btnG.Bind(wx.EVT_BUTTON, self.onAddG)
        gbs3.Add(add_btnG, pos=(0,1), flag=wx.ALL, border=5)

        gbs3.AddGrowableCol(0)
        boxSizer3.Add(gbs3, flag=wx.EXPAND)
        self.sizer.Add(boxSizer3, pos=(9, 0),span=(1,3), flag=wx.EXPAND|wx.LEFT|wx.RIGHT|wx.BOTTOM, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(10, 0), span=(1, 5),flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=5)

        header2 = wx.StaticText(self,-1,label="Predictors Added")
        header2.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header2, pos=(11, 0), span=(1,2),flag=wx.ALL, border=10)

        self.list_ctrl = wx.ListCtrl(self,-1,style=wx.LC_REPORT)
        self.list_ctrl.InsertColumn(0,"Name",wx.LIST_FORMAT_LEFT, wx.LIST_AUTOSIZE)
        self.list_ctrl.InsertColumn(1,"Type",wx.LIST_FORMAT_CENTRE, wx.LIST_AUTOSIZE)
        self.sizer.Add(self.list_ctrl, pos=(12, 0), span=(1, 3),flag=wx.EXPAND|wx.LEFT|wx.RIGHT|wx.BOTTOM, border=10)

        self.sizer.Add(wx.StaticLine(self), pos=(13, 0), span=(1, 5),flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=5)

        # add next button
        self.nextBtn = wx.Button(self, label="Next >")
        self.nextBtn.Bind(wx.EVT_BUTTON, self.onNext)
        self.sizer.Add(self.nextBtn, pos=(14,1),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)
        #self.nextBtn.Disable()

        # add cancel button
        self.cancBtn1 = wx.Button(self, label="Cancel")
        self.cancBtn1.Bind(wx.EVT_BUTTON, self.onCanc1)
        self.cancBtn1.SetToolTip(wx.ToolTip('Cancel'))
        self.sizer.Add(self.cancBtn1, pos=(14,2),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        self.sizer.AddGrowableCol(1)
        self.sizer.AddGrowableRow(12)
        self.SetSizer(self.sizer)
        self.Layout()
        self.Fit()

    def onHlp1(self,event):
        """Help window for setting predictors"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3_TypeOfPredictor.html")
        htmlViewerInstance.Show()

    def onAddA(self, event):
        """open panel3A"""
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel3A,1,wx.EXPAND)
        newsize = self.Parent.panel3A.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel3A.Show()
        log.write(time.strftime("\nPolygon in Buffer - Start Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel3','stop',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel3A','start',datetime())''')
        db.commit()

    def onAddB(self, event):
        """open panel3B"""
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel3B,1,wx.EXPAND)
        newsize = self.Parent.panel3B.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel3B.Show()
        log.write(time.strftime("\nLine in Buffer - Start Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel3','stop',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel3B','start',datetime())''')
        db.commit()

    def onAddC(self, event):
        """open panel3C"""
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel3C,1,wx.EXPAND)
        newsize = self.Parent.panel3C.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel3C.Show()
        log.write(time.strftime("\nPoint in Buffer - Start Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel3','stop',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel3C','start',datetime())''')
        db.commit()

    def onAddD(self, event):
        """open panel3D"""
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel3D,1,wx.EXPAND)
        newsize = self.Parent.panel3D.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel3D.Show()
        log.write(time.strftime("\nPolygon Distance - Start Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel3','stop',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel3D','start',datetime())''')
        db.commit()

    def onAddE(self, event):
        """open panel3E"""
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel3E,1,wx.EXPAND)
        newsize = self.Parent.panel3E.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel3E.Show()
        log.write(time.strftime("\nLine Distance - Start Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel3','stop',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel3E','start',datetime())''')
        db.commit()

    def onAddF(self, event):
        """open panel3F"""
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel3F,1,wx.EXPAND)
        newsize = self.Parent.panel3F.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel3F.Show()
        log.write(time.strftime("\nPoint Distance - Start Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel3','stop',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel3F','start',datetime())''')
        db.commit()

    def onAddG(self, event):
        """open panel3G"""
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel3G,1,wx.EXPAND)
        newsize = self.Parent.panel3G.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel3G.Show()
        log.write(time.strftime("\nRaster Value - Start Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel3','stop',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel3G','start',datetime())''')
        db.commit()

    def onNext(self, event):
        """"""
        # add predictors to panel 4
        cursor = conn.execute("SELECT * FROM dat4stats")
        varnames = list([x[0] for x in cursor.description])
        prednames = [name for name in varnames if (name[0] =='p')]
        self.Parent.panel4.chkbx1.Append(prednames)
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel4,1,wx.EXPAND)
        newsize = self.Parent.panel4.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel4.Show()
        log.write('\nPredictor variables created: '+str(prednames))
        log.write('\nNumber of predictor variables created: '+str(len(prednames))+'\n')
        log.write(time.strftime('\nPredictors - End Time: %A %d %b %Y %H:%M:%S\n', time.localtime()))
        log.write(time.strftime('\nModel - Start Time: %A %d %b %Y %H:%M:%S\n', time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel3','stop',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel4','start',datetime())''')
        db.commit()

    def onCanc1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Cancel Wizard"""
        dlg = wx.MessageDialog(None, "Cancelling the wizard will delete all progress. Do you wish to cancel the wizard?",'Cancel',wx.YES_NO | wx.ICON_WARNING)
        result = dlg.ShowModal()
        if result == wx.ID_YES:
            arcpy.AddMessage((time.strftime("Cancelled by user: %A %d %b %Y %H:%M:%S", time.localtime())))
            try:
                log.close()
                logging.shutdown()
                db.close()
                shutil.rmtree(out_path)
            except NameError as e:
                arcpy.AddMessage(e)
            except OSError as e:
                arcpy.AddMessage(e)
            del wait
            self.Parent.Close()
        else:
            self.Parent.statusbar.SetStatusText('Ready')
            del wait

#-------------------------------------------------------------------------------
class WizardPanel3A(wx.Panel):
    """Page 3A"""

    def __init__(self, parent, title=None):
        """Constructor"""
        wx.Panel.__init__(self, parent)
        self.locale = wx.Locale(wx.LANGUAGE_ENGLISH)
        wx.ToolTip.Enable(True)

        self.sizer = wx.GridBagSizer(0,0)

        title2 = wx.StaticText(self, -1, "Polygon Area or Value within Buffer")
        title2.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(title2, pos=(0,0), span=(1,5),flag=wx.TOP|wx.LEFT|wx.BOTTOM, border=10)

        self.sizer.Add(wx.StaticLine(self), pos=(1, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM, border=5)

        header0 = wx.StaticText(self,-1,label="Set Variable Name")
        header0.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header0, pos=(2, 0), flag=wx.ALL, border=10)

        help_btn0 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn0.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn0.SetForegroundColour(wx.Colour(0,0,255))
        help_btn0.Bind(wx.EVT_BUTTON, self.onHlp0)
        self.sizer.Add(help_btn0, pos=(2,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text1 = wx.StaticText(self,-1,label="Variable Name")
        self.sizer.Add(text1, pos=(3, 0), flag=wx.ALL, border=10)

        self.tc0 = wx.TextCtrl(self, -1)
        self.tc0.SetMaxLength(20)
        self.tc0.SetToolTip(wx.ToolTip('Enter a name for the predictor variable'))
        self.tc0.Bind(wx.EVT_CHAR,self.onChar)
        self.sizer.Add(self.tc0, pos=(3, 1), span=(1,2), flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=5)

        self.enter_btn0 = wx.Button(self, label="Enter")
        self.enter_btn0.Bind(wx.EVT_BUTTON, self.onEnt0)
        self.sizer.Add(self.enter_btn0, pos=(3,3), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)

        self.mark1 = wx.StaticText(self,-1,label=mark_empty)
        self.mark1.SetForegroundColour((0,255,0))
        self.mark1.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark1, pos=(3,4), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(4, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        header1 = wx.StaticText(self,-1,label="Set Buffer Sizes")
        header1.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header1, pos=(5, 0), flag=wx.ALL, border=10)

        help_btn1 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn1.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn1.SetForegroundColour(wx.Colour(0,0,255))
        help_btn1.Bind(wx.EVT_BUTTON, self.onHlp1)
        self.sizer.Add(help_btn1, pos=(5,1), flag=wx.TOP|wx.BOTTOM, border=5)

        self.radio_bx1 = wx.RadioBox(self,-1,label="Buffers",choices=["Create new buffer","Use previous buffer"], majorDimension=0, style=wx.RA_SPECIFY_COLS)
        self.sizer.Add(self.radio_bx1, pos=(6,0), span=(1,6), flag=wx.ALL|wx.EXPAND, border=10)
        self.radio_bx1.Bind(wx.EVT_RADIOBOX,self.onRadBx1)
        self.radio_bx1.Disable()

        text2 = wx.StaticText(self,-1,label="Create Buffer Distance")
        self.sizer.Add(text2, pos=(7, 0), flag=wx.ALL, border=10)

        self.tc1 = NumCtrl(self, integerWidth=6, fractionWidth=0, allowNegative=False, allowNone = True)
        self.tc1.SetToolTip(wx.ToolTip('Enter one or more buffer distances'))
        self.sizer.Add(self.tc1, pos=(7, 1), span=(1,2), flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=5)
        self.tc1.Disable()

        self.enter_btn1 = wx.Button(self, label="Add")
        self.enter_btn1.Bind(wx.EVT_BUTTON, self.onEnt1)
        self.sizer.Add(self.enter_btn1, pos=(7,3), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)
        self.enter_btn1.Disable()

        self.list_bx1 = wx.ListBox(self,-1,choices=[])
        self.sizer.Add(self.list_bx1, pos=(8,1), span=(2,2), flag=wx.EXPAND|wx.BOTTOM, border=5)

        self.del_btn1 = wx.Button(self, label="Remove")
        self.del_btn1.SetToolTip(wx.ToolTip('Remove selected buffer distance'))
        self.del_btn1.Bind(wx.EVT_BUTTON, self.onDel1)
        self.sizer.Add(self.del_btn1, pos=(8,3), flag=wx.RIGHT|wx.LEFT, border=5)
        self.del_btn1.Disable()

        self.done_btn1 = wx.Button(self, label="Done")
        self.done_btn1.SetToolTip(wx.ToolTip('Create buffers'))
        self.done_btn1.Bind(wx.EVT_BUTTON, self.onDone1)
        self.sizer.Add(self.done_btn1, pos=(9,3), flag=wx.RIGHT|wx.LEFT|wx.TOP|wx.BOTTOM, border=5)
        self.done_btn1.Disable()

        self.mark2 = wx.StaticText(self,-1,label=mark_empty)
        self.mark2.SetForegroundColour((0,255,0))
        self.mark2.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark2, pos=(9,4), flag=wx.ALL, border=5)

        text2a = wx.StaticText(self,-1,label="Select Buffer Distance")
        self.sizer.Add(text2a, pos=(10, 0), flag=wx.ALL, border=10)

        self.cb0 = wx.ComboBox(self,choices=[],style=wx.CB_DROPDOWN)
        self.cb0.SetToolTip(wx.ToolTip('From the dropdown list select a list of previously used buffer distances'))
        self.sizer.Add(self.cb0, pos=(10, 1), span=(1,3),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.cb0.Bind(wx.EVT_COMBOBOX, self.onCb0)
        self.cb0.Disable()

        self.done_btn1a = wx.Button(self, label="Done")
        self.done_btn1a.SetToolTip(wx.ToolTip('Select buffers'))
        self.done_btn1a.Bind(wx.EVT_BUTTON, self.onDone1a)
        self.sizer.Add(self.done_btn1a, pos=(10,4), flag=wx.RIGHT|wx.LEFT|wx.TOP|wx.BOTTOM, border=5)
        self.done_btn1a.Disable()

        self.mark2a = wx.StaticText(self,-1,label=mark_empty)
        self.mark2a.SetForegroundColour((0,255,0))
        self.mark2a.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark2a, pos=(10,5), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(11, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        header2 = wx.StaticText(self,-1,label="Set Input Data")
        header2.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header2, pos=(12, 0), flag=wx.ALL, border=10)

        help_btn2 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn2.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn2.SetForegroundColour(wx.Colour(0,0,255))
        help_btn2.Bind(wx.EVT_BUTTON, self.onHlp2)
        self.sizer.Add(help_btn2, pos=(12,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text3 = wx.StaticText(self,-1,label="Polygon Feature Class")
        self.sizer.Add(text3, pos=(13, 0), flag=wx.ALL, border=10)

        self.cb1 = wx.ComboBox(self,choices=[],style=wx.CB_DROPDOWN)
        self.cb1.SetToolTip(wx.ToolTip('From the dropdown list select the feature class containing the polygon data'))
        self.sizer.Add(self.cb1, pos=(13, 1), span=(1,2),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.cb1.Bind(wx.EVT_COMBOBOX, self.onCb1)
        self.cb1.Disable()

        self.mark3 = wx.StaticText(self,-1,label=mark_empty)
        self.mark3.SetForegroundColour((0,255,0))
        self.mark3.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark3, pos=(13,3), flag=wx.ALL, border=5)

        text4 = wx.StaticText(self,-1,label="Category Field")
        self.sizer.Add(text4, pos=(14, 0), flag=wx.ALL, border=10)

        self.cb2 = wx.ComboBox(self,choices=[],style=wx.CB_DROPDOWN)
        self.cb2.SetToolTip(wx.ToolTip('From the dropdown list select the field containing the categories'))
        self.sizer.Add(self.cb2, pos=(14, 1), span=(1,2),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.cb2.Bind(wx.EVT_COMBOBOX, self.onCb2)
        self.cb2.Disable()

        self.mark4 = wx.StaticText(self,-1,label=mark_empty)
        self.mark4.SetForegroundColour((0,255,0))
        self.mark4.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark4, pos=(14,3), flag=wx.ALL, border=5)

        self.radio_bx = wx.RadioBox(self,-1,label="Aggregation Method",choices=["Total area","Area weighted value","Area * Value"], majorDimension=0, style=wx.RA_SPECIFY_COLS)
        self.sizer.Add(self.radio_bx, pos=(15,0), span=(1,6), flag=wx.ALL|wx.EXPAND, border=10)
        self.radio_bx.Bind(wx.EVT_RADIOBOX,self.onRadBx)
        self.radio_bx.Disable()

        self.text6 = wx.StaticText(self,-1,label="Value Field")
        self.sizer.Add(self.text6, pos=(16, 0), flag=wx.ALL, border=10)
        self.text6.Disable()

        self.cb3 = wx.ComboBox(self,choices=[],style=wx.CB_DROPDOWN)
        self.cb3.SetToolTip(wx.ToolTip('From the dropdown list select the field containing values to be area weighted'))
        self.sizer.Add(self.cb3, pos=(16, 1), span=(1,2),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.cb3.Bind(wx.EVT_COMBOBOX, self.onCb3)
        self.cb3.Disable()

        self.mark6 = wx.StaticText(self,-1,label=mark_empty)
        self.mark6.SetForegroundColour((0,255,0))
        self.mark6.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark6, pos=(16,3), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(17, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        header3 = wx.StaticText(self,-1,label="Set Direction of Effect")
        header3.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header3, pos=(18, 0), flag=wx.ALL, border=10)

        help_btn3 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn3.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn3.SetForegroundColour(wx.Colour(0,0,255))
        help_btn3.Bind(wx.EVT_BUTTON, self.onHlp3)
        self.sizer.Add(help_btn3, pos=(18,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text5 = wx.StaticText(self,-1,label="Define Direction of Effect")
        self.sizer.Add(text5, pos=(19, 0), flag=wx.ALL, border=10)

        self.ulist1 = ULC.UltimateListCtrl(self, wx.ID_ANY, agwStyle=ULC.ULC_HAS_VARIABLE_ROW_HEIGHT | wx.LC_REPORT | wx.LC_VRULES | wx.LC_HRULES)
        self.ulist1.InsertColumn(col=0, heading="Variable Name",format=0)
        self.ulist1.InsertColumn(col=1, heading="Positive",format=0)
        self.ulist1.InsertColumn(col=2, heading="Negative",format=0)
        self.ulist1.SetColumnWidth(0,ULC.ULC_AUTOSIZE_FILL)
        self.ulist1.SetColumnWidth(1,ULC.ULC_AUTOSIZE_USEHEADER)
        self.ulist1.SetColumnWidth(2,ULC.ULC_AUTOSIZE_USEHEADER)
        self.sizer.Add(self.ulist1,pos=(19,1),span=(5,4),flag=wx.TOP|wx.BOTTOM|wx.EXPAND, border=5)
        self.ulist1.Disable()

        self.enter_btn2 = wx.Button(self, label="Done")
        self.enter_btn2.Bind(wx.EVT_BUTTON, self.onEnt2)
        self.sizer.Add(self.enter_btn2, pos=(19,5), flag=wx.TOP|wx.BOTTOM|wx.RIGHT|wx.LEFT, border=5)
        self.enter_btn2.Disable()

        self.mark5 = wx.StaticText(self,-1,label=mark_empty)
        self.mark5.SetForegroundColour((0,255,0))
        self.mark5.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark5, pos=(19,6), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(24, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        # add previous Button
        self.backBtn = wx.Button(self, label="< Back")
        self.backBtn.Bind(wx.EVT_BUTTON, self.onBack)
        self.sizer.Add(self.backBtn, pos=(25,4),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        # add next button
        self.nextBtn = wx.Button(self, label="Next >")
        self.nextBtn.Bind(wx.EVT_BUTTON, self.onNext)
        self.sizer.Add(self.nextBtn, pos=(25,5),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)
        self.nextBtn.Disable()

        # add cancel button
        self.cancBtn1 = wx.Button(self, label="Cancel")
        self.cancBtn1.Bind(wx.EVT_BUTTON, self.onCanc1)
        self.cancBtn1.SetToolTip(wx.ToolTip('Cancel'))
        self.sizer.Add(self.cancBtn1, pos=(25,6),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        self.SetSizer(self.sizer)
        self.sizer.AddGrowableRow(7)
        self.sizer.AddGrowableRow(17)
        self.sizer.AddGrowableCol(1)
        self.GetParent().SendSizeEvent()
        self.Layout()
        self.Fit()

    def onHlp0(self,event):
        """Help window for variable name"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3A_SetVariableName.html")
        htmlViewerInstance.Show()

    def onChar(self,event):
        keycode = event.GetKeyCode()
        if keycode in ascii_char: #only allow chars in alphabet
            event.Skip()

    def onEnt0(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """enter variable name"""
        inp_var = self.tc0.GetValue()
        if not inp_var:
            wx.MessageBox('Please enter a variable name','Error',wx.OK|wx.ICON_ERROR)
        else: # change this to be based on the database, so that entry can be changed
            global pA_name
            pA_name = "pA_"+inp_var
            if pA_name in var_list:
                wx.MessageBox('The variable already exists','Error',wx.OK|wx.ICON_ERROR)
            else:
                var_list.append(pA_name)
                self.mark1.SetLabel(mark_done) # change tick mark to done
                log.write('\nVariable name: '+inp_var)
                self.radio_bx1.Enable()
                self.tc1.Enable()
                self.enter_btn1.Enable()
                self.tc0.Disable()
                self.enter_btn0.Disable()
            global pAdist
            pAdist = []
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onHlp1(self,event):
        """Help window for buffers"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3ABC_SetBufferSizes.html")
        htmlViewerInstance.Show()

    def onRadBx1(self,event):
        """New buffer or previous buffer"""
        if self.radio_bx1.GetStringSelection()=="Create new buffer": #activate section
            self.tc1.Enable()
            self.enter_btn1.Enable()
            self.cb0.Disable()
            self.done_btn1a.Disable()
        elif self.radio_bx1.GetStringSelection()=="Use previous buffer": #activate other section
            self.cb0.Enable()
            self.done_btn1a.Enable()
            self.tc1.Disable()
            self.enter_btn1.Disable()

    def onEnt1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Enter buffer distances"""
        new_dist = self.tc1.GetValue()
        if not new_dist:
            wx.MessageBox('Please enter a buffer distance','Error',wx.OK|wx.ICON_ERROR)
        else:
            pAdist.append(new_dist)
            self.list_bx1.Append(str(new_dist))
            self.done_btn1.Enable()
            self.del_btn1.Enable()
            self.tc1.Clear()
            self.Parent.statusbar.SetStatusText('Ready')
            del wait

    def onDel1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Remove buffer distance from list"""
        sel1_id = self.list_bx1.GetSelection()
        if sel1_id==-1:
            wx.MessageBox('Click on a buffer distance in the box first, then click Remove.','Error',wx.OK|wx.ICON_ERROR)
        else:
            del_dist = self.list_bx1.GetString(sel1_id)
            pAdist.remove(int(del_dist))
            self.list_bx1.Delete(sel1_id)
            if not pAdist:
                self.done_btn1.Disable()
                self.del_btn1.Disable()
            self.Parent.statusbar.SetStatusText('Ready')
            del wait

    def onDone1(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Create buffers"""
        global pA_buffer
        if sorted(pAdist) not in bufDists.values(): #check if multiple ring buffer already exists, if not create one and add to dictionary
            k = 'buffer'+time.strftime('%y%m%d_%H%M%S') #create new key
            bufDists.update({k : sorted(pAdist)}) # add to dictionary
            pA_buffer= out_fds+"\\"+k
            arcpy.MultipleRingBuffer_analysis(sites, pA_buffer, pAdist, "", "", "NONE")
            # append buffer distances to cb0 in panel 3A
            self.cb0.SetValue('')
            self.cb0.Clear()
            self.cb0.Append([str(i) for i in bufDists.values()])
            # append buffer distances to cb0 in panel 3B
            self.Parent.panel3B.cb0.SetValue('')
            self.Parent.panel3B.cb0.Clear()
            self.Parent.panel3B.cb0.Append([str(i) for i in bufDists.values()])
            # append buffer distances to cb0 in panel 3C
            self.Parent.panel3C.cb0.SetValue('')
            self.Parent.panel3C.cb0.Clear()
            self.Parent.panel3C.cb0.Append([str(i) for i in bufDists.values()])

            arcpy.AddMessage('bufDists:')
            arcpy.AddMessage(bufDists)

        else:
            for key,value in bufDists.items():
                if value == pAdist:
                    pA_buffer = out_fds+"\\"+key # use existing buffer

        self.mark2.SetLabel(mark_done) # change tick mark to done
        log.write('\nBuffer distances: '+str(sorted(pAdist)))
        log.write('\nBuffer feature class: '+str(pA_buffer))
        arcpy.AddMessage('bufDists:')
        arcpy.AddMessage(bufDists)
        self.cb1.Enable()
        self.ulist1.Enable()
        self.radio_bx1.Disable()
        self.radio_bx1.SetSelection(0)
        self.tc1.Disable()
        self.enter_btn1.Disable()
        self.del_btn1.Disable()
        self.done_btn1.Disable()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onCb0(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Select list of buffers"""
        prev_dist_str = self.cb0.GetValue().strip('[]').split(',')
        prev_dist = [int(i) for i in prev_dist_str]
        pAdist.extend(prev_dist)
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onDone1a(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Create buffers"""
        global pA_buffer
        for key,value in bufDists.items():
            if value == pAdist: # look up key associated with value
                pA_buffer = out_fds+"\\"+key # use existing buffer

        self.mark2a.SetLabel(mark_done) # change tick mark to done
        log.write('\nBuffer distances: '+str(sorted(pAdist)))
        log.write('\nBuffer feature class: '+str(pA_buffer))
        self.cb1.Enable()
        self.ulist1.Enable()
        self.radio_bx1.Disable()
        self.radio_bx1.SetSelection(0)
        self.cb0.Disable()
        self.done_btn1a.Disable()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onHlp2(self,event):
        """Help window for input data"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3A_SetInputData.html")
        htmlViewerInstance.Show()

    def onCb1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Select Polygon feature class"""
        self.cb2.Clear()
        self.cb3.Clear()
        self.ulist1.DeleteAllItems()
        global pA
        pA = self.cb1.GetValue()
        global pA_fldsNamesDict
        pA_fldsNamesDict={}
        num_row=int(arcpy.GetCount_management(pA).getOutput(0))
        if num_row>0: #check it contains text fields
            str_fields = [f.name for f in arcpy.ListFields(pA,"","String")] # get text fields
            str_fields.sort()
            if not str_fields:
                wx.MessageBox('The selected feature class does not contain any text fields','Error',wx.OK|wx.ICON_ERROR)
            else:
                self.cb2.Append(str_fields)
                log.write('\nPolygon Feature Class: '+pA)
                self.mark3.SetLabel(mark_done) # change tick mark to done

        else:#check that it contains features
            wx.MessageBox('The selected feature class is empty','Error',wx.OK|wx.ICON_ERROR)

        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        self.cb2.Enable()

    def onCb2(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Select category"""
        self.ulist1.DeleteAllItems()
        for k,v in list(pA_fldsNamesDict.items()): # check if value is already in fldsNamesDict
            if v == 'pA_cat':
                del pA_fldsNamesDict[k]
        global pA_cat
        pA_cat = self.cb2.GetValue()
        # strip any endspaces from strings and remove underscore,space and full stop
        with arcpy.da.UpdateCursor(pA, pA_cat) as cursor:
            for row in cursor:
                row=[i.strip() if i is not None else None for i in row]
                row=[i.replace(" ","") if i is not None else None for i in row]
                row=[i.replace("_","") if i is not None else None for i in row]
                row=[i.replace(".","") if i is not None else None for i in row]
                cursor.updateRow(row)
        # Get list of unique values in field
        pACats= unique_values(pA,pA_cat)
        # add buffer distances to names
        global pACatsBuffer
        pACatsBuffer=[]
        for cat in pACats:
            if cat is None:
                wx.MessageBox('The selected field contains missing values','Error',wx.OK|wx.ICON_ERROR)
            else:
                for buf in pAdist:
                    pACatsBuffer.append(pA_name+'_'+cat+'_'+str(buf)+'_sum')
        # Check for missing data
        self.radios = []
        myMethods = ['Positive','Negative']
        for item in range(len(pACatsBuffer)):
            self.ulist1.InsertStringItem(item, str(pACatsBuffer[item]))
            for rad in range(1,3):
                cat = pACatsBuffer[item]
                met = myMethods[rad - 1]
                name_of_radio = "{cat}_{met}".format(cat=cat, met=met)
                self.ulist1.SetStringItem(item, rad, "")
                if rad==1:
                    style=wx.RB_GROUP
                else:
                    style=0
                self.radBt = wx.RadioButton(self.ulist1, wx.ID_ANY, "", style=style,name=name_of_radio)
                self.ulist1.SetItemWindow(item,rad,self.radBt,expand=True)
                self.radios.append(self.radBt)

        # add field name to dictionary
        new_entry={str(pA_cat):'pA_cat'}
        pA_fldsNamesDict.update(new_entry)

        self.mark4.SetLabel(mark_done) # change tick mark to done
        self.radio_bx.Enable()
        self.ulist1.Enable()
        self.enter_btn2.Enable()
        log.write('\nCategory field: '+str(pA_cat))
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onRadBx(self,event):
        """Aggregation method"""
        if self.radio_bx.GetStringSelection()=="Area weighted value" or self.radio_bx.GetStringSelection()=="Area * Value": #check it contains numeric fields
            num_fields = [f.name for f in arcpy.ListFields(pA,"",'Double') if not f.required] #get numeric fields
            num_fields.extend([f.name for f in arcpy.ListFields(pA,"",'Integer')])
            num_fields.extend([f.name for f in arcpy.ListFields(pA,"",'Single')])
            num_fields.extend([f.name for f in arcpy.ListFields(pA,"",'SmallInteger')])
            num_fields.sort()
            if not num_fields:
                wx.MessageBox('The selected feature class contains no numeric fields','Error',wx.OK|wx.ICON_ERROR)
            else:
                self.cb3.Append(num_fields)
                self.text6.Enable()
                self.cb3.Enable()
                self.nextBtn.Disable()
                self.ulist1.Disable()
                self.enter_btn2.Disable()
                self.ulist1.DeleteAllItems()

        else:
            self.text6.Disable()
            self.cb3.Disable()
            for k,v in list(pA_fldsNamesDict.items()):
                if v == 'pA_val':
                    del pA_fldsNamesDict[k]


    def onCb3(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Select value field"""
        for k,v in list(pA_fldsNamesDict.items()):
            if v == 'pA_val':
                del pA_fldsNamesDict[k]
        pA_val = self.cb3.GetValue()
        # attach method to pBCatsBuffer
        if self.radio_bx.GetStringSelection()=="Area weighted value":
            for index,o in enumerate(pACatsBuffer):
                pACatsBuffer[index] = o[:-4]+'_wtv'
        else:
            for index,o in enumerate(pACatsBuffer):
                pACatsBuffer[index] = o[:-4]+'_mtv'
        # populate ulist
        self.radios = []
        myMethods = ['Positive','Negative']
        for item in range(len(pACatsBuffer)):
            self.ulist1.InsertStringItem(item, str(pACatsBuffer[item]))
            for rad in range(1,3):
                cat = pACatsBuffer[item]
                met = myMethods[rad - 1]
                name_of_radio = "{cat}_{met}".format(cat=cat, met=met)
                self.ulist1.SetStringItem(item, rad, "")
                if rad==1:
                    style=wx.RB_GROUP
                else:
                    style=0
                self.radBt = wx.RadioButton(self.ulist1, wx.ID_ANY, "", style=style,name=name_of_radio)
                self.ulist1.SetItemWindow(item,rad,self.radBt,expand=True)
                self.radios.append(self.radBt)
        self.ulist1.Enable()
        self.enter_btn2.Enable()
        # add field name to dictionary
        new_entry={str(pA_val):'pA_val'}
        pA_fldsNamesDict.update(new_entry)
        self.mark6.SetLabel(mark_done) # change tick mark to done
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onHlp3(self,event):
        """Help window for direction of effect"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3ABC_SetSourceSink.html")
        htmlViewerInstance.Show()

    def onEnt2(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Define Direction of Effect"""
        results_dict = {}
        for i in self.radios:
            if i.GetValue()==True:
                n = i.GetName()
                index = n.rfind('_')
                cat = n[0:index]
                met = n[index+1:]
                results_dict[cat]= met
        sourcesink.update(results_dict)

        if arcpy.Exists(out_fds+"\\buffer_"+pA_name):
            self.nextBtn.Enable()

        self.mark5.SetLabel(mark_done) # change tick mark to done
        log.write('\nAggregation method: '+self.radio_bx.GetStringSelection())
        if self.radio_bx.GetStringSelection()=="Area weighted value" or self.radio_bx.GetStringSelection()=="Area * Value":
            log.write('\nValue field: '+self.cb3.GetValue())
        log.write('\nVariable definitions: ')
        for k, v in sorted(results_dict.items()):
            log.write('\n{:<30}: {:<6}'.format(k,v))
        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        self.cb1.Disable()
        self.cb2.Disable()
        self.ulist1.Disable()
        self.enter_btn2.Disable()
        self.nextBtn.Enable()

    def onBack(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Return to panel3"""
        try:
            var_list.remove(pA_name)
            for i in self.radios: # remove from sourcesink dictionary
                if i.GetValue()==True:
                    n = i.GetName()
                    index = n.rfind('_')
                    cat = pA_name+'_'+n[0:index]
                    sourcesink.pop(cat, None)
        except:
            pass
        if arcpy.Exists(out_fds+"\\buffer_"+pA_name):
            arcpy.Delete_management(out_fds+"\\buffer_"+pA_name)
            del pAdist[:]
        self.tc0.Clear()
        self.tc1.Clear()
        self.list_bx1.Clear()
        self.cb2.Clear()
        self.ulist1.DeleteAllItems()
        self.cb3.Clear()
        self.radio_bx.SetSelection(0)
        self.radio_bx.Disable()
        self.mark1.SetLabel(mark_empty)
        self.mark2.SetLabel(mark_empty)
        self.mark2a.SetLabel(mark_empty)
        self.mark3.SetLabel(mark_empty)
        self.mark4.SetLabel(mark_empty)
        self.mark5.SetLabel(mark_empty)
        self.mark6.SetLabel(mark_empty)
        self.tc1.Disable()
        self.enter_btn1.Disable()
        self.del_btn1.Disable()
        self.done_btn1.Disable()
        self.radio_bx1.Disable()
        self.radio_bx1.SetSelection(0)
        self.cb0.Disable()
        self.done_btn1a.Disable()
        self.cb1.Disable()
        self.cb2.Disable()
        self.ulist1.Disable()
        self.nextBtn.Disable()
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel3,1,wx.EXPAND)
        newsize = self.Parent.panel3.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel3.Show()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        log.write(time.strftime("\n\nPolygon in Buffer - Stopped by user without completion: %A %d %b %Y %H:%M:%S", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel3A','back',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel3','start',datetime())''')
        db.commit()
        self.tc0.Enable()
        self.enter_btn0.Enable()

    def onNext(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """"""
        fieldmappings=customFieldMap(pA,pA_fldsNamesDict) # create fieldmap
        arcpy.FeatureClassToFeatureClass_conversion(pA,out_fds,pA_name+"_temp", "", fieldmappings) # copy the feature class
        max_dist=max(pAdist) # maximum buffer distance
        arcpy.Buffer_analysis(out_fds+"\\studyArea",out_fds+"\\analysisExtent_"+pA_name,max_dist) # create analysis extent
        # Compare the spatial extent against the analysis extent
        ext_an = arcpy.Describe(out_fds+"\\analysisExtent_"+pA_name).extent
        ext_2 = arcpy.Describe(out_fds+"\\"+pA_name+"_temp").extent
        if ext_2.contains(ext_an)==False:
            wx.MessageBox('The spatial extent of the polygon feature class is too small or completely outside of the study area.','Warning',wx.OK|wx.ICON_WARNING)
            log.write('\n+++ WARNING+++ The spatial extent of the polygon feature class is too small or completely outside of the study area. This may result in empty intersects. If all intersects are empty, no predictor variable will be created.\n')
        arcpy.Clip_analysis(out_fds+"\\"+pA_name+"_temp",out_fds+"\\analysisExtent_"+pA_name,out_fds+"\\"+pA_name) # Clip data to analysis extent
        arcpy.Delete_management(out_fds+"\\"+pA_name+"_temp") # delete temp data
        arcpy.Delete_management(out_fds+"\\analysisExtent_"+pA_name) # delete temp data
        arcpy.AddField_management(out_fds+"\\"+pA_name,"pA_cat_INT","SHORT") # add integer id
        valueSet = set([r[0] for r in arcpy.da.SearchCursor(out_fds+"\\"+pA_name, ["pA_cat"])])
        valueList = list(valueSet)
        valueList.sort()
        updateRows = arcpy.da.UpdateCursor(out_fds+"\\"+pA_name, ["pA_cat","pA_cat_INT"])
        for updateRow in updateRows:
            updateRow[1] = valueList.index(updateRow[0]) + 1
            updateRows.updateRow(updateRow)
        if FieldExist(out_fds+"\\"+pA_name,"pA_val"):
            arcpy.AddField_management(out_fds+"\\"+pA_name,"origArea","DOUBLE") # add field for original area
            cursor=arcpy.da.UpdateCursor(out_fds+"\\"+pA_name, ["origArea","Shape_Area"])
            for row in cursor:
                row[0]=row[1]
                cursor.updateRow(row)
        arcpy.PairwiseIntersect_analysis ([out_fds+"\\"+pA_name, pA_buffer], out_fds+"\\"+pA_name+"_Intersect", "ALL", "", "")
        intersect_var = pA_name+"_Intersect"
        # move data to sql database
        fc_to_sql(conn,out_fds+"\\"+pA_name)
        conn.execute("CREATE INDEX "+pA_name+"_idx on "+pA_name+"(pA_cat_INT);")
        fc_to_sql(conn,out_fds+"\\"+intersect_var)
        conn.execute("CREATE INDEX "+intersect_var+"_idx on "+intersect_var+"(siteID_INT, pA_cat_INT, distance);")
        db.commit()
        arcpy.Delete_management(out_fds+"\\"+intersect_var)
        # add weighted value
        conn.execute("ALTER TABLE "+intersect_var+" ADD pA_wtval float64")
        if FieldExist(out_fds+"\\"+pA_name,"pA_val") and self.radio_bx.GetStringSelection()=="Area weighted value":
            conn.execute("UPDATE "+intersect_var+" SET pA_wtval=(Shape_Area/origArea)*pA_val")
        elif FieldExist(out_fds+"\\"+pA_name,"pA_val") and self.radio_bx.GetStringSelection()=="Area * Value":
            conn.execute("UPDATE "+intersect_var+" SET pA_wtval=Shape_Area*pA_val")
        else:
            conn.execute("UPDATE "+intersect_var+" SET pA_wtval=Shape_Area")
        db.commit()
        # calculate sum by site ID, category and buffer size, indicate aggregation method in variable name
        if self.radio_bx.GetStringSelection()=="Total area":
            qry = "CREATE TABLE temp AS \
                        SELECT siteID_INT \
                             ,pA_cat_INT \
                             ,distance \
                             ,SUM(pA_wtval) AS value \
	                         ,('"+pA_name+"_'||pA_cat||'_'||cast(distance as int)||'_sum') AS varName \
                        FROM "+intersect_var+" \
                        GROUP BY siteID_INT, pA_cat_INT, distance"
        elif self.radio_bx.GetStringSelection()=="Area weighted value":
            qry = "CREATE TABLE temp AS \
                        SELECT siteID_INT \
                             ,pA_cat_INT \
                             ,distance \
                             ,SUM(pA_wtval) AS value \
	                         ,('"+pA_name+"_'||pA_cat||'_'||cast(distance as int)||'_wtv') AS varName \
                        FROM "+intersect_var+" \
                        GROUP BY siteID_INT, pA_cat_INT, distance"
        else:
            qry = "CREATE TABLE temp AS \
                        SELECT siteID_INT \
                             ,pA_cat_INT \
                             ,distance \
                             ,SUM(pA_wtval) AS value \
	                         ,('"+pA_name+"_'||pA_cat||'_'||cast(distance as int)||'_mtv') AS varName \
                        FROM "+intersect_var+" \
                        GROUP BY siteID_INT, pA_cat_INT, distance"
        conn.execute(qry)
        vars_sql_temp=list(conn.execute("SELECT DISTINCT varName FROM temp")) # make a list of variable names
        # add a column for each variable name
        for var in vars_sql_temp:
            conn.execute('ALTER TABLE dat4stats ADD {} float64'.format(var[0])) # if I add second query I get an operational error
        db.commit()
        # add values to each column
        for var in vars_sql_temp:
            qry="UPDATE dat4stats \
                 SET "+var[0]+" = (\
                 SELECT value \
                 FROM temp \
                 WHERE siteID_INT = dat4stats.siteID_INT and varName='"+var[0]+"')"
            conn.execute(qry)
        db.commit()
        # replace missing with zeros
        for var in vars_sql_temp:
            qry="UPDATE dat4stats \
                 SET "+var[0]+" = 0 \
                 WHERE "+var[0]+" IS NULL"
            conn.execute(qry)
        db.commit()
        conn.execute("DROP TABLE temp")
        conn.execute("DROP TABLE "+intersect_var)
        db.commit()
        # predictor names to panel 3
        WizardPanel3.list_ctrl.DeleteAllItems()
        cursor = conn.execute("SELECT * FROM dat4stats")
        varnames = list([x[0] for x in cursor.description])
        prednames = [name for name in varnames if (name[0] =='p')]
        index=0
        for i in prednames:
            WizardPanel3.list_ctrl.InsertItem(index,i)
            WizardPanel3.list_ctrl.SetItem(index,1,sourcesink[i])
            if index % 2:
                WizardPanel3.list_ctrl.SetItemBackgroundColour(index, col="white")
            else:
                WizardPanel3.list_ctrl.SetItemBackgroundColour(index, col=(222,234,246))
            index+=1
        WizardPanel3.list_ctrl.SetColumnWidth(0,wx.LIST_AUTOSIZE)
        WizardPanel3.list_ctrl.SetColumnWidth(1,wx.LIST_AUTOSIZE)
        log.write(time.strftime("\nPolygon in Buffer - End Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel3A','stop',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel3','start',datetime())''')
        db.commit()
        # clear p3A
        del pAdist[:]
        self.tc0.Clear()
        self.tc1.Clear()
        self.list_bx1.Clear()
        self.cb2.Clear()
        self.ulist1.DeleteAllItems()
        self.cb3.Clear()
        self.mark1.SetLabel(mark_empty)
        self.mark2.SetLabel(mark_empty)
        self.mark2a.SetLabel(mark_empty)
        self.mark3.SetLabel(mark_empty)
        self.mark4.SetLabel(mark_empty)
        self.mark5.SetLabel(mark_empty)
        self.mark6.SetLabel(mark_empty)
        self.tc1.Disable()
        self.enter_btn1.Disable()
        self.del_btn1.Disable()
        self.done_btn1.Disable()
        self.radio_bx1.Disable()
        self.radio_bx1.SetSelection(0)
        self.cb0.Disable()
        self.done_btn1a.Disable()
        self.cb1.Disable()
        self.cb2.Disable()
        self.ulist1.Disable()
        self.nextBtn.Disable()
        self.radio_bx.SetSelection(0)
        self.radio_bx.Disable()
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel3,1,wx.EXPAND)
        newsize = self.Parent.panel3.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel3.Show()
        self.tc0.Enable()
        self.enter_btn0.Enable()

        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onCanc1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Cancel Wizard"""
        dlg = wx.MessageDialog(None, "Cancelling the wizard will delete all progress. Do you wish to cancel the wizard?",'Cancel',wx.YES_NO | wx.ICON_WARNING)
        result = dlg.ShowModal()
        if result == wx.ID_YES:
            arcpy.AddMessage((time.strftime("Cancelled by user: %A %d %b %Y %H:%M:%S", time.localtime())))
            try:
                log.close()
                logging.shutdown()
                db.close()
                shutil.rmtree(out_path)
            except NameError as e:
                arcpy.AddMessage(e)
            except OSError as e:
                arcpy.AddMessage(e)
            del wait
            self.Parent.Close()
        else:
            self.Parent.statusbar.SetStatusText('Ready')
            del wait
#-------------------------------------------------------------------------------
class WizardPanel3B(wx.Panel):
    """Page 3B"""

    def __init__(self, parent, title=None):
        """Constructor"""
        wx.Panel.__init__(self, parent)
        self.locale = wx.Locale(wx.LANGUAGE_ENGLISH)
        wx.ToolTip.Enable(True)

        self.sizer = wx.GridBagSizer(0,0)

        title2 = wx.StaticText(self, -1, "Line Length or Value within Buffer")
        title2.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(title2, pos=(0,0), span=(1,5),flag=wx.TOP|wx.LEFT|wx.BOTTOM, border=10)

        self.sizer.Add(wx.StaticLine(self), pos=(1, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM, border=5)

        header0 = wx.StaticText(self,-1,label="Set Variable Name")
        header0.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header0, pos=(2, 0), flag=wx.ALL, border=10)

        help_btn0 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn0.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn0.SetForegroundColour(wx.Colour(0,0,255))
        help_btn0.Bind(wx.EVT_BUTTON, self.onHlp0)
        self.sizer.Add(help_btn0, pos=(2,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text1 = wx.StaticText(self,-1,label="Variable Name")
        self.sizer.Add(text1, pos=(3, 0), flag=wx.ALL, border=10)

        self.tc0 = wx.TextCtrl(self, -1)
        self.tc0.SetMaxLength(20)
        self.tc0.SetToolTip(wx.ToolTip('Enter a name for the predictor variable'))
        self.tc0.Bind(wx.EVT_CHAR,self.onChar)
        self.sizer.Add(self.tc0, pos=(3, 1), span=(1,2), flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=5)

        self.enter_btn0 = wx.Button(self, label="Enter")
        self.enter_btn0.Bind(wx.EVT_BUTTON, self.onEnt0)
        self.sizer.Add(self.enter_btn0, pos=(3,3), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)

        self.mark1 = wx.StaticText(self,-1,label=mark_empty)
        self.mark1.SetForegroundColour((0,255,0))
        self.mark1.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark1, pos=(3,4), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(4, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        header1 = wx.StaticText(self,-1,label="Set Buffer Sizes")
        header1.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header1, pos=(5, 0), flag=wx.ALL, border=10)

        help_btn1 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn1.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn1.SetForegroundColour(wx.Colour(0,0,255))
        help_btn1.Bind(wx.EVT_BUTTON, self.onHlp1)
        self.sizer.Add(help_btn1, pos=(5,1), flag=wx.TOP|wx.BOTTOM, border=5)

        self.radio_bx1 = wx.RadioBox(self,-1,label="Buffers",choices=["Create new buffer","Use previous buffer"], majorDimension=0, style=wx.RA_SPECIFY_COLS)
        self.sizer.Add(self.radio_bx1, pos=(6,0), span=(1,6), flag=wx.ALL|wx.EXPAND, border=10)
        self.radio_bx1.Bind(wx.EVT_RADIOBOX,self.onRadBx1)
        self.radio_bx1.Disable()

        text2 = wx.StaticText(self,-1,label="Create Buffer Distance")
        self.sizer.Add(text2, pos=(7, 0), flag=wx.ALL, border=10)

        self.tc1 = NumCtrl(self, integerWidth=6, fractionWidth=0, allowNegative=False, allowNone = True)
        self.tc1.SetToolTip(wx.ToolTip('Enter one or more buffer distances'))
        self.sizer.Add(self.tc1, pos=(7, 1), span=(1,2), flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=5)
        self.tc1.Disable()

        self.enter_btn1 = wx.Button(self, label="Add")
        self.enter_btn1.Bind(wx.EVT_BUTTON, self.onEnt1)
        self.sizer.Add(self.enter_btn1, pos=(7,3), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)
        self.enter_btn1.Disable()

        self.list_bx1 = wx.ListBox(self,-1,choices=[])
        self.sizer.Add(self.list_bx1, pos=(8,1), span=(2,2), flag=wx.EXPAND|wx.BOTTOM, border=5)

        self.del_btn1 = wx.Button(self, label="Remove")
        self.del_btn1.SetToolTip(wx.ToolTip('Remove selected buffer distance'))
        self.del_btn1.Bind(wx.EVT_BUTTON, self.onDel1)
        self.sizer.Add(self.del_btn1, pos=(8,3), flag=wx.RIGHT|wx.LEFT, border=5)
        self.del_btn1.Disable()

        self.done_btn1 = wx.Button(self, label="Done")
        self.done_btn1.SetToolTip(wx.ToolTip('Create buffers'))
        self.done_btn1.Bind(wx.EVT_BUTTON, self.onDone1)
        self.sizer.Add(self.done_btn1, pos=(9,3), flag=wx.RIGHT|wx.LEFT|wx.TOP|wx.BOTTOM, border=5)
        self.done_btn1.Disable()

        self.mark2 = wx.StaticText(self,-1,label=mark_empty)
        self.mark2.SetForegroundColour((0,255,0))
        self.mark2.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark2, pos=(9,4), flag=wx.ALL, border=5)

        text2a = wx.StaticText(self,-1,label="Select Buffer Distance")
        self.sizer.Add(text2a, pos=(10, 0), flag=wx.ALL, border=10)

        self.cb0 = wx.ComboBox(self,choices=[],style=wx.CB_DROPDOWN)
        self.cb0.SetToolTip(wx.ToolTip('From the dropdown list select a list of previously used buffer distances'))
        self.sizer.Add(self.cb0, pos=(10, 1), span=(1,3),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.cb0.Bind(wx.EVT_COMBOBOX, self.onCb0)
        self.cb0.Disable()

        self.done_btn1a = wx.Button(self, label="Done")
        self.done_btn1a.SetToolTip(wx.ToolTip('Select buffers'))
        self.done_btn1a.Bind(wx.EVT_BUTTON, self.onDone1a)
        self.sizer.Add(self.done_btn1a, pos=(10,4), flag=wx.RIGHT|wx.LEFT|wx.TOP|wx.BOTTOM, border=5)
        self.done_btn1a.Disable()

        self.mark2a = wx.StaticText(self,-1,label=mark_empty)
        self.mark2a.SetForegroundColour((0,255,0))
        self.mark2a.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark2a, pos=(10,5), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(11, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        header2 = wx.StaticText(self,-1,label="Set Input Data")
        header2.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header2, pos=(12, 0), flag=wx.ALL, border=10)

        help_btn2 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn2.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn2.SetForegroundColour(wx.Colour(0,0,255))
        help_btn2.Bind(wx.EVT_BUTTON, self.onHlp2)
        self.sizer.Add(help_btn2, pos=(12,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text3 = wx.StaticText(self,-1,label="Line Feature Class")
        self.sizer.Add(text3, pos=(13, 0), flag=wx.ALL, border=10)

        self.cb1 = wx.ComboBox(self,choices=[],style=wx.CB_DROPDOWN)
        self.cb1.SetToolTip(wx.ToolTip('From the dropdown list select the feature class containing the line data'))
        self.sizer.Add(self.cb1, pos=(13, 1), span=(1,2),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.cb1.Bind(wx.EVT_COMBOBOX, self.onCb1)
        self.cb1.Disable()

        self.mark3 = wx.StaticText(self,-1,label=mark_empty)
        self.mark3.SetForegroundColour((0,255,0))
        self.mark3.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark3, pos=(13,3), flag=wx.ALL, border=5)

        text4 = wx.StaticText(self,-1,label="Category Field")
        self.sizer.Add(text4, pos=(14, 0), flag=wx.ALL, border=10)

        self.cb2 = wx.ComboBox(self,choices=[],style=wx.CB_DROPDOWN)
        self.cb2.SetToolTip(wx.ToolTip('From the dropdown list select the field containing the categories'))
        self.sizer.Add(self.cb2, pos=(14, 1), span=(1,2),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.cb2.Bind(wx.EVT_COMBOBOX, self.onCb2)
        self.cb2.Disable()

        self.mark4 = wx.StaticText(self,-1,label=mark_empty)
        self.mark4.SetForegroundColour((0,255,0))
        self.mark4.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark4, pos=(14,3), flag=wx.ALL, border=5)

        self.radio_bx = wx.RadioBox(self,-1,label="Aggregation Method",choices=["Total length","Length weighted value","Length * Value"], majorDimension=0, style=wx.RA_SPECIFY_COLS)
        self.sizer.Add(self.radio_bx, pos=(15,0), span=(1,6), flag=wx.TOP|wx.ALL|wx.EXPAND, border=10)
        self.radio_bx.Bind(wx.EVT_RADIOBOX,self.onRadBx)
        self.radio_bx.Disable()

        self.text6 = wx.StaticText(self,-1,label="Value Field")
        self.sizer.Add(self.text6, pos=(16, 0), flag=wx.ALL, border=10)
        self.text6.Disable()

        self.cb3 = wx.ComboBox(self,choices=[],style=wx.CB_DROPDOWN)
        self.cb3.SetToolTip(wx.ToolTip('From the dropdown list select the field containing values to be length weighted'))
        self.sizer.Add(self.cb3, pos=(16, 1), span=(1,2),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.cb3.Bind(wx.EVT_COMBOBOX, self.onCb3)
        self.cb3.Disable()

        self.mark6 = wx.StaticText(self,-1,label=mark_empty)
        self.mark6.SetForegroundColour((0,255,0))
        self.mark6.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark6, pos=(16,3), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(17, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        header3 = wx.StaticText(self,-1,label="Set Direction of Effect")
        header3.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header3, pos=(18, 0), flag=wx.ALL, border=10)

        help_btn3 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn3.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn3.SetForegroundColour(wx.Colour(0,0,255))
        help_btn3.Bind(wx.EVT_BUTTON, self.onHlp3)
        self.sizer.Add(help_btn3, pos=(18,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text5 = wx.StaticText(self,-1,label="Define Direction of Effect")
        self.sizer.Add(text5, pos=(19, 0), flag=wx.ALL, border=10)

        self.ulist1 = ULC.UltimateListCtrl(self, wx.ID_ANY, agwStyle=ULC.ULC_HAS_VARIABLE_ROW_HEIGHT | wx.LC_REPORT | wx.LC_VRULES | wx.LC_HRULES)
        self.ulist1.InsertColumn(col=0, heading="Variable Name",format=0)
        self.ulist1.InsertColumn(col=1, heading="Positive",format=0)
        self.ulist1.InsertColumn(col=2, heading="Negative",format=0)
        self.ulist1.SetColumnWidth(0,ULC.ULC_AUTOSIZE_FILL)
        self.ulist1.SetColumnWidth(1,ULC.ULC_AUTOSIZE_USEHEADER)
        self.ulist1.SetColumnWidth(2,ULC.ULC_AUTOSIZE_USEHEADER)
        self.sizer.Add(self.ulist1,pos=(19,1),span=(5,4),flag=wx.TOP|wx.BOTTOM|wx.EXPAND, border=5)
        self.ulist1.Disable()

        self.enter_btn2 = wx.Button(self, label="Done")
        self.enter_btn2.Bind(wx.EVT_BUTTON, self.onEnt2)
        self.sizer.Add(self.enter_btn2, pos=(19,5), flag=wx.TOP|wx.BOTTOM|wx.RIGHT|wx.LEFT, border=5)
        self.enter_btn2.Disable()

        self.mark5 = wx.StaticText(self,-1,label=mark_empty)
        self.mark5.SetForegroundColour((0,255,0))
        self.mark5.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark5, pos=(19,6), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(24, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        # add previous Button
        self.backBtn = wx.Button(self, label="< Back")
        self.backBtn.Bind(wx.EVT_BUTTON, self.onBack)
        self.sizer.Add(self.backBtn, pos=(25,4),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        # add next button
        self.nextBtn = wx.Button(self, label="Next >")
        self.nextBtn.Bind(wx.EVT_BUTTON, self.onNext)
        self.sizer.Add(self.nextBtn, pos=(25,5),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)
        self.nextBtn.Disable()

        # add cancel button
        self.cancBtn1 = wx.Button(self, label="Cancel")
        self.cancBtn1.Bind(wx.EVT_BUTTON, self.onCanc1)
        self.cancBtn1.SetToolTip(wx.ToolTip('Cancel'))
        self.sizer.Add(self.cancBtn1, pos=(25,6),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        self.SetSizer(self.sizer)
        self.sizer.AddGrowableRow(7)
        self.sizer.AddGrowableRow(17)
        self.sizer.AddGrowableCol(1)
        self.GetParent().SendSizeEvent()
        self.Layout()
        self.Fit()

    def onHlp0(self,event):
        """Help window for variable name"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3B_SetVariableName.html")
        htmlViewerInstance.Show()

    def onChar(self,event):
        keycode = event.GetKeyCode()
        if keycode in ascii_char: #only allow chars in alphabet
            event.Skip()

    def onEnt0(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """enter variable name"""
        inp_var = self.tc0.GetValue()
        if not inp_var:
            wx.MessageBox('Please enter a variable name','Error',wx.OK|wx.ICON_ERROR)
        else: # change this to be based on the database, so that entry can be changed
            global pB_name
            pB_name = "pB_"+inp_var
            if pB_name in var_list:
                wx.MessageBox('The variable already exists','Error',wx.OK|wx.ICON_ERROR)
            else:
                var_list.append(pB_name)
                self.mark1.SetLabel(mark_done) # change tick mark to done
                log.write('\nVariable name: '+inp_var)
                self.radio_bx1.Enable()
                self.tc1.Enable()
                self.enter_btn1.Enable()
                self.tc0.Disable()
                self.enter_btn0.Disable()
            global pBdist
            pBdist = []
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onHlp1(self,event):
        """Help window for buffers"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3ABC_SetBufferSizes.html")
        htmlViewerInstance.Show()

    def onRadBx1(self,event):
        """New buffer or previous buffer"""
        if self.radio_bx1.GetStringSelection()=="Create new buffer": #activate section
            self.tc1.Enable()
            self.enter_btn1.Enable()
            self.cb0.Disable()
            self.done_btn1a.Disable()
        elif self.radio_bx1.GetStringSelection()=="Use previous buffer": #activate other section
            self.cb0.Enable()
            self.done_btn1a.Enable()
            self.tc1.Disable()
            self.enter_btn1.Disable()

    def onEnt1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Enter buffer distances"""
        new_dist = self.tc1.GetValue()
        if not new_dist:
            wx.MessageBox('Please enter a buffer distance','Error',wx.OK|wx.ICON_ERROR)
        else:
            pBdist.append(new_dist)
            self.list_bx1.Append(str(new_dist))
            self.done_btn1.Enable()
            self.del_btn1.Enable()
            self.tc1.Clear()
            self.Parent.statusbar.SetStatusText('Ready')
            del wait

    def onDel1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Remove buffer distance from list"""
        sel1_id = self.list_bx1.GetSelection()
        if sel1_id==-1:
            wx.MessageBox('Click on a buffer distance in the box first, then click Remove.','Error',wx.OK|wx.ICON_ERROR)
        else:
            del_dist = self.list_bx1.GetString(sel1_id)
            pBdist.remove(int(del_dist))
            self.list_bx1.Delete(sel1_id)
            if not pBdist:
                self.done_btn1.Disable()
                self.del_btn1.Disable()
            self.Parent.statusbar.SetStatusText('Ready')
            del wait

    def onDone1(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Create buffers"""
        global pB_buffer
        if sorted(pBdist) not in bufDists.values(): #check if multiple ring buffer already exists, if not create one and add to dictionary
            k = 'buffer'+time.strftime('%y%m%d_%H%M%S') #create new key
            bufDists.update({k : sorted(pBdist)}) # add to dictionary
            pB_buffer= out_fds+"\\"+k
            arcpy.MultipleRingBuffer_analysis(sites, pB_buffer, pBdist, "", "", "NONE")
            # append buffer distances to cb0 in panel 3B
            self.cb0.SetValue('')
            self.cb0.Clear()
            self.cb0.Append([str(i) for i in bufDists.values()])
            # append buffer distances to cb0 in panel 3A
            self.Parent.panel3A.cb0.SetValue('')
            self.Parent.panel3A.cb0.Clear()
            self.Parent.panel3A.cb0.Append([str(i) for i in bufDists.values()])
            # append buffer distances to cb0 in panel 3C
            self.Parent.panel3C.cb0.SetValue('')
            self.Parent.panel3C.cb0.Clear()
            self.Parent.panel3C.cb0.Append([str(i) for i in bufDists.values()])

        else:
            for key,value in bufDists.items():
                if value == pBdist:
                    pB_buffer = out_fds+"\\"+key # use existing buffer

        self.mark2.SetLabel(mark_done) # change tick mark to done
        log.write('\nBuffer distances: '+str(sorted(pBdist)))
        log.write('\nBuffer feature class: '+str(pB_buffer))
        arcpy.AddMessage('bufDists:')
        arcpy.AddMessage(bufDists)
        self.cb1.Enable()
        self.ulist1.Enable()
        self.radio_bx1.Disable()
        self.radio_bx1.SetSelection(0)
        self.tc1.Disable()
        self.enter_btn1.Disable()
        self.del_btn1.Disable()
        self.done_btn1.Disable()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onCb0(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Select list of buffers"""
        prev_dist_str = self.cb0.GetValue().strip('[]').split(',')
        prev_dist = [int(i) for i in prev_dist_str]
        pBdist.extend(prev_dist)
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onDone1a(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Create buffers"""
        global pB_buffer
        for key,value in bufDists.items():
            if value == pBdist: # look up key associated with value
                pB_buffer = out_fds+"\\"+key # use existing buffer

        self.mark2a.SetLabel(mark_done)
        log.write('\nBuffer distances: '+str(sorted(pBdist)))
        log.write('\nBuffer feature class: '+str(pB_buffer))
        self.cb1.Enable()
        self.ulist1.Enable()
        self.radio_bx1.Disable()
        self.radio_bx1.SetSelection(0)
        self.cb0.Disable()
        self.done_btn1a.Disable()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onHlp2(self,event):
        """Help window for input data"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3B_SetInputData.html")
        htmlViewerInstance.Show()

    def onCb1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Select Line feature class"""
        self.cb2.Clear()
        self.cb3.Clear()
        self.ulist1.DeleteAllItems()
        global pB
        pB = self.cb1.GetValue()
        global pB_fldsNamesDict
        pB_fldsNamesDict={}
        num_row=int(arcpy.GetCount_management(pB).getOutput(0))
        if num_row>0: #check it contains text fields
            str_fields = [f.name for f in arcpy.ListFields(pB,"","String")] # get text fields
            str_fields.sort()
            if not str_fields:
                wx.MessageBox('The selected feature class does not contain any text fields','Error',wx.OK|wx.ICON_ERROR)
            else:
                self.cb2.Append(str_fields)
                self.mark3.SetLabel(mark_done) # change tick mark to done
                log.write('\nLine Feature Class: '+pB)
        elif num_row==0:#check that it contains features
            wx.MessageBox('The selected feature class is empty','Error',wx.OK|wx.ICON_ERROR)

        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        self.cb2.Enable()

    def onCb2(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Select category"""
        self.ulist1.DeleteAllItems()
        for k,v in list(pB_fldsNamesDict.items()): # check if value is already in fldsNamesDict
            if v == 'pB_cat':
                del pB_fldsNamesDict[k] # delete if there
        global pB_cat
        pB_cat = self.cb2.GetValue()
        # strip any endspaces from strings and replace whitespace with underscore
        with arcpy.da.UpdateCursor(pB, pB_cat) as cursor:
            for row in cursor:
                row=[i.strip() if i is not None else None for i in row]
                row=[i.replace(" ","") if i is not None else None for i in row]
                row=[i.replace("_","") if i is not None else None for i in row]
                row=[i.replace(".","") if i is not None else None for i in row]
                cursor.updateRow(row)
        # Get list of unique values in field
        pBCats= unique_values(pB,pB_cat)
        # add buffer distances to names
        global pBCatsBuffer
        pBCatsBuffer=list()
        for cat in pBCats:
            if cat is None:
                wx.MessageBox('The selected field contains missing values','Error',wx.OK|wx.ICON_ERROR)
            else:
                for buf in pBdist:
                    pBCatsBuffer.append(pB_name+'_'+cat+'_'+str(buf)+'_sum')

        # populate ulist
        self.radios = []
        myMethods = ['Positive','Negative']
        for item in range(len(pBCatsBuffer)):
            self.ulist1.InsertStringItem(item, str(pBCatsBuffer[item]))
            for rad in range(1,3):
                cat = pBCatsBuffer[item]
                met = myMethods[rad - 1]
                name_of_radio = "{cat}_{met}".format(cat=cat, met=met)
                self.ulist1.SetStringItem(item, rad, "")
                if rad==1:
                    style=wx.RB_GROUP
                else:
                    style=0
                self.radBt = wx.RadioButton(self.ulist1, wx.ID_ANY, "", style=style,name=name_of_radio)
                self.ulist1.SetItemWindow(item,rad,self.radBt,expand=True)
                self.radios.append(self.radBt)
        self.ulist1.Enable()
        self.enter_btn2.Enable()
        # add field name to dictionary
        new_entry={str(pB_cat):'pB_cat'}
        pB_fldsNamesDict.update(new_entry)

        self.mark4.SetLabel(mark_done) # change tick mark to done
        self.Parent.statusbar.SetStatusText('Ready')
        self.radio_bx.Enable()
        del wait
        log.write('\nCategory field: '+str(pB_cat))

    def onRadBx(self,event):
        """Aggregation method"""
        if self.radio_bx.GetStringSelection()=="Length weighted value" or self.radio_bx.GetStringSelection()=="Length * Value":
            num_fields = [f.name for f in arcpy.ListFields(pB,"",'Double') if not f.required] #get numeric fields
            num_fields.extend([f.name for f in arcpy.ListFields(pB,"",'Integer')])
            num_fields.extend([f.name for f in arcpy.ListFields(pB,"",'Single')])
            num_fields.extend([f.name for f in arcpy.ListFields(pB,"",'SmallInteger')])
            num_fields.sort()
            if not num_fields:
                wx.MessageBox('The selected feature class contains no numeric fields','Error',wx.OK|wx.ICON_ERROR)
            else:
                self.cb3.Append(num_fields)
                self.text6.Enable()
                self.cb3.Enable()
                self.ulist1.Disable()
                self.enter_btn2.Disable()
                self.ulist1.DeleteAllItems()
        else:
            self.text6.Disable()
            self.cb3.Disable()
            for k,v in list(pB_fldsNamesDict.items()):
                if v == 'pB_val':
                    del pB_fldsNamesDict[k]

    def onCb3(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Select value field"""
        for k,v in list(pB_fldsNamesDict.items()):
            if v == 'pB_val':
                del pB_fldsNamesDict[k]
        pB_val = self.cb3.GetValue()
        # attach method to pBCatsBuffer
        if self.radio_bx.GetStringSelection()=="Length weighted value":
            for index,o in enumerate(pBCatsBuffer):
                pBCatsBuffer[index] = o[:-4]+'_wtv'
        else:
            for index,o in enumerate(pBCatsBuffer):
                pBCatsBuffer[index] = o[:-4]+'_mtv'
        # populate ulist
        self.radios = []
        myMethods = ['Positive','Negative']
        for item in range(len(pBCatsBuffer)):
            self.ulist1.InsertStringItem(item, str(pBCatsBuffer[item]))
            for rad in range(1,3):
                cat = pBCatsBuffer[item]
                met = myMethods[rad - 1]
                name_of_radio = "{cat}_{met}".format(cat=cat, met=met)
                self.ulist1.SetStringItem(item, rad, "")
                if rad==1:
                    style=wx.RB_GROUP
                else:
                    style=0
                self.radBt = wx.RadioButton(self.ulist1, wx.ID_ANY, "", style=style,name=name_of_radio)
                self.ulist1.SetItemWindow(item,rad,self.radBt,expand=True)
                self.radios.append(self.radBt)
        self.ulist1.Enable()
        self.enter_btn2.Enable()
        # add field name to dictionary
        new_entry={str(pB_val):'pB_val'}
        pB_fldsNamesDict.update(new_entry)
        self.mark6.SetLabel(mark_done) # change tick mark to done
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onHlp3(self,event):
        """Help window for direction of effect"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3ABC_SetSourceSink.html")
        htmlViewerInstance.Show()

    def onEnt2(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Define Direction of Effect"""
        results_dict = {}
        for i in self.radios:
            if i.GetValue()==True:
                n = i.GetName()
                index = n.rfind('_')
                cat = n[0:index]
                met = n[index+1:]
                results_dict[cat]= met
        sourcesink.update(results_dict)

        if arcpy.Exists(out_fds+"\\buffer_"+pB_name):
            self.nextBtn.Enable()

        self.mark5.SetLabel(mark_done) # change tick mark to done
        self.Parent.statusbar.SetStatusText('Ready')
        self.cb1.Disable()
        self.cb2.Disable()
        self.ulist1.Disable()
        self.enter_btn2.Disable()
        self.nextBtn.Enable()
        del wait
        log.write('\nAggregation method: '+self.radio_bx.GetStringSelection())
        if self.radio_bx.GetStringSelection()=="Length weighted value" or self.radio_bx.GetStringSelection()=="Length * Value":
            log.write('\nValue field: '+self.cb3.GetValue())
        log.write('\nVariable definitions: ')
        for k, v in sorted(results_dict.items()):
            log.write('\n{:<30}: {:<6}'.format(k,v))

    def onBack(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Return to panel3"""
        try:
            var_list.remove(pB_name)
            for i in self.radios: # remove from sourcesink dictionary
                if i.GetValue()==True:
                    n = i.GetName()
                    index = n.rfind('_')
                    cat = pB_name+'_'+n[0:index]
                    sourcesink.pop(cat, None)
        except:
            pass
        if arcpy.Exists(out_fds+"\\buffer_"+pB_name):
            arcpy.Delete_management(out_fds+"\\buffer_"+pB_name)
            del pBdist[:]
        self.tc0.Clear()
        self.tc1.Clear()
        self.list_bx1.Clear()
        self.cb2.Clear()
        self.ulist1.DeleteAllItems()
        self.cb3.Clear()
        self.mark1.SetLabel(mark_empty)
        self.mark2.SetLabel(mark_empty)
        self.mark2a.SetLabel(mark_empty)
        self.mark3.SetLabel(mark_empty)
        self.mark4.SetLabel(mark_empty)
        self.mark5.SetLabel(mark_empty)
        self.mark6.SetLabel(mark_empty)
        self.tc1.Disable()
        self.enter_btn1.Disable()
        self.del_btn1.Disable()
        self.done_btn1.Disable()
        self.radio_bx1.Disable()
        self.radio_bx1.SetSelection(0)
        self.cb0.Disable()
        self.done_btn1a.Disable()
        self.cb1.Disable()
        self.cb2.Disable()
        self.ulist1.Disable()
        self.nextBtn.Disable()
        self.tc0.Enable()
        self.enter_btn0.Enable()
        self.radio_bx.SetSelection(0)
        self.radio_bx.Disable()
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel3,1,wx.EXPAND)
        newsize = self.Parent.panel3.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel3.Show()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        log.write(time.strftime("\n\nLine in Buffer - Stopped by user without completion: %A %d %b %Y %H:%M:%S", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel3B','back',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel3','start',datetime())''')
        db.commit()

    def onNext(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """"""
        fieldmappings=customFieldMap(pB,pB_fldsNamesDict) # create fieldmap
        arcpy.FeatureClassToFeatureClass_conversion(pB,out_fds,pB_name+"_temp", "", fieldmappings) # copy the feature class
        max_dist=max(pBdist) # maximum buffer distance
        arcpy.Buffer_analysis(out_fds+"\\studyArea",out_fds+"\\analysisExtent_"+pB_name,max_dist) # create analysis extent
        # Compare the spatial extent against the analysis extent
        ext_an = arcpy.Describe(out_fds+"\\analysisExtent_"+pB_name).extent
        ext_2 = arcpy.Describe(out_fds+"\\"+pB_name+"_temp").extent
        if ext_2.contains(ext_an)==False:
            wx.MessageBox('The spatial extent of the line feature class is too small or completely outside of the study area','Warning',wx.OK|wx.ICON_WARNING)
            log.write('\n+++ WARNING+++ The spatial extent of the line feature class is too small or completely outside of the study area. This may result in empty intersects. If all intersects are empty, no predictor variable will be created.\n')
        arcpy.Clip_analysis(out_fds+"\\"+pB_name+"_temp",out_fds+"\\analysisExtent_"+pB_name,out_fds+"\\"+pB_name) # Clip data to analysis extent
        arcpy.Delete_management(out_fds+"\\"+pB_name+"_temp") # delete temp data
        arcpy.Delete_management(out_fds+"\\analysisExtent_"+pB_name) # delete temp data
        arcpy.AddField_management(out_fds+"\\"+pB_name,"pB_cat_INT","SHORT") # add integer id
        valueSet = set([r[0] for r in arcpy.da.SearchCursor(out_fds+"\\"+pB_name, ["pB_cat"])])
        valueList = list(valueSet)
        valueList.sort()
        updateRows = arcpy.da.UpdateCursor(out_fds+"\\"+pB_name, ["pB_cat","pB_cat_INT"])
        for updateRow in updateRows:
            updateRow[1] = valueList.index(updateRow[0]) + 1
            updateRows.updateRow(updateRow)
        if FieldExist(out_fds+"\\"+pB_name,"pB_val"):
            arcpy.AddField_management(out_fds+"\\"+pB_name,"origLength","DOUBLE") # add field for original area
            cursor=arcpy.da.UpdateCursor(out_fds+"\\"+pB_name, ["origLength","Shape_Length"])
            for row in cursor:
                row[0]=row[1]
                cursor.updateRow(row)
        arcpy.PairwiseIntersect_analysis ([out_fds+"\\"+pB_name, pB_buffer], out_fds+"\\"+pB_name+"_Intersect", "ALL", "", "")
        intersect_var = pB_name+"_Intersect"
        # move data to sql database
        fc_to_sql(conn,out_fds+"\\"+pB_name)
        conn.execute("CREATE INDEX "+pB_name+"_idx on "+pB_name+"(pB_cat_INT);")
        fc_to_sql(conn,out_fds+"\\"+intersect_var)
        conn.execute("CREATE INDEX "+intersect_var+"_idx on "+intersect_var+"(siteID_INT, pB_cat_INT, distance);")
        db.commit()
        arcpy.Delete_management(out_fds+"\\"+intersect_var)
        # add weighted value
        conn.execute("ALTER TABLE "+intersect_var+" ADD pB_wtval float64")
        if FieldExist(out_fds+"\\"+pB_name,"pB_val") and self.radio_bx.GetStringSelection()=="Length weighted value":
            conn.execute("UPDATE "+intersect_var+" SET pB_wtval=(Shape_Length/origLength)*pB_val")
        elif FieldExist(out_fds+"\\"+pB_name,"pB_val") and self.radio_bx.GetStringSelection()=="Length * Value":
            conn.execute("UPDATE "+intersect_var+" SET pB_wtval=Shape_Length*pB_val")
        else:
            conn.execute("UPDATE "+intersect_var+" SET pB_wtval=Shape_Length")
        db.commit()
        # calculate sum by site ID, category and buffer size
        if self.radio_bx.GetStringSelection()=="Total length":
            qry = "CREATE TABLE temp AS \
                        SELECT siteID_INT \
                             ,pB_cat_INT \
                             ,distance \
                             ,SUM(pB_wtval) AS value \
	                         ,('"+pB_name+"_'||pB_cat||'_'||cast(distance as int)||'_sum') AS varName \
                        FROM "+intersect_var+" \
                        GROUP BY siteID_INT, pB_cat_INT, distance"
        elif self.radio_bx.GetStringSelection()=="Length weighted value":
            qry = "CREATE TABLE temp AS \
                        SELECT siteID_INT \
                             ,pB_cat_INT \
                             ,distance \
                             ,SUM(pB_wtval) AS value \
	                         ,('"+pB_name+"_'||pB_cat||'_'||cast(distance as int)||'_wtv') AS varName \
                        FROM "+intersect_var+" \
                        GROUP BY siteID_INT, pB_cat_INT, distance"
        else:
            qry = "CREATE TABLE temp AS \
                        SELECT siteID_INT \
                             ,pB_cat_INT \
                             ,distance \
                             ,SUM(pB_wtval) AS value \
	                         ,('"+pB_name+"_'||pB_cat||'_'||cast(distance as int)||'_mtv') AS varName \
                        FROM "+intersect_var+" \
                        GROUP BY siteID_INT, pB_cat_INT, distance"
        conn.execute(qry)
        vars_sql_temp=list(conn.execute("SELECT DISTINCT varName FROM temp")) # make a list of variable names
        # add a column for each variable name
        for var in vars_sql_temp:
            conn.execute('ALTER TABLE dat4stats ADD {} float64'.format(var[0])) # if I add second query I get an operational error
        db.commit()
        # add values to each column
        for var in vars_sql_temp:
            qry="UPDATE dat4stats \
                 SET "+var[0]+" = (\
                 SELECT value \
                 FROM temp \
                 WHERE siteID_INT = dat4stats.siteID_INT and varName='"+var[0]+"')"
            conn.execute(qry)
        db.commit()
        # replace missing with zeros
        for var in vars_sql_temp:
            qry="UPDATE dat4stats \
                 SET "+var[0]+" = 0 \
                 WHERE "+var[0]+" IS NULL"
            conn.execute(qry)
        db.commit()
        conn.execute("DROP TABLE temp")
        conn.execute("DROP TABLE "+intersect_var)
        db.commit()
        # predictor names to panel 3
        WizardPanel3.list_ctrl.DeleteAllItems()
        cursor = conn.execute("SELECT * FROM dat4stats")
        varnames = list([x[0] for x in cursor.description])
        prednames = [name for name in varnames if (name[0] =='p')]
        # pred_list.extend(prednames)
        index=0
        for i in prednames:
            WizardPanel3.list_ctrl.InsertItem(index,i)
            WizardPanel3.list_ctrl.SetItem(index,1,sourcesink[i])
            if index % 2:
                WizardPanel3.list_ctrl.SetItemBackgroundColour(index, col="white")
            else:
                WizardPanel3.list_ctrl.SetItemBackgroundColour(index, col=(222,234,246))
            index+=1
        WizardPanel3.list_ctrl.SetColumnWidth(0,wx.LIST_AUTOSIZE)
        WizardPanel3.list_ctrl.SetColumnWidth(1,wx.LIST_AUTOSIZE)
        log.write(time.strftime("\nLine in Buffer - End Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel3B','stop',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel3','start',datetime())''')
        db.commit()
        # clear p3B
        del pBdist[:]
        self.tc0.Clear()
        self.tc1.Clear()
        self.list_bx1.Clear()
        self.cb2.Clear()
        self.ulist1.DeleteAllItems()
        self.cb3.Clear()
        self.mark1.SetLabel(mark_empty)
        self.mark2.SetLabel(mark_empty)
        self.mark2a.SetLabel(mark_empty)
        self.mark3.SetLabel(mark_empty)
        self.mark4.SetLabel(mark_empty)
        self.mark5.SetLabel(mark_empty)
        self.mark6.SetLabel(mark_empty)
        self.tc1.Disable()
        self.enter_btn1.Disable()
        self.del_btn1.Disable()
        self.done_btn1.Disable()
        self.radio_bx1.Disable()
        self.radio_bx1.SetSelection(0)
        self.cb0.Disable()
        self.done_btn1a.Disable()
        self.cb1.Disable()
        self.cb2.Disable()
        self.ulist1.Disable()
        self.nextBtn.Disable()
        self.tc0.Enable()
        self.enter_btn0.Enable()
        self.radio_bx.SetSelection(0)
        self.radio_bx.Disable()
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel3,1,wx.EXPAND)
        newsize = self.Parent.panel3.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel3.Show()

        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onCanc1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Cancel Wizard"""
        dlg = wx.MessageDialog(None, "Cancelling the wizard will delete all progress. Do you wish to cancel the wizard?",'Cancel',wx.YES_NO | wx.ICON_WARNING)
        result = dlg.ShowModal()
        if result == wx.ID_YES:
            arcpy.AddMessage((time.strftime("Cancelled by user: %A %d %b %Y %H:%M:%S", time.localtime())))
            try:
                log.close()
                logging.shutdown()
                db.close()
                shutil.rmtree(out_path)
            except NameError as e:
                arcpy.AddMessage(e)
            except OSError as e:
                arcpy.AddMessage(e)
            del wait
            self.Parent.Close()
        else:
            self.Parent.statusbar.SetStatusText('Ready')
            del wait
#-------------------------------------------------------------------------------
class WizardPanel3C(wx.Panel):
    """Page 3C"""

    def __init__(self, parent, title=None):
        """Constructor"""
        wx.Panel.__init__(self, parent)
        self.locale = wx.Locale(wx.LANGUAGE_ENGLISH)
        wx.ToolTip.Enable(True)

        self.sizer = wx.GridBagSizer(0,0)

        title2 = wx.StaticText(self, -1, "Point Count or Value within Buffer")
        title2.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(title2, pos=(0,0), span=(1,5),flag=wx.TOP|wx.LEFT|wx.BOTTOM, border=10)

        self.sizer.Add(wx.StaticLine(self), pos=(1, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM, border=5)

        header0 = wx.StaticText(self,-1,label="Set Variable Name")
        header0.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header0, pos=(2, 0), flag=wx.ALL, border=10)

        help_btn0 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn0.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn0.SetForegroundColour(wx.Colour(0,0,255))
        help_btn0.Bind(wx.EVT_BUTTON, self.onHlp0)
        self.sizer.Add(help_btn0, pos=(2,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text1 = wx.StaticText(self,-1,label="Variable Name")
        self.sizer.Add(text1, pos=(3, 0), flag=wx.ALL, border=10)

        self.tc0 = wx.TextCtrl(self, -1)
        self.tc0.SetMaxLength(20)
        self.tc0.SetToolTip(wx.ToolTip('Enter a name for the predictor variable'))
        self.tc0.Bind(wx.EVT_CHAR,self.onChar)
        self.sizer.Add(self.tc0, pos=(3, 1), span=(1,2), flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=5)

        self.enter_btn0 = wx.Button(self, label="Enter")
        self.enter_btn0.Bind(wx.EVT_BUTTON, self.onEnt0)
        self.sizer.Add(self.enter_btn0, pos=(3,3), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)

        self.mark1 = wx.StaticText(self,-1,label=mark_empty)
        self.mark1.SetForegroundColour((0,255,0))
        self.mark1.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark1, pos=(3,4), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(4, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        header1 = wx.StaticText(self,-1,label="Set Buffer Sizes")
        header1.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header1, pos=(5, 0), flag=wx.ALL, border=10)

        help_btn1 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn1.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn1.SetForegroundColour(wx.Colour(0,0,255))
        help_btn1.Bind(wx.EVT_BUTTON, self.onHlp1)
        self.sizer.Add(help_btn1, pos=(5,1), flag=wx.TOP|wx.BOTTOM, border=5)

        self.radio_bx1 = wx.RadioBox(self,-1,label="Buffers",choices=["Create new buffer","Use previous buffer"], majorDimension=0, style=wx.RA_SPECIFY_COLS)
        self.sizer.Add(self.radio_bx1, pos=(6,0), span=(1,6), flag=wx.ALL|wx.EXPAND, border=10)
        self.radio_bx1.Bind(wx.EVT_RADIOBOX,self.onRadBx1)
        self.radio_bx1.Disable()

        text2 = wx.StaticText(self,-1,label="Create Buffer Distance")
        self.sizer.Add(text2, pos=(7, 0), flag=wx.ALL, border=10)

        self.tc1 = NumCtrl(self, integerWidth=6, fractionWidth=0, allowNegative=False, allowNone = True)
        self.tc1.SetToolTip(wx.ToolTip('Enter one or more buffer distances'))
        self.sizer.Add(self.tc1, pos=(7, 1), span=(1,2), flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=5)
        self.tc1.Disable()

        self.enter_btn1 = wx.Button(self, label="Add")
        self.enter_btn1.Bind(wx.EVT_BUTTON, self.onEnt1)
        self.sizer.Add(self.enter_btn1, pos=(7,3), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)
        self.enter_btn1.Disable()

        self.list_bx1 = wx.ListBox(self,-1,choices=[])
        self.sizer.Add(self.list_bx1, pos=(8,1), span=(2,2), flag=wx.EXPAND|wx.BOTTOM, border=5)

        self.del_btn1 = wx.Button(self, label="Remove")
        self.del_btn1.SetToolTip(wx.ToolTip('Remove selected buffer distance'))
        self.del_btn1.Bind(wx.EVT_BUTTON, self.onDel1)
        self.sizer.Add(self.del_btn1, pos=(8,3), flag=wx.RIGHT|wx.LEFT, border=5)
        self.del_btn1.Disable()

        self.done_btn1 = wx.Button(self, label="Done")
        self.done_btn1.SetToolTip(wx.ToolTip('Create buffers'))
        self.done_btn1.Bind(wx.EVT_BUTTON, self.onDone1)
        self.sizer.Add(self.done_btn1, pos=(9,3), flag=wx.RIGHT|wx.LEFT|wx.TOP|wx.BOTTOM, border=5)
        self.done_btn1.Disable()

        self.mark2 = wx.StaticText(self,-1,label=mark_empty)
        self.mark2.SetForegroundColour((0,255,0))
        self.mark2.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark2, pos=(9,4), flag=wx.ALL, border=5)

        text2a = wx.StaticText(self,-1,label="Select Buffer Distance")
        self.sizer.Add(text2a, pos=(10, 0), flag=wx.ALL, border=10)

        self.cb0 = wx.ComboBox(self,choices=[],style=wx.CB_DROPDOWN)
        self.cb0.SetToolTip(wx.ToolTip('From the dropdown list select a list of previously used buffer distances'))
        self.sizer.Add(self.cb0, pos=(10, 1), span=(1,3),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.cb0.Bind(wx.EVT_COMBOBOX, self.onCb0)
        self.cb0.Disable()

        self.done_btn1a = wx.Button(self, label="Done")
        self.done_btn1a.SetToolTip(wx.ToolTip('Select buffers'))
        self.done_btn1a.Bind(wx.EVT_BUTTON, self.onDone1a)
        self.sizer.Add(self.done_btn1a, pos=(10,4), flag=wx.RIGHT|wx.LEFT|wx.TOP|wx.BOTTOM, border=5)
        self.done_btn1a.Disable()

        self.mark2a = wx.StaticText(self,-1,label=mark_empty)
        self.mark2a.SetForegroundColour((0,255,0))
        self.mark2a.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark2a, pos=(10,5), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(11, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        header2 = wx.StaticText(self,-1,label="Set Input Data")
        header2.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header2, pos=(12, 0), flag=wx.ALL, border=10)

        help_btn2 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn2.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn2.SetForegroundColour(wx.Colour(0,0,255))
        help_btn2.Bind(wx.EVT_BUTTON, self.onHlp2)
        self.sizer.Add(help_btn2, pos=(12,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text3 = wx.StaticText(self,-1,label="Point Feature Class")
        self.sizer.Add(text3, pos=(13, 0), flag=wx.ALL, border=10)

        self.cb1 = wx.ComboBox(self,choices=[],style=wx.CB_DROPDOWN)
        self.cb1.SetToolTip(wx.ToolTip('From the dropdown list select the feature class containing the point data'))
        self.sizer.Add(self.cb1, pos=(13, 1), span=(1,2),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.cb1.Bind(wx.EVT_COMBOBOX, self.onCb1)
        self.cb1.Disable()

        self.mark3 = wx.StaticText(self,-1,label=mark_empty)
        self.mark3.SetForegroundColour((0,255,0))
        self.mark3.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark3, pos=(13,3), flag=wx.ALL, border=5)

        text4 = wx.StaticText(self,-1,label="Category Field")
        self.sizer.Add(text4, pos=(14, 0), flag=wx.ALL, border=10)

        self.cb2 = wx.ComboBox(self,choices=[],style=wx.CB_DROPDOWN)
        self.cb2.SetToolTip(wx.ToolTip('From the dropdown list select the field containing the categories'))
        self.sizer.Add(self.cb2, pos=(14, 1), span=(1,2),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.cb2.Bind(wx.EVT_COMBOBOX, self.onCb2)
        self.cb2.Disable()

        self.mark4 = wx.StaticText(self,-1,label=mark_empty)
        self.mark4.SetForegroundColour((0,255,0))
        self.mark4.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark4, pos=(14,3), flag=wx.ALL, border=5)

        self.radio_bx = wx.RadioBox(self,-1,label="Aggregation Method",choices=["Point count","Sum of values","Mean of values","Median of values"], majorDimension=0, style=wx.RA_SPECIFY_COLS)
        self.sizer.Add(self.radio_bx, pos=(15,0), span=(1,6), flag=wx.TOP|wx.ALL, border=10)
        self.radio_bx.Bind(wx.EVT_RADIOBOX,self.onRadBx)
        self.radio_bx.Disable()

        self.text6 = wx.StaticText(self,-1,label="Value Field")
        self.sizer.Add(self.text6, pos=(16, 0), flag=wx.ALL, border=10)
        self.text6.Disable()

        self.cb3 = wx.ComboBox(self,choices=[],style=wx.CB_DROPDOWN)
        self.cb3.SetToolTip(wx.ToolTip('From the dropdown list select the field containing values to be summarised'))
        self.sizer.Add(self.cb3, pos=(16, 1), span=(1,2),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.cb3.Bind(wx.EVT_COMBOBOX, self.onCb3)
        self.cb3.Disable()

        self.mark6 = wx.StaticText(self,-1,label=mark_empty)
        self.mark6.SetForegroundColour((0,255,0))
        self.mark6.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark6, pos=(16,3), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(17, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        header3 = wx.StaticText(self,-1,label="Set Direction of Effect")
        header3.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header3, pos=(18, 0), flag=wx.ALL, border=10)

        help_btn3 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn3.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn3.SetForegroundColour(wx.Colour(0,0,255))
        help_btn3.Bind(wx.EVT_BUTTON, self.onHlp3)
        self.sizer.Add(help_btn3, pos=(18,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text5 = wx.StaticText(self,-1,label="Define Direction of Effect")
        self.sizer.Add(text5, pos=(19, 0), flag=wx.ALL, border=10)

        self.ulist1 = ULC.UltimateListCtrl(self, wx.ID_ANY, agwStyle=ULC.ULC_HAS_VARIABLE_ROW_HEIGHT | wx.LC_REPORT | wx.LC_VRULES | wx.LC_HRULES)
        self.ulist1.InsertColumn(col=0, heading="Variable Name",format=0)
        self.ulist1.InsertColumn(col=1, heading="Positive",format=0)
        self.ulist1.InsertColumn(col=2, heading="Negative",format=0)
        self.ulist1.SetColumnWidth(0,ULC.ULC_AUTOSIZE_FILL)
        self.ulist1.SetColumnWidth(1,ULC.ULC_AUTOSIZE_USEHEADER)
        self.ulist1.SetColumnWidth(2,ULC.ULC_AUTOSIZE_USEHEADER)
        self.sizer.Add(self.ulist1,pos=(19,1),span=(5,4),flag=wx.TOP|wx.BOTTOM|wx.EXPAND, border=5)
        self.ulist1.Disable()

        self.enter_btn2 = wx.Button(self, label="Done")
        self.enter_btn2.Bind(wx.EVT_BUTTON, self.onEnt2)
        self.sizer.Add(self.enter_btn2, pos=(19,5), flag=wx.TOP|wx.BOTTOM|wx.RIGHT|wx.LEFT, border=5)
        self.enter_btn2.Disable()

        self.mark5 = wx.StaticText(self,-1,label=mark_empty)
        self.mark5.SetForegroundColour((0,255,0))
        self.mark5.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark5, pos=(19,6), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(24, 0), span=(1, 6),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        # add previous Button
        self.backBtn = wx.Button(self, label="< Back")
        self.backBtn.Bind(wx.EVT_BUTTON, self.onBack)
        self.sizer.Add(self.backBtn, pos=(25,4),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        # add next button
        self.nextBtn = wx.Button(self, label="Next >")
        self.nextBtn.Bind(wx.EVT_BUTTON, self.onNext)
        self.sizer.Add(self.nextBtn, pos=(25,5),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)
        self.nextBtn.Disable()

        # add cancel button
        self.cancBtn1 = wx.Button(self, label="Cancel")
        self.cancBtn1.Bind(wx.EVT_BUTTON, self.onCanc1)
        self.cancBtn1.SetToolTip(wx.ToolTip('Cancel'))
        self.sizer.Add(self.cancBtn1, pos=(25,6),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        self.SetSizer(self.sizer)
        self.sizer.AddGrowableRow(7)
        self.sizer.AddGrowableRow(17)
        self.sizer.AddGrowableCol(1)
        self.GetParent().SendSizeEvent()
        self.Layout()
        self.Fit()

    def onHlp0(self,event):
        """Help window for variable name"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3C_SetVariableName.html")
        htmlViewerInstance.Show()

    def onChar(self,event):
        keycode = event.GetKeyCode()
        if keycode in ascii_char: #only allow chars in alphabet
            event.Skip()

    def onEnt0(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """enter variable name"""
        inp_var = self.tc0.GetValue()
        if not inp_var:
            wx.MessageBox('Please enter a variable name','Error',wx.OK|wx.ICON_ERROR)
        else: # change this to be based on the database, so that entry can be changed
            global pC_name
            pC_name = "pC_"+inp_var
            if pC_name in var_list:
                wx.MessageBox('The variable already exists','Error',wx.OK|wx.ICON_ERROR)
            else:
                var_list.append(pC_name)
                self.mark1.SetLabel(mark_done) # change tick mark to done
                log.write('\nVariable name: '+inp_var)
                self.radio_bx1.Enable()
                self.tc1.Enable()
                self.enter_btn1.Enable()
                self.tc0.Disable()
                self.enter_btn0.Disable()
            global pCdist
            pCdist = []
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onHlp1(self,event):
        """Help window for buffers"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3ABC_SetBufferSizes.html")
        htmlViewerInstance.Show()

    def onRadBx1(self,event):
        """New buffer or previous buffer"""
        if self.radio_bx1.GetStringSelection()=="Create new buffer": #activate create section
            self.tc1.Enable()
            self.enter_btn1.Enable()
            self.cb0.Disable()
            self.done_btn1a.Disable()
        elif self.radio_bx1.GetStringSelection()=="Use previous buffer": #activate select section
            self.cb0.Enable()
            self.done_btn1a.Enable()
            self.tc1.Disable()
            self.enter_btn1.Disable()

    def onEnt1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Enter buffer distances"""
        new_dist = self.tc1.GetValue()
        if not new_dist:
            wx.MessageBox('Please enter a buffer distance','Error',wx.OK|wx.ICON_ERROR)
        else:
            pCdist.append(new_dist)
            self.list_bx1.Append(str(new_dist))
            self.del_btn1.Enable()
            self.done_btn1.Enable()
            self.tc1.Clear()
            self.Parent.statusbar.SetStatusText('Ready')
            del wait

    def onDel1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Remove buffer distance from list"""
        sel1_id = self.list_bx1.GetSelection()
        if sel1_id==-1:
            wx.MessageBox('Click on a buffer distance in the box first, then click Remove.','Error',wx.OK|wx.ICON_ERROR)
        else:
            del_dist = self.list_bx1.GetString(sel1_id)
            pCdist.remove(int(del_dist))
            self.list_bx1.Delete(sel1_id)
            if not pCdist:
                self.del_btn1.Disable()
                self.done_btn1.Disable()
            self.Parent.statusbar.SetStatusText('Ready')
            del wait

    def onDone1(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Create buffers"""
        global pC_buffer
        if sorted(pCdist) not in bufDists.values(): #check if multiple ring buffer already exists, if not create one and add to dictionary
            k = 'buffer'+time.strftime('%y%m%d_%H%M%S') #create new key
            bufDists.update({k : sorted(pCdist)}) # add to dictionary
            pC_buffer= out_fds+"\\"+k
            arcpy.MultipleRingBuffer_analysis(sites, pC_buffer, pCdist, "", "", "NONE")
            # append buffer distances to cb0 in panel 3C
            self.cb0.SetValue('')
            self.cb0.Clear()
            self.cb0.Append([str(i) for i in bufDists.values()])
            # append buffer distances to cb0 in panel 3A
            self.Parent.panel3A.cb0.SetValue('')
            self.Parent.panel3A.cb0.Clear()
            self.Parent.panel3A.cb0.Append([str(i) for i in bufDists.values()])
            # append buffer distances to cb0 in panel 3B
            self.Parent.panel3B.cb0.SetValue('')
            self.Parent.panel3B.cb0.Clear()
            self.Parent.panel3B.cb0.Append([str(i) for i in bufDists.values()])

        else:
            for key,value in bufDists.items():
                if value == pCdist:
                    pC_buffer = out_fds+"\\"+key # use existing buffer

        self.mark2.SetLabel(mark_done) # change tick mark to done
        log.write('\nBuffer distances: '+str(sorted(pCdist)))
        log.write('\nBuffer feature class: '+str(pC_buffer))
        arcpy.AddMessage('bufDists:')
        arcpy.AddMessage(bufDists)
        self.cb1.Enable()
        self.ulist1.Enable()
        self.radio_bx1.Disable()
        self.radio_bx1.SetSelection(0)
        self.tc1.Disable()
        self.enter_btn1.Disable()
        self.del_btn1.Disable()
        self.done_btn1.Disable()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onCb0(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Select list of buffers"""
        prev_dist_str = self.cb0.GetValue().strip('[]').split(',')
        prev_dist = [int(i) for i in prev_dist_str]
        pCdist.extend(prev_dist)
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onDone1a(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Create buffers"""
        global pC_buffer
        for key,value in bufDists.items():
            if value == pCdist: # look up key associated with value
                pC_buffer = out_fds+"\\"+key # use existing buffer

        self.mark2a.SetLabel(mark_done)
        log.write('\nBuffer distances: '+str(sorted(pCdist)))
        log.write('\nBuffer feature class: '+str(pC_buffer))
        self.cb1.Enable()
        self.ulist1.Enable()
        self.radio_bx1.Disable()
        self.radio_bx1.SetSelection(0)
        self.cb0.Disable()
        self.done_btn1a.Disable()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onHlp2(self,event):
        """Help window for input data"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3C_SetInputData.html")
        htmlViewerInstance.Show()

    def onCb1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Select Line feature class"""
        self.cb2.Clear()
        self.cb3.Clear()
        self.ulist1.DeleteAllItems()
        global pC
        pC = self.cb1.GetValue()
        global pC_fldsNamesDict
        pC_fldsNamesDict={}
        num_row=int(arcpy.GetCount_management(pC).getOutput(0))
        if num_row>0: #check it contains text field
            str_fields = [f.name for f in arcpy.ListFields(pC,"","String")] # get text fields
            str_fields.sort()
            if not str_fields:
                wx.MessageBox('The selected feature class does not contain any text fields','Error',wx.OK|wx.ICON_ERROR)
            else:
                self.cb2.Append(str_fields)
                self.mark3.SetLabel(mark_done) # change tick mark to done
                log.write('\nPoint Feature Class: '+pC)
        elif num_row==0:#check that it contains features
            wx.MessageBox('The selected feature class is empty','Error',wx.OK|wx.ICON_ERROR)

        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        self.cb2.Enable()

    def onCb2(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Select category"""
        self.ulist1.DeleteAllItems()
        for k,v in list(pC_fldsNamesDict.items()): # check if value is already in fldsNamesDict
            if v == 'pC_cat':
                del pC_fldsNamesDict[k]
        global pC_cat
        pC_cat = self.cb2.GetValue()
        # strip any endspaces from strings and replace whitespace with underscore
        with arcpy.da.UpdateCursor(pC, pC_cat) as cursor:
            for row in cursor:
                row=[i.strip() if i is not None else None for i in row]
                row=[i.replace(" ","") if i is not None else None for i in row]
                row=[i.replace("_","") if i is not None else None for i in row]
                row=[i.replace(".","") if i is not None else None for i in row]
                cursor.updateRow(row)
        # Get list of unique values in field
        pCCats= unique_values(pC,pC_cat)
        # add buffer distances to names
        global pCCatsBuffer
        pCCatsBuffer=[]
        for cat in pCCats:
            if cat is None:
                wx.MessageBox('The selected field contains missing values','Error',wx.OK|wx.ICON_ERROR)
            else:
                for buf in pCdist:
                    pCCatsBuffer.append(pC_name+'_'+cat+'_'+str(buf)+'_num')
        # Check for missing data
        self.radios = []
        myMethods = ['Positive','Negative']
        for item in range(len(pCCatsBuffer)):
            self.ulist1.InsertStringItem(item, str(pCCatsBuffer[item]))
            for rad in range(1,3):
                cat = pCCatsBuffer[item]
                met = myMethods[rad - 1]
                name_of_radio = "{cat}_{met}".format(cat=cat, met=met)
                self.ulist1.SetStringItem(item, rad, "")
                if rad==1:
                    style=wx.RB_GROUP
                else:
                    style=0
                self.radBt = wx.RadioButton(self.ulist1, wx.ID_ANY, "", style=style,name=name_of_radio)
                self.ulist1.SetItemWindow(item,rad,self.radBt,expand=True)
                self.radios.append(self.radBt)

        # add field name to dictionary
        new_entry={str(pC_cat):'pC_cat'}
        pC_fldsNamesDict.update(new_entry)

        self.mark4.SetLabel(mark_done) # change tick mark to done
        self.radio_bx.Enable()
        self.ulist1.Enable()
        self.enter_btn2.Enable()
        self.Parent.statusbar.SetStatusText('Ready')
        self.enter_btn2.Enable()
        del wait
        log.write('\nCategory field: '+str(pC_cat))

    def onRadBx(self,event):
        """Aggregation method"""
        if self.radio_bx.GetStringSelection()!="Point count":
            num_fields = [f.name for f in arcpy.ListFields(pC,"",'Double') if not f.required] #get numeric fields
            num_fields.extend([f.name for f in arcpy.ListFields(pC,"",'Integer')])
            num_fields.extend([f.name for f in arcpy.ListFields(pC,"",'Single')])
            num_fields.extend([f.name for f in arcpy.ListFields(pC,"",'SmallInteger')])
            num_fields.sort()
            if not num_fields:
                wx.MessageBox('The selected feature class contains no numeric fields','Error',wx.OK|wx.ICON_ERROR)
            else:
                self.cb3.Append(num_fields)
                self.text6.Enable()
                self.cb3.Enable()
                self.nextBtn.Disable()
                self.ulist1.Disable()
                self.enter_btn2.Disable()
                self.ulist1.DeleteAllItems()
        else:
            self.text6.Disable()
            self.cb3.Disable()
            for k,v in list(pC_fldsNamesDict.items()):
                if v == 'pC_val':
                    del pC_fldsNamesDict[k]


    def onCb3(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Select value field"""
        for k,v in list(pC_fldsNamesDict.items()):
            if v == 'pC_val':
                del pC_fldsNamesDict[k]
        pC_val = self.cb3.GetValue()
        # attach method to pCCatsBuffer
        if self.radio_bx.GetStringSelection()=="Sum of values":
            for index,o in enumerate(pCCatsBuffer):
                pCCatsBuffer[index] = o[:-4]+'_sum'
        elif self.radio_bx.GetStringSelection()=="Mean of values":
            for index,o in enumerate(pCCatsBuffer):
                pCCatsBuffer[index] = o[:-4]+'_avg'
        else:
            for index,o in enumerate(pCCatsBuffer):
                pCCatsBuffer[index] = o[:-4]+'_med'
        # populate ulist
        self.radios = []
        myMethods = ['Positive','Negative']
        for item in range(len(pCCatsBuffer)):
            self.ulist1.InsertStringItem(item, str(pCCatsBuffer[item]))
            for rad in range(1,3):
                cat = pCCatsBuffer[item]
                met = myMethods[rad - 1]
                name_of_radio = "{cat}_{met}".format(cat=cat, met=met)
                self.ulist1.SetStringItem(item, rad, "")
                if rad==1:
                    style=wx.RB_GROUP
                else:
                    style=0
                self.radBt = wx.RadioButton(self.ulist1, wx.ID_ANY, "", style=style,name=name_of_radio)
                self.ulist1.SetItemWindow(item,rad,self.radBt,expand=True)
                self.radios.append(self.radBt)
        self.ulist1.Enable()
        self.enter_btn2.Enable()
        # add field name to dictionary
        new_entry={str(pC_val):'pC_val'}
        pC_fldsNamesDict.update(new_entry)
        self.mark6.SetLabel(mark_done) # change tick mark to done
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onHlp3(self,event):
        """Help window for direction of effect"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3ABC_SetSourceSink.html")
        htmlViewerInstance.Show()

    def onEnt2(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Define Direction of Effect"""
        results_dict = {}
        for i in self.radios:
            if i.GetValue()==True:
                n = i.GetName()
                index = n.rfind('_')
                cat = n[0:index]
                met = n[index+1:]
                results_dict[cat]= met
        sourcesink.update(results_dict)

        if arcpy.Exists(out_fds+"\\buffer_"+pC_name):
            self.nextBtn.Enable()

        self.mark5.SetLabel(mark_done) # change tick mark to done
        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        self.cb1.Disable()
        self.cb2.Disable()
        self.ulist1.Disable()
        self.enter_btn2.Disable()
        self.nextBtn.Enable()
        log.write('\nAggregation method: '+self.radio_bx.GetStringSelection())
        if self.radio_bx.GetStringSelection()!="Point count":
            log.write('\nValue field: '+self.cb3.GetValue())
        log.write('\nVariable definitions: ')
        for k, v in sorted(results_dict.items()):
            log.write('\n{:<30}: {:<6}'.format(k,v))

    def onBack(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Return to panel3"""
        try:
            var_list.remove(pC_name)
            for i in self.radios: # remove from sourcesink dictionary
                if i.GetValue()==True:
                    n = i.GetName()
                    index = n.rfind('_')
                    cat = pC_name+'_'+n[0:index]
                    sourcesink.pop(cat, None)
        except:
            pass
        if arcpy.Exists(out_fds+"\\buffer_"+pC_name):
            arcpy.Delete_management(out_fds+"\\buffer_"+pC_name)
            del pCdist[:]
        self.tc0.Clear()
        self.tc1.Clear()
        self.list_bx1.Clear()
        self.cb2.Clear()
        self.ulist1.DeleteAllItems()
        self.cb3.Clear()
        self.mark1.SetLabel(mark_empty)
        self.mark2.SetLabel(mark_empty)
        self.mark2a.SetLabel(mark_empty)
        self.mark3.SetLabel(mark_empty)
        self.mark4.SetLabel(mark_empty)
        self.mark5.SetLabel(mark_empty)
        self.mark6.SetLabel(mark_empty)
        self.tc1.Disable()
        self.enter_btn1.Disable()
        self.del_btn1.Disable()
        self.done_btn1.Disable()
        self.radio_bx1.Disable()
        self.radio_bx1.SetSelection(0)
        self.cb0.Disable()
        self.done_btn1a.Disable()
        self.cb1.Disable()
        self.cb2.Disable()
        self.ulist1.Disable()
        self.nextBtn.Disable()
        self.tc0.Enable()
        self.enter_btn0.Enable()
        self.radio_bx.SetSelection(0)
        self.radio_bx.Disable()
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel3,1,wx.EXPAND)
        newsize = self.Parent.panel3.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel3.Show()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        log.write(time.strftime("\n\nPoint in Buffer - Stopped by user without completion: %A %d %b %Y %H:%M:%S", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel3C','back',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel3','start',datetime())''')
        db.commit()

    def onNext(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """"""
        fieldmappings=customFieldMap(pC,pC_fldsNamesDict) # create fieldmap
        arcpy.FeatureClassToFeatureClass_conversion(pC,out_fds,pC_name+"_temp", "", fieldmappings) # copy the feature class
        max_dist=max(pCdist) # maximum buffer distance
        arcpy.Buffer_analysis(out_fds+"\\studyArea",out_fds+"\\analysisExtent_"+pC_name,max_dist) # create analysis extent
        # Compare the spatial extent against the analysis extent
        ext_an = arcpy.Describe(out_fds+"\\analysisExtent_"+pC_name).extent
        ext_2 = arcpy.Describe(out_fds+"\\"+pC_name+"_temp").extent
        if ext_2.contains(ext_an)==False:
            wx.MessageBox('The spatial extent of the point feature class is too small or completely outside of the study area','Warning',wx.OK|wx.ICON_WARNING)
            log.write('\n+++ WARNING+++ The spatial extent of the point feature class is too small or completely outside of the study area. This may result in empty intersects. If all intersects are empty, no predictor variable will be created.\n')
        arcpy.Clip_analysis(out_fds+"\\"+pC_name+"_temp",out_fds+"\\analysisExtent_"+pC_name,out_fds+"\\"+pC_name) # Clip data to analysis extent
        arcpy.Delete_management(out_fds+"\\"+pC_name+"_temp") # delete temp data
        arcpy.Delete_management(out_fds+"\\analysisExtent_"+pC_name) # delete temp data
        arcpy.AddField_management(out_fds+"\\"+pC_name,"pC_cat_INT","SHORT") # add integer id
        valueSet = set([r[0] for r in arcpy.da.SearchCursor(out_fds+"\\"+pC_name, ["pC_cat"])])
        valueList = list(valueSet)
        valueList.sort()
        updateRows = arcpy.da.UpdateCursor(out_fds+"\\"+pC_name, ["pC_cat","pC_cat_INT"])
        for updateRow in updateRows:
            updateRow[1] = valueList.index(updateRow[0]) + 1
            updateRows.updateRow(updateRow)
        arcpy.PairwiseIntersect_analysis ([out_fds+"\\"+pC_name, pC_buffer], out_fds+"\\"+pC_name+"_Intersect", "ALL", "", "")
        intersect_var = pC_name+"_Intersect"
        # move data to sql database
        fc_to_sql(conn,out_fds+"\\"+pC_name)
        conn.execute("CREATE INDEX "+pC_name+"_idx on "+pC_name+"(pC_cat_INT);")
        fc_to_sql(conn,out_fds+"\\"+intersect_var)
        conn.execute("CREATE INDEX "+intersect_var+"_idx on "+intersect_var+"(siteID_INT, pC_cat_INT, distance);")
        db.commit()
        arcpy.Delete_management(out_fds+"\\"+intersect_var)
        # add aggregate value
        conn.execute("ALTER TABLE "+intersect_var+" ADD pC_aggval float64")
        if FieldExist(out_fds+"\\"+pC_name,"pC_val"):
            conn.execute("UPDATE "+intersect_var+" SET pC_aggval=pC_val")
        else:
            conn.execute("UPDATE "+intersect_var+" SET pC_aggval=1")
        db.commit()
        # calculate sum by site ID, category and buffer size
        if self.radio_bx.GetStringSelection()=="Point count":
            conn.execute("CREATE TABLE temp AS \
                        SELECT siteID_INT \
                             ,pC_cat_INT \
                             ,distance \
                             ,SUM(pC_aggval) AS value \
	                         ,('"+pC_name+"_'||pC_cat||'_'||cast(distance as int)||'_num') AS varName \
                        FROM "+intersect_var+" \
                        GROUP BY siteID_INT, pC_cat_INT, distance")
        elif self.radio_bx.GetStringSelection()=="Sum of values":
            conn.execute("CREATE TABLE temp AS \
                        SELECT siteID_INT \
                             ,pC_cat_INT \
                             ,distance \
                             ,SUM(pC_aggval) AS value \
	                         ,('"+pC_name+"_'||pC_cat||'_'||cast(distance as int)||'_sum') AS varName \
                        FROM "+intersect_var+" \
                        GROUP BY siteID_INT, pC_cat_INT, distance")
        elif self.radio_bx.GetStringSelection()=="Mean of values":
            conn.execute("CREATE TABLE temp AS \
                        SELECT siteID_INT \
                             ,pC_cat_INT \
                             ,distance \
                             ,AVG(pC_aggval) AS value \
	                         ,('"+pC_name+"_'||pC_cat||'_'||cast(distance as int)||'_avg') AS varName \
                        FROM "+intersect_var+" \
                        GROUP BY siteID_INT, pC_cat_INT, distance")
        elif self.radio_bx.GetStringSelection()=="Median of values": #loads of code to calculate median
            conn.executescript("CREATE TABLE pC_median_temp1 AS \
                SELECT siteID_INT \
                    ,pC_cat_INT \
                    ,distance \
                    ,pC_aggval \
                    ,('"+pC_name+"_'||pC_cat||'_'||cast(distance as int)||'_med') AS varName \
                FROM "+intersect_var+" \
                ORDER BY siteID_INT, pC_cat_INT, distance, pC_aggval; \
                ALTER TABLE pC_median_temp1 ADD row_num int; \
                UPDATE pC_median_temp1 SET row_num=rowid; \
                CREATE TABLE pC_median_temp2 AS \
                SELECT  siteID_INT \
                   ,pC_cat_INT \
                   ,distance \
                   ,(MIN(row_num*1.0)+MAX(row_num*1.0))/2 AS midrow \
                   ,CASE WHEN COUNT(pC_aggval)%2=0 THEN 'even' ELSE 'odd' END AS type \
                FROM pC_median_temp1 \
                GROUP BY siteID_INT, pC_cat_INT, distance; \
                CREATE TABLE pC_median_temp3 AS \
                SELECT siteID_INT \
                    ,pC_cat_INT \
                    ,distance \
                    ,midrow \
                FROM pC_median_temp2 \
                WHERE type='even' \
                UNION ALL \
                SELECT siteID_INT \
                    ,pC_cat_INT \
                    ,distance \
                    ,midrow \
                FROM pC_median_temp2 \
                WHERE type='even'; \
                CREATE TABLE pC_median_temp4 AS \
                SELECT * \
                FROM pC_median_temp3 \
                ORDER BY midrow; \
                ALTER TABLE pC_median_temp4 ADD row_num int; \
                UPDATE pC_median_temp4 SET row_num=rowid; \
                ALTER TABLE pC_median_temp4 ADD midrow_new float; \
                UPDATE pC_median_temp4 SET midrow_new= CASE WHEN (row_num)%2=1 THEN midrow-0.5 WHEN (row_num)%2=0 THEN midrow+0.5 END; \
                CREATE TABLE pC_median_temp5 AS \
                SELECT siteID_INT \
                    ,pC_cat_INT \
                    ,distance \
                    ,midrow \
                FROM pC_median_temp2 \
                WHERE type = 'odd' \
                UNION ALL \
                SELECT siteID_INT \
                	,pC_cat_INT \
                    ,distance \
                    ,midrow_new as midrow \
                FROM pC_median_temp4; \
                CREATE TABLE pC_median_temp6 AS \
                SELECT a.siteID_INT \
                    ,a.pC_cat_INT \
                    ,a.distance \
                    ,a.midrow \
                    ,b.row_num \
                    ,b.pC_aggval \
                    ,b.varName \
                FROM pC_median_temp5 as a \
                JOIN pC_median_temp1 as b \
                ON a.midrow = b.row_num; \
                CREATE TABLE temp AS \
                SELECT siteID_INT \
                    ,pC_cat_INT \
                    ,distance \
                    ,AVG(pC_aggval) AS value \
                    ,MIN(varName) AS varName \
                FROM pC_median_temp6 \
                GROUP BY siteID_INT	,pC_cat_INT ,distance; \
                DROP TABLE pC_median_temp1; \
                DROP TABLE pC_median_temp2; \
                DROP TABLE pC_median_temp3; \
                DROP TABLE pC_median_temp4; \
                DROP TABLE pC_median_temp5; \
                DROP TABLE pC_median_temp6;")

        vars_sql_temp=list(conn.execute("SELECT DISTINCT varName FROM temp")) # make a list of variable names
        # add a column for each variable name
        for var in vars_sql_temp:
            conn.execute('ALTER TABLE dat4stats ADD {} float64'.format(var[0])) # if I add second query I get an operational error
        db.commit()
        # add values to each column
        for var in vars_sql_temp:
            qry="UPDATE dat4stats \
                 SET "+var[0]+" = (\
                 SELECT value \
                 FROM temp \
                 WHERE siteID_INT = dat4stats.siteID_INT and varName='"+var[0]+"')"
            conn.execute(qry)
        db.commit()
        # replace missing with zeros
        for var in vars_sql_temp:
            qry="UPDATE dat4stats \
                SET "+var[0]+" = 0 \
                WHERE "+var[0]+" IS NULL"
            conn.execute(qry)
        db.commit()
        conn.execute("DROP TABLE temp")
        conn.execute("DROP TABLE "+intersect_var)
        db.commit()
        # predictor names to panel 3
        WizardPanel3.list_ctrl.DeleteAllItems()
        cursor = conn.execute("SELECT * FROM dat4stats")
        varnames = list([x[0] for x in cursor.description])
        prednames = [name for name in varnames if (name[0] =='p')]
        index=0
        for i in prednames:
            WizardPanel3.list_ctrl.InsertItem(index,i)
            WizardPanel3.list_ctrl.SetItem(index,1,sourcesink[i])
            if index % 2:
                WizardPanel3.list_ctrl.SetItemBackgroundColour(index, col="white")
            else:
                WizardPanel3.list_ctrl.SetItemBackgroundColour(index, col=(222,234,246))
            index+=1
        WizardPanel3.list_ctrl.SetColumnWidth(0,wx.LIST_AUTOSIZE)
        WizardPanel3.list_ctrl.SetColumnWidth(1,wx.LIST_AUTOSIZE)
        log.write(time.strftime("\nPoint in Buffer - End Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel3C','stop',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel3','start',datetime())''')
        db.commit()
        # clear p3C
        del pCdist[:]
        self.tc0.Clear()
        self.tc1.Clear()
        self.list_bx1.Clear()
        self.cb2.Clear()
        self.ulist1.DeleteAllItems()
        self.cb3.Clear()
        self.mark1.SetLabel(mark_empty)
        self.mark2.SetLabel(mark_empty)
        self.mark2a.SetLabel(mark_empty)
        self.mark3.SetLabel(mark_empty)
        self.mark4.SetLabel(mark_empty)
        self.mark5.SetLabel(mark_empty)
        self.mark6.SetLabel(mark_empty)
        self.tc1.Disable()
        self.enter_btn1.Disable()
        self.del_btn1.Disable()
        self.done_btn1.Disable()
        self.radio_bx1.Disable()
        self.radio_bx1.SetSelection(0)
        self.cb0.Disable()
        self.done_btn1a.Disable()
        self.cb1.Disable()
        self.cb2.Disable()
        self.ulist1.Disable()
        self.nextBtn.Disable()
        self.tc0.Enable()
        self.enter_btn0.Enable()
        self.radio_bx.SetSelection(0)
        self.radio_bx.Disable()
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel3,1,wx.EXPAND)
        newsize = self.Parent.panel3.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel3.Show()

        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onCanc1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Cancel Wizard"""
        dlg = wx.MessageDialog(None, "Cancelling the wizard will delete all progress. Do you wish to cancel the wizard?",'Cancel',wx.YES_NO | wx.ICON_WARNING)
        result = dlg.ShowModal()
        if result == wx.ID_YES:
            arcpy.AddMessage((time.strftime("Cancelled by user: %A %d %b %Y %H:%M:%S", time.localtime())))
            try:
                log.close()
                logging.shutdown()
                db.close()
                shutil.rmtree(out_path)
            except NameError as e:
                arcpy.AddMessage(e)
            except OSError as e:
                arcpy.AddMessage(e)
            del wait
            self.Parent.Close()
        else:
            self.Parent.statusbar.SetStatusText('Ready')
            del wait
#-------------------------------------------------------------------------------
class WizardPanel3D(wx.Panel):
    """Page 3D"""

    def __init__(self, parent, title=None):
        """Constructor"""
        wx.Panel.__init__(self, parent)
        self.locale = wx.Locale(wx.LANGUAGE_ENGLISH)
        wx.ToolTip.Enable(True)

        self.sizer = wx.GridBagSizer(0,0)

        title2 = wx.StaticText(self, -1, "Distance to and/or value of nearest Polygon")
        title2.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(title2, pos=(0,0), span=(1,5),flag=wx.TOP|wx.LEFT|wx.BOTTOM, border=10)

        self.sizer.Add(wx.StaticLine(self), pos=(1, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM, border=5)

        header0 = wx.StaticText(self,-1,label="Set Variable Name")
        header0.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header0, pos=(2, 0), flag=wx.ALL, border=10)

        help_btn0 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn0.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn0.SetForegroundColour(wx.Colour(0,0,255))
        help_btn0.Bind(wx.EVT_BUTTON, self.onHlp0)
        self.sizer.Add(help_btn0, pos=(2,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text1 = wx.StaticText(self,-1,label="Variable Name")
        self.sizer.Add(text1, pos=(3, 0), flag=wx.ALL, border=10)

        self.tc0 = wx.TextCtrl(self, -1)
        self.tc0.SetMaxLength(20)
        self.tc0.SetToolTip(wx.ToolTip('Enter a name for the predictor variable'))
        self.tc0.Bind(wx.EVT_CHAR,self.onChar)
        self.sizer.Add(self.tc0, pos=(3, 1), span=(1,2), flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=5)

        self.enter_btn0 = wx.Button(self, label="Enter")
        self.enter_btn0.Bind(wx.EVT_BUTTON, self.onEnt0)
        self.sizer.Add(self.enter_btn0, pos=(3,3), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)

        self.mark1 = wx.StaticText(self,-1,label=mark_empty)
        self.mark1.SetForegroundColour((0,255,0))
        self.mark1.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark1, pos=(3,4), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(4, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        header1 = wx.StaticText(self,-1,label="Set Method")
        header1.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header1, pos=(5, 0), flag=wx.ALL, border=10)

        help_btn1 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn1.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn1.SetForegroundColour(wx.Colour(0,0,255))
        help_btn1.Bind(wx.EVT_BUTTON, self.onHlp1)
        self.sizer.Add(help_btn1, pos=(5,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text2 = wx.StaticText(self,-1,label="Data to be extracted")
        self.sizer.Add(text2, pos=(6, 0), flag=wx.ALL, border=10)

        self.chklbx1 = wx.CheckListBox(self,-1,choices=sorted(extrmet.keys()))
        self.chklbx1.SetToolTip(wx.ToolTip('Select one or more pieces of information to be extracted'))
        self.sizer.Add(self.chklbx1, pos=(6,1), span=(1,3), flag=wx.TOP|wx.LEFT|wx.BOTTOM|wx.EXPAND, border=5)
        self.chklbx1.Disable()

        self.sel_btn1 = wx.Button(self, label="Select")
        self.sel_btn1.Bind(wx.EVT_BUTTON, self.onSel1)
        self.sizer.Add(self.sel_btn1, pos=(6,4), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)
        self.sel_btn1.Disable()

        self.mark2 = wx.StaticText(self,-1,label=mark_empty)
        self.mark2.SetForegroundColour((0,255,0))
        self.mark2.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark2, pos=(6,5), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(7, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        header2 = wx.StaticText(self,-1,label="Set Input Data")
        header2.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header2, pos=(8, 0), flag=wx.ALL, border=10)

        help_btn2 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn2.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn2.SetForegroundColour(wx.Colour(0,0,255))
        help_btn2.Bind(wx.EVT_BUTTON, self.onHlp2)
        self.sizer.Add(help_btn2, pos=(8,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text3 = wx.StaticText(self,-1,label="Polygon Feature Class")
        self.sizer.Add(text3, pos=(9, 0), flag=wx.ALL, border=10)

        self.cb1 = wx.ComboBox(self,choices=[],style=wx.CB_DROPDOWN)
        self.cb1.SetToolTip(wx.ToolTip('From the dropdown list select the feature class containing the polygon data'))
        self.sizer.Add(self.cb1, pos=(9, 1), span=(1,2),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.cb1.Bind(wx.EVT_COMBOBOX, self.onCb1)
        self.cb1.Disable()

        self.mark3 = wx.StaticText(self,-1,label=mark_empty)
        self.mark3.SetForegroundColour((0,255,0))
        self.mark3.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark3, pos=(9,3), flag=wx.ALL, border=5)

        text4 = wx.StaticText(self,-1,label="Value Field(s)")
        self.sizer.Add(text4, pos=(10, 0), flag=wx.ALL, border=10)

        self.chklbx2 = wx.CheckListBox(self,choices=[])
        self.chklbx2.SetToolTip(wx.ToolTip('Tick all fields that contain values to be extracted'))
        self.sizer.Add(self.chklbx2, pos=(10, 1), span=(4,3),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.chklbx2.Disable()

        self.sel_btn2 = wx.Button(self, label="Select")
        self.sel_btn2.Bind(wx.EVT_BUTTON, self.onSel2)
        self.sizer.Add(self.sel_btn2, pos=(10,4), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)
        self.sel_btn2.Disable()

        self.mark4 = wx.StaticText(self,-1,label=mark_empty)
        self.mark4.SetForegroundColour((0,255,0))
        self.mark4.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark4, pos=(10,5), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(14, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        header3 = wx.StaticText(self,-1,label="Set Direction of Effect")
        header3.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header3, pos=(15, 0), flag=wx.ALL, border=10)

        help_btn3 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn3.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn3.SetForegroundColour(wx.Colour(0,0,255))
        help_btn3.Bind(wx.EVT_BUTTON, self.onHlp3)
        self.sizer.Add(help_btn3, pos=(15,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text5 = wx.StaticText(self,-1,label="Define Direction of Effect")
        self.sizer.Add(text5, pos=(16, 0), flag=wx.ALL, border=10)

        self.ulist1 = ULC.UltimateListCtrl(self, wx.ID_ANY, agwStyle=ULC.ULC_HAS_VARIABLE_ROW_HEIGHT | wx.LC_REPORT | wx.LC_VRULES | wx.LC_HRULES)
        self.ulist1.InsertColumn(col=0, heading="Variable Name",format=0)
        self.ulist1.InsertColumn(col=1, heading="Positive",format=0)
        self.ulist1.InsertColumn(col=2, heading="Negative",format=0)
        self.ulist1.SetColumnWidth(0,ULC.ULC_AUTOSIZE_FILL)
        self.ulist1.SetColumnWidth(1,ULC.ULC_AUTOSIZE_USEHEADER)
        self.ulist1.SetColumnWidth(2,ULC.ULC_AUTOSIZE_USEHEADER)
        self.sizer.Add(self.ulist1,pos=(16,1),span=(5,4),flag=wx.TOP|wx.BOTTOM|wx.EXPAND|wx.RIGHT, border=5)
        self.ulist1.Disable()

        self.enter_btn2 = wx.Button(self, label="Done")
        self.enter_btn2.Bind(wx.EVT_BUTTON, self.onEnt2)
        self.sizer.Add(self.enter_btn2, pos=(16,5), flag=wx.TOP|wx.BOTTOM|wx.RIGHT|wx.LEFT, border=5)
        self.enter_btn2.Disable()

        self.mark5 = wx.StaticText(self,-1,label=mark_empty)
        self.mark5.SetForegroundColour((0,255,0))
        self.mark5.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark5, pos=(16,6), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(21, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        # add previous Button
        self.backBtn = wx.Button(self, label="< Back")
        self.backBtn.Bind(wx.EVT_BUTTON, self.onBack)
        self.sizer.Add(self.backBtn, pos=(22,4),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        # add next button
        self.nextBtn = wx.Button(self, label="Next >")
        self.nextBtn.Bind(wx.EVT_BUTTON, self.onNext)
        self.sizer.Add(self.nextBtn, pos=(22,5),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)
        self.nextBtn.Disable()

        # add cancel button
        self.cancBtn1 = wx.Button(self, label="Cancel")
        self.cancBtn1.Bind(wx.EVT_BUTTON, self.onCanc1)
        self.cancBtn1.SetToolTip(wx.ToolTip('Cancel'))
        self.sizer.Add(self.cancBtn1, pos=(22,6),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        self.SetSizer(self.sizer)
        self.sizer.AddGrowableRow(10)
        self.sizer.AddGrowableRow(16)
        self.sizer.AddGrowableCol(1)
        self.GetParent().SendSizeEvent()
        self.Layout()
        self.Fit()

    def onHlp0(self,event):
        """Help window for variable name"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3D_SetVariableName.html")
        htmlViewerInstance.Show()

    def onChar(self,event):
        """textbox for variable name"""
        keycode = event.GetKeyCode()
        if keycode in ascii_char: #only allow chars in alphabet
            event.Skip()

    def onEnt0(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """enter variable name"""
        inp_var = self.tc0.GetValue()
        if not inp_var:
            wx.MessageBox('Please enter a variable name','Error',wx.OK|wx.ICON_ERROR)
        else: # change this to be based on the database, so that entry can be changed
            global pD_name
            pD_name = "pD_"+inp_var
            if pD_name in var_list:
                wx.MessageBox('The variable already exists','Error',wx.OK|wx.ICON_ERROR)
            else:
                var_list.append(pD_name)
                self.mark1.SetLabel(mark_done) # change tick mark to done
                log.write('\nVariable name: '+inp_var)
                self.chklbx1.Enable()
                self.sel_btn1.Enable()
                self.tc0.Disable()
                self.enter_btn0.Disable()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onHlp1(self,event):
        """Help window value/distance methods"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3D_SetMethod.html")
        htmlViewerInstance.Show()

    def onSel1(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """select distance/value method for extraction"""
        met = self.chklbx1.GetCheckedStrings()
        if not met:
            wx.MessageBox('Select at least one method','Error',wx.OK|wx.ICON_ERROR)
        global metcodenone # list to hold distance only method codes selected
        metcodenone=[]
        global metcodeval # list to hold value*distance method codes selected
        metcodeval=[]
        for field in met:
            newmethod=extrmet[field]
            if newmethod=='dist' or newmethod=='invd' or newmethod=='invsq':
                metcodenone.append(newmethod)
            else:
                metcodeval.append(newmethod)

        self.mark2.SetLabel(mark_done) # change tick mark to done
        self.cb1.Enable()
        if metcodeval:
            self.chklbx2.Enable()
            self.sel_btn2.Enable()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        self.chklbx1.Disable()
        self.sel_btn1.Disable()
        log.write('\nMethod(s) selected: '+str(met))

    def onHlp2(self,event):
        """Help window for input data"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3D_SetInputData.html")
        htmlViewerInstance.Show()

    def onCb1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Select Polygon feature class"""
        self.chklbx2.Clear()
        global pD
        pD = self.cb1.GetValue()
        num_row=int(arcpy.GetCount_management(pD).getOutput(0))
        if num_row==0:#check that it contains features
            wx.MessageBox('The selected feature class is empty','Error',wx.OK|wx.ICON_ERROR)

        if metcodeval and num_row>0: #if value is wanted add value fields
            num_fields = [f.name for f in arcpy.ListFields(pD,"",'Double') if not f.required] #get numeric fields
            num_fields.extend([f.name for f in arcpy.ListFields(pD,"",'Integer')])
            num_fields.extend([f.name for f in arcpy.ListFields(pD,"",'Single')])
            num_fields.extend([f.name for f in arcpy.ListFields(pD,"",'SmallInteger')])
            num_fields.sort()
            if not num_fields:
                wx.MessageBox('The selected feature class contains no numeric fields','Error',wx.OK|wx.ICON_ERROR)
            else:
                self.chklbx2.Append(num_fields)
                self.mark3.SetLabel(mark_done) # change tick mark to done
                log.write('\nPolygon Feature Class: '+pD)
        elif num_row>0: # if value fields are not required just copy the featureclass
            arcpy.FeatureClassToFeatureClass_conversion(pD,out_fds,pD_name, "", "") # copy the feature class into the feature dataset
            pDCats_display=[] #make a display list of the variables that will be created
            pDCats=[] # make a reference list
            for item in metcodenone:
                met = [key for (key, value) in extrmet.items() if value == item] # get key from value
                pDCats_display.append(met[0])
                pDCats.append(pD_name+'_none_'+item)
            #populate ulist
            self.ulist1.DeleteAllItems()
            self.radios = []
            myMethods = ['Positive','Negative']
            for item in range(len(pDCats)):
                self.ulist1.InsertStringItem(item, str(pDCats_display[item]))
                for rad in range(1,3):
                    cat = pDCats[item]
                    met = myMethods[rad - 1]
                    name_of_radio = "{cat}_{met}".format(cat=cat, met=met)
                    self.ulist1.SetStringItem(item, rad, "")
                    if rad==1:
                        style=wx.RB_GROUP
                    else:
                        style=0
                    self.radBt = wx.RadioButton(self.ulist1, wx.ID_ANY, "", style=style,name=name_of_radio)
                    self.ulist1.SetItemWindow(item,rad,self.radBt,expand=True)
                    self.radios.append(self.radBt)

            self.ulist1.Enable()
            self.enter_btn2.Enable()
            self.mark3.SetLabel(mark_done)

        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onSel2(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """select value fields"""
        global pD_cat
        pD_cat = list(self.chklbx2.GetCheckedStrings())
        pD_fldsNamesDict={}
        if not pD_cat:
            wx.MessageBox('Please select one or more fields','Error',wx.OK|wx.ICON_ERROR)
        else:
            for field in pD_cat:
                newfield = field.strip().replace(" ","").replace("_","").replace(".","")
                new_entry={str(field) : str(newfield)}
                pD_fldsNamesDict.update(new_entry)
            fieldmappings=customFieldMap(pD,pD_fldsNamesDict) # create fieldmap
            arcpy.FeatureClassToFeatureClass_conversion(pD,out_fds,pD_name, "", fieldmappings) # copy the feature class into the feature dataset with the value fields

        pDCats_display=[] #make a display list of the variables that will be created
        pDCats=[] # make a reference list
        if metcodenone:
            for item in metcodenone:
                met = [key for (key, value) in extrmet.items() if value == item] # get key from value
                pDCats_display.append(met[0])
                pDCats.append(pD_name+'_none_'+item)
        if metcodeval:
            for val in pD_cat:
                for item in metcodeval:
                    val = val.strip().replace(" ","").replace("_","").replace(".","")
                    met = [key for (key, value) in extrmet.items() if value == item] # get key from value
                    index = met[0].find('*')
                    if index<0: # index is -1 if 'value' has been selected
                        pDCats_display.append(val)
                        pDCats.append(pD_name+'_'+val+'_'+item)
                    else:
                        met2 = met[0][index+2:]
                        pDCats_display.append(val+' * '+str(met2))
                        pDCats.append(pD_name+'_'+val+'_'+item)

        #populate ulist
        self.ulist1.DeleteAllItems()
        self.radios = []
        myMethods = ['Positive','Negative']
        for item in range(len(pDCats)):
            self.ulist1.InsertStringItem(item, str(pDCats_display[item]))
            for rad in range(1,3):
                cat = pDCats[item]
                met = myMethods[rad - 1]
                name_of_radio = "{cat}_{met}".format(cat=cat, met=met)
                self.ulist1.SetStringItem(item, rad, "")
                if rad==1:
                    style=wx.RB_GROUP
                else:
                    style=0
                self.radBt = wx.RadioButton(self.ulist1, wx.ID_ANY, "", style=style,name=name_of_radio)
                self.ulist1.SetItemWindow(item,rad,self.radBt,expand=True)
                self.radios.append(self.radBt)

        self.ulist1.Enable()
        self.enter_btn2.Enable()
        self.mark4.SetLabel(mark_done) # change tick mark to done
        self.chklbx2.Disable()
        self.Parent.statusbar.SetStatusText('Ready')
        self.cb1.Disable()
        self.chklbx2.Disable()
        self.sel_btn2.Disable()
        del wait
        log.write('\nValue field(s): '+str(pD_cat))

    def onHlp3(self,event):
        """Help window for direction of effect"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3D_SetSourceSink.html")
        htmlViewerInstance.Show()

    def onEnt2(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Define Direction of Effect"""
        results_dict = {}
        for i in self.radios:
            if i.GetValue()==True:
                n = i.GetName()
                index = n.rfind('_')
                cat = n[0:index]
                met = n[index+1:]
                results_dict[cat]= met
        sourcesink.update(results_dict)

        if arcpy.Exists(out_fds+"\\"+pD_name):
            self.nextBtn.Enable()

        self.mark5.SetLabel(mark_done) # change tick mark to done
        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        log.write('\nVariable definitions: ')
        for k, v in sorted(results_dict.items()):
            log.write('\n{:<30}: {:<6}'.format(k,v))
        self.ulist1.Disable()
        self.enter_btn2.Disable()

    def onBack(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Return to panel3"""
        try:
            var_list.remove(pD_name)
            for i in self.radios: # remove from sourcesink dictionary
                if i.GetValue()==True:
                    n = i.GetName()
                    index = n.rfind('_')
                    cat = n[0:index]
                    sourcesink.pop(cat, None)
        except:
            pass
        if arcpy.Exists(out_fds+"\\"+pD_name):
            arcpy.Delete_management(out_fds+"\\"+pD_name)
        self.tc0.Clear()
        for cb in self.chklbx1.GetCheckedItems():
            self.chklbx1.Check(cb,False)
        self.chklbx2.Clear()
        self.ulist1.DeleteAllItems()
        self.mark1.SetLabel(mark_empty)
        self.mark2.SetLabel(mark_empty)
        self.mark3.SetLabel(mark_empty)
        self.mark4.SetLabel(mark_empty)
        self.mark5.SetLabel(mark_empty)
        self.chklbx1.Disable()
        self.cb1.Disable()
        self.chklbx2.Disable()
        self.nextBtn.Disable()
        self.tc0.Enable()
        self.enter_btn0.Enable()
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel3,1,wx.EXPAND)
        newsize = self.Parent.panel3.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel3.Show()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        log.write(time.strftime("\n\nPolygon Distance - Stopped by user without completion: %A %d %b %Y %H:%M:%S", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel3D','back',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel3','start',datetime())''')
        db.commit()

    def onNext(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """"""
        arcpy.SpatialJoin_analysis(out_fds+"\\sites", out_fds+"\\"+pD_name, out_fds+"\\"+pD_name+"_join", "JOIN_ONE_TO_ONE", "KEEP_ALL","","CLOSEST","","distance") #spatial join of polygon to monitoring sites
        if 'invsq' in metcodenone or 'valinvsq' in metcodeval:
            arcpy.AddField_management(out_fds+"\\"+pD_name+"_join","distsqu","DOUBLE")
            cursor=arcpy.da.UpdateCursor(out_fds+"\\"+pD_name+"_join", ["distance","distsqu"])
            for row in cursor:
                row[1]=row[0]**2
                cursor.updateRow(row)
        # move data to sql database
        fc_to_sql(conn,out_fds+"\\"+pD_name+"_join")
        db.commit()
        arcpy.Delete_management(out_fds+"\\"+pD_name+"_join")
        # calcaulate Variables
        vars_sql_temp=list()
        if 'dist' in metcodenone:
            conn.execute("ALTER TABLE "+pD_name+"_join ADD "+pD_name+"_none_dist float64")
            conn.execute("UPDATE "+pD_name+"_join SET "+pD_name+"_none_dist=distance")
            vars_sql_temp.append(pD_name+"_none_dist")

        if 'invd' in metcodenone:
            conn.execute("ALTER TABLE "+pD_name+"_join ADD "+pD_name+"_none_invd float64")
            conn.execute("UPDATE "+pD_name+"_join SET "+pD_name+"_none_invd=1/distance")
            vars_sql_temp.append(pD_name+"_none_invd")

        if 'invsq' in metcodenone:
            conn.execute("ALTER TABLE "+pD_name+"_join ADD "+pD_name+"_none_invsq float64")
            conn.execute("UPDATE "+pD_name+"_join SET "+pD_name+"_none_invsq=1/distsqu")
            vars_sql_temp.append(pD_name+"_none_invsq")

        if 'val' in metcodeval:
            for i in pD_cat:
                i = i.strip().replace(" ","").replace("_","").replace(".","")
                conn.execute("ALTER TABLE "+pD_name+"_join ADD "+pD_name+"_"+i+"_val float64")
                conn.execute("UPDATE "+pD_name+"_join SET "+pD_name+"_"+i+"_val="+i)
                vars_sql_temp.append(pD_name+"_"+i+"_val")

        if 'valdist' in metcodeval:
            for i in pD_cat:
                i = i.strip().replace(" ","").replace("_","").replace(".","")
                conn.execute("ALTER TABLE "+pD_name+"_join ADD "+pD_name+"_"+i+"_valdist float64")
                conn.execute("UPDATE "+pD_name+"_join SET "+pD_name+"_"+i+"_valdist="+i+"*distance")
                vars_sql_temp.append(pD_name+"_"+i+"_valdist")

        if 'valinvd' in metcodeval:
            for i in pD_cat:
                i = i.strip().replace(" ","").replace("_","").replace(".","")
                conn.execute("ALTER TABLE "+pD_name+"_join ADD "+pD_name+"_"+i+"_valinvd float64")
                conn.execute("UPDATE "+pD_name+"_join SET "+pD_name+"_"+i+"_valinvd="+i+"*1/distance")
                vars_sql_temp.append(pD_name+"_"+i+"_valinvd")

        if 'valinvsq' in metcodeval:
            for i in pD_cat:
                i = i.strip().replace(" ","").replace("_","").replace(".","")
                conn.execute("ALTER TABLE "+pD_name+"_join ADD "+pD_name+"_"+i+"_valinvsq float64")
                conn.execute("UPDATE "+pD_name+"_join SET "+pD_name+"_"+i+"_valinvsq="+i+"*1/distsqu")
                vars_sql_temp.append(pD_name+"_"+i+"_valinvsq")

        conn.execute("CREATE UNIQUE INDEX "+pD_name+"_idx on "+pD_name+"_join (siteID_INT);")
        db.commit()
        # add to dat4status
        for var in vars_sql_temp:
            conn.execute('ALTER TABLE dat4stats ADD {} float64'.format(var)) # if I add second query I get an operational error
        db.commit()

        for var in vars_sql_temp:
            qry="UPDATE dat4stats \
                 SET "+var+" = ( \
                 SELECT "+var+" \
                 FROM  "+pD_name+"_join \
                 WHERE siteID_INT = dat4stats.siteID_INT)"
            conn.execute(qry)
        db.commit()
        conn.execute("DROP TABLE "+pD_name+"_join")
        db.commit()
        # push predictor names to panel 3
        WizardPanel3.list_ctrl.DeleteAllItems()
        cursor = conn.execute("SELECT * FROM dat4stats")
        varnames = list([x[0] for x in cursor.description])
        prednames = [name for name in varnames if (name[0] =='p')]
        index=0
        for i in prednames:
            WizardPanel3.list_ctrl.InsertItem(index,i)
            WizardPanel3.list_ctrl.SetItem(index,1,sourcesink[i])
            if index % 2:
                WizardPanel3.list_ctrl.SetItemBackgroundColour(index, col="white")
            else:
                WizardPanel3.list_ctrl.SetItemBackgroundColour(index, col=(222,234,246))
            index+=1
            WizardPanel3.list_ctrl.SetColumnWidth(0,wx.LIST_AUTOSIZE)
            WizardPanel3.list_ctrl.SetColumnWidth(1,wx.LIST_AUTOSIZE)
        log.write(time.strftime("\nPolygon Distance - End Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel3D','stop',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel3','start',datetime())''')
        db.commit()
        # clear p3D
        self.tc0.Clear()
        for cb in self.chklbx1.GetCheckedItems():
            self.chklbx1.Check(cb,False)
        self.chklbx2.Clear()
        self.ulist1.DeleteAllItems()
        self.mark1.SetLabel(mark_empty)
        self.mark2.SetLabel(mark_empty)
        self.mark3.SetLabel(mark_empty)
        self.mark4.SetLabel(mark_empty)
        self.mark5.SetLabel(mark_empty)
        self.chklbx1.Disable()
        self.cb1.Disable()
        self.chklbx2.Disable()
        self.nextBtn.Disable()
        self.tc0.Enable()
        self.enter_btn0.Enable()
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel3,1,wx.EXPAND)
        newsize = self.Parent.panel3.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel3.Show()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onCanc1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Cancel Wizard"""
        dlg = wx.MessageDialog(None, "Cancelling the wizard will delete all progress. Do you wish to cancel the wizard?",'Cancel',wx.YES_NO | wx.ICON_WARNING)
        result = dlg.ShowModal()
        if result == wx.ID_YES:
            arcpy.AddMessage((time.strftime("Cancelled by user: %A %d %b %Y %H:%M:%S", time.localtime())))
            try:
                log.close()
                logging.shutdown()
                db.close()
                shutil.rmtree(out_path)
            except NameError as e:
                arcpy.AddMessage(e)
            except OSError as e:
                arcpy.AddMessage(e)
            del wait
            self.Parent.Close()
        else:
            self.Parent.statusbar.SetStatusText('Ready')
            del wait
#-------------------------------------------------------------------------------
class WizardPanel3E(wx.Panel):
    """Page 3E"""

    def __init__(self, parent, title=None):
        """Constructor"""
        wx.Panel.__init__(self, parent)
        self.locale = wx.Locale(wx.LANGUAGE_ENGLISH)
        wx.ToolTip.Enable(True)

        self.sizer = wx.GridBagSizer(0,0)

        title2 = wx.StaticText(self, -1, "Distance to and/or value of nearest Line")
        title2.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(title2, pos=(0,0), span=(1,5),flag=wx.TOP|wx.LEFT|wx.BOTTOM, border=10)

        self.sizer.Add(wx.StaticLine(self), pos=(1, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM, border=5)

        header0 = wx.StaticText(self,-1,label="Set Variable Name")
        header0.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header0, pos=(2, 0), flag=wx.ALL, border=10)

        help_btn0 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn0.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn0.SetForegroundColour(wx.Colour(0,0,255))
        help_btn0.Bind(wx.EVT_BUTTON, self.onHlp0)
        self.sizer.Add(help_btn0, pos=(2,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text1 = wx.StaticText(self,-1,label="Variable Name")
        self.sizer.Add(text1, pos=(3, 0), flag=wx.ALL, border=10)

        self.tc0 = wx.TextCtrl(self, -1)
        self.tc0.SetMaxLength(20)
        self.tc0.SetToolTip(wx.ToolTip('Enter a name for the predictor variable'))
        self.tc0.Bind(wx.EVT_CHAR,self.onChar)
        self.sizer.Add(self.tc0, pos=(3, 1), span=(1,2), flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=5)

        self.enter_btn0 = wx.Button(self, label="Enter")
        self.enter_btn0.Bind(wx.EVT_BUTTON, self.onEnt0)
        self.sizer.Add(self.enter_btn0, pos=(3,3), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)

        self.mark1 = wx.StaticText(self,-1,label=mark_empty)
        self.mark1.SetForegroundColour((0,255,0))
        self.mark1.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark1, pos=(3,4), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(4, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        header1 = wx.StaticText(self,-1,label="Set Method")
        header1.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header1, pos=(5, 0), flag=wx.ALL, border=10)

        help_btn1 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn1.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn1.SetForegroundColour(wx.Colour(0,0,255))
        help_btn1.Bind(wx.EVT_BUTTON, self.onHlp1)
        self.sizer.Add(help_btn1, pos=(5,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text2 = wx.StaticText(self,-1,label="Data to be extracted")
        self.sizer.Add(text2, pos=(6, 0), flag=wx.ALL, border=10)

        self.chklbx1 = wx.CheckListBox(self,-1,choices=sorted(extrmet.keys()))
        self.chklbx1.SetToolTip(wx.ToolTip('Select one or more pieces of information to be extracted'))
        self.sizer.Add(self.chklbx1, pos=(6,1), span=(1,3), flag=wx.TOP|wx.LEFT|wx.BOTTOM|wx.EXPAND, border=5)
        self.chklbx1.Disable()

        self.sel_btn1 = wx.Button(self, label="Select")
        self.sel_btn1.Bind(wx.EVT_BUTTON, self.onSel1)
        self.sizer.Add(self.sel_btn1, pos=(6,4), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)
        self.sel_btn1.Disable()

        self.mark2 = wx.StaticText(self,-1,label=mark_empty)
        self.mark2.SetForegroundColour((0,255,0))
        self.mark2.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark2, pos=(6,5), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(7, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        header2 = wx.StaticText(self,-1,label="Set Input Data")
        header2.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header2, pos=(8, 0), flag=wx.ALL, border=10)

        help_btn2 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn2.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn2.SetForegroundColour(wx.Colour(0,0,255))
        help_btn2.Bind(wx.EVT_BUTTON, self.onHlp2)
        self.sizer.Add(help_btn2, pos=(8,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text3 = wx.StaticText(self,-1,label="Line Feature Class")
        self.sizer.Add(text3, pos=(9, 0), flag=wx.ALL, border=10)

        self.cb1 = wx.ComboBox(self,choices=[],style=wx.CB_DROPDOWN)
        self.cb1.SetToolTip(wx.ToolTip('From the dropdown list select the feature class containing the line data'))
        self.sizer.Add(self.cb1, pos=(9, 1), span=(1,2),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.cb1.Bind(wx.EVT_COMBOBOX, self.onCb1)
        self.cb1.Disable()

        self.mark3 = wx.StaticText(self,-1,label=mark_empty)
        self.mark3.SetForegroundColour((0,255,0))
        self.mark3.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark3, pos=(9,3), flag=wx.ALL, border=5)

        text4 = wx.StaticText(self,-1,label="Value Field(s)")
        self.sizer.Add(text4, pos=(10, 0), flag=wx.ALL, border=10)

        self.chklbx2 = wx.CheckListBox(self,choices=[])
        self.chklbx2.SetToolTip(wx.ToolTip('Tick all fields that contain values to be extracted'))
        self.sizer.Add(self.chklbx2, pos=(10, 1), span=(4,3),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.chklbx2.Disable()

        self.sel_btn2 = wx.Button(self, label="Select")
        self.sel_btn2.Bind(wx.EVT_BUTTON, self.onSel2)
        self.sizer.Add(self.sel_btn2, pos=(10,4), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)
        self.sel_btn2.Disable()

        self.mark4 = wx.StaticText(self,-1,label=mark_empty)
        self.mark4.SetForegroundColour((0,255,0))
        self.mark4.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark4, pos=(10,5), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(14, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        header3 = wx.StaticText(self,-1,label="Set Direction of Effect")
        header3.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header3, pos=(15, 0), flag=wx.ALL, border=10)

        help_btn3 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn3.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn3.SetForegroundColour(wx.Colour(0,0,255))
        help_btn3.Bind(wx.EVT_BUTTON, self.onHlp3)
        self.sizer.Add(help_btn3, pos=(15,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text5 = wx.StaticText(self,-1,label="Define Direction of Effect")
        self.sizer.Add(text5, pos=(16, 0), flag=wx.ALL, border=10)

        self.ulist1 = ULC.UltimateListCtrl(self, wx.ID_ANY, agwStyle=ULC.ULC_HAS_VARIABLE_ROW_HEIGHT | wx.LC_REPORT | wx.LC_VRULES | wx.LC_HRULES)
        self.ulist1.InsertColumn(col=0, heading="Variable Name",format=0)
        self.ulist1.InsertColumn(col=1, heading="Positive",format=0)
        self.ulist1.InsertColumn(col=2, heading="Negative",format=0)
        self.ulist1.SetColumnWidth(0,ULC.ULC_AUTOSIZE_FILL)
        self.ulist1.SetColumnWidth(1,ULC.ULC_AUTOSIZE_USEHEADER)
        self.ulist1.SetColumnWidth(2,ULC.ULC_AUTOSIZE_USEHEADER)
        self.sizer.Add(self.ulist1,pos=(16,1),span=(5,4),flag=wx.TOP|wx.BOTTOM|wx.EXPAND|wx.RIGHT, border=5)
        self.ulist1.Disable()

        self.enter_btn2 = wx.Button(self, label="Done")
        self.enter_btn2.Bind(wx.EVT_BUTTON, self.onEnt2)
        self.sizer.Add(self.enter_btn2, pos=(16,5), flag=wx.TOP|wx.BOTTOM|wx.RIGHT|wx.LEFT, border=5)
        self.enter_btn2.Disable()

        self.mark5 = wx.StaticText(self,-1,label=mark_empty)
        self.mark5.SetForegroundColour((0,255,0))
        self.mark5.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark5, pos=(16,6), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(21, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        # add previous Button
        self.backBtn = wx.Button(self, label="< Back")
        self.backBtn.Bind(wx.EVT_BUTTON, self.onBack)
        self.sizer.Add(self.backBtn, pos=(22,4),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        # add next button
        self.nextBtn = wx.Button(self, label="Next >")
        self.nextBtn.Bind(wx.EVT_BUTTON, self.onNext)
        self.sizer.Add(self.nextBtn, pos=(22,5),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)
        self.nextBtn.Disable()

        # add cancel button
        self.cancBtn1 = wx.Button(self, label="Cancel")
        self.cancBtn1.Bind(wx.EVT_BUTTON, self.onCanc1)
        self.cancBtn1.SetToolTip(wx.ToolTip('Cancel'))
        self.sizer.Add(self.cancBtn1, pos=(22,6),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        self.SetSizer(self.sizer)
        self.sizer.AddGrowableRow(10)
        self.sizer.AddGrowableRow(16)
        self.sizer.AddGrowableCol(1)
        self.GetParent().SendSizeEvent()
        self.Layout()
        self.Fit()

    def onHlp0(self,event):
        """Help window for variable name"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3E_SetVariableName.html")
        htmlViewerInstance.Show()

    def onChar(self,event):
        """textbox for variable name"""
        keycode = event.GetKeyCode()
        if keycode in ascii_char: #only allow chars in alphabet
            event.Skip()

    def onEnt0(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """enter variable name"""
        inp_var = self.tc0.GetValue()
        if not inp_var:
            wx.MessageBox('Please enter a variable name','Error',wx.OK|wx.ICON_ERROR)
        else: # change this to be based on the database, so that entry can be changed
            global pE_name
            pE_name = "pE_"+inp_var
            if pE_name in var_list:
                wx.MessageBox('The variable already exists','Error',wx.OK|wx.ICON_ERROR)
            else:
                var_list.append(pE_name)
                self.mark1.SetLabel(mark_done) # change tick mark to done
                log.write('\nVariable name: '+inp_var)
                self.chklbx1.Enable()
                self.tc0.Disable()
                self.enter_btn0.Disable()
                self.sel_btn1.Enable()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onHlp1(self,event):
        """Help window value/distance methods"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3E_SetMethod.html")
        htmlViewerInstance.Show()

    def onSel1(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """select distance/value method for extraction"""
        met = self.chklbx1.GetCheckedStrings()
        if not met:
            wx.MessageBox('Select at least one method','Error',wx.OK|wx.ICON_ERROR)
        global metcodenone # list to hold distance only method codes selected => move to top?
        metcodenone=[]
        global metcodeval # list to hold value*distance method codes selected => move to top?
        metcodeval=[]
        for field in met:
            newmethod=extrmet[field]
            if newmethod=='dist' or newmethod=='invd' or newmethod=='invsq':
                metcodenone.append(newmethod)
            else:
                metcodeval.append(newmethod)

        self.mark2.SetLabel(mark_done) # change tick mark to done
        self.cb1.Enable()
        if metcodeval:
            self.chklbx2.Enable()
            self.sel_btn2.Enable()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        self.chklbx1.Disable()
        self.sel_btn1.Disable()
        log.write('\nMethod(s) selected: '+str(met))

    def onHlp2(self,event):
        """Help window for data input"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3E_SetInputData.html")
        htmlViewerInstance.Show()

    def onCb1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Select Polygon feature class"""
        self.chklbx2.Clear()
        global pE
        pE = self.cb1.GetValue()
        num_row=int(arcpy.GetCount_management(pE).getOutput(0))
        if num_row==0:#check that it contains features
            wx.MessageBox('The selected feature class is empty','Error',wx.OK|wx.ICON_ERROR)

        if metcodeval and num_row>0: #if value is wanted add value fields
            num_fields = [f.name for f in arcpy.ListFields(pE,"",'Double') if not f.required] #get numeric fields
            num_fields.extend([f.name for f in arcpy.ListFields(pE,"",'Integer')])
            num_fields.extend([f.name for f in arcpy.ListFields(pE,"",'Single')])
            num_fields.extend([f.name for f in arcpy.ListFields(pE,"",'SmallInteger')])
            num_fields.sort()
            if not num_fields:
                wx.MessageBox('The selected feature class contains no numeric fields','Error',wx.OK|wx.ICON_ERROR)
            else:
                self.chklbx2.Append(num_fields)
                self.mark3.SetLabel(mark_done) # change tick mark to done
                log.write('\nLine Feature Class: '+pE)
        elif num_row>0: # if value fields are not required just copy the featureclass
            arcpy.FeatureClassToFeatureClass_conversion(pE,out_fds,pE_name, "", "") # copy the feature class into the feature dataset
            pECats_display=[] #make a display list of the variables that will be created
            pECats=[] # make a reference list
            for item in metcodenone:
                # met = list(extrmet.keys())[list(extrmet.values().index(item))]
                met = [key for (key, value) in extrmet.items() if value == item] # get key from value
                pECats_display.append(met[0])
                pECats.append(pE_name+'_none_'+item)
            #populate ulist
            self.ulist1.DeleteAllItems()
            self.radios = []
            myMethods = ['Positive','Negative']
            for item in range(len(pECats)):
                self.ulist1.InsertStringItem(item, str(pECats_display[item]))
                for rad in range(1,3):
                    cat = pECats[item]
                    met = myMethods[rad - 1]
                    name_of_radio = "{cat}_{met}".format(cat=cat, met=met)
                    self.ulist1.SetStringItem(item, rad, "")
                    if rad==1:
                        style=wx.RB_GROUP
                    else:
                        style=0
                    self.radBt = wx.RadioButton(self.ulist1, wx.ID_ANY, "", style=style,name=name_of_radio)
                    self.ulist1.SetItemWindow(item,rad,self.radBt,expand=True)
                    self.radios.append(self.radBt)

            self.ulist1.Enable()
            self.enter_btn2.Enable()
            self.mark3.SetLabel(mark_done)

        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onSel2(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """select value fields"""
        global pE_cat
        pE_cat = list(self.chklbx2.GetCheckedStrings())
        pE_fldsNamesDict={}
        if not pE_cat:
            wx.MessageBox('Please select one or more fields','Error',wx.OK|wx.ICON_ERROR)
        else:
            for field in pE_cat:
                newfield = field.strip().replace(" ","").replace("_","").replace(".","")
                new_entry={str(field):str(newfield)}
                pE_fldsNamesDict.update(new_entry)
            fieldmappings=customFieldMap(pE,pE_fldsNamesDict) # create fieldmap
            arcpy.FeatureClassToFeatureClass_conversion(pE,out_fds,pE_name, "", fieldmappings) # copy the feature class into the feature dataset with the value fields

        pECats_display=[] #make a display list of the variables that will be created
        pECats=[] # make a reference list
        if metcodenone:
            for item in metcodenone:
                # met = list(extrmet.keys())[list(extrmet.values().index(item))]
                met = [key for (key, value) in extrmet.items() if value == item] # get key from value
                pECats_display.append(met[0])
                pECats.append(pE_name+'_none_'+item)
        if metcodeval:
            for val in pE_cat:
                val = val.strip().replace(" ","").replace("_","").replace(".","")
                for item in metcodeval:
                    # met = list(extrmet.keys())[list(extrmet.values().index(item))]
                    met = [key for (key, value) in extrmet.items() if value == item] # get key from value
                    index = met[0].find('*')
                    if index<0: # index is -1 if 'value' has been selected
                        pECats_display.append(val)
                        pECats.append(pE_name+'_'+val+'_'+item)
                    else:
                        met2 = met[0][index+2:]
                        pECats_display.append(val+' * '+str(met2))
                        pECats.append(pE_name+'_'+val+'_'+item)

        #populate ulist
        self.ulist1.DeleteAllItems()
        self.radios = []
        myMethods = ['Positive','Negative']
        for item in range(len(pECats)):
            self.ulist1.InsertStringItem(item, str(pECats_display[item]))
            for rad in range(1,3):
                cat = pECats[item]
                met = myMethods[rad - 1]
                name_of_radio = "{cat}_{met}".format(cat=cat, met=met)
                self.ulist1.SetStringItem(item, rad, "")
                if rad==1:
                    style=wx.RB_GROUP
                else:
                    style=0
                self.radBt = wx.RadioButton(self.ulist1, wx.ID_ANY, "", style=style,name=name_of_radio)
                self.ulist1.SetItemWindow(item,rad,self.radBt,expand=True)
                self.radios.append(self.radBt)

        self.ulist1.Enable()
        self.enter_btn2.Enable()
        self.mark4.SetLabel(mark_done) # change tick mark to done
        self.chklbx2.Disable()
        self.sel_btn2.Disable()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        log.write('\nValue field(s): '+str(pE_cat))

    def onHlp3(self,event):
        """Help window for direction of effect"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3E_SetSourceSink.html")
        htmlViewerInstance.Show()

    def onEnt2(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Define Direction of Effect"""
        results_dict = {}
        for i in self.radios:
            if i.GetValue()==True:
                n = i.GetName()
                index = n.rfind('_')
                cat = n[0:index]
                met = n[index+1:]
                results_dict[cat]= met
        sourcesink.update(results_dict)

        if arcpy.Exists(out_fds+"\\"+pE_name):
            self.nextBtn.Enable()

        self.mark5.SetLabel(mark_done) # change tick mark to done
        self.Parent.statusbar.SetStatusText('Ready')
        self.enter_btn2.Disable()
        del wait
        log.write('\nVariable definitions: ')
        for k, v in sorted(results_dict.items()):
            log.write('\n{:<30}: {:<6}'.format(k,v))

    def onBack(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Return to panel3"""
        try:
            var_list.remove(pE_name)
            for i in self.radios: # remove from sourcesink dictionary
                if i.GetValue()==True:
                    n = i.GetName()
                    index = n.rfind('_')
                    cat = n[0:index]
                    sourcesink.pop(cat, None)
        except:
            pass
        if arcpy.Exists(out_fds+"\\"+pE_name):
            arcpy.Delete_management(out_fds+"\\"+pE_name)
        self.tc0.Clear()
        for cb in self.chklbx1.GetCheckedItems():
            self.chklbx1.Check(cb,False)
        self.chklbx2.Clear()
        self.ulist1.DeleteAllItems()
        self.mark1.SetLabel(mark_empty)
        self.mark2.SetLabel(mark_empty)
        self.mark3.SetLabel(mark_empty)
        self.mark4.SetLabel(mark_empty)
        self.mark5.SetLabel(mark_empty)
        self.chklbx1.Disable()
        self.cb1.Disable()
        self.chklbx2.Disable()
        self.nextBtn.Disable()
        self.tc0.Enable()
        self.enter_btn0.Enable()
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel3,1,wx.EXPAND)
        newsize = self.Parent.panel3.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel3.Show()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        log.write(time.strftime("\n\nLine Distance - Stopped by user without completion: %A %d %b %Y %H:%M:%S", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel3E','back',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel3','start',datetime())''')
        db.commit()

    def onNext(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """"""
        arcpy.SpatialJoin_analysis(out_fds+"\\sites", out_fds+"\\"+pE_name, out_fds+"\\"+pE_name+"_join", "JOIN_ONE_TO_ONE", "KEEP_ALL","","CLOSEST","","distance") #spatial join of polygon to monitoring sites
        if 'invsq' in metcodenone or 'valinvsq' in metcodeval:
            arcpy.AddField_management(out_fds+"\\"+pE_name+"_join","distsqu","DOUBLE")
            cursor=arcpy.da.UpdateCursor(out_fds+"\\"+pE_name+"_join", ["distance","distsqu"])
            for row in cursor:
                row[1]=row[0]**2
                cursor.updateRow(row)
        # move data to sql database
        fc_to_sql(conn,out_fds+"\\"+pE_name+"_join")
        db.commit()
        arcpy.Delete_management(out_fds+"\\"+pE_name+"_join")
        # calcaulate Variables
        vars_sql_temp=list()
        if 'dist' in metcodenone:
            conn.execute("ALTER TABLE "+pE_name+"_join ADD "+pE_name+"_none_dist float64")
            conn.execute("UPDATE "+pE_name+"_join SET "+pE_name+"_none_dist=distance")
            vars_sql_temp.append(pE_name+"_none_dist")

        if 'invd' in metcodenone:
            conn.execute("ALTER TABLE "+pE_name+"_join ADD "+pE_name+"_none_invd float64")
            conn.execute("UPDATE "+pE_name+"_join SET "+pE_name+"_none_invd=1/distance")
            vars_sql_temp.append(pE_name+"_none_invd")

        if 'invsq' in metcodenone:
            conn.execute("ALTER TABLE "+pE_name+"_join ADD "+pE_name+"_none_invsq float64")
            conn.execute("UPDATE "+pE_name+"_join SET "+pE_name+"_none_invsq=1/distsqu")
            vars_sql_temp.append(pE_name+"_none_invsq")

        if 'val' in metcodeval:
            for i in pE_cat:
                i = i.strip().replace(" ","").replace("_","").replace(".","")
                conn.execute("ALTER TABLE "+pE_name+"_join ADD "+pE_name+"_"+i+"_val float64")
                conn.execute("UPDATE "+pE_name+"_join SET "+pE_name+"_"+i+"_val="+i)
                vars_sql_temp.append(pE_name+"_"+i+"_val")

        if 'valdist' in metcodeval:
            for i in pE_cat:
                i = i.strip().replace(" ","").replace("_","").replace(".","")
                conn.execute("ALTER TABLE "+pE_name+"_join ADD "+pE_name+"_"+i+"_valdist float64")
                conn.execute("UPDATE "+pE_name+"_join SET "+pE_name+"_"+i+"_valdist="+i+"*distance")
                vars_sql_temp.append(pE_name+"_"+i+"_valdist")

        if 'valinvd' in metcodeval:
            for i in pE_cat:
                i = i.strip().replace(" ","").replace("_","").replace(".","")
                conn.execute("ALTER TABLE "+pE_name+"_join ADD "+pE_name+"_"+i+"_valinvd float64")
                conn.execute("UPDATE "+pE_name+"_join SET "+pE_name+"_"+i+"_valinvd="+i+"*1/distance")
                vars_sql_temp.append(pE_name+"_"+i+"_valinvd")

        if 'valinvsq' in metcodeval:
            for i in pE_cat:
                i = i.strip().replace(" ","").replace("_","").replace(".","")
                conn.execute("ALTER TABLE "+pE_name+"_join ADD "+pE_name+"_"+i+"_valinvsq float64")
                conn.execute("UPDATE "+pE_name+"_join SET "+pE_name+"_"+i+"_valinvsq="+i+"*1/distsqu")
                vars_sql_temp.append(pE_name+"_"+i+"_valinvsq")

        conn.execute("CREATE UNIQUE INDEX "+pE_name+"_idx on "+pE_name+"_join (siteID_INT);")
        db.commit()
        # add to dat4status
        for var in vars_sql_temp:
            conn.execute('ALTER TABLE dat4stats ADD {} float64'.format(var)) # if I add second query I get an operational error
        db.commit()

        for var in vars_sql_temp:
            qry="UPDATE dat4stats \
                 SET "+var+" = ( \
                 SELECT "+var+" \
                 FROM  "+pE_name+"_join \
                 WHERE siteID_INT = dat4stats.siteID_INT)"
            conn.execute(qry)
        db.commit()
        conn.execute("DROP TABLE "+pE_name+"_join")
        db.commit()
        # push predictor names to panel 3
        WizardPanel3.list_ctrl.DeleteAllItems()
        cursor = conn.execute("SELECT * FROM dat4stats")
        varnames = list([x[0] for x in cursor.description])
        prednames = [name for name in varnames if (name[0] =='p')]
        index=0
        for i in prednames:
            WizardPanel3.list_ctrl.InsertItem(index,i)
            WizardPanel3.list_ctrl.SetItem(index,1,sourcesink[i])
            if index % 2:
                WizardPanel3.list_ctrl.SetItemBackgroundColour(index, col="white")
            else:
                WizardPanel3.list_ctrl.SetItemBackgroundColour(index, col=(222,234,246))
            index+=1
            WizardPanel3.list_ctrl.SetColumnWidth(0,wx.LIST_AUTOSIZE)
            WizardPanel3.list_ctrl.SetColumnWidth(1,wx.LIST_AUTOSIZE)
        log.write(time.strftime("\nLine Distance - End Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel3E','stop',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel3','start',datetime())''')
        db.commit()
        # clear p3E
        self.tc0.Clear()
        for cb in self.chklbx1.GetCheckedItems():
            self.chklbx1.Check(cb,False)
        self.chklbx2.Clear()
        self.ulist1.DeleteAllItems()
        self.mark1.SetLabel(mark_empty)
        self.mark2.SetLabel(mark_empty)
        self.mark3.SetLabel(mark_empty)
        self.mark4.SetLabel(mark_empty)
        self.mark5.SetLabel(mark_empty)
        self.chklbx1.Disable()
        self.cb1.Disable()
        self.chklbx2.Disable()
        self.nextBtn.Disable()
        self.tc0.Enable()
        self.enter_btn0.Enable()
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel3,1,wx.EXPAND)
        newsize = self.Parent.panel3.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel3.Show()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onCanc1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Cancel Wizard"""
        dlg = wx.MessageDialog(None, "Cancelling the wizard will delete all progress. Do you wish to cancel the wizard?",'Cancel',wx.YES_NO | wx.ICON_WARNING)
        result = dlg.ShowModal()
        if result == wx.ID_YES:
            arcpy.AddMessage((time.strftime("Cancelled by user: %A %d %b %Y %H:%M:%S", time.localtime())))
            try:
                log.close()
                logging.shutdown()
                db.close()
                shutil.rmtree(out_path)
            except NameError as e:
                arcpy.AddMessage(e)
            except OSError as e:
                arcpy.AddMessage(e)
            del wait
            self.Parent.Close()
        else:
            self.Parent.statusbar.SetStatusText('Ready')
            del wait
#-------------------------------------------------------------------------------
class WizardPanel3F(wx.Panel):
    """Page 3F"""

    def __init__(self, parent, title=None):
        """Constructor"""
        wx.Panel.__init__(self, parent)
        self.locale = wx.Locale(wx.LANGUAGE_ENGLISH)
        wx.ToolTip.Enable(True)

        self.sizer = wx.GridBagSizer(0,0)

        title2 = wx.StaticText(self, -1, "Distance to and/or value of nearest Point")
        title2.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(title2, pos=(0,0), span=(1,5),flag=wx.TOP|wx.LEFT|wx.BOTTOM, border=10)

        self.sizer.Add(wx.StaticLine(self), pos=(1, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM, border=5)

        header0 = wx.StaticText(self,-1,label="Set Variable Name")
        header0.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header0, pos=(2, 0), flag=wx.ALL, border=10)

        help_btn0 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn0.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn0.SetForegroundColour(wx.Colour(0,0,255))
        help_btn0.Bind(wx.EVT_BUTTON, self.onHlp0)
        self.sizer.Add(help_btn0, pos=(2,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text1 = wx.StaticText(self,-1,label="Variable Name")
        self.sizer.Add(text1, pos=(3, 0), flag=wx.ALL, border=10)

        self.tc0 = wx.TextCtrl(self, -1)
        self.tc0.SetMaxLength(20)
        self.tc0.SetToolTip(wx.ToolTip('Enter a name for the predictor variable'))
        self.tc0.Bind(wx.EVT_CHAR,self.onChar)
        self.sizer.Add(self.tc0, pos=(3, 1), span=(1,2), flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=5)

        self.enter_btn0 = wx.Button(self, label="Enter")
        self.enter_btn0.Bind(wx.EVT_BUTTON, self.onEnt0)
        self.sizer.Add(self.enter_btn0, pos=(3,3), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)

        self.mark1 = wx.StaticText(self,-1,label=mark_empty)
        self.mark1.SetForegroundColour((0,255,0))
        self.mark1.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark1, pos=(3,4), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(4, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        header1 = wx.StaticText(self,-1,label="Set Method")
        header1.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header1, pos=(5, 0), flag=wx.ALL, border=10)

        help_btn1 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn1.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn1.SetForegroundColour(wx.Colour(0,0,255))
        help_btn1.Bind(wx.EVT_BUTTON, self.onHlp1)
        self.sizer.Add(help_btn1, pos=(5,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text2 = wx.StaticText(self,-1,label="Data to be extracted")
        self.sizer.Add(text2, pos=(6, 0), flag=wx.ALL, border=10)

        self.chklbx1 = wx.CheckListBox(self,-1,choices=sorted(extrmet.keys()))
        self.chklbx1.SetToolTip(wx.ToolTip('Select one or more pieces of information to be extracted'))
        self.sizer.Add(self.chklbx1, pos=(6,1), span=(1,3), flag=wx.TOP|wx.LEFT|wx.BOTTOM|wx.EXPAND, border=5)
        self.chklbx1.Disable()

        self.sel_btn1 = wx.Button(self, label="Select")
        self.sel_btn1.Bind(wx.EVT_BUTTON, self.onSel1)
        self.sizer.Add(self.sel_btn1, pos=(6,4), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)
        self.sel_btn1.Disable()

        self.mark2 = wx.StaticText(self,-1,label=mark_empty)
        self.mark2.SetForegroundColour((0,255,0))
        self.mark2.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark2, pos=(6,5), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(7, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        header2 = wx.StaticText(self,-1,label="Set Input Data")
        header2.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header2, pos=(8, 0), flag=wx.ALL, border=10)

        help_btn2 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn2.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn2.SetForegroundColour(wx.Colour(0,0,255))
        help_btn2.Bind(wx.EVT_BUTTON, self.onHlp2)
        self.sizer.Add(help_btn2, pos=(8,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text3 = wx.StaticText(self,-1,label="Point Feature Class")
        self.sizer.Add(text3, pos=(9, 0), flag=wx.ALL, border=10)

        self.cb1 = wx.ComboBox(self,choices=[],style=wx.CB_DROPDOWN)
        self.cb1.SetToolTip(wx.ToolTip('From the dropdown list select the feature class containing the point data'))
        self.sizer.Add(self.cb1, pos=(9, 1), span=(1,2),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.cb1.Bind(wx.EVT_COMBOBOX, self.onCb1)
        self.cb1.Disable()

        self.mark3 = wx.StaticText(self,-1,label=mark_empty)
        self.mark3.SetForegroundColour((0,255,0))
        self.mark3.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark3, pos=(9,3), flag=wx.ALL, border=5)

        text4 = wx.StaticText(self,-1,label="Value Field(s)")
        self.sizer.Add(text4, pos=(10, 0), flag=wx.ALL, border=10)

        self.chklbx2 = wx.CheckListBox(self,choices=[])
        self.chklbx2.SetToolTip(wx.ToolTip('Tick all fields that contain values to be extracted'))
        self.sizer.Add(self.chklbx2, pos=(10, 1), span=(4,3),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.chklbx2.Disable()

        self.sel_btn2 = wx.Button(self, label="Select")
        self.sel_btn2.Bind(wx.EVT_BUTTON, self.onSel2)
        self.sizer.Add(self.sel_btn2, pos=(10,4), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)
        self.sel_btn2.Disable()

        self.mark4 = wx.StaticText(self,-1,label=mark_empty)
        self.mark4.SetForegroundColour((0,255,0))
        self.mark4.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark4, pos=(10,5), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(14, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        header3 = wx.StaticText(self,-1,label="Set Direction of Effect")
        header3.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header3, pos=(15, 0), flag=wx.ALL, border=10)

        help_btn3 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn3.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn3.SetForegroundColour(wx.Colour(0,0,255))
        help_btn3.Bind(wx.EVT_BUTTON, self.onHlp3)
        self.sizer.Add(help_btn3, pos=(15,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text5 = wx.StaticText(self,-1,label="Define Direction of Effect")
        self.sizer.Add(text5, pos=(16, 0), flag=wx.ALL, border=10)

        self.ulist1 = ULC.UltimateListCtrl(self, wx.ID_ANY, agwStyle=ULC.ULC_HAS_VARIABLE_ROW_HEIGHT | wx.LC_REPORT | wx.LC_VRULES | wx.LC_HRULES)
        self.ulist1.InsertColumn(col=0, heading="Variable Name",format=0)
        self.ulist1.InsertColumn(col=1, heading="Positive",format=0)
        self.ulist1.InsertColumn(col=2, heading="Negative",format=0)
        self.ulist1.SetColumnWidth(0,ULC.ULC_AUTOSIZE_FILL)
        self.ulist1.SetColumnWidth(1,ULC.ULC_AUTOSIZE_USEHEADER)
        self.ulist1.SetColumnWidth(2,ULC.ULC_AUTOSIZE_USEHEADER)
        self.sizer.Add(self.ulist1,pos=(16,1),span=(5,4),flag=wx.TOP|wx.BOTTOM|wx.EXPAND|wx.RIGHT, border=5)
        self.ulist1.Disable()

        self.enter_btn2 = wx.Button(self, label="Done")
        self.enter_btn2.Bind(wx.EVT_BUTTON, self.onEnt2)
        self.sizer.Add(self.enter_btn2, pos=(16,5), flag=wx.TOP|wx.BOTTOM|wx.RIGHT|wx.LEFT, border=5)
        self.enter_btn2.Disable()

        self.mark5 = wx.StaticText(self,-1,label=mark_empty)
        self.mark5.SetForegroundColour((0,255,0))
        self.mark5.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark5, pos=(16,6), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(21, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        # add previous Button
        self.backBtn = wx.Button(self, label="< Back")
        self.backBtn.Bind(wx.EVT_BUTTON, self.onBack)
        self.sizer.Add(self.backBtn, pos=(22,4),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        # add next button
        self.nextBtn = wx.Button(self, label="Next >")
        self.nextBtn.Bind(wx.EVT_BUTTON, self.onNext)
        self.sizer.Add(self.nextBtn, pos=(22,5),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)
        self.nextBtn.Disable()

        # add cancel button
        self.cancBtn1 = wx.Button(self, label="Cancel")
        self.cancBtn1.Bind(wx.EVT_BUTTON, self.onCanc1)
        self.cancBtn1.SetToolTip(wx.ToolTip('Cancel'))
        self.sizer.Add(self.cancBtn1, pos=(22,6),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        self.SetSizer(self.sizer)
        self.sizer.AddGrowableRow(10)
        self.sizer.AddGrowableRow(16)
        self.sizer.AddGrowableCol(1)
        self.GetParent().SendSizeEvent()
        self.Layout()
        self.Fit()

    def onHlp0(self,event):
        """Help window for variable name"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3F_SetVariableName.html")
        htmlViewerInstance.Show()

    def onChar(self,event):
        """textbox for variable name"""
        keycode = event.GetKeyCode()
        if keycode in ascii_char: #only allow chars in alphabet
            event.Skip()

    def onEnt0(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """enter variable name"""
        inp_var = self.tc0.GetValue()
        if not inp_var:
            wx.MessageBox('Please enter a variable name','Error',wx.OK|wx.ICON_ERROR)
        else: # change this to be based on the database, so that entry can be changed
            global pF_name
            pF_name = "pF_"+inp_var
            if pF_name in var_list:
                wx.MessageBox('The variable already exists','Error',wx.OK|wx.ICON_ERROR)
            else:
                var_list.append(pF_name)
                self.mark1.SetLabel(mark_done) # change tick mark to done
                log.write('\nVariable name: '+inp_var)
                self.chklbx1.Enable()
                self.sel_btn1.Enable()
                self.tc0.Disable()
                self.enter_btn0.Disable()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onHlp1(self,event):
        """Help window value/distance methods"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3F_SetMethod.html")
        htmlViewerInstance.Show()

    def onSel1(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """select distance/value method for extraction"""
        met = self.chklbx1.GetCheckedStrings()
        if not met:
            wx.MessageBox('Select at least one method','Error',wx.OK|wx.ICON_ERROR)
        global metcodenone # list to hold distance only method codes selected => move to top?
        metcodenone=[]
        global metcodeval # list to hold value*distance method codes selected => move to top?
        metcodeval=[]
        for field in met:
            newmethod=extrmet[field]
            if newmethod=='dist' or newmethod=='invd' or newmethod=='invsq':
                metcodenone.append(newmethod)
            else:
                metcodeval.append(newmethod)

        self.mark2.SetLabel(mark_done) # change tick mark to done
        self.cb1.Enable()
        if metcodeval:
            self.chklbx2.Enable()
            self.sel_btn2.Enable()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        log.write('\nMethod(s) selected: '+str(met))
        self.chklbx1.Disable()
        self.sel_btn1.Disable()

    def onHlp2(self,event):
        """Help window for data input"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3F_SetInputData.html")
        htmlViewerInstance.Show()

    def onCb1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Select Polygon feature class"""
        self.chklbx2.Clear()
        global pF
        pF = self.cb1.GetValue()
        num_row=int(arcpy.GetCount_management(pF).getOutput(0))
        if num_row==0:#check that it contains features
            wx.MessageBox('The selected feature class is empty','Error',wx.OK|wx.ICON_ERROR)

        if metcodeval and num_row>0: #if value is wanted add value fields
            num_fields = [f.name for f in arcpy.ListFields(pF,"",'Double') if not f.required] #get numeric fields
            num_fields.extend([f.name for f in arcpy.ListFields(pF,"",'Integer')])
            num_fields.extend([f.name for f in arcpy.ListFields(pF,"",'Single')])
            num_fields.extend([f.name for f in arcpy.ListFields(pF,"",'SmallInteger')])
            num_fields.sort()
            if not num_fields:
                wx.MessageBox('The selected feature class contains no numeric fields','Error',wx.OK|wx.ICON_ERROR)
            else:
                self.chklbx2.Append(num_fields)
                self.mark3.SetLabel(mark_done) # change tick mark to done
                log.write('\nPoint Feature Class: '+pF)
        elif num_row>0: # if value fields are not required just copy the featureclass
            arcpy.FeatureClassToFeatureClass_conversion(pF,out_fds,pF_name, "", "") # copy the feature class into the feature dataset
            pFCats_display=[] #make a display list of the variables that will be created
            pFCats=[] # make a reference list
            for item in metcodenone:
                # met = list(extrmet.keys())[list(extrmet.values().index(item))]
                met = [key for (key, value) in extrmet.items() if value == item] # get key from value
                pFCats_display.append(met[0])
                pFCats.append(pF_name+'_none_'+item)
            #populate ulist
            self.ulist1.DeleteAllItems()
            self.radios = []
            myMethods = ['Positive','Negative']
            for item in range(len(pFCats)):
                self.ulist1.InsertStringItem(item, str(pFCats_display[item]))
                for rad in range(1,3):
                    cat = pFCats[item]
                    met = myMethods[rad - 1]
                    name_of_radio = "{cat}_{met}".format(cat=cat, met=met)
                    self.ulist1.SetStringItem(item, rad, "")
                    if rad==1:
                        style=wx.RB_GROUP
                    else:
                        style=0
                    self.radBt = wx.RadioButton(self.ulist1, wx.ID_ANY, "", style=style,name=name_of_radio)
                    self.ulist1.SetItemWindow(item,rad,self.radBt,expand=True)
                    self.radios.append(self.radBt)

            self.ulist1.Enable()
            self.enter_btn2.Enable()
            self.mark3.SetLabel(mark_done)

        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onSel2(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """select value fields"""
        global pF_cat
        pF_cat = list(self.chklbx2.GetCheckedStrings())
        pF_fldsNamesDict={}
        if not pF_cat:
            wx.MessageBox('Please select one or more fields','Error',wx.OK|wx.ICON_ERROR)
        else:
            for field in pF_cat:
                newfield = field.strip().replace(" ","").replace("_","").replace(".","")
                new_entry={str(field):str(newfield)}
                pF_fldsNamesDict.update(new_entry)
            fieldmappings=customFieldMap(pF,pF_fldsNamesDict) # create fieldmap
            arcpy.FeatureClassToFeatureClass_conversion(pF,out_fds,pF_name, "", fieldmappings) # copy the feature class into the feature dataset with the value fields

        pFCats_display=[] #make a display list of the variables that will be created
        pFCats=[] # make a reference list
        if metcodenone:
            for item in metcodenone:
                # met = list(extrmet.keys())[list(extrmet.values().index(item))]
                met = [key for (key, value) in extrmet.items() if value == item] # get key from value
                pFCats_display.append(met[0])
                pFCats.append(pF_name+'_none_'+item)
        if metcodeval:
            for val in pF_cat:
                val = val.strip().replace(" ","").replace("_","").replace(".","")
                for item in metcodeval:
                    # met = list(extrmet.keys())[list(extrmet.values().index(item))]
                    met = [key for (key, value) in extrmet.items() if value == item] # get key from value
                    index = met[0].find('*')
                    if index<0: # index is -1 if 'value' has been selected
                        pFCats_display.append(val)
                        pFCats.append(pF_name+'_'+val+'_'+item)
                    else:
                        met2 = met[0][index+2:]
                        pFCats_display.append(val+' * '+str(met2))
                        pFCats.append(pF_name+'_'+val+'_'+item)

        #populate ulist
        self.ulist1.DeleteAllItems()
        self.radios = []
        myMethods = ['Positive','Negative']
        for item in range(len(pFCats)):
            self.ulist1.InsertStringItem(item, str(pFCats_display[item]))
            for rad in range(1,3):
                cat = pFCats[item]
                met = myMethods[rad - 1]
                name_of_radio = "{cat}_{met}".format(cat=cat, met=met)
                self.ulist1.SetStringItem(item, rad, "")
                if rad==1:
                    style=wx.RB_GROUP
                else:
                    style=0
                self.radBt = wx.RadioButton(self.ulist1, wx.ID_ANY, "", style=style,name=name_of_radio)
                self.ulist1.SetItemWindow(item,rad,self.radBt,expand=True)
                self.radios.append(self.radBt)

        self.ulist1.Enable()
        self.enter_btn2.Enable()
        self.mark4.SetLabel(mark_done) # change tick mark to done
        self.chklbx2.Disable()
        self.sel_btn2.Disable()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        log.write('\nValue field(s): '+str(pF_cat))

    def onHlp3(self,event):
        """Help window for direction of effect"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3F_SetSourceSink.html")
        htmlViewerInstance.Show()

    def onEnt2(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Define Direction of Effect"""
        results_dict = {}
        for i in self.radios:
            if i.GetValue()==True:
                n = i.GetName()
                index = n.rfind('_')
                cat = n[0:index]
                met = n[index+1:]
                results_dict[cat]= met
        sourcesink.update(results_dict)

        if arcpy.Exists(out_fds+"\\"+pF_name):
            self.nextBtn.Enable()

        self.mark5.SetLabel(mark_done) # change tick mark to done
        self.Parent.statusbar.SetStatusText('Ready')
        self.enter_btn2.Disable()
        del wait
        log.write('\nVariable definitions: ')
        for k, v in sorted(results_dict.items()):
            log.write('\n{:<30}: {:<6}'.format(k,v))

    def onBack(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Return to panel3"""
        try:
            var_list.remove(pF_name)
            for i in self.radios: # remove from sourcesink dictionary
                if i.GetValue()==True:
                    n = i.GetName()
                    index = n.rfind('_')
                    cat = n[0:index]
                    sourcesink.pop(cat, None)
        except:
            pass
        if arcpy.Exists(out_fds+"\\"+pF_name):
            arcpy.Delete_management(out_fds+"\\"+pF_name)
        self.tc0.Clear()
        for cb in self.chklbx1.GetCheckedItems():
            self.chklbx1.Check(cb,False)
        self.chklbx2.Clear()
        self.ulist1.DeleteAllItems()
        self.mark1.SetLabel(mark_empty)
        self.mark2.SetLabel(mark_empty)
        self.mark3.SetLabel(mark_empty)
        self.mark4.SetLabel(mark_empty)
        self.mark5.SetLabel(mark_empty)
        self.chklbx1.Disable()
        self.cb1.Disable()
        self.chklbx2.Disable()
        self.nextBtn.Disable()
        self.tc0.Enable()
        self.enter_btn0.Enable()
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel3,1,wx.EXPAND)
        newsize = self.Parent.panel3.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel3.Show()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        log.write(time.strftime("\n\nPoint Distance - Stopped by user without completion: %A %d %b %Y %H:%M:%S", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel3F','back',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel3','start',datetime())''')
        db.commit()

    def onNext(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """"""
        arcpy.SpatialJoin_analysis(out_fds+"\\sites", out_fds+"\\"+pF_name, out_fds+"\\"+pF_name+"_join", "JOIN_ONE_TO_ONE", "KEEP_ALL","","CLOSEST","","distance") #spatial join of polygon to monitoring sites
        if 'invsq' in metcodenone or 'valinvsq' in metcodeval:
            arcpy.AddField_management(out_fds+"\\"+pF_name+"_join","distsqu","DOUBLE")
            cursor=arcpy.da.UpdateCursor(out_fds+"\\"+pF_name+"_join", ["distance","distsqu"])
            for row in cursor:
                row[1]=row[0]**2
                cursor.updateRow(row)
        # move data to sql database
        fc_to_sql(conn,out_fds+"\\"+pF_name+"_join")
        db.commit()
        arcpy.Delete_management(out_fds+"\\"+pF_name+"_join")
        # calcaulate Variables
        vars_sql_temp=list()
        if 'dist' in metcodenone:
            conn.execute("ALTER TABLE "+pF_name+"_join ADD "+pF_name+"_none_dist float64")
            conn.execute("UPDATE "+pF_name+"_join SET "+pF_name+"_none_dist=distance")
            vars_sql_temp.append(pF_name+"_none_dist")

        if 'invd' in metcodenone:
            conn.execute("ALTER TABLE "+pF_name+"_join ADD "+pF_name+"_none_invd float64")
            conn.execute("UPDATE "+pF_name+"_join SET "+pF_name+"_none_invd=1/distance")
            vars_sql_temp.append(pF_name+"_none_invd")

        if 'invsq' in metcodenone:
            conn.execute("ALTER TABLE "+pF_name+"_join ADD "+pF_name+"_none_invsq float64")
            conn.execute("UPDATE "+pF_name+"_join SET "+pF_name+"_none_invsq=1/distsqu")
            vars_sql_temp.append(pF_name+"_none_invsq")

        if 'val' in metcodeval:
            for i in pF_cat:
                i = i.strip().replace(" ","").replace("_","").replace(".","")
                conn.execute("ALTER TABLE "+pF_name+"_join ADD "+pF_name+"_"+i+"_val float64")
                conn.execute("UPDATE "+pF_name+"_join SET "+pF_name+"_"+i+"_val="+i)
                vars_sql_temp.append(pF_name+"_"+i+"_val")

        if 'valdist' in metcodeval:
            for i in pF_cat:
                i = i.strip().replace(" ","").replace("_","").replace(".","")
                conn.execute("ALTER TABLE "+pF_name+"_join ADD "+pF_name+"_"+i+"_valdist float64")
                conn.execute("UPDATE "+pF_name+"_join SET "+pF_name+"_"+i+"_valdist="+i+"*distance")
                vars_sql_temp.append(pF_name+"_"+i+"_valdist")

        if 'valinvd' in metcodeval:
            for i in pF_cat:
                i = i.strip().replace(" ","").replace("_","").replace(".","")
                conn.execute("ALTER TABLE "+pF_name+"_join ADD "+pF_name+"_"+i+"_valinvd float64")
                conn.execute("UPDATE "+pF_name+"_join SET "+pF_name+"_"+i+"_valinvd="+i+"*1/distance")
                vars_sql_temp.append(pF_name+"_"+i+"_valinvd")

        if 'valinvsq' in metcodeval:
            for i in pF_cat:
                i = i.strip().replace(" ","").replace("_","").replace(".","")
                conn.execute("ALTER TABLE "+pF_name+"_join ADD "+pF_name+"_"+i+"_valinvsq float64")
                conn.execute("UPDATE "+pF_name+"_join SET "+pF_name+"_"+i+"_valinvsq="+i+"*1/distsqu")
                vars_sql_temp.append(pF_name+"_"+i+"_valinvsq")

        conn.execute("CREATE UNIQUE INDEX "+pF_name+"_idx on "+pF_name+"_join (siteID_INT);")
        db.commit()
        # add to dat4status
        for var in vars_sql_temp:
            conn.execute('ALTER TABLE dat4stats ADD {} float64'.format(var)) # if I add second query I get an operational error
        db.commit()

        for var in vars_sql_temp:
            qry="UPDATE dat4stats \
                 SET "+var+" = ( \
                 SELECT "+var+" \
                 FROM  "+pF_name+"_join \
                 WHERE siteID_INT = dat4stats.siteID_INT)"
            conn.execute(qry)
        db.commit()
        conn.execute("DROP TABLE "+pF_name+"_join")
        db.commit()
        # push predictor names to panel 3
        WizardPanel3.list_ctrl.DeleteAllItems()
        cursor = conn.execute("SELECT * FROM dat4stats")
        varnames = list([x[0] for x in cursor.description])
        prednames = [name for name in varnames if (name[0] =='p')]
        index=0
        for i in prednames:
            WizardPanel3.list_ctrl.InsertItem(index,i)
            WizardPanel3.list_ctrl.SetItem(index,1,sourcesink[i])
            if index % 2:
                WizardPanel3.list_ctrl.SetItemBackgroundColour(index, col="white")
            else:
                WizardPanel3.list_ctrl.SetItemBackgroundColour(index, col=(222,234,246))
            index+=1
            WizardPanel3.list_ctrl.SetColumnWidth(0,wx.LIST_AUTOSIZE)
            WizardPanel3.list_ctrl.SetColumnWidth(1,wx.LIST_AUTOSIZE)
        log.write(time.strftime("\nPoint Distance - End Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel3F','stop',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel3','start',datetime())''')
        db.commit()
        # clear p3F
        self.tc0.Clear()
        for cb in self.chklbx1.GetCheckedItems():
            self.chklbx1.Check(cb,False)
        self.chklbx2.Clear()
        self.ulist1.DeleteAllItems()
        self.mark1.SetLabel(mark_empty)
        self.mark2.SetLabel(mark_empty)
        self.mark3.SetLabel(mark_empty)
        self.mark4.SetLabel(mark_empty)
        self.mark5.SetLabel(mark_empty)
        self.chklbx1.Disable()
        self.cb1.Disable()
        self.chklbx2.Disable()
        self.nextBtn.Disable()
        self.tc0.Enable()
        self.enter_btn0.Enable()
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel3,1,wx.EXPAND)
        newsize = self.Parent.panel3.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel3.Show()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onCanc1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Cancel Wizard"""
        dlg = wx.MessageDialog(None, "Cancelling the wizard will delete all progress. Do you wish to cancel the wizard?",'Cancel',wx.YES_NO | wx.ICON_WARNING)
        result = dlg.ShowModal()
        if result == wx.ID_YES:
            arcpy.AddMessage((time.strftime("Cancelled by user: %A %d %b %Y %H:%M:%S", time.localtime())))
            try:
                log.close()
                logging.shutdown()
                db.close()
                shutil.rmtree(out_path)
            except NameError as e:
                arcpy.AddMessage(e)
            except OSError as e:
                arcpy.AddMessage(e)
            del wait
            self.Parent.Close()
        else:
            self.Parent.statusbar.SetStatusText('Ready')
            del wait
#-------------------------------------------------------------------------------
class WizardPanel3G(wx.Panel):
    """Page 3G"""

    def __init__(self, parent, title=None):
        """Constructor"""
        wx.Panel.__init__(self, parent)
        self.locale = wx.Locale(wx.LANGUAGE_ENGLISH)
        wx.ToolTip.Enable(True)

        self.sizer = wx.GridBagSizer(0,0)

        title2 = wx.StaticText(self, -1, "Value of Raster cell")
        title2.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(title2, pos=(0,0), span=(1,5),flag=wx.TOP|wx.LEFT|wx.BOTTOM, border=10)

        self.sizer.Add(wx.StaticLine(self), pos=(1, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM, border=5)

        header0 = wx.StaticText(self,-1,label="Set Variable Name")
        header0.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header0, pos=(2, 0), flag=wx.ALL, border=10)

        help_btn0 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn0.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn0.SetForegroundColour(wx.Colour(0,0,255))
        help_btn0.Bind(wx.EVT_BUTTON, self.onHlp0)
        self.sizer.Add(help_btn0, pos=(2,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text1 = wx.StaticText(self,-1,label="Variable Name")
        self.sizer.Add(text1, pos=(3, 0), flag=wx.ALL, border=10)

        self.tc0 = wx.TextCtrl(self, -1)
        self.tc0.SetMaxLength(20)
        self.tc0.SetToolTip(wx.ToolTip('Enter a name for the predictor variable'))
        self.tc0.Bind(wx.EVT_CHAR,self.onChar)
        self.sizer.Add(self.tc0, pos=(3, 1), span=(1,2), flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=5)

        self.enter_btn0 = wx.Button(self, label="Enter")
        self.enter_btn0.Bind(wx.EVT_BUTTON, self.onEnt0)
        self.sizer.Add(self.enter_btn0, pos=(3,3), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)

        self.mark1 = wx.StaticText(self,-1,label=mark_empty)
        self.mark1.SetForegroundColour((0,255,0))
        self.mark1.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark1, pos=(3,4), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(4, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        header2 = wx.StaticText(self,-1,label="Set Input Data")
        header2.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header2, pos=(5, 0), flag=wx.ALL, border=10)

        help_btn2 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn2.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn2.SetForegroundColour(wx.Colour(0,0,255))
        help_btn2.Bind(wx.EVT_BUTTON, self.onHlp2)
        self.sizer.Add(help_btn2, pos=(5,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text3 = wx.StaticText(self,-1,label="Raster file")
        self.sizer.Add(text3, pos=(6, 0), flag=wx.ALL, border=10)

        self.cb1 = wx.ComboBox(self,choices=[],style=wx.CB_DROPDOWN)
        self.cb1.SetToolTip(wx.ToolTip('From the dropdown list select the file containing the raster data'))
        self.sizer.Add(self.cb1, pos=(6, 1), span=(1,2),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.cb1.Bind(wx.EVT_COMBOBOX, self.onCb1)
        self.cb1.Disable()

        self.mark3 = wx.StaticText(self,-1,label=mark_empty)
        self.mark3.SetForegroundColour((0,255,0))
        self.mark3.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark3, pos=(6,3), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(7, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        header3 = wx.StaticText(self,-1,label="Set Direction of Effect")
        header3.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header3, pos=(8, 0), flag=wx.ALL, border=10)

        help_btn3 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn3.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn3.SetForegroundColour(wx.Colour(0,0,255))
        help_btn3.Bind(wx.EVT_BUTTON, self.onHlp3)
        self.sizer.Add(help_btn3, pos=(8,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text5 = wx.StaticText(self,-1,label="Define Direction of Effect")
        self.sizer.Add(text5, pos=(9, 0), flag=wx.ALL, border=10)

        self.ulist1 = ULC.UltimateListCtrl(self, wx.ID_ANY, agwStyle=ULC.ULC_HAS_VARIABLE_ROW_HEIGHT | wx.LC_REPORT | wx.LC_VRULES | wx.LC_HRULES)
        self.ulist1.InsertColumn(col=0, heading="Variable  Name",format=0)
        self.ulist1.InsertColumn(col=1, heading="Positive",format=0)
        self.ulist1.InsertColumn(col=2, heading="Negative",format=0)
        self.ulist1.SetColumnWidth(0,ULC.ULC_AUTOSIZE_FILL)
        self.ulist1.SetColumnWidth(1,ULC.ULC_AUTOSIZE_USEHEADER)
        self.ulist1.SetColumnWidth(2,ULC.ULC_AUTOSIZE_USEHEADER)
        self.sizer.Add(self.ulist1,pos=(9,1),span=(2,4),flag=wx.TOP|wx.BOTTOM|wx.EXPAND|wx.RIGHT, border=5)
        self.ulist1.Disable()

        self.enter_btn2 = wx.Button(self, label="Done")
        self.enter_btn2.Bind(wx.EVT_BUTTON, self.onEnt2)
        self.sizer.Add(self.enter_btn2, pos=(9,5), flag=wx.TOP|wx.BOTTOM|wx.RIGHT|wx.LEFT, border=5)
        self.enter_btn2.Disable()

        self.mark5 = wx.StaticText(self,-1,label=mark_empty)
        self.mark5.SetForegroundColour((0,255,0))
        self.mark5.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark5, pos=(9,6), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(11, 0), span=(1, 7),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        # add previous Button
        self.backBtn = wx.Button(self, label="< Back")
        self.backBtn.Bind(wx.EVT_BUTTON, self.onBack)
        self.sizer.Add(self.backBtn, pos=(12,4),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        # add next button
        self.nextBtn = wx.Button(self, label="Next >")
        self.nextBtn.Bind(wx.EVT_BUTTON, self.onNext)
        self.sizer.Add(self.nextBtn, pos=(12,5),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)
        self.nextBtn.Disable()

        # add cancel button
        self.cancBtn1 = wx.Button(self, label="Cancel")
        self.cancBtn1.Bind(wx.EVT_BUTTON, self.onCanc1)
        self.cancBtn1.SetToolTip(wx.ToolTip('Cancel'))
        self.sizer.Add(self.cancBtn1, pos=(12,6),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        self.sizer.AddGrowableCol(1)
        self.sizer.AddGrowableRow(9)
        self.SetSizer(self.sizer)
        self.GetParent().SendSizeEvent()
        self.Layout()
        self.Fit()

        #check for spatial analyst extension, warning if no license
        if arcpy.CheckExtension("Spatial") == "Available":
            arcpy.CheckOutExtension("Spatial")
        else:
            wx.MessageBox('The Spatial Analyst license is unavailable. If you are planning to analyse raster data, you must have a valid license for Spatial Analyst.','Error',wx.OK|wx.ICON_ERROR)


    def onHlp0(self,event):
        """Help window for variable name"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3G_SetVariableName.html")
        htmlViewerInstance.Show()

    def onChar(self,event):
        keycode = event.GetKeyCode()
        if keycode in ascii_char: #only allow chars in alphabet
            event.Skip()

    def onEnt0(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """enter variable name"""
        inp_var = self.tc0.GetValue()
        if not inp_var:
            wx.MessageBox('Please enter a variable name','Error',wx.OK|wx.ICON_ERROR)
        else: # change this to be based on the database, so that entry can be changed
            global pG_name
            pG_name = "pG_"+inp_var
            if pG_name in var_list:
                wx.MessageBox('The variable already exists','Error',wx.OK|wx.ICON_ERROR)
            elif arcpy.Exists(out_fgdb+"\\"+pG_name):
                wx.MessageBox('A raster file with the same name already exists.\nPlease select a different name.','Error',wx.OK|wx.ICON_ERROR)
            else:
                var_list.append(pG_name)
                self.mark1.SetLabel(mark_done) # change tick mark to done
                log.write('\nVariable name: '+inp_var)
                self.cb1.Enable()
                self.tc0.Disable()
                self.enter_btn0.Disable()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onHlp2(self,event):
        """Help window for input data"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3G_SetInputData.html")
        htmlViewerInstance.Show()

    def onCb1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Select raster file"""
        global pG
        pG = self.cb1.GetValue()
        if arcpy.Describe(in_fgdb+"\\"+pG).spatialReference.name==arcpy.Describe(out_fds).spatialReference.name: # don't know how to check for errors in raster data
            arcpy.CopyRaster_management(in_fgdb+"\\"+pG,out_fgdb+"\\"+pG_name) # just copy if it's already in the correct coordinate system
            self.mark3.SetLabel(mark_done)
            self.ulist1.Enable()
        else:
            if arcpy.Exists(out_fgdb+"\\temp_wgs_proj"):
                arcpy. Delete_management(out_fgdb+"\\temp_wgs_proj")
            arcpy.ProjectRaster_management(in_fgdb+"\\"+pG,out_fgdb+"\\temp_wgs_proj",4326,"BILINEAR") # project raster into wgs coordinate system
            arcpy.ProjectRaster_management(out_fgdb+"\\temp_wgs_proj",out_fgdb+"\\"+pG_name,out_fds,"BILINEAR") # project raster into wanted coordinate System
            arcpy. Delete_management(out_fgdb+"\\temp_wgs_proj")
            self.mark3.SetLabel(mark_done)
            self.ulist1.Enable()
        #populate ulist1
        self.ulist1.DeleteAllItems()
        self.radios = []
        myMethods = ['Positive','Negative']
        self.ulist1.InsertStringItem(0, pG_name+"_raster_val")
        for rad in range(1,3):
            cat = pG_name+"_raster_val"
            met = myMethods[rad - 1]
            name_of_radio = "{cat}_{met}".format(cat=cat, met=met)
            self.ulist1.SetStringItem(0, rad, "")
            if rad==1:
                style=wx.RB_GROUP
            else:
                style=0
            self.radBt = wx.RadioButton(self.ulist1, wx.ID_ANY, "", style=style,name=name_of_radio)
            self.ulist1.SetItemWindow(0,rad,self.radBt,expand=True)
            self.radios.append(self.radBt)
        self.Parent.statusbar.SetStatusText('Ready')
        self.enter_btn2.Enable()
        del wait
        log.write('\nRaster File: '+pG)

    def onHlp3(self,event):
        """Help window for direction of effect"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3G_SetSourceSink.html")
        htmlViewerInstance.Show()

    def onEnt2(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Define Direction of Effect"""
        results_dict = {}
        for i in self.radios:
            if i.GetValue()==True:
                n = i.GetName()
                index = n.rfind('_')
                cat = n[0:index]
                met = n[index+1:]
                results_dict[cat]= met
        sourcesink.update(results_dict)

        if arcpy.Exists(out_fgdb+"\\"+pG_name):
            self.nextBtn.Enable()

        self.mark5.SetLabel(mark_done) # change tick mark to done
        self.cb1.Disable()
        self.ulist1.Disable()
        self.enter_btn2.Disable()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        log.write('\nVariable definitions: ')
        for k, v in sorted(results_dict.items()):
            log.write('\n{:<30}: {:<6}'.format(k,v))

    def onBack(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Return to panel3"""
        try:
            var_list.remove(pG_name)
            for i in self.radios: # remove from sourcesink dictionary
                if i.GetValue()==True:
                    n = i.GetName()
                    index = n.rfind('_')
                    cat = n[0:index]
                    sourcesink.pop(cat, None)
        except:
            pass
        if arcpy.Exists(out_fgdb+"\\"+pG_name):
            arcpy.Delete_management(out_fgdb+"\\"+pG_name)
        self.tc0.Clear()
        self.mark1.SetLabel(mark_empty)
        self.mark3.SetLabel(mark_empty)
        self.mark5.SetLabel(mark_empty)
        self.cb1.Disable()
        self.ulist1.Disable()
        self.nextBtn.Disable()
        self.tc0.Enable()
        self.enter_btn0.Enable()
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel3,1,wx.EXPAND)
        newsize = self.Parent.panel3.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel3.Show()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        log.write(time.strftime("\n\nRaster Value - Stopped by user without completion: %A %d %b %Y %H:%M:%S", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel3G','back',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel3','start',datetime())''')
        db.commit()

    def onNext(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """"""
        if arcpy.Exists(out_fgdb+"\\"+pG_name+"_sites"):
            arcpy.Delete_management(out_fgdb+"\\"+pG_name+"_sites")
        arcpy.sa.ExtractValuesToPoints(out_fds+"\\sites",out_fgdb+"\\"+pG_name,out_fgdb+"\\"+pG_name+"_sites","","VALUE_ONLY") #join cell values to points
        fc_to_sql(conn,out_fgdb+"\\"+pG_name+"_sites") # copy to sqlite
        conn.execute("CREATE UNIQUE INDEX "+pG_name+"_idx on "+pG_name+"_sites (siteID_INT);")
        db.commit()
        arcpy.Delete_management(out_fgdb+"\\"+pG_name+"_sites")
        # add to dat4status
        var = pG_name+"_raster_val"
        conn.execute('ALTER TABLE dat4stats ADD {} float64'.format(var))
        db.commit()
        qry="UPDATE dat4stats \
                 SET "+var+" = ( \
                 SELECT RASTERVALU \
                 FROM  "+pG_name+"_sites \
                 WHERE siteID_INT = dat4stats.siteID_INT)"
        conn.execute(qry)
        db.commit()
        # push predictor names to panel 3
        WizardPanel3.list_ctrl.DeleteAllItems()
        cursor = conn.execute("SELECT * FROM dat4stats")
        varnames = list([x[0] for x in cursor.description])
        prednames = [name for name in varnames if (name[0] =='p')]
        index=0
        for i in prednames:
            WizardPanel3.list_ctrl.InsertItem(index,i)
            WizardPanel3.list_ctrl.SetItem(index,1,sourcesink[i])
            if index % 2:
                WizardPanel3.list_ctrl.SetItemBackgroundColour(index, col="white")
            else:
                WizardPanel3.list_ctrl.SetItemBackgroundColour(index, col=(222,234,246))
            index+=1
            WizardPanel3.list_ctrl.SetColumnWidth(0,wx.LIST_AUTOSIZE)
            WizardPanel3.list_ctrl.SetColumnWidth(1,wx.LIST_AUTOSIZE)
        log.write(time.strftime("\nRaster Value - End Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel3G','stop',datetime())''')
        conn.execute('''INSERT INTO timings VALUES('panel3','start',datetime())''')
        db.commit()
        # clear p3G
        self.tc0.Clear()
        self.mark1.SetLabel(mark_empty)
        self.mark3.SetLabel(mark_empty)
        self.mark5.SetLabel(mark_empty)
        self.cb1.Disable()
        self.ulist1.Disable()
        self.nextBtn.Disable()
        self.tc0.Enable()
        self.enter_btn0.Enable()
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel3,1,wx.EXPAND)
        newsize = self.Parent.panel3.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel3.Show()
        self.Parent.statusbar.SetStatusText('Ready')
        #reset environment?
        arcpy.env.workspace = in_fgdb
        del wait

    def onCanc1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Cancel Wizard"""
        dlg = wx.MessageDialog(None, "Cancelling the wizard will delete all progress. Do you wish to cancel the wizard?",'Cancel',wx.YES_NO | wx.ICON_WARNING)
        result = dlg.ShowModal()
        if result == wx.ID_YES:
            arcpy.AddMessage((time.strftime("Cancelled by user: %A %d %b %Y %H:%M:%S", time.localtime())))
            try:
                log.close()
                logging.shutdown()
                db.close()
                shutil.rmtree(out_path)
            except NameError as e:
                arcpy.AddMessage(e)
            except OSError as e:
                arcpy.AddMessage(e)
            del wait
            self.Parent.Close()
        else:
            self.Parent.statusbar.SetStatusText('Ready')
            del wait

#-------------------------------------------------------------------------------
class WizardPanel4(wx.Panel):
    """Fourth page"""

    def __init__(self, parent, title=None):
        """Constructor"""
        wx.Panel.__init__(self, parent)
        wx.ToolTip.Enable(True)

        self.sizer = wx.GridBagSizer(0,0)

        title2 = wx.StaticText(self, -1, "Model")
        title2.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(title2, pos=(0,0), flag=wx.TOP|wx.LEFT|wx.BOTTOM, border=10)

        self.sizer.Add(wx.StaticLine(self), pos=(1, 0), span=(1, 5),flag=wx.EXPAND|wx.BOTTOM, border=5)

        header1 = wx.StaticText(self,-1,label="Export data (optional)")
        header1.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header1, pos=(2, 0), flag=wx.ALL, border=10)

        help_btn1 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn1.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn1.SetForegroundColour(wx.Colour(0,0,255))
        help_btn1.Bind(wx.EVT_BUTTON, self.onHlp1)
        self.sizer.Add(help_btn1, pos=(2,1), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)

        text1 = wx.StaticText(self,-1,label="Save extracted data as text file")
        self.sizer.Add(text1, pos=(3, 0), span=(1,1),flag=wx.ALL|wx.EXPAND, border=10)

        export_btn = wx.Button(self, label="Export")
        export_btn.Bind(wx.EVT_BUTTON, self.onExport)
        export_btn.SetToolTip(wx.ToolTip('Click here to export the outcome and predictor variables into a standalone text file'))
        self.sizer.Add(export_btn, pos=(3,1), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)

        self.mark1 = wx.StaticText(self,-1,label=mark_empty)
        self.mark1.SetForegroundColour((0,255,0))
        self.mark1.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark1, pos=(3,2), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(4, 0), span=(1, 5),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        header2 = wx.StaticText(self,-1,label="Build LUR model")
        header2.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header2, pos=(5, 0), flag=wx.ALL, border=10)

        help_btn2 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn2.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn2.SetForegroundColour(wx.Colour(0,0,255))
        help_btn2.Bind(wx.EVT_BUTTON, self.onHlp2)
        self.sizer.Add(help_btn2, pos=(5,1), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)

        self.radio_bx = wx.RadioBox(self,-1,label="Type of model",choices=["Classic LUR","Hybrid LUR"], majorDimension=0, style=wx.RA_SPECIFY_COLS|wx.ALIGN_CENTRE)
        self.radio_bx.SetToolTip(wx.ToolTip('Select the type of model that you want to run'))
        self.sizer.Add(self.radio_bx, pos=(6,0), span=(1,3), flag=wx.LEFT|wx.BOTTOM|wx.TOP|wx.EXPAND, border=5)
        self.radio_bx.Bind(wx.EVT_RADIOBOX,self.onRadBx)

        self.text2 = wx.StaticText(self, label="Mandatory Variables")
        self.sizer.Add(self.text2, pos=(7, 0), flag=wx.ALL, border=10)
        self.text2.Disable()

        self.chkbx1 = wx.CheckListBox(self,choices=[])
        self.chkbx1.SetToolTip(wx.ToolTip('Tick all variables that you want to force into the model'))
        self.sizer.Add(self.chkbx1, pos=(7, 1), span=(2,2),flag=wx.TOP|wx.BOTTOM|wx.EXPAND|wx.RIGHT, border=5)
        self.chkbx1.Disable()

        self.sel_btn1 = wx.Button(self, label="Select")
        self.sel_btn1.Bind(wx.EVT_BUTTON, self.onSel1)
        self.sizer.Add(self.sel_btn1, pos=(7,3), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)
        self.sel_btn1.Disable()

        self.mark2 = wx.StaticText(self,-1,label=mark_empty)
        self.mark2.SetForegroundColour((0,255,0))
        self.mark2.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark2, pos=(7,4), flag=wx.ALL, border=5)

        self.model_btn = wx.Button(self, label="Build model")
        self.model_btn.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.model_btn.SetToolTip(wx.ToolTip('Click here to build the LUR model'))
        self.model_btn.Bind(wx.EVT_BUTTON, self.onModel)
        self.sizer.Add(self.model_btn, pos=(9,1),span=(2,2),flag=wx.TOP|wx.BOTTOM|wx.EXPAND, border=20)

        self.mark3 = wx.StaticText(self,-1,label=mark_empty)
        self.mark3.SetForegroundColour((0,255,0))
        self.mark3.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark3, pos=(9,3),span=(2,1), flag=wx.LEFT|wx.TOP|wx.BOTTOM|wx.EXPAND, border=20)

        self.sizer.Add(wx.StaticLine(self), pos=(11, 0), span=(1, 6),flag=wx.EXPAND|wx.BOTTOM, border=5)

        # add finish button
        self.finBtn = wx.Button(self, label="Finish")
        self.finBtn.Bind(wx.EVT_BUTTON, self.onFin)
        self.sizer.Add(self.finBtn, pos=(12,3),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        # add cancel button
        self.cancBtn1 = wx.Button(self, label="Cancel")
        self.cancBtn1.Bind(wx.EVT_BUTTON, self.onCanc1)
        self.cancBtn1.SetToolTip(wx.ToolTip('Cancel'))
        self.sizer.Add(self.cancBtn1, pos=(12,4),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        self.SetSizer(self.sizer)
        self.sizer.AddGrowableRow(7)
        self.sizer.AddGrowableCol(1)
        self.GetParent().SendSizeEvent()
        self.Layout()
        self.Fit()

    def onHlp1(self,event):
        """Help window for export data"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p4_ExportData.html")
        htmlViewerInstance.Show()

    def onExport(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """export data"""
        ts = time.strftime("%d%b%Y_%H%M%S", time.localtime())
        csvWriter = csv.writer(open(out_path+"\\lur_var_data_"+ts+".csv", "w", newline=''))
        conn.row_factory=sqlite3.Row
        crsr=conn.execute("SELECT * FROM dat4stats")
        row=crsr.fetchone()
        titles=list(row.keys())
        rows = run("SELECT * FROM dat4stats")
        csvWriter.writerow(titles)
        csvWriter.writerows(rows)
        self.mark1.SetLabel(mark_done)
        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        log.write('\nData file exported to: '+out_path+'\\lur_var_data_'+ts+'.csv')

    def onHlp2(self,event):
        """Help window for building model"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p4_BuildModel.html")
        htmlViewerInstance.Show()

    def onRadBx(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """select classic or hybrid"""
        if self.radio_bx.GetStringSelection() =="Hybrid LUR":
            self.text2.Enable()
            self.chkbx1.Enable()
            self.sel_btn1.Enable()
            self.model_btn.Disable()
            self.finBtn.Disable()
        else:
            self.chkbx1.Disable()
            self.sel_btn1.Disable()
            self.model_btn.Enable()

        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onSel1(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """select variables to be forced in"""
        global force_vars
        force_vars = list(self.chkbx1.GetCheckedStrings())
        if not force_vars:
            wx.MessageBox('Please select one or more variables','Error',wx.OK|wx.ICON_ERROR)
        else:
            self.model_btn.Enable()
            self.mark2.SetLabel(mark_done)
            self.Parent.statusbar.SetStatusText('Ready')
            del wait
            self.sel_btn1.Disable()

    def onModel(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        log.write('\nType of model: '+self.radio_bx.GetStringSelection())
        if force_vars:
            log.write('\nMandatory variables selected:'+str(force_vars)+'\n')
        """build LUR model"""
        time1 = time.clock() # to time code
        log.write(time.strftime("\nStarting statistical analysis: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        df = pd.read_sql_query("SELECT * FROM dat4stats",db) # load data from sqlite into dataframe
        deps = list(df.filter(regex='^dep',axis=1)) # list of dependent variables
        preds = list(df.filter(regex='^p', axis=1)) # list of predictor variables
        dfdep = df[deps] # dataframe of dependent variables
        dfstats = df.drop(['siteID','siteID_INT'],axis=1) # dataframe of all variables except site id ones
        # check number of observations
        preds_ok = list(preds)
        for column in dfstats:
            vals=dfstats[column].dropna().values.tolist() # extract column as list, drop missing values
            if len(vals)<8:
                log.write('\n+++ WARNING +++ '+column+' has only '+str(len(vals))+' observations! This variable will not be used in the analysis. Please check your data.')
                if column in deps:
                    deps.remove(column)
                elif column in preds:
                    preds_ok.remove(column)
                elif column in force_vars:
                    force_vars.remove(column)
        # create look up dictionary
        dict_lkup = dict(sourcesink)
        for key,value in list(dict_lkup.items()):
            if key=='p_XCOORD' or key=='p_YCOORD':
                dict_lkup[key] = 0
            elif value == "Positive":
                dict_lkup[key] = 1
            elif value == "Negative":
                dict_lkup[key] = -1
        # Descriptive analyses
        timestamp=time.strftime("%Y%m%d_%H%M%S")
        with PdfPages(out_path+'\\Descriptive_analyses_'+timestamp+'.pdf') as pdf:
            title_page('Boxplots and descriptive statistics of depedent variables')
            pdf.savefig()
            plt.close()
            for dep in deps:
                plot_bxp(df,dep)
                pdf.savefig()
                plt.close()
            title_page('Boxplots and descriptive statistics of predictor variables')
            pdf.savefig()
            plt.close()
            for pred in preds:
                plot_bxp(df,pred)
                pdf.savefig()
                plt.close()
            note='A csv file showing correlations of all variables has been saved at:\n'+out_path+'\CorrelationMatrix_predVars_'+timestamp+'.csv'
            title_pagenote('Correlation matrix of dependent variables',note)
            pdf.savefig()
            plt.close()
            dfdep=df[deps]
            plot_corr(dfdep,11.69,8.27)
            pdf.savefig()
            plt.close()
            title_page('Pairwise regression plots of dependent variables')
            pdf.savefig()
            plt.close()
            plot_pwreg(dfdep)
            pdf.savefig()
            plt.close()

        log.write('\nFile of descriptive statistics created: '+out_path+'\Descriptive_analyses_'+timestamp+'.pdf')
        log.write(time.strftime("\nDescriptive statistics completed: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        arcpy.AddMessage((time.strftime("Descriptive statistics completed: %A %d %b %Y %H:%M:%S", time.localtime())))
        # Correlation matrix of all variables as csv
        corrmat=dfstats.corr()
        corrmat.to_csv(out_path+'\\CorrelationMatrix_Vars_'+timestamp+'.csv')
        log.write('\nCorrelation matrix of all variables: '+out_path+'\CorrelationMatrix_Vars_'+timestamp+'.csv')
        log.write(time.strftime("\nCorrelation matrix completed: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        arcpy.AddMessage((time.strftime("Correlation matrix completed: %A %d %b %Y %H:%M:%S", time.localtime())))

        # if forced predictors remove them from preds_ok
        if force_vars:
            preds_ok = list(set(preds_ok)-set(force_vars))

        # linear regression model
        global df_resid
        df_resid=pd.DataFrame(df,columns=['siteID_INT','siteID']) # create dataframe to store all residuals

        for dep in deps: # iterate through dependent variables
            log.write('\n-------------------------------------------------------')
            log.write('\n\n'+dep+'\n') # write name of dependent variable to log
            log.write(time.strftime("\nStarting linear regression: %A %d %b %Y %H:%M:%S\n", time.localtime()))
            dict_r={} # create empty dictionary for adjusted R2
            dict_params={} # create empty dictionary for paramters
            dict_df={} # create empty dictionary for residual degrees of freedom
            sel_preds = list() # create empty list for selected predictors
            if force_vars:
                sel_preds = list(force_vars) # if forced predictors exist add them to selected predictors list
                frml_snip="+".join(sel_preds)
                frml=dep+'~'+frml_snip # create formula
                model=smf.ols(formula=frml,data=df,missing='drop').fit() # fit model with forced variables only
                log.write('\nModel with mandatory variables only :\n')
                log.write(model.summary().as_text()) # record summary of starting model
                log.write('\n\n')

            preds = list(preds_ok) # reset list of predictors for each dependent variable
            for pred in preds: # iterate through all predictors
                if force_vars:
                    frml_snip="+".join(sel_preds)
                    frml=dep+'~'+frml_snip+'+'+pred # create formula
                else:
                    frml=dep+'~'+pred
                model=smf.ols(formula=frml,data=df,missing='drop').fit() # fit model
                dict_r.update({pred:model.rsquared_adj}) # add predictors as keys and adjusted R2 as values
                dict_params.update({pred:[model.params[1:]]}) # add predictors as keys and coefficients as values
                dict_df.update({pred:model.df_resid}) # add predictors as keys and residual dof as values
            arcpy.AddMessage('Start model loop completed.')

            pred_del=[k for k,v in list(dict_df.items()) if v==0] # get predictors with zero residual dof
            for k in pred_del: # delete these predictors from the dictionaries, because they can't be used
                dict_r.pop(k,None)
                dict_params.pop(k,None)
                preds.remove(k)
                log.write('+++ WARNING +++ Model with predictor '+k+': No residual degrees of freedom! This predictor cannot be used. Check the number of observations.\n')
            if not dict_r: # if all predictors are removed, stop. Shouldn't happen because X,Y coordinates are in list of predictors
                log.write('\n +++ ERROR +++ No starting model can be found! Check the data.\n')
                continue

            max_key = max(dict_r, key=dict_r.get) # find the key with the highest adjusted R2, this is the start model
            max_r = dict_r[max_key] # save the adjusted R2
            max_b=dict_params[max_key][0].to_dict() # save the coefficient
            recode_coeff(max_b) # recode into positive/negative, so that it can be checked against dictionary
            if force_vars: # if forced vars remove them from coefficient check
                for var in force_vars:
                    del max_b[var]

            startmodel_loop=True
            while startmodel_loop:
                if direction_effect(max_b,dict_lkup)==True: # check that direction of effect of new predictor matches assumption
                    sel_preds.append(max_key) # if it does, this is the first predictor
                    frml_snip="+".join(sel_preds)
                    frml=dep+'~'+frml_snip # create formula
                    model_start=smf.ols(formula=frml,data=df,missing='drop').fit() # fit model
                    startmodel_loop=False
                else:
                    del dict_r[max_key] # if it doesn't, remove predictor from dictionary
                    max_key = max(dict_r, key=dict_r.get) # find the new best predictor
                    max_r = dict_r[max_key] # get the new adjusted R2
                    max_b = dict_params[max_key][0].to_dict() # get new coefficient
                    recode_coeff(max_b) # recode into positive/negative and check again
                    if force_vars: # if forced vars remove them from coefficient check
                        for var in force_vars:
                            del max_b[var]

            log.write('\nStart model for '+dep+':\n')
            log.write(model_start.summary().as_text()) # record summary of starting model
            log.write('\n\n')
            # Intermediate models
            i=1 # counter for intermediate models
            outer_loop=True
            while outer_loop:
                preds.remove(max_key) # remove max_key from list of names
                dict_r.clear() # empty dictionary
                dict_params.clear() # empty dictionary
                dict_df.clear() # empty dictionary
                frml_snip="+".join(sel_preds) # create string to be added into the formula
                for pred in preds: # iterate through all remaining predictors
                    frml=dep+'~'+frml_snip+'+'+pred # create formula
                    model=smf.ols(formula=frml,data=df,missing='drop').fit() # fit model
                    dict_r.update({pred:model.rsquared_adj}) # add predictors as keys and adjusted R2 as values
                    dict_params.update({pred:[model.params[1:]]}) # add predictors as keys and coefficients as values
                    dict_df.update({pred:model.df_resid}) # add predictors as keys and residual dof as values
                arcpy.AddMessage('Intermediate model loop completed.')

                pred_del=[k for k,v in list(dict_df.items()) if v==0] # delete predictor if residual dof is zero
                for k in pred_del: # remove predictor data from dictionaries
                    dict_r.pop(k,None)
                    dict_params.pop(k,None)
                    preds.remove(k)
                    log.write('+++ WARNING +++ Model with predictor '+k+': No residual degrees of freedom! This predictor cannot be used. Check the number of observations.\n')

                if not dict_r: # if no predictors are left
                    log.write('\n+++ WARNING +++ No intermediate model can be found. The starting model will be used as the intermediate model.')
                    outer_loop = False # break out of while loop

                max_key = max(dict_r, key=dict_r.get) # find the key with the highest adjusted R2
                max_r_new = dict_r[max_key] # find the new R squared value
                max_b=dict_params[max_key][0].to_dict() # get coefficients
                recode_coeff(max_b) # recode the coefficients into positive or negative
                if force_vars: # forced vars remove them from coefficient check
                    for var in force_vars:
                        del max_b[var]

                inner_loop=True  #inner loop to check R2 change and direction of effect
                while inner_loop: # check change in adjusted R2>0.01 and direction of effect of all predictors
                    if (max_r_new-max_r)>0.01 and direction_effect(max_b,dict_lkup)==True: # if change in R2 is greater than 1% and the direction of effect matches assumption, accept predictor and continue adding predictors
                        log.write('\nIntermediate model '+str(i)+': '+dep+'~'+frml_snip+'+'+max_key)
                        log.write('\nAdj. R-squared:'+str(round(max_r_new,2))+'\n')
                        frml = dep+'~'+frml_snip+'+'+max_key
                        model=smf.ols(formula=frml,data=df,missing='drop').fit()
                        log.write(model.summary().tables[1].as_text())
                        log.write('\n>>> Change in adjusted R-squared>1% and direction of effect of predictors as assumed, model accepted.\n')
                        sel_preds.append(max_key)  # add predictor to list
                        i+=1 # increase iterator
                        max_r=max_r_new # change maximum adjusted R2 to current value
                        inner_loop=False # stop inner loop
                    elif (max_r_new-max_r)>0.01 and direction_effect(max_b,dict_lkup)==False: # if change in R2 is greater than 1%, but direction of effect is wrong, use the next best predictor from the dictionary
                        log.write('\nIntermediate model '+str(i)+': '+dep+'~'+frml_snip+'+'+max_key)
                        log.write('\nAdj. R-squared:'+str(round(max_r_new,2))+'\n')
                        frml = dep+'~'+frml_snip+'+'+max_key
                        model=smf.ols(formula=frml,data=df,missing='drop').fit()
                        log.write(model.summary().tables[1].as_text())
                        log.write('\n>>> Change in adjusted R-squared>1%, but direction of effect of predictors not as assumed. The next best predictor will be used instead.\n')
                        del dict_r[max_key] # remove current predictor from dictionary
                        max_key = max(dict_r, key=dict_r.get) # find the next best predictor
                        max_r_new = dict_r[max_key] # get adjusted R2 of next best predictor
                        max_b=dict_params[max_key][0].to_dict() # get coefficient of next best predictor
                        recode_coeff(max_b) # recode the coefficients into positive or negative, then this while loop starts again
                        if force_vars: # forced vars remove them from coefficient check
                            for var in force_vars:
                                del max_b[var]
                        i+=1 # increase iterator, it then goes back to the top of this inner loop
                    elif (max_r_new-max_r)<0.01: # if change in R2 is less than 1%, stop intermediate models
                        log.write('\nIntermediate model '+str(i)+': '+dep+'~'+frml_snip+'+'+max_key)
                        log.write('\nAdj. R-squared:'+str(round(max_r_new,2))+'\n')
                        frml = dep+'~'+frml_snip+'+'+max_key
                        model=smf.ols(formula=frml,data=df,missing='drop').fit()
                        log.write(model.summary().tables[1].as_text())
                        log.write('\n>>> Change in adjusted R-squared<1%, stopping iterations.\n')
                        outer_loop=False # reset command for outer loop, this will stop the outer loop
                        inner_loop=False # break out of this loop, the outer loop will also stop because outer_loop = False
            #end of outer loop
            arcpy.AddMessage("End of outer model loop.")

            # create intermediate model found based on list of selected predictors
            frml_snip="+".join(sel_preds) # create string to be added into the formula
            frml_int=dep+'~'+frml_snip # create formula
            model_int=smf.ols(formula=frml_int,data=df,missing='drop').fit() # run intermediate model
            log.write('\n\nIntermediate model for '+dep+':\n')
            log.write(model_int.summary().as_text()) # record summary of model
            log.write('\n')

            # Check p values in intermediate model
            int_p=model_int.pvalues[1:].to_dict() #get p values into dictionary
            if force_vars: # if forced vars, remove them from p value check
                for var in force_vars:
                    del int_p[var]
            bad_monkey=dict((k, v) for k, v in list(int_p.items()) if v > 0.1) # make dictionary of all variables with p >0.1
            bad_monkey_ranked={key: rank for rank, key in enumerate(sorted(bad_monkey, key=bad_monkey.get, reverse=True), 1)} #replace p value with rank, highest p = 1 and so on
            # if all p values in intermediate model are <0.1
            if not bad_monkey:
                final_model_out(dep,model_int,frml_int,df,out_path,db,df_resid) # intermediate model=final model
                # outer_loop=False # break out of do until loop
            # if not all p values are <0.1
            failed=False # indicator if (first) removal didn't work
            if len(sel_preds)==1 and bad_monkey: # if there is only one predictor and it has a p value >0.1
                log.write('\nCheck significance of predictors in intermediate model:\n')
                log.write('\n+++ ERROR +++ The p-value of the predictor is greater than 0.1. However, the model contains only one predictor, therefore this is the best attainable model.\n')
                final_model_out(dep,model_int,frml_int,df,out_path,db,df_resid)
                # outer_loop=False # break out of do until loop

            elif len(sel_preds)>1 and bad_monkey:
                log.write('\nCheck significance of predictors in intermediate model:\n')
                # make a list with all possible combinations of the ranks from https://stackoverflow.com/questions/8371887/making-all-possible-combinations-of-a-list-in-python
                lst = list(bad_monkey_ranked.values()) # get list of ranks
                lst.sort()
                combs = list() # empty list for combinations
                for item in range(1, len(lst)+1): # makes combinations based on the number of predictors
                    els = [list(x) for x in itertools.combinations(lst, item)]
                    combs.extend(els)
                if bad_monkey.keys()==int_p.keys(): # if all predictors have p-values>0.1, then exclude combination that removes all predictors
                    del combs[-1]
                rev_bad_monkey_ranked = dict((v,k) for k,v in bad_monkey_ranked.items()) #make ranks keys and variable names values
                replace_matched_items(combs,rev_bad_monkey_ranked) # replace numbers with var names in combinations list

                j=1 # counter
                for p in combs: # run model with every combination (p) of poor variable removed, starts by removing highest p
                    test_preds=list(set(sel_preds)-set(p)) # remove p from list of predictors
                    frml_snip="+".join(test_preds)
                    frml_test=dep+'~'+frml_snip
                    model_test=smf.ols(formula=frml_test,data=df,missing='drop').fit()
                    test_p=model_test.pvalues[1:].to_dict() #get p values into dictionary
                    if force_vars: # if forced vars, remove them from p-value check
                        for var in force_vars:
                            del test_p[var]
                    trash_panda=dict((k, v) for k, v in list(test_p.items()) if v > 0.1) # add values >0.1 to dictionary
                    test_params=model_test.params[1:].to_dict() # save parameter coefficients into a dictionary
                    recode_coeff(test_params) # recode into positive/negative
                    if force_vars: # if forced vars, remove them from direction check
                        for var in force_vars:
                            del test_params[var]
                    #check results of models
                    if len(trash_panda)==0 and direction_effect(test_params,dict_lkup)==True: # if all p values<0.1 and direction of effect still meets assumption use this model
                        log.write('\n('+str(j)+') Predictor(s) removed from intermediate model: '+str(p)+'.\n')
                        log.write(model_test.summary().tables[1].as_text())
                        log.write('\n>>> All p-values<0.1, direction of effect of predictors as assumed, model accepted.\n')
                        final_model_out(dep,model_test,frml_test,df,out_path,db,df_resid)
                        break
                    elif len(trash_panda)==0 and direction_effect(test_params,dict_lkup)==False: # if all p-values<0.1, but direction of effect does not meet assumptions
                        log.write('\n('+str(j)+') Predictor(s) removed from intermediate model: '+str(p)+'.\n')
                        log.write(model_test.summary().tables[1].as_text())
                        log.write('\n>>> All p-values<0.1, but direction of effect of predictors not as assumed, model not accepted.\n')
                        j+=1
                        if j>len(combs): # this was the last item in combs and all have failed
                            failed=True # set failed marker to true
                            break # stop loop , would stop anyway?
                        else:
                            continue
                    elif len(trash_panda)>0 and direction_effect(test_params,dict_lkup)==True: # if p values are still greater than 0.1, even though direction of effect is still ok
                        log.write('\n('+str(j)+') Predictor(s) removed from intermediate model: '+str(p)+'.\n')
                        log.write(model_test.summary().tables[1].as_text())
                        log.write('\n>>> Some p-values>0.1, direction of effect of predictors as assumed, model not accepted.\n')
                        j+=1
                        if j>len(combs): # this was the last item in combs and all have failed
                            failed=True # set failed marker to true
                            break # stop loop , would stop anyway?
                        else:
                            continue
                    elif len(trash_panda)>0 and direction_effect(test_params,dict_lkup)==False: # both are wrong
                        log.write('\n('+str(j)+') Predictor(s) removed from intermediate model: '+str(p)+'.\n')
                        log.write(model_test.summary().tables[1].as_text())
                        log.write('\n>>> Some p-values>0.1 and direction of effect of predictors not as assumed, model not accepted.\n')
                        j+=1
                        if j>len(combs): # this was the last item in combs and all have failed
                            failed=True # set failed marker to true
                            break # stop loop , would stop anyway?
                        else:
                            continue

            #if that whole shebang doesn't work and not all predictors have p>0.1, make combination of all predictors
            failed_again=False # indicator if (second) removal failed
            if failed==True and bad_monkey.keys()<int_p.keys():
                log.write('\nNo acceptable model could be found by removing predictors with p-values>0.1. Now all possible combinations of predictors will be tested:\n')
                bad_monkey_all=dict(int_p) # put all p values from intermediate model into dictionary
                bad_monkey_allranked={key: rank for rank, key in enumerate(sorted(bad_monkey_all, key=bad_monkey_all.get, reverse=True), 1)} # sort by p value, i.e. highest p is worst
                lst = list(bad_monkey_allranked.values()) # make a list with all possible combinations of the ranks from https://stackoverflow.com/questions/8371887/making-all-possible-combinations-of-a-list-in-python
                lst.sort()
                combsall = list()
                for item in range(1, len(lst)+1):
                    els = [list(x) for x in itertools.combinations(lst, item)]
                    combsall.extend(els)
                del combsall[-1] # remove combination of all predictors
                rev_bad_monkey_allranked = dict((v,k) for k,v in bad_monkey_allranked.items())  #make ranks keys and variable names values
                replace_matched_items(combsall,rev_bad_monkey_allranked) # replace numbers with predictor name in list
                combsall = [c for c in combsall if c not in combs] # remove combinations already tried

                k=1 # counter
                for p in combsall: # go through every item in list to remove it from the model
                    testall_preds=list(set(sel_preds)-set(p)) # remove p from list of predictors
                    frml_snip="+".join(testall_preds)
                    frml_testall=dep+'~'+frml_snip
                    model_testall=smf.ols(formula=frml_testall,data=df,missing='drop').fit()
                    testall_p=model_testall.pvalues[1:].to_dict() #get p values into dictionary
                    if force_vars: # if forced vars, remove them from p-value check
                        for var in force_vars:
                            del testall_p[var]
                    trash_panda_all=dict((k, v) for k, v in list(testall_p.items()) if v > 0.1) # add values >0.1 to dictionary
                    testall_params=model_testall.params[1:].to_dict() # save parameter coefficients into a dictionary
                    recode_coeff(testall_params) # recode into positive/negative
                    if force_vars: # if forced vars, remove them from direction check
                        for var in force_vars:
                            del testall_params[var]

                    #check results of models
                    if len(trash_panda_all)==0 and direction_effect(testall_params,dict_lkup)==True:
                        log.write('\n('+str(k)+') Predictor(s) removed from intermediate model: '+str(p)+'.\n')
                        log.write(model_testall.summary().tables[1].as_text())
                        log.write('\n>>> All p-values<0.1, direction of effect of predictors as assumed, model accepted.\n')
                        final_model_out(dep,model_test,frml_test,df,out_path,db,df_resid)
                        break
                    elif len(trash_panda_all)==0 and direction_effect(testall_params,dict_lkup)==False:
                        log.write('\n('+str(k)+') Predictor(s) removed from intermediate model: '+str(p)+'.\n')
                        log.write(model_testall.summary().tables[1].as_text())
                        log.write('\n>>> All p-values<0.1, but direction of effect of predictors not as assumed, model not accepted.\n')
                        k+=1
                        if k>len(combs): # this was the last item in combs and all have failed
                            failed_again=True # set failed marker to true
                            break # stop loop , would stop anyway?
                        else:
                            continue
                    elif len(trash_panda_all)>0 and direction_effect(testall_params,dict_lkup)==True:
                        log.write('\n('+str(k)+') Predictor(s) removed from intermediate model: '+str(p)+'.\n')
                        log.write(model_testall.summary().tables[1].as_text())
                        log.write('\n>>> Some p-values>0.1, direction of effect of all predictors as assumed, model not accepted.\n')
                        i+=1
                        if k>len(combs): # this was the last item in combs and all have failed
                            failed_again=True # set failed marker to true
                            break # stop loop , would stop anyway?
                        else:
                            continue
                    elif len(trash_panda_all)>0 and direction_effect(testall_params,dict_lkup)==False:
                        log.write('\n('+str(k)+') Predictor(s) removed from intermediate model: '+str(p)+'.\n')
                        log.write(model_testall.summary().tables[1].as_text())
                        log.write('\n>>> Some p-values>0.1 and direction of effect of predictors not as assumed, model not accepted.\n')
                        i+=1
                        if k>len(combs): # this was the last item in combs and all have failed
                            failed_again=True # set failed marker to true
                            break # stop loop , would stop anyway?
                        else:
                            continue

            elif failed==True and bad_monkey.keys()==int_p.keys(): # if all keys in bad_monkey are in int_p the first p value check has already gone through all possible models
                log.write('''\n+++ ERROR +++ No model with p-values<0.1 can be found! Therefore this is the best attainable model. ''')
                final_model_out(dep,model_int,frml_int,df,out_path,db,df_resid)

            # if this still hasn't worked
            if failed_again==True:
                log.write('''\n+++ ERROR +++ No model with p-values<0.1 can be found! Therefore this is the best attainable model. ''')
                final_model_out(dep,model_int,frml_int,df,out_path,db,df_resid)

        # check spatial autocorrelation of residuals
        df_resid.to_csv(out_path+'\\Residuals.csv',index=False) # save residuals in csv file
        resVar=list(df_resid) # make list of variables
        try:
            resVar.remove('siteID_INT')
        except ValueError:
            pass
        try:
            resVar.remove('siteID')
        except ValueError:
            pass

        if arcpy.Exists(out_fgdb+'\\modelResid'): # delete file if exists
            arcpy.Delete_management(out_fgdb+'\\modelResid')
        arcpy.TableToTable_conversion (out_path+'\\Residuals.csv', out_fgdb, "modelResid") # save csv to out_fgdb
        arcpy.JoinField_management (depVar, "siteID_INT", out_fgdb+'\\modelResid', "siteID_INT") # join to monitoring sites

        df=pd.DataFrame()
        for var in resVar:
            arcpy.SpatialAutocorrelation_stats (depVar, var, "NO_REPORT","INVERSE_DISTANCE", "EUCLIDEAN_DISTANCE", "NONE")
            messages=arcpy.GetMessages()
            sIndex = messages.find('''Moran's Index:''')
            eIndex = messages.find('''Distance measured in Meters''') # what if unit isn't metres?
            moransSum = messages[sIndex : eIndex]
            res_dict=dict(item.split(':') for item in moransSum.rstrip('\n').split('\n'))
            res_dict.update({'VarName':str(var)})
            df = df.append(res_dict,ignore_index=True)

        log.write("\n\nSpatial autocorrelation of residuals\n")
        log.write(df.to_string())


        #-------------------------------------------------------------------------------
        time2=time.clock()
        m, s = divmod((time2-time1), 60)
        h, m = divmod(m, 60)
        log.write(time.strftime("\nEnding statistical analysis: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        arcpy.AddMessage(("Elapsed time statistical analysis: %d:%02d:%02d" % (h, m, s)))
        log.write("\n\nElapsed time statistical analysis: %d:%02d:%02d" % (h, m, s))

        self.mark3.SetLabel(mark_done)
        self.finBtn.Enable()
        self.model_btn.Disable()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait

    def onFin(self, event):
        """"""
        wait = wx.BusyCursor()
        arcpy.AddMessage((time.strftime("Finished: %A %d %b %Y %H:%M:%S", time.localtime())))
        log.write(time.strftime("\n\nModel - End Time: %A %d %b %Y %H:%M:%S", time.localtime()))
        log.write(time.strftime("\n\nFinished by user: %A %d %b %Y %H:%M:%S", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('panel4','stop',datetime())''')
        conn.executescript("ALTER TABLE timings \
                            ADD elapsedTime REAL NULL; \
                            CREATE TABLE tempTimings AS \
                            SELECT t1.Panel \
	                           ,t1.Step \
	                           ,t1.vTime \
	                           ,strftime('%H:%M:%S',CAST ((julianday(t1.vTime) - julianday(t2.vTime)) AS REAL),'12:00:00') As ElapsedTime \
                            FROM timings AS t1 \
                            LEFT OUTER JOIN timings AS t2 \
                            ON t1.rowid = t2.rowid+1; \
                            UPDATE timings \
                            SET elapsedTime = (SELECT ElapsedTime \
				                                FROM tempTimings \
				                                WHERE timings.Panel=tempTimings.Panel AND timings.Step=tempTimings.Step AND timings.vTime=tempTimings.vTime); \
                            DROP TABLE tempTimings;")
        db.commit()
        eltimen = time.clock()
        m, s = divmod((eltimen-eltime0), 60)
        h, m = divmod(m, 60)
        log.write("\n\nTotal time: %d:%02d:%02d" % (h, m, s))
        log.close()
        db.close()
        logging.shutdown()
        del wait
        self.Parent.Close()

    def onCanc1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Cancel Wizard"""
        dlg = wx.MessageDialog(None, "Cancelling the wizard will delete all progress. Do you wish to cancel the wizard?",'Cancel',wx.YES_NO | wx.ICON_WARNING)
        result = dlg.ShowModal()
        if result == wx.ID_YES:
            arcpy.AddMessage((time.strftime("Cancelled by user: %A %d %b %Y %H:%M:%S", time.localtime())))
            try:
                log.close()
                logging.shutdown()
                db.close()
                shutil.rmtree(out_path)
            except NameError as e:
                arcpy.AddMessage(e)
            except OSError as e:
                arcpy.AddMessage(e)
            del wait
            self.Parent.Close()
        else:
            self.Parent.statusbar.SetStatusText('Ready')
            del wait

#------------------------------------------------------------------------------
class HtmlViewer(wx.Frame):
    """View help files"""
    def __init__(self, parent, filepath):
        wx.Frame.__init__(self, None, title="Help",size = (600, 450))

        self.html_view = wx.html2.WebView.New(self)
        self.html_view.Bind(wx.html2.EVT_WEBVIEW_NEWWINDOW, self.onNewWindow) #to open links from html
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.html_view, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.html_view.LoadURL(filepath)

        icon = wx.Icon(wx.ArtProvider.GetBitmap(wx.ART_HELP, wx.ART_FRAME_ICON, (16, 16)))
        self.SetIcon(icon)

    def onNewWindow(self, event):
        url = event.GetURL()
        webbrowser.open(url)
#-------------------------------------------------------------------------------
class MainFrame(wx.Frame):
    """"""

    def __init__(self):
        """Constructor"""
        #displaySize= wx.DisplaySize()
        wx.Frame.__init__(self, None, title="Build LUR")#,size=(displaySize[0]/2, displaySize[1]/2))

        self.panel1 = WizardPanel1(self)
        # self.panel1.Hide()
        self.panel2 = WizardPanel2(self)
        self.panel2.Hide()
        self.panel3 = WizardPanel3(self)
        self.panel3.Hide()
        self.panel3A = WizardPanel3A(self)
        self.panel3A.Hide()
        self.panel3B = WizardPanel3B(self)
        self.panel3B.Hide()
        self.panel3C = WizardPanel3C(self)
        self.panel3C.Hide()
        self.panel3D = WizardPanel3D(self)
        self.panel3D.Hide()
        self.panel3E = WizardPanel3E(self)
        self.panel3E.Hide()
        self.panel3F = WizardPanel3F(self)
        self.panel3F.Hide()
        self.panel3G = WizardPanel3G(self)
        self.panel3G.Hide()
        self.panel4 = WizardPanel4(self)
        self.panel4.Hide()


        self.mainsizer = wx.BoxSizer(wx.VERTICAL)
        self.mainsizer.Add(self.panel1, 1, wx.EXPAND)

        # add status bar
        self.statusbar = self.CreateStatusBar(1)
        self.statusbar.SetStatusText('Ready')

        self.SetSizer(self.mainsizer)
        self.Layout()
        self.Fit()
        self.Show()
        self.Centre()

        image = wx.Image(curPath+'\\Documentation\\Images\\magic_wand4_resize.png', wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        icon = wx.Icon()
        icon.CopyFromBitmap(image)
        self.SetIcon(icon)




if __name__ == "__main__":
    app = wx.App(False)
    frame = MainFrame()
    # wx.lib.inspection.InspectionTool().Show()
    app.MainLoop()
