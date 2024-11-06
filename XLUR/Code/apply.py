import wx
import wx.html2
import webbrowser
from wx.lib.masked import NumCtrl
from wx.lib.wordwrap import wordwrap
from wx.lib.agw import ultimatelistctrl as ULC
import wx.lib.inspection
import arcpy
from arcpy import env
import sys
import os
import logging
import sqlite3
import time
import pandas as pd
try:
    from pandas.tools.plotting import table
except ImportError:
    from pandas.plotting import table
import numpy as np
import itertools
import math
import string
import random
import csv
time.clock = time.time
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
        outputFieldName.aliasName = fldsNamesDict[field]#aslo need to change alias name?
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

#------------------------------------------------------------------------------
# global variables, lists, dictionaries
#------------------------------------------------------------------------------
# current directory
curPath = os.path.dirname(os.path.abspath(__file__))
# tick marks
mark_empty = ' '
mark_done = '\u2713'

#list of ASCII character codes
ascii_lower = list(range(97,122))
ascii_upper = list(range(65,90))
ascii_char = ascii_lower + ascii_upper + [8,9,127,314,316] #add delete etc

#list of ASCII char, number and underscore
ascii_num = list(range(48,58))
ascii_underscore = [95]
ascii_all = ascii_num + ascii_underscore + ascii_char
#------------------------------------------------------------------------------
env.overwriteOutput = True
#------------------------------------------------------------------------------
# Time code
eltime0 = time.clock()
time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
time_strt_sql = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
time_start = time.localtime()
arcpy.AddMessage((time.strftime("Start Time: %A %d %b %Y %H:%M:%S", time_start)))
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
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

        header0 = wx.StaticText(self,-1,label="Set Output Name")
        header0.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header0, pos=(2, 0), flag=wx.ALL, border=10)

        help_btn0 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn0.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn0.SetForegroundColour(wx.Colour(0,0,255))
        help_btn0.Bind(wx.EVT_BUTTON, self.onHlp0)
        self.sizer.Add(help_btn0, pos=(2,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text1 = wx.StaticText(self,-1,label="Output Name")
        self.sizer.Add(text1, pos=(3, 0), flag=wx.ALL, border=10)

        self.tc0 = wx.TextCtrl(self, -1)
        self.tc0.SetMaxLength(10)
        self.tc0.SetToolTip(wx.ToolTip('Enter a name for the output feature class'))
        self.tc0.Bind(wx.EVT_CHAR,self.onChar0)
        self.sizer.Add(self.tc0, pos=(3, 1), span=(1,2), flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=5)

        self.enter_btn0 = wx.Button(self, label="Enter")
        self.enter_btn0.Bind(wx.EVT_BUTTON, self.onEnt0)
        self.sizer.Add(self.enter_btn0, pos=(3,3), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)

        self.mark0 = wx.StaticText(self,-1,label=mark_empty)
        self.mark0.SetForegroundColour((0,255,0))
        self.mark0.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark0, pos=(3,4), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(4, 0), span=(1, 6),flag=wx.EXPAND|wx.BOTTOM, border=5)

        header1 = wx.StaticText(self,-1,label="Set Data Source")
        header1.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header1, pos=(5, 0), flag=wx.ALL, border=10)

        help_btn1 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn1.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn1.SetForegroundColour(wx.Colour(0,0,255))
        help_btn1.Bind(wx.EVT_BUTTON, self.onHlp1)
        self.sizer.Add(help_btn1, pos=(5,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text2 = wx.StaticText(self,-1,label="LUR File Geodatabase")
        self.sizer.Add(text2, pos=(6, 0), flag=wx.ALL, border=10)

        self.browse_btn1 = wx.DirPickerCtrl(self)
        self.browse_btn1.Bind(wx.EVT_DIRPICKER_CHANGED, self.onBrw1)
        self.browse_btn1.SetToolTip(wx.ToolTip('Select the File Geodatabase containing the LUR data'))
        self.sizer.Add(self.browse_btn1, pos=(6,1), span=(1, 4), flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.browse_btn1.Disable()

        self.mark1 = wx.StaticText(self,-1,label=mark_empty)
        self.mark1.SetForegroundColour((0,255,0))
        self.mark1.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark1, pos=(6,5), flag=wx.TOP|wx.BOTTOM|wx.RIGHT, border=5)

        text4 = wx.StaticText(self, label="LUR SQLite Database")
        self.sizer.Add(text4, pos=(7, 0), flag=wx.ALL, border=10)

        self.browse_btn2 = wx.FilePickerCtrl(self)
        self.browse_btn2.Bind(wx.EVT_FILEPICKER_CHANGED, self.onBrw2)
        self.browse_btn2.SetToolTip(wx.ToolTip('Select the LUR SQLite database that contains the LUR model'))
        self.sizer.Add(self.browse_btn2, pos=(7, 1), span=(1, 4), flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.browse_btn2.Disable()

        self.mark2 = wx.StaticText(self,-1,label=mark_empty)
        self.mark2.SetForegroundColour((0,255,0))
        self.mark2.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark2, pos=(7,5), flag=wx.TOP|wx.BOTTOM|wx.RIGHT, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(8, 0), span=(1, 6),flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=5)

        header2 = wx.StaticText(self,-1,label="Set Model")
        header2.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header2, pos=(9, 0), flag=wx.ALL, border=10)

        help_btn2 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn2.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn2.SetForegroundColour(wx.Colour(0,0,255))
        help_btn2.Bind(wx.EVT_BUTTON, self.onHlp2)
        self.sizer.Add(help_btn2, pos=(9,1), flag=wx.TOP|wx.BOTTOM, border=5)

        text5 = wx.StaticText(self,-1,label="Select LUR Model(s)")
        self.sizer.Add(text5, pos=(10, 0), flag=wx.ALL, border=10)

        self.chklbx1 = wx.CheckListBox(self,-1,choices=[])
        self.chklbx1.SetToolTip(wx.ToolTip('Select one or more LUR models to be used for prediction'))
        self.sizer.Add(self.chklbx1, pos=(10,1), span=(1,3), flag=wx.TOP|wx.LEFT|wx.BOTTOM|wx.EXPAND, border=5)
        self.chklbx1.Disable()

        self.sel_btn1 = wx.Button(self, label="Select")
        self.sel_btn1.Bind(wx.EVT_BUTTON, self.onSel1)
        self.sizer.Add(self.sel_btn1, pos=(10,4), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)
        self.sel_btn1.Disable()

        self.mark3 = wx.StaticText(self,-1,label=mark_empty)
        self.mark3.SetForegroundColour((0,255,0))
        self.mark3.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark3, pos=(10,5), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(11, 0), span=(1, 6),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        # add Next button
        self.nextBtn1 = wx.Button(self, label="Next >")
        self.nextBtn1.Bind(wx.EVT_BUTTON, self.onNext)
        self.sizer.Add(self.nextBtn1, pos=(12,4),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)
        self.nextBtn1.Disable()

        # add cancel button
        self.cancBtn1 = wx.Button(self, label="Cancel")
        self.cancBtn1.Bind(wx.EVT_BUTTON, self.onCanc)
        self.cancBtn1.SetToolTip(wx.ToolTip('Cancel'))
        self.sizer.Add(self.cancBtn1, pos=(12,5),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        self.sizer.AddGrowableCol(1)
        self.sizer.AddGrowableRow(10)
        self.SetSizer(self.sizer)
        self.Layout()
        self.Fit()

    def onHlp0(self,event):
        """Help window for setting ouput name"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p1_SetOutputName.html")
        htmlViewerInstance.Show()

    def onChar0(self,event):
        keycode = event.GetKeyCode()
        if keycode in ascii_all: #restrict input
            event.Skip()

    def onEnt0(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """enter output name"""
        global out_name
        out_name = self.tc0.GetValue()
        if not out_name:
            wx.MessageBox('Please enter an output name','Error',wx.OK|wx.ICON_ERROR)
        elif out_name[0].isalpha()==False:
            wx.MessageBox('The output name must start with a letter','Error',wx.OK|wx.ICON_ERROR)
        else:
            global out_recp
            out_recp = out_name+'_receptors' # set name of receptor fc
            self.Parent.statusbar.SetStatusText('Ready')
            del wait
            self.browse_btn1.Enable()
            self.tc0.Disable()
            self.enter_btn0.Disable()
            self.mark0.SetLabel(mark_done) # change tick mark to done

    def onHlp1(self,event): #probably need to change this to MessageDialog to make it look nicer
        """Help window for selecting LUR data"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p1_SetDataSource.html")
        htmlViewerInstance.Show()

    def onBrw1(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Get input fgdb path"""
        global lur_fgdb
        lur_fgdb = self.browse_btn1.GetPath()
        if lur_fgdb[-4:]!='.gdb':
            wx.MessageBox('Invalid selection.\nPlease select a File Geodatabase.','Error',wx.OK|wx.ICON_ERROR)
        elif not arcpy.Exists(lur_fgdb+'\\LURdata'):
            wx.MessageBox('The LURdata feature dataset is missing from the selected File Geodatabase.','Error',wx.OK|wx.ICON_ERROR)
        elif not arcpy.Exists(lur_fgdb+"\\LURdata\\studyArea"):
            wx.MessageBox('The studyArea feature class is missing from LURdata.','Error',wx.OK|wx.ICON_ERROR)
        else:
            global lur_fds
            lur_fds = lur_fgdb+"\\LURdata"
            self.browse_btn2.Enable()
            self.mark1.SetLabel(mark_done) # change tick mark to done
            self.Parent.statusbar.SetStatusText('Ready') # change status bar
            self.browse_btn1.Disable()
            del wait

    def onBrw2(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Get sqlite path """
        global lur_sql_path
        lur_sql_path = self.browse_btn2.GetPath()
        arcpy.AddMessage(lur_sql_path)
        if lur_sql_path[-7:]!='.sqlite':
            wx.MessageBox('Invalid selection.\nPlease select the SQLite database.','Error',wx.OK|wx.ICON_ERROR)
        else:
            #get outer folder
            global prj_folder
            prj_folder=lur_sql_path[0:lur_sql_path.rfind("\\")]
            env.workspace = lur_fgdb
            # make output folder
            global out_folder
            folder_name = out_name+"_"+time_str
            out_folder = os.path.join(prj_folder,folder_name)
            os.mkdir(out_folder) # create output folder in project folder
            #make output fgdb
            arcpy.CreateFileGDB_management(out_folder, out_name)
            global out_fgdb
            out_fgdb = out_folder+"\\"+out_name+".gdb"
            #make output fds
            global out_fds
            out_fds = out_fgdb+"\\LURdata"
            arcpy.CreateFeatureDataset_management(out_fgdb, 'LURdata', arcpy.Describe(lur_fds).SpatialReference) # create feature dataset
            # make log
            global log
            log = open(out_folder+'\\Apply_LOG.txt', 'w')# create log file
            log.write(time.strftime("Start Time: %A %d %b %Y %H:%M:%S", time_start))
            log.write(time.strftime("\n\nSettings - Start Time: %A %d %b %Y %H:%M:%S", time_start))
            log.write('\n\nLUR Source Data File Geodatabase: '+lur_fgdb)
            log.write('\nSQLite Database: '+lur_sql_path)
            #create error file
            logging.basicConfig(filename=out_folder+'\\GOTCHA.log',filemode='w',level=logging.DEBUG,format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',datefmt="%Y-%m-%d %H:%M:%S")
            sys.excepthook = exception_hook
            # Connect to LUR sqlite Database
            global lur_db
            lur_db = sqlite3.connect(lur_sql_path)
            # cursor object
            global lur_conn
            lur_conn = lur_db.cursor()
            # get models
            model_list = lur_conn.execute('''SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'MODEL_%';''').fetchall()
            lur_db.commit()
            arcpy.AddMessage(('\n model_list size: '+str(sys.getsizeof(model_list))))
            if not model_list:
                wx.MessageBox('No models were found in the SQLite database.','Error',wx.OK|wx.ICON_ERROR)
            #append to CheckListBox
            self.chklbx1.Clear()
            for item in model_list:
                self.chklbx1.Append(str(item[0]))
            #set scratch
            global TEMP
            TEMP = os.getenv("TEMP") # this is a folder
            env.scratchWorkspace = TEMP
            env.scratchWorkspace = env.scratchGDB
            tmpFC = arcpy.CreateScratchName(workspace=arcpy.env.scratchGDB)

            self.chklbx1.Enable()
            self.sel_btn1.Enable()
            self.mark2.SetLabel(mark_done)# change tick mark to done
            self.Parent.statusbar.SetStatusText('Ready') # change status bar
            del wait
            self.browse_btn2.Disable()
            log.flush()

    def onHlp2(self,event):
        """Help window for selecting model"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p1_SetModel.html")
        htmlViewerInstance.Show()

    def onSel1(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """select LUR models"""
        global models
        models = list(self.chklbx1.GetCheckedStrings())
        if not models:
            wx.MessageBox('Select at least one model','Error',wx.OK|wx.ICON_ERROR)
        log.write('\n\nLUR Model(s) selected: '+str(models))
        #check that model data is there
        lur_conn.execute("DROP TABLE IF EXISTS predictor_check;")
        qry1 = "CREATE TABLE predictor_check AS "
        for item in models:
            qry1 +="SELECT X FROM "+item+" UNION "
        qry1 = qry1[:-7]
        lur_conn.execute(qry1)
        lur_conn.execute("DELETE FROM predictor_check WHERE X='Intercept' OR X='p_XCOORD' OR X='p_YCOORD';")
        pred_sql = lur_conn.execute("SELECT X FROM predictor_check").fetchall()
        lur_db.commit()
        global preds
        preds = list()
        for p in pred_sql:
            arcpy.AddMessage((str(p[0])))
            preds.append(str(p[0]))

        p1and2 = list() # extract first two parts of variable name
        for p in preds:
            p1and2.append(p[:p.find('_',3)])

        p1and2 = list(set(p1and2)) # only keep unique ones
        #check if raster data and check out extension, warning if no license
        for item in p1and2:
            if item[0:2]=='pG':
                if arcpy.CheckExtension("Spatial") == "Available":
                    arcpy.CheckOutExtension("Spatial")
                else:
                    wx.MessageBox('The Spatial Analyst license is unavailable. Please check your licenses.','Error',wx.OK|wx.ICON_ERROR)

                    del wait

        #check that files exist and for A,B,C that cat field exists
        for item in p1and2:
            if not arcpy.Exists(lur_fgdb+"\\"+item):
                log.write('\n +++ERROR+++ '+item+' not found in LUR file geodatabase.')
                wx.MessageBox(item+' not found in LUR file geodatabase. Please check your data.','Error',wx.OK|wx.ICON_ERROR)
                del wait
                self.Parent.Close()
            elif item[0:2] in ['pA','pB','pC']:
                fieldnm = item[0:2]+'_cat'
                if not FieldExist(lur_fgdb+"\\"+item,fieldnm):
                    log.write('\n +++ERROR+++ '+fieldnm+' field not found in '+item+'.')
                    wx.MessageBox(fieldnm+' field not found in '+item+'. Please check your data.','Error',wx.OK|wx.ICON_ERROR)
                    del wait
                    self.Parent.Close()

        #check that values exist in cat field for A,B,C or that field exists for D,E,F
        p1to3 = list() # extract first three parts of variable name
        for p in preds:
            if p[0:2] in ['pA','pB','pC']:
                p1to3.append(p[:p.rfind('_',0,len(p)-4)])
            else:
                p1to3.append(p[:p.rfind('_')])
        p1to3 = list(set(p1to3))

        for item in p1to3:
            if item[0:2] in ['pA','pB','pC']:
                fieldnm = item[0:2]+'_cat'
                fc = item[0:item.find('_',4)]
                vals = unique_values(fc, fieldnm)
                p3 = item[item.rfind('_')+1:]
                if p3 not in vals:
                    log.write('\n +++ERROR+++ '+p3+' not in '+fieldnm+'.')
                    wx.MessageBox(p3+' not in '+fieldnm+'. Please check your data.','Error',wx.OK|wx.ICON_ERROR)
                    del wait
                    self.Parent.Close()
            if item[0:2] in ['pD','pE','pF']:
                fc = item[0:item.find('_',4)]
                p3 = item[item.rfind('_')+1:]
                if p3!='none' and not FieldExist(fc,p3):
                    log.write('\n +++ERROR+++ '+p3+' field not in '+fc+'.')
                    wx.MessageBox(p3+' field not in '+fc+'. Please check your data.','Error',wx.OK|wx.ICON_ERROR)
                    del wait
                    self.Parent.Close()

        self.mark3.SetLabel(mark_done) # change tick mark to done
        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        self.chklbx1.Disable()
        self.sel_btn1.Disable()
        self.nextBtn1.Enable()
        log.flush()

    def onNext(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Next Page"""
        # copy feature classes into output fds/fgdb
        p1and2 = list() # extract first two parts of variable name
        for p in preds:
            p1and2.append(p[:p.find('_',3)])
        p1and2 = list(set(p1and2)) # only keep unique ones
        for item in p1and2:
            if item[0:2]=='pG':
                arcpy.CopyRaster_management(lur_fgdb+"\\"+item,out_fgdb+"\\"+item)
            else:
                arcpy.FeatureClassToFeatureClass_conversion(lur_fds+"\\"+item,out_fds,item)
        # copy study area
        arcpy.FeatureClassToFeatureClass_conversion(lur_fds+"\\studyArea",out_fds,"studyArea")
        # create new db
        global db
        db = sqlite3.connect(out_folder+"\\OutputSqlDB.sqlite")
        # cursor object
        global conn
        conn = db.cursor()
        #turn journal off, i.e. no rollback
        conn.execute('''PRAGMA journal_mode=OFF;''')
        conn.execute('''PRAGMA auto_vacuum=FULL;''')
        conn.execute('''PRAGMA SQLITE_DEFAULT_CACHE_SIZE=-32000;''')
        conn.execute('''PRAGMA SQLITE_DEFAULT_SYNCHRONOUS=0;''') # will lead to corruption during power loss or OS crash, but not app crash!
        conn.execute('''PRAGMA SQLITE_THREADSAFE=0;''') # doesn't seem to reset
        db.commit()
        arcpy.AddMessage ((conn.execute('PRAGMA compile_options').fetchall()))
        #create timings table
        conn.execute('''CREATE TABLE timings (Panel Int, Step Varchar, vTime DateTime, elapsedTime REAL)''')
        conn.execute("INSERT INTO timings VALUES('apply_panel1','start','"+time_strt_sql+"',NULL)")
        db.commit()
        # copy models to SQLITE database
        conn.execute('''ATTACH "'''+lur_sql_path+'''" AS srcdb''')
        for m in models:
            conn.execute("CREATE TABLE "+str(m)+" AS SELECT * FROM srcdb."+str(m)+";")
            db.commit()
        for item in p1and2:
            if item[0:2] in ['pA','pB','pC']:
                conn.execute("CREATE TABLE "+str(item)+" AS SELECT * FROM srcdb."+str(item)+";")
                db.commit()
        conn.execute("DETACH DATABASE srcdb")
        db.commit()
        # make predictor check table
        qry1 = "CREATE TABLE predictor_check AS "
        for item in models:
            qry1 +="SELECT X FROM "+item+" UNION "
        qry1 = qry1[:-7]
        conn.execute(qry1)
        conn.execute("DELETE FROM predictor_check WHERE X='Intercept' OR X='p_XCOORD' OR X='p_YCOORD';")
        db.commit()

        log.write(time.strftime("\n\nSettings - End Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        log.write(time.strftime("\nReceptors - Start Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('apply_panel1','stop',datetime(),NULL)''')
        conn.execute('''INSERT INTO timings VALUES('apply_panel2','start',datetime(),NULL)''')
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

    def onCanc(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Cancel Wizard"""
        dlg = wx.MessageDialog(None, "Cancelling the wizard will delete all progress. Do you wish to cancel the wizard?",'Cancel',wx.YES_NO | wx.ICON_WARNING)
        result = dlg.ShowModal()
        if result == wx.ID_YES:
            arcpy.AddMessage((time.strftime("Aborted by user: %A %d %b %Y %H:%M:%S", time.localtime())))
            try:
                log.close()
                logging.shutdown()
                db.close()
                os.remove(out_folder)
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

        title2 = wx.StaticText(self, -1, "Receptors")
        title2.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(title2, pos=(0,0),flag=wx.TOP|wx.LEFT|wx.BOTTOM, border=10)

        line1 = wx.StaticLine(self)
        self.sizer.Add(line1, pos=(1, 0), span=(1, 5),flag=wx.EXPAND|wx.BOTTOM, border=5)

        header1 = wx.StaticText(self,-1,label="Select Receptor Points")
        header1.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header1, pos=(2, 0), span=(1,2),flag=wx.ALL, border=10)

        help_btn1 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn1.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn1.SetForegroundColour(wx.Colour(0,0,255))
        help_btn1.Bind(wx.EVT_BUTTON, self.onHlp1)
        self.sizer.Add(help_btn1, pos=(2,2), flag=wx.TOP|wx.BOTTOM, border=5)

        text2 = wx.StaticText(self,-1,label="A. From Feature Class")
        self.sizer.Add(text2, pos=(3, 0), flag=wx.ALL, border=10)

        add_btnA = wx.Button(self, label="Select")
        add_btnA.Bind(wx.EVT_BUTTON, self.onSelA)
        self.sizer.Add(add_btnA, pos=(3,1), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)

        text3 = wx.StaticText(self, label="B. Regular Points")
        self.sizer.Add(text3, pos=(4, 0), flag=wx.ALL, border=10)

        add_btnB = wx.Button(self, label="Select")
        add_btnB.Bind(wx.EVT_BUTTON, self.onSelB)
        self.sizer.Add(add_btnB, pos=(4,1), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)

        text4 = wx.StaticText(self, label="C. Random Points")
        self.sizer.Add(text4, pos=(5, 0), flag=wx.ALL, border=10)

        add_btnC = wx.Button(self, label="Select")
        add_btnC.Bind(wx.EVT_BUTTON, self.onSelC)
        self.sizer.Add(add_btnC, pos=(5,1), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(6, 0), span=(1, 5),flag=wx.EXPAND|wx.BOTTOM, border=5)

        # add cancel button
        self.cancBtn1 = wx.Button(self, label="Cancel")
        self.cancBtn1.Bind(wx.EVT_BUTTON, self.onCanc1)
        self.cancBtn1.SetToolTip(wx.ToolTip('Cancel'))
        self.sizer.Add(self.cancBtn1, pos=(7,2),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        self.sizer.AddGrowableCol(1)
        self.SetSizer(self.sizer)
        self.Layout()
        self.Fit()

    def onHlp1(self,event):
        """Help window for setting predictors"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p2_SelectReceptors.html")
        htmlViewerInstance.Show()

    def onSelA(self, event):
        """open panel2A"""
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel2A,1,wx.EXPAND)
        newsize = self.Parent.panel2A.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel2A.Show()
        log.write(time.strftime("\nReceptors from Feature Class - Start Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('apply_panel2','stop',datetime(),NULL)''')
        conn.execute('''INSERT INTO timings VALUES('apply_panel2A','start',datetime(),NULL)''')
        db.commit()
        log.flush()

    def onSelB(self, event):
        """open panel2B"""
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel2B,1,wx.EXPAND)
        newsize = self.Parent.panel2B.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel2B.Show()
        log.write(time.strftime("\nReceptors from Regular Points - Start Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('apply_panel2','stop',datetime(),NULL)''')
        conn.execute('''INSERT INTO timings VALUES('apply_panel2B','start',datetime(),NULL)''')
        db.commit()
        log.flush()

    def onSelC(self, event):
        """open panel2C"""
        self.Hide()
        self.Parent.mainsizer.Clear()
        self.Parent.mainsizer.Add(self.Parent.panel2C,1,wx.EXPAND)
        newsize = self.Parent.panel2C.sizer.GetSize()
        self.Parent.mainsizer.Layout()
        self.Parent.mainsizer.SetItemMinSize(self.Parent,newsize[0],newsize[1])
        self.Parent.SetSizer(self.Parent.mainsizer)
        self.Parent.Layout()
        self.Parent.SetClientSize(newsize[0],newsize[1])
        self.Parent.PostSizeEventToParent()
        self.Parent.panel2C.Show()
        log.write(time.strftime("\nReceptors from Random Points - Start Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('apply_panel2','stop',datetime(),NULL)''')
        conn.execute('''INSERT INTO timings VALUES('apply_panel2C','start',datetime(),NULL)''')
        db.commit()
        log.flush()

    def onCanc1(self, event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Cancel Wizard"""
        dlg = wx.MessageDialog(None, "Cancelling the wizard will delete all progress. Do you wish to cancel the wizard?",'Cancel',wx.YES_NO | wx.ICON_WARNING)
        result = dlg.ShowModal()
        if result == wx.ID_YES:
            arcpy.AddMessage((time.strftime("Aborted by user: %A %d %b %Y %H:%M:%S", time.localtime())))
            try:
                log.close()
                logging.shutdown()
                db.close()
                os.remove(out_folder)
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
class WizardPanel2A(wx.Panel):
    """Page 2A"""

    def __init__(self, parent, title=None):
        """Constructor"""
        wx.Panel.__init__(self, parent)
        wx.ToolTip.Enable(True)
        self.sizer = wx.GridBagSizer(0,0)

        title2 = wx.StaticText(self, -1, "Receptors from Feature Class")
        title2.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(title2, pos=(0,0),flag=wx.TOP|wx.LEFT|wx.BOTTOM, border=10)

        line1 = wx.StaticLine(self)
        self.sizer.Add(line1, pos=(1, 0), span=(1, 6),flag=wx.EXPAND|wx.BOTTOM, border=5)

        header1 = wx.StaticText(self,-1,label="Set Data Source")
        header1.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header1, pos=(2, 0), span=(1,2),flag=wx.ALL, border=10)

        help_btn1 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn1.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn1.SetForegroundColour(wx.Colour(0,0,255))
        help_btn1.Bind(wx.EVT_BUTTON, self.onHlp1)
        self.sizer.Add(help_btn1, pos=(2,2), flag=wx.TOP|wx.BOTTOM, border=5)

        text2 = wx.StaticText(self,-1,label="Select File Geodatabase")
        self.sizer.Add(text2, pos=(3, 0), flag=wx.ALL, border=10)

        self.browse_btn1 = wx.DirPickerCtrl(self)
        self.browse_btn1.Bind(wx.EVT_DIRPICKER_CHANGED, self.onBrw1)
        self.browse_btn1.SetToolTip(wx.ToolTip('Select the File Geodatabase containing the receptor point feature class'))
        self.sizer.Add(self.browse_btn1, pos=(3,1), span=(1, 4), flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)

        self.mark1 = wx.StaticText(self,-1,label=mark_empty)
        self.mark1.SetForegroundColour((0,255,0))
        self.mark1.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark1, pos=(3,5), flag=wx.TOP|wx.BOTTOM|wx.RIGHT, border=5)

        text2 = wx.StaticText(self,-1,label="Receptor Points")
        self.sizer.Add(text2, pos=(4, 0), flag=wx.ALL, border=10)

        self.cb1 = wx.ComboBox(self,choices=[],style=wx.CB_DROPDOWN)
        self.cb1.SetToolTip(wx.ToolTip('From the dropdown list select the feature class containing the receptor points'))
        self.sizer.Add(self.cb1, pos=(4, 1), span=(1,3),flag=wx.TOP|wx.RIGHT|wx.BOTTOM|wx.EXPAND, border=5)
        self.cb1.Bind(wx.EVT_COMBOBOX, self.onCb1)
        self.cb1.Disable()

        self.mark2 = wx.StaticText(self,-1,label=mark_empty)
        self.mark2.SetForegroundColour((0,255,0))
        self.mark2.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark2, pos=(4,4), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(5, 0), span=(1, 6),flag=wx.EXPAND|wx.BOTTOM, border=5)

        # add previous Button
        self.backBtn = wx.Button(self, label="< Back")
        self.backBtn.Bind(wx.EVT_BUTTON, self.onBack)
        self.sizer.Add(self.backBtn, pos=(6,3),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        # add next button
        self.nextBtn = wx.Button(self, label="Next >")
        self.nextBtn.Bind(wx.EVT_BUTTON, self.onNext)
        self.sizer.Add(self.nextBtn, pos=(6,4),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)
        self.nextBtn.Disable()

        # add cancel button
        self.cancBtn1 = wx.Button(self, label="Cancel")
        self.cancBtn1.Bind(wx.EVT_BUTTON, self.onCanc1)
        self.cancBtn1.SetToolTip(wx.ToolTip('Cancel'))
        self.sizer.Add(self.cancBtn1, pos=(6,5),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        self.sizer.AddGrowableCol(1)
        self.SetSizer(self.sizer)
        self.Layout()
        self.Fit()

    def onHlp1(self,event):
        """Help window for setting predictors"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p2A_SetInput.html")
        htmlViewerInstance.Show()

    def onBrw1(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """select fgdb that holds receptors"""
        global recept_fgdb
        recept_fgdb = self.browse_btn1.GetPath()
        if recept_fgdb[-4:]!='.gdb':
            wx.MessageBox('Invalid selection.\nPlease select a File Geodatabase.','Error',wx.OK|wx.ICON_ERROR)
        else:
            env.workspace = recept_fgdb
            points = list()
            datasets = arcpy.ListDatasets(feature_type='feature')
            datasets = [''] + datasets if datasets is not None else []
            for ds in datasets:
                for fc in arcpy.ListFeatureClasses('','Point',feature_dataset=ds):
                    path = os.path.join(ds, fc)
                    points.append(path)
            if not points:
                wx.MessageBox('Invalid selection.\nThe File Geodatabase does not contain a point feature class.','Error',wx.OK|wx.ICON_ERROR)
            else:
                points.sort()
                self.cb1.SetValue('')
                self.cb1.Clear()
                self.cb1.Append(points)
                self.cb1.Enable()
                self.mark1.SetLabel(mark_done) # change tick mark to done
                self.Parent.statusbar.SetStatusText('Ready') # change status bar
                del wait
                self.browse_btn1.Disable()

    def onCb1(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """select point fc"""
        global fc_recp
        fc_recp = self.cb1.GetValue()
        # check for spatial duplicates
        stop = 0
        statDict = {} # create an empty dictionary
        searchRows = arcpy.da.SearchCursor(fc_recp, ["SHAPE@WKT","OID@"]) # use data access search cursor to get OID and geometry
        for searchRow in searchRows:
            geomValue,oidValue = searchRow
            if geomValue in statDict:
                wx.MessageBox('Spatial duplicates found','Error',wx.OK|wx.ICON_ERROR)
                stop = 1
                break
            else:
                statDict[geomValue] = [oidValue]

        # check it contains data
        num_row=int(arcpy.GetCount_management(fc_recp).getOutput(0))
        if num_row==0:
            wx.MessageBox('The selected feature class is empty','Error',wx.OK|wx.ICON_ERROR)

        # Check that all points are within study area
        arcpy.MakeFeatureLayer_management(fc_recp, out_fds+"\\recp_lyr")
        arcpy.SelectLayerByLocation_management(out_fds+"\\recp_lyr", 'WITHIN', out_fds+"\\studyArea")
        #count features
        selcount=int(arcpy.GetCount_management(out_fds+"\\recp_lyr").getOutput(0))
        if num_row>selcount:
            wx.MessageBox('One or more sites are located outside of the study area.','Error',wx.OK|wx.ICON_ERROR)
            arcpy.Delete_management(out_fds+"\\recp_lyr")
            stop==1

        #copy feature class without the fields
        if stop==0 and num_row>0:
            arcpy.CreateFeatureclass_management(out_fds,out_recp, "POINT") #create empty feature class
            arcpy.Append_management(fc_recp,out_fds+"\\"+out_recp, "NO_TEST") #append to empty fc without copying columns
            env.workspace = out_fgdb # change workspace to LUR fgdb
            arcpy.AddField_management(out_recp,"RecpID","LONG","","","","","","REQUIRED") #add field for integer ID
            with arcpy.da.UpdateCursor(out_recp,"RecpID") as cursor: #add integer IDs
                i=0
                for row in cursor:
                    i=i+1
                    row[0]=i
                    cursor.updateRow(row)
            log.write("\nInitial number of receptor points: "+arcpy.GetCount_management(out_recp).getOutput(0))

        self.mark2.SetLabel(mark_done) # change tick mark to done
        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        self.nextBtn.Enable()

    def onBack(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Return to panel 2"""
        if arcpy.Exists(out_fds+"\\recp_lyr"):
            arcpy.Delete_management(out_fds+"\\recp_lyr")
        if arcpy.Exists(out_fds+"\\"+out_recp):
            arcpy.Delete_management(out_fds+"\\"+out_recp)
        self.cb1.SetValue('')
        self.cb1.Clear()
        self.mark1.SetLabel(mark_empty)
        self.mark2.SetLabel(mark_empty)
        self.cb1.Disable()
        self.nextBtn.Disable()
        self.browse_btn1.Enable()
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
        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        log.write(time.strftime("\n\nReceptors from Feature Class - Stopped by user without completion: %A %d %b %Y %H:%M:%S", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('apply_panel2A','back',datetime(),NULL)''')
        conn.execute('''INSERT INTO timings VALUES('apply_panel2','start',datetime(),NULL)''')
        db.commit()
        log.flush()


    def onNext(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """go to panel 3"""
        log.write('\n\nReceptor points feature class:'+recept_fgdb+'\\'+fc_recp)
        p4 = list() # extract last part of variable name
        for p in preds:
            if p[0:2] in ['pD','pE','pF']:
                p4.append(p[p.rfind('_')+1:])
        p4 = list(set(p4)) # only keep unique ones
        if any(x in p4 for x in ['invd','invsq','valinvd','valinvsq']):
            problem_fc = [i[:i.find('_',3)] for i in preds if i[i.rfind('_')+1:] in ['invd','invsq','valinvd','valinvsq']]
            problem_fc = list(set(problem_fc))
            arcpy.MakeFeatureLayer_management(out_recp,'recp_lyr')
            for fc in problem_fc:
                arcpy.SelectLayerByLocation_management('recp_lyr','INTERSECT',fc,'','ADD_TO_SELECTION')
            arcpy.SelectLayerByAttribute_management('recp_lyr', 'SWITCH_SELECTION')
            if int(arcpy.GetCount_management('recp_lyr').getOutput(0))<int(arcpy.GetCount_management(out_recp).getOutput(0)):
                arcpy.FeatureClassToFeatureClass_conversion('recp_lyr',out_fds,'test_fc')
                arcpy.Delete_management(out_recp)
                arcpy.Rename_management("test_fc", out_recp)
                wx.MessageBox('Invalid receptor points. See log for details.','Warning',wx.OK|wx.ICON_WARNING)
                log.write('\n+++WARNING+++ Receptor points that are located on top of '+str(problem_fc)+' will cause a division bey zero error. These receptor points have been deleted prior to analysis.\n')
            arcpy.Delete_management('recp_lyr')

        log.write("\nFinal number of receptor points: "+arcpy.GetCount_management(out_recp).getOutput(0))
        log.write(time.strftime("\n\nReceptors from Feature Class - End Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        log.write(time.strftime("\nReceptors  - End Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        log.write(time.strftime("\nApply model - Start Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('apply_panel2A','stop',datetime(),NULL)''')
        db.commit()
        arcpy.AddMessage(("\nFinal number of receptor points: "+arcpy.GetCount_management(out_recp).getOutput(0)))
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
            arcpy.AddMessage((time.strftime("Aborted by user: %A %d %b %Y %H:%M:%S", time.localtime())))
            try:
                log.close()
                logging.shutdown()
                db.close()
                os.remove(out_folder)
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
class WizardPanel2B(wx.Panel):
    """Page 2B"""

    def __init__(self, parent, title=None):
        """Constructor"""
        wx.Panel.__init__(self, parent)
        wx.ToolTip.Enable(True)
        self.sizer = wx.GridBagSizer(0,0)

        title2 = wx.StaticText(self, -1, "Receptors from Regular Points")
        title2.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(title2, pos=(0,0),flag=wx.TOP|wx.LEFT|wx.BOTTOM, border=10)

        line1 = wx.StaticLine(self)
        self.sizer.Add(line1, pos=(1, 0), span=(1, 6),flag=wx.EXPAND|wx.BOTTOM, border=5)

        header1 = wx.StaticText(self,-1,label="Set Distances")
        header1.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header1, pos=(2, 0), span=(1,2),flag=wx.ALL, border=10)

        help_btn1 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn1.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn1.SetForegroundColour(wx.Colour(0,0,255))
        help_btn1.Bind(wx.EVT_BUTTON, self.onHlp1)
        self.sizer.Add(help_btn1, pos=(2,2), flag=wx.TOP|wx.BOTTOM, border=5)

        text2 = wx.StaticText(self,-1,label="Horizontal Distance")
        self.sizer.Add(text2, pos=(3, 0), flag=wx.ALL, border=10)

        self.tc1 = NumCtrl(self, integerWidth=6, fractionWidth=0, allowNegative=False, allowNone = True)
        self.tc1.SetToolTip(wx.ToolTip('Enter the horizontal distance between grid points'))
        self.sizer.Add(self.tc1, pos=(3, 1), span=(1,2), flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=5)

        self.enter_btn1 = wx.Button(self, label="Enter")
        self.enter_btn1.Bind(wx.EVT_BUTTON, self.onEnt1)
        self.sizer.Add(self.enter_btn1, pos=(3,3), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)

        self.mark1 = wx.StaticText(self,-1,label=mark_empty)
        self.mark1.SetForegroundColour((0,255,0))
        self.mark1.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark1, pos=(3,4), flag=wx.ALL, border=5)

        text3 = wx.StaticText(self,-1,label="Vertical Distance")
        self.sizer.Add(text3, pos=(4, 0), flag=wx.ALL, border=10)

        self.tc2 = NumCtrl(self, integerWidth=6, fractionWidth=0, allowNegative=False, allowNone = True)
        self.tc2.SetToolTip(wx.ToolTip('Enter the vertical distance between grid points'))
        self.sizer.Add(self.tc2, pos=(4, 1), span=(1,2), flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=5)
        self.tc2.Disable()

        self.enter_btn2 = wx.Button(self, label="Enter")
        self.enter_btn2.Bind(wx.EVT_BUTTON, self.onEnt2)
        self.sizer.Add(self.enter_btn2, pos=(4,3), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)
        self.enter_btn2.Disable()

        self.mark2 = wx.StaticText(self,-1,label=mark_empty)
        self.mark2.SetForegroundColour((0,255,0))
        self.mark2.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark2, pos=(4,4), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(5, 0), span=(1, 6),flag=wx.EXPAND|wx.BOTTOM, border=5)

        # add previous Button
        self.backBtn = wx.Button(self, label="< Back")
        self.backBtn.Bind(wx.EVT_BUTTON, self.onBack)
        self.sizer.Add(self.backBtn, pos=(6,3),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        # add next button
        self.nextBtn = wx.Button(self, label="Next >")
        self.nextBtn.Bind(wx.EVT_BUTTON, self.onNext)
        self.sizer.Add(self.nextBtn, pos=(6,4),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)
        self.nextBtn.Disable()

        # add cancel button
        self.cancBtn1 = wx.Button(self, label="Cancel")
        self.cancBtn1.Bind(wx.EVT_BUTTON, self.onCanc1)
        self.cancBtn1.SetToolTip(wx.ToolTip('Cancel'))
        self.sizer.Add(self.cancBtn1, pos=(6,5),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        self.sizer.AddGrowableCol(1)
        self.SetSizer(self.sizer)
        self.Layout()
        self.Fit()

    def onHlp1(self,event):
        """Help window for setting predictors"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p2B_SetDistance.html")
        htmlViewerInstance.Show()

    def onEnt1(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """get x distance"""
        x_dist = self.tc1.GetValue()
        if not x_dist:
            wx.MessageBox('Please enter a value','Error',wx.OK|wx.ICON_ERROR)
        else:
            xMin = arcpy.Describe(out_fds+'\\studyArea').extent.XMin
            xMax = arcpy.Describe(out_fds+'\\studyArea').extent.XMax
            xlist = list(np.arange(xMin,xMax,x_dist))
            conn.execute("DROP TABLE IF EXISTS GRID_x;")
            conn.execute("CREATE TABLE GRID_x (X Double);")
            for x in xlist:
                conn.execute("INSERT INTO GRID_x VALUES("+str(x)+")")
                db.commit()
            self.mark1.SetLabel(mark_done)
            self.tc1.Disable()
            self.enter_btn1.Disable()
            self.tc2.Enable()
            self.enter_btn2.Enable()
            self.Parent.statusbar.SetStatusText('Ready')
            del wait

    def onEnt2(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """get y distance"""
        y_dist = self.tc2.GetValue()
        if not y_dist:
            wx.MessageBox('Please enter a value','Error',wx.OK|wx.ICON_ERROR)
        else:
            env.workspace = out_fgdb
            yMin = arcpy.Describe(out_fds+'\\studyArea').extent.YMin
            yMax = arcpy.Describe(out_fds+'\\studyArea').extent.YMax
            ylist = list(np.arange(yMin,yMax,y_dist))
            conn.execute("DROP TABLE IF EXISTS GRID_y;")
            conn.execute("CREATE TABLE GRID_y (Y Double);")
            for y in ylist:
                conn.execute("INSERT INTO GRID_y VALUES("+str(y)+")")
                db.commit()
            conn.execute("DROP TABLE IF EXISTS GRID_xy;")
            conn.execute("CREATE TABLE GRID_xy AS SELECT * FROM GRID_x CROSS JOIN GRID_y;")
            db.commit()
            arcpy.MakeXYEventLayer_management(out_folder+"\\OutputSqlDB.sqlite\\GRID_xy", "X", "Y", out_fds+"\\recp_lyr",arcpy.Describe(out_fds).SpatialReference, "")
            arcpy.FeatureClassToFeatureClass_conversion(out_fds+"\\recp_lyr",out_fds,"out_temp")
            arcpy.Delete_management("recp_lyr")
            arcpy.Clip_analysis(out_fds+"\\out_temp",out_fds+"\\studyArea",out_fds+"\\"+out_recp,"")
            arcpy.Delete_management(out_fgdb+"\\out_temp")
            arcpy.AddField_management(out_fds+"\\"+out_recp,"RecpID","LONG","","","","","","REQUIRED") #add field for integer ID
            with arcpy.da.UpdateCursor(out_fds+"\\"+out_recp,"RecpID") as cursor: #add integers
                i=0
                for row in cursor:
                    i=i+1
                    row[0]=i
                    cursor.updateRow(row)
            log.write('\nReceptor points feature class created:'+out_fds+'\\'+out_recp)
            conn.execute("DROP TABLE GRID_x;")
            conn.execute("DROP TABLE GRID_y;")
            conn.execute("DROP TABLE GRID_xy;")
            self.mark2.SetLabel(mark_done)
            self.tc2.Disable()
            self.enter_btn2.Disable()
            self.nextBtn.Enable()
            self.Parent.statusbar.SetStatusText('Ready')
            del wait
            log.flush()


    def onBack(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Return to panel 2"""
        if arcpy.Exists(out_fds+"\\"+out_recp):
            arcpy.Delete_management(out_fds+"\\"+out_recp)
        if arcpy.Exists(out_fds+"\\out_temp"):
            arcpy.Delete_management(out_fds+"\\out_temp")
        self.tc1.Clear()
        self.tc2.Clear()
        self.mark1.SetLabel(mark_empty)
        self.mark2.SetLabel(mark_empty)
        self.enter_btn1.Enable()
        self.tc1.Enable()
        self.enter_btn2.Disable()
        self.tc2.Disable()
        self.nextBtn.Disable()
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
        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        log.write(time.strftime("\n\nReceptors from Regular Points - Stopped by user without completion: %A %d %b %Y %H:%M:%S", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('apply_panel2B','back',datetime(),NULL)''')
        conn.execute('''INSERT INTO timings VALUES('apply_panel2','start',datetime(),NULL)''')
        db.commit()
        log.flush()


    def onNext(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """go to panel 3"""
        log.write("\nInitial number of receptor points: "+arcpy.GetCount_management(out_recp).getOutput(0))
        p4 = list() # extract last part of variable name
        for p in preds:
            if p[0:2] in ['pD','pE','pF']:
                p4.append(p[p.rfind('_')+1:])
        p4 = list(set(p4)) # only keep unique ones
        if any(x in p4 for x in ['invd','invsq','valinvd','valinvsq']):
            problem_fc = [i[:i.find('_',3)] for i in preds if i[i.rfind('_')+1:] in ['invd','invsq','valinvd','valinvsq']]
            problem_fc = list(set(problem_fc))
            arcpy.MakeFeatureLayer_management(out_recp,'recp_lyr')
            for fc in problem_fc:
                arcpy.SelectLayerByLocation_management('recp_lyr','INTERSECT',fc,'','ADD_TO_SELECTION')
            arcpy.SelectLayerByAttribute_management('recp_lyr', 'SWITCH_SELECTION')
            if int(arcpy.GetCount_management('recp_lyr').getOutput(0))<int(arcpy.GetCount_management(out_recp).getOutput(0)):
                arcpy.FeatureClassToFeatureClass_conversion('recp_lyr',out_fds+'\\','test_fc')
                arcpy.Delete_management(out_recp)
                arcpy.Rename_management("test_fc", out_recp)
                wx.MessageBox('Invalid receptor points. See log for details.','Warning',wx.OK|wx.ICON_WARNING)
                log.write('\n+++WARNING+++ Receptor points that are located on top of '+str(problem_fc)+' will cause a division bey zero error. These receptor points have been deleted prior to analysis.\n')
            arcpy.Delete_management("recp_lyr")

        log.write("\nFinal number of receptor points: "+arcpy.GetCount_management(out_recp).getOutput(0))
        log.write(time.strftime("\n\nReceptors from Regular Points - End Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        log.write(time.strftime("\nReceptors  - End Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        log.write(time.strftime("\nApply model - Start Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('apply_panel2B','stop',datetime(),NULL)''')
        db.commit()
        arcpy.AddMessage(("\nFinal number of receptor points: "+arcpy.GetCount_management(out_recp).getOutput(0)))
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
            arcpy.AddMessage((time.strftime("Aborted by user: %A %d %b %Y %H:%M:%S", time.localtime())))
            try:
                log.close()
                logging.shutdown()
                db.close()
                os.remove(out_folder)
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
class WizardPanel2C(wx.Panel):
    """Page 2C"""

    def __init__(self, parent, title=None):
        """Constructor"""
        wx.Panel.__init__(self, parent)
        wx.ToolTip.Enable(True)
        self.sizer = wx.GridBagSizer(0,0)

        title2 = wx.StaticText(self, -1, "Receptors from Random Points")
        title2.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(title2, pos=(0,0),flag=wx.TOP|wx.LEFT|wx.BOTTOM, border=10)

        line1 = wx.StaticLine(self)
        self.sizer.Add(line1, pos=(1, 0), span=(1, 6),flag=wx.EXPAND|wx.BOTTOM, border=5)

        header1 = wx.StaticText(self,-1,label="Set Points")
        header1.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header1, pos=(2, 0), span=(1,2),flag=wx.ALL, border=10)

        help_btn1 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn1.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn1.SetForegroundColour(wx.Colour(0,0,255))
        help_btn1.Bind(wx.EVT_BUTTON, self.onHlp1)
        self.sizer.Add(help_btn1, pos=(2,2), flag=wx.TOP|wx.BOTTOM, border=5)

        text2 = wx.StaticText(self,-1,label="Number of Points")
        self.sizer.Add(text2, pos=(3, 0), flag=wx.ALL, border=10)

        self.tc1 = NumCtrl(self, integerWidth=6, fractionWidth=0, allowNegative=False, allowNone = True)
        self.tc1.SetToolTip(wx.ToolTip('Enter the number of points to be created'))
        self.sizer.Add(self.tc1, pos=(3, 1), span=(1,2), flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=5)

        self.enter_btn1 = wx.Button(self, label="Enter")
        self.enter_btn1.Bind(wx.EVT_BUTTON, self.onEnt1)
        self.sizer.Add(self.enter_btn1, pos=(3,3), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)

        self.mark1 = wx.StaticText(self,-1,label=mark_empty)
        self.mark1.SetForegroundColour((0,255,0))
        self.mark1.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark1, pos=(3,4), flag=wx.ALL, border=5)

        text3 = wx.StaticText(self,-1,label="Minimum Distance")
        self.sizer.Add(text3, pos=(4, 0), flag=wx.ALL, border=10)

        self.tc2 = NumCtrl(self, integerWidth=6, fractionWidth=0, allowNegative=False, allowNone = True)
        self.tc2.SetToolTip(wx.ToolTip('Enter the minimum distance between points'))
        self.sizer.Add(self.tc2, pos=(4, 1), span=(1,2), flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=5)
        self.tc2.Disable()

        self.enter_btn2 = wx.Button(self, label="Enter")
        self.enter_btn2.Bind(wx.EVT_BUTTON, self.onEnt2)
        self.sizer.Add(self.enter_btn2, pos=(4,3), flag=wx.TOP|wx.BOTTOM|wx.LEFT, border=5)
        self.enter_btn2.Disable()

        self.mark2 = wx.StaticText(self,-1,label=mark_empty)
        self.mark2.SetForegroundColour((0,255,0))
        self.mark2.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark2, pos=(4,4), flag=wx.ALL, border=5)

        self.sizer.Add(wx.StaticLine(self), pos=(5, 0), span=(1, 6),flag=wx.EXPAND|wx.BOTTOM, border=5)

        # add previous Button
        self.backBtn = wx.Button(self, label="< Back")
        self.backBtn.Bind(wx.EVT_BUTTON, self.onBack)
        self.sizer.Add(self.backBtn, pos=(6,3),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        # add next button
        self.nextBtn = wx.Button(self, label="Next >")
        self.nextBtn.Bind(wx.EVT_BUTTON, self.onNext)
        self.sizer.Add(self.nextBtn, pos=(6,4),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)
        self.nextBtn.Disable()

        # add cancel button
        self.cancBtn1 = wx.Button(self, label="Cancel")
        self.cancBtn1.Bind(wx.EVT_BUTTON, self.onCanc1)
        self.cancBtn1.SetToolTip(wx.ToolTip('Cancel'))
        self.sizer.Add(self.cancBtn1, pos=(6,5),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        self.sizer.AddGrowableCol(1)
        self.SetSizer(self.sizer)
        self.Layout()
        self.Fit()

    def onHlp1(self,event):
        """Help window for setting predictors"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p2C_SetPoints.html")
        htmlViewerInstance.Show()

    def onEnt1(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """get number of points"""
        global point_num
        point_num = self.tc1.GetValue()
        if not point_num:
            wx.MessageBox('Please enter a value','Error',wx.OK|wx.ICON_ERROR)
        else:
            self.mark1.SetLabel(mark_done)
            self.tc1.Disable()
            self.enter_btn1.Disable()
            self.tc2.Enable()
            self.enter_btn2.Enable()
            self.Parent.statusbar.SetStatusText('Ready')
            del wait

    def onEnt2(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """get minimum distance"""
        min_dist = self.tc2.GetValue()
        if not min_dist:
            wx.MessageBox('Please enter a value','Error',wx.OK|wx.ICON_ERROR)
        else:
            env.workspace = out_fgdb
            arcpy.CreateRandomPoints_management(out_fds, out_recp , out_fds+"\\studyArea" , "", point_num, min_dist, "POINT")
            arcpy.AddField_management(out_recp,"RecpID","LONG","","","","","","REQUIRED") #add field for integer ID
            with arcpy.da.UpdateCursor(out_recp,"RecpID") as cursor: #add integers
                i=0
                for row in cursor:
                    i=i+1
                    row[0]=i
                    cursor.updateRow(row)
            log.write('\nReceptor points feature class created:'+out_fds+'\\'+out_recp)
            self.mark2.SetLabel(mark_done)
            self.tc2.Disable()
            self.enter_btn2.Disable()
            self.nextBtn.Enable()
            self.Parent.statusbar.SetStatusText('Ready')
            del wait
            log.flush()

    def onBack(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """Return to panel 2"""
        if arcpy.Exists(out_fgdb+"\\"+out_recp):
            arcpy.Delete_management(out_fgdb+"\\"+out_recp)
        self.tc1.Clear()
        self.tc2.Clear()
        self.mark1.SetLabel(mark_empty)
        self.mark2.SetLabel(mark_empty)
        self.enter_btn1.Enable()
        self.tc1.Enable()
        self.enter_btn2.Disable()
        self.tc2.Disable()
        self.nextBtn.Disable()
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
        self.Parent.statusbar.SetStatusText('Ready')
        del wait
        log.write(time.strftime("\n\nReceptors from Random Points - Stopped by user without completion: %A %d %b %Y %H:%M:%S", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('apply_panel2C','back',datetime(),NULL)''')
        conn.execute('''INSERT INTO timings VALUES('apply_panel2','start',datetime(),NULL)''')
        db.commit()
        log.flush()


    def onNext(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """go to panel 3"""
        log.write("\nInitial number of receptor points: "+arcpy.GetCount_management(out_recp).getOutput(0))
        p4 = list() # extract last part of variable name
        for p in preds:
            if p[0:2] in ['pD','pE','pF']:
                p4.append(p[p.rfind('_')+1:])
        p4 = list(set(p4)) # only keep unique ones
        if any(x in p4 for x in ['invd','invsq','valinvd','valinvsq']):
            problem_fc = [i[:i.find('_',3)] for i in preds if i[i.rfind('_')+1:] in ['invd','invsq','valinvd','valinvsq']]
            problem_fc = list(set(problem_fc))
            arcpy.MakeFeatureLayer_management(out_recp,'recp_lyr')
            for fc in problem_fc:
                arcpy.SelectLayerByLocation_management('recp_lyr','INTERSECT',fc,'','ADD_TO_SELECTION')
            arcpy.SelectLayerByAttribute_management('recp_lyr', 'SWITCH_SELECTION')
            if int(arcpy.GetCount_management('recp_lyr').getOutput(0))<int(arcpy.GetCount_management(out_recp).getOutput(0)):
                arcpy.FeatureClassToFeatureClass_conversion('recp_lyr',out_fds+'\\','test_fc')
                arcpy.Delete_management(out_recp)
                arcpy.Rename_management("test_fc", out_recp)
                wx.MessageBox('Invalid receptor points. See log for details.','Warning',wx.OK|wx.ICON_WARNING)
                log.write('\n+++WARNING+++ Receptor points that are located on top of '+str(problem_fc)+' will cause a division bey zero error. These receptor points have been deleted prior to analysis.\n')
            arcpy.Delete_management("recp_lyr")

        log.write("\nFinal number of receptor points: "+arcpy.GetCount_management(out_recp).getOutput(0))
        log.write(time.strftime("\n\nReceptors from Random Points - End Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        log.write(time.strftime("\nReceptors  - End Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        log.write(time.strftime("\nApply model - Start Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('apply_panel2C','stop',datetime(),NULL)''')
        db.commit()
        arcpy.AddMessage(("\nFinal number of receptor points: "+arcpy.GetCount_management(out_recp).getOutput(0)))
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
            arcpy.AddMessage((time.strftime("Aborted by user: %A %d %b %Y %H:%M:%S", time.localtime())))
            try:
                log.close()
                logging.shutdown()
                db.close()
                os.remove(out_folder)
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
    """Page 3"""

    def __init__(self, parent, title=None):
        """Constructor"""
        wx.Panel.__init__(self, parent)
        wx.ToolTip.Enable(True)
        self.sizer = wx.GridBagSizer(0,0)

        title2 = wx.StaticText(self, -1, "Apply Model")
        title2.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(title2, pos=(0,0),flag=wx.TOP|wx.LEFT|wx.BOTTOM, border=10)

        line1 = wx.StaticLine(self)
        self.sizer.Add(line1, pos=(1, 0), span=(1, 6),flag=wx.EXPAND|wx.BOTTOM, border=5)

        header1 = wx.StaticText(self,-1,label="Apply LUR Model")
        header1.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(header1, pos=(2, 0), span=(1,2),flag=wx.ALL, border=10)

        help_btn1 = wx.Button(self,label="?",style=wx.BU_EXACTFIT)
        help_btn1.SetFont(wx.Font(10,wx.SWISS,wx.NORMAL,wx.BOLD))
        help_btn1.SetForegroundColour(wx.Colour(0,0,255))
        help_btn1.Bind(wx.EVT_BUTTON, self.onHlp1)
        self.sizer.Add(help_btn1, pos=(2,2), flag=wx.TOP|wx.BOTTOM, border=5)

        self.apply_btn = wx.Button(self, label="Apply model")
        self.apply_btn.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.apply_btn.SetToolTip(wx.ToolTip('Click here to apply the LUR model to the receptor points'))
        self.apply_btn.Bind(wx.EVT_BUTTON, self.onApply)
        self.sizer.Add(self.apply_btn, pos=(3,1),span=(2,2),flag=wx.TOP|wx.BOTTOM|wx.EXPAND, border=20)

        self.mark7 = wx.StaticText(self,-1,label=mark_empty)
        self.mark7.SetForegroundColour((0,255,0))
        self.mark7.SetFont(wx.Font(12,wx.SWISS, wx.NORMAL,wx.BOLD))
        self.sizer.Add(self.mark7, pos=(3,3),span=(2,1), flag=wx.LEFT|wx.TOP|wx.BOTTOM|wx.EXPAND, border=20)

        self.sizer.Add(wx.StaticLine(self), pos=(5, 0), span=(1, 6),flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=5)

        # add next button
        self.finBtn = wx.Button(self, label="Finish")
        self.finBtn.Bind(wx.EVT_BUTTON, self.onFin)
        self.sizer.Add(self.finBtn, pos=(6,4),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)
        self.finBtn.Disable()

        # add cancel button
        self.cancBtn1 = wx.Button(self, label="Cancel")
        self.cancBtn1.Bind(wx.EVT_BUTTON, self.onCanc1)
        self.cancBtn1.SetToolTip(wx.ToolTip('Cancel'))
        self.sizer.Add(self.cancBtn1, pos=(6,5),flag=wx.ALIGN_RIGHT|wx.ALL, border=5)

        self.sizer.AddGrowableCol(1)
        self.SetSizer(self.sizer)
        self.Layout()
        self.Fit()

    def onHlp1(self,event):
        """Help window for setting predictors"""
        htmlViewerInstance = HtmlViewer(None, curPath+"\\Documentation\\p3_ApplyModel.html")
        htmlViewerInstance.Show()

    def onApply(self,event):
        self.Parent.statusbar.SetStatusText('Processing...')
        wait = wx.BusyCursor()
        """apply model"""
        env.workspace = out_fgdb
        conn.execute('''INSERT INTO timings VALUES('apply_panel3','start',datetime(),NULL)''')
        db.commit()
        arcpy.AddMessage('\nstart apply\n')
        #add XY and copy to sql
        #remove xy fields if they exist
        if FieldExist(out_recp,'X') and FieldExist(out_recp,'Y'):
            arcpy.AlterField_management(out_recp, 'X', 'p_XCOORD', 'p_XCOORD')
            arcpy.AlterField_management(out_recp, 'Y', 'p_YCOORD', 'p_YCOORD')
        else:
            arcpy.AddXY_management(out_recp)
            arcpy.AlterField_management(out_recp, 'POINT_X', 'p_XCOORD', 'p_XCOORD')
            arcpy.AlterField_management(out_recp, 'POINT_Y', 'p_YCOORD', 'p_YCOORD')

        # copy to sql db
        fc_to_sql(conn,out_recp) # write everything to this table
        # add a column for each variable name
        for var in preds:
            conn.execute('ALTER TABLE '+out_recp+' ADD {} float64'.format(var))
        db.commit()
        conn.execute("CREATE UNIQUE INDEX "+out_recp+"_idx on "+out_recp+"(RecpID);")
        db.commit()

        conn.execute('''INSERT INTO timings VALUES('apply_panel3_ABC','start',datetime(),NULL)''')
        #Buffer+Intersect predictors
        preds_ABC = [i for i in preds if i[:2] in ['pA','pB','pC']]
        if len(preds_ABC)>0:#check that there are ABC predictors
            # extract first two parts and filter unique
            preds_ABC_1to2 = [i[:i.find('_',3)] for i in preds_ABC]
            preds_ABC_1to2 = list(set(preds_ABC_1to2)) #these are the fc that need to be analysed
            #extract buffer distances and run multiple ring buffer
            p4 = [int(i[i.rfind('_',0,len(i)-4)+1:-4]) for i in preds_ABC] # get buffer distances
            p4 = list(set(p4)) # only unique values
            arcpy.MultipleRingBuffer_analysis(out_recp, out_fds+"\\MultBuffer", p4, "", "", "NONE") # buffer
            arcpy.AddMessage('\nbuffer created')

            #iterate through feature classes
            for item in preds_ABC_1to2:
                arcpy.AddMessage(("\nStarting: "+item))
                temp_preds_ABC = [i for i in preds_ABC if i.startswith(item+'_')] # get predictor variables belonging to item
                temp_p4 = [int(i[i.rfind('_',0,len(i)-4)+1:-4]) for i in temp_preds_ABC] # get buffer distances from predictors
                temp_p4 = list(set(temp_p4)) # only keep unique values
                temp_p3 = [i[i.find('_',3)+1:i.rfind('_',0,len(i)-4)] for i in temp_preds_ABC] # get categories from predictors
                temp_p3 = list(set(temp_p3)) # only keep unique values
                cat_list = conn.execute("SELECT DISTINCT("+item[:2]+"_cat) FROM "+item+";").fetchall() # get all categories in feature class
                db.commit()
                arcpy.AddMessage(('\ncat_list size: '+str(sys.getsizeof(cat_list))))
                cat_list = [str(i[0]) for i in cat_list] # make into list


                if sorted(temp_p4)==sorted(p4) and sorted(temp_p3)==sorted(cat_list): #all buffers are used and all categories are used
                    arcpy.AddMessage('All buffers are used, all categories are used.')
                    arcpy.PairwiseIntersect_analysis ([out_fds+"\\MultBuffer",out_fds+"\\"+item], out_fds+"\\"+item+"_Intersect", "ALL", "", "") # intersect
                    arcpy.AddMessage(('\nIntersect completed:'+str(item)))

                elif sorted(temp_p4)==sorted(p4) and sorted(temp_p3)!=sorted(cat_list): #all buffers are used, some categories are used
                    arcpy.AddMessage('All buffers are used, only {0} categories are used.'.format(str(temp_p3)))
                    where_clause = item[:2]+"_cat IN ('{0}')".format("', '".join(str(p3) for p3 in temp_p3)) # make where clause for sqlite
                    int_list = conn.execute("SELECT DISTINCT("+item[:2]+"_cat_INT) FROM "+item+" WHERE "+where_clause).fetchall() # get integer IDs from database
                    db.commit()
                    arcpy.AddMessage(('\n int_list size: '+str(sys.getsizeof(int_list))))
                    int_list = [i[0] for i in int_list] # make into list
                    where_clause_int = item[:2]+"_cat_INT IN ({0})".format(", ".join(str(p3_int) for p3_int in int_list)) #where clause for arc
                    arcpy.MakeFeatureLayer_management(out_fds+"\\"+item,"item_lyr") # turn fc into layer
                    arcpy.SelectLayerByAttribute_management('item_lyr','NEW_SELECTION',where_clause) # select from layer
                    arcpy.FeatureClassToFeatureClass_conversion('item_lyr',out_fds,'apply_'+item) # copy into fc
                    arcpy.Delete_management(out_fgdb+"\\item_lyr") # delete layer
                    arcpy.PairwiseIntersect_analysis ([out_fds+"\\MultBuffer", out_fds+"\\apply_"+item], out_fds+"\\"+item+"_Intersect", "ALL", "", "") # intersect
                    arcpy.AddMessage(('\nIntersect completed:'+str(item)))

                elif sorted(temp_p4)!=sorted(p4) and sorted(temp_p3)==sorted(cat_list): # only some buffers are used, but all categories
                    arcpy.AddMessage('Only {0} buffers are used, all categories are used.'.format(str(temp_p4)))
                    where_clause = "distance IN ({0})".format(", ".join(str(dist) for dist in temp_p4)) # create where clause for arc
                    arcpy.MakeFeatureLayer_management(out_fds+"\\MultBuffer","buf_lyr") # turn buffer fc into layer
                    arcpy.SelectLayerByAttribute_management('buf_lyr','NEW_SELECTION',where_clause) # select from layer
                    arcpy.FeatureClassToFeatureClass_conversion('buf_lyr',out_fds,'Buf_'+item) # copy selection into fc
                    arcpy.Delete_management(out_fgdb+"\\buf_lyr") # delete fc
                    arcpy.PairwiseIntersect_analysis ([out_fds+"\\Buf_"+item, out_fds+"\\"+item], out_fds+"\\"+item+"_Intersect", "ALL", "", "") # intersect
                    arcpy.AddMessage(('\nIntersect completed:'+str(item)))

                elif sorted(temp_p4)!=sorted(p4) and sorted(temp_p3)!=sorted(cat_list): # only some buffers are used, only some categories are used
                    arcpy.AddMessage('Only {0} buffers are used, only {1} categories are used.'.format(str(temp_p4),str(temp_p3)))
                    where_clause = "distance IN ({0})".format(", ".join(str(dist) for dist in temp_p4)) # create where clause for arc
                    arcpy.MakeFeatureLayer_management(out_fds+"\\MultBuffer","buf_lyr") # turn buffer fc into layer
                    arcpy.SelectLayerByAttribute_management('buf_lyr','NEW_SELECTION',where_clause) # select from layer
                    arcpy.FeatureClassToFeatureClass_conversion('buf_lyr',out_fds,'Buf_'+item) # copy selection into fc
                    arcpy.Delete_management(out_fgdb+"\\buf_lyr") # delete fc
                    where_clause = item[:2]+"_cat IN ('{0}')".format("', '".join(str(p3) for p3 in temp_p3)) # make where clause for sqlite
                    int_list = conn.execute("SELECT DISTINCT("+item[:2]+"_cat_INT) FROM "+item+" WHERE "+where_clause).fetchall() # get integer IDs from database
                    db.commit()
                    arcpy.AddMessage(('\n int_list size: '+str(sys.getsizeof(int_list))))
                    int_list = [i[0] for i in int_list] # make into list
                    where_clause_int = item[:2]+"_cat_INT IN ({0})".format(", ".join(str(p3_int) for p3_int in int_list)) #where clause for arc
                    arcpy.MakeFeatureLayer_management(out_fds+"\\"+item,"item_lyr") # turn fc into layer
                    arcpy.SelectLayerByAttribute_management('item_lyr','NEW_SELECTION',where_clause) # select from layer
                    arcpy.FeatureClassToFeatureClass_conversion('item_lyr',out_fds,'apply_'+item) # copy into fc
                    arcpy.Delete_management(out_fgdb+"\\item_lyr") # delete layer
                    arcpy.PairwiseIntersect_analysis ([out_fds+"\\Buf_"+item, out_fds+"\\apply_"+item], out_fds+"\\"+item+"_Intersect", "ALL", "", "") # intersect
                    arcpy.AddMessage(('\nIntersect completed:'+str(item)))

                conn.execute("DROP TABLE IF EXISTS "+item+"_Intersect;")
                db.commit()
                fc_to_sql(conn,out_fds+"\\"+item+"_Intersect") # move to sql
                db.commit()
                arcpy.Delete_management(out_fgdb+"\\"+item+"_Intersect")
                if arcpy.Exists(out_fgdb+"\\apply_"+item):
                    arcpy.Delete_management(out_fgdb+"\\apply_"+item)
                arcpy.Delete_management(out_fgdb+"\\Buf_"+item)

                # populate field for weighted/aggregated value
                temp_p1 = item[0:2]
                temp_p5 = temp_preds_ABC[0]
                temp_p5 = temp_p5[len(temp_p5)-3:]
                conn.execute("ALTER TABLE "+item+"_Intersect ADD wtval float64;") # add field for weighted value
                #conn.execute("BEGIN TRANSACTION;")
                if temp_p1=='pA' and temp_p5=='sum':
                    conn.execute("UPDATE "+item+"_Intersect SET wtval=Shape_Area;")
                    db.commit()
                elif temp_p1=='pA' and temp_p5=='wtv':
                    conn.execute("UPDATE "+item+"_Intersect SET wtval=(Shape_Area/origArea)*pA_val;")
                    db.commit()
                elif temp_p1=='pA' and temp_p5=='mtv':
                    conn.execute("UPDATE "+item+"_Intersect SET wtval=Shape_Area*pA_val;")
                    db.commit()
                elif temp_p1=='pB' and temp_p5=='sum':
                    conn.execute("UPDATE "+item+"_Intersect SET wtval=Shape_Length;")
                    db.commit()
                elif temp_p1=='pB' and temp_p5=='wtv':
                    conn.execute("UPDATE "+item+"_Intersect SET wtval=(Shape_Length/origLength)*pB_val;")
                    db.commit()
                elif temp_p1=='pB' and temp_p5=='mtv':
                    conn.execute("UPDATE "+item+"_Intersect SET wtval=Shape_Length*pB_val;")
                    db.commit()
                elif temp_p1=='pC' and temp_p5=='num':
                    conn.execute("UPDATE "+item+"_Intersect SET wtval=1;")
                    db.commit()
                else:
                    conn.execute("UPDATE "+item+"_Intersect SET wtval=pC_val;")
                    db.commit()

                conn.execute("CREATE INDEX "+item+"_Intersect_idx on "+item+"_Intersect (RecpID, "+temp_p1+"_cat_INT, distance);")
                db.commit()
                arcpy.AddMessage(('\nupdate wtval field '+str(item)))

                # aggregate values
                conn.execute("DROP TABLE IF EXISTS temp;")
                if temp_p1=='pA' or temp_p1=='pB' or (temp_p1=='pC' and temp_p5=='num') or (temp_p1=='pC' and temp_p5=='sum'):
                    conn.execute("CREATE TABLE "+item+"_intermediate AS \
                                SELECT RecpID \
                                     ,"+temp_p1+"_cat_INT \
                                     ,distance \
                                     ,SUM(wtval) AS value \
        	                         ,('"+item+"_'||"+temp_p1+"_cat||'_'||cast(distance as int)||'_"+temp_p5+"') AS varName \
                                FROM "+item+"_Intersect \
                                GROUP BY RecpID, "+temp_p1+"_cat_INT, distance")

                    vars_sql_temp=list(conn.execute("SELECT DISTINCT varName FROM "+item+"_intermediate")) # make a list of variable names
                    # add values to each column
                    for var in vars_sql_temp:
                        if var[0] in preds:
                            arcpy.AddMessage(("writing value to: "+str(var[0])))
                            qry="UPDATE "+out_recp+" \
                                SET "+var[0]+" = ( \
                                SELECT value \
                                FROM "+item+"_intermediate \
                                WHERE RecpID = "+out_recp+".RecpID and varName='"+var[0]+"')"
                            conn.execute(qry)
                            db.commit()
                            log.write(time.strftime("\nPredictor created: "+str(var[0])+"  Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
                            log.flush()
                            arcpy.AddMessage(('Predictor created '+str(var[0])))
                        else:
                            arcpy.AddMessage((str(var[0])+' not required'))
                elif temp_p5=='avg':
                    conn.execute("CREATE TABLE "+item+"_intermediate AS \
                                SELECT RecpID \
                                     ,pC_cat_INT \
                                     ,distance \
                                     ,AVG(wtval) AS value \
        	                         ,('"+item+"_'||pC_cat||'_'||cast(distance as int)||'_avg') AS varName \
                                FROM "+item+"_Intersect \
                                GROUP BY RecpID, pC_cat_INT, distance")
                    vars_sql_temp=list(conn.execute("SELECT DISTINCT varName FROM "+item+"_intermediate")) # make a list of variable names
                    # add values to each column
                    for var in vars_sql_temp:
                        if var[0] in preds:
                            arcpy.AddMessage(("writing value to: "+str(var[0])))
                            qry="UPDATE "+out_recp+" \
                                SET "+var[0]+" = (\
                                SELECT value \
                                FROM "+item+"_intermediate \
                                WHERE RecpID = "+out_recp+".RecpID and varName='"+var[0]+"')"
                            conn.execute(qry)
                            log.write(time.strftime("\nPredictor created: "+str(var[0])+"  Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
                            log.flush()
                            arcpy.AddMessage(('Predictor created '+str(var[0])))
                            db.commit()
                        else:
                            arcpy.AddMessage((str(var[0])+' not required'))
                elif temp_p5=='med':
                    conn.executescript("CREATE TABLE pC_median_temp1 AS \
                        SELECT RecpID \
                            ,pC_cat_INT \
                            ,distance \
                            ,wtval \
                            ,('"+item+"_'||pC_cat||'_'||cast(distance as int)||'_med') AS varName \
                        FROM "+item+"_Intersect \
                        ORDER BY RecpID, pC_cat_INT, distance, wtval; \
                        ALTER TABLE pC_median_temp1 ADD row_num int; \
                        UPDATE pC_median_temp1 SET row_num=rowid; \
                        CREATE TABLE pC_median_temp2 AS \
                        SELECT  RecpID \
                           ,pC_cat_INT \
                           ,distance \
                           ,(MIN(row_num*1.0)+MAX(row_num*1.0))/2 AS midrow \
                           ,CASE WHEN COUNT(wtval)%2=0 THEN 'even' ELSE 'odd' END AS type \
                        FROM pC_median_temp1 \
                        GROUP BY RecpID, pC_cat_INT, distance; \
                        CREATE TABLE pC_median_temp3 AS \
                        SELECT RecpID \
                            ,pC_cat_INT \
                            ,distance \
                            ,midrow \
                        FROM pC_median_temp2 \
                        WHERE type='even' \
                        UNION ALL \
                        SELECT RecpID \
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
                        SELECT RecpID \
                            ,pC_cat_INT \
                            ,distance \
                            ,midrow \
                        FROM pC_median_temp2 \
                        WHERE type = 'odd' \
                        UNION ALL \
                        SELECT RecpID \
                        	,pC_cat_INT \
                            ,distance \
                            ,midrow_new as midrow \
                        FROM pC_median_temp4; \
                        CREATE TABLE pC_median_temp6 AS \
                        SELECT a.RecpID \
                            ,a.pC_cat_INT \
                            ,a.distance \
                            ,a.midrow \
                            ,b.row_num \
                            ,b.wtval \
                            ,b.varName \
                        FROM pC_median_temp5 as a \
                        JOIN pC_median_temp1 as b \
                        ON a.midrow = b.row_num; \
                        CREATE TABLE "+item+"_intermediate AS \
                        SELECT RecpID \
                            ,pC_cat_INT \
                            ,distance \
                            ,AVG(wtval) AS value \
                            ,MIN(varName) AS varName \
                        FROM pC_median_temp6 \
                        GROUP BY RecpID	,pC_cat_INT ,distance; \
                        DROP TABLE pC_median_temp1; \
                        DROP TABLE pC_median_temp2; \
                        DROP TABLE pC_median_temp3; \
                        DROP TABLE pC_median_temp4; \
                        DROP TABLE pC_median_temp5; \
                        DROP TABLE pC_median_temp6;")

                    vars_sql_temp=list(conn.execute("SELECT DISTINCT varName FROM  "+item+"_intermediate")) # make a list of variable names
                    # add values to each column
                    for var in vars_sql_temp:
                        if var[0] in preds:
                            arcpy.AddMessage(("writing value to: "+str(var[0])))
                            qry="UPDATE "+out_recp+" \
                                SET "+var[0]+" = ( \
                                SELECT value \
                                FROM "+item+"_intermediate \
                                WHERE RecpID = "+out_recp+".RecpID and varName='"+var[0]+"')"
                            conn.execute(qry)
                            log.write(time.strftime("\nPredictor created: "+str(var[0])+"  Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
                            log.flush()
                            arcpy.AddMessage(('Predictor created '+str(var[0])))
                            db.commit()
                        else:
                            arcpy.AddMessage((str(var[0])+' not required'))

                conn.execute("DROP TABLE "+item+"_Intersect;")
                conn.execute("DROP TABLE "+item+"_intermediate;")

            # replace missing values with zeros if p5 not sum
            for var in preds:
                if var[0:2] in ['pA','pB','pC']:
                    qry="UPDATE "+out_recp+" \
                            SET "+var+" = 0 \
                            WHERE "+var+" IS NULL"
                    arcpy.AddMessage(qry)
                    conn.execute(qry)
                    db.commit()

            conn.execute('''INSERT INTO timings VALUES('apply_panel3_ABC','stop',datetime(),NULL)''')
            conn.execute('''INSERT INTO timings VALUES('apply_panel3_DEF','start',datetime(),NULL)''')
            db.commit()
        #Distance predictors
        preds_DEF = [i for i in preds if i[:2] in ['pD','pE','pF']]
        if len(preds_DEF)>0:
            # extract first two parts and filter unique
            preds_DEF_1to2 = [i[:i.find('_',3)] for i in preds_DEF]
            preds_DEF_1to2 = list(set(preds_DEF_1to2)) #these are the fc that need to be analysed
            #iterate through them
            for item in preds_DEF_1to2:
                arcpy.AddMessage(("\nStarting: "+item))
                temp_preds_DEF = [i for i in preds_DEF if i.startswith(item+'_')] # get relevant predictor variables
                temp_p4 = [i[i.rfind('_')+1:] for i in temp_preds_DEF] # get method
                temp_p4 = list(set(temp_p4)) # only unique values
                temp_p3 = [i[i.find('_',3)+1:i.rfind('_')] for i in temp_preds_DEF] # get field name
                temp_p3 = list(set(temp_p3)) # only unique values
                temp_p3 = list([x for x in temp_p3 if x!='none']) # remove 'none'
                arcpy.SpatialJoin_analysis(out_recp, out_fds+"\\"+item, out_fds+"\\"+item+"_join", "JOIN_ONE_TO_ONE", "KEEP_ALL","","CLOSEST","","distance") #spatial join of fc to monitoring sites
                arcpy.AddMessage(('\nSpatial join completed:'+str(item)))

                if 'invsq' in temp_p4 or 'valinvsq' in temp_p4:
                    arcpy.AddField_management(out_fds+"\\"+item+"_join","distsqu","DOUBLE")
                    with arcpy.da.UpdateCursor(out_fds+"\\"+item+"_join", ["distance","distsqu"]) as cursor:
                        for row in cursor:
                            row[1]=row[0]**2
                            cursor.updateRow(row)
                # move data to sql database
                conn.execute("DROP TABLE IF EXISTS "+item+"_join;")
                db.commit()
                fc_to_sql(conn,out_fds+"\\"+item+"_join")
                db.commit()
                arcpy.Delete_management(out_fgdb+"\\"+item+"_join")
                # calcaulate Variables
                vars_sql_temp=list()
                #conn.execute("BEGIN TRANSACTION;")
                if 'dist' in temp_p4:
                    conn.execute("ALTER TABLE "+item+"_join ADD "+item+"_none_dist float64;")
                    conn.execute("UPDATE "+item+"_join SET "+item+"_none_dist=distance;")
                    db.commit()
                    vars_sql_temp.append(item+"_none_dist")

                if 'invd' in temp_p4:
                    conn.execute("ALTER TABLE "+item+"_join ADD "+item+"_none_invd float64;")
                    conn.execute("UPDATE "+item+"_join SET "+item+"_none_invd=1/distance;")
                    db.commit()
                    vars_sql_temp.append(item+"_none_invd")

                if 'invsq' in temp_p4:
                    conn.execute("ALTER TABLE "+item+"_join ADD "+item+"_none_invsq float64;")
                    conn.execute("UPDATE "+item+"_join SET "+item+"_none_invsq=1/distsqu;")
                    db.commit()
                    vars_sql_temp.append(item+"_none_invsq")

                if 'val' in temp_p4:
                    for i in temp_p3:
                        conn.execute("ALTER TABLE "+item+"_join ADD "+item+"_"+i+"_val float64;")
                        conn.execute("UPDATE "+item+"_join SET "+item+"_"+i+"_val="+i+";")
                        db.commit()
                        vars_sql_temp.append(item+"_"+i+"_val")

                if 'valdist' in temp_p4:
                    for i in temp_p3:
                        conn.execute("ALTER TABLE "+item+"_join ADD "+item+"_"+i+"_valdist float64;")
                        conn.execute("UPDATE "+item+"_join SET "+item+"_"+i+"_valdist="+i+"*distance;")
                        db.commit()
                        vars_sql_temp.append(item+"_"+i+"_valdist")

                if 'valinvd' in temp_p4:
                    for i in temp_p3:
                        conn.execute("ALTER TABLE "+item+"_join ADD "+item+"_"+i+"_valinvd float64;")
                        conn.execute("UPDATE "+item+"_join SET "+item+"_"+i+"_valinvd="+i+"*1/distance;")
                        db.commit()
                        vars_sql_temp.append(item+"_"+i+"_valinvd")

                if 'valinvsq' in temp_p4:
                    for i in temp_p3:
                        conn.execute("ALTER TABLE "+item+"_join ADD "+item+"_"+i+"_valinvsq float64;")
                        conn.execute("UPDATE "+item+"_join SET "+item+"_"+i+"_valinvsq="+i+"*1/distsqu;")
                        db.commit()
                        vars_sql_temp.append(item+"_"+i+"_valinvsq")

                conn.execute("CREATE UNIQUE INDEX "+item+"_idx on "+item+"_join (RecpID);")
                db.commit()
                for var in vars_sql_temp:
                    qry="UPDATE "+out_recp+" \
                         SET "+var+" = ( \
                         SELECT "+var+" \
                         FROM  "+item+"_join \
                         WHERE RecpID = "+out_recp+".RecpID)"
                    try:
                        conn.execute(qry)
                        log.write(time.strftime("\nPredictor created: "+str(var)+"  Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
                        log.flush()
                        arcpy.AddMessage(('\nPredictor created: '+str(var)))
                        db.commit()
                    except:
                        pass
                conn.execute("DROP TABLE "+item+"_join")
                db.commit()

            conn.execute('''INSERT INTO timings VALUES('apply_panel3_DEF','stop',datetime(),NULL)''')
            conn.execute('''INSERT INTO timings VALUES('apply_panel3_G','start',datetime(),NULL)''')

        #Raster predictors
        preds_G = [i for i in preds if i[:2]=='pG']
        if len(preds_G)>0:
            # extract first two parts and filter unique
            preds_G_1to2 = [i[:i.find('_',3)] for i in preds_G]
            preds_G_1to2 = list(set(preds_G_1to2)) #these are the fc that need to be analysed
            #iterate through them
            for item in preds_G_1to2:
                arcpy.AddMessage(("\nStarting: "+item))
                arcpy.sa.ExtractValuesToPoints(out_recp,out_fgdb+"\\"+item,out_fgdb+"\\"+item+"_sites","","VALUE_ONLY") #join cell values to points
                arcpy.AddMessage(('\nRaster value extracted: '+str(item)))

                fc_to_sql(conn,out_fgdb+"\\"+item+"_sites") # copy to sqlite
                conn.execute("CREATE UNIQUE INDEX "+item+"_idx on "+item+"_sites (RecpID);")
                db.commit()
                arcpy.Delete_management(out_fgdb+"\\"+item+"_sites")

                # add to combined table
                var = item+"_raster_val"
                qry="UPDATE "+out_recp+" \
                         SET "+var+" = ( \
                         SELECT RASTERVALU \
                         FROM "+item+"_sites \
                         WHERE RecpID = "+out_recp+".RecpID);"
                conn.execute(qry)
                db.commit()
                log.write(time.strftime("\nPredictor created: "+str(var)+"  Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
                log.flush()
                arcpy.AddMessage(('\nPredictor created '+str(var)))

            conn.execute('''INSERT INTO timings VALUES('apply_panel3_G','stop',datetime(),NULL)''')
            conn.execute('''INSERT INTO timings VALUES('apply_panel3_calc','start',datetime(),NULL)''')

        fldsNamesDict={}
        new_entry={"RecpID":"RecpID"}
        fldsNamesDict.update(new_entry)
        new_entry={"p_XCOORD":"p_XCOORD"}
        fldsNamesDict.update(new_entry)
        new_entry={"p_YCOORD":"p_YCOORD"}
        fldsNamesDict.update(new_entry)

        for i in models:
            new_entry={str(i.replace('MODEL_dep','pred')):str(i.replace('MODEL_dep','pred'))}
            fldsNamesDict.update(new_entry)
            #conn.execute("BEGIN TRANSACTION;")
            conn.execute("ALTER TABLE "+out_recp+" ADD "+i.replace('MODEL_dep','pred')+" float64;")
            coeffs = conn.execute("SELECT X,coefficient FROM "+i+";").fetchall()
            db.commit()
            arcpy.AddMessage(('\n coeffs size: '+str(sys.getsizeof(coeffs))))

            formula =""
            for p in coeffs:
                if p[0]=='Intercept':
                    formula+="("+str(p[1])+")+"
                else:
                    formula+="("+str(p[1])+")*"+p[0]+"+"
            log.write("\nModel applied: "+formula[:-1]+"\n")
            conn.execute("UPDATE "+out_recp+" SET "+i.replace('MODEL_dep','pred')+"=("+formula[:-1]+");")
            db.commit()
            arcpy.AddMessage(('\nvalues predicted for '+str(i)))


        qry="SELECT * FROM "+out_recp+";"
        conn.execute(qry) # get data from results table
        with open(out_folder+"\\out_pred.csv", "w", newline='') as csv_file: # write table to csv file
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([i[0] for i in conn.description])
            csv_writer.writerows(conn)
        db.commit()
        #import to file geodatabase
        arcpy.management.XYTableToPoint(out_folder+"\\out_pred.csv",out_fds+"\\pred_lyr","p_XCOORD", "p_YCOORD", "",arcpy.Describe(out_fds).SpatialReference)

        arcpy.AddMessage('\nfeatureclass created \n')

        log.write(time.strftime("\nPredicted values calculated - Time: %A %d %b %Y %H:%M:%S\n", time.localtime()))
        conn.execute('''INSERT INTO timings VALUES('apply_panel3_calc','stop',datetime(),NULL)''')
        conn.execute('''INSERT INTO timings VALUES('apply_panel3','stop',datetime(),NULL)''')
        db.commit()

        self.mark7.SetLabel(mark_done)
        self.finBtn.Enable()
        self.apply_btn.Disable()
        self.Parent.statusbar.SetStatusText('Ready')
        del wait


    def onFin(self,event):
        """finish"""
        wait = wx.BusyCursor()
        arcpy.AddMessage((time.strftime("Finished: %A %d %b %Y %H:%M:%S", time.localtime())))
        log.write(time.strftime("\n\nApply - End Time: %A %d %b %Y %H:%M:%S", time.localtime()))
        log.write(time.strftime("\n\nFinished by user: %A %d %b %Y %H:%M:%S", time.localtime()))
        conn.executescript("CREATE TABLE tempTimings AS \
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
            arcpy.AddMessage((time.strftime("Aborted by user: %A %d %b %Y %H:%M:%S", time.localtime())))
            try:
                log.close()
                logging.shutdown()
                db.close()
                os.remove(out_folder)
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
        wx.Frame.__init__(self, None, title="Apply LUR")#,size=(displaySize[0]/2, displaySize[1]/2))

        self.panel1 = WizardPanel1(self)
        # self.panel1.Hide()
        self.panel2 = WizardPanel2(self)
        self.panel2.Hide()
        self.panel2A = WizardPanel2A(self)
        self.panel2A.Hide()
        self.panel2B = WizardPanel2B(self)
        self.panel2B.Hide()
        self.panel2C = WizardPanel2C(self)
        self.panel2C.Hide()
        self.panel3 = WizardPanel3(self)
        self.panel3.Hide()

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
