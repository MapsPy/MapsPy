'''
Created on 14 May 2013

@author: Mirna Lerotic, 2nd Look Consulting
         http://www.2ndlookconsulting.com/


Copyright (c) 2013, Stefan Vogt, Argonne National Laboratory 
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this 
        list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this 
        list of conditions and the following disclaimer in the documentation and/or 
        other materials provided with the distribution.
    Neither the name of the Argonne National Laboratory nor the names of its 
    contributors may be used to endorse or promote products derived from this 
    software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED 
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER 
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING 
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF 
SUCH DAMAGE.
'''

import os
import getopt
import multiprocessing
import numpy as np
from time import gmtime, strftime
import h5py
import shutil

import maps_generate_img_dat
import maps_definitions
import maps_elements
from file_io import maps_hdf5
import maps_fit_parameters
import maps_calibration
import make_maps
import maps_tools



# ------------------------------------------------------------------------------------------------
def check_output_dirs(main):
    
    dir = main['output_dir']
    if not os.path.exists(dir):   
        os.makedirs(dir)
        if not os.path.exists(dir):
            print 'warning: did not find the output directory, and could not create a new output directory. Will abort this action'
            return 0
        
    dir = main['mda_dir']
    if not os.path.exists(dir):   
        os.makedirs(dir)
        if not os.path.exists(dir):
            print 'warning: did not find the output directory, and could not create a new output directory. Will abort this action'
            return 0
        
    dir = main['pca_dir']
    if not os.path.exists(dir):   
        os.makedirs(dir)
        if not os.path.exists(dir):
            print 'warning: did not find the pca directory, and could not create a new pca directory. Will abort this action'
            return 0
        
    dir = main['img_dat_dir']
    if not os.path.exists(dir):   
        os.makedirs(dir)
        if not os.path.exists(dir):
            print 'warning: did not find the img_dat directory, and could not create a new img_datdirectory. Will abort this action'
            return 0
        
    dir = main['line_dat_dir']
    if not os.path.exists(dir):   
        os.makedirs(dir)
        if not os.path.exists(dir):
            print 'warning: did not find the line_dat directory, and could not create a new line_dat directory. Will abort this action'
            return 0
        
    dir = main['xanes_dat_dir']
    if not os.path.exists(dir):   
        os.makedirs(dir)
        if not os.path.exists(dir):
            print 'warning: did not find the xanes_dat directory, and could not create a new xanes_dat directory. Will abort this action'
            return 0

    dir = main['fly_dat_dir']
    if not os.path.exists(dir):   
        os.makedirs(dir)
        if not os.path.exists(dir):
            print 'warning: did not find the fly_dat directory, and could not create a new fly_dat directory. Will abort this action'
            return 0
        
    dir = os.path.join(main['master_dir'], 'lookup')
    if not os.path.exists(dir):   
        os.makedirs(dir)
        
    dir = os.path.join(main['master_dir'],'rois')
    if not os.path.exists(dir):   
        os.makedirs(dir)
        
    return 1

# ------------------------------------------------------------------------------------------------
def select_beamline(main, make_maps_conf, this_beamline):
    
    make_maps_conf.use_det[:] = 0
    make_maps_conf.use_beamline = this_beamline
  
    print 'make_maps_conf.version', make_maps_conf.version
    print 'main[beamline]', main['beamline']   
    
    if main['beamline'] == '2-ID-E' :
        make_maps_conf.use_det[0] = 1
        make_maps_conf.fit_t_be = 12000. #[8  microns]

        make_maps_conf.dmaps_names  = ['SRcurrent', 'us_ic', 'ds_ic', 'abs_ic', \
                                      'abs_cfg', 'H_dpc_cfg', 'V_dpc_cfg', 'dia1_dpc_cfg', 'dia2_dpc_cfg', \
                                      'H_dpc_norm', 'V_dpc_norm', 'phase', 'ELT1', 'ERT1', 'ICR1', 'OCR1', \
                                      'deadT', 'x_coord', 'y_coord', \
                                      'dummy', 'dummy', 'dummy', 'dummy']    
    
    if (main['beamline'] =='2-ID-D') or (main['beamline'] == '2-ID-B') or (main['beamline'] == '2-BM') :
        make_maps_conf.use_det[0] = 1
        make_maps_conf.fit_t_be = 8000. #[8  microns]
         
        make_maps_conf.dmaps_names  = ['SRcurrent', 'us_ic', 'ds_ic', 'abs_ic', \
                                        'abs_cfg', 'H_dpc_cfg', 'V_dpc_cfg', 'dia1_dpc_cfg', 'dia2_dpc_cfg', \
                                        'H_dpc_norm', 'V_dpc_norm', 'phase', 'ELT1', 'ERT1', 'ICR1', 'OCR1', \
                                        'deadT', 'x_coord', 'y_coord', \
                                        'dummy', 'dummy', 'dummy', 'dummy']    

        print 'make_maps_conf.dmaps_names', make_maps_conf.dmaps_names
    
      
    if (main['beamline'] == 'Bio-CAT') :
        print 'now it is Bio-CAT'
        make_maps_conf.use_det[0] = 1
        make_maps_conf.fit_t_be = 24000. #[8  microns]

        make_maps_conf.dmaps_names  = ['SRcurrent', 'us_ic', 'ds_ic', 'abs_ic', 'ELT1', 'ERT1', \
                                      'x_coord', 'y_coord', 'dummy', 'dummy', 'dummy', \
                                      'dummy', 'dummy', 'dummy', 'dummy', 'dummy', 'dummy', \
                                      'dummy', 'dummy', 'dummy', 'dummy', 'dummy', 'dummy']    
 
    
    if main['beamline'] == 'GSE-CARS' :
        make_maps_conf.use_det[0] = 1
        make_maps_conf.fit_t_be = 24000. #[8  microns]

        make_maps_conf.dmaps_names  = ['SRcurrent', 'us_ic', 'ds_ic', 'abs_ic', \
                                       'abs_cfg', 'H_dpc_cfg', 'V_dpc_cfg', 'dia1_dpc_cfg', 'dia2_dpc_cfg', \
                                       'H_dpc_norm', 'V_dpc_norm', 'phase', 'ELT1', 'ERT1', 'ICR1', 'OCR1', \
                                       'deadT', 'x_coord', 'y_coord', \
                                       'dummy', 'dummy', 'dummy', 'dummy']    
     
    if main['beamline'] == 'Bionanoprobe' :
        make_maps_conf.use_det[0] = 1
        make_maps_conf.fit_t_be = 24000. 

        make_maps_conf.dmaps_names  = ['SRcurrent', 'us_ic', 'ds_ic', 'abs', \
                                       'H_dpc_cfg', 'V_dpc_cfg', 'dia1_dpc_cfg', 'dia2_dpc_cfg', \
                                       'H_dpc_norm', 'V_dpc_norm', 'phase', 'ELT1', 'ERT1', 'dummy', 'dummy', \
                                       'dummy', 'dummy', 'dummy',  'dummy', \
                                       'dummy', 'dummy', 'dummy', 'dummy']    
    
    print main['beamline']
    
    if main['beamline'] == 'DLS-I08' :
        make_maps_conf.use_det[0] = 1
        make_maps_conf.fit_t_be = 24000. 

        make_maps_conf.dmaps_names  = ['dummy', 'dummy', 'dummy', 'dummy', \
                                       'dummy', 'dummy', 'dummy', 'dummy', \
                                       'dummy', 'dummy', 'dummy', 'dummy', 'dummy', 'dummy', 'dummy', \
                                       'dummy', 'dummy', 'dummy',  'dummy', \
                                       'dummy', 'dummy', 'dummy', 'dummy']    
    
    for i in range(len(make_maps_conf.dmaps_names)):
        make_maps_conf.dmaps[i].name == make_maps_conf.dmaps_names[i]
        if make_maps_conf.dmaps_names[i] != 'dummy':
            make_maps_conf.dmaps[i].use = 1

        #print make_maps_conf.dmaps[i].name, make_maps_conf.dmaps[i].use
      
    return



# ------------------------------------------------------------------------------------------------
def load_spectrum( main, filename, spectra, append = 1):

    ic_us = 0
    ic_ds = 0
    us_amp = np.zeros((3))
    ds_amp = np.zeros((3))
    
    current = 0
    
    real_time = 0
    live_time = 0
    
    try:
        f = open(filename, 'rt')                       
    except: 
        print 'Could not open file:', filename
        return
        

    line = f.readline() # 1. line is version
    #print line
    line = f.readline() # 2. is # elements
    slist = line.split(':')
    tag = slist[0]
    value = ''.join(slist[1:])
    n_detector_elements  = int(value)  
    if n_detector_elements < 1 : n_detector_elements = 1
    #print 'n_detector_elements', n_detector_elements
    line = f.readline()
    line = f.readline()
    slist = line.split(':')
    tag = slist[0]
    value = ''.join(slist[1:])
    n_channels = int(value)
    #print 'n_channels', n_channels
    f.close()
    
    amp = np.zeros((8, 3))       # 8 amplifiers, each with a numerical value(0) and a unit(1), resulting in  a factor (3)
    amp[:, 0] = 1. # put in a numerical value default of 1.


    real_time = []
    live_time = []
    current = []
    
    calibration = { 'offset' : np.zeros((n_detector_elements)), 
                   'slope'   : np.zeros((n_detector_elements)), 
                   'quad'   : np.zeros((n_detector_elements)) }
    
    counts_us_ic = 0.
    counts_ds_ic = 0.
    a_num = ['','','','','','']
    a_unit = ['','','','','','']
    year = 0
    month = 0
    day = 0
    hour = 0
    minute = 0
    roi_area = -1.
    roi_pixels = -1

    found_data = 0     
    f = open(filename)
    lines = f.readlines()
    
    for line in lines:
        if ':' in line : 
            slist = line.split(':')
            tag = slist[0]
            value = ''.join(slist[1:])
            
            if   tag == 'VERSION': version = float(value)
            elif tag == 'DATE' : date  =  value
            elif tag == 'ELEMENTS' : n_detector_elements = int(value)
            elif tag == 'CHANNELS' : n_channels = int(value)
            elif tag == 'REAL_TIME'  : 
                real_time = np.zeros((n_detector_elements))
                value = value.split(' ')
                real_time = [float(x) for x in value if x != '']                                                         
            elif tag == 'LIVE_TIME'  :  
                live_time = np.zeros((n_detector_elements))
                value = value.split(' ')
                live_time = [float(x) for x in value if x != '']   
            elif tag == 'CAL_OFFSET'  :  
                value = value.split(' ')
                valuelist = [float(x) for x in value if x != '']
                calibration['offset'][:] = valuelist 
            elif tag == 'CAL_SLOPE'  :  
                value = value.split(' ')
                valuelist = [float(x) for x in value if x != '']
                calibration['slope'][:] = valuelist              
            elif tag == 'CAL_QUAD'  :  
                value = value.split(' ')
                valuelist = [float(x) for x in value if x != '']
                calibration['quad'][:] = valuelist 
            elif tag == 'TWO_THETA'  :  
                two_theta = np.zeros((n_detector_elements))
                value = value.split(' ')
                valuelist = [float(x) for x in value if x != '']
                two_theta[:] = valuelist
            elif tag == 'UPSTREAM_IONCHAMBER' :  
                counts_us_ic = np.zeros((n_detector_elements))
                value = value.split(' ')
                valuelist = [float(x) for x in value if x != '']
                counts_us_ic[:] = valuelist
            elif tag == 'DOWNSTREAM_IONCHAMBER' :  
                counts_ds_ic = np.zeros((n_detector_elements))
                value = value.split(' ')
                valuelist = [float(x) for x in value if x != '']
                counts_ds_ic[:] = valuelist
            elif tag == 'SRcurrent' :  
                current = np.zeros((n_detector_elements))
                value = value.split(' ')
                valuelist = [float(x) for x in value if x != '']
                current[:] = valuelist                
                
            elif tag == 'ENVIRONMENT' :  
                value = ':'.join(slist[1:])
                pos = value.find('=')
                etag = value[0:pos].strip()
                vallist = value.split('"')
                temp = vallist[1]
                if etag == 'S:SRcurrentAI' :
                    current = float(temp)    
                elif etag == '2xfm:scaler1_cts1.B' :
                    if counts_us_ic == 0 : counts_us_ic = float(temp)           
                elif etag == '2xfm:scaler1_cts1.C' : 
                    if counts_ds_ic == 0 : counts_ds_ic = float(temp)        
                elif etag == '2xfm:scaler3_cts1.B' : 
                    counts_us_ic = float(temp)           
                elif etag == '2xfm:scaler3_cts1.C' : 
                    counts_ds_ic = float(temp) 
                elif etag == '2idd:scaler1_cts1.C' : 
                    counts_us_ic = float(temp)           
                elif etag == '2idd:scaler1_cts1.B' : 
                    counts_ds_ic = float(temp)
                elif etag == '8bmb:3820:scaler1_cts1.B' : 
                    counts_us_ic = float(temp)           
                elif etag == '8bmb:3820:scaler1_cts1.C' : 
                    counts_ds_ic = float(temp)       
                elif etag[5:] == 'A1sens_num.VAL' : 
                    a_num[0] = temp        
                elif etag[5:] == 'A2sens_num.VAL' : 
                    a_num[1] = temp         
                elif etag[5:] == 'A3sens_num.VAL' : 
                    a_num[2] = temp         
                elif etag[5:] == 'A4sens_num.VAL' : 
                    a_num[3] = temp        
                elif etag[5:] == 'A1sens_unit.VAL' :                 
                    a_unit[0] = temp
                elif etag[5:] == 'A2sens_unit.VAL' :                 
                    a_unit[1] = temp
                elif etag[5:] == 'A3sens_unit.VAL' :                 
                    a_unit[2] = temp
                elif etag[5:] == 'A4sens_unit.VAL' :                 
                    a_unit[3] = temp
                    
                                            
            elif tag == 'DATA' :  
                found_data = 1
                dataindex = lines.index(line)
                break
            
    if found_data: 
        data = np.zeros((n_channels, n_detector_elements))
        for i in range(n_channels):
            for j in range(n_detector_elements):
                dataindex += 1
                line = lines[dataindex]
                counts = float(line)
                data[i, j] = counts
                                       
    f.close()

    if data.size == 0: 
        print 'Not a valid data file:', filename
        return
    

    for i in range(8):
        amp[i, 2] = amp[i, 0] 
        if amp[i, 1] == 0 : amp[i, 2] = amp[i, 2] /1000.        # pA/V
        if amp[i, 1] == 1 : amp[i, 2] = amp[i, 2]               # nA/V
        if amp[i, 1] == 2 : amp[i, 2] = amp[i, 2] *1000.        #uA/V
        if amp[i, 1] == 3 : amp[i, 2] = amp[i, 2] *1000. *1000. #mA/V

 
    if counts_ds_ic == 0 :
        print 'warning downstream IC counts zero'
        counts_ds_ic = 1.

    if counts_us_ic == 0  : 
        print 'warning upstream IC counts zero'
        counts_us_ic = 1.


    if append > 0: 
        temp_used = []
        for item in spectra: temp_used.append(item.used)
        wo = np.where(np.array(temp_used) > 0.)
        wo = wo[0]
        if wo.size != 0 : 
            wo = np.amax(wo) 
        else:
            wo = -1 
    else:
        wo =  -1
            

 
    month = 0
    year = 0
    day = 0
    hour = 0
    minute = 0
    date = date.strip()
    if date != '':
        test = date[0:3]
        # test which of the two formats is used
        if (test == 'Mon') or (test == 'Tue') or (test == 'Wed') or (test == 'Thu') or (test == 'Fri') or (test == 'Sat') or (test == 'Sun') :
            year_pos = 20
            month_pos = 4
            day_pos = 8
            hour_pos = 13
            minute_pos = 16
        else:
            year_pos = 8
            month_pos = 0
            day_pos = 4
            hour_pos = 13
            minute_pos = 16
    
        test = date[month_pos: month_pos + 3].lower()
        if test == 'jan' : month = 1
        if test == 'feb' : month = 2
        if test == 'mar' : month = 3
        if test == 'apr' : month = 4
        if test == 'may' : month = 5
        if test == 'jun' : month = 6
        if test == 'jul' : month = 7
        if test == 'aug' : month = 8
        if test == 'sep' : month = 9
        if test == 'oct' : month = 10
        if test == 'nov' : month = 11
        if test == 'dec' : month = 12
        try:
            test = date[year_pos:(year_pos + 4)]
            year = int(test)
            test = date[day_pos: day_pos + 2]
            day = int(test)
            test = date[hour_pos:hour_pos + 4]
            hour = int(test)
            test = date[minute_pos: minute_pos + 4]
            minute = int(test)
        except:
            print ' Could not convert date.'


    for l in range(n_detector_elements):        
        i = int(l)
        j = int(i+wo+1)
        if np.sum(data[:, i]) > 0. : 
            shortname = filename.split('/')  
            shortname = shortname[-1]
            shortname = shortname.split('\\')  
            shortname = shortname[-1]
            shortname, ext = os.path.splitext(shortname)
            
            spectra[j].name = shortname.strip()
            spectra[j].used_chan = n_channels
            spectra[j].used = 1
            spectra[j].data[0:spectra[j].used_chan] = data[0:spectra[j].used_chan, i]
            spectra[j].real_time = real_time[l]
            spectra[j].live_time = live_time[l]
            spectra[j].SRcurrent = current[0]
            spectra[j].calib['off'] = calibration['offset']
            spectra[j].calib['lin'] = calibration['slope']
            spectra[j].calib['quad'] = calibration['quad']
            spectra[j].IC[0]['cts'] = counts_us_ic
            for kk in range(2) : 
                if kk == 0 : temp = us_amp
                if kk == 1 : temp = ds_amp
                spectra[j].IC[kk]['sens_num'] = float(temp[0])
                spectra[j].IC[kk]['sens_unit'] = float(temp[1])
                spectra[j].IC[kk]['sens_factor'] = float(temp[2])

            for kk in range(2) : 
                if 'A/V' in a_unit[kk] :
                    spectra[j].IC[kk]['sens_factor'] = float(a_num[kk])
                    spectra[j].IC[kk]['sens_num'] = float(a_num[kk])
                    if 'pA/' in a_unit[kk] : spectra[j].IC[kk]['sens_unit'] = 0
                    if 'nA/' in a_unit[kk] : spectra[j].IC[kk]['sens_unit'] = 1
                    if 'uA/' in a_unit[kk] : spectra[j].IC[kk]['sens_unit'] = 2
                    if 'mA/' in a_unit[kk] : spectra[j].IC[kk]['sens_unit'] = 3
                else:  
                    spectra[j].IC[kk]['sens_unit'] = float(a_unit[kk])
                    if (float(a_num[kk]) == 0) : spectra[j].IC[kk]['sens_factor'] = 1
                    if (float(a_num[kk]) == 1) : spectra[j].IC[kk]['sens_factor'] = 2
                    if float(a_num[kk]) == 2 : spectra[j].IC[kk]['sens_factor'] = 5
                    if float(a_num[kk]) == 3 : spectra[j].IC[kk]['sens_factor'] = 10
                    if float(a_num[kk]) == 4 : spectra[j].IC[kk]['sens_factor'] = 20
                    if float(a_num[kk]) == 5 : spectra[j].IC[kk]['sens_factor'] = 50
                    if float(a_num[kk]) == 6 : spectra[j].IC[kk]['sens_factor'] = 100
                    if float(a_num[kk]) == 7 : spectra[j].IC[kk]['sens_factor'] = 200
                    if float(a_num[kk]) == 8 : spectra[j].IC[kk]['sens_factor'] = 500
                    spectra[j].IC[kk]['sens_num'] = spectra[j].IC[kk]['sens_factor'] 
                               
                spectra[j].IC[kk]['sens_factor'] = float(spectra[j].IC[kk]['sens_factor']) /1000. *np.power(1000.,float(spectra[j].IC[kk]['sens_unit']))
               
                                  
            spectra[j].IC[1]['cts'] = counts_ds_ic
            spectra[j].date['year'] = year
            spectra[j].date['month'] = month
            spectra[j].date['day'] = day
            spectra[j].date['hour'] = hour
            spectra[j].date['minute'] = minute
            #spectra[j].roi['area'] = roi_area[k]
            #spectra[j].roi['pixels'] = roi_pixels[k]
        else:
            spectra[j].used_chan = 0L


    return 

# ------------------------------------------------------------------------------------------------
def save_spectrum( main, filename, sfilename):
    
    
    #Get info from .h5 file 
    no_specs = 1
    real_time = 0.0
    live_time = 0.0
    srcurrent = 0.0
    uICcts = 0
    dICcts = 0
    
    amp = np.zeros((8,3), dtype=np.float)   
    for i in range(8):
        if amp[i, 0] == 0. : amp[i, 2] = 1.
        if amp[i, 0] == 1. : amp[i, 2] = 2.
        if amp[i, 0] == 2. : amp[i, 2] = 5.
        if amp[i, 0] == 3. : amp[i, 2] = 10.
        if amp[i, 0] == 4. : amp[i, 2] = 20.
        if amp[i, 0] == 5. : amp[i, 2] = 50.
        if amp[i, 0] == 6. : amp[i, 2] = 100.
        if amp[i, 0] == 7. : amp[i, 2] = 200.
        if amp[i, 0] == 8. : amp[i, 2] = 500.
        if amp[i, 1] == 0. : amp[i, 2] = amp[i, 2] /1000.        # pA/V
        if amp[i, 1] == 1. : amp[i, 2] = amp[i, 2]               # nA/V
        if amp[i, 1] == 2. : amp[i, 2] = amp[i, 2] *1000.        #uA/V
        if amp[i, 1] == 3. : amp[i, 2] = amp[i, 2] *1000. *1000. #mA/V


    us_amp = np.zeros(3) 
    ds_amp = np.zeros(3)

    if main['beamline'] == '2-ID-D' : 
        us_amp[:] = amp[1, :]
        ds_amp[:] = amp[3, :]

    if main['beamline'] == '2-ID-E' :
        us_amp[:] = amp[0, :]
        ds_amp[:] = amp[1, :]

    if main['beamline'] == 'Bio-CAT' :
        us_amp[:] = amp[0, :]
        ds_amp[:] = amp[1, :]

    ic0 = {'cts':0., 'sens_num':0., 'sens_unit':0., 'sens_factor':0.}
    ic1 = {'cts':0., 'sens_num':0., 'sens_unit':0., 'sens_factor':0.}
    
                  
    for kk in range(1) : 
        if kk == 0 : 
            temp = us_amp
            ic0['sens_num'] = float(temp[0])
            ic0['sens_unit'] = float(temp[1])
            ic0['sens_factor'] = float(temp[2])
        if kk == 1 : 
            temp = ds_amp
            ic1['sens_num'] = float(temp[0])
            ic1['sens_unit'] = float(temp[1])
            ic1['sens_factor'] = float(temp[2])
    
    ch5 = maps_hdf5.h5()
    
    fh5 = h5py.File(filename, 'r') 
            
    if 'MAPS' not in fh5:
        print 'error, hdf5 file does not contain the required MAPS group. I am aborting this action'
        return 

    maps_group_id = fh5['MAPS']

    entryname = 'scan_time_stamp'
    this_data, valid_read = ch5.read_hdf5_core(maps_group_id, entryname) 
    if valid_read : scan_time_stamp = this_data
        
    mmcGrp = maps_group_id['make_maps_conf']
    
    calibration_offset = 0
    entryname = 'calibration_offset'
    this_data, valid_read = ch5.read_hdf5_core(mmcGrp, entryname) 
    if valid_read : calibration_offset = this_data

    calibration_slope = 0
    entryname = 'calibration_slope'
    this_data, valid_read = ch5.read_hdf5_core(mmcGrp, entryname) 
    if valid_read : calibration_slope = this_data

    calibration_quad = 0
    entryname = 'calibration_quad'
    this_data, valid_read = ch5.read_hdf5_core(mmcGrp, entryname) 
    if valid_read : calibration_quad = this_data
    
    int_spec = []
    entryname = 'int_spec'
    this_data, valid_read = ch5.read_hdf5_core(maps_group_id, entryname)
    if valid_read : int_spec = this_data
            
    fh5.close()    
    
    print 'saving', sfilename
    f = open(sfilename, 'w')
    print>>f, 'VERSION:    3.1'
    print>>f,  'ELEMENTS:  ' + str(no_specs)
    line = 'DATE: '+ str(scan_time_stamp)
    print>>f, line
    line = 'CHANNELS: '+ str(main['max_spec_channels'])
    print>>f, line
    line = 'REAL_TIME: ' + str(real_time)
    print>>f, line
    line = 'LIVE_TIME: ' + str(live_time)
    print>>f, line
    line = 'SRcurrent: ' + str(srcurrent)
    print>>f, line
    line = 'UPSTREAM_IONCHAMBER: ' + str(uICcts)
    print>>f, line
    line = 'DOWNSTREAM_IONCHAMBER: ' + str(dICcts)
    print>>f, line
    line = 'CAL_OFFSET:  ' + str(calibration_offset[0])
    print>>f, line
    line = 'CAL_SLOPE: ' + str(calibration_slope[0])
    print>>f, line
    line = 'CAL_QUAD: ' + str(calibration_quad[0])
    print>>f, line


    if main['beamline'] == '2-ID-E' : 
        line = ''
        if ic0['sens_num'] == 0 : line = 'ENVIRONMENT: 2xfm:A1sens_num.VAL="1"'
        if ic0['sens_num'] == 1 : line = 'ENVIRONMENT: 2xfm:A1sens_num.VAL="2"'
        if ic0['sens_num'] == 2 : line = 'ENVIRONMENT: 2xfm:A1sens_num.VAL="5"'
        if ic0['sens_num'] == 3 : line = 'ENVIRONMENT: 2xfm:A1sens_num.VAL="10"'
        if ic0['sens_num'] == 4 : line = 'ENVIRONMENT: 2xfm:A1sens_num.VAL="20"'
        if ic0['sens_num'] == 5 : line = 'ENVIRONMENT: 2xfm:A1sens_num.VAL="50"'
        if ic0['sens_num'] == 6 : line = 'ENVIRONMENT: 2xfm:A1sens_num.VAL="100"'
        if ic0['sens_num'] == 7 : line = 'ENVIRONMENT: 2xfm:A1sens_num.VAL="200"'
        if ic0['sens_num'] == 8 : line = 'ENVIRONMENT: 2xfm:A1sens_num.VAL="500"'
        print>>f, line
        if ic0['sens_unit'] == 0 : line = 'ENVIRONMENT: 2xfm:A1sens_unit.VAL="pA/V"'
        if ic0['sens_unit'] == 1 : line = 'ENVIRONMENT: 2xfm:A1sens_unit.VAL="nA/V"'
        if ic0['sens_unit'] == 2 : line = 'ENVIRONMENT: 2xfm:A1sens_unit.VAL="mA/V"'
        if ic0['sens_unit'] == 3 : line = 'ENVIRONMENT: 2xfm:A1sens_unit.VAL="uA/V"'
        print>>f, line
        if ic1['sens_num'] == 0 : line = 'ENVIRONMENT: 2xfm:A2sens_num.VAL="1"'
        if ic1['sens_num'] == 1 : line = 'ENVIRONMENT: 2xfm:A2sens_num.VAL="2"'
        if ic1['sens_num'] == 2 : line = 'ENVIRONMENT: 2xfm:A2sens_num.VAL="5"'
        if ic1['sens_num'] == 3 : line = 'ENVIRONMENT: 2xfm:A2sens_num.VAL="10"'
        if ic1['sens_num'] == 4 : line = 'ENVIRONMENT: 2xfm:A2sens_num.VAL="20"'
        if ic1['sens_num'] == 5 : line = 'ENVIRONMENT: 2xfm:A2sens_num.VAL="50"'
        if ic1['sens_num'] == 6 : line = 'ENVIRONMENT: 2xfm:A2sens_num.VAL="100"'
        if ic1['sens_num'] == 7 : line = 'ENVIRONMENT: 2xfm:A2sens_num.VAL="200"'
        if ic1['sens_num'] == 8 : line = 'ENVIRONMENT: 2xfm:A2sens_num.VAL="500"'
        print>>f, line
        if ic1['sens_unit'] == 0 : line = 'ENVIRONMENT: 2xfm:A2sens_unit.VAL="pA/V"'
        if ic1['sens_unit'] == 1 : line = 'ENVIRONMENT: 2xfm:A2sens_unit.VAL="nA/V"'
        if ic1['sens_unit'] == 2 : line = 'ENVIRONMENT: 2xfm:A2sens_unit.VAL="mA/V"'
        if ic1['sens_unit'] == 3 : line = 'ENVIRONMENT: 2xfm:A2sens_unit.VAL="uA/V"'
        print>>f, line
  
    if main['beamline'] == '2-ID-D' : 
        line = ['']
        if ic0['sens_num'] == 0 : line = 'ENVIRONMENT: 2idd:A2sens_num.VAL="1"'
        if ic0['sens_num'] == 1 : line = 'ENVIRONMENT: 2idd:A2sens_num.VAL="2"'
        if ic0['sens_num'] == 2 : line = 'ENVIRONMENT: 2idd:A2sens_num.VAL="5"'
        if ic0['sens_num'] == 3 : line = 'ENVIRONMENT: 2idd:A2sens_num.VAL="10"'
        if ic0['sens_num'] == 4 : line = 'ENVIRONMENT: 2idd:A2sens_num.VAL="20"'
        if ic0['sens_num'] == 5 : line = 'ENVIRONMENT: 2idd:A2sens_num.VAL="50"'
        if ic0['sens_num'] == 6 : line = 'ENVIRONMENT: 2idd:A2sens_num.VAL="100"'
        if ic0['sens_num'] == 7 : line = 'ENVIRONMENT: 2idd:A2sens_num.VAL="200"'
        if ic0['sens_num'] == 8 : line = 'ENVIRONMENT: 2idd:A2sens_num.VAL="500"'
        print>>f, line
        if ic0['sens_unit'] == 0 : line = 'ENVIRONMENT: 2idd:A2sens_unit.VAL="pA/V"'
        if ic0['sens_unit'] == 1 : line = 'ENVIRONMENT: 2idd:A2sens_unit.VAL="nA/V"'
        if ic0['sens_unit'] == 2 : line = 'ENVIRONMENT: 2idd:A2sens_unit.VAL="mA/V"'
        if ic0['sens_unit'] == 3 : line = 'ENVIRONMENT: 2idd:A2sens_unit.VAL="uA/V"'
        print>>f, line
        if ic1['sens_num'] == 0 : line = 'ENVIRONMENT: 2idd:A4sens_num.VAL="1"'
        if ic1['sens_num'] == 1 : line = 'ENVIRONMENT: 2idd:A4sens_num.VAL="2"'
        if ic1['sens_num'] == 2 : line = 'ENVIRONMENT: 2idd:A4sens_num.VAL="5"'
        if ic1['sens_num'] == 3 : line = 'ENVIRONMENT: 2idd:A4sens_num.VAL="10"'
        if ic1['sens_num'] == 4 : line = 'ENVIRONMENT: 2idd:A4sens_num.VAL="20"'
        if ic1['sens_num'] == 5 : line = 'ENVIRONMENT: 2idd:A4sens_num.VAL="50"'
        if ic1['sens_num'] == 6 : line = 'ENVIRONMENT: 2idd:A4sens_num.VAL="100"'
        if ic1['sens_num'] == 7 : line = 'ENVIRONMENT: 2idd:A4sens_num.VAL="200"'
        if ic1['sens_num'] == 8 : line = 'ENVIRONMENT: 2idd:A4sens_num.VAL="500"'
        print>>f, line
        if ic1['sens_unit'] == 0 : line = 'ENVIRONMENT: 2idd:A4sens_unit.VAL="pA/V"'
        if ic1['sens_unit'] == 1 : line = 'ENVIRONMENT: 2idd:A4sens_unit.VAL="nA/V"'
        if ic1['sens_unit'] == 2 : line = 'ENVIRONMENT: 2idd:A4sens_unit.VAL="mA/V"'
        if ic1['sens_unit'] == 3 : line = 'ENVIRONMENT: 2idd:A4sens_unit.VAL="uA/V"'
        print>>f, line





    print>>f, 'DATA:'
    for ie in range(len(int_spec)):
        print>>f, '%.6f' %(int_spec[ie])
    
    f.close()    
   
    return


""" ------------------------------------------------------------------------------------------------"""
def main(wdir='', a=1,b=0,c=0,d=0,e=0):
    

    verbose = True
    time_started = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    
    first_run = 1
    
    #remove quotations marks if any in wdir
    wdir.strip('"') 
    if "'" in wdir: wdir = wdir[1:-1]

    wdir = os.path.normpath(wdir)

    if verbose: print 'working directory =', wdir
        
    if not os.path.exists(wdir):
        print 'Error - Directory ', wdir, ' does not exist. Please specify working directory.'
        return
    
    current_directory = wdir
    

    #define main
    main = {'mapspy_version':'1.1', 
            'maps_date':'01. March, 2013', 
            'beamline':'2-ID-E', 
            'S_font':'', 
            'M_font':'', 
            'L_font':'', 
            'master_dir':   current_directory, 
            'output_dir':   os.path.join(current_directory, 'output'), 
            'img_dat_dir':  os.path.join(current_directory, 'img.dat'), 
            'line_dat_dir': os.path.join(current_directory, 'line.dat'), 
            'xanes_dat_dir':os.path.join(current_directory, 'xanes.dat'), 
            'fly_dat_dir':  os.path.join(current_directory, 'fly.dat'), 
            'mda_dir':      os.path.join(current_directory, 'mda'), 
            'pca_dir':      os.path.join(current_directory, 'pca.dat'), 
            'XRFmaps_dir':  os.path.join(current_directory, 'img.dat'), 
            'XRFmaps_names':[''], 
            'XRFmaps_id':0, 
            'print_annotations':1, 
            'black_background':0, 
            'max_spec_channels':2048L, 
            'max_spectra':4096L, 
            'max_ICs':6L}
    
    
    # Get info from maps_settings.txt
    total_number_detectors = 1
    max_no_processors = 1
    write_hdf = 0
    use_fit = 0
    
    quick_dirty = 0
    xrf_bin = 0    
    
    nnls = 0    
    standard_filenames = []
    maps_settingsfile = 'maps_settings.txt'     
    try:
        f = open(os.path.join(main['master_dir'],maps_settingsfile), 'rt')
        for line in f:
            if ':' in line : 
                slist = line.split(':')
                tag = slist[0]
                value = ''.join(slist[1:])
                
                if   tag == 'VERSION': version = float(value)
                elif tag == 'DETECTOR_ELEMENTS' : total_number_detectors  =  int(value)
                elif tag == 'MAX_NUMBER_OF_FILES_TO_PROCESS' : max_no_processors_files = int(value)
                elif tag == 'MAX_NUMBER_OF_LINES_TO_PROCESS' : max_no_processors_lines = int(value)
                elif tag == 'WRITE_HDF5' : write_hdf = int(value)
                elif tag == 'USE_FIT' : use_fit = int(value)
                elif tag == 'QUICK_DIRTY'  :  quick_dirty  = int(value)
                elif tag == 'XRF_BIN'  :  xrf_bin  = int(value)
                elif tag == 'NNLS'  :  nnls  = int(value)
                elif tag == 'XANES_SCAN'  :  xanes_scan  = int(value)
                elif tag == 'DETECTOR_TO_START_WITH'  :  detector_to_start_with  = int(value)
                elif tag == 'BEAMLINE'  :  main['beamline']  = str(value).strip()
                elif tag == 'STANDARD'  :  standard_filenames.append(str(value).strip())
                

        f.close()
                  
    except: print 'Could not open maps_settings.txt.'    

    
    me = maps_elements.maps_elements()
    info_elements = me.get_element_info()
    
    maps_def = maps_definitions.maps_definitions()
    maps_conf = maps_def.set_maps_definitions(main['beamline'], info_elements)

    print 'main beamline: ', main['beamline'], '  maps_config version: ', str(version)
        
    select_beamline(main, maps_conf, main['beamline'])
    calib = maps_calibration.calibration(main, maps_conf)
    
    for st in range(len(standard_filenames)):
        print 'Started reading in standards from:', standard_filenames[st]
    
        nbs = calib.read_nbsstds(os.path.join(main['master_dir'],standard_filenames[st]))
    
        if ("1832" in standard_filenames[st]) :
            maps_conf.nbs32 = nbs
        if ("1833" in standard_filenames[st]) :
            maps_conf.nbs33 = nbs
  
    print 'total number of detectors:', total_number_detectors
   
    
  
    #Section a converts mda to h5 and does ROI and ROI+ fits
    if (a > 0) :
        print '\n Section A \n'
        check_output_dirs(main) 
        #maps_test_xrffly
        maps_conf.use_fit = 0        

        filenames = []
        dirList=os.listdir(main['mda_dir'])
        for fname in dirList:
            if fname[-4:] == '.mda' : 
                filenames.append(fname)
        no_files =len(filenames)

        #If no .mda files were found look for .h5
        if no_files == 0: 
            for fname in dirList:
                if fname[-3:] == '.h5' : 
                    filenames.append(fname)  
        no_files = len(filenames)
        
        if no_files == 0:              
            print 'Did not find any .mda files in /mda directory.'
            return    
        
        make_maps.main(wdir=wdir, no_fit = True)
        
#        for this_detector in range(0, total_number_detectors): 
#            header, scan_ext= os.path.splitext(filenames[0])
#            print header
#            mdafilename = os.path.join(main['mda_dir'],header+scan_ext)
#            print 'doing filen #: ',  mdafilename
#            makemaps = maps_generate_img_dat.analyze(info_elements, main, maps_conf, use_fit = maps_conf.use_fit)
#            makemaps.generate_img_dat_threaded(header, mdafilename, this_detector, total_number_detectors, quick_dirty, nnls, 
#                                               max_no_processors_lines, xrf_bin)

            
            


    #Section b loads 8 largest h5 files, fits them and saves fit parameters 
    if (b > 0): 
        print '\n Section B \n'
        for this_detector_element in range(total_number_detectors):
            print 'this_detector_element', this_detector_element, 'total_number_detectors', total_number_detectors
            if (total_number_detectors > 1) :
                suffix = str(this_detector_element)
            else : suffix = ''

           
            # if b then lets load the 4
            # largest img.at files, extract the spectra, and do the fits, then
            # rename the average override file

        
            main['XRFmaps_dir'] = main['img_dat_dir']
            files = os.listdir(main['XRFmaps_dir'])
            imgdat_filenames = []
            extension = '.h5'+suffix
            for f in files:
                if extension in f.lower():
                    imgdat_filenames.append(f)
            imgdat_filenames.sort()
      
            if len(imgdat_filenames) > 8 : 
                imgdat_filesizes = np.zeros((len(imgdat_filenames)))
                for ii in range(len(imgdat_filenames)): 
                    fsize = os.path.getsize(os.path.join(main['img_dat_dir'],imgdat_filenames[ii]))
                    imgdat_filesizes[ii] = fsize               
                    #print imgdat_filenames[ii], imgdat_filesizes[ii]
                sorted_index = np.argsort(np.array(imgdat_filesizes))
                imgdat_filenames = [imgdat_filenames for (imgdat_filesizes,imgdat_filenames) in sorted(zip(imgdat_filesizes,imgdat_filenames))]
                imgdat_filenames.reverse()
                
                imgdat_filenames = imgdat_filenames[0:8]
                       
            print '8 largest h5 files:', imgdat_filenames
           
            
            main['XRFmaps_names'] = imgdat_filenames
 
            main['XRFmaps_id'] = 0


            spectra_filenames = []
            #Get integrated spectra from .h5 files and save them as text files
            for ii in range(len(imgdat_filenames)):
                
                sfile = os.path.join(main['XRFmaps_dir'], imgdat_filenames[ii])
            

                this_filename = 'intspec'+imgdat_filenames[ii]+'.txt'
                savefile = os.path.join(main['output_dir'], this_filename)
                test = check_output_dirs(main)
                if test == 0 :
                    return                 
                save_spectrum(main, sfile, savefile)
                spectra_filenames.append(savefile)
              

            #Load spectra into spectra structure 
            spectra = maps_def.define_spectra(main['max_spec_channels'], main['max_spectra'], main['max_ICs'], mode = 'plot_spec')

            if len(spectra_filenames) == 1:
                    load_spectrum(main, spectra_filenames[0], spectra, append=0)
                
            if len(spectra_filenames) > 1:
                for iii in range(len(spectra_filenames)):
                    load_spectrum(main, spectra_filenames[iii], spectra)
    
       
        
            # now start the fitting of the integrated spectra we just loaded
            fp = maps_fit_parameters.maps_fit_parameters()
            fitp = fp.define_fitp(main['beamline'], info_elements)   
            fitp.g.no_iters = 4             
            #this_w_uname = "DO_FIT_ALL_W_TAILS"
            this_w_uname = "DO_MATRIX_FIT" 
            dofit_spec = 1
            avg_fitp = fp.define_fitp(main['beamline'], info_elements)
#             if (first_run == 1) and (this_detector_element == 0) : 
#                 fitp, avg_fitp, spectra = calib.do_fits(this_w_uname, fitp, dofit_spec, spectra, maxiter = 10, per_pix = 1, generate_img = 1,  suffix = suffix, info_elements = info_elements)  # do the first fit twice, because the very first spectrum is nevere fitted right (not sure why), need to fix it later
#                 first_run = 0
            fitp, avg_fitp, spectra = calib.do_fits(this_w_uname, fitp, dofit_spec, spectra, maxiter = 500, per_pix = 1, generate_img = 1, suffix = suffix, info_elements = info_elements) 
    

    
            #move AND rename the old AND new override files:
            try:
                os.remove(os.path.join(current_directory, 'old_maps_fit_parameters_override.txt'))
            except:
                pass
            try:
                os.remove(os.path.join(current_directory, 'maps_fit_parameters_override.txt'+suffix))
            except:
                pass            
            if total_number_detectors <= 1 :
                try:
                    os.rename(os.path.join(current_directory,'maps_fit_parameters_override.txt'), 
                              os.path.join(current_directory,'old_maps_fit_parameters_override.txt')) 
                except:
                    print 'could not rename file', os.path.join(current_directory,'maps_fit_parameters_override.txt')
            try:
                os.remove(os.path.join(current_directory, 'maps_fit_parameters_override.txt'+suffix))
            except:
                pass            
            os.rename(os.path.join(current_directory,'average_resulting_maps_fit_parameters_override.txt'),
                os.path.join(current_directory,'maps_fit_parameters_override.txt'+suffix))
            
            
        dirlist = os.listdir(current_directory)
        if 'output_old' in dirlist: 
            #print ' delete files in output_old directory'
            filelist = os.listdir(os.path.join(current_directory,'output_old'))
            for fl in filelist:
                thisfile = os.path.join(os.path.join(current_directory,'output_old'), fl)
                os.remove(thisfile)
        else:
            os.makedirs(os.path.join(current_directory,'output_old'))
            
        #Copy files to output_fits
        src_files = os.listdir(os.path.join(current_directory,'output'))
        print src_files
        for fn in src_files:
            full_file_name = os.path.join(os.path.join(current_directory,'output'), fn)
            if (os.path.isfile(full_file_name)):
                shutil.copy(full_file_name, os.path.join(current_directory,'output_old'))     
                os.remove(full_file_name)      
        

 
    #Section c converts mda to h5 files and does ROI/ROI+/FITS
    if (c > 0): 
        print '\n Section C \n'
        #Call make_maps and force fitting. Overrides USE_FIT in maps_setting.txt
        make_maps.main(wdir = current_directory, force_fit = 1)          

        dirlist = os.listdir(current_directory)
        if 'output.fits' in dirlist: 
            #print ' delete files in output.fits directory'
            filelist = os.listdir(os.path.join(current_directory,'output.fits'))
            for fl in filelist:
                thisfile = os.path.join(os.path.join(current_directory,'output.fits'), fl)
                os.remove(thisfile)
        else:
            os.makedirs(os.path.join(current_directory,'output.fits'))
            
        #Copy files to output_fits
        src_files = os.listdir(os.path.join(current_directory,'output'))
        for file_name in src_files:
            full_file_name = os.path.join(os.path.join(current_directory,'output'), file_name)
            if (os.path.isfile(full_file_name)):
                shutil.copy(full_file_name, os.path.join(current_directory,'output.fits'))     
                os.remove(full_file_name)
                                

    #Section d extracts images
    if (d > 0): 
        
        print 'Image extraction not implemented.'
        
#         main['XRFmaps_dir'] = main['img_dat_dir']
#         files = os.listdir(main['XRFmaps_dir'])
#         imgdat_filenames = []
#         extension = '.h5'
#         for f in files:
#             if extension in f.lower():
#                 imgdat_filenames.append(f)
#         
#         no_files = len(imgdat_filenames)
#         current_directory = main['master_dir']
#         main['XRFmaps_names'] = imgdat_filenames
# 
#         
#         main['XRFmaps_id'] = 0
#         
#         temp_string = []
#         try:
#             f = open('maps_fit_parameters_override.txt', 'rt')
#             for line in f:
#                 if ':' in line : 
#                     slist = line.split(':')
#                     tag = slist[0]
#                     value = ''.join(slist[1:])
#     
#                     
#                     if tag == 'ELEMENTS_TO_FIT' :  
#                         temp_string = value.split(',')
#                         temp_string = [x.strip() for x in temp_string]    
#     
#                                
#             f.close()
#         except:
#             print 'Could not read maps_fit_parameters_override.txt'
#         
#         test_string = ['abs_ic', 'H_dpc_cfg', 'V_dpc_cfg', 'phase']
#         for istring in temp_string:
#             test_string.append(istring)
#         test_string.append('s_a')
# 
#         maps_tools.extract_all(main, test_string)


    #Generate average images
    if (total_number_detectors > 1) : 
        print ' we are now going to create the maps_generate_average...'
        n_channels = 2048
        energy_channels = spectra[0].calib['off'] + spectra[0].calib['lin'] * np.arange((n_channels), dtype=np.float)
        makemaps = maps_generate_img_dat.analyze(info_elements, main, maps_conf)
        makemaps.generate_average_img_dat(total_number_detectors, maps_conf, energy_channels)
 
            

    #Section e adds exchange information
    if (e > 0): 
        print 'Adding exchange information'
        ch5 = maps_hdf5.h5()
        
        ch5.add_exchange(main, maps_conf)

        
    
    print 'time started:  ', time_started
    print 'time finished: ', strftime("%Y-%m-%d %H:%M:%S", gmtime())
        
    return

#-----------------------------------------------------------------------------   
if __name__ == '__main__':
    
    
    import sys
    dirct = sys.argv[1]
    print dirct
    
    a=0
    b=0
    c=0
    d=0
    e=0
    
    options, extraParams = getopt.getopt(sys.argv[2:], 'abcde', ['a', 'b', 'c','d','f','full'])
    for opt, arg in options:
        if opt in ('-a', '--a'):
            a=1
        elif opt in ('-b', '--b'):
            b=1
        elif opt in ('-c', '--c'):
            c=1
        elif opt in ('-d', '--d'):
            d=1
        elif opt in ('-e', '--e'):
            e=1                                    
        elif opt in ('--full'):
            a=1
            b=1
            c=1
            d=1
            e=1  
        
    main(wdir=dirct, a=a, b=b, c=c, d=d, e=e)
