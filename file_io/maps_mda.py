'''
Created on Nov 9, 2011

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

from __future__ import division
from xdrlib import *
from file_util import open_file_with_retry, call_function_with_retry
import string
import h5py
import os

import numpy as np

#import scan_data

#----------------------------------------------------------------------
class scanPositioner:
    def __init__(self):
        self.number = 0
        self.fieldName = ""
        self.name = ""
        self.desc = ""
        self.step_mode = ""
        self.unit = ""
        self.readback_name = ""
        self.readback_desc = ""
        self.readback_unit = ""
        self.data = []
        
#----------------------------------------------------------------------
class scanDetector:
    def __init__(self):
        self.number = 0
        self.fieldName = ""
        self.name = ""
        self.desc = ""
        self.unit = ""
        self.data = []
  
#----------------------------------------------------------------------
class scanTrigger:
    def __init__(self):
        self.number = 0
        self.name = ""
        self.command = 0.0      
        
#----------------------------------------------------------------------        

class scanInfo:
    def __init__(self):
        self.rank = 0
        self.dims = 0
        self.spectrum = 0
        self.no_positioners = 0
        self.no_detectors = 0
        self.no_triggers = 0
        self.time = ''
        self.no_extra_pvs = 0
        
#----------------------------------------------------------------------        

class scanData:
    def __init__(self):
        self.rank = 0
        self.dim = 0
        self.npts = 0
        self.curr_pt = 0
        self.scan_name = ""
        self.time = ""
        self.no_positioners = 0
        self.p = []                # list of scanPositioner instances
        self.no_detectors = 0
        self.d = []                # list of scanDetector instances
        self.no_triggers = 0
        self.t = []                # list of scanTrigger instances

        
        
#----------------------------------------------------------------------

class scanClass:
    def __init__(self):
        self.rank = 0
        self.npts = 0
        self.curr_pt = 0
        self.plower_scans = 0
        self.name = ""
        self.time = ""
        self.no_positioners = 0
        self.no_detectors = 0
        self.no_triggers = 0
        self.p = []
        self.d = []
        self.t = []
        

class scan:
    def __init__(self):
        self.scan_name = ''
        self.scan_time_stamp = ''
        self.mca_calib_arr = []  #mca calibration array
        self.mca_calib_description_arr = []  #mca calib description array
        self.y_coord_arr = []    #y coordinates in mm
        self.x_coord_arr = []    #x coordinates in mm
        self.y_pixels = 0        #m pixel
        self.x_pixels = 0        #n pixel
        self.detector_arr = []   #nxmxo array  ( o detectors)
        self.detector_description_arr = []    #ox1 array
        self.mca_arr = []        #nxmx2000xno.detectors array  ( 2000 energies)
        self.extra_pv = []
        self.extra_pv_key_list = []

#----------------------------------------------------------------------

def detName(i):
    if i < 15:
        return string.upper("D%s"%(hex(i+1)[2]))
    elif i < 85:
        return "D%02d"%(i-14)
    else:
        return "?"
    
#----------------------------------------------------------------------

def posName(i):
    if i < 4:
        return "P%d" % (i+1)
    else:
        return "?"
    
    

#----------------------------------------------------------------------
class mda:
    def __init__(self):
        pass
        #self.scan = scan_data.scan()
    
#----------------------------------------------------------------------
    def read_scan_info(self, filename):
        
        verbose = False
        
        file = open_file_with_retry(str(filename), 'rb')
        if file == None:
            return -1
        
        if verbose:
            print 'File: ', filename
            
        scan_info = scanInfo()
            
        buf = file.read(100)        # to read header for scan of up to 5 dimensions
        u = Unpacker(buf)

        # read file header
        version = u.unpack_float()
        scan_number = u.unpack_int()
        rank = u.unpack_int()
        
        dimensions = u.unpack_farray(rank, u.unpack_int)
        isRegular = u.unpack_int()
        extra_pv_ptr = u.unpack_int()
        pmain_scan = file.tell() - (len(buf) - u.get_position())
        
        file.seek(pmain_scan)
        
        
        scan_info.rank = rank
        scan_info.dims = dimensions
        
        scan_info.spectrum = [-1]*rank
        scan_info.no_positioners = [-1]*rank
        scan_info.no_detectors = [-1]*rank
        scan_info.no_triggers = [-1]*rank
        scan_info.time = ['']*rank
        
        
        if verbose:
            print 'Version: ', version
            print 'Scan no: ', scan_number
            print 'Rank:    ', rank
            print 'Dims:    ', dimensions
            print 'Scan is Regular:', isRegular
            print 'Pointer to extra pvs:', extra_pv_ptr
        
        
        remaining_ranks = np.arange(1, rank+1)
        
        i_pointer = 0l
        ptr = np.array([])
        
        while remaining_ranks[0] != -1:
            #pointer points to position of scan
            #read the scan header
            
            buf = file.read(10000) # enough to read scan header
            u = Unpacker(buf)
            
            scan_rank = u.unpack_int()
            scan_npts = u.unpack_int()
            scan_cpt = u.unpack_int()
      
            
            if verbose:
                print 'Scan rank:', scan_rank
                print 'Scan npts:', scan_npts
                print 'Scan cpt: ', scan_cpt    
                

            
            #if the rank of this scan is *new* store the into and read on
            # otherwise just skip to the next scan
            rank_i = (remaining_ranks == scan_rank).nonzero()
            rank_i = rank_i[0]
                        
            if rank_i != -1:
                if verbose:
                    print 'NEW scan type found: ', scan_rank
                
                # read the ptr to the sub-scans of this scan, and 
                #introduce it to the ptr array
                if scan_rank > 1:
                    tmp_ptr = u.unpack_farray(scan_npts, u.unpack_int)

                    if ptr.size:
                        ptr = np.hstack((ptr, tmp_ptr))
                        ptr.sort()
                    else:
                        tmp_ptr=np.array(tmp_ptr)
                        ptr = tmp_ptr.copy()
                    if verbose:
                        print 'ptr = ', ptr
                    
                # The aim is to determine which (if any) of the scan dimensions is an MCA scan.
                # AT PRESENT, a scan is determined to be an MCA scan if it satisfies:
                # must be inner loop of scan  (info.name = crate:scanH)
                # and must have no positioners    (info.no_positioners = 1)
                # which are deemed to be necessary and sufficient.

                # read first scan information
                
                namelength = u.unpack_int()
                scan_name = u.unpack_string()

                timelength = u.unpack_int()
                scan_time = u.unpack_string()

                scan_no_positioners = u.unpack_int()

                scan_no_detectors = u.unpack_int()

                scan_no_triggers = u.unpack_int()
                
                
                if verbose:
                    print 'Scan name: ', scan_name
                    print 'Scan time: ', scan_time
                    print 'no_positioners ', scan_no_positioners
                    print 'no_detectors ', scan_no_detectors
                    print 'no_triggers ', scan_no_triggers
                
                # test to determine whether scan is a spectrum or not
                test_string = scan_name.split('scan')

                if len(test_string) > 1:
                    test_string = test_string[1]
                else:
                    test_string = ' '
                if (test_string == 'H') and (scan_no_positioners == 0):
                    if verbose:
                        print 'Found an MCA scan'
                    scan_spectrum = 1
                else:
                    if verbose:
                        print 'Not a MCA scan'
                    scan_spectrum = 0
                    
                scan_info.no_positioners[rank-scan_rank] = scan_no_positioners
                scan_info.no_detectors[rank-scan_rank] = scan_no_detectors
                scan_info.no_triggers[rank-scan_rank] = scan_no_triggers
                scan_info.time[rank-scan_rank] = scan_time
                scan_info.spectrum[rank-scan_rank] = scan_spectrum
                    
                if len(remaining_ranks) > 1:
                    indx = np.where(remaining_ranks != scan_rank)
                    remaining_ranks = remaining_ranks[indx[0]]
                else:
                    if verbose:
                        print 'All ranks located.'
                    break
                
                    
            #move the file pointer to the next position if it exists
            if i_pointer < ptr.size-1:
                file.seek(ptr[i_pointer], 0)
                i_pointer=i_pointer+1     
            else:
                break

      
        self.no_extra_pv = 0l
        file.seek(extra_pv_ptr, 0)
        buf = file.read(100)       
        u = Unpacker(buf)
        self.no_extra_pv = u.unpack_int()
        
        if verbose:
            print 'No extra pvs: ', self.no_extra_pv
            
        if (self.no_extra_pv > 10000l) or (self.no_extra_pv < 0l):
            # if the number of extra PVs is very large, it is likely that there is going to
            # be a problem with the file. attempting to read the extra pvs can then cause
            # the program to crash. 
            print 'error: cannot read the number of extra PVs accurately. set them to zero to be on the safe side'
            self.no_extra_pv = 0L
        
        file.close()
        if verbose:
            print 'Finished reading scan info.'
        
        return scan_info

#----------------------------------------------------------------------    
    def read_scan_data(self, file):
        
        v=0
        scan = scanClass()
        
        buf = file.read(5000) # enough to read scan header
        u = Unpacker(buf)
        
        scan.rank = u.unpack_int()
        if v: print "scan.rank = ", `scan.rank`
        scan.npts = u.unpack_int()
        if v: print "scan.npts = ", `scan.npts`
        scan.curr_pt = u.unpack_int()
        if v: print "scan.curr_pt = ", `scan.curr_pt`
        if (scan.rank > 1):
            # if curr_pt < npts, plower_scans will have garbage for pointers to
            # scans that were planned for but not written
            scan.plower_scans = u.unpack_farray(scan.npts, u.unpack_int)
            if v: print "scan.plower_scans = ", `scan.plower_scans`
        namelength = u.unpack_int()
        scan.name = u.unpack_string()
        if v: print "scan.name = ", `scan.name`
        timelength = u.unpack_int()
        scan.time = u.unpack_string()
        if v: print "scan.time = ", `scan.time`
        scan.no_positioners = u.unpack_int()
        if v: print "scan.no_positioners = ", `scan.no_positioners`
        scan.no_detectors = u.unpack_int()
        if v: print "scan.no_detectors = ", `scan.no_detectors`
        scan.no_triggers = u.unpack_int()
        if v: print "scan.no_triggers = ", `scan.no_triggers`
        
        for j in range(scan.no_positioners):
            scan.p.append(scanPositioner())
            scan.p[j].number = u.unpack_int()
            scan.p[j].fieldName = posName(scan.p[j].number)
            if v: print "positioner ", j
            length = u.unpack_int() # length of name string
            if length: scan.p[j].name = u.unpack_string()
            if v: print "scan.p[%d].name = %s" % (j, `scan.p[j].name`)
            length = u.unpack_int() # length of desc string
            if length: scan.p[j].desc = u.unpack_string()
            if v: print "scan.p[%d].desc = %s" % (j, `scan.p[j].desc`)
            length = u.unpack_int() # length of step_mode string
            if length: scan.p[j].step_mode = u.unpack_string()
            if v: print "scan.p[%d].step_mode = %s" % (j, `scan.p[j].step_mode`)
            length = u.unpack_int() # length of unit string
            if length: scan.p[j].unit = u.unpack_string()
            if v: print "scan.p[%d].unit = %s" % (j, `scan.p[j].unit`)
            length = u.unpack_int() # length of readback_name string
            if length: scan.p[j].readback_name = u.unpack_string()
            if v: print "scan.p[%d].readback_name = %s" % (j, `scan.p[j].readback_name`)
            length = u.unpack_int() # length of readback_desc string
            if length: scan.p[j].readback_desc = u.unpack_string()
            if v: print "scan.p[%d].readback_desc = %s" % (j, `scan.p[j].readback_desc`)
            length = u.unpack_int() # length of readback_unit string
            if length: scan.p[j].readback_unit = u.unpack_string()
            if v: print "scan.p[%d].readback_unit = %s" % (j, `scan.p[j].readback_unit`)

        for j in range(scan.no_detectors):
            scan.d.append(scanDetector())
            scan.d[j].number = u.unpack_int()
            scan.d[j].fieldName = detName(scan.d[j].number)
            if v: print "detector ", j
            length = u.unpack_int() # length of name string
            if length: scan.d[j].name = u.unpack_string()
            if v: print "scan.d[%d].name = %s" % (j, `scan.d[j].name`)
            length = u.unpack_int() # length of desc string
            if length: scan.d[j].desc = u.unpack_string()
            if v: print "scan.d[%d].desc = %s" % (j, `scan.d[j].desc`)
            length = u.unpack_int() # length of unit string
            if length: scan.d[j].unit = u.unpack_string()
            if v: print "scan.d[%d].unit = %s" % (j, `scan.d[j].unit`)

        for j in range(scan.no_triggers):
            scan.t.append(scanTrigger())
            scan.t[j].number = u.unpack_int()
            if v: print "trigger ", j
            length = u.unpack_int() # length of name string
            if length: scan.t[j].name = u.unpack_string()
            if v: print "scan.t[%d].name = %s" % (j, `scan.t[j].name`)
            scan.t[j].command = u.unpack_float()
            if v: print "scan.t[%d].command = %s" % (j, `scan.t[j].command`)

        ### read data
        # positioners
        file.seek(file.tell() - (len(buf) - u.get_position()))
        buf = file.read(scan.no_positioners * scan.npts * 8)
        u = Unpacker(buf)
        for j in range(scan.no_positioners):
            if v: print "read %d pts for pos. %d at buf loc %x" % (scan.npts,
                                                                   j, u.get_position())
            scan.p[j].data = u.unpack_farray(scan.npts, u.unpack_double)    
            if v: print "scan.p[%d].data = %s" % (j, `scan.p[j].data`)
        
        # detectors
        file.seek(file.tell() - (len(buf) - u.get_position()))
        buf = file.read(scan.no_detectors * scan.npts * 4)
        u = Unpacker(buf)
        for j in range(scan.no_detectors):
            scan.d[j].data = u.unpack_farray(scan.npts, u.unpack_float)
            if v: print "scan.d[%d].data = %s" % (j, `scan.d[j].data`)

        return scan
        
        
#----------------------------------------------------------------------
    def read_mda(self, filename):
        
        
        verbose = False
        scan_info = self.read_scan_info(filename)
        
        ''' Create a structure to contain the file info, piece by piece    
            1 corresponds to innermost loop, 2 to the next outer loop, 3 typically to the
            outermost loop '''
        if scan_info.rank >= 4: 
            print 'This file has too deep dimensions, I cannot read it and will skip'
            return -1
 
        file = open_file_with_retry(str(filename), 'rb')
        if file == None:
            return -1
 
        buf = file.read(100)        # to read header for scan of up to 5 dimensions
        u = Unpacker(buf)

        # read file header
        version = u.unpack_float()
        scan_number = u.unpack_int()
        rank = u.unpack_int()
        
        dimensions = u.unpack_farray(rank, u.unpack_int)
        isRegular = u.unpack_int()
        extra_pv_ptr = u.unpack_int()
        pmain_scan = file.tell() - (len(buf) - u.get_position())
        
        file.seek(pmain_scan)
        
        dim = []

        for i in range(rank):
            dim.append(scanData())
            dim[i].dim = i+1
            dim[i].rank = rank-i

        #read the first 1D scan 
        s0 = self.read_scan_data(file)
        dim[0].npts = s0.npts
        dim[0].curr_pt = s0.curr_pt
        dim[0].scan_name = s0.name
        dim[0].time = s0.time
        dim[0].no_positioners = s0.no_positioners
        for i in range(s0.no_positioners): dim[0].p.append(s0.p[i])
        
        dim[0].no_triggers = s0.no_triggers
        for j in range(s0.no_triggers): dim[0].t.append(s0.t[j])
        
        dim[0].no_detectors = s0.no_detectors
        for i in range(s0.no_detectors): dim[0].d.append(s0.d[i])
       
       
        
        if (rank > 1):
        # collect 2D data
            for i in range(s0.curr_pt):
                file.seek(s0.plower_scans[i])
                s = self.read_scan_data(file)
                if i == 0:
                    dim[1].npts = s.npts
                    dim[1].curr_pt = s.curr_pt
                    dim[1].scan_name = s.name
                    dim[1].time = s.time
                    # copy positioner, trigger, detector instances
                    dim[1].no_positioners = s.no_positioners
                    for j in range(s.no_positioners):
                        dim[1].p.append(s.p[j])
                        tmp = s.p[j].data[:]
                        dim[1].p[j].data = []
                        dim[1].p[j].data.append(tmp)
                    dim[1].no_triggers = s.no_triggers
                    for j in range(s.no_triggers): dim[1].t.append(s.t[j])
                    dim[1].no_detectors = s.no_detectors
                    for j in range(s.no_detectors):
                        dim[1].d.append(s.d[j])
                        tmp = s.d[j].data[:]
                        dim[1].d[j].data = []
                        dim[1].d[j].data.append(tmp)
                else:
                    # append data arrays
                    for j in range(s.no_positioners): dim[1].p[j].data.append(s.p[j].data)
                    for j in range(s.no_detectors): dim[1].d[j].data.append(s.d[j].data) 
                    
        
        if (rank > 2):
        # collect 3D data
            for i in range(s0.curr_pt):
                file.seek(s0.plower_scans[i])
                s1 = self.read_scan_data(file)
                for j in range(s1.curr_pt):
                    file.seek(s1.plower_scans[j])
                    s = self.read_scan_data(file)
                    if ((i == 0) and (j == 0)):
                        dim[2].npts = s.npts
                        dim[2].curr_pt = s.curr_pt
                        dim[2].scan_name = s.name
                        dim[2].time = s.time
                        # copy positioner, trigger, detector instances
                        dim[2].no_positioners = s.no_positioners
                        for k in range(s.no_positioners):
                            dim[2].p.append(s.p[k])
                            tmp = s.p[k].data[:]
                            dim[2].p[k].data = [[]]
                            dim[2].p[k].data[i].append(tmp)
                        dim[2].no_triggers = s.no_triggers
                        for k in range(s.no_triggers): dim[2].t.append(s.t[k])
                        dim[2].no_detectors = s.no_detectors
                        for k in range(s.no_detectors):
                            dim[2].d.append(s.d[k])
                            tmp = s.d[k].data[:]
                            dim[2].d[k].data = [[]]
                            dim[2].d[k].data[i].append(tmp)
                    elif j == 0:
                        for k in range(s.no_positioners):
                            dim[2].p[k].data.append([])
                            dim[2].p[k].data[i].append(s.p[k].data)
                        for k in range(s.no_detectors):
                            dim[2].d[k].data.append([])
                            dim[2].d[k].data[i].append(s.d[k].data)
                    else:
                        # append data arrays
                        for k in range(s.no_positioners): dim[2].p[k].data[i].append(s.p[k].data)
                        for k in range(s.no_detectors): dim[2].d[k].data[i].append(s.d[k].data)
  
        print 'read 3d data'
        # Collect scan-environment variables into a dictionary
        dict = {}
        dict['sampleEntry'] = ("description", "unit string", "value")
        dict['filename'] = filename
        dict['rank'] = rank
        dict['dimensions'] = dimensions                    
        
        if extra_pv_ptr:
            file.seek(extra_pv_ptr)
            buf = file.read()       # Read all scan-environment data
            u = Unpacker(buf)
            numExtra = u.unpack_int()
            for i in range(numExtra):
                name = ''
                n = u.unpack_int()      # length of name string
                if n: name = u.unpack_string()
                desc = ''
                n = u.unpack_int()      # length of desc string
                if n: desc = u.unpack_string()
                type = u.unpack_int()

                unit = ''
                value = ''
                count = 0
                if type != 0:   # not DBR_STRING
                    count = u.unpack_int()  # 
                    n = u.unpack_int()      # length of unit string
                    if n: unit = u.unpack_string()

                if type == 0: # DBR_STRING
                    n = u.unpack_int()      # length of value string
                    if n: value = u.unpack_string()
                elif type == 32: # DBR_CTRL_CHAR
                    #value = u.unpack_fstring(count)
                    v = u.unpack_farray(count, u.unpack_int)
                    value = ""
                    for i in range(len(v)):
                        # treat the byte array as a null-terminated string
                        if v[i] == 0: break
                        value = value + chr(v[i])

                elif type == 29: # DBR_CTRL_SHORT
                    value = u.unpack_farray(count, u.unpack_int)
                elif type == 33: # DBR_CTRL_LONG
                    value = u.unpack_farray(count, u.unpack_int)
                elif type == 30: # DBR_CTRL_FLOAT
                    value = u.unpack_farray(count, u.unpack_float)
                elif type == 34: # DBR_CTRL_DOUBLE
                    value = u.unpack_farray(count, u.unpack_double)
                    
                dict[name] = (desc, unit, value)
                
        dim.reverse()
        dim.append(dict)
        dim.reverse()
        if verbose:
            print "%s is a %d-D file; %d dimensions read in." % (filename, dim[0]['rank'], len(dim)-1)
            print "dim[0] = dictionary of %d scan-environment PVs" % (len(dim[0]))
            print "   usage: dim[0]['sampleEntry'] ->", dim[0]['sampleEntry']
            for i in range(1,len(dim)):
                print "dim[%d] = %s" % (i, str(dim[i]))
            print "   usage: dim[1].p[2].data -> 1D array of positioner 2 data"
            print "   usage: dim[2].d[7].data -> 2D array of detector 7 data"
            
            
        file.close()
        
        return dim
        
        
#    if help:
#        print " "
#        print "   each dimension (e.g., dim[1]) has the following fields: "
#        print "      time      - date & time at which scan was started: %s" % (dim[1].time)
#        print "      scan_name - name of scan record that acquired this dimension: '%s'" % (dim[1].scan_name)
#        print "      curr_pt   - number of data points actually acquired: %d" % (dim[1].curr_pt)
#        print "      npts      - number of data points requested: %d" % (dim[1].npts)
#        print "      nd        - number of detectors for this scan dimension: %d" % (dim[1].nd)
#        print "      d[]       - list of detector-data structures"
#        print "      np        - number of positioners for this scan dimension: %d" % (dim[1].np)
#        print "      p[]       - list of positioner-data structures"
#        print "      nt        - number of detector triggers for this scan dimension: %d" % (dim[1].nt)
#        print "      t[]       - list of trigger-info structures"
#
#    if help:
#        print " "
#        print "   each detector-data structure (e.g., dim[1].d[0]) has the following fields: "
#        print "      desc      - description of this detector"
#        print "      data      - data list"
#        print "      unit      - engineering units associated with this detector"
#        print "      fieldName - scan-record field (e.g., 'D01')"


#    if help:
#        print " "
#        print "   each positioner-data structure (e.g., dim[1].p[0]) has the following fields: "
#        print "      desc          - description of this positioner"
#        print "      data          - data list"
#        print "      step_mode     - scan mode (e.g., Linear, Table, On-The-Fly)"
#        print "      unit          - engineering units associated with this positioner"
#        print "      fieldName     - scan-record field (e.g., 'P1')"
#        print "      name          - name of EPICS PV (e.g., 'xxx:m1.VAL')"
#        print "      readback_desc - description of this positioner"
#        print "      readback_unit - engineering units associated with this positioner"
#        print "      readback_name - name of EPICS PV (e.g., 'xxx:m1.VAL')"

                          

#----------------------------------------------------------------------
    def read_scan(self, filename, threeD_only = 1,  invalid_file = [0],
                  extra_pvs = False, save_ram = 0):
        
        verbose = False
        
        # the following variables are created or read with this routine:
        scan_name =  ' '
        scan_time_stamp = ' '
        mca_calib_arr = 0.              #  array
        mca_calib_description_arr = [] # array
        y_coord_arr = 0.                # y coordinates in mm
        x_coord_arr = [0.]               # x coordinates in mm
        y_pixels = 0                    # m pixels
        x_pixels = 0                    # n pixels
        detector_arr = 0.               # nxmxo array  ( o detectors)
        detector_description_arr = []  #  ox1 array
        mca_arr = 0.                    # nxmx2000xno.detectors array  ( 2000 energies)
        invalid_file[0] = 0
    
    
        res = 0
        ndet = 85                   # 15
        ntot = ndet+4
    
        file = open_file_with_retry(str(filename), 'rb')
        if file == None:
            return -1
 
        buf = file.read(100)        # to read header for scan of up to 5 dimensions
        print 'buf', buf
        u = Unpacker(buf)

        # read file header
        version = u.unpack_float()
        scan_number = u.unpack_int()
        rank = u.unpack_int()
        
        scan_size = u.unpack_farray(rank, u.unpack_int)
        add_scan_s_regular = u.unpack_int()
        pointer_extra_PVs = u.unpack_int()
        pmain_scan = file.tell() - (len(buf) - u.get_position())
    
    
        if verbose:
            print 'Version: ', version
            print 'Scan no: ', scan_number
            print 'Rank:    ', rank
            print 'Dims:    ', scan_size
            print 'Scan is Regular:', add_scan_s_regular
            print 'Pointer to extra pvs:', pointer_extra_PVs
            
                
        
        file.seek(pmain_scan)
        buf = file.read(5000) # enough to read scan header
        u = Unpacker(buf)
            
        scan_rank = u.unpack_int()
        scan_npts = u.unpack_int()
        scan_cpt = u.unpack_int()
        
        if verbose:
            print 'scan_rank: ', scan_rank
            print 'scan_npts: ', scan_npts
            print 'scan_cpt:    ', scan_cpt
            
        if scan_rank > 2048 : return
        
        if scan_cpt <= 0 :
            print 'error: scan_cpt = ', scan_cpt
            invalid_file[0] = 2
            return
        
        #outer_pointer_lower_scans = np.array((scan_header.npts) , dtype =np.int32) # points set in scan
        #readu, lun, outer_pointer_lower_scans
        outer_pointer_lower_scans = u.unpack_farray(scan_npts, u.unpack_int) # points set in scan
    
        if verbose: print 'outer_pointer_lower_scans = ', outer_pointer_lower_scans
        outer_pointer_lower_scans = np.array(outer_pointer_lower_scans)
        outer_pointer_lower_scans = outer_pointer_lower_scans[np.nonzero(outer_pointer_lower_scans)]
        
        # read first scan information  
        namelength = u.unpack_int()
        scan_name = u.unpack_string()

        timelength = u.unpack_int()
        scan_time = u.unpack_string()

        scan_no_positioners = u.unpack_int()

        scan_no_detectors = u.unpack_int()

        scan_no_triggers = u.unpack_int()
                
                
        if verbose:
            print 'Scan name: ', scan_name
            print 'Scan time: ', scan_time
            print 'no_positioners ', scan_no_positioners
            print 'no_detectors ', scan_no_detectors
            print 'no_triggers ', scan_no_triggers
        
        one_d_info = (scan_name, scan_time, scan_no_positioners, scan_no_detectors, scan_no_triggers)
        if verbose: print '1D info: ', one_d_info
        
        if scan_no_detectors > 0 :
            mca_calib_arr = np.zeros((scan_no_detectors)) # create mca calibration array
            mca_calib_description_arr = []                # create mca calib description array
    
        y_pixels = outer_pointer_lower_scans.size # pixels really in the scan
        one_d_time_stamp = []
        
        for j in range(scan_no_positioners):
            positioner = scanPositioner()
            positioner.number = u.unpack_int()
            positioner.fieldName = posName(positioner.number)
            if verbose: print "positioner ", j
            length = u.unpack_int() # length of name string
            if length: positioner.name = u.unpack_string()
            if verbose: print "positioner[%d].name = %s" % (j, `positioner.name`)
            length = u.unpack_int() # length of desc string
            if length: positioner.desc = u.unpack_string()
            if verbose: print "positioner[%d].desc = %s" % (j, `positioner.desc`)
            length = u.unpack_int() # length of step_mode string
            if length: positioner.step_mode = u.unpack_string()
            if verbose: print "positioner[%d].step_mode = %s" % (j, `positioner.step_mode`)
            length = u.unpack_int() # length of unit string
            if length: positioner.unit = u.unpack_string()
            if verbose: print "positioner[%d].unit = %s" % (j, `positioner.unit`)
            length = u.unpack_int() # length of readback_name string
            if length: positioner.readback_name = u.unpack_string()
            if verbose: print "positioner[%d].readback_name = %s" % (j, `positioner.readback_name`)
            length = u.unpack_int() # length of readback_desc string
            if length: positioner.readback_desc = u.unpack_string()
            if verbose: print "positioner[%d].readback_desc = %s" % (j, `positioner.readback_desc`)
            length = u.unpack_int() # length of readback_unit string
            if length: positioner.readback_unit = u.unpack_string()
            if verbose: print "positioner[%d].readback_unit = %s" % (j, `positioner.readback_unit`)
    
        one_d_positioner = positioner
        
        for j in range(scan_no_detectors):
            detector = scanDetector()
            detector.number = u.unpack_int()
            detector.fieldName = detName(detector.number)
            if verbose: print "detector ", j
            try:
                length = u.unpack_int() # length of name string
                if length: detector.name = u.unpack_string()
                if verbose: print "detector[%d].name = %s" % (j, `detector.name`)
                length = u.unpack_int() # length of desc string
                if length: detector.desc = u.unpack_string()
                if verbose: print "detector[%d].desc = %s" % (j, `detector.desc`)
                length = u.unpack_int() # length of unit string
                if length: detector.unit = u.unpack_string()
                if verbose: print "detector[%d].unit = %s" % (j, `detector.unit`)
                mca_calib_description_arr.append(detector.name)
            except:
                print 'read_scan(): Error reading detector description strings'
            
        if verbose: print 'mca_calib_description_arr: ', mca_calib_description_arr

        for j in range(scan_no_triggers):
            trigger = scanTrigger()
            trigger.number = u.unpack_int()
            if verbose: print "trigger ", j
            length = u.unpack_int() # length of name string
            if length: trigger.name = u.unpack_string()
            if verbose: print "trigger[%d].name = %s" % (j, `trigger.name`)
            trigger.command = u.unpack_float()
            if verbose: print "trigger[%d].command = %s" % (j, `trigger.command`)
            
            
    
        ### read data
        # positioners
        file.seek(file.tell() - (len(buf) - u.get_position()))
        buf = file.read(scan_no_positioners * scan_npts * 8)
        u = Unpacker(buf)
        for j in range(scan_no_positioners):
            if verbose: print "read %d pts for pos. %d at buf loc %x" % (scan_npts,
                                                                   j, u.get_position())
            readback_array = u.unpack_farray(scan_npts, u.unpack_double)    
            if verbose: print "readback_array = ",  readback_array
            
            if j == 0 : 
                readback_array = np.array(readback_array)
                y_coord_arr = readback_array.copy()
            if y_coord_arr.size != y_pixels :
                # remove those y positions that are incorrect for aborted scans
                if y_pixels < 3 :
                    invalid_file[0] = 2
                    print 'scanned y_pixels less than 3 in an aborted array'
                    return 
          
                y_coord_arr = y_coord_arr[0:y_pixels]
                y_coord_arr[y_pixels-1] = y_coord_arr[y_pixels-2] + (y_coord_arr[y_pixels-2] - y_coord_arr[y_pixels-3])
                if verbose:
                    print 'y coord array before correction : ', readback_array
                    print 'y coord array after correction : ', y_coord_arr
   
        #if verbose: print 'y coord array after correction : ', y_coord_arr           
       
        # detectors
        file.seek(file.tell() - (len(buf) - u.get_position()))
        buf = file.read(scan_no_detectors * scan_npts * 4)
        u = Unpacker(buf)
        for j in range(scan_no_detectors):
            detector_array = u.unpack_farray(scan_npts, u.unpack_float)
            mca_calib_arr[j] = detector_array[0]
            #if verbose: print "detector_array" , detector_array
        
        if verbose: print 'mca_calib_arr ', mca_calib_arr
        
        for i_outer_loop in range(len(outer_pointer_lower_scans)):
            verbose = 0
            if outer_pointer_lower_scans[i_outer_loop] == 0 :
                print 'skipping rest of scan, because outer_pointer_lower_scans(i_outer_loop) EQ 0 '
                print 'current y position: ', i_outer_loop,  ' of total ', len(outer_pointer_lower_scans)-1
                continue
            
            file.seek(outer_pointer_lower_scans[i_outer_loop])
            buf = file.read(5000) # enough to read scan header
            u = Unpacker(buf)
            
            scan_rank = u.unpack_int()
            scan_npts = u.unpack_int()
            scan_cpt = u.unpack_int()
        
            if verbose:
                print 'scan_rank: ', scan_rank
                print 'scan_npts: ', scan_npts
                print 'scan_cpt:    ', scan_cpt
            
            if scan_rank > 2048 : return
        
            if scan_cpt <= 0 :
                print 'error: scan_cpt = ', scan_cpt
                invalid_file[0] = 2
                return 
        
            x_pixels = scan_npts
            two_d_time_stamp = []
            pointer_lower_scans = np.zeros((scan_npts), dtype=np.int)
            if scan_rank > 1 :
                pointer_lower_scans = u.unpack_farray(scan_npts, u.unpack_int)
       
            if verbose:
                print 'pointer_lower_scans    : ', pointer_lower_scans
                print 'x pixels    : ', x_pixels
   
            if (scan_npts > 2999) :
                invalid_file[0] = 3
                return 
      
        
            # read scan information  
            namelength = u.unpack_int()
            scan_name = u.unpack_string()

            timelength = u.unpack_int()
            scan_time = u.unpack_string()

            scan_no_positioners = u.unpack_int()
            scan_no_detectors = u.unpack_int()
            scan_no_triggers = u.unpack_int()
                
                
            if verbose:
                print 'Scan name: ', scan_name
                print 'Scan time: ', scan_time
                print 'no_positioners ', scan_no_positioners
                print 'no_detectors ', scan_no_detectors
                print 'no_triggers ', scan_no_triggers
               
            one_d_time_stamp.append(scan_time)
            
            #need to define the 2d detector array FOR the first time only
            if i_outer_loop == 0 :
                detector_arr = np.zeros((x_pixels, y_pixels, scan_no_detectors))
                
            
            for j in range(scan_no_positioners):
                positioner = scanPositioner()
                positioner.number = u.unpack_int()
                positioner.fieldName = posName(positioner.number)
                if verbose: print "positioner ", j
                length = u.unpack_int() # length of name string
                if length: positioner.name = u.unpack_string()
                if verbose: print "positioner[%d].name = %s" % (j, `positioner.name`)
                length = u.unpack_int() # length of desc string
                if length: positioner.desc = u.unpack_string()
                if verbose: print "positioner[%d].desc = %s" % (j, `positioner.desc`)
                length = u.unpack_int() # length of step_mode string
                if length: positioner.step_mode = u.unpack_string()
                if verbose: print "positioner[%d].step_mode = %s" % (j, `positioner.step_mode`)
                length = u.unpack_int() # length of unit string
                if length: positioner.unit = u.unpack_string()
                if verbose: print "positioner[%d].unit = %s" % (j, `positioner.unit`)
                length = u.unpack_int() # length of readback_name string
                if length: positioner.readback_name = u.unpack_string()
                if verbose: print "positioner[%d].readback_name = %s" % (j, `positioner.readback_name`)
                length = u.unpack_int() # length of readback_desc string
                if length: positioner.readback_desc = u.unpack_string()
                if verbose: print "positioner[%d].readback_desc = %s" % (j, `positioner.readback_desc`)
                length = u.unpack_int() # length of readback_unit string
                if length: positioner.readback_unit = u.unpack_string()
                if verbose: print "positioner[%d].readback_unit = %s" % (j, `positioner.readback_unit`)
                
                if (i_outer_loop == 0) :
                    two_d_info = (scan_name, scan_time, scan_no_positioners, scan_no_detectors, scan_no_triggers)
                    two_d_positioner = positioner
    
            for j in range(scan_no_detectors):
                detector = scanDetector()
                detector.number = u.unpack_int()
                detector.fieldName = detName(detector.number)
                if verbose: print "detector ", j
                length = u.unpack_int() # length of name string
                if length: detector.name = u.unpack_string()
                if verbose: print "detector[%d].name = %s" % (j, `detector.name`)
                length = u.unpack_int() # length of desc string
                if length: detector.desc = u.unpack_string()
                if verbose: print "detector[%d].desc = %s" % (j, `detector.desc`)
                length = u.unpack_int() # length of unit string
                if length: detector.unit = u.unpack_string()
                if verbose: print "detector[%d].unit = %s" % (j, `detector.unit`)
                
                if (i_outer_loop == 0) :
                    detector_description_arr.append(detector.name)
                    
            for j in range(scan_no_triggers):
                trigger = scanTrigger()
                trigger.number = u.unpack_int()
                if verbose: print "trigger ", j
                length = u.unpack_int() # length of name string
                if length: trigger.name = u.unpack_string()
                if verbose: print "trigger[%d].name = %s" % (j, `trigger.name`)
                trigger.command = u.unpack_float()
                if verbose: print "trigger[%d].command = %s" % (j, `trigger.command`)
                
  
            ### read data
            # positioners
            file.seek(file.tell() - (len(buf) - u.get_position()))
            buf = file.read(scan_no_positioners * scan_npts * 8)
            u = Unpacker(buf)
            for j in range(scan_no_positioners):
                if verbose: print "read %d pts for pos. %d at buf loc %x" % (scan_npts,
                                                                             j, u.get_position())
                readback_array = u.unpack_farray(scan_npts, u.unpack_double)    
                if verbose: print "readback_array = ",  readback_array
            
                if x_coord_arr[0] == 0 :
                    readback_array = np.array(readback_array)
                    x_coord_arr = readback_array.copy()
          
            if verbose: print 'x coord array : ', x_coord_arr
   
                  
#            #This is slow so read directly 
#            # detectors
#            file.seek(file.tell() - (len(buf) - u.get_position()))
#            buf = file.read(scan_no_detectors * scan_npts * 4)
#            u = Unpacker(buf)
#            for j in range(scan_no_detectors):
#                detector_array = u.unpack_farray(scan_npts, u.unpack_float)
#                detector_arr[:, i_outer_loop, j] = detector_array[:]
                
                
            import struct
            # detectors
            file.seek(file.tell() - (len(buf) - u.get_position()))
            detector_array = np.zeros((scan_npts), dtype=np.float32)
            for j in range(scan_no_detectors):
                buf = file.read(scan_npts * 4)
                detector_array = struct.unpack('>'+str(scan_npts)+'f', buf)
                detector_arr[:, i_outer_loop, j] = detector_array[:]
                    
            
            if verbose: print "detector_array" , detector_array
    
            if rank == 2 : continue
            
            temp_timestamp = []
            
            for i_innermost_loop in range(len(pointer_lower_scans)) :
                if (i_innermost_loop > 0) and (rank > 2) :
                    if (pointer_lower_scans[i_innermost_loop] == 0) or \
                        (pointer_lower_scans[i_innermost_loop] < outer_pointer_lower_scans[i_outer_loop] ):
                        print 'skipping rest of line, because either pointer_lower_scans(i_innermost_loop) EQ 0 or lt outer_pointer'
                        print 'current y position: ', i_outer_loop,  ' of total ', len(outer_pointer_lower_scans)-1
                        print 'current x position: ', i_innermost_loop,  ' of total ', len(pointer_lower_scans) -1
                        print 'pointer_lower_scans(i_innermost_loop): ', pointer_lower_scans[i_innermost_loop], \
                        ' outer_pointer_lower_scans(i_outer_loop) : ', outer_pointer_lower_scans[i_outer_loop]
                        continue

       
                    file.seek(pointer_lower_scans[i_innermost_loop])
                    
                              
                buf = file.read(5000) # enough to read scan header
                u = Unpacker(buf)
            
                scan_rank = u.unpack_int()
                scan_npts = u.unpack_int()
                scan_cpt = u.unpack_int()
        
                if verbose:
                    print 'scan_rank: ', scan_rank
                    print 'scan_npts: ', scan_npts
                    print 'scan_cpt:    ', scan_cpt
            
                if scan_rank > 2048 : return
                if scan_cpt == 0 : print  'warning: scan_header.cpt EQ 0 '

                # read scan information  
                namelength = u.unpack_int()
                scan_name = u.unpack_string()

                timelength = u.unpack_int()
                scan_time = u.unpack_string()

                scan_no_positioners = u.unpack_int()
                scan_no_detectors = u.unpack_int()
                scan_no_triggers = u.unpack_int()
                
                
                if verbose:
                    print 'Scan name: ', scan_name
                    print 'Scan time: ', scan_time
                    print 'no_positioners ', scan_no_positioners
                    print 'no_detectors ', scan_no_detectors
                    print 'no_triggers ', scan_no_triggers
               
                temp_timestamp.append(scan_time)
                
    
                for j in range(scan_no_positioners):
                    positioner = scanPositioner()
                    positioner.number = u.unpack_int()
                    positioner.fieldName = posName(positioner.number)
                    if verbose: print "positioner ", j
                    length = u.unpack_int() # length of name string
                    if length: positioner.name = u.unpack_string()
                    if verbose: print "positioner[%d].name = %s" % (j, `positioner.name`)
                    length = u.unpack_int() # length of desc string
                    if length: positioner.desc = u.unpack_string()
                    if verbose: print "positioner[%d].desc = %s" % (j, `positioner.desc`)
                    length = u.unpack_int() # length of step_mode string
                    if length: positioner.step_mode = u.unpack_string()
                    if verbose: print "positioner[%d].step_mode = %s" % (j, `positioner.step_mode`)
                    length = u.unpack_int() # length of unit string
                    if length: positioner.unit = u.unpack_string()
                    if verbose: print "positioner[%d].unit = %s" % (j, `positioner.unit`)
                    length = u.unpack_int() # length of readback_name string
                    if length: positioner.readback_name = u.unpack_string()
                    if verbose: print "positioner[%d].readback_name = %s" % (j, `positioner.readback_name`)
                    length = u.unpack_int() # length of readback_desc string
                    if length: positioner.readback_desc = u.unpack_string()
                    if verbose: print "positioner[%d].readback_desc = %s" % (j, `positioner.readback_desc`)
                    length = u.unpack_int() # length of readback_unit string
                    if length: positioner.readback_unit = u.unpack_string()
                    if verbose: print "positioner[%d].readback_unit = %s" % (j, `positioner.readback_unit`)
                
                  
                for j in range(scan_no_detectors):
                    detector = scanDetector()
                    detector.number = u.unpack_int()
                    detector.fieldName = detName(detector.number)
                    if verbose: print "detector ", j
                    length = u.unpack_int() # length of name string
                    if length: detector.name = u.unpack_string()
                    if verbose: print "detector[%d].name = %s" % (j, `detector.name`)
                    length = u.unpack_int() # length of desc string
                    if length: detector.desc = u.unpack_string()
                    if verbose: print "detector[%d].desc = %s" % (j, `detector.desc`)
                    length = u.unpack_int() # length of unit string
                    if length: detector.unit = u.unpack_string()
                    if verbose: print "detector[%d].unit = %s" % (j, `detector.unit`)
                

                    
                for j in range(scan_no_triggers):
                    trigger = scanTrigger()
                    trigger.number = u.unpack_int()
                    if verbose: print "trigger ", j
                    length = u.unpack_int() # length of name string
                    if length: trigger.name = u.unpack_string()
                    if verbose: print "trigger[%d].name = %s" % (j, `trigger.name`)
                    trigger.command = u.unpack_float()
                    if verbose: print "trigger[%d].command = %s" % (j, `trigger.command`)
                
                
                ### read data
                # positioners
                file.seek(file.tell() - (len(buf) - u.get_position()))
                buf = file.read(scan_no_positioners * scan_npts * 8)
                u = Unpacker(buf)
                for j in range(scan_no_positioners):
                    if verbose: print "read %d pts for pos. %d at buf loc %x" % (scan_npts,
                                                                                 j, u.get_position())
                    readback_array = u.unpack_farray(scan_cpt, u.unpack_double)    
                    if verbose: print "readback_array = ",  readback_array
            
                if rank == 2 : continue      
           
                if (i_outer_loop == 0) and (i_innermost_loop == 0) :
                    if save_ram:
                        no_energy_channels = save_ram
                    else:
                        no_energy_channels = scan_npts
                    if scan_no_detectors > 1:
                        mca_arr = np.zeros((x_pixels, y_pixels, no_energy_channels, scan_no_detectors), dtype=np.float32)
                    else:
                        mca_arr = np.zeros((x_pixels, y_pixels, no_energy_channels), dtype=np.float32)
                        
#                #This is very slow to unpack so read directly 
#                # detectors
#                pos = file.tell() - (len(buf) - u.get_position())
#                file.seek(file.tell() - (len(buf) - u.get_position()))
#                buf = file.read(scan_no_detectors * scan_npts * 4)
#                u = Unpacker(buf)
#                detector_array = np.zeros((scan_npts), dtype=np.float32)
#                for j in range(scan_no_detectors):
#                    detector_array = u.unpack_farray(scan_npts, u.unpack_float)
#                    if scan_no_detectors > 1: mca_arr[i_innermost_loop, i_outer_loop, :, j] = detector_array[:]          
#                    else: mca_arr[i_innermost_loop, i_outer_loop, :] = detector_array[:] 
       

                import struct
                # detectors
                file.seek(file.tell() - (len(buf) - u.get_position()))
                detector_array = np.zeros((scan_npts), dtype=np.float32)
                for j in range(scan_no_detectors):
                    buf = file.read(scan_npts * 4)
                    detector_array = struct.unpack('>'+str(scan_npts)+'f', buf)
                    if scan_no_detectors > 1: mca_arr[i_innermost_loop, i_outer_loop, :, j] = detector_array[:]          
                    else: mca_arr[i_innermost_loop, i_outer_loop, :] = detector_array[:] 


            two_d_time_stamp.append(temp_timestamp)
        
        
        if extra_pvs:
            extra_pv_key_list = []
            extra_pv_dict = {}
            file.seek(pointer_extra_PVs)
            buf = file.read()       # Read all scan-environment data
            u = Unpacker(buf)
            numExtra = u.unpack_int()
            for i in range(numExtra):
                name = ''
                n = u.unpack_int()      # length of name string
                if n: name = u.unpack_string()
                desc = ''
                n = u.unpack_int()      # length of desc string
                if n: desc = u.unpack_string()
                type = u.unpack_int()

                unit = ''
                value = ''
                count = 0
                if type != 0:   # not DBR_STRING
                    count = u.unpack_int()  # 
                    n = u.unpack_int()      # length of unit string
                    if n: unit = u.unpack_string()

                if type == 0: # DBR_STRING
                    n = u.unpack_int()      # length of value string
                    if n: value = u.unpack_string()
                elif type == 32: # DBR_CTRL_CHAR
                    #value = u.unpack_fstring(count)
                    v = u.unpack_farray(count, u.unpack_int)
                    value = ""
                    for i in range(len(v)):
                        # treat the byte array as a null-terminated string
                        if v[i] == 0: break
                        value = value + chr(v[i])

                elif type == 29: # DBR_CTRL_SHORT
                    value = u.unpack_farray(count, u.unpack_int)[0]
                elif type == 33: # DBR_CTRL_LONG
                    value = u.unpack_farray(count, u.unpack_int)[0]
                elif type == 30: # DBR_CTRL_FLOAT
                    value = u.unpack_farray(count, u.unpack_float)[0]
                elif type == 34: # DBR_CTRL_DOUBLE
                    value = u.unpack_farray(count, u.unpack_double)[0]
                    
                extra_pv_dict[name] = (desc, unit, value)
                
                extra_pv_key_list.append(name)
                
        
        scan_data = scan()

        
        scan_data.scan_name = one_d_info[0]
        scan_data.scan_time_stamp = one_d_info[1]
        
        scan_data.y_pixels = y_pixels
        scan_data.y_coord_arr = y_coord_arr
        

        # create mca calib description array
        scan_data.mca_calib_description_arr = mca_calib_description_arr 
        
        # create mca calibration array
        scan_data.mca_calib_arr = mca_calib_arr
        
        
        scan_data.x_pixels = x_pixels
        scan_data.x_coord_arr = x_coord_arr
        
        
        #detector_arr = fltarr(x_pixels, y_pixels, info.no_detectors)
        scan_data.detector_arr = detector_arr
        
        scan_data.detector_description_arr = detector_description_arr
           
        
        #mca_arr = fltarr(x_pixels, y_pixels, no_energy_channels, info.no_detectors)
        scan_data.mca_arr = mca_arr
        

                    
        if extra_pvs == True:
            scan_data.extra_pv = extra_pv_dict
            scan_data.extra_pv_key_list = extra_pv_key_list
                    
        return scan_data
    
    
    
#----------------------------------------------------------------------
    def read_combined_flyscan(self, path, mdafilename, this_detector):
        
        mdapath, mdafile = os.path.split(mdafilename)
        header, extension = os.path.splitext(mdafile)  

        
        try:
            scan = self.read_scan(mdafilename, extra_pvs = True)
            invalid_file = 0
        except:
            invalid_file = 1

        if scan == None or  invalid_file == 1:      
            print 'not read a valid mda flyscan file, filename: ', mdafilename
            #maps_change_xrf_resetvars, n_ev, n_rows, n_cols, n_energy, energy, energy_spec, scan_time_stamp, dataset_orig
            return
     
        detector_description_arr = scan.detector_description_arr
        detector_arr = scan.detector_arr
        
        
        x_pixels = detector_arr.shape[0]
        y_pixels = detector_arr.shape[1]

        wo = -1
        if '2xfm:mcs:mca1.VAL' in detector_description_arr:
            wo = detector_description_arr.index('2xfm:mcs:mca1.VAL')

        time = detector_arr[:, :, 0]/25000000.
        if wo != -1 : 
            time = time 
        else: 
            time[:, :] = 1.

        wo = -1
        if '2xfm:mcs:mca2.VAL' in detector_description_arr:
            wo =  detector_description_arr.index('2xfm:mcs:mca2.VAL')
        if wo != -1 : detector_description_arr[wo] = '2xfm:scaler3_cts1.B'
        if wo != -1 : detector_arr[:, :, wo] = detector_arr[:, :, wo]/time
        wo = -1
        if '2xfm:mcs:mca3.VAL' in detector_description_arr:
            wo =  detector_description_arr.index('2xfm:mcs:mca3.VAL')
        if wo != -1 : detector_description_arr[wo] = '2xfm:scaler3_cts1.C'
        if wo != -1 : detector_arr[:, :, wo] = detector_arr[:, :, wo]/time
        wo = -1
        if '2xfm:mcs:mca4.VAL' in detector_description_arr:
            wo =  detector_description_arr.index('2xfm:mcs:mca4.VAL')
        if wo != -1 : detector_description_arr[wo] = '2xfm:scaler3_cts1.D'
        wo = -1
        if '2xfm:mcs:mca5.VAL' in detector_description_arr:
            wo =  detector_description_arr.index('2xfm:mcs:mca5.VAL')
        if wo != -1 : detector_description_arr[wo] = '2xfm:scaler3_cts2.A'
        wo = -1
        if '2xfm:mcs:mca6.VAL' in detector_description_arr:
            wo =  detector_description_arr.index('2xfm:mcs:mca6.VAL')
        if wo != -1 : detector_description_arr[wo] = '2xfm:scaler3_cts2.B'
        wo = -1
        if '2xfm:mcs:mca7.VAL' in detector_description_arr:
            wo =  detector_description_arr.index('2xfm:mcs:mca7.VAL')
        if wo != -1 : detector_description_arr[wo] = '2xfm:scaler3_cts2.C'
        wo = -1
        if '2xfm:mcs:mca8.VAL' in detector_description_arr:
            wo =  detector_description_arr.index('2xfm:mcs:mca8.VAL')
        if wo != -1 : detector_description_arr[wo] = '2xfm:scaler3_cts2.D'
        wo = -1
        if '2xfm:mcs:mca9.VAL' in detector_description_arr:
            wo =  detector_description_arr.index('2xfm:mcs:mca9.VAL')
        if wo != -1 : detector_description_arr[wo] = '2xfm:scaler3_cts3.A'
        wo = -1
        if '2xfm:mcs:mca10.VAL' in detector_description_arr:
            wo =  detector_description_arr.index('2xfm:mcs:mca10.VAL')
        if wo != -1 : detector_description_arr[wo] = '2xfm:scaler3_cts3.B'
        wo = -1
        if '2xfm:mcs:mca11.VAL' in detector_description_arr:
            wo =  detector_description_arr.index('2xfm:mcs:mca11.VAL')
        if wo != -1 : detector_description_arr[wo] = '2xfm:scaler3_cts3.C'
        wo = -1
        if '2xfm:mcs:mca12.VAL' in detector_description_arr:
            wo =  detector_description_arr.index('2xfm:mcs:mca12.VAL')
        if wo != -1 : detector_description_arr[wo] = '2xfm:scaler3_cts3.D'
        wo = -1
        if '2xfm:mcs:mca13.VAL' in detector_description_arr:
            wo =  detector_description_arr.index('2xfm:mcs:mca13.VAL')
        if wo != -1 : detector_description_arr[wo] = '2xfm:scaler3_cts4.A'
        wo = -1
        if '2xfm:mcs:mca14.VAL' in detector_description_arr:
            wo =  detector_description_arr.index('2xfm:mcs:mca14.VAL')
        if wo != -1 : detector_description_arr[wo] = '2xfm:scaler3_cts4.B'
        wo = -1
        if '2xfm:mcs:mca15.VAL' in detector_description_arr:
            wo =  detector_description_arr.index('2xfm:mcs:mca15.VAL')
        if wo != -1 : detector_description_arr[wo] = '2xfm:scaler3_cts4.C'
        wo = -1
        if '2xfm:mcs:mca16.VAL' in detector_description_arr:
            wo =  detector_description_arr.index('2xfm:mcs:mca16.VAL')
        if wo != -1 : detector_description_arr[wo] = '2xfm:scaler3_cts4.D'
        wo = -1
        if '2xfm:mcs:mca17.VAL' in detector_description_arr:
            wo =  detector_description_arr.index('2xfm:mcs:mca17.VAL')
        if wo != -1 : detector_description_arr[wo] = '2xfm:scaler3_cts5.A'
        wo = -1
        if '2xfm:mcs:mca18.VAL' in detector_description_arr:
            wo =  detector_description_arr.index('2xfm:mcs:mca18.VAL')
        if wo != -1 : detector_description_arr[wo] = '2xfm:scaler3_cts5.B'
        wo = -1
        if '2xfm:mcs:mca19.VAL' in detector_description_arr:
            wo =  detector_description_arr.index('2xfm:mcs:mca19.VAL')
        if wo != -1 : detector_description_arr[wo] = '2xfm:scaler3_cts5.C'
        wo = -1
        if '2xfm:mcs:mca20.VAL' in detector_description_arr:
            wo =  detector_description_arr.index('2xfm:mcs:mca20.VAL')
        if wo != -1 : detector_description_arr[wo] = '2xfm:scaler3_cts5.D'
 

   
        n_ev = 0
        n_rows = y_pixels
        n_cols = x_pixels
    
        mca_arr = 0
        # now fill in XRF information    
      
        new_detector_description_arr = ['dxpXMAP2xfm3:mca1.ELTM', 'dxpXMAP2xfm3:mca2.ELTM', 'dxpXMAP2xfm3:mca3.ELTM', 'dxpXMAP2xfm3:mca4.ELTM']
        old_detector_description_arr = detector_description_arr
        old_detector_arr = detector_arr
        detector_description_arr = old_detector_description_arr+new_detector_description_arr
        detector_arr = np.ones((x_pixels, y_pixels, len(detector_description_arr)))
        detector_arr[:, :, 0:len(old_detector_description_arr)] = old_detector_arr[:, :, 0:len(old_detector_description_arr)]    
      
        h5filename = header+'_2xfm3__'+str(0)+'.h5'
        h5_file = os.path.join(path, os.path.join('flyXRF.h5', h5filename))



        f = call_function_with_retry(h5py.File, 5, 0.1, 1.1, (h5_file, 'r'))
        if f == None:
            print 'Error: Could not open file: ', h5_file

        gid = f['MAPS_RAW']
        if this_detector == 0 : entryname = 'data_a'
        if this_detector == 1 : entryname = 'data_b'
        if this_detector == 2 : entryname = 'data_c'
        if this_detector == 3 : entryname = 'data_d'
        
        dataset_id = gid[entryname]
        data = dataset_id[...]
        data = data.transpose()
        
        entryname = 'livetime'
        dataset_id = gid[entryname]
        livetime = dataset_id[...]
        livetime = livetime.transpose()

      
        entryname = 'realtime'
        dataset_id = gid[entryname]
        realtime = dataset_id[...]
        realtime = realtime.transpose()

      
        entryname = 'inputcounts'
        dataset_id = gid[entryname]
        inputcounts = dataset_id[...]

      
        entryname = 'ouputcounts'
        dataset_id = gid[entryname]
        outputcounts = dataset_id[...]
   
      
        f.close()
         
        hdf_data_size = data.shape
        this_x_pixels = np.amin([x_pixels, hdf_data_size[0]])
        this_y_pixels = np.amin([y_pixels, hdf_data_size[1]])     
      

        #create mca_arr as int_arr to save memory. conversion int to flt will still take the combined memopry allocation FOR both,  but is probably a bit better than before
        scan.mca_arr = np.zeros((x_pixels, y_pixels, 2000), dtype = np.int)## nxmx2000 array  ( 2000 energies)         


        for j_temp in range(20):
            scan.mca_arr[0:this_x_pixels, 0:this_y_pixels, j_temp*100:(99+j_temp*100+1)] = data[0:this_x_pixels, 0:this_y_pixels, j_temp*100:(99+j_temp*100+1)]
        data = 0
        scan.mca_arr.astype(float)
      
        detector_arr[0:this_x_pixels, 0:this_y_pixels, len(detector_description_arr)-4] = livetime[0:this_x_pixels, 0:this_y_pixels, 0]
        detector_arr[0:this_x_pixels, 0:this_y_pixels, len(detector_description_arr)-3] = livetime[0:this_x_pixels, 0:this_y_pixels, 1]
        detector_arr[0:this_x_pixels, 0:this_y_pixels, len(detector_description_arr)-2] = livetime[0:this_x_pixels, 0:this_y_pixels, 2]
        detector_arr[0:this_x_pixels, 0:this_y_pixels, len(detector_description_arr)-1] = livetime[0:this_x_pixels, 0:this_y_pixels, 3]
      
        return scan
