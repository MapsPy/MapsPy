'''
Created on Jan 11, 2012

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
import numpy as np
import os

import h5py
import maps_hdf5

""" ------------------------------------------------------------------------------------------------"""
def main(file1, file2):
    

    verbose = 1
    
    #remove quotations marks 
    file1.strip('"') 
    if "'" in file1: file1 = file1[1:-1]

    file1 = os.path.normpath(file1)

    if verbose: print 'file1 =', file1
        
    if not os.path.isfile(file1):
        print 'Error - File', file1, ' does not exist. Please specify working directory.'
        return
    
    #remove quotations marks 
    file2.strip('"') 
    if "'" in file2: file2 = file2[1:-1]

    file2 = os.path.normpath(file2)

    if verbose: print 'file2 =', file2
        
    if not os.path.isfile(file2):
        print 'Error - File', file2, ' does not exist. Please specify working directory.'
        return  
    
    
    
    f1 = h5py.File(file1, 'r')                
    if 'MAPS' not in f1:
            print 'error, hdf5 file does not contain the required MAPS group. I am aborting this action'
            return 

    maps_group_id1 = f1['MAPS']
        
    l1 = list(maps_group_id1)
        

    f2 = h5py.File(file2, 'r')                
    if 'MAPS' not in f2:
            print 'error, hdf5 file does not contain the required MAPS group. I am aborting this action'
            return 

    maps_group_id2 = f2['MAPS']
        
    l2 = list(maps_group_id2)
    
    s1 = set(l1)
    s2 = set(l2)
    
    if len(s1.difference(s2)):
        print '\nElements in ',os.path.basename(file1),' that are not in ', os.path.basename(file2), ':'
        for i in s1.difference(s2): print i
    elif len(s2.difference(s1)):
        print '\nElements in ',os.path.basename(file2),' that are not in ', os.path.basename(file1), ':'
        print s2.difference(s1)
    else:
        print 'Files have the same groups.'
        
        
    print '\nCompare HDF5 fields in the files.'
    h51 = maps_hdf5.h5()
    h52 = maps_hdf5.h5()    
    
    entryname = 'mca_arr'
    this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
    this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
    if not np.allclose(this_xrfdata1, this_xrfdata2, atol=1.0e-6): 
        print entryname, ' differs.'
    else: print entryname, ' is the same.'
         
    entryname = 'us_amp'
    this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
    this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
    if not np.allclose(this_xrfdata1, this_xrfdata2, atol=np.finfo(float).eps): 
        print entryname, ' differs. RMS=', np.sqrt(np.mean(( this_xrfdata1 -  this_xrfdata2)**2)) 
    else: print entryname, ' is the same.' 
    
    entryname = 'ds_amp'
    this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
    this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
    if not np.allclose(this_xrfdata1, this_xrfdata2, atol=np.finfo(float).eps): 
        print entryname, ' differs. RMS=', np.sqrt(np.mean(( this_xrfdata1 -  this_xrfdata2)**2)) 
    else: print entryname, ' is the same.'        
        
    entryname = 'energy'
    this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
    this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
    if not np.allclose(this_xrfdata1, this_xrfdata2, atol=np.finfo(float).eps): 
        print entryname, ' differs.'
        print this_xrfdata1
        print this_xrfdata2
    else: print entryname, ' is the same.'        
    
    entryname = 'energy_calib'
    this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
    this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
    if not np.allclose(this_xrfdata1, this_xrfdata2, atol=np.finfo(float).eps): 
        print entryname, ' differs. RMS=', np.sqrt(np.mean(( this_xrfdata1 -  this_xrfdata2)**2)) 
    else: print entryname, ' is the same.'      
        
    entryname = 'int_spec'
    this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
    this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
    if not np.allclose(this_xrfdata1, this_xrfdata2, atol=np.finfo(float).eps): 
        rms = np.sqrt(np.mean((this_xrfdata1-this_xrfdata2)**2))
        print entryname, ' differs. RMS =', rms
    else: print entryname, ' is the same.'  
    
    
    entryname = 'max_chan_spec'
    this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
    this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
    if not np.allclose(this_xrfdata1, this_xrfdata2, atol=np.finfo(float).eps): 
        print entryname, ' differs. RMS='
        for i in range(this_xrfdata1.shape[0]):
            print '\t', i,  np.sqrt(np.mean(( this_xrfdata1[i,:] -  this_xrfdata2[i,:])**2))
    else: print entryname, ' is the same.' 
    
    if verbose == 2:
        import matplotlib.pyplot as plt 
        for i in range(5):
            plt.plot(this_xrfdata1[i,:])
            plt.plot(this_xrfdata2[i,:])
            plt.show()
    
    
    entryname = 'scalers'
    this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
    this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
    #IDL scalers  have Nan on row 16 (17th scaler) - zero out
    this_xrfdata1 = np.nan_to_num(this_xrfdata1)
    this_xrfdata2 = np.nan_to_num(this_xrfdata2)    
    if not np.allclose(this_xrfdata1, this_xrfdata2, atol=1.e-6): 
        print entryname, ' differ.'
        for i in range(this_xrfdata1.shape[0]):
                if np.sum(np.abs(this_xrfdata1[i,:,:])) > 0:
                    print '\t', i,  'RMS= ', np.sqrt(np.mean(( this_xrfdata1[i,:,:] -  this_xrfdata2[i,:,:])**2)), ',', 100*np.sum(np.abs(this_xrfdata1[i,:,:] -  this_xrfdata2[i,:,:]))/np.sum(np.abs(this_xrfdata1[i,:,:])), '%'
                else:
                    print '\t', i,  'RMS= ', np.sqrt(np.mean(( this_xrfdata1[i,:,:] -  this_xrfdata2[i,:,:])**2))
    else: print entryname, 'are the same.' 
    

                
    entryname = 'XRF_roi'
    this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
    this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
    if this_xrfdata1.shape == this_xrfdata2.shape:
        if not np.allclose(this_xrfdata1[:,:,:], this_xrfdata2[:,:,:], atol=np.finfo(float).eps): 
            print 'XRF_roi differs.'
            for i in range(this_xrfdata1.shape[0]):
                if np.sum(np.abs(this_xrfdata1[i,:,:])) > 0:
                    print '\t', i,  'RMS= ', np.sqrt(np.mean(( this_xrfdata1[i,:,:] -  this_xrfdata2[i,:,:])**2)), ',', 100*np.sum(np.abs(this_xrfdata1[i,:,:] -  this_xrfdata2[i,:,:]))/np.sum(np.abs(this_xrfdata1[i,:,:])), '%'
                else:
                    print '\t', i,  'RMS= ', np.sqrt(np.mean(( this_xrfdata1[i,:,:] -  this_xrfdata2[i,:,:])**2))
        else: print entryname, ' is the same.' 
    else: 
        print entryname, ' have different shapes.'
    
    entryname = 'XRF_fits'
    channames, valid_read = h51.read_hdf5_core(maps_group_id1, 'channel_names')
    if (entryname in maps_group_id1) and (entryname in maps_group_id2):
        this_xrfdata1, valid_read1 = h51.read_hdf5_core(maps_group_id1, entryname)
        this_xrfdata2, valid_read2 = h52.read_hdf5_core(maps_group_id2, entryname)
        if this_xrfdata1.shape == this_xrfdata2.shape:
            if not np.allclose(this_xrfdata1, this_xrfdata2, atol=np.finfo(float).eps): 
                print 'XRF_fits differs. RMS='
                for i in range(this_xrfdata1.shape[0]):
                    ind = np.where(this_xrfdata2[i,:,:] > 0)
                    
                    data = this_xrfdata1[i,:,:]
                    #print ind
                    #print '\t', i,  np.sqrt(np.mean(( this_xrfdata1[i,:,:] -  this_xrfdata2[i,:,:])**2))
                    print channames[i], 'fits max:', np.amax(data[ind]), np.amax(this_xrfdata2[i,:,:]), '\t diff=', np.abs(np.amax(data[ind])-np.amax(this_xrfdata2[i,:,:])), ',\t', np.abs(np.amax(data[ind])-np.amax(this_xrfdata2[i,:,:]))/np.abs(np.amax(data[ind])), '%'
                    #print channames[i], 'fits max:', this_xrfdata1[i,20, 31], this_xrfdata2[i,20,31], '\t diff=', np.abs(this_xrfdata1[i,20,31]-this_xrfdata2[i,20,31])
            else: print entryname, ' is the same.' 
        else: print entryname, ' have different shapes.'

    
    entryname = 'XRF_roi_plus'
    this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
    this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
    if this_xrfdata1.shape == this_xrfdata2.shape:
        if not np.allclose(this_xrfdata1, this_xrfdata2, atol=np.finfo(float).eps): 
            print 'XRF_roi_plus differs.'
            for i in range(this_xrfdata1.shape[0]):
                if np.sum(np.abs(this_xrfdata1[i,:,:])) > 0:
                    print '\t', i,  'RMS= ', np.sqrt(np.mean(( this_xrfdata1[i,:,:] -  this_xrfdata2[i,:,:])**2)), ',', 100*np.sum(np.abs(this_xrfdata1[i,:,:] -  this_xrfdata2[i,:,:]))/np.sum(np.abs(this_xrfdata1[i,:,:])), '%'
                else:
                    print '\t', i,  'RMS= ', np.sqrt(np.mean(( this_xrfdata1[i,:,:] -  this_xrfdata2[i,:,:])**2))
        else: print entryname, ' is the same.' 
    else: print entryname, ' have different shapes.'
    
    
    entryname = 'x_axis'
    this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
    this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
    if not np.allclose(this_xrfdata1, this_xrfdata2, atol=np.finfo(float).eps): 
        print 'x_axis differs. RMS=', np.sqrt(np.mean(( this_xrfdata1 -  this_xrfdata2)**2)) 
    else: print entryname, ' is same.' 

    entryname = 'y_axis'
    this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
    this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
    if not np.allclose(this_xrfdata1, this_xrfdata2, atol=np.finfo(float).eps): 
        print 'y_axis differs. RMS=', np.sqrt(np.mean(( this_xrfdata1 -  this_xrfdata2)**2)) 
    else: print entryname, ' is the same.' 
    
            
    entryname = 'XRF_roi_quant'
    this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
    this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
    if this_xrfdata1.shape == this_xrfdata2.shape:
        if not np.allclose(this_xrfdata1, this_xrfdata2, atol=np.finfo(float).eps): 
            print 'XRF_roi_quant differs. '
            for i in range(this_xrfdata1.shape[0]):
                if np.sum(np.abs(this_xrfdata1[i,:,:])) > 0:
                    print '\t', i,  'RMS= ', np.sqrt(np.mean(( this_xrfdata1[i,:,:] -  this_xrfdata2[i,:,:])**2)), ',', 100*np.sum(np.abs(this_xrfdata1[i,:,:] -  this_xrfdata2[i,:,:]))/np.sum(np.abs(this_xrfdata1[i,:,:])), '%'
                else:
                    print '\t', i,  'RMS= ', np.sqrt(np.mean(( this_xrfdata1[i,:,:] -  this_xrfdata2[i,:,:])**2))
        else: print entryname, ' is the same.' 
    else: print entryname, ' have different shapes.'
    
    entryname = 'XRF_roi_plus_quant'
    this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
    this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
    if this_xrfdata1.shape == this_xrfdata2.shape:
        if not np.allclose(this_xrfdata1, this_xrfdata2, atol=np.finfo(float).eps): 
            print 'XRF_roi_plus_quant differs. '
            for i in range(this_xrfdata1.shape[0]):
                if np.sum(np.abs(this_xrfdata1[i,:,:])) > 0:
                    print '\t', i,  'RMS= ', np.sqrt(np.mean(( this_xrfdata1[i,:,:] -  this_xrfdata2[i,:,:])**2)), ',', 100*np.sum(np.abs(this_xrfdata1[i,:,:] -  this_xrfdata2[i,:,:]))/np.sum(np.abs(this_xrfdata1[i,:,:])), '%'
                else:
                    print '\t', i,  'RMS= ', np.sqrt(np.mean(( this_xrfdata1[i,:,:] -  this_xrfdata2[i,:,:])**2))
        else: print entryname, ' is the same.' 
    else: print entryname, ' have different shapes.'
    
    entryname = 'XRF_fits_quant'
    if (entryname in maps_group_id1) and (entryname in maps_group_id2):
        this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
        this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
        if this_xrfdata1.shape == this_xrfdata2.shape:
            if not np.allclose(this_xrfdata1, this_xrfdata2, atol=np.finfo(float).eps): 
                print 'XRF_fits_quant differs. RMS='
                for i in range(this_xrfdata1.shape[0]):
                    if np.sum(np.abs(this_xrfdata1[i,:,:])) > 0:
                        print '\t', i,  'RMS= ', np.sqrt(np.mean(( this_xrfdata1[i,:,:] -  this_xrfdata2[i,:,:])**2)), ',', 100*np.sum(np.abs(this_xrfdata1[i,:,:] -  this_xrfdata2[i,:,:]))/np.sum(np.abs(this_xrfdata1[i,:,:])), '%'
                    else:
                        print '\t', i,  'RMS= ', np.sqrt(np.mean(( this_xrfdata1[i,:,:] -  this_xrfdata2[i,:,:])**2))
            else: print entryname, ' is the same.'   
        else: print entryname, ' have different shapes.'  
    
    
    entryname = 'channel_names'
    this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
    this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
    if len(this_xrfdata1) == len(this_xrfdata2): 
        same = 1
        for i in range(len(this_xrfdata1)):
            if this_xrfdata1[i] != this_xrfdata2[i]: same = 0
        if same == 0:
            print entryname,' differ.'
            for i in range(len(this_xrfdata1)):
                print '\t', i,  this_xrfdata1[i], this_xrfdata2[i]
        else: print entryname, 'are the same.'  
    else: 
        print entryname,' differ:'  
        print this_xrfdata1 
        print this_xrfdata2
        print '\n'
    
    entryname = 'channel_units'
    this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
    this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
    if len(this_xrfdata1) == len(this_xrfdata2): 
        print entryname, 'are the same.'              
    else: 
        print entryname,' differ:'
        print this_xrfdata1
        print this_xrfdata2
        print '\n'
    
    entryname = 'scaler_names'
    this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
    this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
    if len(this_xrfdata1) == len(this_xrfdata2): 
        same = 1
        for i in range(len(this_xrfdata1)):
            if this_xrfdata1[i] != this_xrfdata2[i]: same = 0
        if same == 0:
            print entryname,' differ.'
            for i in range(len(this_xrfdata1)):
                print '\t', i,  this_xrfdata1[i], this_xrfdata2[i]
        else: print entryname, 'are the same.'  
    else: print entryname,' differ.'   
    
    entryname = 'scaler_units'
    this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
    this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
    if len(this_xrfdata1) == len(this_xrfdata2): 
        same = 1
        for i in range(len(this_xrfdata1)):
            if this_xrfdata1[i] != this_xrfdata2[i]: same = 0
        if same == 0:
            print entryname,' differs.'
            for i in range(len(this_xrfdata1)):
                print '\t', i,  this_xrfdata1[i], this_xrfdata2[i]
        else: print entryname, 'are the same.'  
    else: print entryname,' differ.'   
    
    
    entryname = 'add_long'
    this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
    this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
    if not np.allclose(this_xrfdata1, this_xrfdata2, atol=np.finfo(float).eps): 
        print entryname, 'differs. RMS=', np.sqrt(np.mean(( this_xrfdata1 -  this_xrfdata2)**2)) 
    else: print entryname, ' is same.' 
    
    entryname = 'add_float'
    this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
    this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
    if not np.allclose(this_xrfdata1, this_xrfdata2, atol=np.finfo(float).eps): 
        print entryname, 'differs. RMS=', np.sqrt(np.mean(( this_xrfdata1 -  this_xrfdata2)**2)) 
    else: print entryname, ' is same.' 
    
    entryname = 'add_string'
    this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
    this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
    if len(this_xrfdata1) == len(this_xrfdata2): 
        same = 1
        for i in range(len(this_xrfdata1)):
            if this_xrfdata1[i].strip() != this_xrfdata2[i].strip(): 
                same = 0
        if same == 0:
            print entryname,' differs.'                
        else: print entryname, 'are the same.'  
    else: print entryname,' differ.'     
    
#     #Extra_strings are the same if extra_pvs are the same
#     entryname = 'extra_strings'
#     this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
#     this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
#     if len(this_xrfdata1) == len(this_xrfdata2): 
#         same = 1
#         for i in range(len(this_xrfdata1)):
#             if this_xrfdata1[i] != this_xrfdata2[i]: 
#                 same = 0
#                 print this_xrfdata1[i],this_xrfdata2[i]
#         if same == 0:
#             print entryname,' differ.'
#         else: print entryname, 'are the same.'  
#     else: print entryname,' differ.'       
    
    
    entryname = 'extra_pvs'
    this_xrfdata1, valid_read = h51.read_hdf5_core(maps_group_id1, entryname)
    this_xrfdata2, valid_read = h52.read_hdf5_core(maps_group_id2, entryname)
       
    l1 = []
    l2 = []
    for i in range(this_xrfdata1.shape[1]): l1.append(this_xrfdata1[0,i])
    for i in range(this_xrfdata2.shape[1]): l2.append(this_xrfdata2[0,i])        
    s1 = set(l1)
    s2 = set(l2)
    if len(s1.difference(s2)):
        print '\nElements in 1 that are not in 2:'
        for i in s1.difference(s2): print i
    elif len(s2.difference(s1)):
        print '\nElements in 2 that are not in 1:'
        print s2.difference(s1)
    else:
        print 'Files have the same extra_pvs.' 
        
       
    f1.close()
    f2.close()
    
        
    print '\nCompare XRFmaps.'

    XRFmaps1, valid_read1 = h51.read_hdf5(file1)
    
    if valid_read1 == 0: 
        print 'Could not read ', file1
        return
    

    XRFmaps2, valid_read2 = h52.read_hdf5(file2)
    
    if valid_read2 == 0: 
        print 'Could not read ', file2
        return
    
    print '\nComparing XRFmaps_info'
    
    found_diff = 0
    
    if XRFmaps1.n_ev != XRFmaps2.n_ev:
        print 'n_ev differs.'
        found_diff = 1
        
    if XRFmaps1.version != XRFmaps2.version:
        print 'version differs.'
        found_diff = 1
        
    if XRFmaps1.n_cols != XRFmaps2.n_cols:
        print 'n_cols differs.'
        found_diff = 1
        
    if XRFmaps1.n_rows != XRFmaps2.n_rows:
        print 'n_rows differs.'
        found_diff = 1
        
    n_used_dmaps_same = 1
    if XRFmaps1.n_used_dmaps != XRFmaps2.n_used_dmaps:
        print 'n_used_dmaps differs. XRFmaps1.n_used_dmaps ', XRFmaps1.n_used_dmaps, ', XRFmaps2.n_used_dmaps ', XRFmaps2.n_used_dmaps
        found_diff = 1
        n_used_dmaps_same = 0
        
    s1 = set(XRFmaps1.dmaps_names)
    s2 = set(XRFmaps2.dmaps_names)
    if len(s1.difference(s2)):
        print '\nElements in XRFmaps1.dmaps_names that are not in XRFmaps2.dmaps_names:'
        for i in s1.difference(s2): print i
    elif len(s2.difference(s1)):
        print '\nElements in XRFmaps2.dmaps_names that are not in XRFmaps1.dmaps_names:'
        print s2.difference(s1)
    else:
        print 'Files have same XRFmaps.dmaps_names' 
        
        
    if XRFmaps1.n_used_chan != XRFmaps2.n_used_chan:
        print 'n_used_chan differs.'
        found_diff = 1
        
        
    s1 = set(XRFmaps1.chan_names)
    s2 = set(XRFmaps2.chan_names)
    if len(s1.difference(s2)):
        print '\nElements in XRFmaps1.chan_names that are not in XRFmaps2.chan_names:'
        for i in s1.difference(s2): print i
    elif len(s2.difference(s1)):
        print '\nElements in XRFmaps2.chan_names that are not in XRFmaps1.chan_names:'
        print s2.difference(s1)
    else:
        print '\nFiles have same XRFmaps.chan_names' 

    if not np.allclose(XRFmaps1.x_coord_arr, XRFmaps2.x_coord_arr, atol=np.finfo(float).eps): 
        print 'x_coord_arr differs.'
        found_diff = 1
        
    if not np.allclose(XRFmaps1.y_coord_arr, XRFmaps2.y_coord_arr, atol=np.finfo(float).eps): 
        print 'y_coord_arr differs'
        found_diff = 1

    if n_used_dmaps_same == 1:
        if not np.allclose(XRFmaps1.dmaps_set, XRFmaps1.dmaps_set, atol=np.finfo(float).eps):  
            print 'dmaps_set differs.'
            found_diff = 1
               
        if not np.allclose(XRFmaps1.dataset, XRFmaps1.dataset, atol=np.finfo(float).eps):  
            print 'dataset differs.'
            found_diff = 1


    if not np.allclose(XRFmaps1.dataset_orig[:, :, :, 0], XRFmaps1.dataset_orig[:, :, :, 0], atol=np.finfo(float).eps):  
        print 'dataset_orig[:, :, :, 0] differs.'
        found_diff = 1        
              
    if not np.allclose(XRFmaps1.dataset_orig[:, :, :, 1], XRFmaps1.dataset_orig[:, :, :, 1], atol=np.finfo(float).eps):  
        print 'dataset_orig[:, :, :, 1] differs.'
        found_diff = 1    
        
    if not np.allclose(XRFmaps1.dataset_orig[:, :, :, 2], XRFmaps1.dataset_orig[:, :, :, 2], atol=np.finfo(float).eps):  
        print 'dataset_orig[:, :, :, 2] differs.'
        found_diff = 1    
        
        
    s1 = set(XRFmaps1.dataset_names)
    s2 = set(XRFmaps2.dataset_names)
    if len(s1.difference(s2)):
        print '\nElements in XRFmaps1.dataset_names that are not in XRFmaps2.dataset_names:'
        for i in s1.difference(s2): print i
    elif len(s2.difference(s1)):
        print '\nElements in XRFmaps2.dataset_names that are not in XRFmaps1.dataset_names:'
        print s2.difference(s1)
    else:
        print 'Files have same XRFmaps.dataset_names' 
        

    try:
        if not np.allclose(XRFmaps1.dataset_calibration, XRFmaps2.dataset_calibration, atol=np.finfo(float).eps): 
            print 'dataset_calibration differs. '
            found_diff = 1 
    except: print 'dataset_calibration differs.'
        
    
    if XRFmaps1.n_energy != XRFmaps2.n_energy:
        print 'n_energy differs.'
        found_diff = 1


    if not np.allclose(XRFmaps1.energy,XRFmaps2.energy, atol=np.finfo(float).eps):
        print 'energy differs.'
        found_diff = 1  
               

    if not np.allclose(XRFmaps1.energy_spec[0:XRFmaps2.energy_spec.shape[0]], XRFmaps2.energy_spec, atol=np.finfo(float).eps): 
        print 'energy_spec differs.'
        found_diff = 1        
        

    if not np.allclose(XRFmaps1.max_chan_spec[2,0:XRFmaps2.max_chan_spec.shape[1]], XRFmaps2.max_chan_spec[2,:], atol=np.finfo(float).eps): 
        print 'max_chan_spec differs.'
        found_diff = 1   
        
        
    if not np.allclose(XRFmaps1.raw_spec, XRFmaps2.raw_spec, atol=np.finfo(float).eps): 
        print 'raw_spec differs.'
        found_diff = 1  
        

    if XRFmaps1.n_raw_det != XRFmaps2.n_raw_det:
        print 'n_raw_det differs.'
        found_diff = 1
        
    if XRFmaps1.img_type != XRFmaps2.img_type:
        print 'img_type differs.'
        found_diff = 1

    if not np.allclose(XRFmaps1.us_amp, XRFmaps2.us_amp, atol=np.finfo(float).eps): 
        print 'us_amp differs.'
        found_diff = 1  
        
    if not np.allclose(XRFmaps1.ds_amp, XRFmaps2.ds_amp, atol=np.finfo(float).eps): 
        print 'ds_amp differs.'
        found_diff = 1  

    if not np.allclose(XRFmaps1.energy_fit, XRFmaps2.energy_fit, atol=np.finfo(float).eps): 
        print 'energy_fit differs.'
        found_diff = 1  


    s1 = set(XRFmaps1.extra_str_arr)
    s2 = set(XRFmaps2.extra_str_arr)
    if len(s1.difference(s2)):
        print '\nElements in XRFmaps1.extra_str_arr that are not in XRFmaps2.extra_str_arr:'
        for i in s1.difference(s2): print i
    elif len(s2.difference(s1)):
        print '\nElements in XRFmaps2.extra_str_arr that are not in XRFmaps1.extra_str_arr:'
        print s2.difference(s1)
    else:
        print 'Files have same XRFmaps.extra_str_arr' 
        
        
    print 'Compare maps_config:'
    found_diff_mmc = 0
    mmc1 = XRFmaps1.make_maps_conf
    mmc2 = XRFmaps2.make_maps_conf
    
    if mmc1.use_default_dirs != mmc1.use_default_dirs :
        print 'use_default_dirs differs.'
        found_diff_mmc = 1

    if mmc1.use_beamline != mmc1.use_beamline :
        print 'use_beamline differs.'
        found_diff_mmc = 1
    if mmc1.version != mmc1.version :
        print 'version differs.'
        found_diff_mmc = 1
        
    if mmc1.fit_t_be != mmc1.fit_t_be :
        print 'fit_t_be differs.'
        found_diff_mmc = 1

    if mmc1.fit_t_GE != mmc1.fit_t_GE :
        print 'fit_t_GE differs.'
        found_diff_mmc = 1                

    if mmc1.n_chan != mmc1.n_chan :
        print 'n_chan differs.'
        found_diff_mmc = 1  

    if mmc1.n_used_chan != mmc1.n_used_chan :
        print 'n_used_chan differs.'
        found_diff_mmc = 1  
        
    if mmc1.active_chan != mmc1.active_chan :
        print 'active_chan differs.'
        found_diff_mmc = 1  

    if mmc1.n_dmaps != mmc1.n_dmaps :
        print 'n_dmaps differs.'
        found_diff_mmc = 1  
    
    if mmc1.n_used_dmaps != mmc1.n_used_dmaps :
        print 'n_used_dmaps differs.'
        found_diff_mmc = 1  
        
    if mmc1.incident_E != mmc1.incident_E :
        print 'incident_E differs.'
        found_diff_mmc = 1  
        
    if mmc1.use_fit != mmc1.use_fit :
        print 'use_fit differs.'
        found_diff_mmc = 1  

    if mmc1.use_pca != mmc1.use_pca :
        print 'use_pca differs.'
        found_diff_mmc = 1                  

    if not np.allclose(mmc1.use_det, mmc2.use_det, atol=np.finfo(float).eps): 
        print 'mmc1.use_det differs.'
        found_diff_mmc = 1  
        
    if not np.allclose(mmc1.e_cal, mmc2.e_cal, atol=np.finfo(float).eps): 
        print 'mmc1.e_cal differs.'
        found_diff_mmc = 1  
        
        
    #Standards not yet implemented: 
    #        mmc1.nbs32 = standard()
    #        mmc1.nbs33 = standard()
    #        mmc1.calibration = calibration()    
   
 
    if found_diff_mmc == 1: 
        print 'XRFmaps.make_maps_conf differ.' 
        found_diff = 1  
    else:
        print 'XRFmaps.make_maps_conf are the same.'          

    
    if found_diff == 1: print 'XRFmaps differ.'
    print 'Finished!'
    
                
                

#-----------------------------------------------------------------------------   
if __name__ == '__main__':
    

    import sys

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    main(file1, file2)