'''
Created on May 2014

@author: Mirna Lerotic, 2nd Look Consulting
         http://www.2ndlookconsulting.com/
'''

from __future__ import division
import string

import numpy as np
import h5py 
import xml.etree.ElementTree as ET


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
class nxs:
    def __init__(self):
        pass
    
#----------------------------------------------------------------------
    def read_scan(self, filename):
        
        
        # Open HDF5 file
        f = h5py.File(filename, 'r') 
    
        if 'entry1' in f:
            e1Grp = f['entry1'] 
            
                        
            if 'xml' in e1Grp:
                xmlGrp = e1Grp['xml']
                
                if 'ScanParameters' in xmlGrp:
                    scanpars = xmlGrp['ScanParameters']
                    scanpars = scanpars[...]
                    scanpars = scanpars[0]          

                    root = ET.fromstring(scanpars)
                    keys = []
                    values = []
                    for child_of_root in root:
                        keys.append(child_of_root.tag.strip())
                        values.append(child_of_root.text.strip())
                        
                    dScanPars = dict(zip(keys, values))
                    #print dScanPars
                    
                if 'OutputParameters' in xmlGrp:
                    outpars = xmlGrp['OutputParameters']
                    outpars = outpars[...]
                    outpars = outpars[0]          

                    root = ET.fromstring(outpars)
                    keys = []
                    values = []
                    for child_of_root in root:
                        keys.append(child_of_root.tag.strip())
                        values.append(child_of_root.text.strip())
                        
                    dOutPars = dict(zip(keys, values))
                    #print dOutPars
                                        
                    
                if 'DetectorParameters' in xmlGrp:
                    detpars = xmlGrp['DetectorParameters']
                    detpars = detpars[...]
                    detpars = detpars[0]          

                    root = ET.fromstring(detpars)
                    keys = []
                    values = []
                    ndetgrp = 1
                    for child_of_root in root:
                        if child_of_root.text.strip() != '':
                            keys.append(child_of_root.tag.strip())
                            values.append(child_of_root.text.strip())
                        else:
                            nionch = 1
                            apstr1 = ''
                            if child_of_root.tag.strip() == 'detectorGroup':
                                apstr1 = str(ndetgrp)
                                ndetgrp += 1
                            ndet = 1
                            for gchild_of_root in child_of_root:
                                apstr2 = ''
                                if gchild_of_root.tag.strip() == 'detector':
                                    apstr2 = str(ndet)
                                    ndet += 1
                                if gchild_of_root.text.strip() != '':
                                    keys.append(child_of_root.tag.strip()+apstr1+'/'+gchild_of_root.tag.strip()+apstr2)
                                    values.append(gchild_of_root.text.strip())
                                else:
                                    apstr3 = ''
                                    if gchild_of_root.tag.strip() == 'ionChamber':
                                        apstr3 = str(nionch)
                                        nionch += 1                                                                                
                                    for ggchild_of_root in gchild_of_root:
                                        if ggchild_of_root.text.strip() != '':
                                            keys.append(child_of_root.tag.strip()+'/'+gchild_of_root.tag.strip()+apstr3+'/'+ggchild_of_root.tag.strip())
                                            values.append(ggchild_of_root.text.strip())
                                                                                

                 
                    dDetPars = dict(zip(keys, values))
                    #print dDetPars
                    
                if 'SampleParameters' in xmlGrp:
                    samplepars = xmlGrp['SampleParameters']
                    samplepars = samplepars[...]
                    samplepars = samplepars[0]          

                    root = ET.fromstring(samplepars)
                    keys = []
                    values = []
                    for child_of_root in root:
                        if child_of_root.text.strip() != '':
                            keys.append(child_of_root.tag.strip())
                            values.append(child_of_root.text.strip())
                        else:
                            for gchild_of_root in child_of_root:
                                if gchild_of_root.text.strip() != '':
                                    keys.append(child_of_root.tag.strip()+'/'+gchild_of_root.tag.strip())
                                    values.append(gchild_of_root.text.strip())


                 
                    dSamlePars = dict(zip(keys, values))
                    #print dSamlePars
                    

            if 'xmapMca' in e1Grp:
                xmapmcaGrp = e1Grp['xmapMca']    
                                 
                        
                         
                if 'allElementSum' in xmapmcaGrp:
                    allElementSum = xmapmcaGrp['allElementSum']
                    allElementSum = allElementSum[...]                    
                
                if 'fullSpectrum' in xmapmcaGrp:
                    fullSpectrum = xmapmcaGrp['fullSpectrum']
                    fullSpectrum = fullSpectrum[...] 
                    
                if 'icr' in xmapmcaGrp:
                    icr = xmapmcaGrp['icr']
                    icr = icr[...] 
                    
                if 'ocr' in xmapmcaGrp:
                    ocr = xmapmcaGrp['ocr']
                    ocr = ocr[...] 
                                                            
                if 'realX' in xmapmcaGrp:
                    realX = xmapmcaGrp['realX']
                    realX = realX[...] 
                    
                if 'sc_MicroFocusSampleY' in xmapmcaGrp:
                    sc_MicroFocusSampleY  = xmapmcaGrp['sc_MicroFocusSampleY']
                    sc_MicroFocusSampleY = sc_MicroFocusSampleY[...] 
                                                        
                        
        # Close
        f.close() 
    

        scan_data = scan()

        
        scan_data.scan_name = 'DLS Scan'
        scan_data.scan_time_stamp = ' '
        

        # create mca calib description array
        scan_data.mca_calib_description_arr = [] 
        
        # create mca calibration array
        scan_data.mca_calib_arr = np.zeros((fullSpectrum.shape[2]))
        
        
        scan_data.x_pixels = fullSpectrum.shape[0]
        scan_data.x_coord_arr = realX
        
        scan_data.y_pixels = fullSpectrum.shape[1]
        scan_data.y_coord_arr = sc_MicroFocusSampleY 
        
        
        #detector_arr = fltarr(x_pixels, y_pixels, info.no_detectors)
        scan_data.detector_arr = np.zeros((fullSpectrum.shape[0],fullSpectrum.shape[1],fullSpectrum.shape[2]))
        
        scan_data.detector_description_arr = []
           
        
        #mca_arr = fltarr(x_pixels, y_pixels, no_energy_channels, info.no_detectors)
        scan_data.mca_arr = fullSpectrum
                   
        return scan_data
    
    
#----------------------------------------------------------------------
    def convert_nxs_to_h5(self, nfilename, hfilename, overwrite = True):
        
        
        # Open HDF5 file
        f = h5py.File(nfilename, 'r') 
    
        if 'entry1' in f:
            e1Grp = f['entry1'] 
            
                        
            if 'xml' in e1Grp:
                xmlGrp = e1Grp['xml']
                
                if 'ScanParameters' in xmlGrp:
                    scanpars = xmlGrp['ScanParameters']
                    scanpars = scanpars[...]
                    scanpars = scanpars[0]          

                    root = ET.fromstring(scanpars)
                    keys = []
                    values = []
                    for child_of_root in root:
                        keys.append(child_of_root.tag.strip())
                        values.append(child_of_root.text.strip())
                        
                    dScanPars = dict(zip(keys, values))
                    #print dScanPars
                    
                if 'OutputParameters' in xmlGrp:
                    outpars = xmlGrp['OutputParameters']
                    outpars = outpars[...]
                    outpars = outpars[0]          

                    root = ET.fromstring(outpars)
                    keys = []
                    values = []
                    for child_of_root in root:
                        keys.append(child_of_root.tag.strip())
                        values.append(child_of_root.text.strip())
                        
                    dOutPars = dict(zip(keys, values))
                    #print dOutPars
                                        
                    
                if 'DetectorParameters' in xmlGrp:
                    detpars = xmlGrp['DetectorParameters']
                    detpars = detpars[...]
                    detpars = detpars[0]          

                    root = ET.fromstring(detpars)
                    keys = []
                    values = []
                    ndetgrp = 1
                    for child_of_root in root:
                        if child_of_root.text.strip() != '':
                            keys.append(child_of_root.tag.strip())
                            values.append(child_of_root.text.strip())
                        else:
                            nionch = 1
                            apstr1 = ''
                            if child_of_root.tag.strip() == 'detectorGroup':
                                apstr1 = str(ndetgrp)
                                ndetgrp += 1
                            ndet = 1
                            for gchild_of_root in child_of_root:
                                apstr2 = ''
                                if gchild_of_root.tag.strip() == 'detector':
                                    apstr2 = str(ndet)
                                    ndet += 1
                                if gchild_of_root.text.strip() != '':
                                    keys.append(child_of_root.tag.strip()+apstr1+'/'+gchild_of_root.tag.strip()+apstr2)
                                    values.append(gchild_of_root.text.strip())
                                else:
                                    apstr3 = ''
                                    if gchild_of_root.tag.strip() == 'ionChamber':
                                        apstr3 = str(nionch)
                                        nionch += 1                                                                                
                                    for ggchild_of_root in gchild_of_root:
                                        if ggchild_of_root.text.strip() != '':
                                            keys.append(child_of_root.tag.strip()+'/'+gchild_of_root.tag.strip()+apstr3+'/'+ggchild_of_root.tag.strip())
                                            values.append(ggchild_of_root.text.strip())
                                                                                

                 
                    dDetPars = dict(zip(keys, values))
                    #print dDetPars
                    
                if 'SampleParameters' in xmlGrp:
                    samplepars = xmlGrp['SampleParameters']
                    samplepars = samplepars[...]
                    samplepars = samplepars[0]          

                    root = ET.fromstring(samplepars)
                    keys = []
                    values = []
                    for child_of_root in root:
                        if child_of_root.text.strip() != '':
                            keys.append(child_of_root.tag.strip())
                            values.append(child_of_root.text.strip())
                        else:
                            for gchild_of_root in child_of_root:
                                if gchild_of_root.text.strip() != '':
                                    keys.append(child_of_root.tag.strip()+'/'+gchild_of_root.tag.strip())
                                    values.append(gchild_of_root.text.strip())


                 
                    dSamlePars = dict(zip(keys, values))
                    #print dSamlePars
                    

            if 'xmapMca' in e1Grp:
                xmapmcaGrp = e1Grp['xmapMca']    
                                 
                        
                         
                if 'allElementSum' in xmapmcaGrp:
                    allElementSum = xmapmcaGrp['allElementSum']
                    allElementSum = allElementSum[...]                    
                
                if 'fullSpectrum' in xmapmcaGrp:
                    fullSpectrum = xmapmcaGrp['fullSpectrum']
                    fullSpectrum = fullSpectrum[...] 
                    
                if 'icr' in xmapmcaGrp:
                    icr = xmapmcaGrp['icr']
                    icr = icr[...] 
                    
                if 'ocr' in xmapmcaGrp:
                    ocr = xmapmcaGrp['ocr']
                    ocr = ocr[...] 
                                                            
                if 'realX' in xmapmcaGrp:
                    realX = xmapmcaGrp['realX']
                    realX = realX[...] 
                    
                if 'sc_MicroFocusSampleY' in xmapmcaGrp:
                    sc_MicroFocusSampleY  = xmapmcaGrp['sc_MicroFocusSampleY']
                    sc_MicroFocusSampleY = sc_MicroFocusSampleY[...] 
                                                        
                        
        # Close
        f.close()
        
        
        mca_arr = allElementSum

       
        # set compression level where applicable:
        gzip = 5
        file_status = 0
        entry_exists = 0
        
        verbose = 0
        
      
        # test whether a file with this filename already exists:
        try:
            # Open HDF5 file
            f = h5py.File(hfilename, 'r')
            if verbose: print 'Have HDF5 file: ', hfilename
            file_exists = 1
            file_is_hdf = 1
            file_status = 2       
            
            #MAPS HDF5 group
            if 'MAPS' in f:
                if verbose: print 'MAPS group found in file: ', hfilename
                mapsGrp = f['MAPS']
                file_status = 3
                if 'mca_arr' in mapsGrp:
                    if verbose: print 'MAPS\\mca_arr found in file: ', hfilename
                    file_status = 4
                # at the moment, simply overwrite the mca_arr section of
                # the file; in the future, may want to test, and only
                # overwrite if specific flag is set.

            f.close()

        except:
            if verbose: print 'Creating new file: ', hfilename
            
        if verbose: print 'file_status: ', file_status
        
        if overwrite : file_status = 0
        
        print hfilename
        if file_status <= 1 : 
            f = h5py.File(hfilename, 'w')
        else : 
            f = h5py.File(hfilename, 'a')

        if file_status <= 3 : 
            # create a group for maps to hold the data
            mapsGrp = f.create_group('MAPS')
            # now set a comment
            mapsGrp.attrs['comments'] = 'This is the group that stores all relevant information created (and read) by the the MAPS analysis software'

        if file_status >= 4 : 
            mapsGrp = f['MAPS']
            entry_exists = 1
            
        if entry_exists == 0:
            # create dataset and save full spectra
            data = np.transpose(mca_arr)
            dimensions = data.shape
            chunk_dimensions = (dimensions[0], 1, 1)
            comment = 'these are the full spectra at each pixel of the dataset'
            ds_data = mapsGrp.create_dataset('mca_arr', data = data, chunks=chunk_dimensions, compression='gzip', compression_opts=gzip)
            ds_data.attrs['comments'] = comment
        else:
            # save the data to existing array
            # delete old dataset, create new and save full spectra
            data = np.transpose(mca_arr)
            dimensions = data.shape
            chunk_dimensions = (dimensions[0], 1, 1)
            comment = 'these are the full spectra at each pixel of the dataset'
            del mapsGrp['mca_arr']
            ds_data = mapsGrp.create_dataset('mca_arr', data = data, chunks=chunk_dimensions, compression='gzip', compression_opts=gzip)
            ds_data.attrs['comments'] = comment
            
        
        
        
        entryname = 'x_axis'
        comment = 'stores the values of the primary fast axis positioner, typically sample x'
        data = realX
        ds_data = mapsGrp.create_dataset(entryname, data = data)
        ds_data.attrs['comments'] = comment

        entryname = 'y_axis'
        comment = 'stores the values of the slow axis positioner, typically sample y'
        data = sc_MicroFocusSampleY
        ds_data = mapsGrp.create_dataset(entryname, data = data)
        ds_data.attrs['comments'] = comment
        
        
        f.close()
        
        
        return                

    

