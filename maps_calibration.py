'''
Created on Apr 10, 2013

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
import time as t
import matplotlib as mplot
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from fitting import leastsqbound
import maps_definitions
import maps_analyze
import maps_fit_parameters
#import maps_generate_img_dat
import maps_tools
import henke




#-----------------------------------------------------------------------------
class calibration:
    def __init__(self, main, maps_conf):  
        self.maps_conf = maps_conf
        self.main = main
        


#-----------------------------------------------------------------------------   
    def read_axo_calibration(self, standard_filenames,
                             maps_conf, 
                             this_detector = 0, 
                             total_number_detectors = 1,
                             no_nbs = 0, 
                             fitmatrix_reduced = 0,
                             fitp = 0,
                             info_elements = 0):
        


        suffix = ''
        
        fit = maps_analyze.analyze()


        print 'called axo_calibration with these specifications: this detector: {0}; total_number_detectors: {1}.'.format(
                                                                this_detector, total_number_detectors)

        if total_number_detectors > 0: 
            if total_number_detectors > 1:
                suffix = str(this_detector) 
                print ' and suffix is: ', suffix
        

        maxiter = 500
        old_ratio = 0

        axo_e_cal = self.maps_conf.axo_e_cal.copy()
        axo_e_cal_factor = np.zeros((self.maps_conf.n_chan, 3))
          
        aux_arr = np.zeros((self.maps_conf.n_chan, 6))
        yield_correction = np.zeros((self.maps_conf.n_chan, 4))

        airpath = 0
        srcurrent_name = ''
        
        overide_files_found = 0 
        maps_overridefile = os.path.join(self.main['master_dir'],'maps_fit_parameters_override.txt')
        try:
            f = open(maps_overridefile, 'rt')    
            print maps_overridefile, ' exists.'
            f.close()
            overide_files_found = 1
        except :
            overide_files_found = 0 
         
        
        if overide_files_found == 1:
            f = open(maps_overridefile, 'rt')
            for line in f:
                if ':' in line : 
                    slist = line.split(':')
                    tag = slist[0]
                    value = ''.join(slist[1:])
                    
                    if   tag == 'AIRPATH'   : airpath = float(value)
                    elif tag == 'SRCURRENT' : srcurrent_name = str(value)
                    elif tag == 'US_IC'     : us_ic_name = str(value)
                    elif tag == 'DS_IC'     : ds_ic_name = str(value)
       
            f.close()
            
        if airpath > 0:
            print 'airpath: ', airpath 
        else :
            print 'no airpath absorption'
        if len(srcurrent_name):
            print 'srcurrent_name: ', srcurrent_name 
        else:
            print 'default srcurrent_name'
        
           
        e_list = ['Ca', 'Fe', 'Cu']

        std = maps_conf.axo
        std.name = 'axo_std.mca'

        if total_number_detectors > 0: 
            if total_number_detectors > 1:
                std.name = std.name + suffix
                print ' and suffix is: ', suffix
        
        have_axo = 0     
        try:
            f = open(os.path.join(self.main['master_dir'],std.name), 'rt') 
            f.close()
            print 'this_axo_filename: ', std.name        
            have_axo = 1
        except: 
            print 'Could not open axo standard: ', std.name
            return
            
        if have_axo == 1 :
            filename = os.path.join(self.main['master_dir'],std.name)

            calibration, data, date, live_time, real_time, current, IC_US, IC_DS, us_amp, ds_amp = self.read_calibration(filename, 
                                                                                                                             srcurrent_name = srcurrent_name,                                                                                                                             us_ic_name = us_ic_name,                                                                                                                             ds_ic_name = ds_ic_name)
            if data.size <=1 :
                print 'error: AXO standard does not contain data : ' + std.name
            no_mca_detectors = data.shape
            if len(no_mca_detectors) == 1 :
                no_mca_detectors = 1 
            else:
                no_mca_detectors = no_mca_detectors[1]
            # if cannot find the SRcurrent, just set it to 100.
            if current == 0 :current = 100.

            std.calibration.offset = calibration['offset']
            std.calibration.slope = calibration['slope']
            std.calibration.quad = calibration['quad']                                         
            std.live_time = live_time
            std.real_time = real_time
            std.current = current
            std.us_ic = IC_US
            std.ds_ic = IC_DS
            std.us_amp = us_amp
            std.ds_amp = ds_amp
        
                              
        #DO_fit = make_maps_conf.use_fit       
        #;; force always fitting of std standards
        DO_fit = 1      
        
        maps_defs = maps_definitions.maps_definitions()
        
        chan_names = []
        chan_calib = []
        for i in range(len(self.maps_conf.chan)):
            chan_names.append(self.maps_conf.chan[i].name)
            chan_calib.append(self.maps_conf.chan[i].calib)
          
        if DO_fit > 0:

            spectra = maps_defs.define_spectra(self.main['max_spec_channels'], 
                                                self.main['max_spectra'], 
                                                self.main['max_ICs'],
                                                mode = 'spec_tool')


            for n in range(no_mca_detectors) :
                data_temp = spectra[0].data.copy()
                data_temp[:] = 0.
                if data[:, n].size > 2048 : 
                    data_temp[0:2047] = data[0:2047, n]
                else: 
                    data_temp[0:data[:, n].size] = data[:, n]

                filename = ' '
                append = 0
                name = std.name.strip()
                name = name.split('/') 
                name = name[-1]
                name = name.split('\\')  
                name = name[-1]  
                
                ic_us = 0
                ic_ds = 0   
                 
                DO_NOT_MOD_name = 1             
                maps_defs.push_spectrum(filename, spectra, append = append, n_channels = data_temp.size, n_detector_elements = 1, real_time = real_time[n], 
                      live_time = live_time[n], current = current, calibration = calibration, counts_us_ic = ic_us, counts_ds_ic = ic_ds, roi_area = 0., 
                      roi_pixels = 1., us_amp = us_amp, ds_amp = ds_amp, n_spatial_rois = 1, year = 0, data = data_temp, name = name, 
                      DO_NOT_MOD_name = DO_NOT_MOD_name, date = date)


                dofit_spec = 1
                if dofit_spec == 0: 
                    #print 'keyword dofit_spec = 0'
                    fit_this_spec = 0
                    current_spec = fit_this_spec
                    used_chan = spectra[current_spec].used_chan
                    this_spectrum = spectra[current_spec].data[0:used_chan-1]
                    this_spectrum = this_spectrum.astype(np.float)
                    first = 1
                    calib = spectra[current_spec].calib
          
                    
                    u, fitted_spec, background, xmin, xmax, perror = fit.fit_spectrum(fitp, this_spectrum, used_chan, calib, 
                                 first = first, matrix = True, maxiter = maxiter)


                fitp.g.no_iters = 4

                if self.maps_conf.use_fit == 2:
                    this_w_uname = "DO_FIT_ALL_W_TAILS" 
                    fitp, avg_fitp, spectra = self.do_fits(this_w_uname, fitp, dofit_spec, spectra, per_pix = 1, generate_img = 1, maxiter = maxiter, suffix = suffix, info_elements = info_elements)            
                    #maps_DO_fits, Event, this_w_uname, fitp, per_pix = 1, generate_img = 1, maxiter = maxiter, suffix = suffix
                else:
                    this_w_uname = "DO_MATRIX_FIT"   
                    fitp, avg_fitp, spectra = self.do_fits(this_w_uname, fitp, dofit_spec, spectra, per_pix = 1, generate_img = 1, maxiter = maxiter, suffix = suffix, info_elements = info_elements)                              
                    #maps_DO_fits, Event, this_w_uname, fitp, per_pix = 1, generate_img = 1, maxiter = maxiter, suffix = suffix
                    

                std.calibration.offset[0] = fitp.s.val[0]
                std.calibration.slope[0] = fitp.s.val[1]
                std.calibration.quad[0] = fitp.s.val[2]
        
                if n == 0:
                    names = fitp.s.name
                    values = np.zeros((fitp.g.n_fitp, no_mca_detectors))
      
                if std.live_time[n] != 0. :
                    values[:, n] = fitp.s.val/std.live_time[n]
                else:
                    values[:, n] = 0.
       
    
            if len(values.shape) == 2 :
                values = np.sum(values, axis = 1)
            for jj in range(len(e_list)): 
                weight_ugr_cm = 0.
                weight_ugr_cm = self.lookup_axo_standard_weight(e_list[jj])
    
                axo_e_cal_factor[jj, 0] = (weight_ugr_cm*std.current)
                axo_e_cal_factor[jj, 1] = (weight_ugr_cm*std.us_ic)
                axo_e_cal_factor[jj, 2] = (weight_ugr_cm*std.ds_ic)
               
                if e_list[jj] not in chan_names: continue
                wo = chan_names.index(e_list[jj])


                wo_a = np.where(fitp.s.name == e_list[jj])
                if len(wo_a[0]) == 0 :
                    continue
                wo_a=wo_a[0]
                
                counts = values[wo_a]
                axo_e_cal[wo, 1, 0] = axo_e_cal_factor[jj, 0]/counts
                axo_e_cal[wo, 1, 1] = axo_e_cal_factor[jj, 1]/counts
                axo_e_cal[wo, 1, 2] = axo_e_cal_factor[jj, 2]/counts
                axo_e_cal[wo, 1, 3] = counts
                axo_e_cal[wo, 1, 4] = np.sqrt(counts)/counts # error fraction
                if self.maps_conf.version >= 8 :
                    axo_e_cal[wo, 1, 5] = counts
                
              
        else:           #IF keyword_set(DO_fit) THEN BEGIN  
            # define fitp, as this also defines add_pars, which are needed below
            fp = maps_fit_parameters.maps_fit_parameters()
            fitp = fp.define_fitp(self.main['beamline'], info_elements) 
                                 
        # The result of the fit is
        # nomalized by live time, below, the spectral data is normalized,
        # before ROIs are applied. 
        for n in range(no_mca_detectors):
            if std.live_time[n] != 0. :
                data[:, n] = data[:, n]/std.live_time[n] 
            else:
                data[:, n] = 0.
                
         
        # did above for fitted quantification, below for roi based quantification
        for jj in range(len(e_list)): 
            weight_ugr_cm = 0.
            weight_ugr_cm = self.lookup_axo_standard_weight(e_list[jj])
    
            axo_e_cal_factor[jj, 0] = (weight_ugr_cm*std.current)
            axo_e_cal_factor[jj, 1] = (weight_ugr_cm*std.us_ic)
            axo_e_cal_factor[jj, 2] = (weight_ugr_cm*std.ds_ic)
            
            if e_list[jj] not in chan_names: continue
            wo = chan_names.index(e_list[jj])

            counts = 0.
            for kk in range(no_mca_detectors):
                # note: center position for peaks/rois is in keV, widths of ROIs
                # is in eV
                left_roi = int(((self.maps_conf.chan[wo].center-self.maps_conf.chan[wo].width/2./1000.) - std.calibration.offset[kk])/std.calibration.slope[kk])
                if left_roi < 0 : left_roi = 0
                right_roi = int(((self.maps_conf.chan[wo].center+self.maps_conf.chan[wo].width/2./1000.) - std.calibration.offset[kk])/std.calibration.slope[kk])
                if right_roi >= data[:, kk].size : right_roi = data[:, kk].size-1 
    
                roi_width = right_roi-left_roi+1
                counts_temp = np.sum(data[left_roi:right_roi+1, kk])
                counts = counts+counts_temp
                    

            #print self.maps_conf.chan[wo].name, 'counts = ', counts
            axo_e_cal[wo, 0, 0] = axo_e_cal_factor[jj, 0]/counts
            axo_e_cal[wo, 0, 1] = axo_e_cal_factor[jj, 1]/counts
            axo_e_cal[wo, 0, 2] = axo_e_cal_factor[jj, 2]/counts
            axo_e_cal[wo, 0, 3] = counts
            axo_e_cal[wo, 0, 4] = np.sqrt(counts)/counts # error fraction
            
            if self.maps_conf.version >= 8 :
                axo_e_cal[wo, 0, 5] = counts
          
               

        # below for ROI+ quantification
        axo_e_cal[:, 2, :] = axo_e_cal[:, 0, :] 
        
#         TEST_crosstalk_removal = 0
#         if TEST_crosstalk_removal == 1 :
#              print, 'TESTING CROSS TALK REMOVAL'
#              kk = 0
#              reuse_fitp = 0
#              element_pos = [keywords.kele_pos, keywords.lele_pos, keywords.mele_pos]
#              fitp.s[*].use = 1
#              fitp.s[min(keywords.kele_pos):max(keywords.mele_pos)-1].val = 1e-10
#              FOR j = 0, keywords.kele_pos[0]-1 DO  fitp.s[j].use = fitp.s[j].batch[1]
#              det = kk
#              pileup_string = ''
#              read_fit_parameters, fitp, maps_overridefile, string, test_string, det = det, pileup_string = pileup_string
#              FOR jj = 0, fitp.g.n_fitp-1 DO BEGIN 
#                 wo_a = where(test_string EQ fitp.s[jj].name)
#                 IF wo_a[0] NE -1 THEN BEGIN 
#                    fitp.s[jj].val = 1.
#                    fitp.s[jj].use = 5
#                    IF NOT(keyword_set(temp)) THEN temp = jj ELSE temp = [temp, jj]                                
#                 endIF 
#              ENDFOR  
#              n_pars = fitp.g.n_fitp
#              parinfo = replicate({value:0.D, fixed:0, limited:[0, 0], limits:[0.D, 0], relstep:0.D, mpmaxstep:0.D, mpminstep:0.D}, n_pars)        
#              FOR i = 0, n_pars-1 DO BEGIN 
#                 parinfo[i].value = double(fitp.s[i].val)
#                 wo = where(keywords.peaks EQ i)
#                 IF wo[0] NE -1 THEN parinfo[i].value = alog10(fitp.s[i].val)
#              ENDFOR                     ;       i = 0, n_pars-1 DO BEGIN         
#              x = dindgen(n_elements(data_temp))        
#              add_matrixfit_pars = dblarr(6)
#              add_matrixfit_pars[0:2] = fitp.s[keywords.energy_pos[0:2]].val
#              add_matrixfit_pars[3:5] = fitp.s[keywords.added_params[1:3]].val        
#              maps_generate_fitmatrix, x, parinfo.value       
#              
#              FOR mm = 0, n_elements(elements_to_use)-1 DO BEGIN         
#                 this_element_spec = fitmatrix[*, elements_to_use[mm]]
#                 this_fraction_arr = fltarr(n_elements(elements_to_use))
#                 FOR jj = 0, n_elements(e_list)-1 DO BEGIN 
#                    weight_ugr_cm = 0.
#                    maps_lookup_axo_standard_weight, e_list[jj], weight_ugr_cm, old_ratio = old_ratio
#                    axo_e_cal_factor[jj, 0] = (weight_ugr_cm*std.current)
#                    axo_e_cal_factor[jj, 1] = (weight_ugr_cm*std.us_ic)
#                    axo_e_cal_factor[jj, 2] = (weight_ugr_cm*std.ds_ic)           
#                    wo = where(make_maps_conf.chan.name EQ e_list[jj])
#                    IF wo[0] EQ -1 THEN continue
#                    counts = 0.
#                    left_roi = fix(((make_maps_conf.chan[wo].center-make_maps_conf.chan[wo].width/2./1000.) - std.calibration.offset[kk])/std.calibration.slope[kk])
#                    IF left_ROI LT 0 THEN LEFT_ROI = 0
#                    right_roi = fix(((make_maps_conf.chan[wo].center+make_maps_conf.chan[wo].width/2./1000.) - std.calibration.offset[kk])/std.calibration.slope[kk])
#                    IF right_roi GE n_elements(data[*, kk]) THEN right_roi = n_elements(data[*, kk])-1 
#                    roi_width = right_roi-left_roi+1           
#                    this_fraction = total(data[left_roi:right_roi, kk])
#                    IF jj NE mm THEN  this_fraction = -1. * this_fraction
#                    this_fraction_arr[jj] = this_fraction
#                 ENDFOR                  ;FOR jj = 0, n_elements(e_list)-1 DO BEGIN 
#         
#                 max = max(abs(this_fraction_arr))
#                 this_fraction_arr = this_fraction_arr / max[0]
#                 where_NOT_finite = where(finite(this_fraction_arr) EQ 0.)
#                 IF where_NOT_finite[0] NE -1 THEN this_fraction_arr[where_NOT_finite] = 0.
#                 where_very_small = where(abs(this_fraction_arr) lt 0.0001)
#                 IF where_very_small[0] NE -1 THEN this_fraction_arr[where_very_small] = 0.
#         
#                 FOR jj = 0, n_elements(elements_to_use)-1 DO BEGIN         
#         
#                    counts = counts + total(data[left_roi:right_roi, kk])* this_fraction_arr[jj]
#         
#                    axo_e_cal[wo[0], 0, 0] = axo_e_cal_factor[jj, 0]/counts
#                    axo_e_cal[wo[0], 0, 1] = axo_e_cal_factor[jj, 1]/counts
#                    axo_e_cal[wo[0], 0, 2] = axo_e_cal_factor[jj, 2]/counts
#                    axo_e_cal[wo[0], 0, 3] = counts
#                    axo_e_cal[wo[0], 0, 4] = sqrt(counts)/counts ; error fraction                      
#                    IF make_maps_conf.version GE 8 THEN BEGIN 
#                       axo_e_cal[jj, 2, 6] = axo_e_cal[wo[0], 2, 6] + axo_e_cal[wo[0], 0, 6] * this_fraction_arr[jj]
#                    ENDIF           
#                 ENDFOR ;;FOR jj = 0, n_elements(elements_to_use)-1 DO BEGIN         
#                 
#              ENDFOR    ;;FOR mm = 0, n_elements(elements_to_use)-1 DO BEGIN         
#           ENDIF        ;;   IF TEST_crosstalk_removal EQ 1 THEN BEGIN 
#         
#         
#         
#           IF (keyword_set(winID.fit_spec)) THEN widget_control, winID.fit_spec, /destroy 
    
        
        #Look for override files in main.master_dir
        if (total_number_detectors > 1) : 
            overide_files_found = 0 
            suffix = str(this_detector) 
            maps_overridefile = os.path.join(self.main['master_dir'],'maps_fit_parameters_override.txt')+suffix
            try:
                f = open(maps_overridefile, 'rt')    
                print maps_overridefile, ' exists.'
                f.close()
            except :
                # if i cannot find an override file specific per detector, assuming
                # there is a single overall file.
                maps_overridefile = os.path.join(self.main['master_dir'],'maps_fit_parameters_override.txt')  
        else:
            maps_overridefile = os.path.join(self.main['master_dir'],'maps_fit_parameters_override.txt')  
       
        
        try:
            f = open(maps_overridefile, 'rt')    
            print 'this override filename: ', maps_overridefile, ' exists.'
        except :
            print  'Warning: did not find the following file: ', maps_overridefile, '   Please make sure the file is present in the parent directory, and try again. For now, I am aborting this action.'
            return 
                        

        for line in f:
            if ':' in line : 
                slist = line.split(':')
                tag = slist[0]
                value = ''.join(slist[1:])
                
                if tag == 'DETECTOR_MATERIAL' : 
                    temp = int(value)
                    # if eq to 1, it is a silicon based detector
                    if temp == 1 : 
                        self.maps_conf.add_long['a'] = 1
                        self.maps_conf.fit_t_ge = 0.

                elif tag == 'COHERENT_SCT_ENERGY' : 
                    self.maps_conf.incident_E = float(value)

                elif tag == 'BE_WINDOW_THICKNESS' : 
                    self.maps_conf.fit_t_be = float(value)*1000.
                 
                elif tag == 'GE_DEAD_LAYER' : 
                    if self.maps_conf.add_long['a'] != 1 : self.maps_conf.fit_t_ge = float(value)*1000.
                
                elif tag == 'DET_CHIP_THICKNESS' : 
                    self.maps_conf.add_float['a'] = float(value)*1000.
                
                elif tag == 'BRANCHING_RATIO_ADJUSTMENT_K' : 
                    temp_string = value.split(' ,')
                    for ts in temp_string: ts.strip()
                    
                    if temp_string[0] in fitp.s.name:
                        wo = fitp.s.name.index(temp_string[0])
                        ii = wo - np.amin(fitp.keywords.kele_pos) 
                        if len(temp_string) == 5 :
                            if temp_string[0] in info_elements.name:
                                j = info_elements.name.index(temp_string[0])
                                # adjust branching ratios within families, but all relative to Ka1
                         
                                fitp.add_pars[ii, 0].ratio = 1.
                                fitp.add_pars[ii, 1].ratio = info_elements[j].xrf_abs_yield.ka2/info_elements[j].xrf_abs_yield.ka1
                                fitp.add_pars[ii, 2].ratio = info_elements[j].xrf_abs_yield.kb1/info_elements[j].xrf_abs_yield.ka1
                                fitp.add_pars[ii, 3].ratio = info_elements[j].xrf_abs_yield.kb2/info_elements[j].xrf_abs_yield.ka1
            
                                if (len(temp_string) >= 5) :
                                    for jj in range(4):
                                        fitp.add_pars[ii, jj].ratio = fitp.add_pars[ii, jj].ratio*float(temp_string[(jj+1)])

        f.close()
        
        
        axo_e_cal[np.where(np.isfinite(axo_e_cal) == False)] = 0.
    
        use_henke = 1

        beta_delta_arr = np.zeros((self.maps_conf.n_chan, 2, 4))
        #*,*,0: photoelectric abosption
        #*,*,1: be 
        #*,*,2: ge 
        #*,*,2: Si
        energy_yield_arr = np.zeros((self.maps_conf.n_chan, 3))
        #*,*,0: energy
        #*,*,1: lambda
        #*,*,2: yield
      
  
        for mm in range(self.maps_conf.n_chan) :        
            jump_factor = 0.
            total_jump_factor = 0.
            element_temp = -1
             
            # if not to be used for calibration, then skip
            if self.maps_conf.chan[mm].calib == 0 : continue
            if self.maps_conf.chan[mm].calib == 1 :
                ienames = []
                for ie in info_elements: ienames.append(ie.name)
                if self.maps_conf.chan[mm].name not in ienames: continue
                element_temp = ienames.index(self.maps_conf.chan[mm].name)
                ev = info_elements[element_temp].xrf['ka1']*1000.
                yieldd = info_elements[element_temp].xrf_abs_yield['ka1']
                rel_yield = info_elements[element_temp].xrf_abs_yield['ka1']/ \
                            np.sum(info_elements[element_temp].xrf_abs_yield['ka1']+info_elements[element_temp].xrf_abs_yield['ka2']+\
                                  info_elements[element_temp].xrf_abs_yield['kb1']+info_elements[element_temp].xrf_abs_yield['kb2'])
                newrel_yield = fitp.add_pars[mm, 0].ratio/ \
                               np.sum(fitp.add_pars[mm, 0].ratio+fitp.add_pars[mm, 1].ratio+fitp.add_pars[mm, 2].ratio+fitp.add_pars[mm, 3].ratio)
        
                yieldd = newrel_yield * info_elements[element_temp].yieldD['k']
        
                name =  self.maps_conf.chan[mm].name
                k_yield = info_elements[element_temp].yieldD['k']         
                #print name, ' old_yield', yieldd, ' rel_yield', rel_yield, ' k_yield', k_yield, ' rel_yield * k_yield:', rel_yield* k_yield, ' newrel_yield', newrel_yield,  \
                #         '[mm, 0] ', fitp.add_pars[mm, 0].ratio, '[mm, 1] ', fitp.add_pars[mm, 1].ratio, '[mm, 2] ', fitp.add_pars[mm, 2].ratio, '[mm, 3] ', fitp.add_pars[mm, 3].ratio, ' old_rel_yield/new_rel_yield:', rel_yield/newrel_yield
                if self.maps_conf.incident_E > info_elements[element_temp].bindingE['K']: 
                    jump_factor = info_elements[element_temp].jump['K'] 

                yieldd = k_yield         #*rel_yield

            
            if self.maps_conf.chan[mm].calib == 2 :
                ienames = []
                for ie in info_elements: ienames.append(ie.name+'_L')
                if self.maps_conf.chan[mm].name not in ienames: continue
                element_temp = ienames.index(self.maps_conf.chan[mm].name)
                ev = info_elements[element_temp].xrf['la1']*1000.
                yieldd = info_elements[element_temp].xrf_abs_yield['la1']
                name =  self.maps_conf.chan[mm].name[:-2]
                jump_factor = info_elements[element_temp].jump['L3'] 
                if self.maps_conf.incident_E > info_elements[element_temp].bindingE['L2'] : 
                    total_jump_factor = info_elements[element_temp].jump['L2']
                if self.maps_conf.incident_E > info_elements[element_temp].bindingE['L1'] : 
                    total_jump_factor = total_jump_factor*info_elements[element_temp].jump['L1']
                #print  name, ' L jump_factor: ', jump_factor
          

            if self.maps_conf.chan[mm].calib == 3 :
                ienames = []
                for ie in info_elements: ienames.append(ie.name+'_M')
                if self.maps_conf.chan[mm].name not in ienames: continue
                element_temp = ienames.index(self.maps_conf.chan[mm].name)
                ev = self.maps_conf.chan[mm].center*1000.
                yieldd = info_elements[element_temp].xrf_abs_yield['ma1']
                name = self.maps_conf.chan[mm].name[:-2]
                jump_factor = info_elements[element_temp].jump['M5'] 
                total_jump_factor = info_elements[element_temp].jump['M1']*info_elements[element_temp].jump['M2']*\
                                    info_elements[element_temp].jump['M3']*info_elements[element_temp].jump['M4']
                #print  name, ' M jump_factor: ', jump_factor, ' yield ', yieldd, ' ev: ', ev, ' total_jump_factor: ', total_jump_factor
        
        
            if element_temp == -1 : continue 
            if jump_factor != 0. :
                if total_jump_factor == 0. :
                    shell_factor = (jump_factor-1.)/jump_factor 
                else: 
                    shell_factor =  (jump_factor-1.)/jump_factor / total_jump_factor

            else:
                shell_factor = 0.
            if (name == 'U') and (self.maps_conf.chan[mm].calib == 3) :
                U_shell_factor = shell_factor
        
            # beta proportional to f2, 
             
            # make sure name is known to henke routine, if not, skip
            # in case this is Pu, use U instead
            if (name == 'Pu') or (name == 'Np') : 
                print 'name was ', name, ' resetting it to uranium to work with henke data'
                name = 'U'
                
            Chenke = henke.henke()
            
            test = []
            test = Chenke.zcompound(name, test)
            if np.sum(test) == 0 : continue
            # replace straight henke routines, with those
            # that take the absorption edges into account
            # make sure we are a bit above the absorption edge to make sure that for calibration purposes we do not eoncouner any weird things.                
            f1, f2, delta, beta = Chenke.get_henke_single(name, info_elements[element_temp].density, (self.maps_conf.incident_E+0.1)*1000.) 
            #f1, f2, delta, beta = Chenke.get_henke(name, info_elements[element_temp].density, (self.maps_conf.incident_E+0.1)*1000.) 
            # stds in microgram/cm2
            # density rho = g/cm3 = 1 microgram/cm2 /1000/1000/cm = 1 microgram/cm2 /1000/1000/*10*1000/um = 1 microgram/cm2 /100/um 
            # thickness for 1 ugr/cm2
            # =1/(density[g/cm3]/10)
            thickness = 1./(info_elements[element_temp].density * 10.)*1000.
            aux_arr[mm, 0] = self.absorption(thickness, beta, 1239.852/((self.maps_conf.incident_E+0.1)*1000.), shell_factor = shell_factor)
            f1, f2, delta, beta  = Chenke.get_henke_single('Be', 1.848, ev) 
        
            aux_arr[mm, 1] = self.transmission(self.maps_conf.fit_t_be, beta, 1239.852/ev)
            f1, f2, delta, beta  = Chenke.get_henke_single( 'Ge', 5.323, ev) 
            aux_arr[mm, 2] = self.transmission(self.maps_conf.fit_t_ge, beta, 1239.852/ev)
            aux_arr[mm, 3] = yieldd
            if self.maps_conf.add_long['a'] == 1 :
                f1, f2, delta, beta  = Chenke.get_henke_single( 'Si', 2.3, ev) 
            aux_arr[mm, 4] = self.transmission(self.maps_conf.add_float['a'], beta, 1239.852/ev)
            if (self.maps_conf.add_float['a'] == 0.) or (self.maps_conf.add_long['a'] != 1) :
                aux_arr[mm, 4] = 0.
            if airpath > 0: 
                density = 1.0
                #f1, f2, delta, beta, graze_mrad, reflect, inverse_mu, atwt = Chenke.get_henke_single('air', density, ev) 
                f1, f2, delta, beta = Chenke.get_henke_single('air', density, ev) 
                aux_arr[mm, 5] = self.transmission(airpath*1000., beta, 1239.852/ev)  # airpath is read in microns, transmission function expects nm
            else:
                aux_arr[mm, 5] = 1.


        if 'Al' in chan_names:
            used_elements = [chan_names.index('Al')]
            used_elements.append(chan_names.index('Si'))
        else:
            used_elements = [chan_names.index('Si')]
      
 
        temp_string = ['K', 'Ca', 'Ti', 'V']
        if self.maps_conf.incident_E > 6.54 : temp_string.append('Mn')
        if self.maps_conf.incident_E > 7.11 : temp_string.append('Fe')
        if self.maps_conf.incident_E > 7.71 : temp_string.append('Co')
        if self.maps_conf.incident_E > 8.98 : temp_string.append('Cu')
        if self.maps_conf.incident_E > 9.66 : temp_string.append('Zn')
        temp_string.append('Pb_M')
        if self.maps_conf.incident_E > 13.03 : temp_string.append('Pb_L')
      
        for i in range(len(temp_string)):
            if temp_string[i] in chan_names:
                wo = chan_names.index(temp_string[i])
                used_elements.append(wo)


        m_used_elements = []
        if 'Pb_M' in chan_names:
            m_used_elements.append(chan_names.index('Pb_M'))
#         if 'Au_M' in chan_names:
#             m_used_elements.append(chan_names.index('Au_M'))
                
        wo_m = [chan_names.index(s) for s in chan_names if "_M" in s]


        e_cal_shape = self.maps_conf.e_cal.shape
        temp_calib = np.zeros((e_cal_shape[0]))
        m_temp_calib = np.zeros((e_cal_shape[0]))
    
    
        self.aux_arr = aux_arr
        rrange = 3 
        for k in range(rrange): 
            for l in range(3):
                if l == 0 : factor =  std.current
                if l == 1 : factor =  std.us_ic
                if l == 2 : factor =  std.ds_ic
                for iel in used_elements:
                    temp_calib[iel] = axo_e_cal[iel, k, l] 

                if len(m_used_elements) > 0 : 
                    m_temp_calib[m_used_elements] = axo_e_cal[m_used_elements, k, l]
        
                x = used_elements[:]
                y = 1./temp_calib[used_elements]
                y = np.array(y)
                err = y/20.+np.sqrt(y)/10.
                #set the last three errors to be very large so
                #they do not impact the fiting. the last three
                #correspond to PbL, M, Zn          
                #err[len(err)-1] = 100000.
                #if 'Pb_L' in temp_string: 
                #    err[len(err)-2] = 100000.
                # to remove Al and Si from fit:
                #err[0] = 100000.
                #err[1] = 10000.
        
                wo = np.where(np.isfinite(y) == False)
        
                if len(wo[0]) > 0 : 
                    y[wo] = 0.
                    err[wo] = 1000.

                
                n_pars = 2
                parinfo_value = np.zeros((n_pars)) 
                parinfo_fixed = np.zeros((n_pars), dtype=np.int)  
                parinfo_limited = np.zeros((n_pars, 2), dtype = np.int)
                parinfo_limits = np.zeros((n_pars, 2)) 
                parinfo_relstep = np.zeros((n_pars)) 
                parinfo_mpmaxstep = np.zeros((n_pars)) 
                parinfo_mpminstep = np.zeros((n_pars))
                            
                parinfo_value[0] = 100000.0/factor
                parinfo_value[1] = 0. # [1 micron initial thickness]     air
                if airpath >0 : parinfo_value[1] = float(airpath)
                parinfo_fixed[1] =  1
                parinfo_limited[0,0] = 1
                parinfo_limits[0,0] = 0.0

                bounds=[]
                have_bounds = 0
                for i in range(n_pars):
                    havelimit = parinfo_limited[i,:]
                    if havelimit[0] == 1:
                        bmin = parinfo_limits[i,0]
                        have_bounds = 1
                    else:
                        bmin = None
                    if havelimit[1] == 1:
                        bmax = parinfo_limits[i,1]
                        have_bounds = 1
                    else:
                        bmax = None
                    bounds.append((bmin,bmax))
    
                
                Clb = leastsqbound

                p0 = np.array(parinfo_value)
                p1,cov,infodict,mesg,self.success = Clb.leastsqbound(self.residuals, p0, bounds, args=(y, x), maxfev=maxiter, full_output = True)      
                
                #perror1 = self.calc_perror(infodict['fjac'], infodict['ipvt'], len(p1))
                
                temp_x = np.where(np.array(chan_calib) >= 1)[0]
                curve = self.fit_calibrationcurve(temp_x, p1)
                #print k, ' ', l, '   u:', p1[0], '   factor[I, IC] ', factor
                axo_e_cal[:, k, l] = 0.
                for ic in range(len(self.maps_conf.chan)):
                    for iun in range(4) : self.maps_conf.chan[ic].units[iun] = 'cts/s'
                for tx in temp_x:
                    for iun in range(3) : self.maps_conf.chan[tx].units[iun] = 'ug/cm^2'          
                axo_e_cal[temp_x, k, l] = curve
                if len(m_used_elements) > 1 :
                    # now do seperate calibration using M lines
                    x = m_used_elements[:]
                    y = 1./m_temp_calib[m_used_elements]
                    err = y/20.+np.sqrt(y)/10.
                    wo = np.where(np.isfinite(y) == False)[0]
                    if len(wo) > 0 :
                        y[wo] = 0.
                        err[wo] = 1000.
                    n_pars = 2
                    parinfo_value = np.zeros((n_pars)) 
                    parinfo_fixed = np.zeros((n_pars), dtype=np.int)  
                    parinfo_limited = np.zeros((n_pars, 2), dtype = np.int)
                    parinfo_limits = np.zeros((n_pars, 2)) 
                    parinfo_relstep = np.zeros((n_pars)) 
                    parinfo_mpmaxstep = np.zeros((n_pars)) 
                    parinfo_mpminstep = np.zeros((n_pars))
   
                    parinfo_value[0] = 100000.0/factor
                    parinfo_value[1] = 0.0 # [1 micron initial thickness]     air
                    if airpath > 0 : parinfo_value[1] = float(airpath)
                    parinfo_fixed[1] =  1
                    parinfo_limited[0,0] = 1
                    parinfo_limits[0,0] = 0.0
                    perror = 0
                    quiet = 1
                    ERRMSG = ''
                    
                    bounds=[]
                    have_bounds = 0
                    for i in range(n_pars):
                        havelimit = parinfo_limited[i,:]
                        if havelimit[0] == 1:
                            bmin = parinfo_limits[i,0]
                            have_bounds = 1
                        else:
                            bmin = None
                        if havelimit[1] == 1:
                            bmax = parinfo_limits[i,1]
                            have_bounds = 1
                        else:
                            bmax = None
                        bounds.append((bmin,bmax))
                    

                    p0 = np.array(parinfo_value)
                    p1,cov,infodict,mesg,self.success = Clb.leastsqbound(self.residuals, p0, bounds, args=(y, x), maxfev=maxiter, full_output = True)      
                    curve = self.fit_calibrationcurve(wo_m, p1)
                    
                    for ie in wo_m:
                        self.maps_conf.chan[ie].units[0:3] = 'ug/cm^2'          
                    axo_e_cal[wo_m, k, l] = curve



        if no_nbs == 1:
            maps_conf.calibration.slope = std.calibration.slope
            maps_conf.calibration.offset = std.calibration.offset
            maps_conf.calibration.quad = std.calibration.quad
            maps_conf.e_cal = axo_e_cal
            
        maps_conf.axo = std
        maps_conf.axo_e_cal = axo_e_cal 
        self.axo_calibration_write_info( axo_e_cal_factor, old_ratio = old_ratio, suffix = suffix, aux_arr = aux_arr, info_elements = info_elements)
        
        
        return

 
#-----------------------------------------------------------------------------   
    def lookup_axo_standard_weight(self, this_element):
        
        weight_ugr_cm = 0

        if this_element == 'Pb' : weight_ugr_cm = 0.761
        if this_element == 'Ca' : weight_ugr_cm = 1.931
        if this_element == 'Fe' : weight_ugr_cm = 0.504
        if this_element == 'Cu'  : weight_ugr_cm = 0.284
        if this_element == 'Mo'  : weight_ugr_cm = 0.132
        if this_element == 'Pd'  : weight_ugr_cm = 0.18
        if this_element == 'La'  : weight_ugr_cm = 1.1
        
        return weight_ugr_cm
                             
#-----------------------------------------------------------------------------   
    def lookup_nbs_standard_weight(self, this_element, old_ratio = 0):
        
        weight_ugr_cm = 0.0

        if this_element == 'Al' : weight_ugr_cm = 13.7374 /1.25 #Al 13.7374
        if this_element == 'Si' : weight_ugr_cm = 33. /1.158    #Si 33.24
        if this_element == 'Ca' : weight_ugr_cm = 18.5 /1.07    #Ca
        if this_element == 'V' : weight_ugr_cm = 4.19/1.03      #V
        if this_element == 'Mn'  : weight_ugr_cm = 4.22 /1.017  #Mn
        if this_element == 'Co'  : weight_ugr_cm = 0.935 /1.01  #Co
        if this_element == 'Cu'  : weight_ugr_cm = 2.246 /1.007 #Cu
        
        #nbs 1833:
        #    if this_element == 'Si' : weight_ugr_cm = 33.24/1.158              #Si 
        if this_element == 'Ar' : weight_ugr_cm = 0.           #Ar for sanity checks
        if this_element == 'K' : weight_ugr_cm = 17.1/1.095    #K 
        if this_element == 'Ti'  : weight_ugr_cm = 12.7/1.04   #Ti
        if this_element == 'Fe'  : weight_ugr_cm = 14.14/1.014 #Fe
        if this_element == 'Zn'  : weight_ugr_cm = 3.89/1.005  #Zn
        if this_element == 'Pb_M'  : weight_ugr_cm = 16.42/1.1 #Pb-M
        if this_element == 'Pb_L'  : weight_ugr_cm = 16.42/1.0 #Pb-L
        
        
        if old_ratio > 0 :              
            if this_element == 'Al' : weight_ugr_cm = 13.7374 /1.17 #Al 13.7374
            if this_element == 'Si' : weight_ugr_cm = 33.00 /1.14   #Si 33.24
            if this_element == 'Ca' : weight_ugr_cm = 18.5 /1.04    #Ca
            if this_element == 'V' : weight_ugr_cm = 4.19/1.02      #V
            if this_element == 'Mn'  : weight_ugr_cm = 4.22 /1.02   #Mn
            if this_element == 'Co'  : weight_ugr_cm = 0.935 /1.01  #Co
            if this_element == 'Cu'  : weight_ugr_cm = 2.246 /1.01  #Cu
        # nbs 1833:
        #     if this_element == 'Si' : weight_ugr_cm = 33.24/1.14               #Si 
            if this_element == 'Ar' : weight_ugr_cm = 0.           #Ar for sanity checks
            if this_element == 'K' : weight_ugr_cm = 17.1/1.07     #K 
            if this_element == 'Ti'  : weight_ugr_cm = 12.7/1.04   #Ti
            if this_element == 'Fe'  : weight_ugr_cm = 14.14/1.02  #Fe
            if this_element == 'Zn'  : weight_ugr_cm = 3.89/1.01   #Zn
            if this_element == 'Pb_M'  : weight_ugr_cm = 16.42/1.1 #Pb-M
            if this_element == 'Pb_L'  : weight_ugr_cm = 16.42/1.0 #Pb-L
          
          
#         if weight_ugr_cm == 0.0:
#             print 'Warning, element ', this_element, ' weight = ', weight_ugr_cm
            
        return weight_ugr_cm



#-----------------------------------------------------------------------------   
    def read_calibration(self, filename, srcurrent_name = '', us_ic_name ='', ds_ic_name = ''):
    
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
            
 
        line = ''
        line = f.readline() # 1. line is version
        #print line
        line = f.readline() # 2. is # elements
        slist = line.split(':')
        tag = slist[0]
        value = ''.join(slist[1:])
        n_detector_elements  = int(value)  
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
    

        calibration = { 'offset' : np.zeros((n_detector_elements)), 
                       'slope'   : np.zeros((n_detector_elements)), 
                       'quad'   : np.zeros((n_detector_elements)) }

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
                elif tag == 'ROIS' : 
                    n_rois = np.zeros((n_detector_elements), dtype=int)
                    value = value.split(' ')
                    valuelist = [int(x) for x in value if x != '']
                    n_rois[:] = valuelist
                    max_rois = np.max(n_rois)
                elif tag == 'REAL_TIME'  : 
                    real_time = np.zeros((n_detector_elements))
                    value = value.split(' ')
                    valuelist = [float(x) for x in value if x != '']
                    real_time[:] = valuelist                                                              
                elif tag == 'LIVE_TIME'  :  
                    live_time = np.zeros((n_detector_elements))
                    value = value.split(' ')
                    valuelist = [float(x) for x in value if x != '']
                    #print 'live time', valuelist
                    live_time[:] = valuelist      
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
                    IC_US = np.zeros((n_detector_elements))
                    value = value.split(' ')
                    valuelist = [float(x) for x in value if x != '']
                    IC_US[:] = valuelist
                elif tag == 'DOWNSTREAM_IONCHAMBER' :  
                    IC_DS = np.zeros((n_detector_elements))
                    value = value.split(' ')
                    valuelist = [float(x) for x in value if x != '']
                    IC_DS[:] = valuelist
                elif tag == 'ENVIRONMENT' :  
                    value = ':'.join(slist[1:])
                    pos = value.find('=')
                    etag = value[0:pos].strip()
                    vallist = value.split('"')
                    temp = vallist[1]
                    if etag == 'S:SRcurrentAI' :
                        current = float(temp)    
                    elif etag == '2xfm:scaler1_cts1.B' :
                        if IC_US == 0 : IC_US = float(temp)           
                    elif etag == '2xfm:scaler1_cts1.C' : 
                        if IC_DS == 0 : IC_DS = float(temp)        
                    elif etag == '2xfm:scaler3_cts1.B' : 
                        IC_US = float(temp)           
                    elif etag == '2xfm:scaler3_cts1.C' : 
                        IC_DS = float(temp) 
                    elif etag == '2idd:scaler1_cts1.C' : 
                        IC_US = float(temp)           
                    elif etag == '2idd:scaler1_cts1.B' : 
                        IC_DS = float(temp)
                    elif etag == '8bmb:3820:scaler1_cts1.B' : 
                        IC_US = float(temp)           
                    elif etag == '8bmb:3820:scaler1_cts1.C' : 
                        IC_DS = float(temp)       
                    elif etag[5:] == 'A1sens_num.VAL' : 
                        amp[0, 0] = float(temp)        
                    elif etag[5:] == 'A2sens_num.VAL' : 
                        amp[1, 0] = float(temp)          
                    elif etag[5:] == 'A3sens_num.VAL' : 
                        amp[2, 0] = float(temp)          
                    elif etag[5:] == 'A4sens_num.VAL' : 
                        amp[3, 0] = float(temp)          
                    elif etag[5:] == 'A1sens_unit.VAL' :                 
                        #print 'now:', temp
                        if (temp == "nA/V") or  (temp == "pA/V") or (temp == "uA/V") or (temp == "mA/V") : 
                            if (temp == "pA/V") : amp[0, 1] = 0
                            if (temp == "nA/V") : amp[0, 1] = 1
                            if (temp == "uA/V") : amp[0, 1] = 2
                            if (temp == "mA/V") : amp[0, 1] = 3
                        else:
                            amp[0, 1] = float(temp)          
                    elif etag[5:] == 'A2sens_unit.VAL' : 
                        if (temp == "nA/V") or  (temp == "pA/V") or (temp == "uA/V") or (temp == "mA/V") : 
                            if (temp == "pA/V") : amp[1, 1] = 0
                            if (temp == "nA/V") : amp[1, 1] = 1
                            if (temp == "uA/V") : amp[1, 1] = 2
                            if (temp == "mA/V") : amp[1, 1] = 3
                        else: 
                            amp[1, 1] = float(temp)
                    elif etag[5:] == 'A3sens_unit.VAL' : 
                        if (temp == "nA/V") or  (temp == "pA/V") or (temp == "uA/V") or (temp == "mA/V") : 
                            if (temp == "pA/V") : amp[2, 1] = 0
                            if (temp == "nA/V") : amp[2, 1] = 1
                            if (temp == "uA/V") : amp[2, 1] = 2
                            if (temp == "mA/V") : amp[2, 1] = 3
                        else: 
                            amp[2, 1] = float(temp)
                    elif etag[5:] == 'A4sens_unit.VAL' : 
                        if (temp == "nA/V") or (temp == "pA/V") or (temp == "uA/V") or (temp == "mA/V") : 
                            if (temp == "pA/V") : amp[3, 1] = 0
                            if (temp == "nA/V") : amp[3, 1] = 1
                            if (temp == "uA/V") : amp[3, 1] = 2
                            if (temp == "mA/V") : amp[3, 1] = 3
                        else:
                            amp[3, 1] = float(temp)
                    if len(srcurrent_name) > 0 : 
                        if etag == srcurrent_name : 
                            current = float(temp)
                    if len(us_ic_name) > 0 : 
                        if etag == us_ic_name : 
                            IC_US = float(temp)
                    if len(ds_ic_name) > 0: 
                        if etag == ds_ic_name : 
                            IC_DS = float(temp)
                            
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
    
    
        if self.main['beamline'] == '2-ID-D' : 
            us_amp[:] = amp[1, :]
            ds_amp[:] = amp[3, :]
    
        if self.main['beamline'] == '2-ID-E' : 
            us_amp[:] = amp[0, :]
            ds_amp[:] = amp[1, :]
   
        if self.main['beamline'] == 'Bio-CAT' :
            us_amp[:] = amp[0, :]
            ds_amp[:] = amp[1, :]
     
        if self.main['beamline'] == 'GSE-CARS' : 
            amp[0, :]= [1, 1, 1]
            amp[1, :]= [1, 1, 1]
            us_amp[:] = amp[0, :]
            ds_amp[:] = amp[1, :]
     
        if IC_DS == 0 :
            print 'warning downstream IC counts zero'
            IC_DS = 1.
    
        if IC_US == 0  : 
            print 'warning upstream IC counts zero'
            IC_US = 1.
            
    
        return calibration, data, date, live_time, real_time, current, IC_US, IC_DS, us_amp, ds_amp

                   
#-----------------------------------------------------------------------------   
    def read_nbs_calibration(self, standard_filenames,
                             this_detector = 0, 
                             total_number_detectors = 0,
                             dofit_spec = 0,
                             fit_this_spec = 0,
                             fitp = 0,
                             info_elements = 0):
        
        suffix = ''
        
        fit = maps_analyze.analyze()

        nnls = 0
        print '\ncalled nbs_calibration with these specifications: this detector: {0}; total_number_detectors: {1}.'.format(
                                                                this_detector, total_number_detectors)

        if total_number_detectors > 0: 
            if total_number_detectors > 1:
                suffix = str(this_detector) 
                print ' and suffix is: ', suffix

        maxiter = 500
        old_ratio = 0
        

        #e_cal = self.maps_conf.e_cal
        e_cal_factor = np.zeros((self.maps_conf.n_chan, 3))

        aux_arr = np.zeros((self.maps_conf.n_chan, 6))
        yield_correction = np.zeros((self.maps_conf.n_chan, 4))

        airpath = 0
        srcurrent_name = ''
        
        overide_files_found = 0 
        maps_overridefile = os.path.join(self.main['master_dir'],'maps_fit_parameters_override.txt')
        try:
            f = open(maps_overridefile, 'rt')    
            print maps_overridefile, ' exists.'
            f.close()
            overide_files_found = 1
        except :
            overide_files_found = 0 
         
        
        if overide_files_found == 1:
            f = open(maps_overridefile, 'rt')
            for line in f:
                if ':' in line : 
                    slist = line.split(':')
                    tag = slist[0]
                    value = ''.join(slist[1:])
                    
                    if   tag == 'AIRPATH'   : airpath = float(value)
                    elif tag == 'SRCURRENT' : srcurrent_name = str(value)
                    elif tag == 'US_IC'     : us_ic_name = str(value)
                    elif tag == 'DS_IC'     : ds_ic_name = str(value)
       
            f.close()
            
        if airpath > 0:
            print 'airpath: ', airpath 
        else :
            print 'no airpath absorption'
        if len(srcurrent_name):
            print 'srcurrent_name: ', srcurrent_name 
        else:
            print 'default srcurrent_name'

        maps_settingsfile = 'maps_settings.txt'     
        try:
            f = open(os.path.join(self.main['master_dir'],maps_settingsfile), 'rt')
            for line in f:
                if ':' in line : 
                    slist = line.split(':')
                    tag = slist[0]
                    value = ''.join(slist[1:])
                    if tag == 'NNLS'  :  nnls  = int(value)    
            f.close()
                      
        except: print 'Could not open maps_settings.txt.'
    
        thisnbs = 0

        for ii in range(2): 

            nbs = maps_definitions.standard()
        


            nbs_files_found = 0
            test_nbs = 0
            for fname in standard_filenames:
                if 'nbs' in fname.lower(): 
                    nbs.name = fname
                    if total_number_detectors > 1 :
                        nbs.name = nbs.name+suffix
                    print '\nthis_nbs_filename: ', nbs.name
                    
                    #Remove item from the list 
                    standard_filenames.remove(fname)
                    test_nbs = 1
                    break              
                
            
            if test_nbs == 0: 
                print 'Did not find nbs files - returning.'
                return
            

            try:
                f = open(os.path.join(self.main['master_dir'],nbs.name), 'rt')     
                nbs_files_found += 1
                f.close()
            except:
                print 'Could not open nbs file - returning.'
                return               
                

            
            #if ii == 0 :
            # NBS 1833
            if '1833' in nbs.name:
                e_list = ['Si', 'K', 'Ti']
                if self.maps_conf.incident_E > 7.11 : e_list.append('Fe')
                if self.maps_conf.incident_E > 9.66 : e_list.append('Zn')
                e_list.append('Pb_M')
                if self.maps_conf.incident_E > 13.03 : e_list.append('Pb_L')
                self.maps_conf.nbs33 = nbs
                thisnbs = 33
     
            #if ii == 1 :
            # NBS 1832
            if '1832' in nbs.name:
                e_list = ['Al', 'Si', 'Ca', 'V']
                if self.maps_conf.incident_E > 6.54 : e_list.append('Mn')
                if self.maps_conf.incident_E > 7.71 : e_list.append('Co')
                if self.maps_conf.incident_E > 8.98 : e_list.append('Cu')
                self.maps_conf.nbs32 = nbs    
                thisnbs = 32
            
            
            if nbs_files_found == 0: 
                print 'Did not find NBS calibration file, returning'
                return
            if nbs_files_found == 1:
                filename = os.path.join(self.main['master_dir'],nbs.name)
                

                calibration, data, date, live_time, real_time, current, IC_US, IC_DS, us_amp, ds_amp = self.read_calibration(filename, 
                                                                                                                             srcurrent_name = srcurrent_name, 
                                                                                                                             us_ic_name = us_ic_name, 
                                                                                                                             ds_ic_name = ds_ic_name)
                
                nbs.calibration.offset = calibration['offset']
                nbs.calibration.slope = calibration['slope']
                nbs.calibration.quad = calibration['quad']
                nbs.date = date
                nbs.live_time = live_time
                nbs.real_time = real_time       
                nbs.current = current
                nbs.us_ic = IC_US
                nbs.ds_ic = IC_DS
                nbs.us_amp = us_amp
                nbs.ds_amp = ds_amp

            
                if data.size <=1 :
                    print 'error: NBS standard does not contain data : ' + nbs.name
                no_mca_detectors = data.shape
                if len(no_mca_detectors) == 1 :
                    no_mca_detectors = 1 
                else:
                    no_mca_detectors = no_mca_detectors[1]
                # if cannot find the SRcurrent, just set it to 100.
                if current == 0 :current = 100.
                         
            #DO_fit = self.maps_conf.use_fit       
            # force always fitting of nbs standards
            DO_fit = 1           
            
            maps_defs = maps_definitions.maps_definitions()
            
            chan_names = []
            chan_calib = []
            for i in range(len(self.maps_conf.chan)):
                chan_names.append(self.maps_conf.chan[i].name)
                chan_calib.append(self.maps_conf.chan[i].calib)
              
            print 'chan_names', chan_names
            if DO_fit > 0:
                if ii == 0 :
                    spectra = maps_defs.define_spectra(self.main['max_spec_channels'], 
                                                       self.main['max_spectra'], 
                                                       self.main['max_ICs'],
                                                       mode = 'spec_tool')


                for n in range(no_mca_detectors) :
                    data_temp = spectra[0].data.copy()
                    data_temp[:] = 0.
                    if data[:, n].size > 2048 : 
                        data_temp[0:2047] = data[0:2047, n]
                    else: 
                        data_temp[0:data[:, n].size] = data[:, n]

                    filename = ' '
                    append = 0
                    name = nbs.name.strip()
                    name = name.split('/') 
                    name = name[-1]
                    name = name.split('\\')  
                    name = name[-1]  
                    
                    ic_us = 0
                    ic_ds = 0   
                     
                    DO_NOT_MOD_name = 1             
                    maps_defs.push_spectrum(filename, spectra, append = append, n_channels = data_temp.size, n_detector_elements = 1, real_time = real_time[n], 
                          live_time = live_time[n], current = current, calibration = calibration, counts_us_ic = ic_us, counts_ds_ic = ic_ds, roi_area = 0., 
                          roi_pixels = 1., us_amp = us_amp, ds_amp = ds_amp, n_spatial_rois = 1, year = 0, data = data_temp, name = name, 
                          DO_NOT_MOD_name = DO_NOT_MOD_name, date = date)



                    if dofit_spec == 0: 
                        #print 'keyword dofit_spec = 0'
                        current_spec = fit_this_spec
                        used_chan = spectra[current_spec].used_chan
                        this_spectrum = spectra[current_spec].data[0:used_chan-1]
                        this_spectrum = this_spectrum.astype(np.float)
                        first = 1
                        calib = spectra[current_spec].calib
              
                        
                        u, fitted_spec, background, xmin, xmax, perror = fit.fit_spectrum(fitp, this_spectrum, used_chan, calib, 
                                     first = first, matrix = True, maxiter = maxiter)


                    fitp.g.no_iters = 4

                    if self.maps_conf.use_fit == 2:
                        this_w_uname = "DO_FIT_ALL_W_TAILS" 
                        fitp, avg_fitp, spectra = self.do_fits(this_w_uname, fitp, dofit_spec, spectra, per_pix = 1, generate_img = 1, maxiter = maxiter, suffix = suffix, info_elements = info_elements)            
                        #maps_DO_fits, Event, this_w_uname, fitp, per_pix = 1, generate_img = 1, maxiter = maxiter, suffix = suffix
                    else:
                        this_w_uname = "DO_MATRIX_FIT"   
                        fitp, avg_fitp, spectra = self.do_fits(this_w_uname, fitp, dofit_spec, spectra, per_pix = 1, generate_img = 1, maxiter = maxiter, suffix = suffix, info_elements = info_elements)                              
                        #maps_DO_fits, Event, this_w_uname, fitp, per_pix = 1, generate_img = 1, maxiter = maxiter, suffix = suffix
                        

                    if thisnbs == 33 : 
                        self.maps_conf.nbs33.calibration.offset[0] = fitp.s.val[0]
                        self.maps_conf.nbs33.calibration.slope[0] = fitp.s.val[1]
                        self.maps_conf.nbs33.calibration.quad[0] = fitp.s.val[2]
            
                    if thisnbs == 32 : 
                        self.maps_conf.nbs32.calibration.offset[0] = fitp.s.val[0]
                        self.maps_conf.nbs32.calibration.slope[0] = fitp.s.val[1]
                        self.maps_conf.nbs32.calibration.quad[0] = fitp.s.val[2]
            
                    if n == 0:
                        names = fitp.s.name
                        values = np.zeros((fitp.g.n_fitp, no_mca_detectors))
          
                    if nbs.live_time[n] != 0. :
                        values[:, n] = fitp.s.val/nbs.live_time[n]
                    else:
                        values[:, n] = 0.
           
        
                if len(values.shape) == 2 :
                    values = np.sum(values, axis = 1)
                for jj in range(len(e_list)): 
                    weight_ugr_cm = 0.
                    weight_ugr_cm = self.lookup_nbs_standard_weight(e_list[jj], old_ratio = old_ratio)
        
                    e_cal_factor[jj, 0] = (weight_ugr_cm*nbs.current)
                    e_cal_factor[jj, 1] = (weight_ugr_cm*nbs.us_ic)
                    e_cal_factor[jj, 2] = (weight_ugr_cm*nbs.ds_ic)
                   
                    if e_list[jj] not in chan_names: continue
                    wo = chan_names.index(e_list[jj])


                    wo_a = np.where(fitp.s.name == e_list[jj])
                    if len(wo_a[0]) == 0 :
                        continue
                    wo_a=wo_a[0]
                    
                    counts = values[wo_a]
                    self.maps_conf.e_cal[wo, 1, 0] = e_cal_factor[jj, 0]/counts
                    self.maps_conf.e_cal[wo, 1, 1] = e_cal_factor[jj, 1]/counts
                    self.maps_conf.e_cal[wo, 1, 2] = e_cal_factor[jj, 2]/counts
                    self.maps_conf.e_cal[wo, 1, 3] = counts
                    self.maps_conf.e_cal[wo, 1, 4] = np.sqrt(counts)/counts # error fraction
                    if self.maps_conf.version >= 8 :
                        if thisnbs == 32 : self.maps_conf.e_cal[wo, 1, 5] = counts
                        if thisnbs == 33 : self.maps_conf.e_cal[wo, 1, 6] = counts
                    
                  
            else:           #IF keyword_set(DO_fit) THEN BEGIN  
                # define fitp, as this also defines add_pars, which are needed below
                fp = maps_fit_parameters.maps_fit_parameters()
                fitp = fp.define_fitp(self.main['beamline'], info_elements) 
                                     
            # The result of the fit is
            # nomalized by live time, below, the spectral data is normalized,
            # before ROIs are applied. 
            for n in range(no_mca_detectors):
                if nbs.live_time[n] != 0. :
                    data[:, n] = data[:, n]/nbs.live_time[n] 
                else:
                    data[:, n] = 0.
             
             
            # did above for fitted quantification, below for roi based quantification
            for jj in range(len(e_list)): 
                weight_ugr_cm = 0.
                weight_ugr_cm = self.lookup_nbs_standard_weight(e_list[jj], old_ratio = old_ratio)
                #print jj, e_list[jj]
        
                e_cal_factor[jj, 0] = (weight_ugr_cm*nbs.current)
                e_cal_factor[jj, 1] = (weight_ugr_cm*nbs.us_ic)
                e_cal_factor[jj, 2] = (weight_ugr_cm*nbs.ds_ic)
                
                if e_list[jj] not in chan_names: continue
                wo = chan_names.index(e_list[jj])

                counts = 0.
                for kk in range(no_mca_detectors):
                    # note: center position for peaks/rois is in keV, widths of ROIs
                    # is in eV
                    left_roi = int(((self.maps_conf.chan[wo].center-self.maps_conf.chan[wo].width/2./1000.) - nbs.calibration.offset[kk])/nbs.calibration.slope[kk])
                    if left_roi < 0 : left_roi = 0
                    right_roi = int(((self.maps_conf.chan[wo].center+self.maps_conf.chan[wo].width/2./1000.) - nbs.calibration.offset[kk])/nbs.calibration.slope[kk])
                    if right_roi >= data[:, kk].size : right_roi = data[:, kk].size-1 
        
                    roi_width = right_roi-left_roi+1
                    counts_temp = np.sum(data[left_roi:right_roi+1, kk])
                    counts = counts+counts_temp
                    #print 'left_roi, right_roi', kk, left_roi, right_roi
                    

                #print self.maps_conf.chan[wo].name, 'counts = ', counts
                self.maps_conf.e_cal[wo, 0, 0] = e_cal_factor[jj, 0]/counts
                self.maps_conf.e_cal[wo, 0, 1] = e_cal_factor[jj, 1]/counts
                self.maps_conf.e_cal[wo, 0, 2] = e_cal_factor[jj, 2]/counts
                self.maps_conf.e_cal[wo, 0, 3] = counts
                self.maps_conf.e_cal[wo, 0, 4] = np.sqrt(counts)/counts # error fraction
                #print self.maps_conf.e_cal[wo, 0, 0], self.maps_conf.e_cal[wo, 0, 1], self.maps_conf.e_cal[wo, 0, 2], self.maps_conf.e_cal[wo, 0, 3], self.maps_conf.e_cal[wo, 0, 4]
                
                if self.maps_conf.version >= 8 :
                    if thisnbs == 32 : self.maps_conf.e_cal[wo, 0, 5] = counts
                    if thisnbs == 33 : self.maps_conf.e_cal[wo, 0, 6] = counts
                   
                   

            if nnls > 0 : 
                n_pars = fitp.g.n_fitp
                parinfo_value = np.zeros((n_pars))
                parinfo_fixed = np.zeros((n_pars), dtype = np.int) 
                parinfo_limited = np.zeros((n_pars, 2), dtype = np.int)
                parinfo_limits = np.zeros((n_pars, 2)) 
                parinfo_relstep = np.zeros((n_pars)) 
                parinfo_mpmaxstep = np.zeros((n_pars)) 
                parinfo_mpminstep = np.zeros((n_pars))
                
             
                for i in range(n_pars) :
                    parinfo_value[i] = float(fitp.s.val[i])
                    wo = np.where(fitp.keywords.peaks == i)
                    if wo[0].size > 0 :
                        parinfo_value[i] = np.log10(fitp.s.val[i])
                    if fitp.s.use[i] == 1:
                        parinfo_fixed[i] = 1             
                     
#                 which_parameters_to_fit = np.where(fitp.s.use != 1)
#                 print 'parameters to fit:'
#                 print fitp.s.name[which_parameters_to_fit]
        
                x = np.arange(float(self.main['max_spec_channels']))     
                add_matrixfit_pars = np.zeros((6))
                add_matrixfit_pars[0] = fitp.s.val[fitp.keywords.energy_pos[0]]
                add_matrixfit_pars[1] = fitp.s.val[fitp.keywords.energy_pos[1]]
                add_matrixfit_pars[2] = fitp.s.val[fitp.keywords.energy_pos[2]]
                add_matrixfit_pars[3] = fitp.s.val[fitp.keywords.added_params[1]]
                add_matrixfit_pars[4] = fitp.s.val[fitp.keywords.added_params[2]]
                add_matrixfit_pars[5] = fitp.s.val[fitp.keywords.added_params[3]]           
             
             
                parinfo_prime_val = parinfo_value[np.amin(fitp.keywords.kele_pos):np.amax(fitp.keywords.mele_pos)+1]
                parinfo_prime_val = np.concatenate((parinfo_prime_val, [parinfo_value[fitp.keywords.coherent_pos[1]], parinfo_value[fitp.keywords.compton_pos[2]]], 
                                                    parinfo_value[fitp.keywords.added_params[4:13]], parinfo_value[fitp.keywords.added_params[1:4]]), axis=0)
                parinfo_prime_fixed = parinfo_fixed[np.amin(fitp.keywords.kele_pos):np.amax(fitp.keywords.mele_pos)+1]
                parinfo_prime_fixed = np.concatenate((parinfo_prime_fixed, [parinfo_fixed[fitp.keywords.coherent_pos[1]], parinfo_fixed[fitp.keywords.compton_pos[2]]], 
                                                      parinfo_fixed[fitp.keywords.added_params[4:13]], parinfo_fixed[fitp.keywords.added_params[1:4]]), axis=0)
        
                parinfo_prime_limited = parinfo_limited[np.amin(fitp.keywords.kele_pos):np.amax(fitp.keywords.mele_pos)+1,:]
                parinfo_prime_limited = np.concatenate((parinfo_prime_limited, [parinfo_limited[fitp.keywords.coherent_pos[1],:], parinfo_limited[fitp.keywords.compton_pos[2],:]], 
                                                      parinfo_limited[fitp.keywords.added_params[4:13],:], parinfo_limited[fitp.keywords.added_params[1:4],:]), axis=0)
                
                parinfo_prime_limits = parinfo_limits[np.amin(fitp.keywords.kele_pos):np.amax(fitp.keywords.mele_pos)+1,:]
                parinfo_prime_limits = np.concatenate((parinfo_prime_limits, [parinfo_limits[fitp.keywords.coherent_pos[1],:], parinfo_limits[fitp.keywords.compton_pos[2],:]], 
                                                      parinfo_limits[fitp.keywords.added_params[4:13],:], parinfo_limits[fitp.keywords.added_params[1:4],:]), axis=0) 
                
                      
                fitp.keywords.use_this_par[:] = 0
                fitp.keywords.use_this_par[np.where(parinfo_prime_fixed != 1)] = 1
                # force the last three to be 0, to make sure they do NOT get fitted as peaks.
                fitp.keywords.use_this_par[parinfo_prime_val.size-3:parinfo_prime_val.size] = 0
                
#                 temp_fitp_use = fitp.s.use[np.amin(fitp.keywords.kele_pos):np.amax(fitp.keywords.mele_pos)+1]
#                 temp_fitp_name = fitp.s.name[np.amin(fitp.keywords.kele_pos):np.amax(fitp.keywords.mele_pos)+1]
#                 which_elements_to_fit = (np.nonzero(temp_fitp_use != 1))[0]
#                 print 'elements to fit:'
#                 print temp_fitp_name[which_elements_to_fit]    
                                 
                fit = maps_analyze.analyze()
            
                fitmatrix = fit.generate_fitmatrix(fitp, x, parinfo_value)            


                wo_use_this_par = (np.nonzero(fitp.keywords.use_this_par[0:(np.max(fitp.keywords.mele_pos)-np.min(fitp.keywords.kele_pos)+1)] == 1))[0]
            
                no_use_pars = wo_use_this_par.size+2
                fitmatrix_reduced = np.zeros((x.size, no_use_pars))



                for mm in range(wo_use_this_par.size):         
                    fitmatrix_reduced[:,mm] = fitmatrix[:,wo_use_this_par[mm]]
                mm = wo_use_this_par.size - 1
                fitmatrix_reduced[:,mm] = fitmatrix[:,np.max(fitp.keywords.mele_pos)-np.min(fitp.keywords.kele_pos)+1] # elastic scatter
                mm = mm+1
                fitmatrix_reduced[:,mm] = fitmatrix[:,np.max(fitp.keywords.mele_pos)-np.min(fitp.keywords.kele_pos)+2]  # inelastic scatter        
        
         
                input_data = data[:].copy()
                
                result = maps_tools.maps_nnls_single(np.zeros((input_data.size)), input_data, fitmatrix_reduced, input_data.size)
                
             
                wo_use_this_par = np.where(fitp.keywords.use_this_par[0:max(fitp.keywords.mele_pos)-min(fitp.keywords.kele_pos)] == 1)[0]
        
                for jj in range(len(e_list)) :  
                    if e_list[jj] not in chan_names : continue 
                    wo = chan_names.index(e_list[jj])
                    if wo not in wo_use_this_par: continue
                    wo_b = np.where(wo_use_this_par == wo)[0]      
                    counts = result[wo_b]
                    print 'nnls', e_list[jj], ' ', chan_names[wo], counts
                    self.maps_conf.e_cal[wo, 2, 0] = e_cal_factor[jj, 0]/counts
                    self.maps_conf.e_cal[wo, 2, 1] = e_cal_factor[jj, 1]/counts
                    self.maps_conf.e_cal[wo, 2, 2] = e_cal_factor[jj, 2]/counts
                    self.maps_conf.e_cal[wo, 2, 3] = counts
                    self.maps_conf.e_cal[wo, 2, 4] = np.sqrt(counts)/counts # error fraction
                   
                    if self.maps_conf.version >= 8 : 
                        if thisnbs == 32 : self.maps_conf.e_cal[wo, 2, 5] = counts
                        if thisnbs == 33 : self.maps_conf.e_cal[wo, 2, 6] = counts


            
            #Look for override files in main.master_dir
            if (total_number_detectors > 1) : 
                overide_files_found = 0 
                suffix = str(this_detector) 
                maps_overridefile = os.path.join(self.main['master_dir'],'maps_fit_parameters_override.txt')+suffix
                try:
                    f = open(maps_overridefile, 'rt')    
                    print maps_overridefile, ' exists.'
                    f.close()
                except :
                    # if i cannot find an override file specific per detector, assuming
                    # there is a single overall file.
                    maps_overridefile = os.path.join(self.main['master_dir'],'maps_fit_parameters_override.txt')  
            else:
                maps_overridefile = os.path.join(self.main['master_dir'],'maps_fit_parameters_override.txt')  
  

        
            # below for ROI+ quantification
            if nnls == 0: 
                self.maps_conf.e_cal[:, 2, :] = self.maps_conf.e_cal[:, 0, :].copy()


        
        # read parameters from override file that pertain to the calibration curve
        
        #Look for override files in main.master_dir
        if (total_number_detectors > 1) : 
            overide_files_found = 0 
            suffix = str(this_detector) 
            maps_overridefile = os.path.join(self.main['master_dir'],'maps_fit_parameters_override.txt')+suffix
            try:
                f = open(maps_overridefile, 'rt')    
                print maps_overridefile, ' exists.'
                f.close()
            except :
                # if i cannot find an override file specific per detector, assuming
                # there is a single overall file.
                maps_overridefile = os.path.join(self.main['master_dir'],'maps_fit_parameters_override.txt')  
        else:
            maps_overridefile = os.path.join(self.main['master_dir'],'maps_fit_parameters_override.txt')  
       
        
        try:
            f = open(maps_overridefile, 'rt')    
            print 'this override filename: ', maps_overridefile, ' exists.'
        except :
            print  'Warning: did not find the following file: ', maps_overridefile, '   Please make sure the file is present in the parent directory, and try again. For now, I am aborting this action.'
            return 
                        

        for line in f:
            if ':' in line : 
                slist = line.split(':')
                tag = slist[0]
                value = ''.join(slist[1:])
                
                if tag == 'DETECTOR_MATERIAL' : 
                    temp = int(value)
                    # if eq to 1, it is a silicon based detector
                    if temp == 1 : 
                        self.maps_conf.add_long['a'] = 1
                        self.maps_conf.fit_t_ge = 0.

                elif tag == 'COHERENT_SCT_ENERGY' : 
                    self.maps_conf.incident_E = float(value)

                elif tag == 'BE_WINDOW_THICKNESS' : 
                    self.maps_conf.fit_t_be = float(value)*1000.
                 
                elif tag == 'GE_DEAD_LAYER' : 
                    if self.maps_conf.add_long['a'] != 1 : self.maps_conf.fit_t_ge = float(value)*1000.
                
                elif tag == 'DET_CHIP_THICKNESS' : 
                    self.maps_conf.add_float['a'] = float(value)*1000.
                
                elif tag == 'BRANCHING_RATIO_ADJUSTMENT_K' : 
                    temp_string = value.split(' ,')
                    for ts in temp_string: ts.strip()
                    
                    if temp_string[0] in fitp.s.name:
                        wo = fitp.s.name.index(temp_string[0])
                        ii = wo - np.amin(fitp.keywords.kele_pos) 
                        if len(temp_string) == 5 :
                            if temp_string[0] in info_elements.name:
                                j = info_elements.name.index(temp_string[0])
                                # adjust branching ratios within families, but all relative to Ka1
                         
                                fitp.add_pars[ii, 0].ratio = 1.
                                fitp.add_pars[ii, 1].ratio = info_elements[j].xrf_abs_yield.ka2/info_elements[j].xrf_abs_yield.ka1
                                fitp.add_pars[ii, 2].ratio = info_elements[j].xrf_abs_yield.kb1/info_elements[j].xrf_abs_yield.ka1
                                fitp.add_pars[ii, 3].ratio = info_elements[j].xrf_abs_yield.kb2/info_elements[j].xrf_abs_yield.ka1
            
                                if (len(temp_string) >= 5) :
                                    for jj in range(4):
                                        fitp.add_pars[ii, jj].ratio = fitp.add_pars[ii, jj].ratio*float(temp_string[(jj+1)])

        f.close()
        
        
        self.maps_conf.e_cal[np.where(np.isfinite(self.maps_conf.e_cal) == False)] = 0.
        

        use_henke = 1

        beta_delta_arr = np.zeros((self.maps_conf.n_chan, 2, 4))
        #*,*,0: photoelectric abosption
        #*,*,1: be 
        #*,*,2: ge 
        #*,*,2: Si
        energy_yield_arr = np.zeros((self.maps_conf.n_chan, 3))
        #*,*,0: energy
        #*,*,1: lambda
        #*,*,2: yield
      
  
        for mm in range(self.maps_conf.n_chan) :        
            jump_factor = 0.
            total_jump_factor = 0.
            element_temp = -1
             
            # if not to be used for calibration, then skip
            if self.maps_conf.chan[mm].calib == 0 : continue
            if self.maps_conf.chan[mm].calib == 1 :
                ienames = []
                for ie in info_elements: ienames.append(ie.name)
                if self.maps_conf.chan[mm].name not in ienames: continue
                element_temp = ienames.index(self.maps_conf.chan[mm].name)
                ev = info_elements[element_temp].xrf['ka1']*1000.
                yieldd = info_elements[element_temp].xrf_abs_yield['ka1']
                rel_yield = info_elements[element_temp].xrf_abs_yield['ka1']/ \
                            np.sum(info_elements[element_temp].xrf_abs_yield['ka1']+info_elements[element_temp].xrf_abs_yield['ka2']+\
                                  info_elements[element_temp].xrf_abs_yield['kb1']+info_elements[element_temp].xrf_abs_yield['kb2'])
                newrel_yield = fitp.add_pars[mm, 0].ratio/ \
                               np.sum(fitp.add_pars[mm, 0].ratio+fitp.add_pars[mm, 1].ratio+fitp.add_pars[mm, 2].ratio+fitp.add_pars[mm, 3].ratio)
        
                yieldd = newrel_yield * info_elements[element_temp].yieldD['k']
        
                name =  self.maps_conf.chan[mm].name
                k_yield = info_elements[element_temp].yieldD['k']         
                #print name, ' old_yield', yieldd, ' rel_yield', rel_yield, ' k_yield', k_yield, ' rel_yield * k_yield:', rel_yield* k_yield, ' newrel_yield', newrel_yield,  \
                #         '[mm, 0] ', fitp.add_pars[mm, 0].ratio, '[mm, 1] ', fitp.add_pars[mm, 1].ratio, '[mm, 2] ', fitp.add_pars[mm, 2].ratio, '[mm, 3] ', fitp.add_pars[mm, 3].ratio, ' old_rel_yield/new_rel_yield:', rel_yield/newrel_yield
                if self.maps_conf.incident_E > info_elements[element_temp].bindingE['K']: 
                    jump_factor = info_elements[element_temp].jump['K'] 

                yieldd = k_yield         #*rel_yield

            
            if self.maps_conf.chan[mm].calib == 2 :
                ienames = []
                for ie in info_elements: ienames.append(ie.name+'_L')
                if self.maps_conf.chan[mm].name not in ienames: continue
                element_temp = ienames.index(self.maps_conf.chan[mm].name)
                ev = info_elements[element_temp].xrf['la1']*1000.
                yieldd = info_elements[element_temp].xrf_abs_yield['la1']
                name =  self.maps_conf.chan[mm].name[:-2]
                jump_factor = info_elements[element_temp].jump['L3'] 
                if self.maps_conf.incident_E > info_elements[element_temp].bindingE['L2'] : 
                    total_jump_factor = info_elements[element_temp].jump['L2']
                if self.maps_conf.incident_E > info_elements[element_temp].bindingE['L1'] : 
                    total_jump_factor = total_jump_factor*info_elements[element_temp].jump['L1']
                #print  name, ' L jump_factor: ', jump_factor
          

            if self.maps_conf.chan[mm].calib == 3 :
                ienames = []
                for ie in info_elements: ienames.append(ie.name+'_M')
                if self.maps_conf.chan[mm].name not in ienames: continue
                element_temp = ienames.index(self.maps_conf.chan[mm].name)
                ev = self.maps_conf.chan[mm].center*1000.
                yieldd = info_elements[element_temp].xrf_abs_yield['ma1']
                name = self.maps_conf.chan[mm].name[:-2]
                jump_factor = info_elements[element_temp].jump['M5'] 
                total_jump_factor = info_elements[element_temp].jump['M1']*info_elements[element_temp].jump['M2']*\
                                    info_elements[element_temp].jump['M3']*info_elements[element_temp].jump['M4']
                #print  name, ' M jump_factor: ', jump_factor, ' yield ', yieldd, ' ev: ', ev, ' total_jump_factor: ', total_jump_factor
        
        
            if element_temp == -1 : continue 
            if jump_factor != 0. :
                if total_jump_factor == 0. :
                    shell_factor = (jump_factor-1.)/jump_factor 
                else: 
                    shell_factor =  (jump_factor-1.)/jump_factor / total_jump_factor

            else:
                shell_factor = 0.
            if (name == 'U') and (self.maps_conf.chan[mm].calib == 3) :
                U_shell_factor = shell_factor
        
            # beta proportional to f2, 
             
            # make sure name is known to henke routine, if not, skip
            # in case this is Pu, use U instead
            if (name == 'Pu') or (name == 'Np') : 
                print 'name was ', name, ' resetting it to uranium to work with henke data'
                name = 'U'
                
            Chenke = henke.henke()
            
            test = []
            test = Chenke.zcompound(name, test)
            if np.sum(test) == 0 : continue
            # replace straight henke routines, with those
            # that take the absorption edges into account
            # make sure we are a bit above the absorption edge to make sure that for calibration purposes we do not eoncouner any weird things.                
            f1, f2, delta, beta = Chenke.get_henke_single(name, info_elements[element_temp].density, (self.maps_conf.incident_E+0.1)*1000.) 
            #f1, f2, delta, beta = Chenke.get_henke(name, info_elements[element_temp].density, (self.maps_conf.incident_E+0.1)*1000.) 
            # stds in microgram/cm2
            # density rho = g/cm3 = 1 microgram/cm2 /1000/1000/cm = 1 microgram/cm2 /1000/1000/*10*1000/um = 1 microgram/cm2 /100/um 
            # thickness for 1 ugr/cm2
            # =1/(density[g/cm3]/10)
            thickness = 1./(info_elements[element_temp].density * 10.)*1000.
            aux_arr[mm, 0] = self.absorption(thickness, beta, 1239.852/((self.maps_conf.incident_E+0.1)*1000.), shell_factor = shell_factor)
            f1, f2, delta, beta  = Chenke.get_henke_single('Be', 1.848, ev) 
        
            aux_arr[mm, 1] = self.transmission(self.maps_conf.fit_t_be, beta, 1239.852/ev)
            f1, f2, delta, beta  = Chenke.get_henke_single( 'Ge', 5.323, ev) 
            aux_arr[mm, 2] = self.transmission(self.maps_conf.fit_t_ge, beta, 1239.852/ev)
            aux_arr[mm, 3] = yieldd
            if self.maps_conf.add_long['a'] == 1 :
                f1, f2, delta, beta  = Chenke.get_henke_single( 'Si', 2.3, ev) 
            aux_arr[mm, 4] = self.transmission(self.maps_conf.add_float['a'], beta, 1239.852/ev)
            if (self.maps_conf.add_float['a'] == 0.) or (self.maps_conf.add_long['a'] != 1) :
                aux_arr[mm, 4] = 0.
            if airpath > 0: 
                density = 1.0
                #f1, f2, delta, beta, graze_mrad, reflect, inverse_mu, atwt = Chenke.get_henke_single('air', density, ev) 
                f1, f2, delta, beta = Chenke.get_henke_single('air', density, ev) 
                aux_arr[mm, 5] = self.transmission(airpath*1000., beta, 1239.852/ev)  # airpath is read in microns, transmission function expects nm
            else:
                aux_arr[mm, 5] = 1.


        if 'Al' in chan_names:
            used_elements = [chan_names.index('Al')]
            used_elements.append(chan_names.index('Si'))
        else:
            used_elements = [chan_names.index('Si')]
      
 
        temp_string = ['K', 'Ca', 'Ti', 'V']
        if self.maps_conf.incident_E > 6.54 : temp_string.append('Mn')
        if self.maps_conf.incident_E > 7.11 : temp_string.append('Fe')
        if self.maps_conf.incident_E > 7.71 : temp_string.append('Co')
        if self.maps_conf.incident_E > 8.98 : temp_string.append('Cu')
        if self.maps_conf.incident_E > 9.66 : temp_string.append('Zn')
        temp_string.append('Pb_M')
        if self.maps_conf.incident_E > 13.03 : temp_string.append('Pb_L')
      
        for i in range(len(temp_string)):
            if temp_string[i] in chan_names:
                wo = chan_names.index(temp_string[i])
                used_elements.append(wo)


        m_used_elements = []
        if 'Pb_M' in chan_names:
            m_used_elements.append(chan_names.index('Pb_M'))
        if 'Au_M' in chan_names:
            m_used_elements.append(chan_names.index('Au_M'))
                
        wo_m = [chan_names.index(s) for s in chan_names if "_M" in s]

        
        e_cal_shape = self.maps_conf.e_cal.shape
        temp_calib = np.zeros((e_cal_shape[0]))
        m_temp_calib = np.zeros((e_cal_shape[0]))
    
    
        self.aux_arr = aux_arr
        rrange = 3 
        for k in range(rrange): 
            for l in range(3):
                if l == 0 : factor =  nbs.current
                if l == 1 : factor =  nbs.us_ic
                if l == 2 : factor =  nbs.ds_ic
                for iel in used_elements:
                    temp_calib[iel] = self.maps_conf.e_cal[iel, k, l] 

                if len(m_used_elements) > 0 : 
                    m_temp_calib[m_used_elements] = self.maps_conf.e_cal[m_used_elements, k, l]
        
                x = used_elements[:]
                y = 1./temp_calib[used_elements]
                y = np.array(y)
                err = y/20.+np.sqrt(y)/10.
                #set the last three errors to be very large so
                #they do not impact the fiting. the last three
                #correspond to PbL, M, Zn          
                err[err.size-1] = 100000.
                if 'Pb_L' in temp_string: 
                    err[err.size-2] = 100000.
                # to remove Al and Si from fit:
                err[0] = 100000.
                err[1] = 10000.
        
                wo = np.where(np.isfinite(y) == False)
        
                if len(wo[0]) > 0 : 
                    y[wo] = 0.
                    err[wo] = 1000.

                
                n_pars = 2
                parinfo_value = np.zeros((n_pars)) 
                parinfo_fixed = np.zeros((n_pars), dtype=np.int)  
                parinfo_limited = np.zeros((n_pars, 2), dtype = np.int)
                parinfo_limits = np.zeros((n_pars, 2)) 
                parinfo_relstep = np.zeros((n_pars)) 
                parinfo_mpmaxstep = np.zeros((n_pars)) 
                parinfo_mpminstep = np.zeros((n_pars))
                            
                parinfo_value[0] = 100000.0/factor
                parinfo_value[1] = 0. # [1 micron initial thickness]     air
                if airpath >0 : parinfo_value[1] = float(airpath)
                parinfo_fixed[1] =  1
                parinfo_limited[0,0] = 1
                parinfo_limits[0,0] = 0.0

                bounds=[]
                have_bounds = 0
                for i in range(n_pars):
                    havelimit = parinfo_limited[i,:]
                    if havelimit[0] == 1:
                        bmin = parinfo_limits[i,0]
                        have_bounds = 1
                    else:
                        bmin = None
                    if havelimit[1] == 1:
                        bmax = parinfo_limits[i,1]
                        have_bounds = 1
                    else:
                        bmax = None
                    bounds.append((bmin,bmax))
    
                
                Clb = leastsqbound

                p0 = np.array(parinfo_value)
                p1,cov,infodict,mesg,self.success = Clb.leastsqbound(self.residuals, p0, bounds, args=(y, x), maxfev=maxiter, full_output = True)      
                
                #perror1 = self.calc_perror(infodict['fjac'], infodict['ipvt'], len(p1))


                temp_x = np.where(np.array(chan_calib) >= 1)[0]
                curve = self.fit_calibrationcurve(temp_x, p1)
                #print k, ' ', l, '   p:', p1[0], '   factor[I, IC] ', factor
                self.maps_conf.e_cal[:, k, l] = 0.
                for ic in range(len(self.maps_conf.chan)):
                    for iun in range(4) : self.maps_conf.chan[ic].units[iun] = 'cts/s'
                for tx in temp_x:
                    for iun in range(3) : self.maps_conf.chan[tx].units[iun] = 'ug/cm^2'          
                self.maps_conf.e_cal[temp_x, k, l] = curve
                
                
                if len(m_used_elements) > 1 :
                    # now do seperate calibration using M lines
                    x = m_used_elements
                    y = 1./m_temp_calib[m_used_elements]
                    err = y/20.+np.sqrt(y)/10.
                    wo = np.where(np.isfinite(y) == False)[0]
                    if len(wo) > 0 :
                        y[wo] = 0.
                        err[wo] = 1000.
                    n_pars = 2
                    parinfo_value = np.zeros((n_pars)) 
                    parinfo_fixed = np.zeros((n_pars), dtype=np.int)  
                    parinfo_limited = np.zeros((n_pars, 2), dtype = np.int)
                    parinfo_limits = np.zeros((n_pars, 2)) 
                    parinfo_relstep = np.zeros((n_pars)) 
                    parinfo_mpmaxstep = np.zeros((n_pars)) 
                    parinfo_mpminstep = np.zeros((n_pars))
   
                    parinfo_value[0] = 100000.0/factor
                    parinfo_value[1] = 0.0 # [1 micron initial thickness]     air
                    if airpath > 0 : parinfo_value[1] = float(airpath)
                    parinfo_fixed[1] =  1
                    parinfo_limited[0,0] = 1
                    parinfo_limits[0,0] = 0.0
                    perror = 0
                    quiet = 1
                    ERRMSG = ''
                    
                    bounds=[]
                    have_bounds = 0
                    for i in range(n_pars):
                        havelimit = parinfo_limited[i,:]
                        if havelimit[0] == 1:
                            bmin = parinfo_limits[i,0]
                            have_bounds = 1
                        else:
                            bmin = None
                        if havelimit[1] == 1:
                            bmax = parinfo_limits[i,1]
                            have_bounds = 1
                        else:
                            bmax = None
                        bounds.append((bmin,bmax))
                    

                    p0 = np.array(parinfo_value)
                    p1,cov,infodict,mesg,self.success = Clb.leastsqbound(self.residuals, p0, bounds, args=(y, x), maxfev=maxiter, full_output = True)      

                    curve = self.fit_calibrationcurve(wo_m, p1)
                    for ie in wo_m:
                        self.maps_conf.chan[ie].units[0:3] = 'ug/cm^2'          
                    self.maps_conf.e_cal[wo_m, k, l] = curve
             
        self.maps_conf.calibration.slope = (self.maps_conf.nbs32.calibration.slope+self.maps_conf.nbs33.calibration.slope)/2.
        self.maps_conf.calibration.offset = (self.maps_conf.nbs32.calibration.offset+self.maps_conf.nbs33.calibration.offset)/2.
        self.maps_conf.calibration.quad = (self.maps_conf.nbs32.calibration.quad+self.maps_conf.nbs33.calibration.quad)/2.

        self.nbs_calibration_write_info( e_cal_factor, old_ratio = old_ratio, suffix = suffix, aux_arr = aux_arr, info_elements = info_elements)
           

        
        return


                   
#-----------------------------------------------------------------------------   
    def read_nbsstds(self, filename):
        

        overide_files_found = 0 
        maps_overridefile = os.path.join(self.main['master_dir'],'maps_fit_parameters_override.txt')
        try:
            f = open(maps_overridefile, 'rt')    
            print maps_overridefile, ' exists.'
            f.close()
            overide_files_found = 1
        except :
            overide_files_found = 0 
         
        
        if overide_files_found == 1:
            f = open(maps_overridefile, 'rt')
            for line in f:
                if ':' in line : 
                    slist = line.split(':')
                    tag = slist[0]
                    value = ''.join(slist[1:])
                    
                    if   tag == 'AIRPATH'   : airpath = float(value)
                    elif tag == 'SRCURRENT' : srcurrent_name = str(value)
                    elif tag == 'US_IC'     : us_ic_name = str(value)
                    elif tag == 'DS_IC'     : ds_ic_name = str(value)
       
            f.close()


        try:
            f = open(filename, 'rt')    
            print filename, ' exists.'
            f.close()
            nbs_files_found = 1
        except :
            nbs_files_found = 0 

                
        if nbs_files_found == 0: 
            print 'Did not find NBS calibration file, returning'
            return
        if nbs_files_found == 1:
            filepath = os.path.join(self.main['master_dir'],filename)

            calibration, data, date, live_time, real_time, current, IC_US, IC_DS, us_amp, ds_amp = self.read_calibration(filepath, 
                                                                                                                         srcurrent_name = srcurrent_name, 
                                                                                                                         us_ic_name = us_ic_name, 
                                                                                                                         ds_ic_name = ds_ic_name)
            if self.maps_conf.use_det.sum() > 0:
                if current == 0:
                    warning_msg = 'Could not find synchrotron current in the NBS standard. Will proceed assuming a SRcurrent of 100 mA'
                    print warning_msg
                    current = 100.                    
                        
                nbs = self.maps_conf.nbs32
                nbs.name = filename
                nbs.calibration.offset = calibration['offset']
                nbs.calibration.slope = calibration['slope']
                nbs.calibration.quad = calibration['quad']
                nbs.date = date
                nbs.live_time = 0
                nbs.real_time = real_time       
                nbs.current = current
                nbs.us_ic = IC_US
                nbs.ds_ic = IC_DS
                nbs.us_amp = us_amp
                nbs.ds_amp = ds_amp
                
                lt_shape = live_time.shape[0]
                
                if lt_shape < self.maps_conf.use_det.sum():
                    print 'warning: number of selected detectors does NOT match number OF detectors found in the mca file'


                wo = np.where(self.maps_conf.use_det == 1)[0]
                if len(wo) == 0 : return 
                elif len(wo) == 1:
                    nbs.live_time = live_time
                    nbs.real_time = real_time                    
                else:
                    for ii in range(len(wo)):
                        nbs.live_time[wo[ii]] = live_time[wo[ii]]
                        nbs.real_time[wo[ii]] = real_time[wo[ii]]

        
        return nbs



#-----------------------------------------------------------------------------   
    def nbs_calibration_write_info(self, 
                                   e_cal_factor, 
                                   old_ratio = 0, 
                                   suffix = '',
                                   aux_arr = 0,
                                   info_elements = 0):
        
        print 'Writing nbs info'
        
        make_maps_conf = self.maps_conf
        
        chan_names = []
        chan_calib = []
        for i in range(len(make_maps_conf.chan)):
            chan_names.append(make_maps_conf.chan[i].name)
            chan_calib.append(make_maps_conf.chan[i].calib)

        dir = self.main['output_dir']
        if not os.path.exists(dir):   
            os.makedirs(dir)
            if not os.path.exists(dir):
                print 'warning: did not find the output directory, and could not create a new output directory. Will abort this action'
                return 0
        
 
        # determine no of maximal supported detectors    n_max
        n_max = len(make_maps_conf.nbs32.real_time)
    
        filename = os.path.join(self.main['output_dir'],'calibration_info')+suffix+'.csv'
        found = 0

        try:
            bufsize = 1
            f = open(filename, 'w', bufsize)    
        except :
            print 'Could not open nbs_info_file:', filename
            return 0
        

        # make_maps_conf
        line = 'calibrated:, '+ t.strftime("%a, %d %b %Y %H:%M:%S")
        print>>f, line.strip()
        line = 'NBS1832:, '+make_maps_conf.nbs32.name
        print>>f, line
        line = str(make_maps_conf.nbs32.real_time[0]) +', '
        for ii in range(n_max) :
            line = line + str(make_maps_conf.nbs32.real_time[ii])+', '
        print>>f, 'realtime[s]:,  '+ line.strip()
        line = str(make_maps_conf.nbs32.live_time[0])+', '
        for ii in range(n_max) :
            line = line + str(make_maps_conf.nbs32.live_time[ii])+', '
        print>>f, 'livetime[s]:,  '+ line.strip()
        line = str(make_maps_conf.nbs32.current)
        print>>f, 'I_[mA]:,  '+ line.strip()
        line = str(make_maps_conf.nbs32.us_ic)
        print>>f, 'US_IC[cts/s]:,  '+ line.strip()
        line = str(make_maps_conf.nbs32.ds_ic)
        print>>f, 'DS_IC[cts/s]:,  '+ line.strip()
        line = str(make_maps_conf.nbs32.us_amp[0])+', '
        for ii in range(2):
            line = line + str(make_maps_conf.nbs32.us_amp[ii])+', '
        print>>f, 'US_AMP[sensitivity/units/factor]:,  '+ line.strip()
        line = str(make_maps_conf.nbs32.ds_amp[0])+', '
        for ii in range(2):
            line = line + str(make_maps_conf.nbs32.ds_amp[ii])+', '
        print>>f,  'DS_AMP[sensitivity/units/factor]:,  '+ line.strip()
        line =  ' '
        print>>f,  line 
    
        line = 'NBS1833:, '+make_maps_conf.nbs33.name
        print>>f, line 
        line = str(make_maps_conf.nbs33.real_time[0]) +', '
        for ii in range(n_max) :
            line = line + str(make_maps_conf.nbs33.real_time[ii])+', '
        print>>f,  'realtime[s]:,  '+ line.strip()
        line = str(make_maps_conf.nbs33.live_time[0])+', '
        for ii in range(n_max) :
            line = line + str(make_maps_conf.nbs33.live_time[ii])+', '
        print>>f,  'livetime[s]:,  '+ line.strip()
        line = str(make_maps_conf.nbs33.current)
        print>>f,  'I_[mA]:,  '+ line.strip()
        line = str(make_maps_conf.nbs33.us_ic)
        print>>f,  'US_IC[cts/s]:,  '+ line.strip()
        line = str(make_maps_conf.nbs33.ds_ic)
        print>>f,  'DS_IC[cts/s]:,  '+ line.strip()
        line = str(make_maps_conf.nbs33.us_amp[0])+', '
        for ii in range(2) :
            line = line + str(make_maps_conf.nbs33.us_amp[ii])+', '
        print>>f,  'US_AMP[sensitivity/units/factor]:,  '+ line.strip()
        line = str(make_maps_conf.nbs33.ds_amp[0])+', '
        for ii in range(2) :
            line = line + str(make_maps_conf.nbs33.ds_amp[ii])+', '
        print>>f,  'DS_AMP[sensitivity/units/factor]:,  '+ line.strip()
        line =  ' '
        print>>f,  line 
    
        line = 'name '+', '
        line = line+'z'+', ' 
        line = line+'units[DS_IC]'+', ' 
        line = line+'units[US_IC]'+', ' 
        line = line+'units[SRCurrent]'+', ' 
        line = line+'units[1]'+', ' 
        line = line+'use'+', ' 
        line = line+'calib'+', ' 
        line = line+'center'+', ' 
        line = line+'shift'+', ' 
        line = line+'width'+', ' 
        line = line+'bkground_left'+', ' 
        line = line+'bkground_right'+', ' 
        line = line+'left_roi[0]'+', ' 
        line = line+'right_roi[0]'+', ' 
        line = line+'left_bkground[0]'+', ' 
        line = line+'right_bkground[0]'+', ' 
        line = line+'absorption_1um_element'+', ' 
        line = line+'Be_window_transmission'+', ' 
        line = line+'Ge_dead_layer_transmission'+', ' 
        line = line+'XRF_yield'+', ' 
        line = line+'fraction_photons_absorbed_in_det'+', ' 
        line = line+'air_absoprtion'+', ' 
        line = line+'total_XRF_efficiency_factor'+', ' 
           
        print>>f,  line.strip()
        
    
        if np.sum(aux_arr) != 0:
            max_eff = aux_arr[:, 0]*aux_arr[:, 1]*aux_arr[:, 2]*aux_arr[:, 3]*(1.-aux_arr[:, 4])*aux_arr[:, 5]
            max_eff = np.amax(max_eff)
        for ii in range(make_maps_conf.n_chan):
            line = make_maps_conf.chan[ii].name +', '
            line = line+str(make_maps_conf.chan[ii].z)+', ' 
            line = line+make_maps_conf.chan[ii].units[0]+', ' 
            line = line+make_maps_conf.chan[ii].units[1]+', ' 
            line = line+make_maps_conf.chan[ii].units[2]+', ' 
            line = line+make_maps_conf.chan[ii].units[3]+', ' 
            line = line+str(make_maps_conf.chan[ii].use)+', ' 
            line = line+str(make_maps_conf.chan[ii].calib)+', ' 
            line = line+str(make_maps_conf.chan[ii].center)+', ' 
            line = line+str(make_maps_conf.chan[ii].shift)+', ' 
            line = line+str(make_maps_conf.chan[ii].width)+', ' 
            line = line+str(make_maps_conf.chan[ii].bkground_left)+', ' 
            line = line+str(make_maps_conf.chan[ii].bkground_right)+', ' 
            line = line+str(make_maps_conf.chan[ii].left_roi[0])+', ' 
            line = line+str(make_maps_conf.chan[ii].right_roi[0])+', ' 
            line = line+str(make_maps_conf.chan[ii].left_bkground[0])+', ' 
            line = line+str(make_maps_conf.chan[ii].right_bkground[0])+', ' 
            if np.sum(aux_arr) != 0 :
                line = line+str(aux_arr[ii, 0])+', ' 
                line = line+str(aux_arr[ii, 1])+', ' 
                line = line+str(aux_arr[ii, 2])+', ' 
                line = line+str(aux_arr[ii, 3])+', ' 
                line = line+str(1-aux_arr[ii, 4])+', ' 
                line = line+str(aux_arr[ii, 5])+', ' 
                line = line+str( aux_arr[ii, 0]*aux_arr[ii, 1]*aux_arr[ii, 2]*aux_arr[ii, 3]*(1.-aux_arr[ii, 4])*aux_arr[ii, 5] / max_eff)+', ' 
              
            print>>f, line.strip()


            
        line = ' '
        print>>f, line.strip()       
        line = 'calibration curve for ROIs'
        print>>f,  line.strip()       
        line = 'desc, name, Z, for_SRCurrent_norm, for_US_IC_norm, for_DS_IC_norm[XRF_counts_per_s/(weight_ugr_cm*nbs.current)], measured_counts, error_bar'
        if make_maps_conf.version >= 8 : line = line+', measured_nb1832, measured_nb1833'
        print>>f,  line.strip()       
        for ii in range(make_maps_conf.n_chan): 
            line = 'calib_curve_roi,' + make_maps_conf.chan[ii].name +', '
            line = line+str(make_maps_conf.chan[ii].z)+', ' 
            for jj in range(5) :
                line = line+str(make_maps_conf.e_cal[ii, 0, jj])+', ' 
            if make_maps_conf.version >= 8 :
                for jj in range(5, 7):
                    line = line+str(make_maps_conf.e_cal[ii, 0, jj])+', '                     
            print>>f,  line.strip()      


    
        line = ' '
        print>>f,  line.strip()       
        line = 'calibration curve for fitted data'
        print>>f,  line.strip()       
        line = 'desc, name, Z, for_SRCurrent_norm, for_US_IC_norm, for_DS_IC_norm, measured_counts, error_bar'
        if make_maps_conf.version >= 8 : line = line+', measured_nb1832, measured_nb1833'
        print>>f,  line.strip()       
        for ii in range(make_maps_conf.n_chan):
            line = 'calib_curve_fitted,' + make_maps_conf.chan[ii].name +', '
            line = line+str(make_maps_conf.chan[ii].z)+', ' 
            for jj in range(5):
                line = line+str(make_maps_conf.e_cal[ii, 1, jj])+', ' 
            if make_maps_conf.version >= 8 :
                for jj in range(5, 7) :
                    line = line+str(make_maps_conf.e_cal[ii, 1, jj])+', '                     
            print>>f,  line.strip()          


        line = ' '
        print>>f,  line.strip()       
        line = 'calibration curve for ROI+'
        print>>f,  line.strip()      
        line = 'desc, name, Z, for_SRCurrent_norm, for_US_IC_norm, for_DS_IC_norm[XRF_counts_per_s/(weight_ugr_cm*nbs.current)], measured_counts, error_bar'
        if make_maps_conf.version >= 8 : line = line+', measured_nb1832, measured_nb1833'
        print>>f,  line.strip()    
        for ii in range(make_maps_conf.n_chan):
            line = 'calib_curve_roi,' + make_maps_conf.chan[ii].name +', '
            line = line+str(make_maps_conf.chan[ii].z)+', ' 
            for jj in range(5):
                line = line+str(make_maps_conf.e_cal[ii, 2, jj])+', ' 
            if make_maps_conf.version >= 8 :
                for jj in range(5, 7) :
                    line = line+str(make_maps_conf.e_cal[ii, 2, jj])+', '                     
            print>>f,  line.strip()        
            
        f.close()


        print "Saved calibration info file"
        
        wo = np.where(np.array(chan_calib) == 1)[0]
        if len(wo) > 0: 

            print 'ploting spectrum'
            
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            mplot.rcParams['pdf.fonttype'] = 42
            
            fontsize = 9
            mplot.rcParams['font.size'] = fontsize
            
            colortable = []
        
            colortable.append((0., 0., 0.)) # ; black
            colortable.append((1., 0., 0.)) # ; red
            colortable.append((0., 1., 0.)) # ; green
            colortable.append((0., 0., 1.)) # ; blue
            colortable.append((0., 1., 1.)) # ; turquois
            colortable.append((1., 0., 1.)) # ; magenta
            colortable.append((1., 1., 0.)) # ; yellow
            colortable.append((0.7, 0.7, 0.7)) # ; light grey
            colortable.append((1., 0.8, 0.75)) # ; flesh
            colortable.append(( 0.35,  0.35,  0.35)) # ; dark grey       
            colortable.append((0., 0.5, 0.5)) # ; sea green
            colortable.append((1., 0., 0.53)) # ; pink-red
            colortable.append((0., 1., 0.68)) # ; bluegreen 
            colortable.append((1., 0.5, 0.)) # ; orange
            colortable.append((0., 0.68, 1.)) # ; another blue
            colortable.append((0.5, 0., 1.)) # ; violet
            colortable.append((1., 1., 1.)) # ; white
            
            foreground_color = colortable[-1]
            background_color = colortable[0]
                
            dpi = 100
            canvas_xsize_in = 1000./dpi
            canvas_ysize_in = 700./dpi
            
            bindingEK = []
            for ib in range(len(info_elements)): bindingEK.append(info_elements[ib].bindingE['K'])
            bindingEK = np.array(bindingEK)
                        
            for k in range(3) :
                for l in range(3) : 
                    
                    fig = mplot.figure.Figure(figsize=(canvas_xsize_in, canvas_ysize_in), dpi=dpi, facecolor=background_color, edgecolor=None)
                    canvas = FigureCanvas(fig)
                    fig.add_axes()
                    axes = fig.gca()
                    for child in axes.get_children():
                        if isinstance(child, mplot.spines.Spine):
                            child.set_color(foreground_color)
                    axes.set_axis_bgcolor(background_color)
                    ya = axes.yaxis                  
                    xa = axes.xaxis                          
                    ya.set_tick_params(labelcolor=foreground_color) 
                    ya.set_tick_params(color=foreground_color) 
                    xa.set_tick_params(labelcolor=foreground_color) 
                    xa.set_tick_params(color=foreground_color)     
    
                    title = 'NBS calibration curve '
                    if l == 0 : title = title +'for ROI sum'
                    if l == 1 : title = title +'for fitted data'
                    if l == 2 : title = title +'for ROIs+'
                    if k == 0 : title = title +' with normalization by Synchrotron current'
                    if k == 1 : title = title +'with normalization by upstream ionchamber'
                    if k == 2 : title = title +'with normalization by downstream ionchamber'
                    if k == 0 : ytitle = 'XRF_counts_per_s/(weight_ugr_cm*nbs.current)'
                    if k == 1 : ytitle = 'XRF_counts_per_s/(weight_ugr_cm*US_IC)'
                    if k == 2 : ytitle = 'XRF_counts_per_s(weight_ugr_cm*DS_IC)'
                    y = make_maps_conf.e_cal[wo, l, k]    
                    measured = y.copy()
                    measured[:] = 0.
                    measured_32 = measured.copy()
                    measured_33 = measured.copy()
                    for ii in range(len(wo)): 
                        weight_ugr_cm = 0.
                        weight_ugr_cm = self.lookup_nbs_standard_weight(make_maps_conf.chan[wo[ii]].name, old_ratio = old_ratio)
                       
                        if weight_ugr_cm == 0. : continue
                        if k ==0 : norm = (make_maps_conf.nbs32.current+make_maps_conf.nbs33.current)/2.
                        if k ==1 : norm = (make_maps_conf.nbs32.us_ic+make_maps_conf.nbs33.us_ic)/2.
                        if k ==2 : norm = (make_maps_conf.nbs32.ds_ic+make_maps_conf.nbs33.ds_ic)/2.
                        measured[ii] = 1.0/(weight_ugr_cm*norm/make_maps_conf.e_cal[wo[ii], l, 3])
                        if make_maps_conf.version > 8 : 
                            if k ==0 : norm = make_maps_conf.nbs32.current
                            if k ==1 : norm = make_maps_conf.nbs32.us_ic
                            if k ==2 : norm = make_maps_conf.nbs32.ds_ic
                            measured_32[ii] = 1.0/(weight_ugr_cm*norm/make_maps_conf.e_cal[wo[ii], l, 5])
                            if k ==0 : norm = make_maps_conf.nbs33.current
                            if k ==1 : norm = make_maps_conf.nbs33.us_ic
                            if k ==2 : norm = make_maps_conf.nbs33.ds_ic
                            measured_33[ii] = 1.0/(weight_ugr_cm*norm/make_maps_conf.e_cal[wo[ii], l, 6])
       
    
                    x = []
                    for iw in range(len(wo)): x.append(make_maps_conf.chan[iw].z)
                    x = np.array(x)
                    y = np.array(y)  
                    
                    temp1 = np.where(bindingEK <= make_maps_conf.incident_E)
                    temp = np.where(bindingEK[temp1] != 0.)[0]
                    
                    wo_2 = np.where(x <= max(temp)+1)[0]
                    # do NOT plot for Na and Mg.
                    wo_2 = np.delete(wo_2,0)
                    wo_2 = np.delete(wo_2,0)

                    x = x[wo_2]
                    y = y[wo_2]       
                    measured = measured[wo_2]       
                    measured_32 = measured_32[wo_2]       
                    measured_33 = measured_33[wo_2]  
 
                    x_p = x.copy()
                    y_p = y.copy()
                    x_p = np.insert(x_p, 0, x_p[0]-1) 
                    x_p = np.append(x_p, max(x_p)+1)
                    y_p = np.insert(y_p, 0, 0)
                    y_p = np.append(y_p, 0)
                    measured_p = measured.copy()
                    measured_p = np.insert(measured_p, 0, 0) 
                    measured_p = np.append(measured_p, 0)
                    measured_p_32 = measured_32.copy()
                    measured_p_32 = np.insert(measured_p_32, 0, 0.0) 
                    measured_p_32 = np.append(measured_p_32, 0.0)
                    measured_p_33 = measured_33.copy()
                    measured_p_33 = np.insert(measured_p_33, 0, 0.0) 
                    measured_p_33 = np.append(measured_p_33, 0.0)
                    xtickv = x_p
                    chnames = np.array(chan_names)
                    chnames = chnames[wo[wo_2]].tolist()
                    xtickname = [' '] + chnames+[' ']
                        
                    wo_3 = np.where(y != 0)
                    if len(wo_3[0]) > 0 : y_min = min(y[wo_3]) 
                    else: y_min = 0.
                    yrange = [y_min*0.8, max(y)*1.5]       
                    if y_min < 1e-10 : yrange = [1e-4, max(y)*1.5]    


                    plot1 = axes.semilogy(x_p+0.5, y_p, color = foreground_color, linewidth=1.0, linestyle='steps')  
                    #plot1a = axes.semilogy(x_p,y_p, color = 'yellow', linewidth=1.0, linestyle='None', marker = '*')  
                    plot2 = axes.semilogy(x_p, measured_p, color = 'blue', linewidth=1.0, linestyle='None', marker = 'x', markersize = 7, mew=1.2)  
                    plot3 = axes.semilogy(x_p, measured_p_32, color = 'green', linewidth=1.0, linestyle='None', marker = 'x', markersize = 7, mew=1.2)  
                    plot4 = axes.semilogy(x_p, measured_p_33, color = 'red', linewidth=1.0, linestyle='None', marker = 'x', markersize = 7, mew=1.2)   

                    axes.xaxis.set_ticks(xtickv)
                    axes.set_ylabel(ytitle, color=foreground_color)
                    
                    axes.set_xlim((x_p[0], x_p[-1]))
                    axes.set_ylim(yrange)
                    axes.set_xticklabels(xtickname)
                    axes.set_position([0.10,0.08,0.85,0.85])
                    axes.set_title(title, color=foreground_color)
                                      
                   
                    axes.text(0.60, 0.13, 'NBS 1832', color = 'green', transform = axes.transAxes) 
                    axes.text(0.60, 0.10, 'NBS 1833', color = 'red', transform = axes.transAxes) 
                    axes.text(0.60, 0.07, 'average(1832+1833)', color = 'blue', transform = axes.transAxes) 
                                         
                    axes.text(0.97, -0.08, 'mapspy', color = foreground_color, transform = axes.transAxes) 
                    image_filename = 'calib'+str(l)+'_'+str(k)+'.png'
                    print 'saving ', os.path.join(dir,image_filename)
                    fig.savefig(os.path.join(dir, image_filename), dpi=dpi, facecolor=background_color, edgecolor=None)
                   


        
        
        return    


#-----------------------------------------------------------------------------   
    def axo_calibration_write_info(self, 
                                   e_cal_factor, 
                                   old_ratio = 0, 
                                   suffix = '',
                                   aux_arr = 0,
                                   info_elements = 0):
        
        print 'Writing axo info'
        
        make_maps_conf = self.maps_conf
        
        chan_names = []
        chan_calib = []
        for i in range(len(make_maps_conf.chan)):
            chan_names.append(make_maps_conf.chan[i].name)
            chan_calib.append(make_maps_conf.chan[i].calib)

        dir = self.main['output_dir']
        if not os.path.exists(dir):   
            os.makedirs(dir)
            if not os.path.exists(dir):
                print 'warning: did not find the output directory, and could not create a new output directory. Will abort this action'
                return 0
        
 
        # determine no of maximal supported detectors    n_max
        n_max = len(make_maps_conf.axo.real_time)
    
        filename = os.path.join(self.main['output_dir'],'calibration_info_axo_')+suffix+'.csv'
        found = 0

        try:
            bufsize = 1
            f = open(filename, 'w', bufsize)    
        except :
            print 'Could not open nbs_info_file:', filename
            return 0
        

        # make_maps_conf
        line = 'calibrated:, '+ t.strftime("%a, %d %b %Y %H:%M:%S")
        print>>f, line.strip()
        line = 'NBS1832:, '+make_maps_conf.nbs32.name
        print>>f, line
        line = str(make_maps_conf.nbs32.real_time[0]) +', '
        for ii in range(n_max) :
            line = line + str(make_maps_conf.nbs32.real_time[ii])+', '
        print>>f, 'realtime[s]:,  '+ line.strip()
        line = str(make_maps_conf.nbs32.live_time[0])+', '
        for ii in range(n_max) :
            line = line + str(make_maps_conf.nbs32.live_time[ii])+', '
        print>>f, 'livetime[s]:,  '+ line.strip()
        line = str(make_maps_conf.nbs32.current)
        print>>f, 'I_[mA]:,  '+ line.strip()
        line = str(make_maps_conf.nbs32.us_ic)
        print>>f, 'US_IC[cts/s]:,  '+ line.strip()
        line = str(make_maps_conf.nbs32.ds_ic)
        print>>f, 'DS_IC[cts/s]:,  '+ line.strip()
        line = str(make_maps_conf.nbs32.us_amp[0])+', '
        for ii in range(2):
            line = line + str(make_maps_conf.nbs32.us_amp[ii])+', '
        print>>f, 'US_AMP[sensitivity/units/factor]:,  '+ line.strip()
        line = str(make_maps_conf.nbs32.ds_amp[0])+', '
        for ii in range(2):
            line = line + str(make_maps_conf.nbs32.ds_amp[ii])+', '
        print>>f,  'DS_AMP[sensitivity/units/factor]:,  '+ line.strip()
        line =  ' '
        print>>f,  line 
    
        line = 'NBS1833:, '+make_maps_conf.nbs33.name
        print>>f, line 
        line = str(make_maps_conf.nbs33.real_time[0]) +', '
        for ii in range(n_max) :
            line = line + str(make_maps_conf.nbs33.real_time[ii])+', '
        print>>f,  'realtime[s]:,  '+ line.strip()
        line = str(make_maps_conf.nbs33.live_time[0])+', '
        for ii in range(n_max) :
            line = line + str(make_maps_conf.nbs33.live_time[ii])+', '
        print>>f,  'livetime[s]:,  '+ line.strip()
        line = str(make_maps_conf.nbs33.current)
        print>>f,  'I_[mA]:,  '+ line.strip()
        line = str(make_maps_conf.nbs33.us_ic)
        print>>f,  'US_IC[cts/s]:,  '+ line.strip()
        line = str(make_maps_conf.nbs33.ds_ic)
        print>>f,  'DS_IC[cts/s]:,  '+ line.strip()
        line = str(make_maps_conf.nbs33.us_amp[0])+', '
        for ii in range(2) :
            line = line + str(make_maps_conf.nbs33.us_amp[ii])+', '
        print>>f,  'US_AMP[sensitivity/units/factor]:,  '+ line.strip()
        line = str(make_maps_conf.nbs33.ds_amp[0])+', '
        for ii in range(2) :
            line = line + str(make_maps_conf.nbs33.ds_amp[ii])+', '
        print>>f,  'DS_AMP[sensitivity/units/factor]:,  '+ line.strip()
        line =  ' '
        print>>f,  line 
    
    
        line = 'AXO:, '+ make_maps_conf.axo.name
        print>>f,  line 
        line = str(make_maps_conf.axo.real_time[0]) +', '
        for ii in range(n_max) :
            line = line + str(make_maps_conf.axo.real_time[ii])+', '
        print>>f,  'realtime[s]:,  '+ line.strip()
        line = str(make_maps_conf.axo.live_time[0])+', '
        for ii in range(n_max) :
            line = line + str(make_maps_conf.axo.live_time[ii])+', '
        print>>f,  'livetime[s]:,  '+ line.strip()
        line = str(make_maps_conf.axo.current)
        print>>f,  'I_[mA]:,  '+ line.strip()
        line = str(make_maps_conf.axo.us_ic)
        print>>f,  'US_IC[cts/s]:,  '+ line.strip()
        line = str(make_maps_conf.axo.ds_ic)
        print>>f,  'DS_IC[cts/s]:,  '+ line.strip()
        line = str(make_maps_conf.axo.us_amp[0])+', '
        for ii in range(2) :
            line = line + str(make_maps_conf.axo.us_amp[ii])+', '
        print>>f,  'US_AMP[sensitivity/units/factor]:,  '+ line.strip()
        line = str(make_maps_conf.axo.ds_amp[0])+', '
        for ii in range(2) :
            line = line + str(make_maps_conf.axo.ds_amp[ii])+', '
        print>>f,  'DS_AMP[sensitivity/units/factor]:,  '+ line.strip()
        line =  ' '
        print>>f,  line     
    
    
        line = 'name '+', '
        line = line+'z'+', ' 
        line = line+'units[DS_IC]'+', ' 
        line = line+'units[US_IC]'+', ' 
        line = line+'units[SRCurrent]'+', ' 
        line = line+'units[1]'+', ' 
        line = line+'use'+', ' 
        line = line+'calib'+', ' 
        line = line+'center'+', ' 
        line = line+'shift'+', ' 
        line = line+'width'+', ' 
        line = line+'bkground_left'+', ' 
        line = line+'bkground_right'+', ' 
        line = line+'left_roi[0]'+', ' 
        line = line+'right_roi[0]'+', ' 
        line = line+'left_bkground[0]'+', ' 
        line = line+'right_bkground[0]'+', ' 
        line = line+'absorption_1um_element'+', ' 
        line = line+'Be_window_transmission'+', ' 
        line = line+'Ge_dead_layer_transmission'+', ' 
        line = line+'XRF_yield'+', ' 
        line = line+'fraction_photons_absorbed_in_det'+', ' 
        line = line+'air_absoprtion'+', ' 
        line = line+'total_XRF_efficiency_factor'+', ' 
           
        print>>f,  line.strip()
        
    
        if np.sum(aux_arr) != 0:
            max_eff = aux_arr[:, 0]*aux_arr[:, 1]*aux_arr[:, 2]*aux_arr[:, 3]*(1.-aux_arr[:, 4])*aux_arr[:, 5]
            max_eff = np.amax(max_eff)
        for ii in range(make_maps_conf.n_chan):
            line = make_maps_conf.chan[ii].name +', '
            line = line+str(make_maps_conf.chan[ii].z)+', ' 
            line = line+make_maps_conf.chan[ii].units[0]+', ' 
            line = line+make_maps_conf.chan[ii].units[1]+', ' 
            line = line+make_maps_conf.chan[ii].units[2]+', ' 
            line = line+make_maps_conf.chan[ii].units[3]+', ' 
            line = line+str(make_maps_conf.chan[ii].use)+', ' 
            line = line+str(make_maps_conf.chan[ii].calib)+', ' 
            line = line+str(make_maps_conf.chan[ii].center)+', ' 
            line = line+str(make_maps_conf.chan[ii].shift)+', ' 
            line = line+str(make_maps_conf.chan[ii].width)+', ' 
            line = line+str(make_maps_conf.chan[ii].bkground_left)+', ' 
            line = line+str(make_maps_conf.chan[ii].bkground_right)+', ' 
            line = line+str(make_maps_conf.chan[ii].left_roi[0])+', ' 
            line = line+str(make_maps_conf.chan[ii].right_roi[0])+', ' 
            line = line+str(make_maps_conf.chan[ii].left_bkground[0])+', ' 
            line = line+str(make_maps_conf.chan[ii].right_bkground[0])+', ' 
            if np.sum(aux_arr) != 0 :
                line = line+str(aux_arr[ii, 0])+', ' 
                line = line+str(aux_arr[ii, 1])+', ' 
                line = line+str(aux_arr[ii, 2])+', ' 
                line = line+str(aux_arr[ii, 3])+', ' 
                line = line+str(1-aux_arr[ii, 4])+', ' 
                line = line+str(aux_arr[ii, 5])+', ' 
                line = line+str( aux_arr[ii, 0]*aux_arr[ii, 1]*aux_arr[ii, 2]*aux_arr[ii, 3]*(1.-aux_arr[ii, 4])*aux_arr[ii, 5] / max_eff)+', ' 
              
            print>>f, line.strip()


            
        line = ' '
        print>>f, line.strip()       
        line = 'calibration curve for ROIs'
        print>>f,  line.strip()       
        line = 'desc, name, Z, for_SRCurrent_norm, for_US_IC_norm, for_DS_IC_norm[XRF_counts_per_s/(weight_ugr_cm*axo.current)], measured_counts, error_bar'
        if make_maps_conf.version >= 8 : line = line+', measured_nb1832, measured_nb1833'
        print>>f,  line.strip()       
        for ii in range(make_maps_conf.n_chan): 
            line = 'calib_curve_roi,' + make_maps_conf.chan[ii].name +', '
            line = line+str(make_maps_conf.chan[ii].z)+', ' 
            for jj in range(5) :
                line = line+str(make_maps_conf.axo_e_cal[ii, 0, jj])+', ' 
            if make_maps_conf.version >= 8 :
                for jj in range(5, 7):
                    line = line+str(make_maps_conf.axo_e_cal[ii, 0, jj])+', '                     
            print>>f,  line.strip()      


    
        line = ' '
        print>>f,  line.strip()       
        line = 'calibration curve for fitted data'
        print>>f,  line.strip()       
        line = 'desc, name, Z, for_SRCurrent_norm, for_US_IC_norm, for_DS_IC_norm, measured_counts, error_bar'
        if make_maps_conf.version >= 8 : line = line+', measured_nb1832, measured_nb1833'
        print>>f,  line.strip()       
        for ii in range(make_maps_conf.n_chan):
            line = 'calib_curve_fitted,' + make_maps_conf.chan[ii].name +', '
            line = line+str(make_maps_conf.chan[ii].z)+', ' 
            for jj in range(5):
                line = line+str(make_maps_conf.axo_e_cal[ii, 1, jj])+', ' 
            if make_maps_conf.version >= 8 :
                for jj in range(5, 7) :
                    line = line+str(make_maps_conf.axo_e_cal[ii, 1, jj])+', '                     
            print>>f,  line.strip()          


        line = ' '
        print>>f,  line.strip()       
        line = 'calibration curve for ROI+'
        print>>f,  line.strip()      
        line = 'desc, name, Z, for_SRCurrent_norm, for_US_IC_norm, for_DS_IC_norm[XRF_counts_per_s/(weight_ugr_cm*axo.current)], measured_counts, error_bar'
        if make_maps_conf.version >= 8 : line = line+', measured_nb1832, measured_nb1833'
        print>>f,  line.strip()    
        for ii in range(make_maps_conf.n_chan):
            line = 'calib_curve_roi,' + make_maps_conf.chan[ii].name +', '
            line = line+str(make_maps_conf.chan[ii].z)+', ' 
            for jj in range(5):
                line = line+str(make_maps_conf.axo_e_cal[ii, 2, jj])+', ' 
            if make_maps_conf.version >= 8 :
                for jj in range(5, 7) :
                    line = line+str(make_maps_conf.axo_e_cal[ii, 2, jj])+', '                     
            print>>f,  line.strip()        
            
        f.close()


        print "Saved calibration info file"
        
        wo = np.where(np.array(chan_calib) == 1)[0]
        if len(wo) > 0: 

            print 'ploting spectrum'
            
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            mplot.rcParams['pdf.fonttype'] = 42
            
            fontsize = 9
            mplot.rcParams['font.size'] = fontsize
            
            colortable = []
        
            colortable.append((0., 0., 0.)) # ; black
            colortable.append((1., 0., 0.)) # ; red
            colortable.append((0., 1., 0.)) # ; green
            colortable.append((0., 0., 1.)) # ; blue
            colortable.append((0., 1., 1.)) # ; turquois
            colortable.append((1., 0., 1.)) # ; magenta
            colortable.append((1., 1., 0.)) # ; yellow
            colortable.append((0.7, 0.7, 0.7)) # ; light grey
            colortable.append((1., 0.8, 0.75)) # ; flesh
            colortable.append(( 0.35,  0.35,  0.35)) # ; dark grey       
            colortable.append((0., 0.5, 0.5)) # ; sea green
            colortable.append((1., 0., 0.53)) # ; pink-red
            colortable.append((0., 1., 0.68)) # ; bluegreen 
            colortable.append((1., 0.5, 0.)) # ; orange
            colortable.append((0., 0.68, 1.)) # ; another blue
            colortable.append((0.5, 0., 1.)) # ; violet
            colortable.append((1., 1., 1.)) # ; white
            
            foreground_color = colortable[-1]
            background_color = colortable[0]
                
            dpi = 100
            canvas_xsize_in = 1000./dpi
            canvas_ysize_in = 700./dpi
            
            bindingEK = []
            for ib in range(len(info_elements)): bindingEK.append(info_elements[ib].bindingE['K'])
            bindingEK = np.array(bindingEK)
                        
            for k in range(3) :
                for l in range(3) : 
                    
                    fig = mplot.figure.Figure(figsize=(canvas_xsize_in, canvas_ysize_in), dpi=dpi, facecolor=background_color, edgecolor=None)
                    canvas = FigureCanvas(fig)
                    fig.add_axes()
                    axes = fig.gca()
                    for child in axes.get_children():
                        if isinstance(child, mplot.spines.Spine):
                            child.set_color(foreground_color)
                    axes.set_axis_bgcolor(background_color)
                    ya = axes.yaxis                  
                    xa = axes.xaxis                          
                    ya.set_tick_params(labelcolor=foreground_color) 
                    ya.set_tick_params(color=foreground_color) 
                    xa.set_tick_params(labelcolor=foreground_color) 
                    xa.set_tick_params(color=foreground_color)     
    
                    title = 'AXO calibration curve '
                    if l == 0 : title = title +'for ROI sum'
                    if l == 1 : title = title +'for fitted data'
                    if l == 2 : title = title +'for ROIs+'
                    if k == 0 : title = title +' with normalization by Synchrotron current'
                    if k == 1 : title = title +'with normalization by upstream ionchamber'
                    if k == 2 : title = title +'with normalization by downstream ionchamber'
                    if k == 0 : ytitle = 'XRF_counts_per_s/(weight_ugr_cm*nbs.current)'
                    if k == 1 : ytitle = 'XRF_counts_per_s/(weight_ugr_cm*US_IC)'
                    if k == 2 : ytitle = 'XRF_counts_per_s(weight_ugr_cm*DS_IC)'
                    y = make_maps_conf.axo_e_cal[wo, l, k]    
                    measured = y.copy()
                    measured[:] = 0.
                    measured_32 = measured.copy()
                    measured_33 = measured.copy()
                    for ii in range(len(wo)): 
                        weight_ugr_cm = 0.
                        weight_ugr_cm = self.lookup_axo_standard_weight(make_maps_conf.chan[wo[ii]].name)
                       
                        if weight_ugr_cm == 0. : continue
                        if k ==0 : norm = make_maps_conf.axo.current
                        if k ==1 : norm = make_maps_conf.axo.us_ic
                        if k ==2 : norm = make_maps_conf.axo.ds_ic
                        measured[ii] = 1.0/(weight_ugr_cm*norm/make_maps_conf.axo_e_cal[wo[ii], l, 3])
                        
                    for ii in range(len(wo)): 
                        weight_ugr_cm = 0.
                        weight_ugr_cm = self.lookup_nbs_standard_weight(make_maps_conf.chan[wo[ii]].name, old_ratio = old_ratio)
                        if weight_ugr_cm == 0. : continue
                        if make_maps_conf.version > 8 : 
                            if k ==0 : norm = make_maps_conf.nbs32.current
                            if k ==1 : norm = make_maps_conf.nbs32.us_ic
                            if k ==2 : norm = make_maps_conf.nbs32.ds_ic
                            measured_32[ii] = 1.0/(weight_ugr_cm*norm/make_maps_conf.e_cal[wo[ii], l, 5])
                            if k ==0 : norm = make_maps_conf.nbs33.current
                            if k ==1 : norm = make_maps_conf.nbs33.us_ic
                            if k ==2 : norm = make_maps_conf.nbs33.ds_ic
                            measured_33[ii] = 1.0/(weight_ugr_cm*norm/make_maps_conf.e_cal[wo[ii], l, 6])
       
    
                    x = []
                    for iw in range(len(wo)): x.append(make_maps_conf.chan[iw].z)
                    x = np.array(x)
                    y = np.array(y)  
                    
                    temp1 = np.where(bindingEK <= make_maps_conf.incident_E)
                    temp = np.where(bindingEK[temp1] != 0.)[0]
                    
                    wo_2 = np.where(x <= max(temp)+1)[0]
                    # do NOT plot for Na and Mg.
                    wo_2 = np.delete(wo_2,0)
                    wo_2 = np.delete(wo_2,0)

                    x = x[wo_2]
                    y = y[wo_2]       
                    measured = measured[wo_2]       
                    measured_32 = measured_32[wo_2]       
                    measured_33 = measured_33[wo_2]  
 
                    x_p = x.copy()
                    y_p = y.copy()
                    x_p = np.insert(x_p, 0, x_p[0]-1) 
                    x_p = np.append(x_p, max(x_p)+1)
                    y_p = np.insert(y_p, 0, 0)
                    y_p = np.append(y_p, 0)
                    measured_p = measured.copy()
                    measured_p = np.insert(measured_p, 0, 0) 
                    measured_p = np.append(measured_p, 0)
                    measured_p_32 = measured_32.copy()
                    measured_p_32 = np.insert(measured_p_32, 0, 0.0) 
                    measured_p_32 = np.append(measured_p_32, 0.0)
                    measured_p_33 = measured_33.copy()
                    measured_p_33 = np.insert(measured_p_33, 0, 0.0) 
                    measured_p_33 = np.append(measured_p_33, 0.0)
                    xtickv = x_p
                    chnames = np.array(chan_names)
                    chnames = chnames[wo[wo_2]].tolist()
                    xtickname = [' '] + chnames+[' ']
                        
                    wo_3 = np.where(y != 0)
                    if len(wo_3[0]) > 0 : y_min = min(y[wo_3]) 
                    else: y_min = 0.
                    yrange = [y_min*0.8, max(y)*1.5]       
                    #if y_min < 1e-10 : yrange = [1e-4, max(y)*1.5]    


                    try:    
                        y_p = y_p + 0.0001
                        plot1 = axes.semilogy(x_p+0.5, y_p, color = foreground_color, linewidth=1.0, linestyle='steps')  
                        #plot1a = axes.semilogy(x_p,y_p, color = 'yellow', linewidth=1.0, linestyle='None', marker = '*')  
                        plot2 = axes.semilogy(x_p, measured_p, color = 'blue', linewidth=1.0, linestyle='None', marker = 'x', markersize = 7, mew=1.2)  
                        plot3 = axes.semilogy(x_p, measured_p_32, color = 'green', linewidth=1.0, linestyle='None', marker = 'x', markersize = 7, mew=1.2)  
                        plot4 = axes.semilogy(x_p, measured_p_33, color = 'red', linewidth=1.0, linestyle='None', marker = 'x', markersize = 7, mew=1.2)   
    
                        axes.xaxis.set_ticks(xtickv)
                        axes.set_ylabel(ytitle, color=foreground_color)
                        
                        axes.set_xlim((x_p[0], x_p[-1]))
                        axes.set_ylim(yrange)
                        axes.set_xticklabels(xtickname)
                        axes.set_position([0.10,0.08,0.85,0.85])
                        axes.set_title(title, color=foreground_color)
                                          
                       
                        axes.text(0.60, 0.13, 'NBS 1832', color = 'green', transform = axes.transAxes) 
                        axes.text(0.60, 0.10, 'NBS 1833', color = 'red', transform = axes.transAxes) 
                        axes.text(0.60, 0.07, 'axo', color = 'blue', transform = axes.transAxes) 
                                             
                        axes.text(0.97, -0.08, 'mapspy', color = foreground_color, transform = axes.transAxes) 
                        image_filename = 'calib'+str(l)+'_'+str(k)+'axo.png'
                        print 'saving ', os.path.join(dir,image_filename)
                        fig.savefig(os.path.join(dir, image_filename), dpi=dpi, facecolor=background_color, edgecolor=None)
                    except:
                        print 'Warning: Could not save axo calibration plot.'
                    
                   


        
        
        return    
    
       
#-----------------------------------------------------------------------------    
    def do_fits(self, this_w_uname, fitp, dofit_spec, spectra, per_pix = 0, generate_img = 0, maxiter = 500, suffix = '', info_elements = 0):
        beamline = self.main['beamline']
        keywords = fitp.keywords
        
        fp = maps_fit_parameters.maps_fit_parameters()
        avg_fitp = fp.define_fitp(beamline, info_elements)

        avg_fitp.s.val[:] = 0.
        avg_n_fitp = 0.        

        if suffix != '':
            maps_overridefile = os.path.join(self.main['master_dir'],'maps_fit_parameters_override.txt')+suffix
            try:
                f = open(maps_overridefile, 'rt')      
                f.close()
            except:      
                # if i cannot find an override file specific per detector, assuming
                # there is a single overall file.
                maps_overridefile = os.path.join(self.main['master_dir'],'maps_fit_parameters_override.txt')

        else:
            maps_overridefile = os.path.join(self.main['master_dir'],'maps_fit_parameters_override.txt')
            
        used_chan = []
        for i in range(self.main['max_spectra']-11):
            used_chan.append(spectra[i].used_chan)
        wo = np.where(np.array(used_chan) > 0)[0]
        tot_wo = len(wo)
        #print 'fiting n spectra', tot_wo
        if tot_wo == 0 :
            return (0, 0, spectra)
        names = ['none']
        for i in range(len(spectra)):
            if spectra[i].name != '': names.append(spectra[i].name)


        
        #n_names = len(names)
        # now go one by one through all spectra loaded into the plot_spec window
        print 'tot_wo', tot_wo
        for i in range(tot_wo):    
            old_fitp = fp.define_fitp(beamline, info_elements)
            old_fitp.s.val[:]=fitp.s.val[:]
           
            for j in range(keywords.kele_pos[0]):  fitp.s.use[j] = fitp.s.batch[j,0]
    
            if spectra[wo[i]].date['year'] == 0 : spectra[wo[i]].date['year'] = 1. # avoid error of julday routine for year number zero
            test_string = ['Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 
                          'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 
                          'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 
                          'In', 'Sn', 'Sb', 'Te', 'I', 'Cd_L', 'I_L', 'Cs_L', 'Ba_L', 'Au_L', 'Hg_L', 'Pb_L', 
                          'Au_M', 'U_M', 'Pu_L', 'Pb_M'] 
            # first disable fitting for all elements, and set the values to be
            # close to zero (i.e., set fit pars small)
            fitp.s.val[keywords.kele_pos] = 1e-10
            fitp.s.val[keywords.mele_pos] = 1e-10
            fitp.s.val[keywords.lele_pos] = 1e-10
            fitp.s.use[keywords.kele_pos] = 1.
            fitp.s.use[keywords.mele_pos] = 1.
            fitp.s.use[keywords.lele_pos] = 1.
                         
            # now enable selected elements, either from read file, or from
            # knowning it is an nbs standard
            #print 'spectra[wo[i]].name', spectra[wo[i]].name
            if 'nbs' in spectra[wo[i]].name:
                if '32' in spectra[wo[i]].name:
                    test_string = ['Al', 'Si', 'Ar', 'Ca', 'V', 'Mn', 'Co', 'Cu']
                    for jj in range(fitp.g.n_fitp) : 
                        if fitp.s.name[jj] in test_string:
                            #wo_a = test_string.index(fitp.s.name[jj])
                            fitp.s.val[jj] = 1.
                            fitp.s.use[jj] = 5
                if '33' in spectra[wo[i]].name:
                    test_string = ['Si', 'Pb_M', 'Ar', 'K', 'Ti', 'Fe', 'Zn', 'Pb_L']
                    for jj in range(fitp.g.n_fitp) : 
                        if fitp.s.name[jj] in test_string:
                            #wo_a = test_string.index(fitp.s.name[jj])
                            fitp.s.val[jj] = 1.
                            fitp.s.use[jj] = 5

            else: 
                print 'fitting spectrum'
                # if not NBS standard then look here
                det = 0   
                try:
                    fitp, test_string, pileup_string = fp.read_fitp(maps_overridefile, info_elements, det=det)
                    which_par_str = 'found override file (maps_fit_parameters_override.txt). Using the contained parameters.', test_string
                except:        
                    print 'warning: did not find override file (maps_fit_parameters_override.txt). Will abort this action'
                    return (0, 0, spectra)
                for jj in range(fitp.g.n_fitp) : 
                    if fitp.s.name[jj] in test_string:
                        #wo_a = test_string.index(fitp.s.name[jj])
                        fitp.s.val[jj] = 1.
                        fitp.s.use[jj] = 5

#             temp_fitp_use = fitp.s.use[np.amin(fitp.keywords.kele_pos):np.amax(fitp.keywords.mele_pos)+1]
#             temp_fitp_name = fitp.s.name[np.amin(fitp.keywords.kele_pos):np.amax(fitp.keywords.mele_pos)+1]
#             which_elements_to_fit = (np.nonzero(temp_fitp_use != 1))[0]
#             print 'elements to fit:'
#             print temp_fitp_name[which_elements_to_fit]  
             
 
 
            det = 0   
            pileup_string = ''
#             try:
#                 fitp, test_string, pileup_string = fp.read_fitp(maps_overridefile, info_elements, det=det)
#             except:
#                 print 'error reading fit paramenters'
                     
            if avg_n_fitp == 0 :
                # make sure that avg_fitp gets redefined here, so that changes,
                # etc, in the override file get translateed into the avg file on
                # the first round
                fp = maps_fit_parameters.maps_fit_parameters()
                avg_fitp = fp.define_fitp(beamline, info_elements)
                avg_fitp.s.val[:] = 0.
       
            if (this_w_uname == "DO_FIT_ALL_W_TAILS") :                 
                for j in range(keywords.kele_pos[0]) :  fitp.s.use[j] = fitp.s.batch[j,2]
            if (this_w_uname == "DO_MATRIX_FIT") :                  
                for j in range(keywords.kele_pos[0]):  fitp.s.use[j] = fitp.s.batch[j,1]
            if (this_w_uname == "DO_FIT_ALL_FREE") :                
                for j in range(keywords.kele_pos[0]):  fitp.s.use[j] = fitp.s.batch[j,3] 
            if (this_w_uname == "DO_FIT_ALL_FREE_E_FIXED_REST") :                  
                for j in range(keywords.kele_pos[0]):  fitp.s.use[j] = fitp.s.batch[j,4]
    
            fp.parse_pileupdef(fitp, pileup_string, info_elements)

  
            if (this_w_uname == "DO_MATRIX_FIT") :
                matrix = 1 
            else: 
                matrix = 0
            first = 1   

            
            temp_fitp_use = fitp.s.use[np.amin(fitp.keywords.kele_pos):np.amax(fitp.keywords.mele_pos)+1]
            temp_fitp_name = fitp.s.name[np.amin(fitp.keywords.kele_pos):np.amax(fitp.keywords.mele_pos)+1]
            which_elements_to_fit = (np.nonzero(temp_fitp_use != 1))[0]
            print 'elements to fit:'
            print temp_fitp_name[which_elements_to_fit]  
        
            fit = maps_analyze.analyze()
            u, fitted_spec, background, xmin, xmax, perror = fit.fit_spectrum(fitp, spectra[wo[i]].data, spectra[wo[i]].used_chan, spectra[wo[i]].calib, 
                            first = first, matrix = matrix, maxiter = maxiter)

            counts_background, counts_ka, counts_kb, counts_l, counts_m, \
            counts_elastic, counts_compton, counts_step, counts_tail, \
            counts_pileup, counts_escape = fit.get_counts()

            if (this_w_uname == "DO_FIT_ALL_FREE") :                  
                fitp.s.val[:] = u[:]       
                fitp.s.val[keywords.peaks] = 10.0**u[keywords.peaks]
                for j in range(keywords.kele_pos[0]): fitp.s.use[j] = fitp.s.batch[j, 3]
                u, fitted_spec, background, xmin, xmax, perror = fit.fit_spectrum(fitp, spectra[wo[i]].data, spectra[wo[i]].used_chan, spectra[wo[i]].calib, 
                                                                                  first = first, matrix = matrix, maxiter = maxiter)                

                counts_background, counts_ka, counts_kb, counts_l, counts_m, \
                counts_elastic, counts_compton, counts_step, counts_tail, \
                counts_pileup, counts_escape = fit.get_counts()
            
            add_plot_spectra = np.zeros((self.main['max_spec_channels'], 12))
            add_plot_names = ['fitted', 'K alpha', 'background', 'K beta', 'L lines', 'M lines', 'step', 'tail', 'elastic', 'compton', 'pileup', 'escape']
    
            add_plot_spectra[xmin:xmax+1, 0] = fitted_spec[:]
            add_plot_spectra[xmin:xmax+1, 1] = counts_ka[:]
            add_plot_spectra[0:np.amin([spectra[wo[i]].used_chan, len(background)-1]), 2] = background[0:np.amin([spectra[wo[i]].used_chan, len(background)-1])] 
            add_plot_spectra[xmin:xmax+1, 3] = counts_kb[:]
            add_plot_spectra[xmin:xmax+1, 4] = counts_l[:]
            add_plot_spectra[xmin:xmax+1, 5] = counts_m[:]
            add_plot_spectra[xmin:xmax+1, 6] = counts_step[:]
            add_plot_spectra[xmin:xmax+1, 7] = counts_tail[:]
            add_plot_spectra[xmin:xmax+1, 8] = counts_elastic[:]
            add_plot_spectra[xmin:xmax+1, 9] = counts_compton[:]
            add_plot_spectra[xmin:xmax+1, 10] = counts_pileup[:]
            add_plot_spectra[xmin:xmax+1, 11] = counts_escape[:]
    

            fitp.s.val[:] = u[:]       
            fitp.s.val[keywords.peaks] = 10.0**u[keywords.peaks]
       

            # this is not quite correct. the 1 sigma values are calculated for
            # the fit pars, which is used in the exponential. to translate them
            # into a meaning ful number, just calculate the upper bound and call
            # that +/- error
            perror[keywords.peaks] = 10.0**(perror[keywords.peaks] + u[keywords.peaks]) - 10.0**u[keywords.peaks]
    
            spectra[wo[i]].calib['off'] = fitp.s.val[keywords.energy_pos[0]]
            spectra[wo[i]].calib['lin'] = fitp.s.val[keywords.energy_pos[1]]
            spectra[wo[i]].calib['quad'] = fitp.s.val[keywords.energy_pos[2]]
           
            spectra[self.main['max_spectra']-8].data[:] = 0.
            spectra[self.main['max_spectra']-8].data[xmin:xmax+1] = fitted_spec[:]
            spectra[self.main['max_spectra']-8].name = 'fitted'
            for isp in range(self.main['max_spectra']-8,self.main['max_spectra']-3):
                spectra[isp].used_chan = spectra[wo[i]].used_chan 
                spectra[isp].calib['off'] = spectra[wo[i]].calib['off'] 
                spectra[isp].calib['lin'] = spectra[wo[i]].calib['lin']
                spectra[isp].calib['quad'] = spectra[wo[i]].calib['quad']
            spectra[self.main['max_spectra']-7].data[:] = 0.
            spectra[self.main['max_spectra']-7].data[xmin:xmax+1] = counts_ka[:]
            spectra[self.main['max_spectra']-7].name = 'ka_only'
            spectra[self.main['max_spectra']-6].data[:] = 0.
            spectra[self.main['max_spectra']-6].data[xmin:xmax+1] = counts_kb[:]
            spectra[self.main['max_spectra']-6].name = 'kb_only'
            spectra[self.main['max_spectra']-5].data[:] = 0.
            spectra[self.main['max_spectra']-5].data[xmin:xmax+1] = counts_tail[:]
            spectra[self.main['max_spectra']-5].name = 'tails'
            spectra[self.main['max_spectra']-4].data[:] = 0.
            spectra[self.main['max_spectra']-4].data[0:np.amin([spectra[wo[i]].used_chan, len(background)-1])] = background[0:np.amin([spectra[wo[i]].used_chan, len(background)-1])] 
            spectra[self.main['max_spectra']-4].name = 'background' 

            filename = 'specfit_'+names[wo[i]+1]+suffix
            maps_tools.plot_spectrum(info_elements, spectra = spectra, i_spectrum = wo[i], add_plot_spectra = add_plot_spectra, 
                                     add_plot_names = add_plot_names, ps = 0, fitp = fitp, filename= filename,
                                     outdir = self.main['output_dir'])
     


            if per_pix == 0 : 

                dirt = self.main['output_dir']
                if not os.path.exists(dirt):   
                    os.makedirs(dirt)
                    if not os.path.exists(dirt):
                        print 'warning: did not find the output directory, and could not create a new output directory. Will abort this action'
                        return (0, 0, spectra)
            else:
                if generate_img > 0 :
                    filename = os.path.join(self.main['output_dir'],'fit_'+names[wo[i]+1])
                    #write_spectrum, main.output_dir+strcompress('fit_'+names[wo[i]+1]), spectra, droplist_spectrum


            avg_fitp.s.val[:] = avg_fitp.s.val[:] + fitp.s.val[:]
            avg_n_fitp = avg_n_fitp + 1
  
        
        avg_fitp.s.val[:] = avg_fitp.s.val[:]/avg_n_fitp
        avg_fitp.s.max[:] = fitp.s.max[:]
        avg_fitp.s.min[:] = fitp.s.min[:]
        avgfilename = os.path.join(self.main['master_dir'],'average_resulting_maps_fit_parameters_override.txt')
        fp.write_fit_parameters(self.main, avg_fitp, avgfilename, test_string, pileup_string = pileup_string, suffix = suffix)
        
        #print 'fitp',fitp, 'avg_fitp',  avg_fitp, 'spectra', spectra
        return fitp, avg_fitp, spectra



        
#-----------------------------------------------------------------------------    
    def transmission(self, thickness, beta, llambda):

        arg = -4.*np.pi*thickness*beta/llambda 
        value = np.abs(np.math.exp(arg))
        return value
   
#-----------------------------------------------------------------------------    
    def absorption(self, thickness, beta, llambda, shell_factor = [] ):    
        
          
        # make sure shell_factor is defined, and if not, set it to 1
        # shell factor <1 is to determine how much is
        # absorbed by a subshell, and is essentially the
        # ratio of jump factor -1 / jump factor
        if shell_factor == 0 : shell_factor = 1
    
        arg = -4.*np.pi*thickness*shell_factor*beta/llambda
        value = 1-np.abs(np.math.exp(arg))
        return value
  
    
#-----------------------------------------------------------------------------    
    def fit_calibrationcurve(self, z_prime, p):

        #       aux_arr[mm, 0] = absorption
        #       aux_arr[mm, 1] = transmission, Be
        #       aux_arr[mm, 2] = transmission, Ge or Si dead layer
        #       aux_arr[mm, 3] = yield
        #       aux_arr[mm, 4] = transmission through Si detector
        #       aux_arr[mm, 5] = transmission through  air (N2)
        
        aux_arr = self.aux_arr

        value = p[0]*aux_arr[z_prime, 0]*aux_arr[z_prime, 1]*aux_arr[z_prime, 2]*aux_arr[z_prime, 3]*(1.-aux_arr[z_prime, 4])*aux_arr[z_prime, 5]

        return value

#-----------------------------------------------------------------------------      
    def residuals(self, p, y, x):
        err = (y-self.fit_calibrationcurve(x,p))
        return err          
                    
