'''
Created on Nov 21, 2011

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
import string 
import datetime
import time as tm
import numpy as np
import logging
import os
import struct
import matplotlib as mplot
import multiprocessing
import csv
import glob

from file_io import maps_mda
from file_io import maps_nc
from file_io import maps_hdf5
import maps_definitions
import maps_detector
import maps_fit_parameters
import maps_analyze
import maps_tools

import h5py
from file_io.file_util import open_file_with_retry, call_function_with_retry

NO_MATRIX = 0

FLYSACN_TYPE_NONE = 0
FLYSCAN_TYPE_NETCDF = 1
FLYSCAN_TYPE_BIONANOPROBE = 3
FLYSCAN_TYPE_HDF = 4

FLYSCAN_TYPE_2ID_NETCDF = 1
FLYSCAN_TYPE_8BM_NETCDF = 2


# ----------------------------------------------------------------------
def rebin(a, *args):
	'''
	rebin ndarray data into a smaller ndarray of the same rank whose dimensions
	are factors of the original dimensions. eg. An array with 6 columns and 4 rows
	can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
	example usages:
	>>> a=rand(6,4); b=rebin(a,3,2)
	>>> a=rand(6); b=rebin(a,2)
	'''
	shape = a.shape
	lenShape = len(shape)
	factor = np.asarray(shape)/np.asarray(args)
	factor = factor[0]
	args = args[0]
	evList = ['a.reshape('] + \
			['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
			[')'] + ['.sum(%d)'%(i+1) for i in range(lenShape)] + \
			['/factor[%d]'%i for i in range(lenShape)]

	return eval(''.join(evList))

# ----------------------------------------------------------------------
def fit_line_threaded(log_name, i_fit, data_line, n_rows,  matrix, spectral_binning, elt_line, fitp, old_fitp, keywords, xrf_bin, calib):

	logger = logging.getLogger(log_name)
	'''
	fHandler = logging.FileHandler(log_name)
	logger.setLevel(logging.DEBUG)
	fHandler.setLevel(logging.DEBUG)
	formatter = logging.Formatter('%(asctime)s | %(levelname)s | PID[%(process)d] | %(funcName)s(): %(message)s')
	fHandler.setFormatter(formatter)
	logger.addHandler(fHandler)
	'''
	logger.info('fitting row number %s', i_fit)

	fit = maps_analyze.analyze(logger)
	fitted_line, ka_line, l_line, bkground_line, values_line, bkgnd_line, tfy_line, xmin, xmax = fit.fit_line(data_line,
						n_rows, matrix, spectral_binning, elt_line, fitp, old_fitp, keywords, xrf_bin, calib)

	return [fitted_line, ka_line, l_line, bkground_line, values_line, bkgnd_line, tfy_line, xmin, xmax]

# ----------------------------------------------------------------------
class analyze:
	def __init__(self, logger, info_elements, main_dict, maps_conf, beamline='2-ID-E', use_fit=0):

		self.info_elements = info_elements
		self.beamline = beamline
		self.logger = logger
		self.integrate = 0
		
		self.main_dict = main_dict
		
		self.pca = 0
		self.save_ram = 0				  
		self.verbose = 2
		self.maxiter = 500 
		
		self.main_max_spectra = main_dict['max_spectra']
		self.max_spec_channels = main_dict['max_spec_channels']
		self.max_ICs = main_dict['max_ICs']
		
		self.show_extra_pvs = 1
		
		self.which_par_str = 0

		self.maps_def = maps_definitions.maps_definitions(self.logger)
		#self.maps_conf = self.maps_def.set_maps_definitions(beamline, info_elements)
		self.maps_conf = maps_conf
		
		self.maps_conf.use_fit = use_fit

		self.netcdf_fly_scan = 0
		self.save_h5 = 1
		self.xanes = 0

		if beamline == '2-ID-E': self.crate = '2xfm'
		if beamline == '2-ID-D': self.crate = '2idd'
		if beamline == '2-ID-B': self.crate = '2idb1'
		if beamline == '2-BM': self.crate = '2bmb'
		if beamline == 'Bio-CAT': self.crate = 'biocat'
		if beamline == 'GSE-CARS': self.crate = 'gsecars'

	# ----------------------------------------------------------------------
	def __binning__(self, scan, xrf_bin, n_cols, n_rows, mca_arr_dimensions, elt1_arr):
		self.logger.info('binning the data')

		this_mca_arr_dimensions = scan.mca_arr.shape
		this_n_channels = min(mca_arr_dimensions[2], self.main_dict['max_spec_channels'])
		if (xrf_bin == 2) and (n_cols > 5) and (n_rows > 5):
			for i_bin in range(n_cols - 1):
				if i_bin % 2 == 0:
					for jj in range(n_rows):
						if jj % 2 == 0:
							scan.mca_arr[i_bin, jj, 0:this_n_channels] = scan.mca_arr[i_bin, jj, 0:this_n_channels] + scan.mca_arr[i_bin + 1, jj, 0:this_n_channels] + scan.mca_arr[i_bin, jj+1, 0:this_n_channels]+scan.mca_arr[i_bin+1, jj+1, 0:this_n_channels]
							elt1_arr[i_bin, jj] = elt1_arr[i_bin, jj] + elt1_arr[i_bin + 1, jj]+elt1_arr[i_bin, jj + 1]+elt1_arr[i_bin + 1, jj + 1]
						else:
							scan.mca_arr[i_bin, jj, 0:this_n_channels] = scan.mca_arr[i_bin, jj - 1, 0:this_n_channels]
							elt1_arr[i_bin, jj] = elt1_arr[i_bin, jj - 1]
				else:
					scan.mca_arr[i_bin, :, 0:this_n_channels] = scan.mca_arr[i_bin - 1, :, 0:this_n_channels]
					elt1_arr[i_bin, :] = elt1_arr[i_bin - 1, :]

		if (xrf_bin == 3) and (n_cols > 5) and (n_rows > 5) :
			current_line = np.zeros((self.main_dict['max_spec_channels'], n_rows))
			previous_line = np.zeros((self.main_dict['max_spec_channels'], n_rows))
			next_line = np.zeros((self.main_dict['max_spec_channels'], n_rows))
			current_elt_line = np.zeros((elt1_arr.size))
			next_elt_line = np.zeros((elt1_arr.size))
			this_mca_arr_dimensions = scan.mca_arr.shape
			this_n_channels = min(this_mca_arr_dimensions[2], self.main_dict['max_spec_channels'])

			for i_bin in range(n_cols):
				if i_bin > 1 :
					previous_elt_line = current_elt_line.copy()
					current_elt_line = next_elt_line.copy()
					previous_line = current_line.copy()
					current_line = next_line.copy()
				else:
					for jj in range(n_rows-1) :
						current_line[0:this_n_channels, jj] = scan.mca_arr[i_bin, jj, 0:this_n_channels]
					current_elt_line = elt1_arr[i_bin, :]
					previous_line = current_line
					previous_elt_line = current_elt_line

				if i_bin < n_cols-1 :
					for jj in range(n_rows) :
						next_line[0:this_n_channels, jj] = scan.mca_arr[i_bin+1, jj, 0:this_n_channels]
					next_elt_line = elt1_arr[i_bin+1, :]

				if n_rows-2 > 1 :
					jj = 0
					scan.mca_arr[i_bin, jj, 0:this_n_channels] = previous_line[0:this_n_channels, jj] + previous_line[0:this_n_channels, jj] + previous_line[0:this_n_channels, jj+1] + \
						current_line[0:this_n_channels, jj] + current_line[0:this_n_channels, jj] + current_line[0:this_n_channels, jj+1] + \
						next_line[0:this_n_channels, jj] + next_line[0:this_n_channels, jj] + next_line[0:this_n_channels, jj+1]
					elt1_arr[i_bin, jj] = np.sum([previous_elt_line[jj:jj+1], current_elt_line[jj:jj+1], next_elt_line[jj:jj+1],
												previous_elt_line[jj], current_elt_line[jj], next_elt_line[jj]])
					for jj in range(n_rows-1) :
						scan.mca_arr[i_bin, jj, 0:this_n_channels] = previous_line[0:this_n_channels, jj-1] + previous_line[0:this_n_channels, jj] + previous_line[0:this_n_channels, jj+1] + \
							current_line[0:this_n_channels, jj-1] + current_line[0:this_n_channels, jj] + current_line[0:this_n_channels, jj+1] + \
							next_line[0:this_n_channels, jj-1] + next_line[0:this_n_channels, jj] + next_line[0:this_n_channels, jj+1]
						elt1_arr[i_bin, jj] = np.sum([previous_elt_line[jj-1:jj+1], current_elt_line[jj-1:jj+1], next_elt_line[jj-1:jj+1]])

					jj = n_rows-1
					scan.mca_arr[i_bin, jj, 0:this_n_channels] = previous_line[0:this_n_channels, jj] + previous_line[0:this_n_channels, jj] + previous_line[0:this_n_channels, jj] + \
						current_line[0:this_n_channels, jj] + current_line[0:this_n_channels, jj] + current_line[0:this_n_channels, jj] + \
						next_line[0:this_n_channels, jj] + next_line[0:this_n_channels, jj] + next_line[0:this_n_channels, jj]
					elt1_arr[i_bin, jj] = np.sum([previous_elt_line[jj-1:jj], current_elt_line[jj-1:jj], next_elt_line[jj-1:jj],
												previous_elt_line[jj], current_elt_line[jj], next_elt_line[jj]])

		if (xrf_bin == 4) and (n_cols > 5) and (n_rows > 5) :
			for i_bin in range(n_cols-2) :
				if i_bin % 3. == 0:
					for jj in range(n_rows-2):
						if jj % 3 == 0:
							scan.mca_arr[i_bin, jj, 0:this_n_channels] = scan.mca_arr[i_bin, jj, 0:this_n_channels]+scan.mca_arr[i_bin+1, jj, 0:this_n_channels]+scan.mca_arr[i_bin+2, jj, 0:this_n_channels]+\
								scan.mca_arr[i_bin, jj+1, 0:this_n_channels]+scan.mca_arr[i_bin+1, jj+1, 0:this_n_channels]+scan.mca_arr[i_bin+2, jj+1, 0:this_n_channels]+\
								scan.mca_arr[i_bin, jj+2, 0:this_n_channels]+scan.mca_arr[i_bin+1, jj+2, 0:this_n_channels]+scan.mca_arr[i_bin+2, jj+2, 0:this_n_channels]
							elt1_arr[i_bin, jj] = elt1_arr[i_bin, jj]+elt1_arr[i_bin+1, jj]+elt1_arr[i_bin+2, jj]+elt1_arr[i_bin+2, jj]+\
											  elt1_arr[i_bin, jj+1]+elt1_arr[i_bin+1, jj+1]+elt1_arr[i_bin, jj+1]+elt1_arr[i_bin+2, jj+1]+\
											  elt1_arr[i_bin, jj+2]+elt1_arr[i_bin+1, jj+2]+elt1_arr[i_bin, jj+2]+elt1_arr[i_bin+2, jj+2]
						else:
							if (jj+2) % 3 == 0:
								scan.mca_arr[i_bin, jj, 0:this_n_channels] = scan.mca_arr[i_bin, jj-1, 0:this_n_channels]
								elt1_arr[i_bin, jj] = elt1_arr[i_bin, jj-1]

							if (jj+1) % 3 == 0:
								scan.mca_arr[i_bin, jj, 0:this_n_channels] = scan.mca_arr[i_bin, jj-2, 0:this_n_channels]
								elt1_arr[i_bin, jj] = elt1_arr[i_bin, jj-2]

				else:
					if (i_bin+2) % 3 == 0:
						scan.mca_arr[i_bin, :, 0:this_n_channels] = scan.mca_arr[i_bin-1, :, 0:this_n_channels]
						elt1_arr[i_bin, :] = elt1_arr[i_bin-1, :]

					if (i_bin+1) % 3 == 0:
						scan.mca_arr[i_bin, :, 0:this_n_channels] = scan.mca_arr[i_bin-2, :, 0:this_n_channels]
						elt1_arr[i_bin, :] = elt1_arr[i_bin-2, :]

		return elt1_arr

	# ----------------------------------------------------------------------
	def maps_generate_img_add_extra(self, extra_pv, thisdata):
		try:
			if extra_pv.name.find('saveData_fileSystem') > -1:
				thisdata.add_str.a = extra_pv[0].str_value
			if extra_pv.name.find('saveData_subDir') > -1:
				thisdata.add_str.b = extra_pv[0].str_value
			if extra_pv.name.find('saveData_scanNumber') > -1:
				thisdata.add_str.c = extra_pv[0].str_value
			if extra_pv.name.find('userStringCalc10.AA') > -1:
				thisdata.add_str.d = extra_pv[0].str_value
			if extra_pv.name.find('userStringCalc10.BB') > -1:
				thisdata.add_str.e = extra_pv[0].str_value
			if extra_pv.name.find('userStringCalc10.CC') > -1:
				thisdata.add_str.f = extra_pv[0].str_value
			if extra_pv.name.find('userStringCalc10.DD') > -1:
				thisdata.add_str.g = extra_pv[0].str_value
			if extra_pv.name.find('userStringCalc10.EE') > -1:
				thisdata.add_str.h = extra_pv[0].str_value
		except:
			pass

		for i in range( min(len(extra_pv.name), 100) ):
			thisdata.extra_str_arr[i] = extra_pv[i].name + '; ' + extra_pv[i].str_value
		return

	# ----------------------------------------------------------------------
	def maps_core_generate_fly_dat(self, header, mdafilename, this_detector, total_number_detectors):
		n_ev, n_rows, n_cols, n_energy, energy, energy_spec, scan_time_stamp, dataset_orig = self.change_xrf_resetvars()
		mda_scan = maps_mda.mda(self.logger)
		scan = mda_scan.read_scan(mdafilename, threeD_only=2)
		if scan == None or scan.detector_description_arr == None:
			self.logger.info('Not a valid mda flyscan file, scan == None . returning: %s', mdafilename)
			return
		if len(scan.detector_description_arr) < 3:
			self.logger.info('Not a valid mda flyscan file, returning: %s', mdafilename)
			return

		new_detector_description_arr = ['us_ic', 'ds_ic', 'sum_mcs', 'H_dpc', 'V_dpc', 'dia1_dpc', 'dia2_dpc', 'sca1',
										'sca2', 'sca3', 'TFY-ICR', 'T', 'norm_H_dpc', 'norm_V_dpc', 'phase', 'abs']

		temp_a_arr = ['2xfm:mcs:mca10.VAL', '2xfm:mcs:mca11.VAL', '2xfm:mcs:mca12.VAL', '2xfm:mcs:mca13.VAL',
					  '2xfm:mcs:mca14.VAL', '2xfm:mcs:mca15.VAL', '2xfm:mcs:mca16.VAL', '2xfm:mcs:mca17.VAL',
					  '2xfm:mcs:mca18.VAL', '2xfm:mcs:mca19.VAL', '2xfm:mcs:mca2.VAL', '2xfm:mcs:mca3.VAL',
					  '2xfm:mcs:mca1.VAL', '2xfm:mcs:mca26.VAL', '2xfm:mcs:mca27.VAL', '2xfm:mcs:mca28.VAL',
					  '2xfm:mcs:mca29.VAL']

		temp_b_arr = ['SD_1', 'SD_2', 'SD_3', 'SD_4', 'SD_5', 'SD_6', 'SD_7', 'SD_8', 'SD_9', 'SD_10',
					'us_ic', 'ds_ic', 'T', 'sca1_det0', 'sca2_det0', 'sca3_det0', 'tfy_det0']

		b_pos = np.zeros(len(temp_a_arr))
		n_gui_chan = len(new_detector_description_arr)
		dataset_orig = np.ndarray((scan.x_pixels, scan.y_pixels, n_gui_chan), dtype=float)
		dataset_orig.fill(1.0)
		try:
			time_idx = scan.detector_description_arr.index('2xfm:mcs:mca1.VAL')
		except:
			time_idx = -1
		if time_idx == -1:
			try:
				time_idx = scan.detector_description_arr.index('8bmb:3820:mca1.VAL')
			except:
				time_idx = -1
			temp_a_arr = ['8bmb:3820:mca10.VAL', '8bmb:3820:mca11.VAL', '8bmb:3820:mca12.VAL', '8bmb:3820:mca13.VAL',
						  '8bmb:3820:mca14.VAL', '8bmb:3820:mca15.VAL', '8bmb:3820:mca16.VAL', '8bmb:3820:mca17.VAL',
						  '8bmb:3820:mca18.VAL', '8bmb:3820:mca19.VAL', '8bmb:3820:mca2.VAL', '8bmb:3820:mca3.VAL',
						  '8bmb:3820:mca1.VAL', '8bmb:3820:mca26.VAL', '8bmb:3820:mca27.VAL', '8bmb:3820:mca28.VAL',
						  '8bmb:3820:mca29.VAL']

		if time_idx == -1:
			self.logger.error('an error occured, cannot read dwell time in fly scan; returning')
			n_ev, n_rows, n_cols, n_energy, energy, energy_spec, scan_time_stamp, dataset_orig = self.change_xrf_resetvars()
			return

		time = scan.detector_arr[:, :, time_idx]
		time_norm = time / np.mean(time)
		for n in range(len(temp_a_arr)):
			try:
				b_pos[n] = scan.detector_description_arr.index(temp_a_arr[n])
				scan.detector_arr[:, :, b_pos[n]] = scan.detector_arr[:, :, b_pos[n]] / time[:, :]  # time_norm[*, *]
				# dataset_orig(*, *, n) = (detector_arr(*, *, idx[0])-32768.)/3276.8
			except:
				self.logger.debug('wo = -1; @ %s temp_a_arr[n] = %s', n, temp_a_arr[n])

		dataset_orig[:, :, 11] = time[:, :] / 25000

		# sum_mcs
		dataset_orig[:, :, 2] = scan.detector_arr[:, :, b_pos[1]] + scan.detector_arr[:, :, b_pos[2]] + scan.detector_arr[:, :, b_pos[3]] + scan.detector_arr[:, :, b_pos[4]]
		# H_dpc
		dataset_orig[:, :, 3] = scan.detector_arr[:, :, b_pos[1]] - scan.detector_arr[:, :, b_pos[2]] - scan.detector_arr[:, :, b_pos[3]] + scan.detector_arr[:, :, b_pos[4]]
		dataset_orig[:, :, 3] = dataset_orig[:, :, 3] / dataset_orig[:, :, 2]
		# V_dpc
		dataset_orig[:, :, 4] = scan.detector_arr[:, :, b_pos[1]] + scan.detector_arr[:, :, b_pos[2]] - scan.detector_arr[:, :, b_pos[3]] - scan.detector_arr[:, :, b_pos[4]]
		dataset_orig[:, :, 4] = dataset_orig[:, :, 3] / dataset_orig[:, :, 2]
		# dia1_dpc
		dataset_orig[:, :, 5] = scan.detector_arr[:, :, b_pos[1]] - scan.detector_arr[:, :, b_pos[3]]
		dataset_orig[:, :, 5] = dataset_orig[:, :, 5] / dataset_orig[:, :, 2]
		# dia2_dpc
		dataset_orig[:, :, 6] = scan.detector_arr[:, :, b_pos[2]] - scan.detector_arr[:, :, b_pos[4]]
		dataset_orig[:, :, 6] = dataset_orig[:, :, 6] / dataset_orig[:, :, 2]
		# us,ds_ic
		dataset_orig[:, :, 0] = scan.detector_arr[:, :, b_pos[10]]
		dataset_orig[:, :, 1] = scan.detector_arr[:, :, b_pos[11]]
		dataset_orig[:, :, 15] = dataset_orig[:, :, 1] / dataset_orig[:, :, 0]

		# scas
		dataset_orig[:, :, 7] = scan.detector_arr[:, :, b_pos[13]]
		dataset_orig[:, :, 8] = scan.detector_arr[:, :, b_pos[14]]
		dataset_orig[:, :, 9] = scan.detector_arr[:, :, b_pos[15]]
		dataset_orig[:, :, 10] = scan.detector_arr[:, :, b_pos[16]]

		nrml = dataset_orig[:, :, 3]
		sz = nrml.shape
		nx = sz[0]
		ny = sz[1]
		if (nx % 2) == 0:
			nx -= 1
		if (ny % 2) == 0:
			ny -= 1
		nrml = dataset_orig[2:nx-3, 0:ny-1, 3]
		ntmb = dataset_orig[2:nx-3, 0:ny-1, 4]

		if self.maps_conf.dmaps[this_detector].name == 'phase':
			no_int = 1
		else:
			no_int = 0
		anl = maps_analyze.analyze(self.logger)
		nrml, ntmb, rdt = anl.maps_simple_dpc_integration(nrml, ntmb, no_int=no_int)
		# notem nrml, ntmb, now normalized (what comes up ust go down)
		# first pad dataset with mean value, so that non assigned pixels are not
		# zero (for scaling purposes)
		dataset_orig[:, :, 12] = np.mean(nrml)
		dataset_orig[2:nx - 3, 0:ny - 1, 12] = nrml
		dataset_orig[:, :, 13] = np.mean(ntmb)
		dataset_orig[2:nx - 3, 0:ny - 1, 13] = ntmb
		dataset_orig[:, :, 14] = np.mean(rdt)
		dataset_orig[2:nx - 3, 0:ny - 1, 14] = rdt

		scan.detector_description_arr = new_detector_description_arr

		no_detectors = 1
		img_type = 0
		n_channels = 2048
		n_energy = 2048

		XRFmaps_info = self.maps_def.define_xrfmaps_info(scan.x_pixels, scan.y_pixels, 3, n_channels, n_energy, no_detectors, self.maps_conf.n_used_chan, self.maps_conf.n_used_dmaps, self.maps_conf)
		XRFmaps_info.img_type = img_type
		if scan.extra_pv:
			self.maps_generate_img_add_extra(scan.extra_pv, XRFmaps_info)

		XRFmaps_info.dataset_names[:] = ['fly', 'dummy', 'dummy']
		XRFmaps_info.scan_time_stamp = scan.scan_time_stamp
		XRFmaps_info.write_date = datetime.datetime.ctime(datetime.datetime.now())
		XRFmaps_info.x_coord_arr = scan.x_coord_arr
		XRFmaps_info.y_coord_arr = scan.y_coord_arr

		for j in range(len(new_detector_description_arr)):
			XRFmaps_info.dmaps_set[:, :, j] = dataset_orig[:, :, j]
		XRFmaps_info.dmaps_names[:] = new_detector_description_arr[:]
		XRFmaps_info.dmaps_units = ['raw']
		XRFmaps_info.chan_names = ['dummy']  # make_maps_conf.chan[wo].name
		XRFmaps_info.version = 5L
		XRFmaps_info.us_amp.fill(0.)       # us_amp
		XRFmaps_info.ds_amp.fill(0.)       # ds_amp
		XRFmaps_info.n_energy = n_energy

		dataset_size = len(XRFmaps_info.dataset_names)

		filename = os.path.join(self.main_dict['fly_dat_dir'], header + '.fly.dat')

		data_file = open_file_with_retry(str(filename), 'wb')
		if data_file == None:
			self.logger.error('Error: (maps_core_generate_fly_dat) failed to open file to write to : %s', filename)
			return None
		data_file.write(struct.pack('>iiiii', 0, XRFmaps_info.version, n_channels, scan.x_pixels, scan.y_pixels))
		data_file.write(struct.pack('>iiii', len(energy), no_detectors, dataset_size, self.maps_conf.n_used_chan))
		data_file.write(struct.pack('>i', self.maps_conf.n_used_dmaps))
		XRFmaps_info.dump(data_file)
		data_file.close()

	# ----------------------------------------------------------------------
	def read_and_parse_scan(self, header, mdafilename, beamline, this_detector, total_number_detectors):

		self.logger.info('beamline: %s', beamline)
		if beamline == 'Bio-CAT':
			self.logger.warning('cannot read biocat scans')
			return None

		if beamline == 'GSE-CARS':
			self.logger.warning('cannot read GSE-CARS scans')
			return None

		if (beamline == '2-ID-E') or (beamline == '2-ID-D') or (beamline == '2-ID-B') or (beamline == '2-BM') or (beamline == 'Bionanoprobe'):

			# read scan info
			mda = maps_mda.mda(self.logger)
			info = mda.read_scan_info(mdafilename)
			if info == None:
				self.logger.warning('Warning: skipping file : %s, due to maps_scan_info error', mdafilename)
				return None

			if np.amin(info.spectrum) == -1:
				self.logger.warning('Warning: skipping file : %s, due to maps_scan_info error', mdafilename)
				return None

			if (info.rank == 2) and (info.spectrum[info.rank - 1] == 1):
				self.xanes = 1
			else:
				self.xanes = 0

			if self.verbose == 1:
				self.logger.debug('info.rank	: %s', info.rank)
				self.logger.debug('info.dims	: %s', info.dims)
				self.logger.debug('info.spectrum: %s', info.spectrum)
				self.logger.debug('xanes		: %s', self.xanes)

			test_textfile = 0
			combined_file_info = os.path.join(self.main_dict['master_dir'], 'lookup', header + '.txt')
			if os.path.isfile(combined_file_info):
				test_textfile = 1
			self.logger.debug('testing test_textfile %s', test_textfile)

			test_netcdf = FLYSACN_TYPE_NONE
			file_path = os.path.join(self.main_dict['master_dir'], os.path.join('flyXRF.h5', header))
			hdf_files = glob.glob(file_path + '*.h5')
			num_files_found = len(hdf_files)
			if num_files_found != 1:
				if num_files_found > 1:
					self.logger.error('Error: too many files found, %s', hdf_files)
				else:
					self.logger.info('Could not find hdf5 file associated with mda file: %s', mdafilename)
			else:
				test_netcdf = FLYSCAN_TYPE_HDF

			if test_netcdf == FLYSACN_TYPE_NONE:
				self.logger.info('testing presence of converted flyscans %s', test_netcdf)
				file_path = os.path.join(self.main_dict['master_dir'], os.path.join('flyXRF', header))
				nc_files = glob.glob(file_path + '*.nc')
				num_files_found = len(nc_files)
				if num_files_found < 1:
					self.logger.info('Could not find netcdf files associated with mda: %s', mdafilename)
				else:
					test_netcdf = FLYSCAN_TYPE_NETCDF

			if test_netcdf == FLYSACN_TYPE_NONE:
				# bnp_fly_18_001.nc
				try:
					scan_number = header[-4:]
					scan_number = int(scan_number)
					bnpfly_header = ''.join('bnp_fly_' + str(scan_number))

					if os.path.isfile(os.path.join(self.main_dict['master_dir'], 'flyXRF', bnpfly_header + '_001.nc')):
						test_netCDF = FLYSCAN_TYPE_BIONANOPROBE
					# use test_netCDF = 3 to indicate this is a fly scan from bionanoprobe
				except:
					self.logger.error('This is not a scan file.')

			if (info.rank == 2) and (np.sum(info.spectrum) == 0 and (test_textfile == 0) and (test_netcdf == FLYSACN_TYPE_NONE)):
				self.logger.info('This is a fly scan, without XRF.')
				self.maps_core_generate_fly_dat(header, mdafilename, this_detector, total_number_detectors)
				return None

			elif (info.rank == 2) and (np.sum(info.spectrum) == 0 and (test_textfile > 0)):
				# this is a fly scan, but i found a text file, which should contain
				# the filename of the XRF file
				self.logger.warning( 'This scan has the combined file with info stored in a text file which is not yet supported - returning.')
				return None

			elif (info.rank == 2) and (np.sum(info.spectrum) == 0 and (test_netcdf == FLYSCAN_TYPE_2ID_NETCDF or test_netcdf == FLYSCAN_TYPE_NETCDF)):
				# this is a fly scan, but i found a netcdf file with matching
				# name. this should be a fly scna with XRF
				self.logger.info('trying to do the combined file')
				nc = maps_nc.nc(self.logger)
				scan = nc.read_combined_nc_scans(mdafilename, self.main_dict['master_dir'], header, this_detector, extra_pvs=True)

				self.logger.info('Finished reading combined nc scan')
				self.netcdf_fly_scan = 1

			elif (info.rank == 2) and (np.sum(info.spectrum) == 0 and (test_netcdf == FLYSCAN_TYPE_8BM_NETCDF)):
				# this is a fly scan, but i found a netcdf file with matching
				# name. this should be a fly scan with XRF from 8bm
				self.logger.warning('This scan with XRF from 8bm which is not yet supported - returning.')
				return None

			elif (info.rank == 2) and (np.sum(info.spectrum) == 0 and (test_netcdf == FLYSCAN_TYPE_BIONANOPROBE)):
				# this is a fly scan, but i found a netcdf file with matching
				# name. this should be a fly scan with XRF from the bionanoprobe
				self.logger.warning('This scan with with XRF from the bionanoprobe which is not yet supported - returning.')
				return None

			elif (info.rank == 2) and (np.sum(info.spectrum) == 0 and (test_netcdf == FLYSCAN_TYPE_HDF)):
				# this is a fly scan, but i found a netcdf file with matching
				# name. this should be a fly scan with XRF
				self.logger.warning('trying to do the combined file')

				scan = mda.read_combined_flyscan(self.main_dict['master_dir'], mdafilename, this_detector)

				self.netcdf_fly_scan = 1

			elif self.xanes == 1:
				self.logger.warning('xanes scans not supported - returning')
				return None

			else:
				# Read mda scan:
				self.logger.info('Reading scan from %s', mdafilename)
				try:
					scan = mda.read_scan(mdafilename, extra_pvs=True)
				except:
					scan = None
				self.logger.info('Finished reading scan from %s', mdafilename)

		if beamline == 'DLS-I08':
			self.logger.info('beamline: %s', beamline)
			self.logger.info('reading DLS-I08 scan from /img.dat/*.h5')
			filenameh5 = os.path.basename(str(mdafilename))
			h5filename = os.path.join(os.path.join(self.main_dict['master_dir'], 'img.dat'), filenameh5)
			self.logger.debug('filename= %s', h5filename)
			h5 = maps_hdf5.h5(self.logger)
			scan = h5.read_scan(h5filename)
			self.save_h5 = 0
			self.xanes = 0

		return scan

	# ----------------------------------------------------------------------
	def generate_img_dat_threaded(self, header, mdafilename, this_detector, total_number_detectors, quick_dirty, nnls,
								no_processors_to_use, xrf_bin, xrf_bin_ext=''):

		info_elements = self.info_elements
		beamline = self.beamline
		make_maps_conf = self.maps_conf

		self.save_h5 = 1
		self.xanes = 0
		# assume by default this is not a fly scan based on netcdf
		self.netcdf_fly_scan = 0
		suffix = ''

		xrfflyscan = 0
		maps_overridefile = os.path.join(self.main_dict['master_dir'], 'maps_fit_parameters_override.txt')
		maps_intermediate_solution_file = 'maps_intermediate_solution.tmp'

		xmin = 0L
		xmax = 0L
		nnls = 0L

		#Look for override files in main.master_dir
		if total_number_detectors > 1:
			overide_files_found = 0
			suffix = str(this_detector)
			self.logger.debug('suff= %s', suffix)
			maps_overridefile = os.path.join(self.main_dict['master_dir'], 'maps_fit_parameters_override.txt') + suffix
			try:
				f = open(maps_overridefile, 'rt')
				self.logger.debug('maps override file %s exists',maps_overridefile)
				f.close()
			except :
				# if i cannot find an override file specific per detector, assuming
				# there is a single overall file.
				maps_overridefile = os.path.join(self.main_dict['master_dir'], 'maps_fit_parameters_override.txt')

		if xrf_bin > 0:
			xrf_bin_ext = '.avg3'

		self.version = make_maps_conf.version
		extra_pv = 0

		scan = self.read_and_parse_scan(header, mdafilename, beamline, this_detector, total_number_detectors)

		if scan == None:
			self.logger.error("Could not read and parse scan " + mdafilename)
			return

		extra_pv = scan.extra_pv

		# Get scan date
		scan_date = datetime.date(0001, 01, 01)
		month_list = ['jan', 'feb', 'mar', 'apr', 'mai', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
		if scan.scan_time_stamp != '':
			monthstr = scan.scan_time_stamp[0:3].lower()
			month = month_list.index(monthstr) + 1
			day = int(scan.scan_time_stamp[4:6])
			year = int(scan.scan_time_stamp[8:12])
			scan_date = datetime.date(year, month, day)

		# Read in detector calibration
		maps_detector.get_detector_calibration(make_maps_conf, beamline, info_elements, scan, maps_overridefile, self.logger)

		amp = np.zeros((8, 3), dtype=np.float)

		if scan.mca_calib_description_arr:
			for i in range(8):
				mca_calib = [self.crate, ':A', str(i + 1), 'sens_num.VAL']
				mca_calib = string.join(mca_calib, '')
				if mca_calib in scan.mca_calib_description_arr:
					amp[i, 0] = float(scan.mca_calib_arr[scan.mca_calib_description_arr.index(mca_calib)])

				mca_calib = [self.crate, ':A', str(i + 1), 'sens_unit.VAL']
				mca_calib = string.join(mca_calib, '')
				if mca_calib in scan.mca_calib_description_arr:
					amp[i, 1] = float(scan.mca_calib_arr[scan.mca_calib_description_arr.index(mca_calib)])

		# if all amp values are 0, it is likely, amps were not found. then try this
		# could not find amplifier sensitivity in data file. will try to look up
		# in a user editable file
		if amp.sum() == 0.0:
			try:
				f = open_file_with_retry(maps_overridefile, 'rt')

				for line in f:
					if ':' in line :
						slist = line.split(':')
						tag = slist[0]
						value = ''.join(slist[1:])

						if tag == 'US_AMP_SENS_NUM':
							amp[0, 0] = value
						elif tag == 'US_AMP_SENS_UNIT':
							amp[0, 1] = value
						elif tag == 'DS_AMP_SENS_NUM':
							amp[1, 0] = value
						elif tag == 'DS_AMP_SENS_UNIT':
							amp[1, 1] = value
				f.close()

				if beamline == '2-ID-D':
					amp[3, :] = amp[1, :]
					amp[1, :] = amp[0, :]
			except:
				self.logger.warning('Warning: amp[] is 0 - could not read override file')

		for i in range(8):
			if amp[i, 0] == 0.0: amp[i, 2] = 1.
			if amp[i, 0] == 1.0: amp[i, 2] = 2.
			if amp[i, 0] == 2.0: amp[i, 2] = 5.
			if amp[i, 0] == 3.0: amp[i, 2] = 10.
			if amp[i, 0] == 4.0: amp[i, 2] = 20.
			if amp[i, 0] == 5.0: amp[i, 2] = 50.
			if amp[i, 0] == 6.0: amp[i, 2] = 100.
			if amp[i, 0] == 7.0: amp[i, 2] = 200.
			if amp[i, 0] == 8.0: amp[i, 2] = 500.
			if amp[i, 1] == 0.0: amp[i, 2] = amp[i, 2] / 1000.0		 # pA/V
			if amp[i, 1] == 1.0: amp[i, 2] = amp[i, 2]				 # nA/V
			if amp[i, 1] == 2.0: amp[i, 2] = amp[i, 2] * 1000.0		 #uA/V
			if amp[i, 1] == 3.0: amp[i, 2] = amp[i, 2] * 1000.0 * 1000.0 #mA/V

		us_amp = np.zeros(3)
		ds_amp = np.zeros(3)

		if beamline == '2-ID-D':
			us_amp[:] = amp[1, :]
			ds_amp[:] = amp[3, :]

		if beamline == '2-ID-E':
			us_amp[:] = amp[0, :]
			ds_amp[:] = amp[1, :]

		if beamline == 'Bio-CAT':
			us_amp[:] = amp[0, :]
			ds_amp[:] = amp[1, :]

		try:
			if scan.mca_arr.size < 1:
				self.logger.warning( 'Skipping file : %s, was not able to read a valid 3D array', scan.scan_name)
				return
		except:
			self.logger.warning('skipping file : %s, was not able to read a valid 3D array', scan.scan_name)
			return

		if scan.mca_arr.sum() == 0:
			self.logger.warning('skipping file : %s, contains 3D array with all elements EQ 0', scan.scan_name)
			return

		mca_arr_dimensions = scan.mca_arr.shape
		n_cols = mca_arr_dimensions[0]
		n_rows = mca_arr_dimensions[1]
		n_channels = 2048
		n_mca_channels = mca_arr_dimensions[2]

		if total_number_detectors > 1 or (total_number_detectors == 1 and len(scan.mca_arr.shape) == 4):
			if (self.netcdf_fly_scan != 1) and (beamline != 'GSE-CARS') and (beamline != 'Bio-CAT'):
				# if it is a fly scan, ie netcdf_fly_scan is 1, the even a multi
				# element detector scan is read in as a single element
				old_mca_arr = scan.mca_arr.copy()
				scan.mca_arr = np.zeros((n_cols, n_rows, n_mca_channels))
				scan.mca_arr[:, :, :] = old_mca_arr[:, :, :, this_detector]
				del old_mca_arr
				mca_arr_dimensions = scan.mca_arr.shape

		# IF quick_dirty is set, just sum up all detector elements and treat
		# them as a single detector, to speed initial analysis up
		if quick_dirty > 0:
			if self.netcdf_fly_scan != 1:
				# if it is a fly scan, ie netcdf_fly_scan is 1, the even a multi
				# element detector scan is read in as a single element
				scan_shape = scan.mca_arr.shape
				old_mca_arr = scan.mca_arr.copy()
				scan.mca_arr = np.zeros((n_cols, n_rows, scan_shape[2]))
				old_mca_no_dets = old_mca_arr.shape
				if len(old_mca_no_dets) == 4 :
					old_mca_no_dets = old_mca_no_dets[3]
					for ii in range(old_mca_no_dets):
						scan.mca_arr[:, :, :] = scan.mca_arr[:, :, :] + old_mca_arr[:, :, :, ii]

				del old_mca_arr
				mca_arr_dimensions = scan.mca_arr.shape

		#dataset = np.zeros((n_cols, n_rows, make_maps_conf.n_used_chan, 3))

		# These are n_mca_channels - why n < 20?
		if len(mca_arr_dimensions) == 4:
			for n in range(mca_arr_dimensions[3]):
				if n < 20 :
					if make_maps_conf.use_det[n] == 0:
						scan.mca_arr[:, :, :, n] = 0

		if len(mca_arr_dimensions) == 4:
			no_detectors = mca_arr_dimensions[3]
		else:
			no_detectors = 1

		h5 = maps_hdf5.h5(self.logger)
		if self.save_h5 == 1:
			# Save full spectra to HDF5 file
			h5file = os.path.join(self.main_dict['img_dat_dir'], header + xrf_bin_ext + '.h5' + suffix)
			self.logger.info('now trying to write the mca spectra into the HDF5 file %s', h5file)
			h5.write_mca_hdf5(h5file, scan.mca_arr)

		max_chan_spec = np.zeros((n_channels, 5))

		if no_detectors > 1:
			for kk in range(n_mca_channels):
				temp = scan.mca_arr[:, :, kk, :].flatten()
				sortind = temp.argsort()
				sortind = sortind[::-1]
				max_chan_spec[kk, 1] = np.sum(temp[sortind[0:np.amin([11, sortind.size])]])
				max_chan_spec[kk, 0] = np.amax(temp)
				del temp
		else:
			for kk in range(n_mca_channels):
				temp = scan.mca_arr[:, :, kk].flatten()
				sortind = temp.argsort()
				sortind = sortind[::-1]
				max_chan_spec[kk, 1] = np.sum(temp[sortind[0:np.amin([11, sortind.size])]])
				max_chan_spec[kk, 0] = np.amax(temp)
				del temp

		temp = 0
		raw_spec = scan.mca_arr.sum(axis=0)
		raw_spec = raw_spec.sum(axis=0)
		if no_detectors > 1:
			spec_all = raw_spec.sum(axis=1)
		else:
			spec_all = raw_spec

		if no_detectors > make_maps_conf.use_det.sum():
			no_detectors = make_maps_conf.use_det.sum()

		dataset_size = 3

		thisdata = self.maps_def.define_xrfmaps_info(scan.x_pixels, scan.y_pixels, dataset_size,
													n_channels, n_channels, no_detectors,
													make_maps_conf.n_used_chan,
													make_maps_conf.n_used_dmaps,
													make_maps_conf,
													version = 9)

		if scan.extra_pv:
			self.maps_def.xrfmaps_add_extra(scan.extra_pv, extra_pv_order = scan.extra_pv_key_list)

		wo = np.where(make_maps_conf.use_det == 1)
		self.logger.debug('wo len %s raw_spec len %s no_detectors %s', len(wo), len(thisdata.raw_spec), no_detectors)
		for ii in range(no_detectors-1):
			thisdata.raw_spec[0:len(spec_all), ii] = raw_spec[:, wo[ii]]

		if beamline == '2-BM':
			det_descr = ['srcurrent', 'us_ic', 'ds_ic', 'ELT1', 'ERT1']

		if beamline =='2-ID-B':
			det_descr = ['srcurrent', 'us_ic', 'ds_ic', 'dpc1_ic', 'dpc2_ic',
						'cfg_1', 'cfg_2', 'cfg_3', 'cfg_4', 'cfg_5', 'cfg_6', 'cfg_7', 'cfg_8',
						'cfg_9', 'ELT1', 'ERT1']

		if beamline == '2-ID-D':
			det_descr = ['srcurrent', 'us_ic', 'ds_ic', 'ELT1', 'ERT1']
			if scan_date > datetime.date(2009, 9, 01) :
				det_descr = ['srcurrent', 'us_ic', 'ds_ic', 'ELT1', 'ERT1', 'ICR1', 'OCR1']
			if scan_date > datetime.date(2009, 01, 01) :
				scan_date = ['srcurrent', 'us_ic', 'ds_ic',
							'cfg_1', 'cfg_2', 'cfg_3', 'cfg_4', 'cfg_5', 'cfg_6', 'cfg_7', 'cfg_8',
							'cfg_9', 'cfg_10', 'ELT1', 'ERT1', 'ICR1', 'OCR1']

		if beamline == '2-ID-E':
			det_descr = ['srcurrent', 'us_ic', 'ds_ic', 'dpc1_ic', 'dpc2_ic',
						'cfg_1', 'cfg_2', 'cfg_3', 'cfg_4', 'cfg_5', 'cfg_6', 'cfg_7', 'cfg_8',
						'ELT1', 'ERT1', 'ELT2', 'ERT2', 'ELT3', 'ERT3']
			if scan_date > datetime.date(2007, 9, 01):
				det_descr = ['srcurrent', 'us_ic', 'ds_ic', 'dpc1_ic', 'dpc2_ic',
							'cfg_1', 'cfg_2', 'cfg_3', 'cfg_4', 'cfg_5', 'cfg_6', 'cfg_7', 'cfg_8',
							'cfg_9', 'ELT1', 'ERT1', 'ELT2', 'ERT2', 'ELT3', 'ERT3', 'ICR1', 'OCR1']

		if beamline == 'Bio-CAT':
			det_descr = ['srcurrent', 'us_ic', 'ds_ic', 'ELT1', 'ERT1']

		if beamline == 'GSE-CARS':
			det_descr = ['srcurrent', 'us_ic', 'ds_ic', 'ELT1', 'ERT1']

		if beamline == 'Bionanoprobe':
			det_descr = ['srcurrent', 'us_ic', 'ds_ic',
						'cfg_1', 'cfg_2', 'cfg_3', 'cfg_4', 'ELT1', 'ERT1', 'ICR1', 'OCR1']

		if (beamline == 'DLS-I08'):
			det_descr = []

		dmaps_set = np.zeros((n_cols, n_rows, make_maps_conf.n_used_dmaps))

		# generate direct maps, such as SR current, ICs, life time in subroutine
		dmaps_set = maps_detector.find_detector_name(det_descr, scan_date, scan.detector_arr, scan.detector_description_arr,
																 make_maps_conf, scan.x_coord_arr, scan.y_coord_arr, beamline,
																 n_cols, n_rows, maps_overridefile, self.logger)


		dmaps_names = []
		for item in make_maps_conf.dmaps:
			dmaps_names.append(item.name)

		# elt1_ = dmaps_set[:, :, dmaps_names.index('ELT1')]
		ert1_ = dmaps_set[:, :, dmaps_names.index('ERT1')]
		icr1_ = dmaps_set[:, :, dmaps_names.index('ICR1')]
		ocr1_ = dmaps_set[:, :, dmaps_names.index('OCR1')]
		if ert1_.sum() > 0 and icr1_.sum() > 0 and ocr1_.sum() > 0:
			#dmaps_set[:, :, dmaps_names.index('ELT1')] = dmaps_set[:, :, dmaps_names.index('ERT1')] * dmaps_set[:, :, dmaps_names.index('OCR1')] / dmaps_set[:, :, dmaps_names.index('ICR1')]
			elt1_ = ert1_ * ocr1_ / icr1_
			dmaps_set[:, :, dmaps_names.index('ELT1')] = elt1_[:]
		#	# ICR = Input Counts/Trigger Filter Livetime OCR = Output Counts / Real Time Energy Filter Livetime = Real Time * OCR/ICR.

		elt1_arr = []
		if 'ELT1' in dmaps_names:
			elt1_arr = dmaps_set[:, :, dmaps_names.index('ELT1')]
		elt1_arr = np.array(elt1_arr)
		if np.sum(elt1_arr) == 0.0:
			self.logger.warning('WARNING: did not find elapsed life time. Will continue assuming ELT1 was 1s, but this is just an ARBITRARY value')
			elt1_arr[:, :] = 1.

		elt2_arr = []
		if 'ELT2' in dmaps_names:
			elt2_arr = dmaps_set[:, :, dmaps_names.index('ELT2')]
			elt2_arr = np.array(elt2_arr)
			if np.sum(elt2_arr) == 0.0:
				self.logger.warning('WARNING: did not find elapsed life time. Will continue assuming ELT2 was 1s, but this is just an ARBITRARY value')
				elt2_arr[:, :] = 1.

		elt3_arr = []
		if 'ELT3' in dmaps_names:
			elt3_arr = dmaps_set[:, :, dmaps_names.index('ELT3')]
			elt3_arr = np.array(elt3_arr)
			if np.sum(elt3_arr) == 0.0:
				self.logger.warning('WARNING: did not find elapsed life time. Will continue assuming ELT3 was 1s, but this is just an ARBITRARY value')
				elt3_arr[:, :] = 1.

		# Bin the data if required
		if xrf_bin > 0:
			elt1_arr = self.__binning__(scan, xrf_bin, n_cols, n_rows, mca_arr_dimensions, elt1_arr)

		if 'ELT2' in dmaps_names:
			elt2_arr = dmaps_set[:, :, dmaps_names.index('ELT2')]
		if 'ELT3' in dmaps_names:
			elt3_arr = dmaps_set[:, :, dmaps_names.index('ELT3')]

		# calculate elemental maps using XRF
		temp_elementsuse = []
		for item in make_maps_conf.chan:
			temp_elementsuse.append(item.use)
		elements_to_use = np.where(np.array(temp_elementsuse) == 1)
		elements_to_use = elements_to_use[0]

		if elements_to_use.size == 0:
			return
		spectra = self.maps_def.define_spectra(self.max_spec_channels, self.main_max_spectra, self.max_ICs, mode='plot_spec')

		fp = maps_fit_parameters.maps_fit_parameters(self.logger)
		fitp = fp.define_fitp(beamline, info_elements)

		text = ' spec_name, inner_coord, outer_coord, '
		for i in range(fitp.g.n_fitp):
			text = text + str(fitp.s.name[i]) + ', '
		text = text + ' live_time, ' + ' total_counts, ' + ' status, ' + ' niter, ' + \
			' total_perror, ' + ' abs_error, ' + ' relative_error, ' + 'fit_time_per_pix, ' + \
			' srcurrent, ' + ' us_ic, ' + ' ds_ic, '
		for i in range(np.amin(fitp.keywords.kele_pos), np.amax(fitp.keywords.mele_pos)):
			text = text + 'perror_' + str(fitp.s.name[i]) + ', '

		roi_start = datetime.datetime.now()
		# below is the routine for straight ROI mapping
		for jj in range(len(elements_to_use)):
			counts = 0.
			for kk in range(no_detectors):
				if kk == 0 : elt_arr = elt1_arr
				if kk == 1 : elt_arr = elt2_arr
				if kk == 2 : elt_arr = elt3_arr

				wo = elements_to_use[jj]
				# note: center position for peaks/rois is in keV, widths of ROIs is in eV
				left_roi = int(((make_maps_conf.chan[wo].center-make_maps_conf.chan[wo].width/2./1000.) - make_maps_conf.calibration.offset[kk])/make_maps_conf.calibration.slope[kk])
				right_roi = int(((make_maps_conf.chan[wo].center+make_maps_conf.chan[wo].width/2./1000.) - make_maps_conf.calibration.offset[kk])/make_maps_conf.calibration.slope[kk])

				if right_roi >= n_mca_channels:
					right_roi = n_mca_channels - 2
				if left_roi > right_roi:
					left_roi = right_roi - 1
				if left_roi < 0:
					left_roi = 1
				if right_roi < 0:
					right_roi = n_mca_channels - 2

				make_maps_conf.chan[wo].left_roi[kk] = left_roi
				make_maps_conf.chan[wo].right_roi[kk] = right_roi

				roi_width = right_roi - left_roi + 1

				if no_detectors == 1:
					if (roi_width < 200) or (n_cols < 50) or (n_rows < 50)	:
						# note: when looking for tfy, or other elements with a large
						# number of channels (big difference between left and right
						# roi, the line below can gobble up a lot of memory (eg, a
						# total of 1/3 of content.
						these_counts = scan.mca_arr[:, :, left_roi:right_roi + 1]
					else:
						these_counts = scan.mca_arr[:, :, left_roi]
						for ii_temp in range(left_roi + 1, right_roi  + 1):
							these_counts = these_counts + scan.mca_arr[:, :, ii_temp]
				else:
					if (roi_width < 200) or (n_cols < 50) or (n_rows < 50)	:
						# note: when looking for tfy, or other elements with a large
						# number of channels (big difference between left and right
						# roi, the line below can gobble up a lot of memory (eg, a
						# total of 1/3 of content.
						these_counts = scan.mca_arr[:, :, left_roi:right_roi + 1, kk]
					else:
						these_counts = scan.mca_arr[:, :, left_roi, kk]
						for ii_temp in range(left_roi + 1, right_roi + 1):
							these_counts = these_counts + scan.mca_arr[:, :, ii_temp, kk]

				if len(these_counts.shape) >= 3:
					these_counts = these_counts.sum(axis=2)
				these_counts = these_counts / elt_arr
				counts = counts + these_counts
			thisdata.dataset_orig[:, :, jj, 0] = counts

		roi_end = datetime.datetime.now()
		total_roi_time = roi_end - roi_start
		total_time_str = '\n\n %%%%%%%% roi total time = '+str(total_roi_time.total_seconds())+'  %%%%%%% \n\n'
		self.logger.info(total_time_str)

		# below is the routine for using matrix math to calculate elemental
		# content with overlap removal
		self.logger.info('now using matrix math for analysis')

		kk = 0
		#element_pos = [fitp.keywords.kele_pos, fitp.keywords.lele_pos, fitp.keywords.mele_pos]
		det = kk
		pileup_string = ''
		test_string = ''
		fitp, test_string, pileup_string = fp.read_fitp(maps_overridefile, info_elements, det)
		for jj in range(fitp.g.n_fitp) :
			if fitp.s.name[jj] in test_string :
				fitp.s.val[jj] = 1.
				fitp.s.use[jj] = 5
			else:
				fitp.s.use[jj] = 1

		n_pars = fitp.g.n_fitp
		parinfo_value = np.zeros((n_pars))
		parinfo_fixed = np.zeros((n_pars), dtype=np.int)
		parinfo_limited = np.zeros((n_pars, 2), dtype=np.int)
		parinfo_limits = np.zeros((n_pars, 2))
		parinfo_relstep = np.zeros((n_pars))
		parinfo_mpmaxstep = np.zeros((n_pars))
		parinfo_mpminstep = np.zeros((n_pars))

		for i in range(n_pars) :
			parinfo_value[i] = float(fitp.s.val[i])
			if fitp.s.use[i] == 1 :
				parinfo_fixed[i] = 1
			else:
				parinfo_fixed[i] = 0
			wo = np.where(fitp.keywords.peaks == i)
			if wo[0].size > 0 :
				parinfo_value[i] = np.log10(fitp.s.val[i])

		thisdata.dataset_orig[:, :, 0, 2] = 0.
		x = np.arange(float(n_channels))

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

		wo_use_this_par = (np.nonzero(fitp.keywords.use_this_par[0:(np.max(fitp.keywords.mele_pos)-np.min(fitp.keywords.kele_pos)+1)] == 1))[0]

		no_use_pars = wo_use_this_par.size+2

		sol_intermediate = np.zeros((no_use_pars, self.main_dict['max_spec_channels']))
		fitmatrix_reduced = np.zeros((self.main_dict['max_spec_channels'], no_use_pars))

		# Read in intermediate solution
		filepath = os.path.join(self.main_dict['output_dir'], maps_intermediate_solution_file) + suffix
		#saveddatafile = np.load(filepath)
		saveddatafile = call_function_with_retry(np.load, 5, 0.1, 1.1, (filepath,))
		if saveddatafile == None:
			self.logger.error('Error opening %s', filepath)
		else:
			sol_intermediate = saveddatafile['sol_intermediate']
			fitmatrix_reduced = saveddatafile['fitmatrix_reduced']
			saveddatafile.close()

		self.logger.info('elements to use as per make_maps_conf')
		maps_conf_chan_elstouse_names = []
		for iel in range(len(elements_to_use)):
			maps_conf_chan_elstouse_names.append(make_maps_conf.chan[elements_to_use[iel]].name)
		self.logger.debug('maps_conf_chan_elstouse_names %s', maps_conf_chan_elstouse_names)

		temp_fitp_use = fitp.s.use[np.amin(fitp.keywords.kele_pos):np.amax(fitp.keywords.mele_pos)+1]
		temp_fitp_name = fitp.s.name[np.amin(fitp.keywords.kele_pos):np.amax(fitp.keywords.mele_pos)+1]
		which_elements_to_fit = (np.nonzero(temp_fitp_use != 1))[0]
		self.logger.info('elements to fit as per fitp: %s', temp_fitp_name[which_elements_to_fit])

		element_lookup_in_reduced = np.zeros((len(elements_to_use)), dtype=int)
		element_lookup_in_reduced[:] = -1
		j_temp = 0
		for i_temp in range(len(elements_to_use)):
			if make_maps_conf.chan[elements_to_use[i_temp]].name in fitp.s.name[min(fitp.keywords.kele_pos) + which_elements_to_fit]:
				wo_temp = np.where(fitp.s.name[np.amin(fitp.keywords.kele_pos)+which_elements_to_fit] == make_maps_conf.chan[elements_to_use[i_temp]].name)
				element_lookup_in_reduced[i_temp] = wo_temp[0]
			if 's_i' == make_maps_conf.chan[elements_to_use[i_temp]].name:
				element_lookup_in_reduced[i_temp] = len(wo_use_this_par) + 1

			if 's_e' == make_maps_conf.chan[elements_to_use[i_temp]].name:
				element_lookup_in_reduced[i_temp] = len(wo_use_this_par) + 0

		self.logger.debug('c %s', tm.time())

		if nnls == 0:
			fitting_start = datetime.datetime.now()
			for i_temp in range(n_cols):
				for j_temp in range(n_rows):
					these_counts = np.zeros((len(x)))
					n_relev_channels = min(len(x), n_mca_channels)
					these_counts[0:n_relev_channels] = scan.mca_arr[i_temp, j_temp, 0:n_relev_channels]
					# mca_arr is the 4d array, where pixel_x, pixel_y,spectrum at
					# this point (eg 2000), adetector number (legacy), typically 1
					solution = np.dot(sol_intermediate[:, 0:len(x)], these_counts)
					solution = solution/elt_arr[i_temp, j_temp]
					for mm in range(len(elements_to_use)):
						if element_lookup_in_reduced[mm] != -1:
							thisdata.dataset_orig[i_temp, j_temp, mm, 2] = solution[element_lookup_in_reduced[mm]]
			fitting_end = datetime.datetime.now()
			total_fitting_time = fitting_end - fitting_start
			total_time_str = '\n\n %%%%%%%% SVD total time = '+str(total_fitting_time.total_seconds())+'  %%%%%%% \n\n'
			self.logger.info(total_time_str)
		if nnls > 0:
			results_line = np.zeros((n_rows, len(elements_to_use)))
			if (no_processors_to_use > 1):

				self.logger.info('no_processors_to_use = %s', no_processors_to_use)
				self.logger.info('cpu_count() = %d\n', multiprocessing.cpu_count())
				self.logger.info('Creating pool with %d processes\n', no_processors_to_use)
				pool = multiprocessing.Pool(no_processors_to_use)

				count = n_cols
				data_lines = np.zeros((self.main_dict['max_spec_channels'],	n_rows, n_cols))
				for i_fit in range(count):
					for jj in range(n_rows):
						data_lines[0:scan.mca_arr[i_fit, jj, :].size, jj, i_fit] = scan.mca_arr[i_fit, jj, :]

					# Single processor version for debugging
					#  for i_fit in range(count):
					#	  self.logger.debug( 'Doing line ', i_fit, ' of ', count
					#	  results_line = maps_tools.maps_nnls_line(data_lines[:, :, i_fit], n_channels, fitmatrix_reduced, n_mca_channels,
					#											   elements_to_use, element_lookup_in_reduced, n_rows)

				results_pool = [pool.apply_async(maps_tools.maps_nnls_line, (data_lines[:, :, i_fit], n_channels, fitmatrix_reduced, n_mca_channels,
												elements_to_use, element_lookup_in_reduced, n_rows)) for i_fit in range(count)]

				self.logger.info('Ordered results using pool.apply_async():')
				results = []
				for r in results_pool:
					results.append(r.get())

				pool.terminate()
				pool.join()

				results = np.array(results)

				for iline in range(count):
					results_line = results[iline, :, :]
					for mm in range(len(elements_to_use)):
						if element_lookup_in_reduced[mm] != -1:
							thisdata.dataset_orig[iline, :, mm, 2] = results_line[:, mm]

			else:
				data_line = np.zeros((self.main_dict['max_spec_channels'], n_rows))
				count = n_cols
				for i_fit in range(count):
					data_line[:, :] = 0.
					for jj in range(n_rows):
						data_line[0:scan.mca_arr[i_fit, jj, :].size, jj] = scan.mca_arr[i_fit, jj, :]
					results_line = maps_tools.maps_nnls_line(data_line, n_channels, fitmatrix_reduced, n_mca_channels,
															elements_to_use, element_lookup_in_reduced, n_rows)
					for mm in range(len(elements_to_use)):
						if element_lookup_in_reduced[mm] != -1:
							thisdata.dataset_orig[i_fit, :, mm, 2] = results_line[:, mm]

		self.logger.debug('d %s', tm.time())

		if ('s_a' in maps_conf_chan_elstouse_names) and \
			('s_i' in maps_conf_chan_elstouse_names) and \
			('s_e' in maps_conf_chan_elstouse_names) :
			wo = maps_conf_chan_elstouse_names.index('s_a')
			wo_i = maps_conf_chan_elstouse_names.index('s_i')
			wo_e = maps_conf_chan_elstouse_names.index('s_e')
			thisdata.dataset_orig[:, :, wo, 2] = thisdata.dataset_orig[:, :, wo_i, 2] + thisdata.dataset_orig[:, :, wo_e, 2]

		if 'TFY' in maps_conf_chan_elstouse_names:
			wo_tfy = maps_conf_chan_elstouse_names.index('TFY')
			thisdata.dataset_orig[:, :, wo_tfy, 2] = thisdata.dataset_orig[:, :, wo_tfy, 0]

		fitp = fp.define_fitp(beamline, info_elements)

		self.logger.debug('make_maps_conf.use_fit = %s', make_maps_conf.use_fit)

		# Spectrum fitting goes here if enabled
		if (make_maps_conf.use_fit > 0) or (self.pca > 0):

			spectral_binning = 2
			if spectral_binning > 0:
				t = scan.mca_arr.shape
				mca_arr_dim = len(t)

				# the statement below is the one that causes trouble with large arrays in IDL
				if mca_arr_dim == 3:
					scan.mca_arr = rebin(scan.mca_arr[:, :, 0:int(t[2] / spectral_binning)*spectral_binning], (t[0], t[1], int(t[2] / spectral_binning)))
				if mca_arr_dim == 4:
					scan.mca_arr = rebin(scan.mca_arr[:, :, 0:int(t[2] / spectral_binning)*spectral_binning, :], (t[0], t[2], int(t[2] / spectral_binning), t[3]))

				mca_arr_dimensions = scan.mca_arr.shape
				n_cols = mca_arr_dimensions[0]
				n_rows = mca_arr_dimensions[1]
				n_channels = mca_arr_dimensions[2]

		keywords = fitp.keywords

		if (make_maps_conf.use_fit > 0) and (xrfflyscan == 0):

			fit = maps_analyze.analyze(self.logger)

			seconds_fit_start = tm.time()
			# note: spectral binning needs to be even !!
			#data_temp = np.zeros((self.max_spec_channels))
			data_line = np.zeros((self.max_spec_channels,  n_rows))
			fitted_line = np.zeros((self.max_spec_channels, n_rows))
			ka_line = np.zeros((self.max_spec_channels, n_rows))
			l_line = np.zeros((self.max_spec_channels, n_rows))
			bkground_line = np.zeros((self.max_spec_channels, n_rows))
			'''
			fitted_line2 = np.zeros((self.max_spec_channels, n_rows))
			ka_line2 = np.zeros((self.max_spec_channels, n_rows))
			l_line2 = np.zeros((self.max_spec_channels, n_rows))
			bkground_line2 = np.zeros((self.max_spec_channels, n_rows))
			'''
			fitted_temp = np.zeros((self.max_spec_channels, no_detectors + 1))
			Ka_temp = np.zeros((self.max_spec_channels, no_detectors + 1))
			l_temp = np.zeros((self.max_spec_channels, no_detectors + 1))
			bkground_temp = np.zeros((self.max_spec_channels, no_detectors + 1))
			raw_temp = np.zeros((self.max_spec_channels, no_detectors + 1))

			add_plot_spectra = np.zeros((self.max_spec_channels, 12, n_rows), dtype=np.float32)
			#temp_add_plot_spectra = np.zeros((self.max_spec_channels, 12, n_rows), dtype=np.float32)
			add_plot_names = ['fitted', 'K alpha', 'background', 'K beta', 'L lines', 'M lines', 'step', 'tail', 'elastic', 'compton', 'pileup', 'escape']

			values = np.zeros((n_cols, n_rows, fitp.g.n_fitp), dtype=np.float32)
			values_line = np.zeros((n_rows, fitp.g.n_fitp), dtype=np.float32)
			bkgnd = np.zeros((n_cols, n_rows), dtype=np.float32)
			bkgnd_line = np.zeros((n_rows), dtype=np.float32)
			tfy = np.zeros((n_cols, n_rows), dtype=np.float32)
			tfy_line = np.zeros((n_rows), dtype=np.float32)
			#sigma = np.zeros((n_cols, n_rows, fitp.g.n_fitp), dtype=np.float32)
			elt_line = np.zeros((n_rows), dtype=np.float32)
			'''
			values_line2 = np.zeros((n_rows, fitp.g.n_fitp), dtype=np.float32)
			bkgnd_line2 = np.zeros((n_rows), dtype=np.float32)
			tfy_line2 = np.zeros((n_rows), dtype=np.float32)
			'''
			#test = np.zeros(self.max_spec_channels)

			which_dets_to_use = np.where(make_maps_conf.use_det == 1)
			for bb in range(no_detectors):
				kk = which_dets_to_use[0][bb]

				if kk == 0:
					elt_arr = elt1_arr
				if kk == 1:
					elt_arr = elt2_arr
				if kk == 2:
					elt_arr = elt3_arr

				matrix = 1
				temp = 0
				if NO_MATRIX:
					matrix = 0
					temp = 1

				fitp.g.no_iters = 4

				fitp.s.use[:] = 1

				fitp.s.val[min(fitp.keywords.kele_pos):max(fitp.keywords.mele_pos)-1] = 1e-10
				# execute below if do fixed fit per pixel
				if make_maps_conf.use_fit == 1 :
					for j in range(fitp.keywords.kele_pos[0]):
						fitp.s.use[j] = fitp.s.batch[j,1]
					# if matrix is not 1, then global variable NO_MATRIX is used to
					# override, and will keep energy calibration floating. at every pixel
					if matrix == 0 :
						for j in range(fitp.keywords.kele_pos[0]):
							fitp.s.use[j] = fitp.s.batch[j,4]
					det = kk
					pileup_string = ''
					test_string = ''
					self.logger.debug('maps_overridefile %s', maps_overridefile)
					#for ie in range(len(info_elements)): self.logger.debug( info_elements[ie].xrf_abs_yield
					fitp, test_string, pileup_string = fp.read_fitp(maps_overridefile, info_elements, det)

					for jj in range(fitp.g.n_fitp):
						if fitp.s.name[jj] in test_string :
							fitp.s.val[jj] = 1.
							fitp.s.use[jj] = 5
							if temp == 0:
								temp = jj
							else:
								temp = [temp, jj]

					calib = {'off':0., 'lin':0., 'quad':0.}
					calib['off'] = fitp.s.val[fitp.keywords.energy_pos[0]]
					calib['lin'] = fitp.s.val[fitp.keywords.energy_pos[1]]
					calib['quad'] = fitp.s.val[fitp.keywords.energy_pos[2]]

					fp.parse_pileupdef(fitp, pileup_string, info_elements)

				add_matrixfit_pars = np.zeros((6))
				add_matrixfit_pars[0] = fitp.s.val[fitp.keywords.energy_pos[0]]
				add_matrixfit_pars[1] = fitp.s.val[fitp.keywords.energy_pos[1]]
				add_matrixfit_pars[2] = fitp.s.val[fitp.keywords.energy_pos[2]]
				add_matrixfit_pars[3] = fitp.s.val[fitp.keywords.added_params[1]]
				add_matrixfit_pars[4] = fitp.s.val[fitp.keywords.added_params[2]]
				add_matrixfit_pars[5] = fitp.s.val[fitp.keywords.added_params[3]]

				#if len(which_par_str) : text = [which_par_str, text]
				old_fitp = fp.define_fitp(beamline, info_elements)
				old_fitp.s.val[:]=fitp.s.val[:]

				if no_processors_to_use > 1:
					self.logger.info('Multi-threaded fitting started')
					self.logger.info('no_processors_to_use = %s', no_processors_to_use)
					self.logger.info('cpu_count() = %s\n', multiprocessing.cpu_count())
					self.logger.info('Creating pool with %s processes\n', no_processors_to_use)
					pool = multiprocessing.Pool(no_processors_to_use)

					count = n_cols

					data_lines = np.zeros((self.main_dict['max_spec_channels'],	n_rows, n_cols))
					for i_fit in range(n_cols):
						for jj in range(n_rows):
							data_lines[0:scan.mca_arr[i_fit, jj, :].size, jj, i_fit] = scan.mca_arr[i_fit, jj, :]

					self.logger.info('Started fitting')

					fitting_start = datetime.datetime.now()
					results_pool = []
					start = 0
					for i_fit in range(count):
						elt_line[:] = elt1_arr[i_fit, :]

						fitp.s.val[:]=old_fitp.s.val[:]

						if (xrf_bin > 0) and (i_fit < count -2) :
							if (xrf_bin == 2) and (n_cols > 5) and (n_rows > 5) :
								if i_fit % 2 != 0 :
									continue

							if (xrf_bin == 4) and (n_cols > 5) and (n_rows > 5) :
								if i_fit % 3 != 0:
									continue

						for jj in range(n_rows):
							raw_temp[:, kk] = raw_temp[:, kk] + data_lines[:, jj, i_fit]

						results_pool.append(pool.apply_async(fit_line_threaded, (self.logger.name, i_fit, data_lines[:,:,i_fit],
										n_rows, matrix, spectral_binning, elt1_arr[i_fit, :], fitp, old_fitp, keywords, xrf_bin, calib)) )
					#self.logger.info( '------ Waiting for fitting to finish ------')
					#del data_lines
					pool.close()
					pool.join()
					results = []
					for r in results_pool:
						results.append(r.get())

					fitting_end = datetime.datetime.now()
					total_fitting_time = fitting_end - fitting_start
					total_time_str = '\n\n %%%%%%%% fitting total time = '+str(total_fitting_time.total_seconds())+'  %%%%%%% \n\n'
					self.logger.info(total_time_str)

					for iline in range(count):
						results_line = results[iline]
						#self.logger.info( 'results_line=', results_line)
						fitted_line[...] = results_line[0][...]
						ka_line[...] = results_line[1][...]
						l_line[...] = results_line[2][...]
						bkground_line[...] = results_line[3][...]
						values_line[...] = results_line[4][...]
						bkgnd_line[...] = results_line[5][...]
						tfy_line[...] = results_line[6][...]
						xmin = results_line[7]
						xmax = results_line[8]
						values[start+iline, :, :] = values_line[:, :]
						bkgnd[start+iline, :] = bkgnd_line[:]
						tfy[start+iline, :] = tfy_line[:]

						if fitted_line == None:
							continue

						for jj in range(n_rows):
							fitted_temp[xmin:xmax+1, kk] = fitted_temp[xmin:xmax+1, kk] + fitted_line[xmin:xmax+1, jj]
							Ka_temp[xmin:xmax+1, kk] = Ka_temp[xmin:xmax+1, kk] + ka_line[xmin:xmax+1, jj]
							l_temp[xmin:xmax+1, kk] = l_temp[xmin:xmax+1, kk] + l_line[xmin:xmax+1, jj]
							bkground_temp[xmin:xmax+1, kk] = bkground_temp[xmin:xmax+1, kk] + bkground_line[xmin:xmax+1, jj]

						#import matplotlib.pyplot as plt
						#plt.plot(range(0,2048), ka_line[:,0])
						##plt.semilogy(range(70,500), ka_line[70:500])
						#plt.show()
						#exit(1)

					self.logger.debug('before %s', thisdata.energy_fit[0, kk])

					thisdata.energy_fit[0, kk] = calib['off']
					thisdata.energy_fit[1, kk] = calib['lin']
					thisdata.energy_fit[2, kk] = calib['quad']

					self.logger.debug('after %s', thisdata.energy_fit[0, kk])

				else:
					count = n_cols
					fitting_start = datetime.datetime.now()
					for i_fit in range(count):

						self.logger.info('fitting row number %s of %s', i_fit, count)

						if (xrf_bin > 0) and (i_fit < count -2) :
							if (xrf_bin == 2) and (n_cols > 5) and (n_rows > 5) :
								if i_fit % 2 != 0 :
									continue

							if (xrf_bin == 4) and (n_cols > 5) and (n_rows > 5) :
								if i_fit % 3 != 0:
									continue

						data_line[:, :] = 0.

						for jj in range(n_rows):
							data_line[0:scan.mca_arr[i_fit, jj, :].size, jj] = scan.mca_arr[i_fit, jj, :]
						elt_line[:] = elt1_arr[i_fit, :]

						fitp.s.val[:]=old_fitp.s.val[:]

						fitted_line, ka_line, l_line, bkground_line, values_line, bkgnd_line, tfy_line, xmin, xmax = fit.fit_line(data_line,
												n_rows, matrix, spectral_binning, elt_line, fitp, old_fitp, keywords, xrf_bin, calib)

						if fitted_line == None:
							continue

						for jj in range(n_rows):
							fitted_temp[xmin:xmax + 1, kk] = fitted_temp[xmin:xmax + 1, kk] + fitted_line[xmin:xmax + 1, jj]
							Ka_temp[xmin:xmax + 1, kk] = Ka_temp[xmin:xmax + 1, kk] + ka_line[xmin:xmax + 1, jj]
							l_temp[xmin:xmax + 1, kk] = l_temp[xmin:xmax + 1, kk] + l_line[xmin:xmax + 1, jj]
							bkground_temp[xmin:xmax + 1, kk] = bkground_temp[xmin:xmax + 1, kk] + bkground_line[xmin:xmax + 1, jj]
							raw_temp[:, kk] = raw_temp[:, kk] + data_line[:, jj]

						values[i_fit, :, :] = values_line[:, :]
						bkgnd[i_fit, :] = bkgnd_line[:]
						tfy[i_fit, :] = tfy_line[:]

					fitting_end = datetime.datetime.now()
					total_fitting_time = fitting_end - fitting_start
					total_time_str = '\n\n %%%%%%%% fitting total time = '+str(total_fitting_time.total_seconds())+'  %%%%%%% \n\n'
					self.logger.info(total_time_str)

					self.logger.debug('before %s', thisdata.energy_fit[0, kk])

					thisdata.energy_fit[0, kk] = calib['off']
					thisdata.energy_fit[1, kk] = calib['lin']
					thisdata.energy_fit[2, kk] = calib['quad']

					self.logger.info('after %s', thisdata.energy_fit[0, kk])

			for i_fit in range(n_cols):
				for j_fit in range(n_rows):
					for jj in range(len(elements_to_use)):
						if make_maps_conf.chan[elements_to_use[jj]].name in fitp.s.name:
							wo= np.where( fitp.s.name == make_maps_conf.chan[elements_to_use[jj]].name)[0]
						else:
							if make_maps_conf.chan[elements_to_use[jj]].name == 's_e':
								wo = np.where(fitp.s.name == 'coherent_sct_amplitude')[0]
							if make_maps_conf.chan[elements_to_use[jj]].name == 's_i':
								wo = np.where(fitp.s.name == 'compton_amplitude')[0]
							if make_maps_conf.chan[elements_to_use[jj]].name == 's_a':
								wo = np.concatenate((np.where(fitp.s.name == 'compton_amplitude')[0], np.where(fitp.s.name == 'coherent_sct_amplitude')[0]), axis=0)

						if len(wo) == 0:
							continue
						thisdata.dataset_orig[i_fit, j_fit, jj, 1] = np.sum(values[i_fit, j_fit, wo])

					for ie in range(len(elements_to_use[:])):
						if 'TFY' == make_maps_conf.chan[elements_to_use[ie]].name:
							thisdata.dataset_orig[i_fit, j_fit, ie, 1] = tfy[i_fit, j_fit]
						if 'Bkgnd' == make_maps_conf.chan[elements_to_use[ie]].name:
							thisdata.dataset_orig[i_fit, j_fit, ie, 1] = bkgnd[i_fit, j_fit]

			seconds_fit_end = tm.time()
			self.logger.info('fitting of this scan  finished in %s seconds', seconds_fit_end-seconds_fit_start)

			if make_maps_conf.use_fit == 2:
				kk_loop_length = no_detectors + 1
			else:
				kk_loop_length = no_detectors
			for kk in range(kk_loop_length):
				name_pre = 'fit_'
				if kk == no_detectors:
					name_after = '_integrated'
				else:
					name_after = '_det'+str(kk).strip()
				spectra[self.main_max_spectra-8].data[:] = fitted_temp[:, kk]
				spectra[self.main_max_spectra-7].data[:] = Ka_temp[:, kk]
				spectra[self.main_max_spectra-4].data[:] = bkground_temp[:, kk]
				spectra[0].data[:] = raw_temp[:, kk]
				spectra[0].name = name_pre+header+name_after
				# need to be in here for B station, for files w/o standards
				if beamline == '2-ID-B' :
					spectra[self.main_max_spectra-8].name = 'fitted'
					spectra[self.main_max_spectra-7].name = 'alpha'
					spectra[self.main_max_spectra-4].name = 'background'

				spectra[0].used_chan = raw_temp[:, 0].size
				spectra[0].calib['off'] = calib['off']
				spectra[0].calib['lin'] = calib['lin']
				if spectral_binning > 0:
					spectra[0].calib['lin'] = spectra[0].calib['lin'] * spectral_binning
				spectra[0].calib['quad'] = calib['quad']

				for isp in range(self.main_max_spectra-8,self.main_max_spectra-3):
					spectra[isp].used_chan = spectra[0].used_chan
					spectra[isp].calib['off'] = spectra[0].calib['off']
					spectra[isp].calib['lin'] = spectra[0].calib['lin']
					spectra[isp].calib['quad'] = spectra[0].calib['quad']
				# need to be in here for B station, for files w/o standards
				if beamline == '2-ID-B':
					names = spectra[np.where(spectra.name != '')].name
					names.insert(0, 'none')
					n_names = len(names)

				temp_this_max = max_chan_spec[:, 0].size
				temp = np.repeat(fitted_temp[:, 0], 2)
				max_chan_spec[0:temp_this_max, 2] = temp[0:temp_this_max]
				temp = np.repeat(Ka_temp[:, 0], 2)
				max_chan_spec[0:temp_this_max, 3] = temp[0:temp_this_max]
				temp = np.repeat(bkground_temp[:, 0], 2)
				max_chan_spec[0:temp_this_max, 4] = temp[0:temp_this_max]

				add_plot_spectra[:, 0, kk] = fitted_temp[:, kk]
				add_plot_spectra[:, 1, kk] = Ka_temp[:, kk]
				add_plot_spectra[:, 2, kk] = bkground_temp[:, kk]
				add_plot_spectra[:, 4, kk] = l_temp[:, kk]
				this_add_plot_spectra = np.zeros((self.max_spec_channels, 12))
				this_add_plot_spectra[:, :] = add_plot_spectra[:, :, kk]
				self.plot_fit_spec(info_elements, spectra=spectra, add_plot_spectra=this_add_plot_spectra, add_plot_names=add_plot_names, fitp=fitp)

		if xrf_bin > 0:
			if (xrf_bin == 2) and (n_cols > 5) and (n_rows > 5):
				for i_bin in range(n_cols-2):
					if i_bin % 2 == 0 :
						for jj in range(n_rows-2):
							if jj % 2 == 0:
								thisdata.dataset_orig[i_bin+1, jj, :, :] = (thisdata.dataset_orig[i_bin, jj, :, :] + thisdata.dataset_orig[i_bin+2, jj, :, :])/2.
								thisdata.dataset_orig[i_bin, jj+1, :, :] = (thisdata.dataset_orig[i_bin, jj, :, :] + thisdata.dataset_orig[i_bin, jj+2, :, :])/2.
								thisdata.dataset_orig[i_bin+1, jj+1, :, :] = (thisdata.dataset_orig[i_bin, jj, :, :] + thisdata.dataset_orig[i_bin+2, jj+2, :, :])/2.

			if (xrf_bin == 4) and (n_cols > 5) and (n_rows > 5) :
				this_dimensions = thisdata.dataset_orig.shape

				congrid_arr = np.zeros((np.floor(n_cols/3.), np.floor(n_rows/3.), this_dimensions[2], this_dimensions[3]))
				for i_bin in range(n_cols-3):
					if i_bin % 3 == 0 :
						for jj in range(n_rows-3):
							if jj % 3 == 0 :
								congrid_arr[i_bin/3, jj/3, :, :] = thisdata.dataset_orig[i_bin, jj, :, :]

				for i_a in range(this_dimensions[2]):
					for i_b in	range(this_dimensions[3]):
						temp_congrid = congrid_arr[:, :, i_a, i_b]
						temp_congrid = maps_tools.congrid(temp_congrid, (n_cols, n_rows), self.logger)
						thisdata.dataset_orig[:, :, i_a, i_b] = temp_congrid[:, :]

		#begin pca part:
		if self.pca > 0:
			seconds_PCA_start = tm.time()
			input_arr = np.zeros((n_cols * n_rows, n_channels))
			l = 0
			for i_x_pixels in range(n_cols):
				for i_y_pixels in range(n_rows - 1):
					input_arr[l, :] = scan.mca_arr[i_x_pixels, i_y_pixels, :]
					l = l + 1

			input_arr = np.transpose(input_arr)

			U, eigen_values_vec, V = np.linalg.svd(input_arr, full_matrices=False)

			temp_filename = os.path.basename(mdafilename)
			basename, extension = os.path.splitext(temp_filename)
			filename = os.path.join(self.main_dict['pca_dir'], basename) + '.pca.h5'

			gzip = 7

			f = call_function_with_retry(h5py.File, 5, 0.1, 1.1, (filename, 'w'))
			if f == None:
				self.logger.error('maps_generate_img_dat: Error opening file %s', filename)
			else:
				# create a group for maps to hold the data
				pcaGrp = f.create_group('PCA')

				ds_data = pcaGrp.create_dataset('n_channels', data=n_channels)
				ds_data = pcaGrp.create_dataset('n_cols', data=n_cols)
				ds_data = pcaGrp.create_dataset('n_rows', data=n_rows)
				data = long(len(scan.detector_description_arr))
				ds_data = pcaGrp.create_dataset('n_detector_description', data=data)
				ds_data = pcaGrp.create_dataset('eigen_vec', data=eigen_values_vec)
				ds_data = pcaGrp.create_dataset('U', data=U, compression='gzip', compression_opts=7)
				ds_data = pcaGrp.create_dataset('V', data=V, compression='gzip', compression_opts=7)
				ds_data = pcaGrp.create_dataset('input_arr', data=input_arr, compression='gzip', compression_opts=7)
				ds_data = pcaGrp.create_dataset('scan_time_stamp', data=scan.scan_time_stamp)
				ds_data = pcaGrp.create_dataset('y_coord_arr', data=scan.y_coord_arr)
				ds_data = pcaGrp.create_dataset('x_coord_arr', data=scan.x_coord_arr)
				ds_data = pcaGrp.create_dataset('x_pixels', data=scan.x_pixels)
				ds_data = pcaGrp.create_dataset('y_pixels', data=scan.y_pixels)
				ds_data = pcaGrp.create_dataset('detector_description_arr', data=scan.detector_description_arr)
				ds_data = pcaGrp.create_dataset('detector_arr', data=scan.detector_arr)

				f.close()

			seconds_PCA_end = tm.time()
			delta_sec = str(seconds_PCA_end-seconds_PCA_start)
			delta_min = str((seconds_PCA_end-seconds_PCA_start) / 60.)
			delta_hour = str((seconds_PCA_end-seconds_PCA_start) / 60. / 24.)
			self.logger.info('PCA part of the analysis took : %s  seconds corresponding to %s minutes corresponding to %s hours', delta_sec, delta_min, delta_hour)

		#########################################################################################

		thisdata.dataset_names = ['ROI sum', 'fitted', 'lin_fit']
		thisdata.scan_time_stamp = scan.scan_time_stamp
		thisdata.write_date = datetime.datetime.utcnow()
		thisdata.x_coord_arr = scan.x_coord_arr
		thisdata.y_coord_arr = scan.y_coord_arr
		thisdata.dmaps_set = dmaps_set
		for item in make_maps_conf.dmaps:
			if item.use == 1:
				thisdata.dmaps_names.append(item.name)
				thisdata.dmaps_units.append(item.units)

		dmaps_use = []
		for item in make_maps_conf.dmaps: dmaps_use.append(item.use)
		dmaps_use = np.array(dmaps_use)
		for i in range(len(make_maps_conf.chan)):
			if make_maps_conf.chan[i].use == 1:
				thisdata.chan_names.append(make_maps_conf.chan[i].name)
				thisdata.chan_units.append([make_maps_conf.chan[i].units[0],make_maps_conf.chan[i].units[1],make_maps_conf.chan[i].units[2]])
		thisdata.version = self.version
		thisdata.us_amp = us_amp
		thisdata.ds_amp = ds_amp

		chan_use = []
		for ii in range(len(make_maps_conf.chan)):
			chan_use.append( make_maps_conf.chan[ii].use)
		wo = np.where(np.array(chan_use) == 1)

		for i in range(3):
			for j in range(3):
				thisdata.dataset_calibration[:, i, j] = make_maps_conf.e_cal[wo, i, j]

		if (make_maps_conf.use_fit == 0) or (xrfflyscan == 1):
			for i in range(no_detectors) : thisdata.energy_fit[0, i] = make_maps_conf.calibration.offset[i]
			for i in range(no_detectors) : thisdata.energy_fit[1, i] = make_maps_conf.calibration.slope[i]
			for i in range(no_detectors) : thisdata.energy_fit[2, i] = make_maps_conf.calibration.quad[i]

		n_channels = 2048
		thisdata.n_energy = n_channels
		thisdata.energy = np.arange(float(n_channels)) * thisdata.energy_fit[1, 0] + thisdata.energy_fit[0, 0]
		thisdata.energy_spec = np.zeros((n_channels))
		thisdata.energy_spec[0:len(spec_all)] = spec_all[:]

		for j in range(5) :
			thisdata.max_chan_spec[0:len(max_chan_spec[:, 0]), j] = max_chan_spec[:, j]

		if self.xanes == 0:
			h5file = os.path.join(self.main_dict['img_dat_dir'], header + xrf_bin_ext + '.h5' + suffix)
			self.logger.info('now trying to write HDF5 file %s', h5file)
			energy_channels = spectra[0].calib['off'] + spectra[0].calib['lin'] * np.arange((n_channels), dtype=np.float)
			try:
				h5.write_hdf5(thisdata, h5file, scan.mca_arr, energy_channels, extra_pv=extra_pv, extra_pv_order=scan.extra_pv_key_list, update=True)
			except:
				self.logger.exception("Error writing "+h5file)
		'''
		#Generate average images
		if (total_number_detectors > 1):
			self.logger.info( ' we are now going to create the maps_generate_average...'
			if this_detector == total_number_detectors -1:
				self.logger.info( 'now doing maps_generate_average_img_dat, total_number_detectors: ', total_number_detectors, '	this_detector: ', this_detector, ' this_file = ', mdafilename
				energy_channels = spectra[0].calib['off'] + spectra[0].calib['lin'] * np.arange((n_channels), dtype=np.float)
				self.generate_average_img_dat(total_number_detectors, make_maps_conf, energy_channels, this_file=mdafilename, extra_pv=extra_pv)
		'''
		return

	# ----------------------------------------------------------------------

	def generate_average_img_dat(self, main_dict, make_maps_conf, energy_channels, extra_pv=None):
		self.logger.info("Generating average image")
		total_number_detectors = main_dict['total_number_detectors']
		h5p = maps_hdf5.h5(self.logger)

		imgdat_filenames = []
		if main_dict['dataset_files_to_proc'][0] == 'all':
			imgdat_filenames = []
			dirList=os.listdir(main_dict['XRFmaps_dir'])
			for fname in dirList:
				if fname[-4:] == '.h50':
					imgdat_filenames.append(fname)
		else:
			imgdat_filenames = [mdafile.replace('.mda', '.h50') for mdafile in main_dict['dataset_files_to_proc']]

		#self.logger.debug('imgdat_filename %s', imgdat_filenames)
		no_files = len(imgdat_filenames)
		for i_temp in range(no_files):
			basename, extension = os.path.splitext(imgdat_filenames[i_temp])
			imgdat_filenames[i_temp] = basename

		main_XRFmaps_names = imgdat_filenames

		for n_filenumber in range(no_files):
			# is the avergae .dat file older than the dat0 file ? if so, generate a
			# new avg file, otherwise skip it.
			valid_read = 0
			avg_XRFmaps_info = None
			try:
				added_number_detectors = 0
				for this_detector_element in range(total_number_detectors):
					sfile = os.path.join(self.main_dict['XRFmaps_dir'], imgdat_filenames[n_filenumber] + '.h5' + str(this_detector_element).strip())
					#self.logger.debug('sfile %s', sfile)
					n_ev, n_rows, n_cols, n_energy, energy, energy_spec, scan_time_stamp, dataset_orig = self.change_xrf_resetvars()
					#temp = max([sfile.split('/'), sfile.split('\\')])
					#if temp == -1:
					#	temp = 0

					if not os.path.isfile(sfile) :
						self.logger.warning('WARNING: did not find :%s skipping to next',sfile)
						continue

					XRFmaps_info, n_cols, n_rows, n_channels, valid_read = h5p.maps_change_xrf_read_hdf5(sfile, make_maps_conf)
					if valid_read == 0 and XRFmaps_info == None:
						self.logger.error('Error calling h5p.maps_change_xrf_read_hdf5(%s,%s)', sfile, make_maps_conf)
						break
					f = call_function_with_retry(h5py.File, 5, 0.1, 1.1, (sfile, 'r'))
					if f == None:
						self.logger.error('Error opening file %s', sfile)
						break
					if 'MAPS' not in f:
						self.logger.error('error, hdf5 file does not contain the required MAPS group. I am aborting this action')
						break

					maps_group_id = f['MAPS']

					entryname = 'mca_arr'
					mca_arr, valid_read = h5p.read_hdf5_core(maps_group_id, entryname)
					mca_arr = np.transpose(mca_arr)

					if valid_read == 0:
						self.logger.warning('warning: did not find the valid mca array in dataset. cannot extract spectra')
						break

					f.close()

					if added_number_detectors == 0:

						avg_XRFmaps_info, n_cols, n_rows, n_channels, valid_read = h5p.maps_change_xrf_read_hdf5(sfile, make_maps_conf)
						if valid_read == 0 and avg_XRFmaps_info == None:
							self.logger.error('Error calling h5p.maps_change_xrf_read_hdf5(%s, %s)', sfile, make_maps_conf)
							break
						avg_mca_arr = mca_arr.copy()

					elif added_number_detectors >= 1:
						avg_XRFmaps_info.dmaps_set[:, :, :] = avg_XRFmaps_info.dmaps_set[:, :, :] + XRFmaps_info.dmaps_set[:, :, :]
						avg_XRFmaps_info.dataset[:, :, :] = avg_XRFmaps_info.dataset[:, :, :] + XRFmaps_info.dataset[:, :, :]
						avg_XRFmaps_info.dataset_orig[:, :, :, :] = avg_XRFmaps_info.dataset_orig[:, :, :, :] + XRFmaps_info.dataset_orig[:, :, :, :]
						avg_XRFmaps_info.dataset_calibration[:, :, :] = avg_XRFmaps_info.dataset_calibration[:, :, :] + XRFmaps_info.dataset_calibration[:, :, :]
						avg_XRFmaps_info.energy_spec[:] = avg_XRFmaps_info.energy_spec[:] + XRFmaps_info.energy_spec[:]
						avg_XRFmaps_info.max_chan_spec[:, :] = avg_XRFmaps_info.max_chan_spec[:, :] + XRFmaps_info.max_chan_spec[:, :]
						avg_XRFmaps_info.raw_spec[:, :] = avg_XRFmaps_info.raw_spec[:, :] + XRFmaps_info.raw_spec[:, :]
						avg_mca_arr = avg_mca_arr + mca_arr

					added_number_detectors = added_number_detectors+1

				if not os.path.isfile(sfile):
					self.logger.warning('WARNING: did not find any of these: %s skipping to next level', sfile)
					continue

				if avg_XRFmaps_info != None:
					avg_XRFmaps_info.dmaps_set[:, :, :] = avg_XRFmaps_info.dmaps_set[:, :, :] / added_number_detectors
					avg_XRFmaps_info.dataset[:, :, :] = avg_XRFmaps_info.dataset[:, :, :] / added_number_detectors
					avg_XRFmaps_info.dataset_orig[:, :, :, :] = avg_XRFmaps_info.dataset_orig[:, :, :, :] / added_number_detectors
					avg_XRFmaps_info.dataset_calibration[:, :, :] = avg_XRFmaps_info.dataset_calibration[:, :, :] / added_number_detectors
					avg_XRFmaps_info.energy_spec[:] = avg_XRFmaps_info.energy_spec[:] / added_number_detectors
					avg_XRFmaps_info.max_chan_spec[:, :] = avg_XRFmaps_info.max_chan_spec[:, :] / added_number_detectors
					avg_XRFmaps_info.raw_spec[:, :] = avg_XRFmaps_info.raw_spec[:, :] / added_number_detectors

					h5p.write_hdf5(avg_XRFmaps_info, os.path.join(self.main_dict['XRFmaps_dir'], imgdat_filenames[n_filenumber] + '.h5'), avg_mca_arr, energy_channels, extra_pv=XRFmaps_info.extra_pv, extra_pv_order=XRFmaps_info.extra_pv_as_csv)
			except:
				self.logger.exception('exception occured')
		return

	# ----------------------------------------------------------------------
	def change_xrf_resetvars(self):

		n_ev = 0L
		n_rows = 3L
		n_cols = 3L
		n_energy = 1100L
		energy = np.zeros(n_energy)
		energy_spec = np.arange(float(n_energy))
		scan_time_stamp = ''
		dataset_orig = 0

		return n_ev, n_rows, n_cols, n_energy, energy, energy_spec, scan_time_stamp, dataset_orig

	# ----------------------------------------------------------------------
	def plot_fit_spec(self, info_elements, spectra=0, add_plot_spectra=0, add_plot_names=0, ps=0, fitp=0, perpix=0, save_csv=1):
		self.logger.info('ploting spectrum')

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

		#foreground_color = colortable[-1]
		#background_color = colortable[0]

		droplist_spectrum = 0
		#droplist_scale = 0
		png = 0
		if ps == 0:
			png = 2
		if spectra == 0:
			return

		have_name = 0
		for isp in range(len(spectra)):
			if spectra[isp].name != '':
				have_name = 1

		if have_name == 0:
			return
		filename = spectra[0].name

		if save_csv == 1:
			csvfilename = 'csv_' + filename + '.csv'
			file_csv = os.path.join(self.main_dict['output_dir'], csvfilename)

		if (png > 0) or (ps > 0):
			if png > 0:
				dpi = 100
				canvas_xsize_in = 900. / dpi
				canvas_ysize_in = 700. / dpi
				fig = mplot.figure.Figure(figsize=(canvas_xsize_in, canvas_ysize_in), dpi=dpi, edgecolor=None)
				canvas = FigureCanvas(fig)
				fig.add_axes()
				axes = fig.gca()
				for child in axes.get_children():
					if isinstance(child, mplot.spines.Spine):
						child.set_color((0., 0., 0.))
				#axes.set_axis_bgcolor(background_color)
				ya = axes.yaxis
				xa = axes.xaxis
				ya.set_tick_params(labelcolor=(0., 0., 0.))
				ya.set_tick_params(color=(0., 0., 0.))
				xa.set_tick_params(labelcolor=(0., 0., 0.))
				xa.set_tick_params(color=(0., 0., 0.))

			if ps > 0:
				ps_filename = 'ps_' + filename + '.pdf'
				if ps_filename == '':
					return
				eps_plot_xsize = 8.
				eps_plot_ysize = 6.

				fig = mplot.figure.Figure(figsize=(eps_plot_xsize, eps_plot_ysize))
				canvas = FigureCanvas(fig)
				fig.add_axes()
				axes = fig.gca()

				file_ps = os.path.join(self.main_dict['output_dir'], ps_filename)

		if spectra[droplist_spectrum].used_chan > 0:
			this_axis_calib = droplist_spectrum

			xaxis = (np.arange(spectra[this_axis_calib].used_chan))**2*spectra[this_axis_calib].calib['quad'] + \
					np.arange(spectra[this_axis_calib].used_chan) * spectra[this_axis_calib].calib['lin'] + \
					spectra[this_axis_calib].calib['off']
			xtitle = 'energy [keV]'

			xmin = fitp.g.xmin * 0.5
			xmax = fitp.g.xmax + (fitp.g.xmax - fitp.g.xmin) * 0.10

			wo_a = np.where(xaxis > xmax)[0]
			if len(wo_a) > 0 :
				wo_xmax = np.amin(wo_a)
			else:
				wo_xmax = spectra[droplist_spectrum].used_chan * 8. / 10.
			wo_b = np.where(xaxis < xmin)[0]
			if len(wo_b) >0:
				wo_xmin = np.amax(wo_b)
			else:
				wo_xmin = 0

			wo = np.where(spectra[droplist_spectrum].data[wo_xmin:wo_xmax + 1] > 0.)
			if len(wo[0]) > 0:
				ymin = np.amin(spectra[droplist_spectrum].data[wo + wo_xmin]) * 0.9
			else:
				ymin = 0.1
			if perpix > 0:
				ymin = 0.001
			if len(wo[0]) > 0:
				ymax = np.amax(spectra[droplist_spectrum].data[wo+wo_xmin] * 1.1)
			else:
				ymax = np.amax(spectra[droplist_spectrum].data)
			# make sure ymax is larger than ymin, so as to avoid a crash during plotting
			if ymax <= ymin:
				ymax = ymin + 0.001

			'''
			yanno = (1.01 + 0.04 * (1 - droplist_scale)) * ymax
			yanno_beta = (1.07 + 0.53 * (1 - droplist_scale)) * ymax
			if droplist_scale == 0:
				yanno_below = 0.8 * ymin
			else:
				yanno_below = ymin -(ymax - ymin) * .04
			yanno_lowest = (0.8 + 0.15 * (1 - (1 - droplist_scale))) * ymin
			'''
			this_spec = spectra[droplist_spectrum].data[0:spectra[droplist_spectrum].used_chan]
			wo = np.where(this_spec <= 0)[0]
			if len(wo) > 0:
				this_spec[wo] = ymin

			plot1 = axes.semilogy(xaxis, this_spec, linewidth=1.0)
			axes.set_xlabel(xtitle)
			axes.set_ylabel('counts')
			axes.set_xlim((xmin, xmax))
			axes.set_ylim((ymin, ymax))

			axes.set_position([0.10,0.18,0.85,0.75])
			self.logger.debug('spectra[droplist_spectrum].name %s', spectra[droplist_spectrum].name)

			axes.text(-0.10, -0.12, spectra[droplist_spectrum].name, transform=axes.transAxes)

			if add_plot_spectra.any():
				size = add_plot_spectra.shape
				if len(size) == 2:
					#for k = size[2]-1, 0, -1 :
					for k in np.arange(size[1] - 1, -1, -1):
						plot2 = axes.semilogy(xaxis, add_plot_spectra[:, k], color=colortable[1 + k], linewidth=1.0)

						if k <= 2:
							axes.text(-0.10 + 0.4 + 0.2 * k, -0.12, add_plot_names[k], color=colortable[1 + k], transform=axes.transAxes)
						if (k >= 3) and (k <= 6):
							axes.text(-0.10 + 0.2 * (k - 3), -0.15, add_plot_names[k], color=colortable[1 + k], transform=axes.transAxes)
						if k >= 7:
							axes.text(-0.10 + 0.2 * (k - 7), -0.18, add_plot_names[k], color=colortable[1 + k], transform=axes.transAxes)

					# plot background next to last
					plot3 = axes.semilogy(xaxis, add_plot_spectra[:, 2], color=colortable[1 + 2], linewidth=1.0)
					# plot fit last
					plot4 = axes.semilogy(xaxis, add_plot_spectra[:, 0], color=colortable[1 + 0], linewidth=1.0)

			# plot xrf ticks
			element_list = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 25, 26, 27, 28, 29, 30, 32, 33, 35]) - 1
			x_positions = []
			for i in range(len(info_elements)): x_positions.append(info_elements[i].xrf['ka1'])
			color = 2

			local_ymax = np.array([1.03, 1.15, 1.3]) * ymax
			local_ymin = ymax * 0.9
			for k in range(len(element_list)):
				i = element_list[k]
				line=mplot.lines.Line2D([x_positions[i], x_positions[i]], [local_ymin, local_ymax[(i - int(i / 3) * 3)]], color=colortable[color])
				line.set_clip_on(False)
				axes.add_line(line)
				axes.text(x_positions[i], local_ymax[(i - int(i / 3) * 3)], info_elements[i].name, ha='center', va='bottom', color=colortable[color])

			if (png > 0) or (ps > 0):
				if png > 0:
					axes.text(0.97, -0.23, 'mapspy', transform=axes.transAxes)
					if (png == 1) or (png == 2):
						image_filename = filename + '.png'
						self.logger.info('saving png %s', os.path.join(self.main_dict['output_dir'], image_filename))
						fig.savefig(os.path.join(self.main_dict['output_dir'], image_filename), dpi=dpi, edgecolor=None)

					if ps > 0:
						fig.savefig(file_ps)

			if save_csv == 1:
				if add_plot_spectra.any():
					size = add_plot_spectra.shape
					if len(size) == 2:
						spectra_names = ['Energy', 'Spectrum']
						for i in range(len(add_plot_names)):
							spectra_names.append(add_plot_names[i])
						dims = add_plot_spectra.shape
						allspectra = np.zeros((dims[0], dims[1] + 2))
						allspectra[:, 2:] = add_plot_spectra

						allspectra[:, 0] = xaxis
						allspectra[:, 1] = this_spec

						file_ptr = open_file_with_retry(file_csv, 'wb')
						if file_ptr == None:
							self.logger.error('Error opening file: %s', file_csv)
						else:
							writer = csv.writer(file_ptr)
							writer.writerow(spectra_names)
							writer.writerows(allspectra)

		return
