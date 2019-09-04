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
import matplotlib as mplot
import logging
from fitting import leastsqbound
import maps_definitions
import maps_analyze
import maps_fit_parameters
import maps_tools
import glob
import henke
import copy
from file_io import maps_mda
from file_io.file_util import open_file_with_retry
#from file_io.mca_io import load_mca


def index_full_str_search(str_list, search_str):
	try:
		return str_list.index(search_str)
	except:
		return -1


def index_partial_str_search(str_list, search_str):
	for idx in range(len(str_list)):
		if str_list[idx].find(search_str) > -1:
			return idx
	return -1


def find_str_idx(str_list, search_list):
	for search_str in search_list:
		idx = index_full_str_search(str_list, search_str)
		if idx > -1:
			return idx


def translate_amps_val(amp_val):
	if amp_val == 0:
		return 1
	if amp_val == 1:
		return 2
	if amp_val == 2:
		return 5
	if amp_val == 3:
		return 10
	if amp_val == 4:
		return 20
	if amp_val == 5:
		return 50
	if amp_val == 6:
		return 100
	if amp_val == 7:
		return 200
	if amp_val == 8:
		return 500

	return 1

def translate_amp_unit_val(amp_unit_val):
	if amp_unit_val == 0:
		return "pA/V"
	if amp_unit_val == 1:
		return "nA/V"
	if amp_unit_val == 2:
		return "uA/V"
	if amp_unit_val == 3:
		return "mA/V"
	return "N/A"

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class calibration:
	def __init__(self, main_dict, maps_conf, logger):
		self.maps_conf = maps_conf
		self.main_dict = main_dict
		self.logger = logger

	def convert_mda_to_mca(self, mda_filename, mca_filename):
		mda_scan = maps_mda.mda(self.logger)
		scan = mda_scan.read_scan(mda_filename)
		num_detectors = 1
		det_shape_len = len(scan.detector_arr.shape)
		if len(scan.mca_arr.shape) > 3:
			num_detectors = scan.mca_arr.shape[3]
		for i in range(num_detectors):
			mca_file = open_file_with_retry(mca_filename + str(i), 'wt')
			mca_file.write('VERSION:    3.1\n')
			mca_file.write('ELEMENTS:    1\n')
			mca_file.write('DATE:    ' + scan.scan_time_stamp + '\n')
			mca_file.write('CHANNELS:    ' + str(scan.mca_arr.shape[2]) + '\n')
			real_time_str_list = ['dxpXMAP2xfm3:mca' + str(i + 1) + '.ERTM',
								  '2xfm:mca1.ERTM',
								  '2xfm2:dxpSaturn:mca1.ERTM',
								  'dxpVortexSDD:mca1.ERTM'
								  '2iddXMAP:mca' + str(i + 1) + '.ERTM',
								  '21:D3win:mca' + str(i + 1) + '.ERTM',
								  'dxpXMAPDP3:mca' + str(i + 1) + '.ERTM',
								  '26idcXMAP:mca' + str(i + 1) + '.ERTM']
			live_time_str_list = ['dxpXMAP2xfm3:mca' + str(i + 1) + '.ELTM',
								  '2xfm2:dxpSaturn:mca1.ELTM',
								  '2iddXMAP:mca' + str(i + 1) + '.ELTM',
								  '21:D3win:mca' + str(i + 1) + '.ELTM',
								  'dxpXMAPDP3:mca' + str(i + 1) + '.ELTM',
								  '26idcXMAP:mca' + str(i + 1) + '.ELTM',
								  'dxpVortexSDD:mca1.ELTM']
			input_cnt_str_list = ['dxpXMAP2xfm3:dxp' + str(i + 1) + ':InputCountRate',
								  '2iddXMAP:dxp' + str(i + 1) + ':InputCountRate',
								  '21:D3win:dxp' + str(i + 1) + ':InputCountRate',
								  'dxpXMAPDP3:dxp' + str(i + 1) + ':InputCountRate',]
			output_cnt_str_list = ['dxpXMAP2xfm3:dxp' + str(i + 1) + ':OutputCountRate',
								  '2iddXMAP:dxp' + str(i + 1) + ':OutputCountRate',
								  '21:D3win:dxp' + str(i + 1) + ':OutputCountRate',
								  'dxpXMAPDP3:dxp' + str(i + 1) + ':OutputCountRate',]
			idx_rt = find_str_idx(scan.detector_description_arr, real_time_str_list)
			idx_lt = find_str_idx(scan.detector_description_arr, live_time_str_list)
			idx_incnt = find_str_idx(scan.detector_description_arr, input_cnt_str_list)
			idx_outcnt = find_str_idx(scan.detector_description_arr, output_cnt_str_list)
			if idx_rt > -1 and idx_lt > -1:
				if det_shape_len == 2:
					mca_file.write('REAL_TIME:    ' + str(scan.detector_arr[:, idx_rt].sum()) + '\n')
					rt_sum = scan.detector_arr[:, idx_rt].sum()
					in_sum = scan.detector_arr[:, idx_incnt].sum()
					out_sum = scan.detector_arr[:, idx_outcnt].sum()
					if rt_sum > 0.0 and in_sum > 0.0 and out_sum > 0.0:
						elt1_arr = rt_sum * out_sum / in_sum
						mca_file.write('LIVE_TIME:    ' + str(elt1_arr) + '\n')
					else:
						mca_file.write('LIVE_TIME:    ' + str(scan.detector_arr[:, idx_lt].sum()) + '\n')
				elif det_shape_len == 3:
					mca_file.write('REAL_TIME:    ' + str(scan.detector_arr[:, :, idx_rt].sum()) + '\n')
					rt_sum = scan.detector_arr[:, :, idx_rt].sum()
					in_sum = scan.detector_arr[:, :, idx_incnt].sum()
					out_sum = scan.detector_arr[:, :, idx_outcnt].sum()
					if rt_sum > 0.0 and in_sum > 0.0 and out_sum > 0.0:
						elt1_arr = rt_sum * out_sum / in_sum
						mca_file.write('LIVE_TIME:    ' + str(elt1_arr) + '\n')
					else:
						mca_file.write('LIVE_TIME:    ' + str(scan.detector_arr[:, :, idx_lt].sum()) + '\n')
			idx = index_partial_str_search(scan.mca_calib_description_arr, 'mca' + str(i + 1) + '.CALO')
			if idx > -1:
				mca_file.write('CAL_OFFSET:    ' + str(scan.mca_calib_arr[idx]) + '\n')
			else:
				mca_file.write('CAL_OFFSET:    0\n')
			idx = index_partial_str_search(scan.mca_calib_description_arr, 'mca' + str(i + 1) + '.CALS')
			if idx > -1:
				mca_file.write('CAL_SLOPE:    ' + str(scan.mca_calib_arr[idx]) + '\n')
			else:
				mca_file.write('CAL_SLOPE:    0\n')
			mca_file.write('CAL_QUAD:    0\n')
			bnp_ds_sum = 0.0
			for idx in range(len(scan.detector_description_arr)):
				if scan.detector_description_arr[idx].find('scaler') > -1 or scan.detector_description_arr[idx].find('current') > -1:
					if det_shape_len == 2:
						mca_file.write('ENVIRONMENT:    ' + str(scan.detector_description_arr[idx]) + '="' + str(scan.detector_arr[:, idx].mean()) + '"\n')
					elif det_shape_len == 3:
						mca_file.write('ENVIRONMENT:    ' + str(scan.detector_description_arr[idx]) + '="' + str(scan.detector_arr[:, :, idx].mean()) + '"\n')
				else:
					if det_shape_len == 2:
						mca_file.write('ENVIRONMENT:    ' + str(scan.detector_description_arr[idx]) + '="' + str(scan.detector_arr[:, idx].sum()) + '"\n')
					elif det_shape_len == 3:
						mca_file.write('ENVIRONMENT:    ' + str(scan.detector_description_arr[idx]) + '="' + str(scan.detector_arr[:, :, idx].sum()) + '"\n')
				if scan.detector_description_arr[idx].find('21:D3:scaler1_cts1.B') > -1:
					bnp_ds_sum += scan.detector_arr[:, :, idx].mean()
				if scan.detector_description_arr[idx].find('21:D3:scaler1_cts1.C') > -1:
					bnp_ds_sum += scan.detector_arr[:, :, idx].mean()
				if scan.detector_description_arr[idx].find('21:D3:scaler1_cts1.D') > -1:
					bnp_ds_sum += scan.detector_arr[:, :, idx].mean()
				if scan.detector_description_arr[idx].find('21:D3:scaler1_cts2.A') > -1:
					bnp_ds_sum += scan.detector_arr[:, :, idx].mean()
			for idx in range(len(scan.mca_calib_description_arr)):
				mca_file.write('ENVIRONMENT:    ' + str(scan.mca_calib_description_arr[idx]) + '="' + str(scan.mca_calib_arr[idx]) + '"\n')
			if bnp_ds_sum > 0.0:
				mca_file.write('ENVIRONMENT:  21:D3:scaler1_cts3.A="' + str(bnp_ds_sum) + '"')
			for idx in range(len(scan.extra_pv_key_list)):
				mca_file.write('ENVIRONMENT:  ' + str(scan.extra_pv_key_list[idx]) + '="' + str(scan.extra_pv[idx]) + '"')
			mca_file.write('DATA:\n')
			for idx in range(scan.mca_arr.shape[2]):
				if det_shape_len == 2:
					mca_file.write(str(scan.mca_arr[:, idx, i].sum()))
				elif det_shape_len == 3:
					mca_file.write(str(int(scan.mca_arr[:, :, idx, i].sum())) + '\n')
			mca_file.close()
		# end for each detector

	def generage_std_info_for_axo(self):
		maps_standardInfoFilename = os.path.join(self.main_dict['master_dir'], 'maps_standardinfo.txt')
		self.logger.info('creating %s', maps_standardInfoFilename)
		standardInfoFile = open_file_with_retry(maps_standardInfoFilename, 'w')
		if standardInfoFile != None:
			standardInfoFile.write('\tThis file describes arbitrary standard reference materials\n')
			standardInfoFile.write('\tplease put the filename that contains the measured standard next. if it was acquired using a multi element detector, please list just the mca0 file\n')
			standardInfoFile.write('FILENAME:      axo_std.mca\n')
			standardInfoFile.write('\tplease put the comma seperated list of elements in the standard\n')
			standardInfoFile.write('ELEMENTS_IN_STANDARD: Ca, Fe, Cu\n')
			standardInfoFile.write('\tplease put the weight (area density) in ug/cm2 for each of the elements above\n')
			standardInfoFile.write('WEIGHT:       1.931, 0.504, 0.284\n')
			standardInfoFile.flush()
			standardInfoFile.close()
			standardInfoFile = open_file_with_retry(maps_standardInfoFilename, 'r')
		return standardInfoFile

	def search_for_axo(self):
		search_path = os.path.join(self.main_dict['master_dir'], 'axo_std.mca*')
		self.logger.info('Searching for %s', search_path)
		file_list = glob.glob(search_path)
		if len(file_list) < 1:
			self.logger.warning('Nothing found.')
			search_path = os.path.join(self.main_dict['mda_dir'], 'axo_std.mda')
			self.logger.info('Searching for %s', search_path)
			file_list = glob.glob(search_path)
			if len(file_list) < 1:
				self.logger.warning('Nothing found.')
				return None
			else:
				# create axo_std
				self.convert_mda_to_mca(file_list[0], os.path.join(self.main_dict['master_dir'], 'axo_std.mca'))
		# found axo_std, make maps_standardinfo.txt and return file handle
		return self.generage_std_info_for_axo()

	# -----------------------------------------------------------------------------
	def read_generic_calibration(self, this_detector, total_number_detectors, no_nbs, fitp, info_elements):
		# read info from maps_standardinfo.txt
		maps_standardInfoFilename = os.path.join(self.main_dict['master_dir'], 'maps_standardinfo.txt')
		self.logger.info('opening %s', maps_standardInfoFilename)
		standardInfoFile = open_file_with_retry(maps_standardInfoFilename, 'rt') 
		if standardInfoFile is None:
			# try to look for axo mda and create std calib from it.
			standardInfoFile = self.search_for_axo()
			if standardInfoFile is None:
				self.logger.warning('Warning: Could not load maps_standardinfo.txt.')
				return False
		# parse the file for filename, elements, and weights
		standardinfo_dict = dict()
		line = standardInfoFile.readline()
		while len(line) > 0:
			# self.logger.debug( 'line = ',line
			if line.find(':') > -1:
				subline = line.split(':')
				standardinfo_dict[subline[0].strip().rstrip()] = subline[1].strip().rstrip()
			line = standardInfoFile.readline()
		# check that all keys exist
		if not 'FILENAME' in standardinfo_dict:
			self.logger.warning('Warning: Could not find key FILENAME in maps_standardinfo.txt, returning')
			return False
		if not 'ELEMENTS_IN_STANDARD' in standardinfo_dict:
			self.logger.warning('Warning: Could not find key ELEMENTS_IN_STANDARD in maps_standardinfo.txt, returning')
			return False
		if not 'WEIGHT' in standardinfo_dict:
			self.logger.warning('Warning: Could not find key WEIGHT in maps_standardinfo.txt, returning')
			return False
		self.logger.debug('standardinfo_dict: %s', standardinfo_dict)
		'''
		#open standards file 
		standardFilename = os.path.join(self.main_dict['master_dir'],standardinfo_dict['FILENAME'])
		if total_number_detectors > 1:
			standardFilename += str(this_detector)
		mca_dict = load_mca(standardFilename)
		if mca_dict['success'] == False:
			self.logger.error( 'Error loading',standardFilename
			return False
		'''
		e_list = [x.strip().rstrip() for x in standardinfo_dict['ELEMENTS_IN_STANDARD'].split(',')]
		weight_list = [float(x) for x in standardinfo_dict['WEIGHT'].split(',')]
		self.logger.debug('element list = %s', e_list)

		suffix = ''

		fit = maps_analyze.analyze(self.logger)

		m_str = 'called calibration with these specifications: this detector: {0}; total_number_detectors: {1}.'.format(this_detector, total_number_detectors)
		self.logger.info('%s', m_str)

		if total_number_detectors > 0:
			if total_number_detectors > 1:
				suffix = str(this_detector)
				self.logger.info(' and suffix is: %s', suffix)

		maxiter = 500
		old_ratio = 0

		e_cal = self.maps_conf.e_cal.copy()
		e_cal_factor = np.zeros((self.maps_conf.n_chan, 3))

		aux_arr = np.zeros((self.maps_conf.n_chan, 6))
		yield_correction = np.zeros((self.maps_conf.n_chan, 4))

		airpath = 0
		srcurrent_name = ''

		overide_files_found = 0
		maps_overridefile = os.path.join(self.main_dict['master_dir'], 'maps_fit_parameters_override.txt')
		try:
			f = open(maps_overridefile, 'rt')
			self.logger.debug('maps override file: %s exists', maps_overridefile)
			f.close()
			overide_files_found = 1
		except:
			overide_files_found = 0

		if overide_files_found == 1:
			f = open(maps_overridefile, 'rt')
			for line in f:
				if ':' in line:
					slist = line.split(':')
					tag = slist[0]
					value = ''.join(slist[1:])

					if tag == 'AIRPATH':
						airpath = float(value)
					elif tag == 'SRCURRENT':
						srcurrent_name = str(value)
					elif tag == 'US_IC':
						us_ic_name = str(value)
					elif tag == 'DS_IC':
						ds_ic_name = str(value)

			f.close()

		if airpath > 0:
			self.logger.info('airpath: %s', airpath)
		else:
			self.logger.info('no airpath absorption')
		if len(srcurrent_name):
			self.logger.info('srcurrent_name: %s', srcurrent_name)
		else:
			self.logger.info('default srcurrent_name')

		#xxo element list
		#e_list = ['Ca', 'Fe', 'Cu']
		#std.name = 'axo_std.mca'

		std = self.maps_conf.element_standard
		std.name = standardinfo_dict['FILENAME']

		if total_number_detectors > 0: 
			if total_number_detectors > 1:
				std.name = std.name + suffix
				self.logger.info(' and suffix is: %s', suffix)

		have_standard = 0
		try:
			f = open(os.path.join(self.main_dict['master_dir'], std.name), 'rt')
			f.close()
			self.logger.info('this_standard_filename: %s', std.name)
			have_standard = 1
		except:
			standardInfoFile = self.search_for_axo()
			try:
				f = open(os.path.join(self.main_dict['master_dir'], std.name), 'rt')
				f.close()
				self.logger.info('this_standard_filename: %s', std.name)
				have_standard = 1
			except:
				self.logger.error('Could not open standard: %s', std.name)
				return False

		if have_standard == 1:
			filename = os.path.join(self.main_dict['master_dir'], std.name)

			calibration, data, date, live_time, real_time, current, IC_US, IC_DS, us_amp, ds_amp = self.read_calibration(filename,
																														srcurrent_name=srcurrent_name,
																														us_ic_name=us_ic_name,
																														ds_ic_name=ds_ic_name)
			if calibration is None:
				return False

			if data.size <=1 :
				self.logger.error('error: standard does not contain data : %s', std.name)
			no_mca_detectors = data.shape
			if len(no_mca_detectors) == 1:
				no_mca_detectors = 1 
			else:
				no_mca_detectors = no_mca_detectors[1]
			# if cannot find the SRcurrent, just set it to 100.
			if current == 0:
				current = 100.

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

		maps_defs = maps_definitions.maps_definitions(self.logger)

		chan_names = []
		chan_calib = []
		for i in range(len(self.maps_conf.chan)):
			chan_names.append(self.maps_conf.chan[i].name)
			chan_calib.append(self.maps_conf.chan[i].calib)

		if DO_fit > 0:

			spectra = maps_defs.define_spectra(self.main_dict['max_spec_channels'],
												self.main_dict['max_spectra'],
												self.main_dict['max_ICs'],
												mode='spec_tool')

			for n in range(no_mca_detectors) :
				data_temp = spectra[0].data.copy()
				data_temp[:] = 0.
				if data[:, n].size > 2048:
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
				maps_defs.push_spectrum(filename, spectra, append = append, n_channels=data_temp.size, n_detector_elements=1, real_time=real_time[n],
					  live_time=live_time[n], current=current, calibration=calibration, counts_us_ic=ic_us, counts_ds_ic=ic_ds, roi_area=0.,
					  roi_pixels=1., us_amp=us_amp, ds_amp=ds_amp, n_spatial_rois=1, year=0, data=data_temp, name=name,
					  DO_NOT_MOD_name=DO_NOT_MOD_name, date=date)

				dofit_spec = 1
				if dofit_spec == 0: 
					#self.logger.debug( 'keyword dofit_spec = 0'
					fit_this_spec = 0
					current_spec = fit_this_spec
					used_chan = spectra[current_spec].used_chan
					this_spectrum = spectra[current_spec].data[0:used_chan - 1]
					this_spectrum = this_spectrum.astype(np.float)
					first = 1
					calib = spectra[current_spec].calib

					u, fitted_spec, background, xmin, xmax, perror = fit.fit_spectrum(fitp, this_spectrum, used_chan, calib,
								first=first, matrix=True, maxiter=maxiter)

					if u is None:
						self.logger.error('Error calling fit_spectrum!. returning')
						return False

				fitp.g.no_iters = 4

				if self.maps_conf.use_fit == 2:
					this_w_uname = "DO_FIT_ALL_W_TAILS"
					fitp, avg_fitp, spectra = self.do_fits(this_w_uname, fitp, spectra, per_pix=1, generate_img=1, maxiter=maxiter, suffix=suffix, info_elements=info_elements)
					if fitp is None:
						return False
				else:
					this_w_uname = "DO_MATRIX_FIT"
					fitp, avg_fitp, spectra = self.do_fits(this_w_uname, fitp, spectra, per_pix=1, generate_img=1, maxiter=maxiter, suffix=suffix, info_elements=info_elements)
					if fitp is None:
						return False

				std.calibration.offset[0] = fitp.s.val[0]
				std.calibration.slope[0] = fitp.s.val[1]
				std.calibration.quad[0] = fitp.s.val[2]

				if n == 0:
					names = fitp.s.name
					values = np.zeros((fitp.g.n_fitp, no_mca_detectors))

				if std.live_time[n] != 0.0:
					values[:, n] = fitp.s.val / std.live_time[n]
				else:
					values[:, n] = 0.

			if len(values.shape) == 2:
				values = np.sum(values, axis=1)
			for jj in range(len(e_list)):
				weight_ugr_cm = 0.
				weight_ugr_cm = weight_list[jj]

				e_cal_factor[jj, 0] = (weight_ugr_cm*std.current)
				e_cal_factor[jj, 1] = (weight_ugr_cm*std.us_ic)
				e_cal_factor[jj, 2] = (weight_ugr_cm*std.ds_ic)

				if e_list[jj] not in chan_names:
					continue
				wo = chan_names.index(e_list[jj])

				wo_a = np.where(fitp.s.name == e_list[jj])
				if len(wo_a[0]) == 0:
					continue
				wo_a=wo_a[0]

				counts = values[wo_a]
				e_cal[wo, 1, 0] = e_cal_factor[jj, 0] / counts
				e_cal[wo, 1, 1] = e_cal_factor[jj, 1] / counts
				e_cal[wo, 1, 2] = e_cal_factor[jj, 2] / counts
				e_cal[wo, 1, 3] = counts
				e_cal[wo, 1, 4] = np.sqrt(counts) / counts # error fraction
				if self.maps_conf.version >= 8:
					e_cal[wo, 1, 5] = counts

		else:			#IF keyword_set(DO_fit) THEN BEGIN	
			# define fitp, as this also defines add_pars, which are needed below
			fp = maps_fit_parameters.maps_fit_parameters(self.logger)
			fitp = fp.define_fitp(self.main_dict['beamline'], info_elements)

		# The result of the fit is
		# nomalized by live time, below, the spectral data is normalized,
		# before ROIs are applied. 
		for n in range(no_mca_detectors):
			if std.live_time[n] != 0.0:
				data[:, n] = data[:, n] / std.live_time[n]
			else:
				data[:, n] = 0.

		# did above for fitted quantification, below for roi based quantification
		for jj in range(len(e_list)): 
			weight_ugr_cm = 0.
			weight_ugr_cm = weight_list[jj]

			e_cal_factor[jj, 0] = (weight_ugr_cm*std.current)
			e_cal_factor[jj, 1] = (weight_ugr_cm*std.us_ic)
			e_cal_factor[jj, 2] = (weight_ugr_cm*std.ds_ic)

			if e_list[jj] not in chan_names:
				continue
			wo = chan_names.index(e_list[jj])

			counts = 0.
			for kk in range(no_mca_detectors):
				# note: center position for peaks/rois is in keV, widths of ROIs
				# is in eV
				left_roi = int(((self.maps_conf.chan[wo].center - self.maps_conf.chan[wo].width / 2. / 1000.) - std.calibration.offset[kk]) / std.calibration.slope[kk])
				if left_roi < 0:
					left_roi = 0
				right_roi = int(((self.maps_conf.chan[wo].center + self.maps_conf.chan[wo].width / 2. / 1000.) - std.calibration.offset[kk]) / std.calibration.slope[kk])
				if right_roi >= data[:, kk].size:
					right_roi = data[:, kk].size - 1

				roi_width = right_roi-left_roi + 1
				counts_temp = np.sum(data[left_roi:right_roi + 1, kk])
				counts = counts+counts_temp

			#self.logger.debug( self.maps_conf.chan[wo].name, 'counts = ', counts
			e_cal[wo, 0, 0] = e_cal_factor[jj, 0] / counts
			e_cal[wo, 0, 1] = e_cal_factor[jj, 1] / counts
			e_cal[wo, 0, 2] = e_cal_factor[jj, 2] / counts
			e_cal[wo, 0, 3] = counts
			e_cal[wo, 0, 4] = np.sqrt(counts) / counts # error fraction

			if self.maps_conf.version >= 8:
				e_cal[wo, 0, 5] = counts

		# below for ROI+ quantification
		e_cal[:, 2, :] = e_cal[:, 0, :]

		#Look for override files in main.master_dir
		if total_number_detectors > 1:
			overide_files_found = 0
			suffix = str(this_detector)
			maps_overridefile = os.path.join(self.main_dict['master_dir'], 'maps_fit_parameters_override.txt') + suffix
			try:
				f = open(maps_overridefile, 'rt')
				self.logger.info('maps override file %s exists', maps_overridefile)
				f.close()
			except :
				# if i cannot find an override file specific per detector, assuming
				# there is a single overall file.
				maps_overridefile = os.path.join(self.main_dict['master_dir'], 'maps_fit_parameters_override.txt')
		else:
			maps_overridefile = os.path.join(self.main_dict['master_dir'], 'maps_fit_parameters_override.txt')

		try:
			f = open(maps_overridefile, 'rt')	 
			self.logger.info('maps override file %s exists', maps_overridefile)
		except :
			self.logger.warning('Warning: did not find the following file: %s Please make sure the file is present in the parent directory, and try again. For now, I am aborting this action.', maps_overridefile)
			return False

		for line in f:
			if ':' in line : 
				slist = line.split(':')
				tag = slist[0]
				value = ''.join(slist[1:])
				
				if tag == 'DETECTOR_MATERIAL':
					temp = int(value)
					# if eq to 1, it is a silicon based detector
					if temp == 1:
						self.maps_conf.add_long['a'] = 1
						self.maps_conf.fit_t_ge = 0.

				elif tag == 'COHERENT_SCT_ENERGY':
					self.maps_conf.incident_E = float(value)

				elif tag == 'BE_WINDOW_THICKNESS':
					self.maps_conf.fit_t_be = float(value) * 1000.

				elif tag == 'GE_DEAD_LAYER':
					if self.maps_conf.add_long['a'] != 1:
						self.maps_conf.fit_t_ge = float(value) * 1000.
				
				elif tag == 'DET_CHIP_THICKNESS':
					self.maps_conf.add_float['a'] = float(value) * 1000.
				
				elif tag == 'BRANCHING_RATIO_ADJUSTMENT_K':
					temp_string = value.split(' ,')
					for ts in temp_string:
						ts.strip()
					
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
			
								if len(temp_string) >= 5:
									for jj in range(4):
										fitp.add_pars[ii, jj].ratio *= float(temp_string[(jj + 1)])

		f.close()

		e_cal[np.where(np.isfinite(e_cal) == False)] = 0.

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

		for mm in range(self.maps_conf.n_chan):
			jump_factor = 0.
			total_jump_factor = 0.
			element_temp = -1

			# if not to be used for calibration, then skip
			if self.maps_conf.chan[mm].calib == 0:
				continue
			if self.maps_conf.chan[mm].calib == 1:
				ienames = []
				for ie in info_elements: ienames.append(ie.name)
				if self.maps_conf.chan[mm].name not in ienames:
					continue
				element_temp = ienames.index(self.maps_conf.chan[mm].name)
				ev = info_elements[element_temp].xrf['ka1'] * 1000.
				yieldd = info_elements[element_temp].xrf_abs_yield['ka1']
				rel_yield = info_elements[element_temp].xrf_abs_yield['ka1'] / \
							np.sum(info_elements[element_temp].xrf_abs_yield['ka1'] + info_elements[element_temp].xrf_abs_yield['ka2'] + \
								info_elements[element_temp].xrf_abs_yield['kb1'] + info_elements[element_temp].xrf_abs_yield['kb2'])
				newrel_yield = fitp.add_pars[mm, 0].ratio / \
					np.sum(fitp.add_pars[mm, 0].ratio + fitp.add_pars[mm, 1].ratio + fitp.add_pars[mm, 2].ratio + fitp.add_pars[mm, 3].ratio)
		
				yieldd = newrel_yield * info_elements[element_temp].yieldD['k']
		
				name =	self.maps_conf.chan[mm].name
				k_yield = info_elements[element_temp].yieldD['k']		  
				#self.logger.debug( name, ' old_yield', yieldd, ' rel_yield', rel_yield, ' k_yield', k_yield, ' rel_yield * k_yield:', rel_yield* k_yield, ' newrel_yield', newrel_yield,  \
				#		  '[mm, 0] ', fitp.add_pars[mm, 0].ratio, '[mm, 1] ', fitp.add_pars[mm, 1].ratio, '[mm, 2] ', fitp.add_pars[mm, 2].ratio, '[mm, 3] ', fitp.add_pars[mm, 3].ratio, ' old_rel_yield/new_rel_yield:', rel_yield/newrel_yield
				if self.maps_conf.incident_E > info_elements[element_temp].bindingE['K']: 
					jump_factor = info_elements[element_temp].jump['K'] 

				yieldd = k_yield		 #*rel_yield

			if self.maps_conf.chan[mm].calib == 2:
				ienames = []
				for ie in info_elements:
					ienames.append(ie.name + '_L')
				if self.maps_conf.chan[mm].name not in ienames:
					continue
				element_temp = ienames.index(self.maps_conf.chan[mm].name)
				ev = info_elements[element_temp].xrf['la1'] * 1000.
				yieldd = info_elements[element_temp].xrf_abs_yield['la1']
				name = self.maps_conf.chan[mm].name[:-2]
				jump_factor = info_elements[element_temp].jump['L3'] 
				if self.maps_conf.incident_E > info_elements[element_temp].bindingE['L2']:
					total_jump_factor = info_elements[element_temp].jump['L2']
				if self.maps_conf.incident_E > info_elements[element_temp].bindingE['L1']:
					total_jump_factor = total_jump_factor*info_elements[element_temp].jump['L1']
				#self.logger.debug(	name, ' L jump_factor: ', jump_factor

			if self.maps_conf.chan[mm].calib == 3:
				ienames = []
				for ie in info_elements:
					ienames.append(ie.name + '_M')
				if self.maps_conf.chan[mm].name not in ienames:
					continue
				element_temp = ienames.index(self.maps_conf.chan[mm].name)
				ev = self.maps_conf.chan[mm].center * 1000.
				yieldd = info_elements[element_temp].xrf_abs_yield['ma1']
				name = self.maps_conf.chan[mm].name[:-2]
				jump_factor = info_elements[element_temp].jump['M5'] 
				total_jump_factor = info_elements[element_temp].jump['M1'] * info_elements[element_temp].jump['M2'] * \
									info_elements[element_temp].jump['M3'] * info_elements[element_temp].jump['M4']
				#self.logger.debug(	name, ' M jump_factor: ', jump_factor, ' yield ', yieldd, ' ev: ', ev, ' total_jump_factor: ', total_jump_factor

			if element_temp == -1:
				continue
			if jump_factor != 0.:
				if total_jump_factor == 0.:
					shell_factor = (jump_factor - 1.) / jump_factor
				else:
					shell_factor = (jump_factor - 1.) / jump_factor / total_jump_factor

			else:
				shell_factor = 0.
			if (name == 'U') and (self.maps_conf.chan[mm].calib == 3):
				U_shell_factor = shell_factor
		
			# beta proportional to f2, 

			# make sure name is known to henke routine, if not, skip
			# in case this is Pu, use U instead
			if (name == 'Pu') or (name == 'Np'):
				self.logger.info('name was %s resetting it to uranium to work with henke data', name)
				name = 'U'
				
			Chenke = henke.henke(self.logger)
			
			test = []
			test = Chenke.zcompound(name, test)
			if np.sum(test) == 0:
				continue
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
			aux_arr[mm, 0] = self.absorption(thickness, beta, 1239.852/((self.maps_conf.incident_E+0.1)*1000.), shell_factor=shell_factor)
			f1, f2, delta, beta  = Chenke.get_henke_single('Be', 1.848, ev) 
		
			aux_arr[mm, 1] = self.transmission(self.maps_conf.fit_t_be, beta, 1239.852/ev)
			f1, f2, delta, beta  = Chenke.get_henke_single( 'Ge', 5.323, ev) 
			aux_arr[mm, 2] = self.transmission(self.maps_conf.fit_t_ge, beta, 1239.852/ev)
			aux_arr[mm, 3] = yieldd
			if self.maps_conf.add_long['a'] == 1:
				f1, f2, delta, beta  = Chenke.get_henke_single( 'Si', 2.3, ev)
			aux_arr[mm, 4] = self.transmission(self.maps_conf.add_float['a'], beta, 1239.852/ev)
			if (self.maps_conf.add_float['a'] == 0.) or (self.maps_conf.add_long['a'] != 1):
				aux_arr[mm, 4] = 0.
			if airpath > 0: 
				#density = 1.0
				density = 0.00117
				#air_ele = 'N78.08O20.95Ar0.93'
				#density = 1.2047e-3
				#f1, f2, delta, beta, graze_mrad, reflect, inverse_mu, atwt = Chenke.get_henke_single('air', density, ev) 
				f1, f2, delta, beta = Chenke.get_henke_single('air', density, ev)
				aux_arr[mm, 5] = self.transmission(airpath*1000., beta, 1239.852/ev)  # airpath is read in microns, transmission function expects nm
			else:
				aux_arr[mm, 5] = 1.

		used_elements = []
		'''
		if 'Al' in chan_names:
			used_elements = [chan_names.index('Al')]
			used_elements.append(chan_names.index('Si'))
		else:
			used_elements = [chan_names.index('Si')]

		temp_string = ['K', 'Ca', 'Ti', 'V']
		if self.maps_conf.incident_E > 6.54: temp_string.append('Mn')
		if self.maps_conf.incident_E > 7.11: temp_string.append('Fe')
		if self.maps_conf.incident_E > 7.71: temp_string.append('Co')
		if self.maps_conf.incident_E > 8.98: temp_string.append('Cu')
		if self.maps_conf.incident_E > 9.66: temp_string.append('Zn')
		temp_string.append('Pb_M')
		if self.maps_conf.incident_E > 13.03: temp_string.append('Pb_L')
		'''
		for i in range(len(e_list)):
			if e_list[i] in chan_names:
				wo = chan_names.index(e_list[i])
				used_elements.append(wo)

		m_used_elements = []
		if 'Pb_M' in chan_names:
			m_used_elements.append(chan_names.index('Pb_M'))
			# if 'Au_M' in chan_names:
			# m_used_elements.append(chan_names.index('Au_M'))
		wo_m = [chan_names.index(s) for s in chan_names if "_M" in s]

		e_cal_shape = self.maps_conf.e_cal.shape
		temp_calib = np.zeros(e_cal_shape[0])
		m_temp_calib = np.zeros(e_cal_shape[0])

		self.aux_arr = aux_arr
		rrange = 3
		for k in range(rrange):
			for l in range(3):
				if l == 0:
					factor = std.current
				if l == 1:
					factor = std.us_ic
				if l == 2:
					factor = std.ds_ic
				for iel in used_elements:
					temp_calib[iel] = e_cal[iel, k, l]

				if len(m_used_elements) > 0:
					m_temp_calib[m_used_elements] = e_cal[m_used_elements, k, l]

				x = used_elements[:]
				y = 1./temp_calib[used_elements]
				y[np.isinf(y)] = 0.
				y = np.array(y)
				err = y / 20. + np.sqrt(y) / 10.
				#set the last three errors to be very large so
				#they do not impact the fiting. the last three
				#correspond to PbL, M, Zn		   
				#err[len(err)-1] = 100000.
				#if 'Pb_L' in temp_string: 
				#	 err[len(err)-2] = 100000.
				# to remove Al and Si from fit:
				#err[0] = 100000.
				#err[1] = 10000.

				wo = np.where(np.isfinite(y) == False)

				if len(wo[0]) > 0:
					y[wo] = 0.
					err[wo] = 1000.

				n_pars = 2
				parinfo_value = np.zeros(n_pars)
				parinfo_fixed = np.zeros(n_pars, dtype=np.int)
				parinfo_limited = np.zeros((n_pars, 2), dtype = np.int)
				parinfo_limits = np.zeros((n_pars, 2)) 

				parinfo_value[0] = 100000.0 / factor
				parinfo_value[1] = 0. # [1 micron initial thickness]	 air
				if airpath >0:
					parinfo_value[1] = float(airpath)
				parinfo_fixed[1] = 1
				parinfo_limited[0, 0] = 1
				parinfo_limits[0, 0] = 0.0

				bounds=[]
				for i in range(n_pars):
					havelimit = parinfo_limited[i, :]
					if havelimit[0] == 1:
						bmin = parinfo_limits[i, 0]
					else:
						bmin = None
					if havelimit[1] == 1:
						bmax = parinfo_limits[i, 1]
					else:
						bmax = None
					bounds.append((bmin, bmax))

				Clb = leastsqbound

				p0 = np.array(parinfo_value)
				p1, cov, infodict, mesg, self.success = Clb.leastsqbound(self.residuals, p0, bounds, args=(y, x), maxfev=maxiter, full_output=True)
				
				#perror1 = self.calc_perror(infodict['fjac'], infodict['ipvt'], len(p1))
				
				temp_x = np.where(np.array(chan_calib) >= 1)[0]
				curve = self.fit_calibrationcurve(temp_x, p1)
				#self.logger.debug( k, ' ', l, '   u:', p1[0], '   factor[I, IC] ', factor
				e_cal[:, k, l] = 0.
				for ic in range(len(self.maps_conf.chan)):
					for iun in range(4):
						self.maps_conf.chan[ic].units[iun] = 'cts/s'
				for tx in temp_x:
					for iun in range(3):
						self.maps_conf.chan[tx].units[iun] = 'ug/cm^2'
				e_cal[temp_x, k, l] = curve
				if len(m_used_elements) > 1:
					# now do seperate calibration using M lines
					x = m_used_elements[:]
					y = 1. / m_temp_calib[m_used_elements]
					err = y / 20. + np.sqrt(y) / 10.
					wo = np.where(np.isfinite(y) == False)[0]
					if len(wo) > 0:
						y[wo] = 0.
						err[wo] = 1000.
					n_pars = 2
					parinfo_value = np.zeros(n_pars)
					parinfo_fixed = np.zeros(n_pars, dtype=np.int)
					parinfo_limited = np.zeros((n_pars, 2), dtype=np.int)
					parinfo_limits = np.zeros((n_pars, 2))

					parinfo_value[0] = 100000.0/factor
					parinfo_value[1] = 0.0 # [1 micron initial thickness]	  air
					if airpath > 0:
						parinfo_value[1] = float(airpath)
					parinfo_fixed[1] = 1
					parinfo_limited[0, 0] = 1
					parinfo_limits[0, 0] = 0.0

					bounds=[]
					for i in range(n_pars):
						havelimit = parinfo_limited[i, :]
						if havelimit[0] == 1:
							bmin = parinfo_limits[i, 0]
						else:
							bmin = None
						if havelimit[1] == 1:
							bmax = parinfo_limits[i, 1]
						else:
							bmax = None
						bounds.append((bmin, bmax))

					p0 = np.array(parinfo_value)
					p1, cov, infodict, mesg, self.success = Clb.leastsqbound(self.residuals, p0, bounds, args=(y, x), maxfev=maxiter, full_output=True)
					curve = self.fit_calibrationcurve(wo_m, p1)

					for ie in wo_m:
						self.maps_conf.chan[ie].units[0:3] = 'ug/cm^2'			
					e_cal[wo_m, k, l] = curve

		if no_nbs == 1:
			self.maps_conf.calibration.slope = std.calibration.slope
			self.maps_conf.calibration.offset = std.calibration.offset
			self.maps_conf.calibration.quad = std.calibration.quad
			self.maps_conf.e_cal = e_cal

		self.maps_conf.element_standard = std
		self.maps_conf.e_cal = e_cal
		self.calibration_write_info(old_ratio=old_ratio, suffix=suffix, aux_arr=aux_arr, info_elements=info_elements)

		return True

	# -----------------------------------------------------------------------------
	def lookup_axo_standard_weight(self, this_element):
		
		weight_ugr_cm = 0

		if this_element == 'Pb': weight_ugr_cm = 0.761
		if this_element == 'Ca': weight_ugr_cm = 1.931
		if this_element == 'Fe': weight_ugr_cm = 0.504
		if this_element == 'Cu': weight_ugr_cm = 0.284
		if this_element == 'Mo': weight_ugr_cm = 0.132
		if this_element == 'Pd': weight_ugr_cm = 0.18
		if this_element == 'La': weight_ugr_cm = 1.1
		
		return weight_ugr_cm

	# -----------------------------------------------------------------------------
	def lookup_nbs_standard_weight(self, this_element, old_ratio=0):
		
		weight_ugr_cm = 0.0

		if this_element == 'Al' : weight_ugr_cm = 13.7374 /1.25 #Al 13.7374
		if this_element == 'Si' : weight_ugr_cm = 33. /1.158	#Si 33.24
		if this_element == 'Ca' : weight_ugr_cm = 18.5 /1.07	#Ca
		if this_element == 'V' : weight_ugr_cm = 4.19/1.03		#V
		if this_element == 'Mn'  : weight_ugr_cm = 4.22 /1.017	#Mn
		if this_element == 'Co'  : weight_ugr_cm = 0.935 /1.01	#Co
		if this_element == 'Cu'  : weight_ugr_cm = 2.246 /1.007 #Cu
		
		#nbs 1833:
		#	 if this_element == 'Si' : weight_ugr_cm = 33.24/1.158				#Si 
		if this_element == 'Ar' : weight_ugr_cm = 0.		   #Ar for sanity checks
		if this_element == 'K' : weight_ugr_cm = 17.1/1.095    #K 
		if this_element == 'Ti'  : weight_ugr_cm = 12.7/1.04   #Ti
		if this_element == 'Fe'  : weight_ugr_cm = 14.14/1.014 #Fe
		if this_element == 'Zn'  : weight_ugr_cm = 3.89/1.005  #Zn
		if this_element == 'Pb_M'  : weight_ugr_cm = 16.42/1.1 #Pb-M
		if this_element == 'Pb_L'  : weight_ugr_cm = 16.42/1.0 #Pb-L

		if old_ratio > 0 :				
			if this_element == 'Al' : weight_ugr_cm = 13.7374 /1.17 #Al 13.7374
			if this_element == 'Si' : weight_ugr_cm = 33.00 /1.14	#Si 33.24
			if this_element == 'Ca' : weight_ugr_cm = 18.5 /1.04	#Ca
			if this_element == 'V' : weight_ugr_cm = 4.19/1.02		#V
			if this_element == 'Mn'  : weight_ugr_cm = 4.22 /1.02	#Mn
			if this_element == 'Co'  : weight_ugr_cm = 0.935 /1.01	#Co
			if this_element == 'Cu'  : weight_ugr_cm = 2.246 /1.01	#Cu
		# nbs 1833:
		#	  if this_element == 'Si' : weight_ugr_cm = 33.24/1.14				 #Si 
			if this_element == 'Ar' : weight_ugr_cm = 0.		   #Ar for sanity checks
			if this_element == 'K' : weight_ugr_cm = 17.1/1.07	   #K 
			if this_element == 'Ti'  : weight_ugr_cm = 12.7/1.04   #Ti
			if this_element == 'Fe'  : weight_ugr_cm = 14.14/1.02  #Fe
			if this_element == 'Zn'  : weight_ugr_cm = 3.89/1.01   #Zn
			if this_element == 'Pb_M'  : weight_ugr_cm = 16.42/1.1 #Pb-M
			if this_element == 'Pb_L'  : weight_ugr_cm = 16.42/1.0 #Pb-L

		# if weight_ugr_cm == 0.0:
		# self.logger.debug( 'Warning, element ', this_element, ' weight = ', weight_ugr_cm

		return weight_ugr_cm

	# -----------------------------------------------------------------------------
	def read_calibration(self, filename, srcurrent_name='', us_ic_name='', ds_ic_name=''):

		IC_US = 0
		IC_DS = 0
		us_amp = np.zeros((3))
		ds_amp = np.zeros((3))

		current = 0

		real_time = 0
		live_time = 0

		f = open_file_with_retry(filename, 'rt')
		if f is None:
			self.logger.error('Could not open file: %s', filename)
			return None, None, None, None, None, None, None, None, None, None

		line = ''
		line = f.readline() # 1. line is version
		#self.logger.debug( line
		line = f.readline() # 2. is # elements
		slist = line.split(':')
		tag = slist[0]
		value = ''.join(slist[1:])
		n_detector_elements  = int(value)  
		#self.logger.debug( 'n_detector_elements', n_detector_elements
		line = f.readline()
		line = f.readline()
		slist = line.split(':')
		tag = slist[0]
		value = ''.join(slist[1:])
		n_channels = int(value)
		#self.logger.debug( 'n_channels', n_channels
		f.close()

		amp = np.zeros((8, 3))		 # 8 amplifiers, each with a numerical value(0) and a unit(1), resulting in  a factor (3)
		amp[:, 0] = 1. # put in a numerical value default of 1.

		calibration = { 'offset' : np.zeros((n_detector_elements)), 
					   'slope'	 : np.zeros((n_detector_elements)), 
					   'quad'	: np.zeros((n_detector_elements)) }

		found_data = 0
		f = open_file_with_retry(filename, 'rt')
		lines = f.readlines()

		for line in lines:
			if ':' in line : 
				slist = line.split(':')
				tag = slist[0]
				value = ''.join(slist[1:])

				if	 tag == 'VERSION': version = float(value)
				elif tag == 'DATE': date  =  value
				elif tag == 'ELEMENTS': n_detector_elements = int(value)
				elif tag == 'CHANNELS': n_channels = int(value)
				elif tag == 'ROIS':
					n_rois = np.zeros((n_detector_elements), dtype=int)
					value = value.split(' ')
					valuelist = [int(x) for x in value if x != '']
					n_rois[:] = valuelist
					max_rois = np.max(n_rois)
				elif tag == 'REAL_TIME':
					real_time = np.zeros((n_detector_elements))
					value = value.split(' ')
					valuelist = [float(x) for x in value if x != '']
					real_time[:] = valuelist
				elif tag == 'LIVE_TIME':
					live_time = np.zeros((n_detector_elements))
					value = value.split(' ')
					valuelist = [float(x) for x in value if x != '']
					#self.logger.debug( 'live time', valuelist
					live_time[:] = valuelist
				elif tag == 'CAL_OFFSET':
					value = value.split(' ')
					valuelist = [float(x) for x in value if x != '']
					calibration['offset'][:] = valuelist 
				elif tag == 'CAL_SLOPE':
					value = value.split(' ')
					valuelist = [float(x) for x in value if x != '']
					calibration['slope'][:] = valuelist				 
				elif tag == 'CAL_QUAD':
					value = value.split(' ')
					valuelist = [float(x) for x in value if x != '']
					calibration['quad'][:] = valuelist 
				elif tag == 'TWO_THETA':
					two_theta = np.zeros((n_detector_elements))
					value = value.split(' ')
					valuelist = [float(x) for x in value if x != '']
					two_theta[:] = valuelist
				elif tag == 'UPSTREAM_IONCHAMBER':
					IC_US = np.zeros((n_detector_elements))
					value = value.split(' ')
					valuelist = [float(x) for x in value if x != '']
					IC_US[:] = valuelist
				elif tag == 'DOWNSTREAM_IONCHAMBER':
					IC_DS = np.zeros((n_detector_elements))
					value = value.split(' ')
					valuelist = [float(x) for x in value if x != '']
					IC_DS[:] = valuelist
				elif tag == 'ENVIRONMENT':
					value = ':'.join(slist[1:])
					pos = value.find('=')
					etag = value[0:pos].strip()
					vallist = value.split('"')
					temp = vallist[1]
					if etag == 'S:SRcurrentAI':
						current = float(temp)	 
					elif etag == '2xfm:scaler1_cts1.B':
						if IC_US == 0 : IC_US = float(temp)			  
					elif etag == '2xfm:scaler1_cts1.C':
						if IC_DS == 0 : IC_DS = float(temp)		   
					elif etag == '2xfm:scaler3_cts1.B':
						IC_US = float(temp)			  
					elif etag == '2xfm:scaler3_cts1.C':
						IC_DS = float(temp) 
					elif etag == '2idd:scaler1_cts1.C':
						IC_US = float(temp)			  
					elif etag == '2idd:scaler1_cts1.B':
						IC_DS = float(temp)
					elif etag == '8bmb:3820:scaler1_cts1.B':
						IC_US = float(temp)			  
					elif etag == '8bmb:3820:scaler1_cts1.C':
						IC_DS = float(temp)		  
					elif etag[5:] == 'A1sens_num.VAL':
						amp[0, 0] = float(temp)		   
					elif etag[5:] == 'A2sens_num.VAL':
						amp[1, 0] = float(temp)			 
					elif etag[5:] == 'A3sens_num.VAL':
						amp[2, 0] = float(temp)			 
					elif etag[5:] == 'A4sens_num.VAL':
						amp[3, 0] = float(temp)			 
					elif etag[5:] == 'A1sens_unit.VAL':
						#self.logger.debug( 'now:', temp
						if (temp == "nA/V") or	(temp == "pA/V") or (temp == "uA/V") or (temp == "mA/V"):
							if (temp == "pA/V") : amp[0, 1] = 0
							if (temp == "nA/V") : amp[0, 1] = 1
							if (temp == "uA/V") : amp[0, 1] = 2
							if (temp == "mA/V") : amp[0, 1] = 3
						else:
							amp[0, 1] = float(temp)			 
					elif etag[5:] == 'A2sens_unit.VAL':
						if (temp == "nA/V") or	(temp == "pA/V") or (temp == "uA/V") or (temp == "mA/V"):
							if (temp == "pA/V") : amp[1, 1] = 0
							if (temp == "nA/V") : amp[1, 1] = 1
							if (temp == "uA/V") : amp[1, 1] = 2
							if (temp == "mA/V") : amp[1, 1] = 3
						else: 
							amp[1, 1] = float(temp)
					elif etag[5:] == 'A3sens_unit.VAL':
						if (temp == "nA/V") or	(temp == "pA/V") or (temp == "uA/V") or (temp == "mA/V"):
							if (temp == "pA/V") : amp[2, 1] = 0
							if (temp == "nA/V") : amp[2, 1] = 1
							if (temp == "uA/V") : amp[2, 1] = 2
							if (temp == "mA/V") : amp[2, 1] = 3
						else: 
							amp[2, 1] = float(temp)
					elif etag[5:] == 'A4sens_unit.VAL':
						if (temp == "nA/V") or (temp == "pA/V") or (temp == "uA/V") or (temp == "mA/V") : 
							if (temp == "pA/V") : amp[3, 1] = 0
							if (temp == "nA/V") : amp[3, 1] = 1
							if (temp == "uA/V") : amp[3, 1] = 2
							if (temp == "mA/V") : amp[3, 1] = 3
						else:
							amp[3, 1] = float(temp)
					if len(srcurrent_name) > 0:
						if etag == srcurrent_name:
							current = float(temp)
					if len(us_ic_name) > 0:
						if etag == us_ic_name:
							IC_US = float(temp)
					if len(ds_ic_name) > 0: 
						if etag == ds_ic_name:
							IC_DS = float(temp)

				elif tag == 'DATA':
					found_data = 1
					dataindex = lines.index(line)
					break

		f.close()

		if found_data: 
			data = np.zeros((n_channels, n_detector_elements))
			for i in range(n_channels):
				for j in range(n_detector_elements):
					dataindex += 1
					line = lines[dataindex]
					counts = float(line)
					data[i, j] = counts
		else:
			self.logger.error('Not a valid data file: %s', filename)
			return

		for i in range(8):
			amp[i, 2] = amp[i, 0] 
			if amp[i, 1] == 0:
				amp[i, 2] = amp[i, 2] / 1000.		# pA/V
			if amp[i, 1] == 1:
				amp[i, 2] = amp[i, 2]				# nA/V
			if amp[i, 1] == 2:
				amp[i, 2] = amp[i, 2] * 1000.		# uA/V
			if amp[i, 1] == 3:
				amp[i, 2] = amp[i, 2] * 1000. * 1000. # mA/V

		if self.main_dict['beamline'] == '2-ID-D':
			us_amp[:] = amp[1, :]
			ds_amp[:] = amp[3, :]

		if self.main_dict['beamline'] == '2-ID-E':
			us_amp[:] = amp[0, :]
			ds_amp[:] = amp[1, :]

		if self.main_dict['beamline'] == 'Bio-CAT':
			us_amp[:] = amp[0, :]
			ds_amp[:] = amp[1, :]

		if self.main_dict['beamline'] == 'GSE-CARS':
			amp[0, :]= [1, 1, 1]
			amp[1, :]= [1, 1, 1]
			us_amp[:] = amp[0, :]
			ds_amp[:] = amp[1, :]

		if IC_DS == 0:
			self.logger.warning('warning downstream IC counts zero')
			IC_DS = 1.

		if IC_US == 0:
			self.logger.warning('warning upstream IC counts zero')
			IC_US = 1.

		return calibration, data, date, live_time, real_time, current, IC_US, IC_DS, us_amp, ds_amp

	# -----------------------------------------------------------------------------
	def read_nbsstds(self, filename):

		overide_files_found = 0 
		maps_overridefile = os.path.join(self.main_dict['master_dir'],'maps_fit_parameters_override.txt')
		try:
			f = open(maps_overridefile, 'rt')	 
			self.logger.info('maps override file: %s exists', maps_overridefile)
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

					if	 tag == 'AIRPATH':
						airpath = float(value)
					elif tag == 'SRCURRENT':
						srcurrent_name = str(value)
					elif tag == 'US_IC':
						us_ic_name = str(value)
					elif tag == 'DS_IC':
						ds_ic_name = str(value)

			f.close()

		try:
			f = open(filename, 'rt')	
			self.logger.inof('filename: %s exists', filename)
			f.close()
			nbs_files_found = 1
		except :
			nbs_files_found = 0 

		if nbs_files_found == 0: 
			self.logger.warning('Did not find NBS calibration file, returning')
			return None
		if nbs_files_found == 1:
			filepath = os.path.join(self.main_dict['master_dir'], filename)

			calibration, data, date, live_time, real_time, current, IC_US, IC_DS, us_amp, ds_amp = self.read_calibration(filepath, 
																														 srcurrent_name = srcurrent_name, 
																														 us_ic_name = us_ic_name, 
																														 ds_ic_name = ds_ic_name)
			if calibration is None:
				return None
			if self.maps_conf.use_det.sum() > 0:
				if current == 0:
					self.logger.warning('Could not find synchrotron current in the NBS standard. Will proceed assuming a SRcurrent of 100 mA')
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
					self.logger.warning('warning: number of selected detectors does NOT match number OF detectors found in the mca file')

				wo = np.where(self.maps_conf.use_det == 1)[0]
				if len(wo) == 0:
					return None
				elif len(wo) == 1:
					nbs.live_time = live_time
					nbs.real_time = real_time					 
				else:
					for ii in range(len(wo)):
						nbs.live_time[wo[ii]] = live_time[wo[ii]]
						nbs.real_time[wo[ii]] = real_time[wo[ii]]

		return nbs

	# -----------------------------------------------------------------------------
	def _write_calibration_segment_(self, file_ptr, make_maps_conf, desc1, desc2, desc3, index):
		line = ' '
		print>>file_ptr, line.strip()
		line = 'calibration curve for ' + desc1
		print>>file_ptr, line.strip()
		line = 'desc, name, Z, for_SRCurrent_norm, for_US_IC_norm, for_DS_IC_norm' + desc3 + ', measured_counts, error_bar'
		if make_maps_conf.version >= 8:
			line = line + ', measured_nb1832, measured_nb1833'
		print>>file_ptr, line.strip()
		for ii in range(make_maps_conf.n_chan):
			line = 'calib_curve_' + desc2 + ',' + make_maps_conf.chan[ii].name + ', '
			line = line + str(make_maps_conf.chan[ii].z) + ', '
			for jj in range(5):
				line = line + str(make_maps_conf.e_cal[ii, index, jj]) + ', '
			if make_maps_conf.version >= 8:
				for jj in range(5, 7):
					line = line + str(make_maps_conf.e_cal[ii, index, jj]) + ', '
			print>>file_ptr, line.strip()

	# -----------------------------------------------------------------------------
	def calibration_write_info(self,
								old_ratio=0,
								suffix='',
								aux_arr=0,
								info_elements=0):

		self.logger.info('Writing standard info')

		make_maps_conf = self.maps_conf

		chan_names = []
		chan_calib = []
		for i in range(len(make_maps_conf.chan)):
			chan_names.append(make_maps_conf.chan[i].name)
			chan_calib.append(make_maps_conf.chan[i].calib)

		directory = self.main_dict['output_dir']
		if not os.path.exists(directory):
			os.makedirs(directory)
			if not os.path.exists(directory):
				self.logger.warning('warning: did not find the output directory, and could not create a new output directory. Will abort this action')
				return

		# determine no of maximal supported detectors	 n_max
		n_max = len(make_maps_conf.element_standard.real_time)

		filename = os.path.join(self.main_dict['output_dir'], 'calibration_info_standard_') + suffix + '.csv'

		try:
			f = open_file_with_retry(filename, 'w')
			if f is None:
				self.logger.error('Could not open info_file: %s', filename)
				return
		except :
			self.logger.error('Could not open info_file: %s', filename)
			return

		line = 'Standard:, ' + make_maps_conf.element_standard.name
		print>>f, line
		line = str(make_maps_conf.element_standard.real_time[0]) + ', '
		for ii in range(n_max):
			line = line + str(make_maps_conf.element_standard.real_time[ii]) + ', '
		print>>f, 'realtime[s]:,  ' + line.strip()
		line = str(make_maps_conf.element_standard.live_time[0])+', '
		for ii in range(n_max) :
			line = line + str(make_maps_conf.element_standard.live_time[ii])+', '
		print>>f, 'livetime[s]:,  ' + line.strip()
		line = str(make_maps_conf.element_standard.current)
		print>>f, 'I_[mA]:,  '+ line.strip()
		line = str(make_maps_conf.element_standard.us_ic)
		print>>f, 'US_IC[cts/s]:,	' + line.strip()
		line = str(make_maps_conf.element_standard.ds_ic)
		print>>f, 'DS_IC[cts/s]:,	' + line.strip()
		line = str(make_maps_conf.element_standard.us_amp[0]) + ', '
		line = line + translate_amp_unit_val( make_maps_conf.element_standard.us_amp[1] ) + ', '
		line = line + str( translate_amps_val( make_maps_conf.element_standard.us_amp[0] ) )
		print>>f, 'US_AMP[sensitivity/units/factor]:,' + line.strip()
		line = str(make_maps_conf.element_standard.ds_amp[0]) + ', '
		line = line + translate_amp_unit_val(make_maps_conf.element_standard.ds_amp[1]) + ', '
		line = line + str(translate_amps_val(make_maps_conf.element_standard.ds_amp[0]))
		print>>f, 'DS_AMP[sensitivity/units/factor]:,' + line.strip()
		line = ' '
		print>>f, line

		line = 'name ' + ', '
		line = line + 'z' + ', '
		line = line + 'units[DS_IC]' + ', '
		line = line + 'units[US_IC]' + ', '
		line = line + 'units[SRCurrent]' + ', '
		line = line + 'units[1]' + ', '
		line = line + 'use' + ', '
		line = line + 'calib' + ', '
		line = line + 'center' + ', '
		line = line + 'shift' + ', '
		line = line + 'width' + ', '
		line = line + 'bkground_left' + ', '
		line = line + 'bkground_right' + ', '
		line = line + 'left_roi[0]' + ', '
		line = line + 'right_roi[0]' + ', '
		line = line + 'left_bkground[0]' + ', '
		line = line + 'right_bkground[0]' + ', '
		line = line + 'absorption_1um_element' + ', '
		line = line + 'Be_window_transmission' + ', '
		line = line + 'Ge_dead_layer_transmission' + ', '
		line = line + 'XRF_yield' + ', '
		line = line + 'fraction_photons_absorbed_in_det' + ', '
		line = line + 'air_absoprtion' + ', '
		line = line + 'total_XRF_efficiency_factor' + ', '

		print>>f,  line.strip()

		if np.sum(aux_arr) != 0:
			max_eff = aux_arr[:, 0]*aux_arr[:, 1]*aux_arr[:, 2]*aux_arr[:, 3]*(1.-aux_arr[:, 4])*aux_arr[:, 5]
			max_eff = np.amax(max_eff)
		for ii in range(make_maps_conf.n_chan):
			line = make_maps_conf.chan[ii].name + ', '
			line = line + str(make_maps_conf.chan[ii].z) + ', '
			line = line + make_maps_conf.chan[ii].units[0] + ', '
			line = line + make_maps_conf.chan[ii].units[1] + ', '
			line = line + make_maps_conf.chan[ii].units[2] + ', '
			line = line + make_maps_conf.chan[ii].units[3] + ', '
			line = line + str(make_maps_conf.chan[ii].use) + ', '
			line = line + str(make_maps_conf.chan[ii].calib) + ', '
			line = line + str(make_maps_conf.chan[ii].center) + ', '
			line = line + str(make_maps_conf.chan[ii].shift) + ', '
			line = line + str(make_maps_conf.chan[ii].width) + ', '
			line = line + str(make_maps_conf.chan[ii].bkground_left) + ', '
			line = line + str(make_maps_conf.chan[ii].bkground_right) + ', '
			line = line + str(make_maps_conf.chan[ii].left_roi[0]) + ', '
			line = line + str(make_maps_conf.chan[ii].right_roi[0]) + ', '
			line = line + str(make_maps_conf.chan[ii].left_bkground[0]) + ', '
			line = line + str(make_maps_conf.chan[ii].right_bkground[0]) + ', '
			if np.sum(aux_arr) != 0:
				line = line + str(aux_arr[ii, 0]) + ', '
				line = line + str(aux_arr[ii, 1]) + ', '
				line = line + str(aux_arr[ii, 2]) + ', '
				line = line + str(aux_arr[ii, 3]) + ', '
				line = line + str(1 - aux_arr[ii, 4]) + ', '
				line = line + str(aux_arr[ii, 5]) + ', '
				line = line + str(aux_arr[ii, 0] * aux_arr[ii, 1] * aux_arr[ii, 2] * aux_arr[ii, 3] * (1. - aux_arr[ii, 4]) * aux_arr[ii, 5] / max_eff) + ', '

			print>>f, line.strip()

		self._write_calibration_segment_(f, make_maps_conf, 'ROIs', 'roi', '[XRF_counts_per_s/(weight_ugr_cm*standard.current)]', 0)
		self._write_calibration_segment_(f, make_maps_conf, 'fitted data', 'fitted', '', 1)
		self._write_calibration_segment_(f, make_maps_conf, 'ROI+', 'roi', '[XRF_counts_per_s/(weight_ugr_cm*standard.current)]', 2)
		f.close()

		self.logger.info("Saved calibration info file")

		wo = np.where(np.array(chan_calib) == 1)[0]
		if len(wo) > 0:
			self.logger.info('ploting spectrum')

			from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
			mplot.rcParams['pdf.fonttype'] = 42

			fontsize = 9
			mplot.rcParams['font.size'] = fontsize

			dpi = 100
			canvas_xsize_in = 1000. / dpi
			canvas_ysize_in = 700. / dpi

			bindingEK = []
			for ib in range(len(info_elements)): bindingEK.append(info_elements[ib].bindingE['K'])
			bindingEK = np.array(bindingEK)

			for k in range(3):
				for l in range(3):

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

					title = 'AXO calibration curve '
					if l == 0: title = title +'for ROI sum'
					if l == 1: title = title +'for fitted data'
					if l == 2: title = title +'for ROIs+'
					if k == 0: title = title +' with normalization by Synchrotron current'
					if k == 1: title = title +'with normalization by upstream ionchamber'
					if k == 2: title = title +'with normalization by downstream ionchamber'
					if k == 0: ytitle = 'XRF_counts_per_s/(weight_ugr_cm*nbs.current)'
					if k == 1: ytitle = 'XRF_counts_per_s/(weight_ugr_cm*US_IC)'
					if k == 2: ytitle = 'XRF_counts_per_s(weight_ugr_cm*DS_IC)'
					y = make_maps_conf.e_cal[wo, l, k]
					measured = y.copy()
					measured[:] = 0.
					#measured_32 = measured.copy()
					#measured_33 = measured.copy()
					for ii in range(len(wo)):
						weight_ugr_cm = 0.
						weight_ugr_cm = self.lookup_axo_standard_weight(make_maps_conf.chan[wo[ii]].name)
						'''
						if e_list.count(make_maps_conf.chan[wo[ii]].name) > 0:
							weight_ugr_cm = weight_list[e_list.index(make_maps_conf.chan[wo[ii]].name)]
						#self.logger.debug( "chan name = ",make_maps_conf.chan[wo[ii]].name, 'weight', weight_ugr_cm
						'''
						if weight_ugr_cm == 0.:
							continue
						if k == 0:
							norm = make_maps_conf.element_standard.current
						if k == 1:
							norm = make_maps_conf.element_standard.us_ic
						if k == 2:
							norm = make_maps_conf.element_standard.ds_ic
						measured[ii] = 1.0 / (weight_ugr_cm * norm / make_maps_conf.e_cal[wo[ii], l, 3])
						measured[np.isinf(measured)] = 0.
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

					x_p = x.copy()
					y_p = y.copy()
					x_p = np.insert(x_p, 0, x_p[0]-1)
					x_p = np.append(x_p, max(x_p)+1)
					y_p = np.insert(y_p, 0, 0)
					y_p = np.append(y_p, 0)
					measured_p = measured.copy()
					measured_p = np.insert(measured_p, 0, 0)
					measured_p = np.append(measured_p, 0)
					xtickv = x_p
					chnames = np.array(chan_names)
					chnames = chnames[wo[wo_2]].tolist()
					xtickname = [' '] + chnames+[' ']
						
					wo_3 = np.where(y != 0)
					if len(wo_3[0]) > 0:
						y_min = min(y[wo_3])
					else:
						y_min = 0.
					yrange = [y_min*0.8, max(y)*1.5]	   
					#if y_min < 1e-10 : yrange = [1e-4, max(y)*1.5]    

					try:	
						plot1 = axes.semilogy(x_p+0.5, y_p,color='black', linewidth=1.0, linestyle='steps')
						#plot1a = axes.semilogy(x_p,y_p, color = 'yellow', linewidth=1.0, linestyle='None', marker = '*')  
						plot2 = axes.semilogy(x_p, measured_p, color='blue', linewidth=1.0, linestyle='None', marker='x', markersize=7, mew=1.2)

						axes.xaxis.set_ticks(xtickv)
						axes.set_ylabel(ytitle)

						axes.set_xlim((x_p[0], x_p[-1]))
						axes.set_ylim(yrange)
						#axes.autoscale_view()
						axes.set_xticklabels(xtickname)
						axes.set_position([0.10,0.08,0.85,0.85])
						axes.set_title(title)

						axes.text(0.60, 0.13, 'NBS 1832', color = 'green', transform = axes.transAxes) 
						axes.text(0.60, 0.10, 'NBS 1833', color = 'red', transform = axes.transAxes) 
						axes.text(0.60, 0.07, 'axo', color = 'blue', transform = axes.transAxes) 

						axes.text(0.97, -0.08, 'mapspy', transform = axes.transAxes)
						image_filename = 'calib'+str(l)+'_'+str(k)+'standard.png'
						self.logger.info('saving standard png %s', os.path.join(directory,image_filename))
						fig.savefig(os.path.join(directory, image_filename), dpi=dpi, edgecolor=None)
					except:
						self.logger.warning('Warning: Could not save standard calibration plot.')

		return

	# -----------------------------------------------------------------------------
	def do_fits(self, this_w_uname, fitp, spectra, per_pix=0, generate_img=0, maxiter=500, suffix='', info_elements=0):
		beamline = self.main_dict['beamline']
		keywords = fitp.keywords

		fp = maps_fit_parameters.maps_fit_parameters(self.logger)
		avg_fitp = fp.define_fitp(beamline, info_elements)

		avg_fitp.s.val[:] = 0.
		avg_n_fitp = 0.

		if suffix != '':
			maps_overridefile = os.path.join(self.main_dict['master_dir'], 'maps_fit_parameters_override.txt')+suffix
			try:
				f = open(maps_overridefile, 'rt')
				f.close()
			except:
				# if i cannot find an override file specific per detector, assuming
				# there is a single overall file.
				maps_overridefile = os.path.join(self.main_dict['master_dir'], 'maps_fit_parameters_override.txt')

		else:
			maps_overridefile = os.path.join(self.main_dict['master_dir'], 'maps_fit_parameters_override.txt')

		used_chan = []
		for i in range(self.main_dict['max_spectra'] - 11):
			used_chan.append(spectra[i].used_chan)
		wo = np.where(np.array(used_chan) > 0)[0]
		tot_wo = len(wo)
		#self.logger.debug( 'fiting n spectra', tot_wo
		if tot_wo == 0:
			return 0, 0, spectra
		names = ['none']
		for i in range(len(spectra)):
			if spectra[i].name != '':
				names.append(spectra[i].name)

		#n_names = len(names)
		# now go one by one through all spectra loaded into the plot_spec window
		self.logger.debug('tot_wo: %s', tot_wo)
		for i in range(tot_wo):
			old_fitp = fp.define_fitp(beamline, info_elements)
			old_fitp.s.val[:] = fitp.s.val[:]

			for j in range(keywords.kele_pos[0]):
				fitp.s.use[j] = fitp.s.batch[j, 0]

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
			#self.logger.debug( 'spectra[wo[i]].name', spectra[wo[i]].name
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
					for jj in range(fitp.g.n_fitp):
						if fitp.s.name[jj] in test_string:
							#wo_a = test_string.index(fitp.s.name[jj])
							fitp.s.val[jj] = 1.
							fitp.s.use[jj] = 5

			else: 
				self.logger.info('fitting spectrum')
				# if not NBS standard then look here
				det = 0
				try:
					fitp, test_string, pileup_string = fp.read_fitp(maps_overridefile, info_elements, det=det)
					# copy fitp
					avg_fitp.g = copy.deepcopy(fitp.g)
					self.logger.info('found override file (maps_fit_parameters_override.txt). Using the contained parameters. %s', test_string)
				except:
					self.logger.warning('warning: did not find override file (maps_fit_parameters_override.txt). Will abort this action')
					return 0, 0, spectra
				for jj in range(fitp.g.n_fitp):
					if fitp.s.name[jj] in test_string:
						#wo_a = test_string.index(fitp.s.name[jj])
						fitp.s.val[jj] = 1.
						fitp.s.use[jj] = 5

			# temp_fitp_use = fitp.s.use[np.amin(fitp.keywords.kele_pos):np.amax(fitp.keywords.mele_pos)+1]
			# temp_fitp_name = fitp.s.name[np.amin(fitp.keywords.kele_pos):np.amax(fitp.keywords.mele_pos)+1]
			# which_elements_to_fit = (np.nonzero(temp_fitp_use != 1))[0]
			# self.logger.debug( 'elements to fit:'
			# self.logger.debug( temp_fitp_name[which_elements_to_fit]

			det = 0
			pileup_string = ''
			# try:
			# 	fitp, test_string, pileup_string = fp.read_fitp(maps_overridefile, info_elements, det=det)
			# except:
			# 	self.logger.debug( 'error reading fit paramenters'

			if avg_n_fitp == 0 :
				# make sure that avg_fitp gets redefined here, so that changes,
				# etc, in the override file get translateed into the avg file on
				# the first round
				fp = maps_fit_parameters.maps_fit_parameters(self.logger)
				avg_fitp = fp.define_fitp(beamline, info_elements)
				avg_fitp.s.val[:] = 0.

			if this_w_uname == "DO_FIT_ALL_W_TAILS":
				for j in range(keywords.kele_pos[0]):
					fitp.s.use[j] = fitp.s.batch[j,2]
			if this_w_uname == "DO_MATRIX_FIT":
				for j in range(keywords.kele_pos[0]):
					fitp.s.use[j] = fitp.s.batch[j,1]
			if this_w_uname == "DO_FIT_ALL_FREE":
				for j in range(keywords.kele_pos[0]):
					fitp.s.use[j] = fitp.s.batch[j,3]
			if this_w_uname == "DO_FIT_ALL_FREE_E_FIXED_REST":
				for j in range(keywords.kele_pos[0]):
					fitp.s.use[j] = fitp.s.batch[j,4]

			fp.parse_pileupdef(fitp, pileup_string, info_elements)

			if this_w_uname == "DO_MATRIX_FIT":
				matrix = 1 
			else:
				matrix = 0
			first = 1	

			temp_fitp_use = fitp.s.use[np.amin(fitp.keywords.kele_pos):np.amax(fitp.keywords.mele_pos)+1]
			temp_fitp_name = fitp.s.name[np.amin(fitp.keywords.kele_pos):np.amax(fitp.keywords.mele_pos)+1]
			which_elements_to_fit = (np.nonzero(temp_fitp_use != 1))[0]
			self.logger.debug('elements to fit: %s', temp_fitp_name[which_elements_to_fit])

			fit = maps_analyze.analyze(self.logger)
			u, fitted_spec, background, xmin, xmax, perror = fit.fit_spectrum(fitp, spectra[wo[i]].data, spectra[wo[i]].used_chan, spectra[wo[i]].calib, 
							first=first, matrix=matrix, maxiter=maxiter)
			if u is None:
				self.logger.error('Error calling fit_spectrum!. returning')
				return None, None, None

			#counts_background = fit.counts_background
			counts_ka = fit.counts_ka
			counts_kb = fit.counts_kb
			counts_l = fit.counts_l
			counts_m = fit.counts_m
			counts_elastic = fit.counts_elastic
			counts_compton = fit.counts_compton
			counts_step = fit.counts_step
			counts_tail = fit.counts_tail
			counts_pileup = fit.counts_pileup
			counts_escape = fit.counts_escape

			if (this_w_uname == "DO_FIT_ALL_FREE"):
				fitp.s.val[:] = u[:]
				fitp.s.val[keywords.peaks] = 10.0**u[keywords.peaks]
				for j in range(keywords.kele_pos[0]): fitp.s.use[j] = fitp.s.batch[j, 3]
				u, fitted_spec, background, xmin, xmax, perror = fit.fit_spectrum(fitp, spectra[wo[i]].data, spectra[wo[i]].used_chan, spectra[wo[i]].calib, 
																				first=first, matrix=matrix, maxiter=maxiter)
				if u is None:
					self.logger.error('Error calling fit_spectrum!. returning')
					return None, None, None

				#counts_background = fit.counts_background
				counts_ka = fit.counts_ka
				counts_kb = fit.counts_kb
				counts_l = fit.counts_l
				counts_m = fit.counts_m
				counts_elastic = fit.counts_elastic
				counts_compton = fit.counts_compton
				counts_step = fit.counts_step
				counts_tail = fit.counts_tail
				counts_pileup = fit.counts_pileup
				counts_escape = fit.counts_escape

			add_plot_spectra = np.zeros((self.main_dict['max_spec_channels'], 12))
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

			spectra[self.main_dict['max_spectra']-8].data[:] = 0.
			spectra[self.main_dict['max_spectra']-8].data[xmin:xmax + 1] = fitted_spec[:]
			spectra[self.main_dict['max_spectra']-8].name = 'fitted'
			for isp in range(self.main_dict['max_spectra'] - 8, self.main_dict['max_spectra'] - 3):
				spectra[isp].used_chan = spectra[wo[i]].used_chan 
				spectra[isp].calib['off'] = spectra[wo[i]].calib['off'] 
				spectra[isp].calib['lin'] = spectra[wo[i]].calib['lin']
				spectra[isp].calib['quad'] = spectra[wo[i]].calib['quad']
			spectra[self.main_dict['max_spectra']-7].data[:] = 0.
			spectra[self.main_dict['max_spectra']-7].data[xmin:xmax + 1] = counts_ka[:]
			spectra[self.main_dict['max_spectra']-7].name = 'ka_only'
			spectra[self.main_dict['max_spectra']-6].data[:] = 0.
			spectra[self.main_dict['max_spectra']-6].data[xmin:xmax + 1] = counts_kb[:]
			spectra[self.main_dict['max_spectra']-6].name = 'kb_only'
			spectra[self.main_dict['max_spectra']-5].data[:] = 0.
			spectra[self.main_dict['max_spectra']-5].data[xmin:xmax + 1] = counts_tail[:]
			spectra[self.main_dict['max_spectra']-5].name = 'tails'
			spectra[self.main_dict['max_spectra']-4].data[:] = 0.
			spectra[self.main_dict['max_spectra']-4].data[0:np.amin([spectra[wo[i]].used_chan, len(background) - 1])] = background[0:np.amin([spectra[wo[i]].used_chan, len(background)-1])]
			spectra[self.main_dict['max_spectra']-4].name = 'background'

			filename = 'specfit_' + names[wo[i] + 1] + suffix
			maps_tools.plot_spectrum(info_elements,
									spectra=spectra,
									i_spectrum=wo[i],
									add_plot_spectra=add_plot_spectra,
									add_plot_names=add_plot_names,
									ps=0,
									fitp=fitp,
									filename=filename,
									outdir=self.main_dict['output_dir'],
									logger=self.logger)

			if per_pix == 0:
				dirt = self.main_dict['output_dir']
				if not os.path.exists(dirt):
					os.makedirs(dirt)
					if not os.path.exists(dirt):
						self.logger.warning('warning: did not find the output directory, and could not create a new output directory. Will abort this action')
						return 0, 0, spectra
			else:
				if generate_img > 0:
					filename = os.path.join(self.main_dict['output_dir'], 'fit_'+names[wo[i]+1])
					#write_spectrum, main.output_dir+strcompress('fit_'+names[wo[i]+1]), spectra, droplist_spectrum

			avg_fitp.s.val[:] = avg_fitp.s.val[:] + fitp.s.val[:]
			avg_n_fitp += 1

		avg_fitp.g = copy.copy(fitp.g)
		avg_fitp.g.det_material |= fitp.g.det_material
		avg_fitp.s.val[:] = avg_fitp.s.val[:]/avg_n_fitp
		avg_fitp.s.max[:] = fitp.s.max[:]
		avg_fitp.s.min[:] = fitp.s.min[:]
		avgfilename = os.path.join(self.main_dict['master_dir'], 'average_resulting_maps_fit_parameters_override.txt')
		fp.write_fit_parameters(self.main_dict, avg_fitp, avgfilename, suffix=suffix)

		#self.logger.debug( 'fitp',fitp, 'avg_fitp',  avg_fitp, 'spectra', spectra
		return fitp, avg_fitp, spectra

	# -----------------------------------------------------------------------------
	def transmission(self, thickness, beta, llambda):

		arg = -4. * np.pi * thickness * beta / llambda
		value = np.abs(np.math.exp(arg))
		return value

	# -----------------------------------------------------------------------------
	def absorption(self, thickness, beta, llambda, shell_factor=[]):
		# make sure shell_factor is defined, and if not, set it to 1
		# shell factor <1 is to determine how much is
		# absorbed by a subshell, and is essentially the
		# ratio of jump factor -1 / jump factor
		if shell_factor == 0:
			shell_factor = 1

		arg = -4. * np.pi * thickness * shell_factor * beta / llambda
		value = 1 - np.abs(np.math.exp(arg))
		return value

	# -----------------------------------------------------------------------------
	def fit_calibrationcurve(self, z_prime, p):
		# aux_arr[mm, 0] = absorption
		# aux_arr[mm, 1] = transmission, Be
		# aux_arr[mm, 2] = transmission, Ge or Si dead layer
		# aux_arr[mm, 3] = yield
		# aux_arr[mm, 4] = transmission through Si detector
		# aux_arr[mm, 5] = transmission through  air (N2)

		aux_arr = self.aux_arr

		value = p[0] * aux_arr[z_prime, 0] * aux_arr[z_prime, 1] * aux_arr[z_prime, 2] * aux_arr[z_prime, 3] * (1. - aux_arr[z_prime, 4]) * aux_arr[z_prime, 5]

		return value

	# -----------------------------------------------------------------------------
	def residuals(self, p, y, x):
		err = (y - self.fit_calibrationcurve(x, p))
		return err			

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == '__main__':
	m_dict = dict()
	m_dict['master_dir'] = '/tmp'
	logger = logging.getLogger('calibrationtest')
	c = calibration(m_dict, None, logger)
	c.read_generic_calibration(None, None, None, None, None)
