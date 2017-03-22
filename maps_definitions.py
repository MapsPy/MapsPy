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

import numpy as np
import string
import struct

# -----------------------------------------------------------------------------
class calibration:
	def __init__(self):
		self.offset = []
		self.slope = []
		self.quad = []


# -----------------------------------------------------------------------------
class standard:
	def __init__(self):
		self.name = ''
		self.date = ''
		self.live_time = []
		self.real_time = []
		self.current = 0.
		self.us_ic = 0.
		self.ds_ic = 0.
		self.us_amp = np.zeros(3)
		self.ds_amp = np.zeros(3)
		self.calibration = calibration()

	# -----------------------------------------------------------------------------


class chan:
	def __init__(self):
		self.z = 0
		self.name = ''
		self.units = ['', '', '', '']
		self.use = 0
		self.calib = 0
		self.center = 0.
		self.shift = 0.
		self.width = 0.
		self.bkground_left = 0.
		self.bkground_right = 0.
		self.left_roi = []
		self.right_roi = []
		self.left_bkground = []
		self.right_bkground = []


# -----------------------------------------------------------------------------
class dmaps:
	def __init__(self):
		self.name = ''
		self.units = ''
		self.use = 0

	# -----------------------------------------------------------------------------


class maps_conf:
	def __init__(self):
		self.use_default_dirs = 1
		self.use_beamline = 0
		self.version = 0
		self.use_det = []
		self.calibration = calibration()
		# self.nbs32 = standard()
		# self.nbs33 = standard()
		# self.axo = standard()
		self.element_standard = standard()
		self.e_cal = []
		# self.axo_e_cal = []
		self.fit_t_be = 8000.
		self.fit_t_GE = 100.
		self.n_chan = 0
		self.n_used_chan = 0
		self.active_chan = 0
		self.chan = []
		self.n_dmaps = 0
		self.n_used_dmaps = 0
		self.dmaps = []
		self.incident_E = 10.
		self.use_fit = 0
		self.use_pca = 0
		self.add_long = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0}
		self.add_float = {'a': 0., 'b': 0., 'c': 0., 'd': 0., 'e': 0.}


# -----------------------------------------------------------------------------
class XRFmaps_info(object):
	def __init__(self, n_cols, n_rows, dataset_size, n_energy, n_raw_spec, n_raw_det, n_used_chan, n_used_dmaps,
				 maps_conf, version):
		self.n_ev = 0
		self.version = version
		self.n_cols = n_cols
		self.n_rows = n_rows
		self.n_used_dmaps = n_used_dmaps
		self.n_used_chan = n_used_chan
		self.scan_time_stamp = ''
		self.write_date = ''
		self.x_coord_arr = np.zeros((n_cols))
		self.y_coord_arr = np.zeros((n_rows))
		self.dmaps_set = np.zeros((n_cols, n_rows, n_used_dmaps))
		self.dataset = np.zeros((n_cols, n_rows, n_used_dmaps + n_used_chan))
		self.dataset_units = []
		self.dmaps_names = []
		self.dmaps_units = []
		self.chan_names = []
		self.chan_units = []
		self.dataset_names = []
		self.dataset_orig = np.zeros((n_cols, n_rows, n_used_chan, dataset_size))
		self.dataset_calibration = np.ones((n_used_chan, 3, dataset_size))
		self.n_energy = n_energy
		self.energy = np.zeros((n_energy))
		self.energy_spec = np.zeros((n_energy))
		self.max_chan_spec = np.zeros((n_energy, 5))
		self.raw_spec = np.zeros((n_raw_spec, n_raw_det))
		self.n_raw_det = n_raw_det
		self.img_type = 0
		self.us_amp = np.zeros((3))
		self.ds_amp = np.zeros((3))
		self.energy_fit = np.zeros((3, n_raw_det))
		self.add_long = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0}
		self.add_float = {'a': 0., 'b': 0., 'c': 0., 'd': 0., 'e': 0.}
		self.add_str = {'a': '', 'b': '', 'c': '', 'd': '', 'e': '', \
						'f': '', 'g': '', 'h': '', 'i': '', 'j': '', \
						'k': '', 'l': '', 'm': '', 'n': '', 'o': ''}
		self.extra_str_arr = []
		self.make_maps_conf = maps_conf

	def dump(self, data_file):
		data_file.write(struct.pack('>iiiii', self.n_ev, self.version, self.n_cols, self.n_rows, self.n_used_dmaps))
		data_file.write(struct.pack('>iss', self.n_used_chan, self.scan_time_stamp, self.write_date,))
		self.x_coord_arr.astype('>f').tofile(data_file)
		self.y_coord_arr.astype('>f').tofile(data_file)
		self.dmaps_set.astype('>f').tofile(data_file)
		self.dataset.astype('>f').tofile(data_file)
		for f in self.dataset_units:
			data_file.write(struct.pack('>s', f))
		for f in self.dmaps_names:
			data_file.write(struct.pack('>s', f))
		for f in self.dmaps_units:
			data_file.write(struct.pack('>s', f))
		for f in self.chan_names:
			data_file.write(struct.pack('>s', f))
		for f in self.chan_units:
			data_file.write(struct.pack('>s', f))
		for f in self.dataset_names:
			data_file.write(struct.pack('>s', f))
		self.dataset_orig.astype('>f').tofile(data_file)
		self.dataset_calibration.astype('>f').tofile(data_file)
		data_file.write(struct.pack('>i', self.n_energy))
		self.energy.astype('>f').tofile(data_file)
		self.energy_spec.astype('>f').tofile(data_file)
		self.max_chan_spec.astype('>f').tofile(data_file)
		self.raw_spec.astype('>f').tofile(data_file)
		data_file.write(struct.pack('>ii', self.n_raw_det, self.img_type))
		self.us_amp.astype('>f').tofile(data_file)
		self.ds_amp.astype('>f').tofile(data_file)
		self.energy_fit.astype('>f').tofile(data_file)
		for key in self.add_long.iterkeys():
			data_file.write(struct.pack('>si', key, self.add_long[key]))
		for key in self.add_float.iterkeys():
			data_file.write(struct.pack('>sf', key, self.add_float[key]))
		for key in self.add_str.iterkeys():
			data_file.write(struct.pack('>ss', key, self.add_str[key]))
		for f in self.extra_str_arr:
			data_file.write(struct.pack('>s', f))



# -----------------------------------------------------------------------------
class spectrum:
	def __init__(self, max_spec_channels, max_ICs, mode='spec_tool'):
		self.mode = mode
		self.name = ''
		self.data = np.zeros(max_spec_channels)
		self.used_chan = 0
		self.base_xsize = 0
		self.base_ysize = 0
		self.offset_xsize = 0
		self.offset_ysize = 0
		self.used = 0
		self.scan_time_stamp = ''
		self.real_time = 0.
		self.live_time = 0.
		self.srcurrent = 0.
		self.normalized = 0
		self.calib = {'off': 0., 'lin': 0., 'quad': 0.}
		self.date = {'year': 0, 'month': 0, 'day': 0, 'hour': 0, 'minute': 0, 'second': 0}
		self.roi = {'number': 0, 'area': 0., 'pixels': 0.}
		ic = {'cts': 0., 'sens_num': 0., 'sens_unit': 0., 'sens_factor': 0.}
		self.IC = [ic for count in range(max_ICs)]


# -----------------------------------------------------------------------------
class maps_definitions:
	def __init__(self, logger):
		self.logger = logger

	# -----------------------------------------------------------------------------
	def set_maps_definitions(self, beamline, info_elements, version=9, latest=True):

		if latest: version = 9

		verbose = False

		n_max = 20  # no of maximal supported detectors

		dmaps_list = ['SRcurrent', 'us_ic', 'ds_ic', 'abs_ic', 'ELT1', 'ERT1',
					  'x_coord', 'y_coord', 'dummy', 'dummy', 'dummy',
					  'dummy', 'dummy', 'dummy', 'dummy', 'dummy', 'dummy',
					  'dummy', 'dummy', 'dummy', 'dummy', 'dummy', 'dummy']

		element_list = ['Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co',
				'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc',
				'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Hf', 'dummy',
				'Mo_L', 'Ag_L', 'Sn_L', 'Cd_L', 'I_L', 'Cs_L', 'Ba_L', 'Eu_L', 'Gd_L', 'W_L', 'Pt_L', 'Au_L',
				'Hg_L', 'Pb_L', 'U_L', 'Pu_L', 'Sm_L', 'Y_L', 'Pr_L', 'Ce_L', 'Zr_L', 'Os_L', 'Rb_L', 'Ru_L', 'Hf_L'
				'Au_M', 'Pb_M', 'U_M', 'noise', 'dummy', 's_i', 's_e', 's_a', 'TFY', 'Bkgnd']

		list_use_these = ['Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Ti', 'Cr', 'Mn', 'Fe', 'Co',
						  'Ni', 'Cu', 'Zn', 'Pt_L', 'Au_M', 'Pb_M', 's_i', 's_e', 's_a', 'TFY', 'Bkgnd']

		if beamline == '2-ID-E':
			dmaps_list = ['SRcurrent', 'us_ic', 'ds_ic', 'abs_ic',
						  'abs_cfg', 'H_dpc_cfg', 'V_dpc_cfg', 'dia1_dpc_cfg', 'dia2_dpc_cfg',
						  'H_dpc_norm', 'V_dpc_norm', 'phase', 'ELT1', 'ERT1', 'ICR1', 'OCR1',
						  'deadT', 'x_coord', 'y_coord',
						  'dummy', 'dummy', 'dummy', 'dummy']

		if beamline == '2-ID-D':
			dmaps_list = ['SRcurrent', 'us_ic', 'ds_ic', 'abs_ic',
						  'abs_cfg', 'H_dpc_cfg', 'V_dpc_cfg', 'dia1_dpc_cfg', 'dia2_dpc_cfg',
						  'H_dpc_norm', 'V_dpc_norm', 'phase', 'ELT1', 'ERT1', 'ICR1', 'OCR1',
						  'deadT', 'x_coord', 'y_coord',
						  'dummy', 'dummy', 'dummy', 'dummy']

		if beamline == '2-ID-D':
			self.logger.info('main beamline is 2iDD')

		mcfg = maps_conf()
		if version == 9:
			mcfg.version = version
			mcfg.use_det = np.zeros(n_max, dtype=np.bool)
			mcfg.calibration.offset = np.zeros(n_max, dtype=np.float)
			mcfg.calibration.slope = np.zeros(n_max, dtype=np.float)
			mcfg.calibration.quad = np.zeros(n_max, dtype=np.float)

			'''
            mcfg.nbs32.name = 'nbs1832_'
            mcfg.nbs32.date = ''
            mcfg.nbs32.live_time = np.zeros(n_max, dtype = np.float)
            mcfg.nbs32.real_time = np.zeros(n_max, dtype = np.float)
            mcfg.nbs32.calibration.offset = np.zeros(n_max, dtype = np.float)
            mcfg.nbs32.calibration.slope = np.zeros(n_max, dtype = np.float)
            mcfg.nbs32.calibration.quad = np.zeros(n_max, dtype = np.float)
            mcfg.nbs33.name = 'nbs1833_'
            mcfg.nbs33.date = ''
            mcfg.nbs33.live_time = np.zeros(n_max, dtype = np.float)
            mcfg.nbs33.real_time = np.zeros(n_max, dtype = np.float)
            mcfg.nbs33.calibration.offset = np.zeros(n_max, dtype = np.float)
            mcfg.nbs33.calibration.slope = np.zeros(n_max, dtype = np.float)
            mcfg.nbs33.calibration.quad = np.zeros(n_max, dtype = np.float)
            mcfg.axo.name = ''
            mcfg.axo.date = ''
            mcfg.axo.live_time = np.zeros(n_max, dtype = np.float)
            mcfg.axo.real_time = np.zeros(n_max, dtype = np.float)
            mcfg.axo.calibration.offset = np.zeros(n_max, dtype = np.float)
            mcfg.axo.calibration.slope = np.zeros(n_max, dtype = np.float)
            mcfg.axo.calibration.quad = np.zeros(n_max, dtype = np.float)
            '''
			mcfg.element_standard.name = ''
			mcfg.element_standard.date = ''
			mcfg.element_standard.live_time = np.zeros(n_max, dtype=np.float)
			mcfg.element_standard.real_time = np.zeros(n_max, dtype=np.float)
			mcfg.element_standard.calibration.offset = np.zeros(n_max, dtype=np.float)
			mcfg.element_standard.calibration.slope = np.zeros(n_max, dtype=np.float)
			mcfg.element_standard.calibration.quad = np.zeros(n_max, dtype=np.float)

			mcfg.e_cal = np.zeros((len(element_list), 3, 7), dtype=np.float)
			# mcfg.axo_e_cal = np.zeros((len(list),3,7), dtype = np.float)
			mcfg.n_chan = len(element_list)
			mcfg.use_det[0] = 1

			for i in range(mcfg.n_chan):
				mcfg.chan.append(chan())
				mcfg.chan[i].name = element_list[i]
				mcfg.chan[i].left_roi = np.zeros(n_max, dtype=np.float)
				mcfg.chan[i].right_roi = np.zeros(n_max, dtype=np.float)
				mcfg.chan[i].left_bkground = np.zeros(n_max, dtype=np.float)
				mcfg.chan[i].right_bkground = np.zeros(n_max, dtype=np.float)

			mcfg.n_dmaps = len(dmaps_list)

			for i in range(mcfg.n_dmaps):
				mcfg.dmaps.append(dmaps())
				mcfg.dmaps[i].name = dmaps_list[i]

			for i in range(mcfg.n_dmaps):
				if dmaps_list[i] != 'dummy':
					mcfg.dmaps[i].use = 1
					mcfg.n_used_dmaps += 1

			for item in list_use_these:
				if item in element_list:
					mcfg.chan[element_list.index(item)].use = 1
					mcfg.n_used_chan += 1

			if 'TFY' in element_list:
				mcfg.chan[mcfg.n_chan - 2].center = 8.
				mcfg.chan[mcfg.n_chan - 2].width = 10000.

			for item in info_elements:
				if item.name in element_list:
					mcfg.chan[element_list.index(item.name)].z = item.z
					mcfg.chan[element_list.index(item.name)].center = item.xrf['ka1']
					mcfg.chan[element_list.index(item.name)].calib = 1
					if verbose:
						self.logger.debug('item.name: %s, item.z: %s, item.xrf: %s',item.name, item.z, item.xrf['ka1'])

				elname = string.join([item.name, '_L'], '')
				if elname in element_list:
					mcfg.chan[element_list.index(elname)].z = item.z
					mcfg.chan[element_list.index(elname)].center = item.xrf['la1']
					mcfg.chan[element_list.index(elname)].calib = 2
					if verbose:
						self.logger.debug('elname: %s, item.z: %s, item.xrf: %s', elname, item.z, item.xrf['la1'])

				elname = 'Pb_M'
				if elname in element_list:
					mcfg.chan[element_list.index(elname)].z = 82
					mcfg.chan[element_list.index(elname)].center = 2.383
					mcfg.chan[element_list.index(elname)].calib = 3

				elname = 'Au_M'
				if elname in element_list:
					mcfg.chan[element_list.index(elname)].z = 79
					mcfg.chan[element_list.index(elname)].center = 2.123
					mcfg.chan[element_list.index(elname)].calib = 3

				elname = 'U_M'
				if elname in element_list:
					mcfg.chan[element_list.index(elname)].z = 92
					mcfg.chan[element_list.index(elname)].center = 3.171
					mcfg.chan[element_list.index(elname)].calib = 3

			energy_res_offset = 150.  # in ev
			energy_res_sqrt = 12.  # for keV

			for ii in range(mcfg.n_chan):
				mcfg.chan[ii].width = np.int(
					np.sqrt(energy_res_offset ** 2 + (mcfg.chan[ii].center * energy_res_sqrt) ** 2))
				mcfg.chan[ii].bkground_left = np.int(
					np.sqrt(energy_res_offset ** 2 + (mcfg.chan[ii].center * energy_res_sqrt) ** 2) / 4.)
				mcfg.chan[ii].bkground_right = np.int(
					np.sqrt(energy_res_offset ** 2 + (mcfg.chan[ii].center * energy_res_sqrt) ** 2) / 4.)
				if verbose:
					self.logger.debug('width: %s bckg_left: %s bckg_right: %s ', mcfg.chan[ii].width, mcfg.chan[ii].bkground_left, mcfg.chan[ii].bkground_right)

			return mcfg

		# -----------------------------------------------------------------------------

	def define_xrfmaps_info(self, n_cols, n_rows, dataset_size, n_energy, n_raw_spec, n_raw_det, n_used_chan,
							n_used_dmaps, maps_conf, version=9, latest=True):

		self.xrf_info = XRFmaps_info(n_cols, n_rows, dataset_size, n_energy, n_raw_spec, n_raw_det, n_used_chan,
									 n_used_dmaps, maps_conf, version)

		return self.xrf_info

	# -----------------------------------------------------------------------------
	def xrfmaps_add_extra(self, extra_pv, extra_pv_order=0):

		stringlist = ['saveData_fileSystem', \
					  'saveData_subDir', \
					  'saveData_scanNumber', \
					  'userStringCalc10.AA', \
					  'userStringCalc10.BB', \
					  'userStringCalc10.CC', \
					  'userStringCalc10.DD', \
					  'userStringCalc10.EE']

		for item in extra_pv:
			# self.logger.debug(item, extra_pv[item]
			for si in range(len(stringlist)):
				if stringlist[si] in item:
					ind = si
					if ind == 0: self.xrf_info.add_str['a'] = extra_pv[item][2]
					if ind == 1: self.xrf_info.add_str['b'] = extra_pv[item][2]
					if ind == 2: self.xrf_info.add_str['c'] = extra_pv[item][2]
					if ind == 3: self.xrf_info.add_str['d'] = extra_pv[item][2]
					if ind == 4: self.xrf_info.add_str['e'] = extra_pv[item][2]
					if ind == 5: self.xrf_info.add_str['f'] = extra_pv[item][2]
					if ind == 6: self.xrf_info.add_str['g'] = extra_pv[item][2]
					if ind == 7: self.xrf_info.add_str['h'] = extra_pv[item][2]

		if extra_pv_order:
			count = 0
			for item in extra_pv_order:
				stemp = item + '; ' + str(extra_pv[item][2])
				self.xrf_info.extra_str_arr.append(stemp)
				count += 1
				if count == 100:
					break
		else:
			count = 0
			for item in extra_pv:
				stemp = item + '; ' + str(extra_pv[item][2])
				self.xrf_info.extra_str_arr.append(stemp)
				count += 1
				if count == 100:
					break

	# -----------------------------------------------------------------------------

	def define_spectra(self, max_spec_channels, max_spectra, max_ICs, mode='spec_tool'):

		spectra = [spectrum(max_spec_channels, max_ICs, mode) for count in range(max_spectra)]

		return spectra

	# -----------------------------------------------------------------------------

	def push_spectrum(self, filename, spectra,
					  append='',
					  n_channels=0,
					  n_detector_elements=0,
					  real_time=0,
					  live_time=0,
					  current=0,
					  calibration=0,
					  counts_us_ic=0,
					  counts_ds_ic=0,
					  roi_area=0,
					  roi_pixels=0,
					  us_amp=0,
					  ds_amp=0,
					  n_spatial_rois=0,
					  year=0,
					  data=0,
					  name=0,
					  DO_NOT_MOD_name='',
					  date=''):

		if append > 0:
			temp_used = []
			for item in spectra.used_chan: temp_used.append(item.use)
			wo = np.where(np.array(temp_used) > 0.)
			wo = wo[0]
			if wo.size != 0:
				wo = np.amax(wo)
			else:
				wo = -1
		else:
			wo = -1

		if append > 0:
			shortname = filename.split('/')
			shortname = shortname[-1]
			shortname = shortname.split('\\')
			shortname = shortname[-1]

		month = 0
		year = 0
		day = 0
		hour = 0
		minute = 0
		date = date.strip()
		if date != '':
			test = date[0:3]
			# test which of the two formats is used
			if (test == 'Mon') or (test == 'Tue') or (test == 'Wed') or (test == 'Thu') or (test == 'Fri') or (
						test == 'Sat') or (test == 'Sun'):
				year_pos = 20
				month_pos = 4
				day_pos = 8
				hour_pos = 11
				minute_pos = 13
			else:
				year_pos = 8
				month_pos = 0
				day_pos = 4
				hour_pos = 13
				minute_pos = 16

			test = date[month_pos: month_pos + 3].lower()
			if test == 'jan': month = 1
			if test == 'feb': month = 2
			if test == 'mar': month = 3
			if test == 'apr': month = 4
			if test == 'may': month = 5
			if test == 'jun': month = 6
			if test == 'jul': month = 7
			if test == 'aug': month = 8
			if test == 'sep': month = 9
			if test == 'oct': month = 10
			if test == 'nov': month = 11
			if test == 'dec': month = 12
			try:
				test = date[year_pos:(year_pos + 4)]
				year = int(test)
				test = date[day_pos: day_pos + 2]
				day = int(test)
				test = date[hour_pos:hour_pos + 2]
				hour = int(test)
				test = date[minute_pos: minute_pos + 2]
				minute = int(test)
			except:
				self.logger.exception('push_spectrum(): Could not convert date.')

		if len(data.shape) > 1:
			for k in range(n_spatial_rois):
				for l in range(n_detector_elements):
					i = int(l + k * n_detector_elements)
					j = int(i + wo + 1)
					if np.sum(data[:, i]) > 0.:
						spectra[j].name = ('det' + str(l) + '_roi' + str(k)).strip()
						if DO_NOT_MOD_name > 0:
							if name != '': spectra[j].name = name.strip()
						else:
							if name != '': spectra[j].name = (name + '_det' + str(l) + '_roi' + str(k)).strip()

						if append > 0: spectra[j].name = (shortname + '_' + spectra[j].name).strip()
						spectra[j].used_chan = n_channels
						spectra[j].data[0:spectra[j].used_chan] = data[0:spectra[j].used_chan, i]
						spectra[j].real_time = real_time[l]
						spectra[j].live_time = live_time[l]
						spectra[j].SRcurrent = current[0]
						spectra[j].calib['off'] = calibration[l]['offset']
						spectra[j].calib['lin'] = calibration[l]['slope']
						spectra[j].calib['quad'] = calibration[l]['quad']
						spectra[j].IC[0]['cts'] = counts_us_ic[0]
						for kk in range(1):
							if kk == 0: temp = us_amp
							if kk == 1: temp = ds_amp
							spectra[j].IC[kk].sens_num = float(temp[0])
							spectra[j].IC[kk].sens_unit = float(temp[1])
							spectra[j].IC[kk].sens_factor = float(temp[2])

						spectra[j].IC[1]['cts'] = counts_ds_ic[0]
						spectra[j].date['year'] = year[0]
						spectra[j].date['month'] = month[0]
						spectra[j].date['day'] = day[0]
						spectra[j].date['hour'] = hour[0]
						spectra[j].date['minute'] = minute[0]
						spectra[j].roi['area'] = roi_area[k]
						spectra[j].roi['pixels'] = roi_pixels[k]
					else:
						spectra[j].used_chan = 0L
		else:
			for k in range(n_spatial_rois):
				for l in range(n_detector_elements):
					i = int(l + k * n_detector_elements)
					j = int(i + wo + 1)
					if np.sum(data[:]) > 0.:
						spectra[j].name = ('det' + str(l) + '_roi' + str(k)).strip()
						if DO_NOT_MOD_name > 0:
							if name != '': spectra[j].name = name.strip()
						else:
							if name != '': spectra[j].name = (name + '_det' + str(l) + '_roi' + str(k)).strip()

						if append > 0: spectra[j].name = (shortname + '_' + spectra[j].name).strip()
						spectra[j].used_chan = n_channels
						spectra[j].data[0:spectra[j].used_chan] = data[0:spectra[j].used_chan]
						spectra[j].real_time = real_time
						spectra[j].live_time = live_time
						spectra[j].SRcurrent = current
						spectra[j].calib['off'] = calibration['offset']
						spectra[j].calib['lin'] = calibration['slope']
						spectra[j].calib['quad'] = calibration['quad']
						spectra[j].IC[0]['cts'] = counts_us_ic
						for kk in range(1):
							if kk == 0: temp = us_amp
							if kk == 1: temp = ds_amp
							spectra[j].IC[kk]['sens_num'] = float(temp[0])
							spectra[j].IC[kk]['sens_unit'] = float(temp[1])
							spectra[j].IC[kk]['sens_factor'] = float(temp[2])

						spectra[j].IC[1]['cts'] = counts_ds_ic
						spectra[j].date['year'] = year
						spectra[j].date['month'] = month
						spectra[j].date['day'] = day
						spectra[j].date['hour'] = hour
						spectra[j].date['minute'] = minute
						spectra[j].roi['area'] = roi_area
						spectra[j].roi['pixels'] = roi_pixels
					else:
						spectra[j].used_chan = 0L

		return
