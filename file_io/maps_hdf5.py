'''
Created on Nov 26, 2011

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
import os
import numpy as np
import h5py 
import time
from file_util import open_file_with_retry, call_function_with_retry

import maps_definitions
import maps_mda

#-----------------------------------------------------------------------------
class h5:
	def __init__(self):  
		pass

#-----------------------------------------------------------------------------	 
	def write_mca_hdf5(self, filename, mca_arr, overwrite = True):
		
		# set compression level where applicable:
		gzip = 5
		file_status = 0
		entry_exists = 0
		
		verbose = 0

		# test whether a file with this filename already exists:
		try:
			# Open HDF5 file
			f = h5py.File(filename, 'r')
			if verbose: print 'Have HDF5 file: ', filename
			file_exists = 1
			file_is_hdf = 1
			file_status = 2		  
			
			#MAPS HDF5 group
			if 'MAPS' in f:
				if verbose: print 'MAPS group found in file: ', filename
				mapsGrp = f['MAPS']
				file_status = 3
				if 'mca_arr' in mapsGrp:
					if verbose: print 'MAPS\\mca_arr found in file: ', filename
					file_status = 4
				# at the moment, simply overwrite the mca_arr section of
				# the file; in the future, may want to test, and only
				# overwrite if specific flag is set.

			f.close()

		except:
			if verbose: print 'Creating new file: ', filename
			
		if verbose: print 'file_status: ', file_status
		
		if overwrite : file_status = 0
		
		if file_status <= 1 : 
			f = call_function_with_retry(h5py.File, 5, 0.1, 1.1, (filename, 'w'))
			#f = h5py.File(filename, 'w')
		else : 
			f = call_function_with_retry(h5py.File, 5, 0.1, 1.1, (filename, 'a'))
			#f = h5py.File(filename, 'a')

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
			
		f.close()
		return
	
#-----------------------------------------------------------------------------	 
	def write_hdf5(self, thisdata, filename, mca_arr, energy_channels, extra_pv=None, extra_pv_order=None, update=False):


		#set compression level where applicable:
		gzip = 7

		if update == False:
			f = call_function_with_retry(h5py.File, 5, 0.1, 1.1, (filename, 'w'))
			#f = h5py.File(filename, 'w')
			# create a group for maps to hold the data
			mapsGrp = f.create_group('MAPS')
			# now set a comment
			mapsGrp.attrs['comments'] = 'This is the group that stores all relevant information created (and read) by the the MAPS analysis software'
		else:
			f = call_function_with_retry(h5py.File, 5, 0.1, 1.1, (filename, 'a'))
			#f = h5py.File(filename, 'a')
			if 'MAPS' not in f:
				print 'error, hdf5 file does not contain the required MAPS group. I am aborting this action'
				return 
			mapsGrp = f['MAPS']

		if 'XRF_roi' in mapsGrp:
			del mapsGrp['XRF_roi']
		# this is the data we want to write into the hdf5 file
		entryname = 'XRF_roi'
		comment = 'these are elemental maps created from full spectra using spectral ROIs'
		data = np.transpose(thisdata.dataset_orig[:, :, :, 0])
		# choose an image / map as a chunk
		dimensions = data.shape
		chunk_dimensions = (1, dimensions[1], dimensions[2])
		ds_data = mapsGrp.create_dataset(entryname, data=data, chunks=chunk_dimensions, compression='gzip', compression_opts=gzip)
		ds_data.attrs['comments'] = comment
		#print 'total of data 0', np.sum(data)
		
		if 'XRF_fits' in mapsGrp:
			del mapsGrp['XRF_fits']		   
		entryname = 'XRF_fits'
		comment = 'these are elemental maps created from full spectra using per pixel fitting'
		data = np.transpose(thisdata.dataset_orig[:, :, :, 1])
		dimensions = data.shape
		chunk_dimensions = (1, dimensions[1], dimensions[2])
		if np.sum(data) != 0.0:
			contains_fitted_data = 1 
		else:
			contains_fitted_data = 0
		if contains_fitted_data :
			ds_data = mapsGrp.create_dataset(entryname, data=data, chunks=chunk_dimensions, compression='gzip', compression_opts=gzip)
			ds_data.attrs['comments'] = comment

		if 'XRF_roi_plus' in mapsGrp:
			del mapsGrp['XRF_roi_plus']			
		entryname = 'XRF_roi_plus'
		comment = 'these are elemental maps created from full spectra accounting for crosstalk between channels / elements'
		data = np.transpose(thisdata.dataset_orig[:, :, :, 2])
		if np.sum(data) != 0.:
			contains_roiplus_data = 1 
		else:
			contains_roiplus_data = 0
		dimensions = data.shape
		chunk_dimensions = (1, dimensions[1], dimensions[2])
		if contains_roiplus_data :
			ds_data = mapsGrp.create_dataset(entryname, data=data, chunks=chunk_dimensions, compression='gzip', compression_opts=gzip)
			ds_data.attrs['comments'] = comment
		print 'total of data 2', np.sum(data)

		entryname = 'scalers'
		comment = 'these are scaler information acquired during the scan'
		data = np.transpose(thisdata.dmaps_set[:, :, :])
		dimensions = data.shape
		chunk_dimensions = (1, dimensions[1], dimensions[2])
		ds_data = mapsGrp.create_dataset(entryname, data=data, chunks=chunk_dimensions, compression='gzip', compression_opts=gzip)
		ds_data.attrs['comments'] = comment

		entryname = 'x_axis'
		if entryname not in mapsGrp:
			comment = 'stores the values of the primary fast axis positioner, typically sample x'
			data = thisdata.x_coord_arr
			ds_data = mapsGrp.create_dataset(entryname, data=data)
			ds_data.attrs['comments'] = comment

		entryname = 'y_axis'
		if entryname not in mapsGrp:
			comment = 'stores the values of the slow axis positioner, typically sample y'
			data = thisdata.y_coord_arr
			ds_data = mapsGrp.create_dataset(entryname, data=data)
			ds_data.attrs['comments'] = comment

		entryname = 'energy'
		comment = 'stores the values of the energy axis'
		data = thisdata.energy
		ds_data = mapsGrp.create_dataset(entryname, data=data)
		ds_data.attrs['comments'] = comment

		# now write integrated spectrum as dataset
		entryname = 'int_spec'
		comment = 'spectrum integrated over the full dataset'
		data = thisdata.energy_spec
		ds_data = mapsGrp.create_dataset(entryname, data=data)
		ds_data.attrs['comments'] = comment

		# now write max channel spectrum as dataset
		entryname = 'max_chan_spec'
		comment = 'several maximum channel spectra integrated over the full dataset'
		data = np.transpose(thisdata.max_chan_spec)
		ds_data = mapsGrp.create_dataset(entryname, data=data)
		ds_data.attrs['comments'] = comment

		# now write quantification of roi dataset
		entryname = 'XRF_roi_quant'
		comment = 'quantification curve for the ROI based dataset'
		data = np.transpose(thisdata.dataset_calibration[:, 0, :])
		data = np.reshape(data, (data.shape[0], 1, data.shape[1]))
		ds_data = mapsGrp.create_dataset(entryname, data=data)
		ds_data.attrs['comments'] = comment

		# now write quantification of fits dataset
		entryname = 'XRF_fits_quant'
		comment = 'quantification curve for the fits based dataset'
		data = np.transpose(thisdata.dataset_calibration[:, 1, :])
		data = np.reshape(data, (data.shape[0], 1, data.shape[1]))
		if contains_fitted_data:
			ds_data = mapsGrp.create_dataset(entryname, data=data)
			ds_data.attrs['comments'] = comment
			
		# now write quantification of ROI+ dataset
		entryname = 'XRF_roi_plus_quant'
		comment = 'quantification curve for the datasets based on ROI + definitions ( with background subtraction)'
		data = np.transpose(thisdata.dataset_calibration[:, 2, :])
		data = np.reshape(data, (data.shape[0], 1, data.shape[1]))
		if contains_roiplus_data:
			ds_data = mapsGrp.create_dataset(entryname, data=data)
			ds_data.attrs['comments'] = comment

		entryname = 'us_amp'
		comment = 'sensitivity of the upstream amplifier'
		data = thisdata.us_amp
		ds_data = mapsGrp.create_dataset(entryname, data=data)
		ds_data.attrs['comments'] = comment

		entryname = 'ds_amp'
		comment = 'sensitivity of the downstream amplifier'
		data = thisdata.ds_amp
		ds_data = mapsGrp.create_dataset(entryname, data=data)
		ds_data.attrs['comments'] = comment

		try:
			entryname = 'energy_calib'
			comment = 'energy calibration'
			data = thisdata.energy_fit
			ds_data = mapsGrp.create_dataset(entryname, data=data)
			ds_data.attrs['comments'] = comment
		except:
			print 'Error: HDF5: Could not write energy calibration'

		entryname = 'version'
		comment = 'this is the version number of the file structure'
		data = thisdata.version
		ds_data = mapsGrp.create_dataset(entryname, data=data)
		ds_data.attrs['comments'] = comment

		try:
			entryname = 'scan_time_stamp'
			comment = 'time the scan was acquired'
			data = thisdata.scan_time_stamp
			ds_data = mapsGrp.create_dataset(entryname, data=data)
			ds_data.attrs['comments'] = comment
		except:
			print 'HDF5: Could not write scan_time_stamp'

		entryname = 'write_date'
		comment = 'time this analysis was carried out'
		data = str(thisdata.write_date)
		ds_data = mapsGrp.create_dataset(entryname, data=data)
		ds_data.attrs['comments'] = comment

		entryname = 'scaler_names'
		comment = 'names of the scalers saved'
		data = thisdata.dmaps_names
		ds_data = mapsGrp.create_dataset(entryname, data=data)
		ds_data.attrs['comments'] = comment
		
		entryname = 'scaler_units'
		comment = 'units of the scalers saved'
		data = thisdata.dmaps_units
		ds_data = mapsGrp.create_dataset(entryname, data=data)
		ds_data.attrs['comments'] = comment  

		entryname = 'channel_names'
		comment = 'names of the channels saved'
		data = thisdata.chan_names
		ds_data = mapsGrp.create_dataset(entryname, data=data)
		ds_data.attrs['comments'] = comment

		entryname = 'channel_units'
		comment = 'units of the channels saved'
		data = thisdata.chan_units[:]
		data = zip(*data)
		ds_data = mapsGrp.create_dataset(entryname, data=data)
		ds_data.attrs['comments'] = comment

		entryname = 'extra_strings'
		comment = 'additional string values saved in the dataset'
		data = thisdata.extra_str_arr
		ds_data = mapsGrp.create_dataset(entryname, data=data)
		ds_data.attrs['comments'] = comment 

		entryname = 'add_long'
		comment = 'additional long values'
		data = [thisdata.add_long['a'], thisdata.add_long['b'], thisdata.add_long['c'], thisdata.add_long['d'], thisdata.add_long['e']]
		ds_data = mapsGrp.create_dataset(entryname, data=data)
		ds_data.attrs['comments'] = comment

		entryname = 'add_float'
		comment = 'additional float values'
		data = [thisdata.add_float['a'], thisdata.add_float['b'], thisdata.add_float['c'], thisdata.add_float['d'], thisdata.add_float['e']]
		ds_data = mapsGrp.create_dataset(entryname, data=data)
		ds_data.attrs['comments'] = comment  
		
		entryname = 'add_string'
		comment = 'additional string values'
		data = [thisdata.add_str['a'], thisdata.add_str['b'], thisdata.add_str['c'], thisdata.add_str['d'], thisdata.add_str['e'], 
				thisdata.add_str['f'], thisdata.add_str['g'], thisdata.add_str['h'], thisdata.add_str['i'], thisdata.add_str['j'], 
				thisdata.add_str['k'], thisdata.add_str['l'], thisdata.add_str['m'], thisdata.add_str['n'], thisdata.add_str['o']]
		ds_data = mapsGrp.create_dataset(entryname, data=data)
		ds_data.attrs['comments'] = comment  
		
		if update == False: 
			print 'saving full spectra to hdf5'
			# now save full spectra
			entryname = 'mca_arr'
			comment = 'these are the full spectra at each pixel of the dataset'
			data = np.transpose(mca_arr)
			dimensions = data.shape
			chunk_dimensions = (dimensions[0], 1, 1)
			ds_data = mapsGrp.create_dataset(entryname, data=data, chunks=chunk_dimensions, compression='gzip', compression_opts=gzip)
			ds_data.attrs['comments'] = comment

		# create a subgroup FOR make_maps_conf
		if 'make_maps_conf' not in mapsGrp:
			mmcGrp = mapsGrp.create_group('make_maps_conf')
		else:
			mmcGrp = mapsGrp['make_maps_conf']
		
		entryname = 'use_default_dirs'
		comment = ''
		data = thisdata.make_maps_conf.use_default_dirs
		ds_data = mmcGrp.create_dataset(entryname, data=data)
		ds_data.attrs['comments'] = comment		   

		entryname = 'use_beamline'
		data = thisdata.make_maps_conf.use_beamline
		ds_data = mmcGrp.create_dataset(entryname, data=data)
		ds_data.attrs['comments'] = comment 

		entryname = 'version'
		data = thisdata.make_maps_conf.version
		ds_data = mmcGrp.create_dataset(entryname, data=data)
		ds_data.attrs['comments'] = comment 

		entryname = 'use_det'
		data = thisdata.make_maps_conf.use_det
		ds_data = mmcGrp.create_dataset(entryname, data=data)
		ds_data.attrs['comments'] = comment  

		entryname = 'calibration_offset'
		data = thisdata.make_maps_conf.calibration.offset
		ds_data = mmcGrp.create_dataset(entryname, data=data)
		ds_data.attrs['comments'] = comment 

		entryname = 'calibration_slope'
		data = thisdata.make_maps_conf.calibration.slope
		ds_data = mmcGrp.create_dataset(entryname, data=data)
		ds_data.attrs['comments'] = comment 

		entryname = 'calibration_quad'
		data = thisdata.make_maps_conf.calibration.quad
		ds_data = mmcGrp.create_dataset(entryname, data=data)
		ds_data.attrs['comments'] = comment 
		'''
		if 'nbs1832' not in mmcGrp:
			nbs1832Grp = mmcGrp.create_group('nbs1832')
		else:
			nbs1832Grp = mmcGrp['nbs1832']
		
		substructure = thisdata.make_maps_conf.nbs32  
		entryname = 'name'
		data = substructure.name
		ds_data = nbs1832Grp.create_dataset(entryname, data = data)  
		entryname = 'date'
		data = substructure.date
		ds_data = nbs1832Grp.create_dataset(entryname, data = data)  
		entryname = 'live_time'
		data = substructure.live_time
		ds_data = nbs1832Grp.create_dataset(entryname, data = data)  
		entryname = 'real_time'
		data = substructure.real_time
		ds_data = nbs1832Grp.create_dataset(entryname, data = data)  
		entryname = 'current'
		data = substructure.current
		ds_data = nbs1832Grp.create_dataset(entryname, data = data)  
		entryname = 'us_ic'
		data = substructure.us_ic
		ds_data = nbs1832Grp.create_dataset(entryname, data = data)  
		entryname = 'ds_ic'
		data = substructure.ds_ic
		ds_data = nbs1832Grp.create_dataset(entryname, data = data)  
		entryname = 'us_amp'
		data = substructure.us_amp
		ds_data = nbs1832Grp.create_dataset(entryname, data = data)
		entryname = 'ds_amp'
		data = substructure.ds_amp
		ds_data = nbs1832Grp.create_dataset(entryname, data = data)

		if 'nbs1833' not in mmcGrp:
			nbs1833Grp = mmcGrp.create_group('nbs1833')
		else:
			nbs1833Grp = mmcGrp['nbs1833']
				   
		substructure = thisdata.make_maps_conf.nbs33  
		entryname = 'name'
		data = substructure.name
		ds_data = nbs1833Grp.create_dataset(entryname, data = data)
		entryname = 'date'
		data = substructure.date
		ds_data = nbs1833Grp.create_dataset(entryname, data = data)  
		entryname = 'live_time'
		data = substructure.live_time
		ds_data = nbs1833Grp.create_dataset(entryname, data = data) 
		entryname = 'real_time'
		data = substructure.real_time
		ds_data = nbs1833Grp.create_dataset(entryname, data = data) 
		entryname = 'current'
		data = substructure.current
		ds_data = nbs1833Grp.create_dataset(entryname, data = data) 
		entryname = 'us_ic'
		data = substructure.us_ic
		ds_data = nbs1833Grp.create_dataset(entryname, data = data) 
		entryname = 'ds_ic'
		data = substructure.ds_ic
		ds_data = nbs1833Grp.create_dataset(entryname, data = data)  
		entryname = 'us_amp'
		data = substructure.us_amp
		ds_data = nbs1833Grp.create_dataset(entryname, data = data)
		entryname = 'ds_amp'
		data = substructure.ds_amp
		ds_data = nbs1833Grp.create_dataset(entryname, data = data)
		'''

		entryname = 'e_cal'
		data = np.transpose(thisdata.make_maps_conf.e_cal)
		ds_data = mmcGrp.create_dataset(entryname, data = data)  
		'''
		if thisdata.version >= 9 :
			entryname = 'axo_e_cal'
			data = np.transpose(thisdata.make_maps_conf.axo_e_cal)
			ds_data = mmcGrp.create_dataset(entryname, data = data)  
		'''
		entryname = 'fit_t_be'
		data = thisdata.make_maps_conf.fit_t_be
		ds_data = mmcGrp.create_dataset(entryname, data = data)

		entryname = 'fit_t_ge'
		data = thisdata.make_maps_conf.fit_t_GE
		ds_data = mmcGrp.create_dataset(entryname, data = data)

		if extra_pv: 
			entryname = 'extra_pvs'
			comment = 'additional process variables saved in the original dataset'
			data = []
			print extra_pv
			if extra_pv_order:
				for k in extra_pv_order:
					v = extra_pv[k]
					data.append([k, str(v[2]), v[0], v[1]])				   
			else:
				for k in sorted(extra_pv.iterkeys()):
					v = extra_pv[k]
					data.append([k, str(v[2]), v[0], v[1]])
			ds_data = mapsGrp.create_dataset(entryname, data = np.transpose(data))

			entryname = 'extra_pvs_as_csv'
			comment = 'additional process variables saved in the original dataset, name and value fields reported as comma seperated values'
			if extra_pv_order:
				data = []
				for k in extra_pv_order:
					v = extra_pv[k]
					data.append(k + ', '+ str(v[2]))
				ds_data = mapsGrp.create_dataset(entryname, data = data) 
			else:
				ds_data = mapsGrp.create_dataset(entryname, data = ds_data)  

		f.close()
		return

#-----------------------------------------------------------------------------	 
	def maps_change_xrf_read_hdf5(self, sfile, make_maps_conf):
	
		maps_def = maps_definitions.maps_definitions()

		f = call_function_with_retry(h5py.File, 5, 0.1, 1.1, (sfile, 'r'))
		if f == None:
			print 'Error could not open file ', sfile
			return None, None, None, None, 0

		if 'MAPS' not in f:
			print 'error, hdf5 file does not contain the required MAPS group. I am aborting this action'
			return None, None, None, None, 0

		maps_group_id = f['MAPS']

		entryname = 'XRF_roi'
		this_xrfdata, valid_read = self.read_hdf5_core(maps_group_id, entryname)
		if valid_read == 0:
			print 'error, reading', entryname
			return None, None, None, None, 0
		this_xrfdata = np.transpose(this_xrfdata)
		dimensions = this_xrfdata.shape

		# if this is a 2D (x, y) scan dimensions should be 3  
		n_cols = dimensions[0]
		n_rows = dimensions[1]
		n_used_chan = dimensions[2]

		entryname = 'scalers'
		this_scalers, valid_read = self.read_hdf5_core(maps_group_id, entryname)
		if valid_read == 0:
			print 'error, reading', entryname
			return None, None, None, None, 0
		this_scalers = this_scalers.transpose()
		dimensions = this_scalers.shape
		n_used_dmaps = dimensions[2]
		
		entryname = 'energy'
		this_energy, valid_read = self.read_hdf5_core(maps_group_id, entryname)
		if valid_read == 0:
			print 'error, reading', entryname
			return None, None, None, None, 0
		dimensions = this_energy.shape
		n_channels = dimensions[0]

		# default, one for roi based , one for fitted images and one for sigma.
		dataset_size = 3

		# any current maps version deal with multiple detectors in creating
		# different files, one for each detector, and at the end one
		# average  
		no_detectors = 1
		
		version = 9
		
		XRFmaps_info = maps_def.define_xrfmaps_info(n_cols, n_rows, dataset_size,
													n_channels, n_channels, no_detectors, 
													n_used_chan, n_used_dmaps, 
													make_maps_conf, version = 9)  

		XRFmaps_info.n_ev = n_channels
		XRFmaps_info.n_energy = n_channels
		XRFmaps_info.energy = this_energy
		XRFmaps_info.dmaps_set = this_scalers

		
		XRFmaps_info.dataset_names = ['ROI sum', 'fitted', 'sigma']
		for i in range(n_used_chan):
			XRFmaps_info.dataset_orig[:, :, i, 0] = this_xrfdata[:,:,i]
		this_xrfdata = 0

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'scan_time_stamp')
		if valid_read:
			XRFmaps_info.scan_time_stamp = this_data

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'write_date')
		if valid_read:
			XRFmaps_info.write_date = this_data

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'x_axis')
		if valid_read:
			XRFmaps_info.x_coord_arr = this_data

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'y_axis')
		if valid_read:
			XRFmaps_info.y_coord_arr = this_data

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'scaler_names')
		if valid_read:
			XRFmaps_info.dmaps_names = this_data

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'scaler_units')
		if valid_read:
			XRFmaps_info.dmaps_units = this_data

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'channel_names')
		if valid_read:
			XRFmaps_info.chan_names = this_data

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'channel_units')
		if valid_read:
			XRFmaps_info.chan_units = zip(*this_data)

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'XRF_fits')
		this_data = np.transpose(this_data)
		if valid_read:
			for i in range(n_used_chan):
				XRFmaps_info.dataset_orig[:, :, i, 1] = this_data[:, :, i]

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'XRF_sigma')
		this_data = np.transpose(this_data)
		if valid_read:
			for i in range(n_used_chan):
				XRFmaps_info.dataset_orig[:, :, i, 2] = this_data[:, :, i]

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'XRF_roi_plus')
		this_data = np.transpose(this_data)
		if valid_read:
			for i in range(n_used_chan):
				XRFmaps_info.dataset_orig[:, :, i, 2] = this_data[:, :, i]
			XRFmaps_info.dataset_names[2] = 'XRF_roi+'

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'XRF_roi_quant')
		this_data = np.transpose(this_data)
		if valid_read:
			XRFmaps_info.dataset_calibration[:, 0, :] = this_data[:, 0, :]

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'XRF_roi_plus_quant')
		this_data = np.transpose(this_data)
		if valid_read:
			XRFmaps_info.dataset_calibration[:, 2, :] = this_data[:, 0, :]

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'XRF_fits_quant')
		this_data = np.transpose(this_data)
		if valid_read:
			XRFmaps_info.dataset_calibration[:, 1, :] = this_data[:, 0, :]

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'energy')
		if valid_read:
			XRFmaps_info.energy_spec = this_data

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'int_spec')
		if valid_read:
			XRFmaps_info.energy_spec = this_data

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'energy_calib')
		if valid_read:
			XRFmaps_info.energy_fit = this_data

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'max_chan_spec')
		this_data = np.transpose(this_data)
		if valid_read:
			XRFmaps_info.max_chan_spec = this_data

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'us_amp')
		if valid_read:
			XRFmaps_info.us_amp = this_data

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'ds_amp')
		if valid_read:
			XRFmaps_info.ds_amp = this_data

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'extra_strings')
		if valid_read:
			XRFmaps_info.extra_str_arr = this_data

		XRFmaps_info.img_type = 7
		
		f.close()
		
		return XRFmaps_info, n_cols, n_rows, n_channels, valid_read
		
			
#-----------------------------------------------------------------------------	 
	def read_hdf5_core(self, gid, entryname, verbose=False):
		valid_read = 0
		this_data = 0

		if entryname not in gid:
			if verbose:
				print 'read error: '
				print 'did not find the entry: ', entryname, 'in:'
				print gid
			return this_data, valid_read

		dataset_id = gid[entryname]
		this_data = dataset_id[...]
		valid_read = 1

		return this_data, valid_read

#-----------------------------------------------------------------------------	 
	def read_hdf5(self, filename):
		
		import maps_elements
		me = maps_elements.maps_elements()
		info_elements = me.get_element_info()
	
		maps_def = maps_definitions.maps_definitions()
		maps_conf = maps_def.set_maps_definitions('2-ID-E', info_elements)

		XRFmaps_info, n_cols, n_rows, n_channels, valid_read = self.maps_change_xrf_read_hdf5(filename, maps_conf)
		
		return XRFmaps_info, valid_read

#-----------------------------------------------------------------------------	 
	def add_exchange(self, main, make_maps_conf):
		

		files = os.listdir(main['XRFmaps_dir'])
		imgdat_filenames = []
		extension = '.h5'
		for f in files:
			if extension in f.lower():
				imgdat_filenames.append(f)	  
				

		gzip = 7

		no_files = len(imgdat_filenames)

		current_directory = main['master_dir']
	
		for n_filenumber in range(no_files):
			sFile = os.path.join(main['XRFmaps_dir'], imgdat_filenames[n_filenumber])
			
			print 'Adding exchange to ', sFile
		
			XRFmaps_info, n_cols, n_rows, n_channels, valid_read = self.maps_change_xrf_read_hdf5(sFile, make_maps_conf)
			if valid_read == 0:
				print 'Error calling h5p.maps_change_xrf_read_hdf5(', sFile, make_maps_conf, ')'
				return

			f = call_function_with_retry(h5py.File, 5, 0.1, 1.1, (sFile, 'a'))
			#f = h5py.File(sFile, 'a')
			if 'MAPS' not in f:
				print 'error, hdf5 file does not contain the required MAPS group. I am aborting this action'
				return 
			mapsGrp = f['MAPS']


			# create a subgroup for exchange
			if 'exchange' not in f:
				excGrp = f.create_group('exchange')
				excGrp.attrs['comments'] = 'This is the group that stores a default analysed dataset'
			else:
				excGrp = f['exchange']
					
			entryname = 'images'
			#comment = 'these are elemental maps'

			drop_val = 1
			data = XRFmaps_info.dataset_orig[:, :, :, drop_val]
			comment = 'these are elemental maps based on per pixel fitting'
			if np.sum(data) == 0.0:
				drop_val = 2
				data = XRFmaps_info.dataset_orig[:, :, :, drop_val]
				comment = 'these are elemental maps based on roi plus'

			if np.sum(data) == 0.0:
				drop_val = 0
				data = XRFmaps_info.dataset_orig[:, :, :, drop_val]
				comment = 'these are elemental maps based on rois'

			dataset = XRFmaps_info.dataset
			dataset[:, :, 0:XRFmaps_info.n_used_dmaps] = XRFmaps_info.dmaps_set[:, :, :]
			dataset[:, :, XRFmaps_info.n_used_dmaps:XRFmaps_info.n_used_dmaps+XRFmaps_info.n_used_chan] = data[:, :, :]
		
			drop_vtwo = 0  # for now just use ds ic for normalization
			if drop_vtwo == 0:
				wo = []
				if 'ds_ic' in XRFmaps_info.dmaps_names: 
					wo = np.where(XRFmaps_info.dmaps_names == 'ds_ic')
			#if drop_vtwo == 1 : wo = XRFmaps_info.dmaps_names.index('us_ic')
			#if drop_vtwo == 2 : wo = XRFmaps_info.dmaps_names.index('SRcurrent')
			ic_correction_factor = 1.		   
			#if (drop_vtwo == 0) and (XRFmaps_info.make_maps_conf.nbs32.ds_amp[2] > 0.) and \
			#	(XRFmaps_info.ds_amp[2] > 0.) :
			#	ic_correction_factor = XRFmaps_info.make_maps_conf.nbs32.ds_amp[2]/XRFmaps_info.ds_amp[2]
			#if (drop_vtwo == 1) and (XRFmaps_info.make_maps_conf.nbs32.us_amp[2] > 0.) and \
			#	(XRFmaps_info.us_amp[2] > 0.) :
			#	ic_correction_factor = XRFmaps_info.make_maps_conf.nbs32.us_amp[2]/XRFmaps_info.us_amp[2]
			if len(wo[0]) > -1 : 
				calib = XRFmaps_info.dmaps_set[:, :, wo].astype(float) 
				calib = calib[:, :, 0, 0]
				for k in range(XRFmaps_info.n_used_dmaps, XRFmaps_info.n_used_dmaps + XRFmaps_info.n_used_chan):
					calib_factor = float(XRFmaps_info.dataset_calibration[k - XRFmaps_info.n_used_dmaps, drop_val, 2-drop_vtwo])
					if calib_factor > 0:
						dataset[:, :, k] = dataset[:, :, k] / calib_factor / calib
					else:
						dataset[:, :, k] = dataset[:, :, k] / calib * np.mean(calib)
					dataset[:, :, k] = dataset[:, :, k] * ic_correction_factor

			data = np.transpose(dataset)
			dimensions = data.shape
			chunk_dimensions = (1, dimensions[1], dimensions[2])

			if entryname not in excGrp:
				ds_data = excGrp.create_dataset(entryname, data = data, chunks=chunk_dimensions, compression='gzip', compression_opts=gzip)
				ds_data.attrs['comments'] = comment
			else:
				dataset_id = excGrp[entryname]
				dataset_id[...] = data

			units = ['-' for x in range(XRFmaps_info.n_used_dmaps+XRFmaps_info.n_used_chan)]
			print 'len units  =', len(units)
			units[0:XRFmaps_info.n_used_dmaps] = XRFmaps_info.dmaps_units[:]
			units[XRFmaps_info.n_used_dmaps:XRFmaps_info.n_used_dmaps + XRFmaps_info.n_used_chan] = XRFmaps_info.chan_units[: 2-drop_vtwo]

			names = ['' for x in range(XRFmaps_info.n_used_dmaps + XRFmaps_info.n_used_chan)]
			names[0:XRFmaps_info.n_used_dmaps] = XRFmaps_info.dmaps_names[:]
			names[XRFmaps_info.n_used_dmaps:XRFmaps_info.n_used_dmaps + XRFmaps_info.n_used_chan] = XRFmaps_info.chan_names[:]

			entryname = 'x_axis'
			comment = 'stores the values of the primary fast axis positioner, typically sample x'
			data = XRFmaps_info.x_coord_arr
			if entryname not in excGrp:
				ds_data = excGrp.create_dataset(entryname, data = data)
				ds_data.attrs['comments'] = comment
			else:
				dataset_id = excGrp[entryname]
				dataset_id[...] = data			 
			
			entryname = 'y_axis'
			comment = 'stores the values of the slow axis positioner, typically sample y'
			data = XRFmaps_info.y_coord_arr
			if entryname not in excGrp:
				ds_data = excGrp.create_dataset(entryname, data=data)
				ds_data.attrs['comments'] = comment
			else:
				dataset_id = excGrp[entryname]
				dataset_id[...] = data

			entryname = 'images_names'
			comment = 'names of the xrf and scaler images'
			data = names
			if entryname not in excGrp:
				ds_data = excGrp.create_dataset(entryname, data=data)
				ds_data.attrs['comments'] = comment
			else:
				dataset_id = excGrp[entryname]
				dataset_id[...] = data

			entryname = 'images_units'
			comment = 'units of the xrf and scaler images'
			data = units
			if entryname not in excGrp:
				ds_data = excGrp.create_dataset(entryname, data=data)
				ds_data.attrs['comments'] = comment
			else:
				dataset_id = excGrp[entryname]
				if (len(data),) == dataset_id.shape:
					dataset_id[...] = data		   
				else:
					print 'Error: could not update ', dataset_id.name, ' dataset shapes are different! dataset(', dataset_id.shape, ') : data(', len(data), ')'

			f.close()
			time.sleep(1.0)

		print '---------------------'
		print 'done adding exchange information'
		print '---------------------'
		print ' '

#-----------------------------------------------------------------------------	 
	def read_scan(self, filename):

		#filename= 'D:/mirna/Phyton/Diamond/src/testMapspy/img.dat/5730_sample588_1.h5'
		
		scan_data = maps_mda.scan() 
				
		print filename
		f = call_function_with_retry(h5py.File, 5, 0.1, 1.1, (filename, 'r'))
		#f = h5py.File(filename, 'r') 
				
		if 'MAPS' not in f:
			print 'error, hdf5 file does not contain the required MAPS group. I am aborting this action'
			return 

		maps_group_id = f['MAPS']

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'mca_arr')
		if valid_read:
			mca_arr = this_data

		mca_arr = mca_arr.T
		dimensions = mca_arr.shape
		print 'mca_arr dims: ', dimensions
		# if this is a 2D (x, y) scan dimensions should be 3  
		x_pixels = dimensions[0]
		y_pixels = dimensions[1]
		n_used_chan = dimensions[2]

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'x_axis')
		if valid_read:
			x_coord_arr = this_data
			if len(x_coord_arr.shape) == 2:
				scan_data.x_coord_arr = x_coord_arr[:, 0]
			else:
				scan_data.x_coord_arr = np.array(x_coord_arr)

		this_data, valid_read = self.read_hdf5_core(maps_group_id, 'y_axis')
		if valid_read:
			y_coord_arr = this_data
			if len(y_coord_arr.shape) == 2:
				scan_data.y_coord_arr = y_coord_arr[0, :]
			else:
				scan_data.y_coord_arr = np.array(y_coord_arr)	  
		
		f.close()

		scan_data.scan_name = ''
		scan_data.scan_time_stamp = ''
		
		scan_data.y_pixels = y_pixels

		# create mca calib description array
		#scan_data.mca_calib_description_arr = mca_calib_description_arr 
		
		# create mca calibration array
		#scan_data.mca_calib_arr = mca_calib_arr
		
		scan_data.x_pixels = x_pixels

		#detector_arr = fltarr(x_pixels, y_pixels, info.no_detectors)
		#scan_data.detector_arr = detector_arr
		
		#scan_data.detector_description_arr = detector_description_arr

		#mca_arr = fltarr(x_pixels, y_pixels, no_energy_channels, info.no_detectors)
		scan_data.mca_arr = mca_arr
		
		return scan_data
