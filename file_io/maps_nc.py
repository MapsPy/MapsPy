'''
Created on Dec 27, 2011

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
import time
import glob

from file_util import call_function_with_retry

try:
	import scipy.io.netcdf
	netcdf_open = scipy.io.netcdf.netcdf_file
# except ImportError:
#	  import netCDF4
#	  netcdf_open = netCDF4.Dataset
except ImportError:
	print 'cannot find a netcdf module'
	raise ImportError('cannot find a netcdf module')

import maps_mda

# ----------------------------------------------------------------------


class nc:
	def __init__(self, logger):
		self.logger = logger
	
	# ----------------------------------------------------------------------
	def read_combined_nc_scans(self, filename, path, header, this_detector, extra_pvs = True):

		#the following variables are created or read with this routine:
		scan_name =  ' ' 
		scan_time_stamp = ' '
		mca_calib_arr = 0.					#  array 
		mca_calib_description_arr = ' '		# array

		detector_arr = 0.					# nxmxo array  ( o detectors)
		detector_description_arr = ' '		#  ox1 array
		invalid_file = [0]

		threeD_only = 2		# read in only files with x and y dimensions
		show_extra_pvs = 1
		extra_pv = 0
		binning = 0


		mda = maps_mda.mda(self.logger)
		scan = mda.read_scan(filename, threeD_only=threeD_only, invalid_file=invalid_file, extra_pvs=True)
		if scan == None:
			return None

		invalid_file = invalid_file[0]
		try:
			maps_mda.convert_pv_names(filename, scan)
		except:
			self.logger.error('Cannot convert_pv_names: %s filename: %s', str(invalid_file), filename)

		if invalid_file > 0:		
			self.logger.error('not a valid mda flyscan file, error number: %s filename: %s', str(invalid_file), filename)
			#  may be live scan
			#return None


		n_ev = 0
		n_rows = scan.y_pixels
		n_cols = scan.x_pixels
		scan.mca_arr = np.zeros((scan.x_pixels, scan.y_pixels, 2000)) # nxmx2000 array	( 2000 energies) 

		# now fill in XRF information	
		
		old_det_len = len(scan.detector_description_arr)
				
		scan.detector_description_arr.append('dxpXMAP2xfm3:mca1.ELTM')
		scan.detector_description_arr.append('dxpXMAP2xfm3:mca2.ELTM')
		scan.detector_description_arr.append('dxpXMAP2xfm3:mca3.ELTM')
		scan.detector_description_arr.append('dxpXMAP2xfm3:mca4.ELTM')

		scan.detector_description_arr.append('dxpXMAP2xfm3:mca1.ERTM')
		scan.detector_description_arr.append('dxpXMAP2xfm3:mca2.ERTM')
		scan.detector_description_arr.append('dxpXMAP2xfm3:mca3.ERTM')
		scan.detector_description_arr.append('dxpXMAP2xfm3:mca4.ERTM')

		scan.detector_description_arr.append('dxpXMAP2xfm3:dxp1:InputCountRate')
		scan.detector_description_arr.append('dxpXMAP2xfm3:dxp2:InputCountRate')
		scan.detector_description_arr.append('dxpXMAP2xfm3:dxp3:InputCountRate')
		scan.detector_description_arr.append('dxpXMAP2xfm3:dxp4:InputCountRate')

		scan.detector_description_arr.append('dxpXMAP2xfm3:dxp1:OutputCountRate')
		scan.detector_description_arr.append('dxpXMAP2xfm3:dxp2:OutputCountRate')
		scan.detector_description_arr.append('dxpXMAP2xfm3:dxp3:OutputCountRate')
		scan.detector_description_arr.append('dxpXMAP2xfm3:dxp4:OutputCountRate')

		new_det_len = len(scan.detector_description_arr)

		new_detector_arr=np.ones((scan.x_pixels, scan.y_pixels, new_det_len))
		if old_det_len > 0:
			new_detector_arr[:, :, 0:old_det_len] = scan.detector_arr[:, :, 0:old_det_len]
		scan.detector_arr = new_detector_arr

		file_path = os.path.join(path, os.path.join('flyXRF', header))

		last_line = 0
		for i_lines in range(n_rows):
			last_line = i_lines
			try:
				nc_files = glob.glob(file_path + '*_' + str(i_lines) + '.nc')
				#ncfile = os.path.join(path, 'flyXRF', header + '_2xfm3__' + str(i_lines) + '.nc' )
				if len(nc_files) > 0:
					ncfile = nc_files[0]
				else:
					continue

				#xmapdat = read_xmap_netcdf(ncfile, self.logger, True)
				xmapdat = read_xmap_netcdf2(ncfile, self.logger, True)

				if xmapdat == None:
					return None
				for ix in range(n_cols):
					if ix < len(xmapdat.liveTime[:, 0]):
						scan.mca_arr[ix, i_lines, 0:2000] = xmapdat.data[ix, this_detector, 0:2000]

						scan.detector_arr[ix, i_lines, new_det_len-16] = xmapdat.liveTime[ix, 0]
						scan.detector_arr[ix, i_lines, new_det_len-15] = xmapdat.liveTime[ix, 1]
						scan.detector_arr[ix, i_lines, new_det_len-14] = xmapdat.liveTime[ix, 2]
						scan.detector_arr[ix, i_lines, new_det_len-13] = xmapdat.liveTime[ix, 3]

						scan.detector_arr[ix, i_lines, new_det_len-12] = xmapdat.realTime[ix, 0]
						scan.detector_arr[ix, i_lines, new_det_len-11] = xmapdat.realTime[ix, 1]
						scan.detector_arr[ix, i_lines, new_det_len-10] = xmapdat.realTime[ix, 2]
						scan.detector_arr[ix, i_lines, new_det_len-9] = xmapdat.realTime[ix, 3]

						scan.detector_arr[ix, i_lines, new_det_len-8] = xmapdat.inputCounts[ix, 0] / xmapdat.liveTime[ix, 0]
						scan.detector_arr[ix, i_lines, new_det_len-7] = xmapdat.inputCounts[ix, 1] / xmapdat.liveTime[ix, 1]
						scan.detector_arr[ix, i_lines, new_det_len-6] = xmapdat.inputCounts[ix, 2] / xmapdat.liveTime[ix, 2]
						scan.detector_arr[ix, i_lines, new_det_len-5] = xmapdat.inputCounts[ix, 3] / xmapdat.liveTime[ix, 3]

						scan.detector_arr[ix, i_lines, new_det_len-4] = xmapdat.outputCounts[ix, 0] / xmapdat.realTime[ix, 0]
						scan.detector_arr[ix, i_lines, new_det_len-3] = xmapdat.outputCounts[ix, 1] / xmapdat.realTime[ix, 1]
						scan.detector_arr[ix, i_lines, new_det_len-2] = xmapdat.outputCounts[ix, 2] / xmapdat.realTime[ix, 2]
						scan.detector_arr[ix, i_lines, new_det_len-1] = xmapdat.outputCounts[ix, 3] / xmapdat.realTime[ix, 3]
			except:
				self.logger.exception('Skipping netcdf line at %d, filename: %s', last_line, filename)
		return scan

#----------------------------------------------------------------------
# From xmap_nc.py
#----------------------------------------------------------------------
def aslong(d):
	"""unravels and converts array of int16 (int) to int32 (long)"""
	# need to unravel the array!!!
	d = d.astype(np.int16).ravel()
	d.dtype = np.int32
	return d

class xMAPBufferHeader(object):
	def __init__(self,buff):
		self.tag0		   = buff[0]  # Tag word 0
		self.tag1		   = buff[1]  # Tag word 1
		self.headerSize    = buff[2]  #  Buffer header size
		#  Mapping mode (1=Full spectrum, 2=Multiple ROI, 3=List mode)
		self.mappingMode   = buff[3]
		self.runNumber	   = buff[4]  # Run number
		# Sequential buffer number, low word first
		self.bufferNumber  = aslong(buff[5:7])[0]
		self.bufferID	   = buff[7]  # 0=A, 1=B
		self.numPixels	   = buff[8]  # Number of pixels in buffer
		# Starting pixel number, low word first
		self.startingPixel = aslong(buff[9:11])[0]
		self.moduleNumber  = buff[11]
		self.channelID	   = np.array(buff[12:20]).reshape((4,2))
		self.channelSize   = buff[20:24]
		self.bufferErrors  = buff[24]
		self.userDefined   = buff[32:64]

class xMAPMCAPixelHeader(object):
	def __init__(self,buff):
		self.tag0		 = buff[0]
		self.tag1		 = buff[1]
		self.headerSize  = buff[2]
		# Mapping mode (1=Full spectrum, 2=Multiple ROI, 3=List mode)
		self.mappingMode = buff[3]
		self.pixelNumber = aslong(buff[4:6])[0]
		self.blockSize	 = aslong(buff[6:8])[0]

class xMAPData(object):
	def __init__(self,npix,nmod,nchan):
		ndet = 4 * nmod
		self.firstPixel   = 0
		self.numPixels	  = 0
		self.data		  = np.zeros((npix, ndet, nchan), dtype='i2')
		self.realTime	  = np.zeros((npix, ndet), dtype='i8')
		self.liveTime	  = np.zeros((npix, ndet), dtype='i8')
		self.inputCounts  = np.zeros((npix, ndet), dtype='i4')
		self.outputCounts = np.zeros((npix, ndet), dtype='i4')

def read_xmap_netcdf(fname, logger, verbose=False):
	# Reads a netCDF file created with the DXP xMAP driver
	# with the netCDF plugin buffers

	if verbose:
		logger.debug('reading %s', fname)

	t0 = time.time()
	# read data from array_data variable of netcdf file
	fh = call_function_with_retry(netcdf_open, 5, 0.1, 1.1, (fname, 'r'))
	#fh = netcdf_open(fname,'r')
	if fh == None:
		return None
	data_var = fh.variables['array_data']
	array_data = data_var.data
	if len(array_data) == 0:
		array_data = np.zeros((5,1,1047808), dtype='f')
	t1 = time.time()
	# array_data will normally be 3d:
	#  shape = (narrays, nmodules, buffersize)
	# but nmodules and narrays could be 1, so that
	# array_data could be 1d or 2d.
	#
	# here we force the data to be 3d
	shape = array_data.shape 
	if len(shape) == 1:
		array_data.shape = (1,1,shape[0])
	elif len(shape) == 2:
		array_data.shape = (1,shape[0],shape[1])

	narrays,nmodules,buffersize = array_data.shape
	modpixs = array_data[0,0,8]
	if modpixs < 124: modpixs = 124
	npix_total = 0
	clocktick  = 320.e-9
	for array in range(narrays):
		for module in range(nmodules):
			d	= array_data[array,module,:]
			bh	= xMAPBufferHeader(d)
			#if verbose and array==0:
			#print	' nc data shape: ', d.shape, d.size
			# logger.debug(modpixs, (d.size-256), (d.size-256)/modpixs
			# logger.debug(modpixs*(d.size-256)/(1.0*modpixs)
			dat = d[256:].reshape(modpixs, (d.size-256)/modpixs )

			npix = bh.numPixels
			if module == 0:
				npix_total += npix
				if array == 0:
					# first time through, (array,module)=(0,0) we
					# read mapping mode, set up how to slice the
					# data, and build data arrays in xmapdat
					mapmode = dat[0, 3]
					if mapmode == 1:  # mapping, full spectra
						nchans = d[20]
						data_slice = slice(256, 8448)
					elif mapmode == 2:  # ROI mode
						# Note:  nchans = number of ROIS !!
						nchans = max(d[264:268])
						data_slice = slice(64, 64 + 8 * nchans)
					else:
						nchans = 0
						data_slice = 0
					xmapdat = xMAPData(narrays*modpixs, nmodules, nchans)
					xmapdat.firstPixel = bh.startingPixel

			# acquistion times and i/o counts data are stored
			# as longs in locations 32:64
			t_times = aslong(dat[:npix, 32:64]).reshape(npix, 4, 4)
			p1 = npix_total - npix
			p2 = npix_total
			xmapdat.realTime[p1:p2, :] = t_times[:, :, 0]
			xmapdat.liveTime[p1:p2, :] = t_times[:, :, 1]
			xmapdat.inputCounts[p1:p2, :] = t_times[:, :, 2]
			xmapdat.outputCounts[p1:p2, :] = t_times[:, :, 3]

			# the data, extracted as per data_slice and mapmode
			t_data = dat[:npix, data_slice]
			if mapmode == 2:
				t_data = aslong(t_data)
			xmapdat.data[p1:p2, :, :] = t_data.reshape(npix, 4, nchans)

	t2 = time.time()
	xmapdat.numPixels = npix_total
	xmapdat.data = xmapdat.data[:npix_total]
	xmapdat.realTime = clocktick * xmapdat.realTime[:npix_total]
	xmapdat.liveTime = clocktick * xmapdat.liveTime[:npix_total]
	xmapdat.inputCounts = xmapdat.inputCounts[:npix_total]
	xmapdat.outputCounts = xmapdat.outputCounts[:npix_total]
	if verbose:
		logger.debug('   time to read file    = %5.1f ms', ((t1 - t0) * 1000))
		logger.debug('   time to extract data = %5.1f ms', ((t2 - t1) * 1000))
		logger.debug('   read %i pixels ', npix_total)
		logger.debug('   data shape:	%s', xmapdat.data.shape)
	fh.close()
	return xmapdat

def read_xmap_netcdf2(fname, logger, verbose=False):
	# Reads a netCDF file created with the DXP xMAP driver
	# with the netCDF plugin buffers

	if verbose:
		logger.debug('reading %s', fname)

	t0 = time.time()
	# read data from array_data variable of netcdf file
	fh = call_function_with_retry(netcdf_open, 5, 0.1, 1.1, (fname, 'r'))
	#fh = netcdf_open(fname,'r')
	if fh == None:
		return None
	data_var = fh.variables['array_data']
	array_data = data_var.data
	if len(array_data) == 0:
		array_data = np.zeros((5,1,1047808), dtype='f')
	t1 = time.time()
	# array_data will normally be 3d:
	#  shape = (narrays, nmodules, buffersize)
	# but nmodules and narrays could be 1, so that
	# array_data could be 1d or 2d.
	#
	# here we force the data to be 3d
	shape = array_data.shape
	if len(shape) == 1:
		array_data.shape = (1,1,shape[0])
	elif len(shape) == 2:
		array_data.shape = (1,shape[0],shape[1])

	narrays,nmodules,buffersize = array_data.shape
	modpixs = array_data[0,0,8]
	ndetectors = 4
	#if modpixs < 124: modpixs = 124
	npix_total = 0
	clocktick  = 320.e-9
	offset = 256
	for array in range(narrays):
		for module in range(nmodules):
			d	= array_data[array,module,:]
			bh	= xMAPBufferHeader(d)
			#if verbose and array==0:
			#print	' nc data shape: ', d.shape, d.size
			# logger.debug(modpixs, (d.size-256), (d.size-256)/modpixs
			# logger.debug(modpixs*(d.size-256)/(1.0*modpixs)
			#dat = d[256:].reshape(modpixs, (d.size-256)/modpixs )

			npix = bh.numPixels
			if module == 0:
				npix_total += npix
				if array == 0:
					# first time through, (array,module)=(0,0) we
					# read mapping mode, set up how to slice the
					# data, and build data arrays in xmapdat
					#mapmode = dat[0, 3]
					if bh.mappingMode == 1:  # mapping, full spectra
						nchans = d[20]
						data_slice = slice(256, 8448)
					elif bh.mappingMode == 2:  # ROI mode
						# Note:  nchans = number of ROIS !!
						nchans = max(d[264:268])
						data_slice = slice(64, 64 + 8 * nchans)
					else:
						nchans = 0
						data_slice = 0
					xmapdat = xMAPData(narrays*modpixs, nmodules, nchans)
					xmapdat.firstPixel = bh.startingPixel

			# acquistion times and i/o counts data are stored
			# as longs in locations 32:64
			#t_times = aslong(dat[:npix, 32:64]).reshape(npix, 4, 4)
			#p1 = npix_total - npix
			#p2 = npix_total
			for pix in range(npix_total):
				if (offset+64) < d.size:
					t_times = aslong(d[offset + 32: offset + 64]).reshape(4, 4)
					for j in range(ndetectors):
						xmapdat.realTime[pix, j] = t_times[j, 0]
						xmapdat.liveTime[pix, j] = t_times[j, 1]
						xmapdat.inputCounts[pix, j] = t_times[j, 2]
						xmapdat.outputCounts[pix, j] = t_times[j, 3]
					# the data, extracted as per data_slice and mapmode
					#t_data = dat[:npix, data_slice]
					spec_size = d[offset + 6]
					spec_offset = offset + spec_size
					t_data = d[offset + 256: spec_offset]
					if t_data.size == 4*nchans:
						if bh.mappingMode == 2:
							t_data = aslong(t_data)
						xmapdat.data[pix, :, :] = t_data.reshape(4, nchans)
						offset += spec_size

	t2 = time.time()
	xmapdat.numPixels = npix_total
	xmapdat.data = xmapdat.data[:npix_total]
	xmapdat.realTime = clocktick * xmapdat.realTime[:npix_total]
	xmapdat.liveTime = clocktick * xmapdat.liveTime[:npix_total]
	xmapdat.inputCounts = xmapdat.inputCounts[:npix_total]
	xmapdat.outputCounts = xmapdat.outputCounts[:npix_total]
	if verbose:
		logger.debug('   time to read file    = %5.1f ms', ((t1 - t0) * 1000))
		logger.debug('   time to extract data = %5.1f ms', ((t2 - t1) * 1000))
		logger.debug('   read %i pixels ', npix_total)
		logger.debug('   data shape:	%s', xmapdat.data.shape)
	fh.close()
	return xmapdat
