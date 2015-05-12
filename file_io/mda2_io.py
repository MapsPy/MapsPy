'''
Created on Apr 27, 2015

@author: Arthur Glowacki
         Argonne APS SSG


Copyright (c) 2015, Stefan Vogt, Argonne National Laboratory 
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

#from __future__ import division
from xdrlib import Unpacker
import string

import numpy as np

#Defines for reading in sizes
SIZE_OF_INT = 4 
SIZE_OF_FLOAT = 4
SIZE_OF_DOUBLE = 8

#----------------------------------------------------------------------
def read_mda_str(file_ptr):
	buf = file_ptr.read(SIZE_OF_INT) #read int, length of string
	upkr = Unpacker(buf)
	str_len = upkr.unpack_int()
	if str_len > 0:
		#print 'str_len',str_len
		byte_len = (str_len+SIZE_OF_INT+3)//4*4
		#print 'byte_len',byte_len
		buf = file_ptr.read(byte_len) 
		upkr = Unpacker(buf)
		return upkr.unpack_fstring(str_len+SIZE_OF_INT)
	else:
		return ''

#----------------------------------------------------------------------
class mda_positioner:
	def __init__(self):
		self.number = 0
		self.name = ""
		self.desc = ""
		self.step_mode = ""
		self.unit = ""
		self.readback_name = ""
		self.readback_desc = ""
		self.readback_unit = ""
	def print_info(self):
		print 'Positioner'
		print 'Number:',self.number
		print 'Name:',self.name
		print 'Desc:',self.desc
		print 'Step Mode:',self.step_mode
		print 'Unit:',self.unit
		print 'Readback Name:',self.readback_name
		print 'Readback Desc:',self.readback_desc
		print 'Readback Unit:',self.readback_unit
		print ' '
	def load_header(self, file_ptr, verbose=False):
		if file_ptr == None:
			raise Exception( 'mda_scan Error: Null file pointer')
		#print 'file loc',hex(file_ptr.tell())
		buf = file_ptr.read(SIZE_OF_INT) #read int
		upkr = Unpacker(buf)
		self.number = upkr.unpack_int()
		self.name = read_mda_str(file_ptr)
		self.desc = read_mda_str(file_ptr)
		self.step_mode = read_mda_str(file_ptr)
		self.unit = read_mda_str(file_ptr)
		self.readback_name = read_mda_str(file_ptr)
		self.readback_desc = read_mda_str(file_ptr)
		self.readback_unit = read_mda_str(file_ptr)
		if verbose:
			self.print_info()

#----------------------------------------------------------------------
class mda_detector:
	def __init__(self):
		self.number = 0
		self.name = ""
		self.desc = ""
		self.unit = ""
	def print_info(self):
		print 'Detector'
		print 'Number:',self.number
		print 'Name:',self.name
		print 'Desc:',self.desc
		print 'Unit:',self.unit
		print ' '
	def load_header(self, file_ptr, verbose=False):
		if file_ptr == None:
			raise Exception( 'mda_scan Error: Null file pointer')
		#print 'file loc',hex(file_ptr.tell())
		buf = file_ptr.read(SIZE_OF_INT) #read int
		upkr = Unpacker(buf)
		self.number = upkr.unpack_int()
		self.name = read_mda_str(file_ptr)
		self.desc = read_mda_str(file_ptr)
		self.unit = read_mda_str(file_ptr)
		if verbose:
			self.print_info()

#----------------------------------------------------------------------
class mda_trigger:
	def __init__(self):
		self.number = 0
		self.name = ""
		self.command = 0.0
	def print_info(self):
		print 'Trigger'
		print 'Number:',self.number
		print 'Name:',self.name
		print 'Command:',self.command
		print ' '
	def load_header(self, file_ptr, verbose=False):
		if file_ptr == None:
			raise Exception( 'mda_scan Error: Null file pointer')
		#print 'file loc',hex(file_ptr.tell())
		buf = file_ptr.read(SIZE_OF_INT) #read int
		upkr = Unpacker(buf)
		self.number = upkr.unpack_int()
		self.name = read_mda_str(file_ptr)
		buf = file_ptr.read(SIZE_OF_INT) #read float
		upkr = Unpacker(buf)
		self.command = upkr.unpack_float()
		if verbose:
			self.print_info()

#----------------------------------------------------------------------		
class mda_extra_pv:
	def __init__(self):
		self.name = ""
		self.desc = ""
		self.pv_type = 0

#----------------------------------------------------------------------		
class mda_scan:
	def __init__(self):
		self.rank = 0				   #short
		self.num_requested_points = 0  #long
		self.current_point = 0		   #long last point captured
		self.scan_locs = []			   #pointer to file location of inner scans
		self.name = ""				   #scan name
		self.date_time = ""            #scan timestamp
		self.num_positioners = 0       #int
		self.num_detectors = 0         #int
		self.num_triggers = 0		   #int
		self.positioners = []		   #list of scanPositioner 
		self.detectors = []            #list of scanDetector 
		self.triggers = []             #list of scanTrigger 
		self.inner_scans = []          #list of inner scans
		self.positioner_data_file_loc = 0
		self.positioner_data = None
		self.detector_data_file_loc = 0
		self.detector_data = None
	def print_info(self):
		print ' '
		print 'Scan:',self.name
		print 'Rank:',self.rank
		print 'Date:',self.date_time
		print 'Number Requested Points:',self.num_requested_points
		print 'Current Point:',self.current_point
		print 'Number of Positioners:',self.num_positioners
		print 'Number of Detectors:',self.num_detectors
		print 'Number of Triggers:',self.num_triggers
		print 'Scan Locations:',self.scan_locs
		print ' '
	def _read_positioner_data_(self, file_ptr):
		#read positioner data
		array_size = self.num_positioners * self.num_requested_points
		read_amt = array_size * SIZE_OF_DOUBLE
		buf = file_ptr.read(read_amt)
		upkr = Unpacker(buf)
		return upkr.unpack_farray(array_size, upkr.unpack_double)
	def _read_detector_data_(self, file_ptr):
		#read detector data
		import struct
		array_size = self.num_detectors * self.num_requested_points
		read_amt = array_size * SIZE_OF_FLOAT
		buf = file_ptr.read(read_amt)
		arr = struct.unpack('>'+str(array_size)+'f', buf)
		#upkr = Unpacker(buf)
		#return upkr.unpack_farray( array_size, upkr.unpack_float)
		return arr
	def _read_positioner_detector_trigger_data_(self, file_ptr):
		#sanity check, unlikely to have more than 10000 positioners
		if self.num_positioners > 10000:
			raise Exception('Possible error reading positioners, total number of positioners = '+str(self.num_positioners) )
		#read positioners
		for i in range(self.num_positioners):
			positioner = mda_positioner()
			positioner.load_header(file_ptr,verbose)
			self.positioners += [positioner]
		#read detectors 
		for i in range(self.num_detectors):
			detector = mda_detector()
			detector.load_header(file_ptr,verbose)
			self.detectors += [detector]
		#read triggers
		for i in range(self.num_triggers):
			trigger = mda_trigger()
			trigger.load_header(file_ptr,verbose)
			self.triggers += [trigger]
		if load_data:
			#read positioner data
			self.positioner_data = self._read_posotioner_data(file_ptr)
			#read detector data
			self.detector_data = self._read_detector_data(file_ptr)
		else:
			#save file location of data
			self.positioner_data_file_loc = file_ptr.tell()
			self.detector_data_file_loc = file_ptr.tell() + (self.num_positioners * SIZE_OF_DOUBLE)
		for loc in self.scan_locs:
			file_ptr.seek(loc,0)
			#print self.name,' file ',hex(file_ptr.tell())
			inner_scan = mda_scan()
			inner_scan.load_header(file_ptr, False, verbose)
			self.inner_scans += [inner_scan]
	def get_child_num_detectors(self):
		if self.rank > 1:
			return self.inner_scans[0].get_child_num_detectors()
		return self.num_detectors
	def scan_header(self, file_ptr, verbose=False):
		if file_ptr == None:
			raise Exception( 'mda_scan Error: Null file pointer')
		#read first 3 bytes in header
		#print 'f',hex(file_ptr.tell())
		buf = file_ptr.read(SIZE_OF_INT * 3)
		upkr = Unpacker(buf)
		self.rank = upkr.unpack_int()
		self.num_requested_points = upkr.unpack_int()
		self.current_point = upkr.unpack_int()
		if self.rank > 1:
			#read in scan file positions and 3 int's after it
			read_amt = (self.num_requested_points) * SIZE_OF_INT  # +1 to get the length of the string after the scans
			buf = file_ptr.read(read_amt)
			upkr = Unpacker(buf)
			self.scan_locs = upkr.unpack_farray(self.num_requested_points, upkr.unpack_int)
		#read a string
		self.name = read_mda_str(file_ptr)
		self.date_time = read_mda_str(file_ptr)
		#read 3 int's
		read_amt = 3 * SIZE_OF_INT
		buf = file_ptr.read(read_amt)
		upkr = Unpacker(buf)
		self.num_positioners = upkr.unpack_int()
		self.num_detectors = upkr.unpack_int()
		self.num_triggers = upkr.unpack_int()
		if self.rank > 1:
			self.positioner_data_file_loc = self.scan_locs[0] - ( (self.num_positioners * SIZE_OF_DOUBLE) + (self.num_detectors * SIZE_OF_FLOAT) )
			self.detector_data_file_loc = self.scan_locs[0] - (self.num_detectors * SIZE_OF_FLOAT)
		#search through inner scans
		num_inner_scans = len(self.scan_locs)
		for i in range(num_inner_scans):
			loc = self.scan_locs[i]
			file_ptr.seek(loc,0)
			#print self.name,' file ',hex(file_ptr.tell())
			inner_scan = mda_scan()
			inner_scan.load_header(file_ptr, False, verbose)
			if inner_scan.rank == 1 and i < (num_inner_scans-1):
				next_loc = self.scan_loc[i+1]
				inner_scan.positioner_data_file_loc = next_loc - ( (inner_scan.num_positioners * SIZE_OF_DOUBLE) + (inner_scan.num_detectors * SIZE_OF_FLOAT) )
				inner_scan.detector_data_file_loc = next_loc - (inner_scan.num_detectors * SIZE_OF_FLOAT)
			else:
				#last one has diff logic
				inner_scan._read_positioner_detector_trigger_data_(file_ptr)
			self.inner_scans += [inner_scan]
 
		if verbose:
			self.print_info()
	def load_header(self, file_ptr, load_data=False, verbose=False):
		if file_ptr == None:
			raise Exception( 'mda_scan Error: Null file pointer')
		#read first 3 bytes in header
		#print 'f',hex(file_ptr.tell())
		buf = file_ptr.read(SIZE_OF_INT * 3)
		upkr = Unpacker(buf)
		self.rank = upkr.unpack_int()
		self.num_requested_points = upkr.unpack_int()
		self.current_point = upkr.unpack_int()
		if self.rank > 1:
			#read in scan file positions and 3 int's after it
			read_amt = (self.num_requested_points) * SIZE_OF_INT  # +1 to get the length of the string after the scans
			buf = file_ptr.read(read_amt)
			upkr = Unpacker(buf)
			self.scan_locs = upkr.unpack_farray(self.num_requested_points, upkr.unpack_int)
		#read a string
		self.name = read_mda_str(file_ptr)
		self.date_time = read_mda_str(file_ptr)
		#read 3 int's
		read_amt = 3 * SIZE_OF_INT
		buf = file_ptr.read(read_amt)
		upkr = Unpacker(buf)
		self.num_positioners = upkr.unpack_int()
		self.num_detectors = upkr.unpack_int()
		self.num_triggers = upkr.unpack_int()
		if verbose:
			self.print_info()
		#sanity check, unlikely to have more than 10000 positioners
		if self.num_positioners > 10000: 
			raise Exception('Possible error reading positioners, total number of positioners = '+str(self.num_positioners) )
		#read positioners
		for i in range(self.num_positioners):
			positioner = mda_positioner()
			positioner.load_header(file_ptr,verbose)
			self.positioners += [positioner]
		#read detectors 
		for i in range(self.num_detectors):
			detector = mda_detector()
			detector.load_header(file_ptr,verbose)
			self.detectors += [detector]
		#read triggers
		for i in range(self.num_triggers):
			trigger = mda_trigger()
			trigger.load_header(file_ptr,verbose)
			self.triggers += [trigger]
		if load_data:
			#read positioner data
			self.positioner_data = self._read_posotioner_data(file_ptr)
			#read detector data
			self.detector_data = self._read_detector_data(file_ptr)
		else:
			#save file location of data
			self.positioner_data_file_loc = file_ptr.tell()
			self.detector_data_file_loc = file_ptr.tell() + (self.num_positioners * SIZE_OF_DOUBLE)
		for loc in self.scan_locs:
			file_ptr.seek(loc,0)
			#print self.name,' file ',hex(file_ptr.tell())
			inner_scan = mda_scan()
			inner_scan.load_header(file_ptr, False, verbose)
			self.inner_scans += [inner_scan]
	def _get_scan_(self, file_ptr, inner_scan):
		#if the dataset is not loaded, read it from the file
		if inner_scan.detector_data == None:
			file_ptr.seek(inner_scan.detector_data_file_loc, 0)
			return inner_scan._read_detector_data_(file_ptr)
		else:
			return inner_scan.detector_data
		
	def get_scan(self, file_ptr, scan_loc, verbose=False):
		#print self.name,':',scan_loc,';',len(scan_loc)
		scan_loc_len = len(scan_loc)
		if scan_loc_len > 1:
			scan_idx = scan_loc[0]
			scan_loc = scan_loc[1:]  #pop off the first value off the tuple
			return self.inner_scans[scan_idx].get_scan(file_ptr, scan_loc, verbose)
		else:
			dataset = []
			if self.rank == 3:
				#get array of data
				if scan_loc_len > 0:
					if scan_loc[0] >= len(self.inner_scans):
						raise Exception('mda_scan Error: scan index:'+str(scan_loc[0])+' is out of range : '+str(self.rank))
					scan_list = self.inner_scans[scan_loc[0]].inner_scans
				else:
					scan_list = self.inner_scans
				for t_scan in scan_list:
					dataset += [ t_scan.get_scan(file_ptr, tuple()) ]
			elif self.rank == 2:
				#get data from the inner scan
				if scan_loc_len > 0:
					if scan_loc[0] >= len(self.inner_scans):
						raise Exception('mda_scan Error: scan index:'+str(scan_loc[0])+' is out of range : '+str(self.rank))
					dataset = self._get_scan_(file_ptr, self.inner_scans[scan_loc[0]])
				else:
					#dataset = self._get_scan_(file_ptr, self)
					for t_scan in self.inner_scans:
						dataset += [ t_scan.get_scan(file_ptr, tuple()) ]
			elif self.rank == 1:
				#get data from self
				dataset = self._get_scan_(file_ptr, self)
			return dataset
#----------------------------------------------------------------------

class mda_header:
	def __init__(self):
		self.version = 0.0     #float
		self.scan_number = 0   #long
		self.data_rank = 0     #short
		self.dims = (0,0)      #vector (rank, int)
		self.is_regular = 0    #regular = 1, not = 0
		self.extra_pv_loc = 0  #long pointer to extra pv's
		
		self.file_size = 0     #in bytes
	def print_info(self):
		print 'Version:',self.version
		print 'Scan Number:',self.scan_number
		print 'Data Rank',self.data_rank
		print 'Dims',self.dims
		print 'Is Regular',self.is_regular
		print 'Extra Pvs Ptr',hex(self.extra_pv_loc)
	def load(self, file_ptr, verbose=False):
		if file_ptr == None:
			raise Exception('mda_header Error: Null file pointer')
		file_ptr.seek(0,2)
		self.file_size = file_ptr.tell()
		file_ptr.seek(0,0)
		print 'file size = ',self.file_size
		if self.file_size < (SIZE_OF_INT * 3):
			raise Exception('mda_header Error: file size ('+str(self.file_size)+') smaller than header read size('+str(SIZE_OF_INT*3)+')')
		#read first 3 bytes in header
		buf = file_ptr.read(SIZE_OF_INT * 3)
		upkr = Unpacker(buf)
		self.version = upkr.unpack_float()
		self.scan_number = upkr.unpack_int()
		self.data_rank = upkr.unpack_int()
		#read in rank + 2 other ints
		read_amt = (self.data_rank + 2) * SIZE_OF_INT# read in rank , + 2 other int
		buf = file_ptr.read(read_amt)
		upkr = Unpacker(buf)
		self.dims = upkr.unpack_farray(self.data_rank, upkr.unpack_int)
		self.is_regular = upkr.unpack_int()
		self.extra_pv_loc = upkr.unpack_int()
		if verbose:
			self.print_info()

#----------------------------------------------------------------------

class mda_file:
	def __init__(self, filename):
		self.header = mda_header()
		self.scan = mda_scan()
		self.extra_pv = mda_extra_pv()
		
		self.filename = filename
		self.file_ptr = None
		try:
			self.file_ptr = open(self.filename, 'rb')
		except Exception, e:
			print 'Exception:', Exception
			print 'Error:', str(e)
			print 'mda_file Error: Failed to open file:',filename
	def load_header(self, load_data=False, verbose=False):
		if self.file_ptr == None:
			return False
		self.header.load(self.file_ptr, verbose)
		self.scan.load_header(self.file_ptr, load_data, verbose)
	def get_scan(self, scan_loc=tuple(), file_ptr=None, verbose=False):
		#passing an empty tuple or nothing into get_scan returns the whole scan
		header_len = len(self.header.dims)-1
		if len(scan_loc) > header_len:
			str_err = 'mda_file Error: scan location size '+str(len(scan_loc))+' is greater than scan dims '+str(len(self.header.dims))
			print str_err
			raise Exception(str_err)
			return []
		if file_ptr == None:
			if self.file_ptr == None:
				raise Exception('mda_file Error: File was never loaded!')
			else:
				return self.scan.get_scan(self.file_ptr, scan_loc, verbose)
		else:
			return self.scan.get_scan(file_ptr, scan_loc, verbose)
	def close_file(self):
		if self.file_ptr != None:
			self.file_ptr.close()
			self.file_ptr = None

#----------------------------------------------------------------------

if __name__ == '__main__':
	#test loading mda file
	print 'test opening test.mda'
	mda_f = mda_file('test4.mda')
	mda_f.load_header()
	print 'Dataset dims:',mda_f.header.dims
	
	dataset =  mda_f.get_scan((0,)  )
	print 'len',len(dataset)
	print 'len',len(dataset[0])
	num_detectors = mda_f.scan.get_child_num_detectors()
	print 'num detectors', num_detectors
	#print 'len',len(dataset[0][0])
	#print dataset
	mda_f.close_file()
	
	#for i in range(mda_f.header.dims[1]):
	#	print len(mda_f.get_scan( (0,i) ) )

