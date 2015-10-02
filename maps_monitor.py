'''
Created on Jun 6, 2013

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

import os
import sys
import shutil
from time import gmtime, strftime
import time
import platform

import Settings
import maps_batch
import traceback

settings_filename = 'settings.ini'

#-----------------------------------------------------------------------------------------------------
def check_for_alias(directory_str, alias_dict):
	ret_str = directory_str
	for key in alias_dict.iterkeys():
		if directory_str.startswith(key):
			ret_str = directory_str.replace(key, alias_dict[key])
			break
	return ret_str

#-----------------------------------------------------------------------------------------------------
def parse_aliases(alias_str):
	all_aliases = alias_str.split(';')
	alias_dict = dict()
	for single_set in all_aliases:
		split_single = single_set.split(',')
		if len(split_single) > 1:
			alias_dict[split_single[0]] = split_single[1]
	return alias_dict

#-----------------------------------------------------------------------------------------------------
def main(mySettings):

	jobs_path = mySettings[Settings.MONITOR_JOBS_PATH]
	processing_path = mySettings[Settings.MONITOR_PROCESSING_PATH]
	info_path = mySettings[Settings.MONITOR_FINISHED_INFO_PATH]
	done_path = mySettings[Settings.MONITOR_DONE_PATH]
	computer = mySettings[Settings.MONITOR_COMPUTER_NAME]
	check_interval = int(mySettings[Settings.MONITOR_CHECK_INTERVAL])

	alias_dict = parse_aliases(mySettings[Settings.MONITOR_DIR_ALIAS])

	working_dir = os.getcwd()	
	#todo: create folders if they don't exist
	#os.chdir(jobs_path)
	print 'Starting maps_monitor with'
	print 'jobs_path = ',jobs_path
	print 'processing_path = ',processing_path
	print 'finished_info_path = ',info_path
	print 'done_path = ',done_path
	print 'computer name = ',computer
	print 'directory aliases = ',alias_dict
	print 'checking every ',check_interval,'seconds'


	#print 'changed into ', jobs_path
  
	#make sure the following are defined:
	keyword_a = 0
	keyword_b = 0
	keyword_c = 0
	keyword_d = 0
	keyword_e = 0
	keyword_f = 0

	statusfile = 'status_'+computer
	print 'changed into ', jobs_path

	print strftime("%Y-%m-%d %H:%M:%S", gmtime())
	true = 1
	  
	while true:
		filenames = []
		try:
			os.chdir(jobs_path)
			dirList=os.listdir(jobs_path)
			for fname in dirList:
				if (fname[0:4] == 'job_') and (fname[-4:] == '.txt') : 
					filenames.append(fname)
		except:
			print 'error changing dir'
			time.sleep(5)
		no_files =len(filenames)
		if no_files == 0 :
			#time.sleep(300.0)
			time.sleep(check_interval)
			print 'no jobs found, waiting ...'
			print strftime("%Y-%m-%d %H:%M:%S", gmtime())
			f = open(statusfile+'_idle.txt', 'w')
			f.write(strftime("%Y-%m-%d %H:%M:%S", gmtime())+'\n')
			f.close()		
			continue
		
		if no_files > 0 :
			try:
				os.remove(statusfile+'_idle.txt')
			except:
				pass

			time_started = strftime("%Y-%m-%d %H:%M:%S", gmtime())
			version = 0
			total_number_detectors = 1
			max_no_processors_files = 1
			max_no_processors_lines = 1
			write_hdf = 0
			quick_dirty = 0
			xrf_bin = 0
			nnls = 0
			xanes_scan = 0
			detector_to_start_with = 0
			#default beamline to use for now is 2-id-e , we will change this in the future
			beamline = '2-ID-E'
	 
			print 'found a job waiting, in file: ', filenames[0]
			print 'read data file'

			f = open(statusfile+'_working.txt', 'w')
			f.write(strftime("%Y-%m-%d %H:%M:%S", gmtime())+'\n')
			f.write('found a job waiting, in file: '+ filenames[0]+'\n')
			f.close()	
						
			time.sleep(5)

			standard_filenames = []
			try:
				f = open(filenames[0], 'rt')
				for line in f:
					if ':' in line : 
						slist = line.split(':')
						tag = slist[0]
						value = ':'.join(slist[1:])
						
						if tag == 'DIRECTORY': directory = value.strip()
						elif tag == 'A'	:  keyword_a = int(value)
						elif tag == 'B'	:  keyword_b = int(value)
						elif tag == 'C'	:  keyword_c = int(value)
						elif tag == 'D'	:  keyword_d = int(value)
						elif tag == 'E'	:  keyword_e = int(value)
						elif tag == 'F'	:  keyword_f = int(value)
						elif tag == 'DETECTOR_ELEMENTS' : total_number_detectors  =  int(value)
						elif tag == 'MAX_NUMBER_OF_FILES_TO_PROCESS' : max_no_processors_files = int(value)
						elif tag == 'MAX_NUMBER_OF_LINES_TO_PROCESS' : max_no_processors_lines = int(value)
						elif tag == 'QUICK_DIRTY'  :  quick_dirty  = int(value)
						elif tag == 'XRF_BIN'  :  xrf_bin  = int(value)
						elif tag == 'NNLS'	:  nnls  = int(value)
						elif tag == 'XANES_SCAN'  :  xanes_scan  = int(value)
						elif tag == 'DETECTOR_TO_START_WITH'  :  detector_to_start_with  = int(value)
						elif tag == 'BEAMLINE'	:  beamline  = str(value).strip()
						elif tag == 'STANDARD'	:  standard_filenames.append(str(value).strip())
								
				f.close()

			except: print 'Could not read file: ', filenames[0]

			directory = check_for_alias(directory, alias_dict)
			print 'move job into processing directory'
			shutil.copy(filenames[0], os.path.join(processing_path, filenames[0]))
			os.remove(filenames[0])   

			if keyword_f == 1:
				keyword_a = 1
				keyword_b = 1
				keyword_c = 1
				keyword_d = 1
				keyword_e = 1
						 
			print 'now moving into directory to analyse ', directory
			os.chdir(directory)
			f = open('maps_settings.txt', 'w')						   
			f.write('	   This file will set some MAPS settings mostly to do with fitting'+'\n')
			f.write('VERSION:' + str(version).strip()+'\n')
			f.write('DETECTOR_ELEMENTS:' + str(total_number_detectors).strip()+'\n')
			f.write('MAX_NUMBER_OF_FILES_TO_PROCESS:' + str(max_no_processors_files).strip()+'\n')
			f.write('MAX_NUMBER_OF_LINES_TO_PROCESS:' + str(max_no_processors_lines).strip()+'\n')
			f.write('QUICK_DIRTY:' + str(quick_dirty).strip()+'\n')
			f.write('XRF_BIN:' + str(xrf_bin).strip()+'\n')
			f.write('NNLS:' + str(nnls).strip()+'\n')
			f.write('XANES_SCAN:' + str(xanes_scan).strip()+'\n')
			f.write('DETECTOR_TO_START_WITH:' + str(detector_to_start_with).strip()+'\n')	
			f.write('BEAMLINE:' + beamline.strip()+'\n')   
			for item in standard_filenames:
				f.write('STANDARD:' + item.strip()+'\n')   
			f.close() 
			 
			os.chdir(working_dir)
			try:
				maps_batch.main(wdir=directory, a=keyword_a, b=keyword_b, c=keyword_c, d=keyword_d, e=keyword_e)
			except:
				print 'Error processing',directory
				traceback.print_exc(file=sys.stdout)
			os.chdir(processing_path)
			print 'move job into processing directory'
			shutil.copy(os.path.join(processing_path,filenames[0]), os.path.join(done_path,filenames[0]))	  
			os.remove(filenames[0])   

			os.chdir(info_path)
		
			f = open('finished_'+filenames[0], 'w') 

			f.write( 'time started: ' + time_started+'\n')
			f.write( 'time finished: '+ strftime("%Y-%m-%d %H:%M:%S", gmtime())+'\n')
			f.write( 'computer that did analysis '+ computer+'\n')
			f.write( '--------------------------------------'+'\n')
			f.write( '')
			f.write( '')
			f.write( '')
			f.write( 'used the following settings'+'\n')
			f.write('VERSION:' + str(version).strip()+'\n')
			f.write( 'A:'+ str(keyword_a).strip()+'\n')
			f.write( 'B:'+ str(keyword_b).strip()+'\n')
			f.write( 'C:'+ str(keyword_c).strip()+'\n')
			f.write( 'D:'+ str(keyword_d).strip()+'\n')
			f.write( 'E:'+ str(keyword_e).strip()+'\n')
			f.write( 'F:'+ str(keyword_f).strip()+'\n')
			f.write('DETECTOR_ELEMENTS:' + str(total_number_detectors).strip()+'\n')
			f.write('MAX_NUMBER_OF_FILES_TO_PROCESS:' + str(max_no_processors_files).strip()+'\n')
			f.write('MAX_NUMBER_OF_LINES_TO_PROCESS:' + str(max_no_processors_lines).strip()+'\n')
			f.write('QUICK_DIRTY:' + str(quick_dirty).strip()+'\n')
			f.write('XRF_BIN:' + str(xrf_bin).strip()+'\n')
			f.write('NNLS:' + str(nnls).strip()+'\n')
			f.write('XANES_SCAN:' + str(xanes_scan).strip()+'\n')
			f.write('DETECTOR_TO_START_WITH:' + str(detector_to_start_with).strip()+'\n')		  
			f.close() 
			
			os.chdir(jobs_path)

			os.remove(statusfile+'_working.txt') 
	  
	 
	return


#-----------------------------------------------------------------------------	 
if __name__ == '__main__':
	if len(sys.argv) > 1:
		settings_filename = sys.argv[1]
	settings = Settings.SettingsIO()
	settings.load(settings_filename)
	if settings.checkSectionKeys(Settings.SECTION_MONITOR, Settings.MONITOR_KEYS) == False:
		print 'Error: Could not find all settings in ',settings_filename
		print 'Please add the following keys to',settings_filename,'under the section',Settings.SECTION_MONITOR
		for key in Settings.MONITOR_KEYS:
			print key
		sys.exit(1)
	monitorSettings = settings.getSetting(Settings.SECTION_MONITOR)
	#computer_name =  str(platform.node())
	main(monitorSettings)
		
