'''
Created on May 2015

@author: Arthur Glowacki, Argonne National Laboratory

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

# include parent directory for imports
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import Settings
import requests
import cherrypy
import json
import traceback
import logging
import threading
import signal
import psutil
import multiprocessing
from datetime import datetime
from plugins.DatabasePlugin import DatabasePlugin
from plugins.SQLiteDB import SQLiteDB
from handlers.ProcessNodeHandlers import ProcessNodeHandler, ProcessNodeJobsWebService
from maps_batch import maps_batch
import math


STR_COMPUTER_NAME = 'ComputerName'
STR_NUM_THREADS = 'NumThreads'
STR_HOSTNAME = 'Hostname'
STR_PORT = 'Port'
STR_STATUS = 'Status'
STR_HEARTBEAT = 'Heartbeat'
STR_PROC_CPU_PERC = 'ProcessCpuPercent'
STR_PROC_CPU_PERC_CHILDREN = 'ProcessCpuPercentChildren'
STR_PROC_MEM_PERC_CHILDREN = 'ProcessMemPercentChildren'
STR_PROC_MEM_PERC = 'ProcessMemPercent'
STR_SYS_CPU_PERC = 'SystemCpuPercent'

STR_JOB_LOG_DIR_NAME = 'job_logs'

JOB_PROCESSING_ID = 1
JOB_COMPLETED_ID = 2
JOB_ERROR_ID = 10


class ProcessNode(object):

	def __init__(self, settings):
		self.settings = settings
		serverSettings = settings.getSetting(Settings.SECTION_SERVER)
		pnSettings = settings.getSetting(Settings.SECTION_PROCESS_NODE)
		print serverSettings
		print pnSettings
		self.pn_info = {STR_COMPUTER_NAME: pnSettings[Settings.PROCESS_NODE_NAME],
					STR_NUM_THREADS: pnSettings[Settings.PROCESS_NODE_THREADS],
					STR_HOSTNAME: serverSettings[Settings.SERVER_HOSTNAME],
					STR_PORT: serverSettings[Settings.SERVER_PORT],
					STR_STATUS: 'Bootup',
					STR_HEARTBEAT: str(datetime.now()),
					STR_PROC_CPU_PERC: 0.0,
					STR_PROC_MEM_PERC: 0.0,
					STR_SYS_CPU_PERC: 0.0,
					STR_PROC_CPU_PERC_CHILDREN: [],
					STR_PROC_MEM_PERC_CHILDREN: []
		}
		cherrypy.config.update({
			'server.socket_host': serverSettings[Settings.SERVER_HOSTNAME],
			'server.socket_port': int(serverSettings[Settings.SERVER_PORT]),
			'log.access_file': "logs/" + str(pnSettings[Settings.PROCESS_NODE_NAME]) + "_access.log",
			'log.error_file': "logs/" + str(pnSettings[Settings.PROCESS_NODE_NAME]) + "_error.log"
		})

		self.conf = {
			'/': {
				'tools.sessions.on': True,
				'tools.staticdir.root': os.path.abspath(os.getcwd())
			},
			'/job_queue': {
				'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
				'tools.response_headers.on': True,
				'tools.response_headers.headers': [('Content-Type', 'text/plain')],
			},
			'/static': {
				'tools.staticdir.on': True,
				'tools.staticdir.dir': './public'
			}
		}
		self.new_job_event = threading.Event()
		self.status_event = threading.Event()
		print 'Setup signal handler'
		if os.name == "nt":
			try:
				import win32api
				win32api.SetConsoleCtrlHandler(self.win_handle_sigint, True)
			except ImportError:
				version = ".".join(map(str, sys.version_info[:2]))
				raise Exception("pywin32 not installed for Python " + version)
		else:
			signal.signal(signal.SIGINT, self.unix_handle_sigint)
		self.status_update_interval = 10
		self.scheduler_host = serverSettings[Settings.SERVER_SCHEDULER_HOSTNAME]
		self.scheduler_port = serverSettings[Settings.SERVER_SCHEDULER_PORT]
		self.path_alias_dict = self.parse_aliases(pnSettings[Settings.PROCESS_NODE_PATH_ALIAS])
		print 'alias paths ',self.path_alias_dict
		self.session = requests.Session()
		self.scheduler_pn_url = 'http://' + self.scheduler_host + ':' + self.scheduler_port + '/process_node'
		self.scheduler_job_url = 'http://' + self.scheduler_host + ':' + self.scheduler_port + '/job'
		self.db_name = pnSettings[Settings.PROCESS_NODE_DATABASE_NAME]
		self.db = DatabasePlugin(cherrypy.engine, SQLiteDB, self.db_name)
		cherrypy.engine.subscribe("new_job", self.callback_new_job)
		cherrypy.engine.subscribe("update_id", self.callback_update_id)
		cherrypy.engine.subscribe("send_job_update", self.callback_send_job_update)
		self.create_directories()
		self.running = True
		self.status_thread = None
		self.this_process = psutil.Process(os.getpid())

	def unix_handle_sigint(self, sig, frame):
		print 'unix_handle_sigint', sig, frame
		self.stop()

	def win_handle_sigint(self, sig):
		print 'win_handle_sigint', sig
		self.stop()

	def create_directories(self):
		if not os.path.exists(STR_JOB_LOG_DIR_NAME):
			os.makedirs(STR_JOB_LOG_DIR_NAME)

	def callback_new_job(self, val):
		self.new_job_event.set()

	def callback_send_job_update(self, val):
		self.send_job_update(val)

	def _setup_logging_(self, log, logtype, logname):
		maxBytes = getattr(log, "rot_maxBytes", 20971520) # 20Mb
		backupCount = getattr(log, "rot_backupCount", 10)
		fname = getattr(log, logtype, logname)
		h = logging.handlers.RotatingFileHandler(fname, 'a', maxBytes, backupCount)
		h.setLevel(logging.DEBUG)
		h.setFormatter(cherrypy._cplogging.logfmt)
		log.error_log.addHandler(h)

	def callback_update_id(self, new_id):
		self.pn_info['Id'] = int(new_id)

	def run(self):
		webapp = ProcessNodeHandler()
		self.db.subscribe()
		self.db.create_tables()
		webapp.job_queue = ProcessNodeJobsWebService(self.db)
		app = cherrypy.tree.mount(webapp, '/', self.conf)
		self._setup_logging_(app.log, "rot_error_file", "logs/" + self.pn_info[STR_COMPUTER_NAME] + "_error.log")
		self._setup_logging_(app.log, "rot_access_file", "logs/" + self.pn_info[STR_COMPUTER_NAME] + "_access.log")
		cherrypy.engine.start()
		try:
			print 'posting to scheduler', self.scheduler_pn_url
			self.session.post(self.scheduler_pn_url, data=json.dumps(self.pn_info))
		except:
			print 'Error sending post'
		self.pn_info[STR_STATUS] = 'Idle'
		# start status thread
		if self.status_thread is None:
			self.status_thread = threading.Thread(target=self.status_thread_func)
			self.status_thread.start()
		self.new_job_event.set() # set it at start to check for unfinished jobs
		try:
			while self.running:
				self.new_job_event.wait(self.status_update_interval)
				if self.new_job_event.is_set():
					self.new_job_event.clear()
					self.process_next_job()
				#else:
				#	self.send_status_update()
				#	#self.process_next_job()
				if cherrypy.engine.state != cherrypy.engine.states.STARTED and self.running:
					# if cherrypy engine stopped but this thread is still alive, restart it.
					print 'CherryPy Engine state = ', cherrypy.engine.state
					print 'Calling cherrypy.engine.start()'
					cherrypy.engine.start()
				if not self.status_thread.is_alive():
					self.status_thread = threading.Thread(target=self.status_thread_func)
					self.status_thread.start()
		except:
			print 'run error'
			traceback.print_exc(file=sys.stdout)
			self.stop()

	def update_proc_info(self):
		self.pn_info[STR_STATUS]

	# thread function for sending status during processing
	def status_thread_func(self):
		try:
			print 'Started Status Thread'
			while self.running:
				self.status_event.wait(self.status_update_interval)
				if self.running:
					self.pn_info[STR_PROC_CPU_PERC] = self.this_process.cpu_percent()
					self.pn_info[STR_PROC_MEM_PERC] = math.floor(self.this_process.memory_percent() * 100) / 100
					self.pn_info[STR_SYS_CPU_PERC] = psutil.cpu_percent()
					#for child in self.this_process.children():
					#	self.pn_info[STR_PROC_CPU_PERC_CHILDREN] += [child.cpu_percent()]
					#	self.pn_info[STR_PROC_MEM_PERC_CHILDREN] += [child.memory_percent()]
					self.send_status_update()
		except:
			print 'status_thread_func error'
			traceback.print_exc(file=sys.stdout)
			#self.stop()
		print 'Stopped Status Thread'

	def parse_aliases(self, alias_str):
		all_aliases = alias_str.split(';')
		alias_dict = dict()
		for single_set in all_aliases:
			split_single = single_set.split(',')
			if len(split_single) > 1:
				alias_dict[split_single[0]] = split_single[1]
		return alias_dict

	def check_for_alias(self, directory_str, alias_dict):
		ret_str = directory_str
		for key in alias_dict.iterkeys():
			if directory_str.startswith(key):
				ret_str = directory_str.replace(key, alias_dict[key])
				break
		return ret_str

	def process_next_job(self):
		if self.running == False:
			return
		print 'checking for jobs to process'
		job_list = self.db.get_all_unprocessed_and_processing_jobs()
		saveout = sys.stdout
		for job_dict in job_list:
			try:
				alias_path = self.check_for_alias(job_dict['DataPath'], self.path_alias_dict)
				alias_path = alias_path.replace('\\', '/')
				print 'processing job:', job_dict['DataPath'], 'alias_path: ', alias_path
				self.pn_info[STR_STATUS] = 'Processing'
				job_dict['Status'] = JOB_PROCESSING_ID
				#job_dict['StartProcWork'] = time.time()
				self.db.update_job(job_dict)
				self.send_job_update(job_dict)
				self.send_status_update()
				maps_set_str = os.path.join(str(alias_path), 'maps_settings.txt')
				f = open(maps_set_str, 'w')
				f.write('	  This file will set some MAPS settings mostly to do with fitting' + '\n')
				f.write('VERSION:' + str(job_dict['Version']).strip() + '\n')
				f.write('DETECTOR_ELEMENTS:' + str(job_dict['DetectorElements']).strip() + '\n')
				f.write('MAX_NUMBER_OF_FILES_TO_PROCESS:' + str(job_dict['MaxFilesToProc']).strip() + '\n')
				f.write('MAX_NUMBER_OF_LINES_TO_PROCESS:' + str(job_dict['MaxLinesToProc']).strip() + '\n')
				f.write('QUICK_DIRTY:' + str(job_dict['QuickAndDirty']).strip() + '\n')
				f.write('XRF_BIN:' + str(job_dict['XRF_Bin']).strip() + '\n')
				f.write('NNLS:' + str(job_dict['NNLS']).strip() + '\n')
				f.write('XANES_SCAN:' + str(job_dict['XANES_Scan']).strip() + '\n')
				f.write('DETECTOR_TO_START_WITH:' + str(job_dict['DetectorToStartWith']).strip() + '\n')
				f.write('BEAMLINE:' + str(job_dict['BeamLine']).strip() + '\n')
				f.write('DatasetFilesToProc:' + str(job_dict['DatasetFilesToProc']).strip() + '\n')
				standard_filenames = job_dict['Standards'].split(';')
				for item in standard_filenames:
					f.write('STANDARD:' + item.strip() + '\n')
				f.close()
				proc_mask = int(job_dict['ProcMask'])
				key_a = 0
				key_b = 0
				key_c = 0
				key_d = 0
				key_e = 0
				key_f = 0 # for netcdf to hdf5 future feature
				if proc_mask & 1 == 1:
					key_a = 1
				if proc_mask & 2 == 2:
					key_b = 1
				if proc_mask & 4 == 4:
					key_c = 1
				if proc_mask & 8 == 8:
					key_d = 1
				if proc_mask & 16 == 16:
					key_e = 1
				if proc_mask & 32 == 32:
					key_f = 1
				#os.chdir(job_dict['DataPath'])
				log_name = 'Job_' + str(job_dict['Id']) + '_' + datetime.strftime(datetime.now(), "%y_%m_%d_%H_%M_%S") + '.log'
				job_dict['Log_Path'] = log_name
				log_path = os.path.join(STR_JOB_LOG_DIR_NAME, log_name)
				job_status = multiprocessing.Value('i', JOB_PROCESSING_ID)
				proc = multiprocessing.Process(target=self.__proc_func__, args=(job_status, log_path, alias_path, key_a, key_b, key_c, key_d, key_e))
				proc.start()
				self.this_process = psutil.Process(proc.pid)
				proc.join()
				self.this_process = psutil.Process(os.getpid())
				job_dict['Status'] = job_status.value
				#logfile = open(log_path, 'wt')
				#sys.stdout = logfile
				#maps_batch(wdir=alias_path, a=key_a, b=key_b, c=key_c, d=key_d, e=key_e)
				#sys.stdout = saveout
				#logfile.close()
				#job_dict['Status'] = JOB_COMPLETED_ID
			except:
				print 'Error processing', job_dict['DataPath']
				traceback.print_exc(file=sys.stdout)
				sys.stdout = saveout
				job_dict['Status'] = JOB_ERROR_ID
			self.db.update_job(job_dict)
			self.send_job_update(job_dict)
			self.send_status_update()
			print 'done processing job', job_dict['DataPath']
		print 'Finished Processing, going to Idle'
		self.pn_info[STR_STATUS] = 'Idle'
		self.send_status_update()

	def __proc_func__(self, job_status, log_name, alias_path, key_a, key_b, key_c, key_d, key_e):
		saveout = sys.stdout
		try:
			logfile = open(log_name, 'wt')
			sys.stdout = logfile
			maps_batch(wdir=alias_path, a=key_a, b=key_b, c=key_c, d=key_d, e=key_e)
			sys.stdout = saveout
			logfile.close()
			job_status.value = JOB_COMPLETED_ID
		except:
			print datetime.now(), 'Error processing', alias_path
			traceback.print_exc(file=sys.stdout)
			sys.stdout = saveout
			job_status.value = JOB_ERROR_ID

	def stop(self):
		self.running = False
		self.status_event.set()
		if self.status_thread is not None:
			print 'Waiting for status thread to join'
			self.status_thread.join()
		self.new_job_event.set()
		try:
			self.pn_info[STR_STATUS] = 'Offline'
			self.send_status_update()
			self.session.delete(self.scheduler_pn_url, data=json.dumps(self.pn_info))
		except:
			print datetime.now(), 'stop error'
			traceback.print_exc(file=sys.stdout)
		cherrypy.engine.exit()

	def send_status_update(self):
		try:
			self.pn_info[STR_HEARTBEAT] = str(datetime.now())
			self.session.put(self.scheduler_pn_url, data=json.dumps(self.pn_info))
		except:
			print datetime.now(), 'Error sending status update'
			#traceback.print_exc(file=sys.stdout)

	def send_job_update(self, job_dict):
		try:
			self.session.put(self.scheduler_job_url, params=self.pn_info, data=json.dumps(job_dict))
			print 'sent status'
		except:
			print datetime.now(), 'Error sending job update'
			#traceback.print_exc(file=sys.stdout)

