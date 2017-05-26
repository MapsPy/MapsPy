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
import logging
import logging.handlers
import threading
import signal
import psutil
import multiprocessing
from datetime import datetime
from RestBase import RestBase
from plugins.DatabasePlugin import DatabasePlugin
from plugins.SQLiteDB import SQLiteDB
from handlers.ProcessNodeHandlers import ProcessNodeHandler, ProcessNodeJobsWebService
import math
import Constants
import modules

class ProcessNode(RestBase):
	def __init__(self, settings):
		RestBase.__init__(self)
		self.settings = settings
		serverSettings = settings.getSetting(Settings.SECTION_SERVER)
		pnSettings = settings.getSetting(Settings.SECTION_PROCESS_NODE)
		#print 'Server settings ', serverSettings
		#print 'ProcessNode Settings ', pnSettings
		self.pn_info = {Constants.PROCESS_NODE_COMPUTERNAME: pnSettings[Settings.PROCESS_NODE_NAME],
					Constants.PROCESS_NODE_NUM_THREADS: pnSettings[Settings.PROCESS_NODE_THREADS],
					Constants.PROCESS_NODE_HOSTNAME: serverSettings[Settings.SERVER_HOSTNAME],
					Constants.PROCESS_NODE_PORT: serverSettings[Settings.SERVER_PORT],
					Constants.PROCESS_NODE_STATUS: Constants.PROCESS_NODE_STATUS_BOOT_UP,
					Constants.PROCESS_NODE_HEARTBEAT: str(datetime.now()),
					Constants.PROCESS_NODE_PROCESS_CPU_PERCENT: 0.0,
					Constants.PROCESS_NODE_PROCESS_MEM_PERCENT: 0.0,
					Constants.PROCESS_NODE_SYSTEM_CPU_PERCENT: 0.0,
					Constants.PROCESS_NODE_SYSTEM_MEM_PERCENT: 0.0,
					Constants.PROCESS_NODE_SYSTEM_SWAP_PERCENT: 0.0,
					Constants.PROCESS_NODE_SUPPORTED_SOFTWARE: pnSettings[Settings.PROCESS_NODE_SOFTWARE_LIST]
					}
		cherrypy.config.update({
			'server.socket_host': serverSettings[Settings.SERVER_HOSTNAME],
			'server.socket_port': int(serverSettings[Settings.SERVER_PORT]),
			#'log.access_file': "logs/" + str(pnSettings[Settings.PROCESS_NODE_NAME]) + "_access.log",
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
				'request.methods_with_bodies': ('POST', 'PUT', 'DELETE')
			},
			'/static': {
				'tools.staticdir.on': True,
				'tools.staticdir.dir': './public'
			}
		}
		self.logger = logging.getLogger(__name__)
		self._setup_logging_(self.logger, "rot_file", "logs/" + self.pn_info[Constants.PROCESS_NODE_COMPUTERNAME] + "_pn.log", True)
		self.logger.info('pnSettings %s', pnSettings)
		self.new_job_event = threading.Event()
		self.status_event = threading.Event()
		self.logger.info('Setup signal handler')
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
		self.software_dict = self.parse_json_file(pnSettings[Settings.PROCESS_NODE_SOFTWARE_LIST])
		self.path_alias_dict = self.parse_json_file(pnSettings[Settings.PROCESS_NODE_PATH_ALIAS])
		#self.xrf_maps_path = pnSettings[Settings.PROCESS_NODE_XRF_MAPS_PATH]
		#self.xrf_maps_exe = pnSettings[Settings.PROCESS_NODE_XRF_MAPS_EXE]
		self.logger.info('alias paths %s', self.path_alias_dict)
		self.session = requests.Session()
		self.scheduler_pn_url = 'http://' + self.scheduler_host + ':' + self.scheduler_port + '/process_node'
		self.scheduler_job_url = 'http://' + self.scheduler_host + ':' + self.scheduler_port + '/job'
		self.db_name = pnSettings[Settings.PROCESS_NODE_DATABASE_NAME]
		self.db = DatabasePlugin(cherrypy.engine, SQLiteDB, self.db_name)
		cherrypy.engine.subscribe("new_job", self.callback_new_job)
		cherrypy.engine.subscribe("update_id", self.callback_update_id)
		cherrypy.engine.subscribe("delete_job", self.callback_delete_job)
		cherrypy.engine.subscribe("send_job_update", self.callback_send_job_update)
		self.create_directories()
		self.running = True
		self.status_thread = None
		self.this_process = psutil.Process(os.getpid())

	def unix_handle_sigint(self, sig, frame):
		self.logger.info('unix_handle_sigint %s %s', sig, frame)
		self.stop()

	def win_handle_sigint(self, sig):
		self.logger.info('win_handle_sigint %s', sig)
		self.stop()

	def create_directories(self):
		if not os.path.exists(Constants.STR_JOB_LOG_DIR_NAME):
			os.makedirs(Constants.STR_JOB_LOG_DIR_NAME)

	def callback_new_job(self, val):
		self.new_job_event.set()

	def callback_send_job_update(self, val):
		self.send_job_update(val)

	def callback_update_id(self, new_id):
		self.pn_info[Constants.PROCESS_NODE_ID] = int(new_id)

	def callback_delete_job(self, job):
		try:
			job[Constants.JOB_STATUS] = Constants.JOB_STATUS_CANCELED
			#self.db.update_job(job)
			self.db.delete_job_by_id(job[Constants.JOB_ID])
			if self.this_process != psutil.Process(os.getpid()):
				parent = psutil.Process(self.this_process.pid)
				for child in parent.children(recursive=True):  # or parent.children() for recursive=False
					child.kill()
				self.this_process.kill()
				self.this_process = psutil.Process(os.getpid())
				job[Constants.JOB_STATUS] = Constants.JOB_STATUS_CANCELED
			self.send_job_update(job)
		except:
			self.logger.exception('callback_delete_job: Error')

	def parse_json_file(self, json_file):
		#list should be comma separated for each software key and ; separated for name;path;exe
		with open(json_file) as data_file:
			parsed_obj = json.load(data_file)
		return parsed_obj

	def run(self):
		webapp = ProcessNodeHandler()
		webapp.set_software_dir(self.software_dict)
		self.db.subscribe()
		self.db.create_tables()
		webapp.job_queue = ProcessNodeJobsWebService(self.db, self.logger)
		app = cherrypy.tree.mount(webapp, '/', self.conf)
		#self._setup_logging_(app.log, "rot_error_file", "logs/" + self.pn_info[Constants.PROCESS_NODE_COMPUTERNAME] + "_error.log", False, True)
		#self._setup_logging_(app.log, "rot_access_file", "logs/" + self.pn_info[Constants.PROCESS_NODE_COMPUTERNAME] + "_access.log", False, True)
		cherrypy.engine.start()
		try:
			self.logger.info('posting to scheduler %s', self.scheduler_pn_url)
			self.session.post(self.scheduler_pn_url, data=json.dumps(self.pn_info))
		except:
			self.logger.error('Error sending post')
			#print datetime.now(), 'Error sending post'
		self.pn_info[Constants.PROCESS_NODE_STATUS] = Constants.PROCESS_NODE_STATUS_IDLE
		# start status thread
		if self.status_thread == None:
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
					self.logger.info( 'CherryPy Engine state = %s', cherrypy.engine.state)
					self.logger.info('Calling cherrypy.engine.start()')
					cherrypy.engine.start()
				if not self.status_thread.is_alive():
					self.status_thread = threading.Thread(target=self.status_thread_func)
					self.status_thread.start()
		except:
			self.logger.exception('run error')
			self.stop()

	def update_proc_info(self):
		self.pn_info[Constants.PROCESS_NODE_STATUS]

	# thread function for sending status during processing
	def status_thread_func(self):
		try:
			self.logger.info('Started Status Thread')
			while self.running:
				self.status_event.wait(self.status_update_interval)
				if self.running:
					self.pn_info[Constants.PROCESS_NODE_PROCESS_CPU_PERCENT] = self.this_process.cpu_percent()
					self.pn_info[Constants.PROCESS_NODE_PROCESS_MEM_PERCENT] = math.floor(self.this_process.memory_percent() * 100) / 100
					self.pn_info[Constants.PROCESS_NODE_SYSTEM_CPU_PERCENT] = psutil.cpu_percent()
					self.pn_info[Constants.PROCESS_NODE_SYSTEM_MEM_PERCENT] = psutil.virtual_memory().percent
					self.pn_info[Constants.PROCESS_NODE_SYSTEM_SWAP_PERCENT] = psutil.swap_memory().percent
					self.send_status_update()
		except:
			self.logger.exception('status_thread_func error')
		self.logger.warning('Stopped Status Thread')

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
		self.logger.info('checking for jobs to process')
		job_list = self.db.get_all_unprocessed_and_processing_jobs()
		for job_dict in job_list:
			try:
				alias_path = self.check_for_alias(job_dict[Constants.JOB_DATA_PATH], self.path_alias_dict)
				alias_path = alias_path.replace('\\', '/')
				self.logger.info('processing job: %s  alias_path: %s', job_dict[Constants.JOB_DATA_PATH], alias_path)
				self.pn_info[Constants.PROCESS_NODE_STATUS] = Constants.PROCESS_NODE_STATUS_PROCESSING

				job_dict[Constants.JOB_STATUS] = Constants.JOB_STATUS_PROCESSING
				job_dict[Constants.JOB_START_PROC_TIME] = datetime.ctime(datetime.now())
				log_name = 'Job_' + str(job_dict[Constants.JOB_ID]) + '_' + datetime.strftime(datetime.now(), "%y_%m_%d_%H_%M_%S") + '.log'
				job_dict[Constants.JOB_LOG_PATH] = log_name

				job_logger = None
				proc = None
				exitcode = -1
				self.db.update_job(job_dict)
				self.send_job_update(job_dict)
				self.send_status_update()

				if job_dict[Constants.JOB_EXPERIMENT] in self.software_dict:
					prog = self.software_dict[job_dict[Constants.JOB_EXPERIMENT]]
					method = getattr(modules, prog['module'])
					proc = multiprocessing.Process(target=method.start_job, args=(log_name, alias_path, job_dict, prog['path'], prog['exe'], exitcode))
				if not proc == None:
					proc.start()
					self.this_process = psutil.Process(proc.pid)
					#  todo: if allow more than 1 job put in a list and check on it
					proc.join()
					exitcode = proc.exitcode
				self.this_process = psutil.Process(os.getpid())
				self.logger.debug("Process finished with exitcode %s", exitcode)
				job_dict[Constants.JOB_FINISH_PROC_TIME] = datetime.ctime(datetime.now())
				if exitcode != 0:
					self.logger.info('finished processing job with status ERROR')
					job_dict[Constants.JOB_STATUS] = Constants.JOB_STATUS_GENERAL_ERROR
				else:
					self.logger.info('finished processing job with status COMPLETED')
					job_dict[Constants.JOB_STATUS] = Constants.JOB_STATUS_COMPLETED
				if not job_logger == None:
					handlers = job_logger.handlers[:]
					for handler in handlers:
						handler.close()
						job_logger.removeHandler(handler)
			except:
				self.logger.exception('Error processing %s', job_dict[Constants.JOB_DATA_PATH])
				job_dict[Constants.JOB_FINISH_PROC_TIME] = datetime.ctime(datetime.now())
				job_dict[Constants.JOB_STATUS] = Constants.JOB_STATUS_GENERAL_ERROR
				try:
					handlers = job_logger.handlers[:]
					for handler in handlers:
						handler.close()
						job_logger.removeHandler(handler)
				except:
					pass
			if self.db.update_job(job_dict):
				self.send_job_update(job_dict)
			self.send_status_update()
			self.logger.info('Done processing job: %s STATUS = %s', job_dict[Constants.JOB_DATA_PATH], job_dict[Constants.JOB_STATUS])
		self.logger.info('Finished Processing, going to Idle')
		self.pn_info[Constants.PROCESS_NODE_STATUS] = Constants.PROCESS_NODE_STATUS_IDLE
		self.send_status_update()

	def stop(self):
		self.running = False
		self.status_event.set()
		if self.status_thread is not None:
			self.logger.info('Waiting for status thread to join')
			self.status_thread.join()
		self.new_job_event.set()
		try:
			self.pn_info[Constants.PROCESS_NODE_STATUS] = Constants.PROCESS_NODE_STATUS_OFFLINE
			self.pn_info[Constants.PROCESS_NODE_PROCESS_CPU_PERCENT] = 0.0
			self.pn_info[Constants.PROCESS_NODE_PROCESS_MEM_PERCENT] = 0.0
			self.pn_info[Constants.PROCESS_NODE_SYSTEM_CPU_PERCENT] = 0.0
			self.pn_info[Constants.PROCESS_NODE_SYSTEM_MEM_PERCENT] = 0.0
			self.pn_info[Constants.PROCESS_NODE_SYSTEM_SWAP_PERCENT] = 0.0
			self.send_status_update()
			self.session.delete(self.scheduler_pn_url, data=json.dumps(self.pn_info))
		except:
			self.logger.exception('stop error')
		cherrypy.engine.exit()

	def send_status_update(self):
		try:
			self.pn_info[Constants.PROCESS_NODE_HEARTBEAT] = str(datetime.now())
			self.session.put(self.scheduler_pn_url, data=json.dumps(self.pn_info))
		except:
			self.logger.error('Error sending status update')

	def send_job_update(self, job_dict):
		try:
			self.session.put(self.scheduler_job_url, params=self.pn_info, data=json.dumps(job_dict))
			self.logger.info('sent job status %s', job_dict)
		except:
			self.logger.exception('Error sending job update')

'''
def call_software(log_name, alias_path, job_dict, software_path, software_exe, exitcode):
	try:
		log_file = open('job_logs/' + log_name, 'w')
		args = [software_exe]
		args += [job_dict[Constants.JOB_ARGS]]
		if os.name == "nt":
			exitcode = subprocess.call(args, cwd=software_path, stdout=log_file, stderr=log_file, shell=True)
		else:
			exitcode = subprocess.call(args, cwd=software_path, stdout=log_file, stderr=log_file, shell=False)
		print 'exitcode = ', exitcode
		log_file.close()
	except:
		exc_str = traceback.format_exc()
		print exc_str
		exitcode = -1
'''