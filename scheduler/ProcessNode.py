# include parent directory for imports
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import Settings
import requests
import cherrypy
import time
import json
import threading
import traceback
from datetime import datetime
from plugins.DatabasePlugin import DatabasePlugin
from plugins.SQLiteDB import SQLiteDB
from handlers.ProcessNodeHandlers import ProcessNodeHandler, ProcessNodeJobsWebService
import maps_batch

STR_COMPUTER_NAME = 'ComputerName'
STR_NUM_THREADS = 'NumThreads'
STR_HOSTNAME = 'Hostname'
STR_PORT = 'Port'
STR_STATUS = 'Status'
STR_HEARTBEAT = 'Heartbeat'

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
					 STR_HEARTBEAT: str(datetime.now()) }
		cherrypy.config.update({
			'server.socket_host': serverSettings[Settings.SERVER_HOSTNAME],
			'server.socket_port': int(serverSettings[Settings.SERVER_PORT]),
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
		self.status_update_interval = 10
		self.scheduler_host = serverSettings[Settings.SERVER_SCHEDULER_HOSTNAME]
		self.scheduler_port = serverSettings[Settings.SERVER_SCHEDULER_PORT]
		self.session = requests.Session()
		self.scheduler_pn_url = 'http://'+self.scheduler_host+':'+self.scheduler_port+'/process_node'
		self.scheduler_job_url = 'http://'+self.scheduler_host+':'+self.scheduler_port+'/job'
		self.db_name = pnSettings[Settings.PROCESS_NODE_DATABASE_NAME]
		self.db = DatabasePlugin(cherrypy.engine, SQLiteDB, self.db_name)
		cherrypy.engine.subscribe("new_job", self.callback_new_job)
		self.running = True

	def callback_new_job(self, val):
		self.new_job_event.set()

	def run(self):
		webapp = ProcessNodeHandler()
		self.db.subscribe()
		self.db.create_tables()
		webapp.job_queue = ProcessNodeJobsWebService(self.db)
		#cherrypy.quickstart(webapp, '/', self.conf)
		cherrypy.tree.mount(webapp, '/', self.conf)
		cherrypy.engine.start()
		try:
			print 'posting to scheduler',self.scheduler_pn_url
			self.session.post(self.scheduler_pn_url, data=json.dumps(self.pn_info))
		except:
			print 'Error sending post'
		self.pn_info[STR_STATUS] = 'Idle'
		self.new_job_event.set() # set it at start to check for unfinished jobs
		try:
			while self.running:
				self.new_job_event.wait(self.status_update_interval)
				if self.new_job_event.is_set():
					self.new_job_event.clear()
					self.process_next_job()
				else:
					self.send_status_update()
		except:
			self.stop()

	def process_next_job(self):
		print 'checking for jobs to process'
		job_list = self.db.get_all_jobs()
		for job_dict in job_list:
			self.pn_info[STR_STATUS] = 'Processing'
			job_dict['Status'] = 1 #1 = processing
			#job_dict['StartProcWork'] = time.time()
			#self.db.update_job(job_dict)
			self.send_job_update(job_dict)
			self.send_status_update()
			print 'processing job', job_dict['DataPath']
			maps_set_str = os.path.join(str(job_dict['DataPath']),'maps_settings.txt')
			try:
				f = open(maps_set_str, 'w')
				f.write('	  This file will set some MAPS settings mostly to do with fitting'+'\n')
				f.write('VERSION:' + str(job_dict['Version']).strip()+'\n')
				f.write('DETECTOR_ELEMENTS:' + str(job_dict['DetectorElements']).strip()+'\n')
				f.write('MAX_NUMBER_OF_FILES_TO_PROCESS:' + str(job_dict['MaxFileToProc']).strip()+'\n')
				f.write('MAX_NUMBER_OF_LINES_TO_PROCESS:' + str(job_dict['MaxLinesToProc']).strip()+'\n')
				f.write('QUICK_DIRTY:' + str(job_dict['QuickAndDirty']).strip()+'\n')
				f.write('XRF_BIN:' + str(job_dict['XRF_Bin']).strip()+'\n')
				f.write('NNLS:' + str(job_dict['NNLS']).strip()+'\n')
				f.write('XANES_SCAN:' + str(job_dict['XANES_Scan']).strip()+'\n')
				f.write('DETECTOR_TO_START_WITH:' + str(job_dict['DetectorToStartWith']).strip()+'\n')
				f.write('BEAMLINE:' + str(job_dict['BeamLine']).strip()+'\n')
				standard_filenames = job_dict['Standards'].split(';')
				for item in standard_filenames:
					f.write('STANDARD:' + item.strip()+'\n')
				f.close()
				proc_mask = int(job_dict['ProcMask'])
				key_a = 0
				key_b = 0
				key_c = 0
				key_d = 0
				key_e = 0
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
				#os.chdir(job_dict['DataPath'])
				maps_batch.main(wdir=job_dict['DataPath'], a=key_a, b=key_b, c=key_c, d=key_d, e=key_e)
			except:
				print 'Error processing',job_dict['DataPath']
				traceback.print_exc(file=sys.stdout)
			print 'done processing job', job_dict['DataPath']
			job_dict['Status'] = 3 #3 = completed
			#job_dict['StopWork'] = time.time()
			self.db.update_job(job_dict)
			self.send_job_update(job_dict)
		self.pn_info[STR_STATUS] = 'Idle'
		self.send_status_update()
	def stop(self):
		self.running = False
		self.new_job_event.set()
		try:
			self.pn_info[STR_STATUS] = 'Offline'
			self.send_status_update()
		except:
			pass
		try:
			self.session.delete(self.scheduler_pn_url, data=json.dumps(self.pn_info))
			cherrypy.engine.exit()
		except:
			pass

	def send_status_update(self):
		try:
			self.pn_info[STR_HEARTBEAT] = str(datetime.now())
			self.session.put(self.scheduler_pn_url, data=json.dumps(self.pn_info))
		except:
			print 'Error sending status update'

	def send_job_update(self, job_dict):
		try:
			self.session.put(self.scheduler_job_url, data=json.dumps(job_dict))
			print 'sent status'
		except:
			print 'Error sending status update'

