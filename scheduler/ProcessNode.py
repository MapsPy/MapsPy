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
		self.scheduler_host = serverSettings[Settings.SERVER_SCHEDULER_HOSTNAME]
		self.scheduler_port = serverSettings[Settings.SERVER_SCHEDULER_PORT]
		self.session = requests.Session()
		self.scheduler_url = 'http://'+self.scheduler_host+':'+self.scheduler_port+'/process_node'
		self.db_name = pnSettings[Settings.PROCESS_NODE_DATABASE_NAME]
		self.db = DatabasePlugin(cherrypy.engine, SQLiteDB, self.db_name)
		self.running = True
	def run(self):
		webapp = ProcessNodeHandler()
		self.db.subscribe()
		self.db.create_tables()
		webapp.job_queue = ProcessNodeJobsWebService(self.db)
		#cherrypy.quickstart(webapp, '/', self.conf)
		cherrypy.tree.mount(webapp, '/', self.conf)
		cherrypy.engine.start()
		try:
			print 'posting to scheduler',self.scheduler_url
			self.session.post(self.scheduler_url, data=json.dumps(self.pn_info))
		except:
			print 'Error sending post'
		self.pn_info[STR_STATUS] = 'Idle'
		try:
			while self.running:
				self.send_status_update()
				time.sleep(10)
		except:
			self.stop()

	def stop(self):
		self.running = False
		self.pn_info[STR_STATUS] = 'Offline'
		self.send_status_update()
		self.session.delete(self.scheduler_url, data=json.dumps(self.pn_info))
		cherrypy.engine.exit()

	def send_status_update(self):
		try:
			self.pn_info[STR_HEARTBEAT] = str(datetime.now())
			self.session.put(self.scheduler_url, data=json.dumps(self.pn_info))
		except:
			print 'Error sending status update'

