import Settings
import requests
import cherrypy
import time
import json
from datetime import datetime

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
		print pnSettings
		self.pn_info = {STR_COMPUTER_NAME: pnSettings[Settings.PROCESS_NODE_NAME],
					 STR_NUM_THREADS: pnSettings[Settings.PROCESS_NODE_THREADS],
					 STR_HOSTNAME: serverSettings[Settings.SERVER_HOSTNAME],
					 STR_PORT: serverSettings[Settings.SERVER_PORT],
					 STR_STATUS: 'Bootup',
					 STR_HEARTBEAT: str(datetime.now()) }
		self.scheduler_host = serverSettings[Settings.SERVER_SCHEDULER_HOSTNAME]
		self.scheduler_port = serverSettings[Settings.SERVER_SCHEDULER_PORT]
		self.session = requests.Session()
		self.scheduler_url = 'http://'+self.scheduler_host+':'+self.scheduler_port+'/process_node'
		self.running = True
		print self.scheduler_url
	def run(self):
		try:
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

	def send_status_update(self):
		try:
			self.pn_info[STR_HEARTBEAT] = str(datetime.now())
			self.session.put(self.scheduler_url, data=json.dumps(self.pn_info))
		except:
			print 'Error sending status update'

