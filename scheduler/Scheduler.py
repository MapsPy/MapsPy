import os, os.path

import Settings
import requests
from handlers.SchedulerHandlers import SchedulerHandler, SchedulerJobsWebService, SchedulerProcessNodeWebService
from plugins.DatabasePlugin import DatabasePlugin
from plugins.SQLiteDB import SQLiteDB
import json
import cherrypy

db = DatabasePlugin(cherrypy.engine, SQLiteDB)

class Scheduler(object):
	def __init__(self, settings):
		self.all_settings = settings
		self.settings = settings.getSetting(Settings.SECTION_SERVER)
		cherrypy.config.update({
			'server.socket_host': self.settings[Settings.SERVER_HOSTNAME],
			'server.socket_port': int(self.settings[Settings.SERVER_PORT]),
		})
		cherrypy.engine.subscribe("new_job", self.callback_new_job)
		cherrypy.engine.subscribe("process_node_update", self.callback_process_node_update)
		self.conf = {
			'/': {
				'tools.sessions.on': True,
				'tools.staticdir.root': os.path.abspath(os.getcwd())
			},
			'/process_node': {
				'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
				'tools.response_headers.on': True,
				'tools.response_headers.headers': [('Content-Type', 'text/plain')],
			},
			'/job': {
				'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
				'tools.response_headers.on': True,
				'tools.response_headers.headers': [('Content-Type', 'text/plain')],
			},
			'/static': {
				'tools.staticdir.on': True,
				'tools.staticdir.dir': './public'
			}
		}
	
	def callback_new_job(self, job):
		print 'callback got new job', job
		node_list = db.get_all_process_nodes()
		for node in node_list:
			if node['Status'] == 'Idle':
				url = 'http://' + str(node['Hostname']) + ':' + str(node['Port']) + '/job_queue'
				print 'sending job to ',node['ComputerName'], 'url',url
				s = requests.Session()
				r = s.post(url, data=json.dumps(job))
				print 'result', r.status_code,':',r.text
				break

	def callback_process_node_update(self, node):
		print node
		if node['Status'] == 'Idle':
			job_list = db.get_all_unprocessed_jobs()
			if len(job_list) > 0:
				job = job_list[0]
				url = 'http://' + str(node['Hostname']) + ':' + str(node['Port']) + '/job_queue'
				print 'sending job to ',node['ComputerName'], 'url',url
				s = requests.Session()
				r = s.post(url, data=json.dumps(job))
				print 'result', r.status_code,':',r.text

	def run(self):
		db.subscribe()
		db.create_tables()
		webapp = SchedulerHandler(db, self.all_settings)
		webapp.process_node = SchedulerProcessNodeWebService(db)
		webapp.job = SchedulerJobsWebService(db)
		cherrypy.quickstart(webapp, '/', self.conf)

