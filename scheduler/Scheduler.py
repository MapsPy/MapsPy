import os, os.path

import Settings
from handlers.JobStatusHandler import JobStatusHandler, ComputerNodeWebService
#from plugins.BaseDatabasePlugin import BaseDatabasePlugin
#from plugins.SQLiteDB import SQLiteDB

import cherrypy

class Scheduler(object):
	def __init__(self, settings):
		self.settings = settings
		HostName = settings[Settings.SERVER_HOSTNAME]
		Port = int(settings[Settings.SERVER_PORT])
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
			'/static': {
				'tools.staticdir.on': True,
				'tools.staticdir.dir': './public'
			}
		}
	
	def run(self):
		webapp = JobStatusHandler()
		#BaseDatabasePlugin(cherrypy.engine, SQLiteDB).subscribe()
		webapp.process_node = ComputerNodeWebService()
		cherrypy.quickstart(webapp, '/', self.conf)

