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


import os, os.path
import Settings
import requests
from handlers.SchedulerHandlers import SchedulerHandler, SchedulerJobsWebService, SchedulerProcessNodeWebService
from plugins.DatabasePlugin import DatabasePlugin
from plugins.SQLiteDB import SQLiteDB
import json
import cherrypy
import traceback
import logging
import logging.handlers
import threading

db = DatabasePlugin(cherrypy.engine, SQLiteDB)

class Scheduler(object):
	def __init__(self, settings):
		self.all_settings = settings
		self.settings = settings.getSetting(Settings.SECTION_SERVER)
		self.job_lock = threading.RLock()
		cherrypy.config.update({
			'server.socket_host': self.settings[Settings.SERVER_HOSTNAME],
			'server.socket_port': int(self.settings[Settings.SERVER_PORT]),
			'log.access_file': "logs/scheduler_access.log",
			'log.error_file': "logs/scheduler_error.log"
		})
		cherrypy.engine.subscribe("new_job", self.callback_new_job)
		cherrypy.engine.subscribe("process_node_update", self.callback_process_node_update)
		if hasattr(cherrypy.engine, 'signal_handler'):
			cherrypy.engine.signal_handler.subscribe()
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
		try:
			self.job_lock.acquire(True)
			p_node = None
			if job['Process_Node_Id'] > -1:
				p_node = db.get_process_node_by_id()
			else:
				node_list = db.get_all_process_nodes()
				print 'searching for idle node'
				for node in node_list:
					if node['Status'] == 'Idle':
						p_node = node
						break
			if p_node != None:
				job['Process_Node_Id'] = p_node['Id']
				url = 'http://' + str(p_node['Hostname']) + ':' + str(p_node['Port']) + '/job_queue'
				print 'sending job to ', p_node['ComputerName'], 'url', url
				s = requests.Session()
				r = s.post(url, data=json.dumps(job))
				print 'result', r.status_code, ':', r.text
				self.job_lock.release()
		except:
			self.job_lock.release()
			exc_str = traceback.format_exc()
			return exc_str

	def callback_process_node_update(self, node):
		print 'callback', node['ComputerName']
		try:
			self.job_lock.acquire(True)
			if node.has_key('Id') == False:
				print 'getting id for node', node
				new_node = db.get_process_node_by_name(node['ComputerName'])
				node['Id'] = new_node['Id']
				print 'updated node', node
				s = requests.Session()
				url = 'http://' + str(node['Hostname']) + ':' + str(node['Port']) + '/update_id'
				r = s.post(url, data={'Id': node['Id']})
				print 'update result', r.status_code, ':', r.text
			if node['Status'] == 'Idle':
				job_list = db.get_all_unprocessed_jobs()
				for job in job_list:
					print 'checking job', job
					if job['Process_Node_Id'] < 0 or job['Process_Node_Id'] == node['Id']:
						job['Process_Node_Id'] = node['Id']
						url = 'http://' + str(node['Hostname']) + ':' + str(node['Port']) + '/job_queue'
						print '_sending job to ', node['ComputerName'], 'url', url
						s = requests.Session()
						r = s.post(url, data=json.dumps(job))
						print 'result', r.status_code, ':', r.text
						break
			self.job_lock.release()
		except:
			self.job_lock.release()
			exc_str = traceback.format_exc()
			print exc_str

	def _setup_logging_(self, log, logtype, logname):
		maxBytes = getattr(log, "rot_maxBytes", 20971520) # 20Mb
		backupCount = getattr(log, "rot_backupCount", 10)
		fname = getattr(log, logtype, logname)
		h = logging.handlers.RotatingFileHandler(fname, 'a', maxBytes, backupCount)
		h.setLevel(logging.DEBUG)
		h.setFormatter(cherrypy._cplogging.logfmt)
		log.error_log.addHandler(h)

	def run(self):
		db.subscribe()
		db.create_tables()
		webapp = SchedulerHandler(db, self.all_settings)
		webapp.process_node = SchedulerProcessNodeWebService(db)
		webapp.job = SchedulerJobsWebService(db)
		app = cherrypy.tree.mount(webapp, '/', self.conf)
		self._setup_logging_(app.log, "rot_error_file", "logs/scheduler_error.log")
		self._setup_logging_(app.log, "rot_access_file", "logs/scheduler_access.log")
		cherrypy.engine.start()
		cherrypy.engine.block()
		print 'done blocking'
