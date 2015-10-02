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
from tools import Mailman
import json
import cherrypy
import traceback
import logging
import logging.handlers
import threading
import datetime
import Constants

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
		self.mailman = Mailman.mainman(self.settings[Settings.SERVER_SMTP_ADDRESS],
								self.settings[Settings.SERVER_FROM_ADDRESS],
								self.settings[Settings.SERVER_MAIL_USERNAME],
								self.settings[Settings.SERVER_MAIL_PASSWORD])
		cherrypy.engine.subscribe("new_job", self.callback_new_job)
		cherrypy.engine.subscribe("update_job", self.callback_update_job)
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
		print datetime.now(), 'callback got new job', job
		try:
			self.job_lock.acquire(True)
			p_node = None
			if job[Constants.JOB_PROCESS_NODE_ID] > -1:
				p_node = db.get_process_node_by_id(int(job[Constants.JOB_PROCESS_NODE_ID]))
			else:
				node_list = db.get_all_process_nodes()
				print datetime.now(), 'searching for idle node'
				for node in node_list:
					if node[Constants.PROCESS_NODE_STATUS] == Constants.PROCESS_NODE_STATUS_IDLE:
						p_node = node
						break
			if p_node != None:
				job[Constants.JOB_PROCESS_NODE_ID] = p_node[Constants.PROCESS_NODE_ID]
				url = 'http://' + str(p_node[Constants.PROCESS_NODE_HOSTNAME]) + ':' + str(p_node[Constants.PROCESS_NODE_PORT]) + '/job_queue'
				print datetime.now(), 'sending job to ', p_node[Constants.PROCESS_NODE_COMPUTERNAME], 'url', url
				s = requests.Session()
				r = s.post(url, data=json.dumps(job))
				print datetime.now(), 'result', r.status_code, ':', r.text
			self.job_lock.release()
		except:
			self.job_lock.release()
			exc_str = traceback.format_exc()
			return exc_str

	def callback_update_job(self, job):
		pass
		#if Constants.JOB_STATUS in job and Constants.JOB_EMAILS in job:
		#	#if job[Constants.JOB_STATUS] > Constants.JOB_STATUS_PROCESSING and len(job[Constants.JOB_EMAILS]) > 0:
		#		print datetime.now(), 'sending completed emails'
		#		mesg = '...'
		#		try:
		#			self.mailman.send(job[Constants.JOB_EMAILS], Constants.EMAIL_SUBJECT, mesg)
		#		except:
		#			exc_str = traceback.format_exc()
		#			print datetime.now(), exc_str

	def callback_process_node_update(self, node):
		print datetime.now(), 'callback', node[Constants.PROCESS_NODE_COMPUTERNAME]
		try:
			self.job_lock.acquire(True)
			if not Constants.PROCESS_NODE_ID in node:
				print datetime.now(), 'getting id for node', node
				new_node = db.get_process_node_by_name(node[Constants.PROCESS_NODE_COMPUTERNAME])
				node[Constants.PROCESS_NODE_ID] = new_node[Constants.PROCESS_NODE_ID]
				print datetime.now(), 'updated node', node
				s = requests.Session()
				url = 'http://' + str(node[Constants.PROCESS_NODE_HOSTNAME]) + ':' + str(node[Constants.PROCESS_NODE_PORT]) + '/update_id'
				r = s.post(url, data={Constants.PROCESS_NODE_ID: node[Constants.PROCESS_NODE_ID]})
				print datetime.now(), 'update result', r.status_code, ':', r.text
			if node[Constants.PROCESS_NODE_STATUS] == Constants.PROCESS_NODE_STATUS_IDLE:
				job_list = db.get_all_unprocessed_jobs_for_pn_id(int(node[Constants.PROCESS_NODE_ID]))
				if len(job_list) < 1:
					job_list = db.get_all_unprocessed_jobs()
				for job in job_list:
					print datetime.now(), 'checking job', job
					if job[Constants.JOB_PROCESS_NODE_ID] < 0 or job[Constants.JOB_PROCESS_NODE_ID] == node[Constants.PROCESS_NODE_ID]:
						job[Constants.JOB_PROCESS_NODE_ID] = node[Constants.PROCESS_NODE_ID]
						url = 'http://' + str(node[Constants.PROCESS_NODE_HOSTNAME]) + ':' + str(node[Constants.PROCESS_NODE_PORT]) + '/job_queue'
						print datetime.now(), '_sending job to ', node[Constants.PROCESS_NODE_COMPUTERNAME], 'url', url
						s = requests.Session()
						r = s.post(url, data=json.dumps(job))
						print datetime.now(), 'result', r.status_code, ':', r.text
						break
			self.job_lock.release()
		except:
			self.job_lock.release()
			exc_str = traceback.format_exc()
			print datetime.now(), exc_str

	def _setup_logging_(self, log, logtype, logname):
		max_bytes = getattr(log, "rot_maxBytes", 20971520)  # 20Mb
		backup_count = getattr(log, "rot_backupCount", 10)
		fname = getattr(log, logtype, logname)
		h = logging.handlers.RotatingFileHandler(fname, 'a', max_bytes, backup_count)
		h.setLevel(logging.DEBUG)
		h.setFormatter(cherrypy._cplogging.logfmt)
		log.error_log.addHandler(h)

	def run(self):
		db.subscribe()
		db.create_tables()
		db.reset_process_nodes_status()
		webapp = SchedulerHandler(db, self.all_settings)
		webapp.process_node = SchedulerProcessNodeWebService(db)
		webapp.job = SchedulerJobsWebService(db)
		app = cherrypy.tree.mount(webapp, '/', self.conf)
		self._setup_logging_(app.log, "rot_error_file", "logs/scheduler_error.log")
		self._setup_logging_(app.log, "rot_access_file", "logs/scheduler_access.log")
		cherrypy.engine.start()
		cherrypy.engine.block()
		print datetime.now(), 'done blocking'
