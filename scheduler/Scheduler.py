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
from datetime import datetime
import Constants
import h5py
import StringIO
import scipy.misc

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
		self.mailman = Mailman.mailman(self.settings[Settings.SERVER_SMTP_ADDRESS],
								self.settings[Settings.SERVER_FROM_ADDRESS],
								self.settings[Settings.SERVER_MAIL_USERNAME],
								self.settings[Settings.SERVER_MAIL_PASSWORD])
		cherrypy.engine.subscribe("new_job", self.callback_new_job)
		cherrypy.engine.subscribe("update_job", self.callback_update_job)
		cherrypy.engine.subscribe("delete_job", self.callback_delete_job)
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
				'request.methods_with_bodies': ('POST', 'PUT', 'DELETE')
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
		if Constants.JOB_STATUS in job and Constants.JOB_EMAILS in job:
			if job[Constants.JOB_STATUS] > Constants.JOB_STATUS_PROCESSING and len(job[Constants.JOB_EMAILS]) > 0:
				print datetime.now(), 'sending completed emails'
				if job[Constants.JOB_STATUS] == Constants.JOB_STATUS_COMPLETED:
					subject = Constants.EMAIL_SUBJECT_COMPLETED
					mesg = Constants.EMAIL_MESSAGE_COMPLETED
				else:
					subject = Constants.EMAIL_SUBJECT_ERROR
					mesg = Constants.EMAIL_MESSAGE_ERROR
				image_dict = self._get_images_from_hdf(job)
				for key in job.iterkeys():
					mesg += key + ': ' + str(job[key]) + '\n'
				try:
					self.mailman.send(job[Constants.JOB_EMAILS], subject, mesg, image_dict)
				except:
					exc_str = traceback.format_exc()
					print datetime.now(), exc_str

	def callback_delete_job(self, job):
		try:
			status = job[Constants.JOB_STATUS]
			if status == Constants.JOB_STATUS_COMPLETED or status == Constants.JOB_STATUS_CANCELED or status == Constants.JOB_STATUS_GENERAL_ERROR:
				db.delete_job_by_id(job[Constants.JOB_ID])
				if status == Constants.JOB_STATUS_COMPLETED or status == Constants.JOB_STATUS_GENERAL_ERROR and job[Constants.JOB_PROCESS_NODE_ID] > -1:
					self.call_delete_job(job)
				return
			elif status == Constants.JOB_STATUS_PROCESSING or status == Constants.JOB_STATUS_NEW or status == Constants.JOB_STATUS_CANCELING:
				self.job_lock.acquire(True)
				if job[Constants.JOB_PROCESS_NODE_ID] > -1:
					job[Constants.JOB_STATUS] = Constants.JOB_STATUS_CANCELING
				else:
					job[Constants.JOB_STATUS] = Constants.JOB_STATUS_CANCELED
				db.update_job(job)
				self.job_lock.release()
				if job[Constants.JOB_PROCESS_NODE_ID] > -1:
					self.call_delete_job(job)
				else:
					print 'Warning: callback_delete_job - No Process Node Id to send cancel to.'
		except:
			self.job_lock.release()
			exc_str = traceback.format_exc()
			print datetime.now(), exc_str

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
					job_list = db.get_all_unprocessed_jobs_for_any_node()
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

	def call_delete_job(self, job):
		p_node = db.get_process_node_by_id(int(job[Constants.JOB_PROCESS_NODE_ID]))
		url = 'http://' + str(p_node[Constants.PROCESS_NODE_HOSTNAME]) + ':' + str(p_node[Constants.PROCESS_NODE_PORT]) + '/job_queue'
		print datetime.now(), 'sending job to ', p_node[Constants.PROCESS_NODE_COMPUTERNAME], 'url', url
		s = requests.Session()
		r = s.delete(url, data=json.dumps(job))
		print datetime.now(), 'update result', r.status_code, ':', r.text

	def _get_images_from_hdf(self, job):
		images_dict = None
		try:
			# create image dictionary
			images_dict = {}
			# check how many datasets are in job
			file_name = ''
			file_dir = os.path.join(job[Constants.JOB_DATA_PATH], Constants.DIR_IMG_DAT)
			# will only check one file for images
			if job[Constants.JOB_DATASET_FILES_TO_PROC] == 'all':
				print 'Warning: Too many datasets to parse images from'
				return None
			else:
				temp_names = job[Constants.JOB_DATASET_FILES_TO_PROC].split(',')
				if len(temp_names) > 1:
					print 'Warning: Can only parse one dataset for images, dataset list is', job[Constants.JOB_DATASET_FILES_TO_PROC]
					return None
				temp_name = job[Constants.JOB_DATASET_FILES_TO_PROC]
				hdf_file_name = temp_name.replace('.mda', '.h5')
				full_file_name = os.path.join(file_dir, hdf_file_name)

			hdf_file = h5py.File(full_file_name, 'r')
			maps_group = hdf_file[Constants.HDF5_GRP_MAPS]
			proc_mask = job[Constants.JOB_PROC_MASK]
			if proc_mask & 1 == 1:
				xrf_roi_dataset = maps_group[Constants.HDF5_GRP_XRF_ROI]
			elif proc_mask & 4 == 4:
				xrf_roi_dataset = maps_group[Constants.HDF5_GRP_XRF_FITS]
			else:
				print 'Warning: ', file_name, ' did not process XRF_ROI or XRF_FITS'
				return None

			channel_names = maps_group[Constants.HDF5_GRP_CHANNEL_NAMES]
			if channel_names.shape[0] != xrf_roi_dataset.shape[0]:
				print 'Warning: ', file_name, ' Datasets', Constants.HDF5_GRP_XRF_ROI, '[', xrf_roi_dataset.shape[0], '] and ', Constants.HDF5_GRP_CHANNEL_NAMES, '[' , channel_names.shape[0], '] length missmatch'
				return None

			for i in range(channel_names.size):
				outbuf = StringIO.StringIO()
				img = scipy.misc.toimage(xrf_roi_dataset[i], mode='L')
				img.save(outbuf, format='PNG')
				name = 'channel_' + channel_names[i] + '.png'
				images_dict[name] = outbuf.getvalue()
		except:
			images_dict = None
		return images_dict

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
