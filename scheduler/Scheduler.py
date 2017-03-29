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
from RestBase import RestBase
import threading
from datetime import datetime
import Constants
import h5py
import StringIO
import scipy.misc
import glob
import time
import shutil

db = DatabasePlugin(cherrypy.engine, SQLiteDB)


class Scheduler(RestBase):
	def __init__(self, settings):
		RestBase.__init__(self)
		self.all_settings = settings
		self.settings = settings.getSetting(Settings.SECTION_SERVER)
		self.job_lock = threading.RLock()
		self.pn_lock = threading.RLock()
		cherrypy.config.update({
			'server.socket_host': self.settings[Settings.SERVER_HOSTNAME],
			'server.socket_port': int(self.settings[Settings.SERVER_PORT]),
			'server.thread_pool': 60,
			'log.error_file': "logs/scheduler_error.log"
		})
		self.mailman = Mailman.mailman(self.settings[Settings.SERVER_SMTP_ADDRESS],
								self.settings[Settings.SERVER_FROM_ADDRESS],
								self.settings[Settings.SERVER_MAIL_USERNAME],
								self.settings[Settings.SERVER_MAIL_PASSWORD])
		# this flag is for checking old idl process node status stored in files
		self.schedule_files = str(self.settings[Settings.SCHEDULE_FILES_PATH]).strip(' ')
		# dictionary of idl process node status
		self.idl_process_node_statuses = {}
		# last time we checked the files status directory
		self.idl_last_time_check = 0
		# check every 10 seconds
		self.idl_check_time = 10.0
		self.logger = logging.getLogger(__name__)
		self._setup_logging_(self.logger, "rot_file", "logs/MapsPy.log", True)
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
				'tools.staticdir.dir': './public',
				'tools.sessions.on': False,
        		'tools.caching.on': False,
        		'tools.caching.force' : False,
        		'tools.caching.delay' : 0,
        		'tools.expires.on' : False,
        		'tools.expires.secs' : 60*24*365
			}
		}

	def send_job_to_process_node(self, job, p_node):
		url = 'http://' + str(p_node[Constants.PROCESS_NODE_HOSTNAME]) + ':' + str(p_node[Constants.PROCESS_NODE_PORT]) + '/job_queue'
		self.logger.info('sending job to %s, url: %s', p_node[Constants.PROCESS_NODE_COMPUTERNAME], url)
		s = requests.Session()
		r = s.post(url, data=json.dumps(job))
		self.logger.info('result %s , %s', r.status_code, r.text)
		return r

	def callback_new_job(self, job):
		self.logger.info('callback got new job %s', job)
		try:
			self.job_lock.acquire(True)
			p_node = None
			if job[Constants.JOB_PROCESS_NODE_ID] > -1:
				p_node = db.get_process_node_by_id(int(job[Constants.JOB_PROCESS_NODE_ID]))
				self.send_job_to_process_node(job, p_node)
			else:
				node_list = db.get_all_process_nodes()
				self.logger.info('searching for idle node')
				for node in node_list:
					if node[Constants.PROCESS_NODE_STATUS] == Constants.PROCESS_NODE_STATUS_IDLE:
						p_node = node
						if p_node != None:
							job[Constants.JOB_PROCESS_NODE_ID] = p_node[Constants.PROCESS_NODE_ID]
							r = self.send_job_to_process_node(job, p_node)
							if r.status_code == 200:
								db.update_job_pn(job[Constants.JOB_ID], p_node[Constants.PROCESS_NODE_ID])
								break
							else:
								job[Constants.JOB_PROCESS_NODE_ID] = -1
			self.job_lock.release()
		except:
			self.job_lock.release()
			exc_str = traceback.format_exc()
			self.logger.error(exc_str)
			return exc_str

	def callback_update_job(self, job):
		if Constants.JOB_STATUS in job and Constants.JOB_EMAILS in job:
			if job[Constants.JOB_STATUS] > Constants.JOB_STATUS_PROCESSING and len(job[Constants.JOB_EMAILS]) > 0:
				self.logger.info('sending completed emails')
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
					self.logger.exception("Error")

	def callback_delete_job(self, job):
		try:
			status = job[Constants.JOB_STATUS]
			if status == Constants.JOB_STATUS_COMPLETED or status == Constants.JOB_STATUS_CANCELED or status == Constants.JOB_STATUS_GENERAL_ERROR:
				db.delete_job_by_id(job[Constants.JOB_ID])
				if status == Constants.JOB_STATUS_COMPLETED or status == Constants.JOB_STATUS_GENERAL_ERROR and job[Constants.JOB_PROCESS_NODE_ID] > -1:
					self.call_delete_job(job)
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
					self.logger.warning('Warning: callback_delete_job - No Process Node Id to send cancel to.')
		except:
			self.job_lock.release()
			self.logger.exception("Error")

	def callback_process_node_update(self, node):
		self.pn_lock.acquire(True)
		# if we should check idl process nodes status
		if len(self.schedule_files) > 0:
			curtime = time.time()
			if (curtime - self.idl_last_time_check) > self.idl_check_time:
				self.idl_last_time_check = time.time()
				self._read_idl_process_node_status()
		#self.logger.info('callback %s', node[Constants.PROCESS_NODE_COMPUTERNAME])
		self.pn_lock.release()
		try:
			if not Constants.PROCESS_NODE_ID in node:
				self.logger.info('getting id for node %s', node)
				new_node = db.get_process_node_by_name(node[Constants.PROCESS_NODE_COMPUTERNAME])
				node[Constants.PROCESS_NODE_ID] = new_node[Constants.PROCESS_NODE_ID]
				self.logger.info('updated node %s', node)
				s = requests.Session()
				url = 'http://' + str(node[Constants.PROCESS_NODE_HOSTNAME]) + ':' + str(node[Constants.PROCESS_NODE_PORT]) + '/update_id'
				r = s.post(url, data={Constants.PROCESS_NODE_ID: node[Constants.PROCESS_NODE_ID]})
				self.logger.info('update result: %s, %s', r.status_code, r.text)
			# This is a hack that should be removed in the future
			if node[Constants.PROCESS_NODE_COMPUTERNAME] in self.idl_process_node_statuses:
				if self.idl_process_node_statuses[node[Constants.PROCESS_NODE_COMPUTERNAME]] == Constants.PROCESS_NODE_STATUS_PROCESSING_IDL:
					node[Constants.PROCESS_NODE_STATUS] = Constants.PROCESS_NODE_STATUS_PROCESSING_IDL
			if node[Constants.PROCESS_NODE_STATUS] == Constants.PROCESS_NODE_STATUS_IDLE:
				self.job_lock.acquire(True)
				job_list = db.get_all_unprocessed_jobs_for_pn_id(int(node[Constants.PROCESS_NODE_ID]))
				if len(job_list) < 1:
					job_list = db.get_all_unprocessed_jobs_for_any_node()
				if len(job_list) > 0:
					job = job_list[0]
					job[Constants.JOB_PROCESS_NODE_ID] = node[Constants.PROCESS_NODE_ID]
					r = self.send_job_to_process_node(job, node)
					if r.status_code == 200:
						db.update_job_pn(job[Constants.JOB_ID], node[Constants.PROCESS_NODE_ID])
				self.job_lock.release()
			db.insert_process_node(node)
		except:
			self.logger.exception('Error')
			self.job_lock.release()

	def call_delete_job(self, job):
		p_node = db.get_process_node_by_id(int(job[Constants.JOB_PROCESS_NODE_ID]))
		url = 'http://' + str(p_node[Constants.PROCESS_NODE_HOSTNAME]) + ':' + str(p_node[Constants.PROCESS_NODE_PORT]) + '/job_queue'
		self.logger.info('sending job to %s, url:%s', p_node[Constants.PROCESS_NODE_COMPUTERNAME], url)
		s = requests.Session()
		r = s.delete(url, data=json.dumps(job))
		self.logger.info('update result %s, %s', r.status_code, r.text)
		if r.status_code == 200:
			db.delete_job_by_id(job[Constants.JOB_ID])

	# This is a hack, should be removed in the future!
	# we are trying to keep track to two states for one node.
	def _read_idl_process_node_status(self):
		pn_list = db.get_all_process_nodes()
		jobs_list = []
		idle_process_nodes = {}
		for file_name in glob.glob(self.schedule_files + '*.txt'):
			split_str = os.path.basename(file_name).split('_')
			if len(split_str) > 2:
				# check if process node status
				if split_str[0] == 'status':
					if split_str[2] == 'idle.txt':
						self.idl_process_node_statuses[split_str[1]] = Constants.PROCESS_NODE_STATUS_IDLE
					elif split_str[2] == 'working.txt':
						self.idl_process_node_statuses[split_str[1]] = Constants.PROCESS_NODE_STATUS_PROCESSING_IDL
					# try to find the process node
					process_node = None
					for pn in pn_list:
						if pn[Constants.PROCESS_NODE_COMPUTERNAME] == split_str[1]:
							process_node = pn
							break
					# check if both are idle
					status = 0
					if process_node != None:
						if process_node[Constants.PROCESS_NODE_STATUS] == Constants.PROCESS_NODE_STATUS_IDLE:
							status = 1
					# if it is None we set it to idle to process on idl
					else:
						status = 1
					if status and split_str[2] == 'idle.txt':
						idle_process_nodes[ split_str[1] ] = file_name
			if len(split_str) > 1:
				# check if process node status
				if split_str[0] == 'job':
					jobs_list += [file_name]
		# if we have idle idl process nodes, submit a job to them
		for file_name in jobs_list:
			if len(idle_process_nodes) > 0:
				file_name = os.path.basename(file_name)
				key, value = idle_process_nodes.popitem()
				try:
					shutil.move(self.schedule_files + file_name, self.schedule_files + key + '/' + file_name)
					os.remove(value)
					process_node = None
					for pn in pn_list:
						if pn[Constants.PROCESS_NODE_COMPUTERNAME] == key:
							process_node = pn
							break
					if process_node != None:
						process_node[Constants.PROCESS_NODE_STATUS] = Constants.PROCESS_NODE_STATUS_PROCESSING_IDL
						self.idl_process_node_statuses[key] = Constants.PROCESS_NODE_STATUS_PROCESSING_IDL
						db.insert_process_node(process_node)
				except:
					self.logger.exception("Error moving file " + value)
			else:
				break

	def _get_images_from_hdf(self, job):
		images_dict = None
		try:
			# create image dictionary
			images_dict = {}
			full_file_name = ''
			# check how many datasets are in job
			file_name = ''
			file_dir = os.path.join(job[Constants.JOB_DATA_PATH], Constants.DIR_IMG_DAT)
			proc_mask = job[Constants.JOB_PROC_MASK]
			# will only check one file for images
			if job[Constants.JOB_DATASET_FILES_TO_PROC] == 'all':
				self.logger.warning('Warning: Too many datasets to parse images from')
				return None
			else:
				temp_names = job[Constants.JOB_DATASET_FILES_TO_PROC].split(',')
				if len(temp_names) > 1:
					self.logger.warning('Warning: Can only parse one dataset for images, dataset list is %s', job[Constants.JOB_DATASET_FILES_TO_PROC])
					return None
				temp_name = job[Constants.JOB_DATASET_FILES_TO_PROC]
				if job[Constants.JOB_XANES_SCAN] == 1:
					if proc_mask & 64 == 64: #generate avg
						full_file_name = os.path.join(file_dir, temp_name + '.h5')
					else:
						full_file_name = os.path.join(file_dir, temp_name + '.h5' + str(job[Constants.JOB_DETECTOR_TO_START_WITH]))
				else:
					hdf_file_name = temp_name.replace('.mda', '.h5')
					full_file_name = os.path.join(file_dir, hdf_file_name)

			hdf_file = h5py.File(full_file_name, 'r')
			maps_group = hdf_file[Constants.HDF5_GRP_MAPS]
			if job[Constants.JOB_XANES_SCAN] == 1:
				h5_grp = None
				analyzed_grp = maps_group[Constants.HDF5_GRP_ANALYZED]
				if analyzed_grp == None:
					self.logger.warning('Warning: %s did not find '+Constants.HDF5_GRP_ANALYZED, file_name)
					return None
				if job[Constants.JOB_NNLS] == 1:
					h5_grp = analyzed_grp[Constants.HDF5_GRP_NNLS]
				elif proc_mask & 4 == 4:
					h5_grp = analyzed_grp[Constants.HDF5_GRP_FITS]
				elif proc_mask & 1 == 1:
					h5_grp = analyzed_grp[Constants.HDF5_GRP_ROI]
				else:
					self.logger.warning('Warning: %s did not process XRF_ROI or XRF_FITS', file_name)
					return None
				if not h5_grp == None:
					xrf_roi_dataset = h5_grp[Constants.HDF5_DSET_COUNTS]
					channel_names = h5_grp[Constants.HDF5_DSET_CHANNELS]
				else:
					return None
			else:
				if proc_mask & 1 == 1:
					xrf_roi_dataset = maps_group[Constants.HDF5_DSET_XRF_ROI]
				elif proc_mask & 4 == 4:
					xrf_roi_dataset = maps_group[Constants.HDF5_DSET_XRF_FITS]
				else:
					self.logger.warning('Warning: %s did not process XRF_ROI or XRF_FITS', file_name)
					return None
				channel_names = maps_group[Constants.HDF5_GRP_CHANNEL_NAMES]

			if channel_names.shape[0] != xrf_roi_dataset.shape[0]:
				self.logger.warning('Warning: file %s : Datasets: %s [%s] and %s [%s] length missmatch', file_name, Constants.HDF5_DSET_XRF_ROI, xrf_roi_dataset.shape[0], Constants.HDF5_GRP_CHANNEL_NAMES, channel_names.shape[0])
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

	def run(self):
		db.subscribe()
		db.create_tables()
		db.reset_process_nodes_status()
		webapp = SchedulerHandler(db, self.all_settings)
		webapp.process_node = SchedulerProcessNodeWebService(db)
		webapp.job = SchedulerJobsWebService(db)
		app = cherrypy.tree.mount(webapp, '/', self.conf)
		#self._setup_logging_(app.log, "rot_error_file", "logs/scheduler_error.log")
		#self._setup_logging_(app.log, "rot_access_file", "logs/scheduler_access.log")
		cherrypy.engine.start()
		cherrypy.engine.block()
		self.logger.info(datetime.now(), 'done blocking')

	def stop(self):
		pass