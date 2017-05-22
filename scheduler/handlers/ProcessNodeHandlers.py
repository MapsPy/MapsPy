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

import string
import json
import cherrypy
import traceback
import os
import re

#todo: this is redefined in ProcessNode.py 
STR_JOB_LOG_DIR_NAME = 'job_logs'


class ProcessNodeHandler(object):

	def __init__(self):
		self.software_ver_dic = {}

	def append_software_dir(self, software_name, software_path):
		self.software_ver_dic[software_name] = software_path

	@cherrypy.expose
	def index(self):
		return file('public/process_node_index.html')

	@cherrypy.expose
	def get_job_log(self, log_path):
		try:
			full_log_path = os.path.join(STR_JOB_LOG_DIR_NAME, log_path)
			full_log_path = full_log_path.replace('..', '')
			retstr = '<!DOCTYPE html><html><head></head><body><pre>'
			with open(full_log_path, "rt") as txt_file:
				retstr += txt_file.read()
			retstr += '</pre></body></html>'
			return retstr
		except:
			exc_str = traceback.format_exc()
			return exc_str

	@cherrypy.expose
	def version(self, software):
		if self.software_ver_dic.has_key(software):
			ver = [re.findall(r'<b>Revision<\/b>:\s*([^\n\r]*)',line) for line in open(self.software_ver_dic[software])]
			# remove empty's
			ver = [x for x in ver if x]
			if len(ver) > 0:
				#print 'ver ', ver
				return ver[0]
				#ret_str = '<!DOCTYPE html><html><head></head><body>' + str(ver[0]) + '</body></html>'
				#return ret_str
			else:
				return file(self.software_ver_dic[software])
		else:
			return 'Unknown software: ' + software

	@cherrypy.expose
	def version_file(self, software):
		if self.software_ver_dic.has_key(software):
			return file(self.software_ver_dic[software])
		else:
			return 'Unknown software: ' + software

	@cherrypy.expose
	def update_id(self, Id):
		cherrypy.engine.publish('update_id', Id)
		return 'Updated'


class ProcessNodeJobsWebService(object):
	'''
	ProcessNode exposed /job_queue
	'''
	exposed = True
	def __init__(self, db, logger):
		self.db = db
		self.logger = logger

	@cherrypy.tools.accept(media='text/plain')
	@cherrypy.tools.json_out()
	# get list of jobs on this nodes queue
	def GET(self, job_id=None):
		result = None
		if job_id == None:
			result = self.db.get_all_jobs()
		else:
			result = self.db.get_job(job_id)
		jenc = json.JSONEncoder()
		return jenc.encode(result)

	# submit a job
	def POST(self):
		cl = cherrypy.request.headers['Content-Length']
		rawbody = cherrypy.request.body.read(int(cl))
		job = json.loads(rawbody)
		if job != None:
			try:
				self.db.insert_job_with_id(job)
			except:
				myJob = self.db.get_job(job['Id'])
				if not myJob == None:
					if int(myJob['Status']) > int(job['Status']):
						self.logger.info('sending updated status for job: %s', myJob)
						cherrypy.engine.publish('send_job_update', myJob)
					else:
						self.logger.info('updating job: %s', job)
						self.db.update_job(job)
				else:
					self.logger.info('-updating job: %s', job)
					self.db.update_job(job)
			cherrypy.engine.publish("new_job", job)
			return 'inserted job'
		else:
			self.logger.error('Error: could not parse json job')
			return 'Error: could not parse json job'

	# update job
	def PUT(self):
		return 'updated process node'

	# remove a job from the queue
	def DELETE(self):
		cl = cherrypy.request.headers['Content-Length']
		rawbody = cherrypy.request.body.read(int(cl))
		job = json.loads(rawbody)
		if job != None:
			self.logger.info('updating job: %s', job)
			self.db.update_job(job)
			cherrypy.engine.publish('delete_job', job)
		else:
			return 'Empty Job dictionary'
		return 'done'
