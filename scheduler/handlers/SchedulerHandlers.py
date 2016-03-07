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


import json
import cherrypy
import glob
import os
import base64
import Settings
import unicodedata

def gen_job_dir_dict(text, opened, children):
	d_dict = dict()
	d_dict['text'] = text
	d_dict['state'] = dict()
	d_dict['state']['opened'] = opened
	d_dict['children'] = children
	return d_dict

def get_dirs(path, level):
	dir_list = [ {'id': os.path.join(path, name), 'parent':path, 'text': name} for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]
	if level > 0:
		level -= 1
		new_list = []
		for sub_path_dict in dir_list:
			new_list += get_dirs(sub_path_dict['id'], level) 
		dir_list += new_list
	return dir_list

class SchedulerHandler(object):

	def __init__(self, db, settings=None):
		self.db = db
		self.settings = settings
		self.api_dict = {'functions': [{'Function': 'api',
										'Parameters': 'N/A',
										'Description': 'Show the API help page'
										},
									   {
									    'Function': 'help',
										'Parameters': 'N/A',
										'Description': 'Show the API help page'
									   },
									   {
										'Function': 'get_output_list',
										'Parameters': 'job_path, process_type (PER_PIXEL or ROI)',
										'Description': 'Gets directory content for $job_path/output_old or $job_path/output.fit'
									   },
									   {
										'Function': 'get_dataset_dirs_list',
										'Parameters': 'job_path, depth',
										'Description': 'Get a list of jobs directories. Search recursively to depth.'
									   },
									   {
										'Function': 'get_spectrum_image_list',
										'Parameters': 'job_path',
										'Description': ''
									   },
									   {
										'Function': 'get_mda_list',
										'Parameters': 'job_path',
										'Description': 'Get a list of mda files in $jobs_path/mda'
									   },
									   {
										'Function': 'get_spectrum_image',
										'Parameters': 'path',
										'Description': 'Get the spectrum image file'
									   },
									   {
										'Function': 'get_spectrum_txt',
										'Parameters': 'path',
										'Description': 'Get the spectrum data txt file'
									   },
									   {
										'Function': 'get_all_unprocessed_jobs',
										'Parameters': 'N/A',
										'Description': 'Get all the queued jobs'
									   },
									   {
										'Function': 'get_all_processing_jobs',
										'Parameters': 'N/A',
										'Description': 'Get all the processing jobs'
									   },
									   {
										'Function': 'get_all_finished_jobs',
										'Parameters': 'N/A',
										'Description': 'Get all the finished jobs'
									   }
									   ]}

	def show_api(self):
		ret_str = '<!DOCTYPE html><html><head></head><body><table>'
		for entry in self.api_dict['functions']:
			ret_str += '<tr><td>Function: </td><td><b>' + entry['Function'] + '</b></td></tr>'
			ret_str += '<tr><td>Parameters: </td><td>' + entry['Parameters'] + '</td><tr>'
			ret_str += '<tr><td>Description: </td><td>' + entry['Description'] + '</td><tr>'
			ret_str += '<tr><td></td><td>----------------------------------------------------</td></tr>'
		ret_str += '</table></body></html>'
		return ret_str

	@cherrypy.expose
	def index(self):
		return file('public/scheduler_index.html')

	@cherrypy.expose
	def api(self):
		return self.show_api()

	@cherrypy.expose
	def help(self):
		return self.show_api()

	@cherrypy.expose
	def get_output_list(self, job_path=None, process_type=None):
		rfile = file('public/get_output_list.html')
		retstr = rfile.read()
		# default directory is output_old, but if it is matrix fit then use output.fit
		img_path = os.path.join(job_path, 'output_old/*.png')
		txt_path = os.path.join(job_path, 'output_old/*.txt')
		if process_type is not None:
			#print 'proc type = ',process_type
			if process_type == 'PER_PIXEL':
				img_path = os.path.join(job_path, 'output.fits/*.png')
				txt_path = os.path.join(job_path, 'output.fits/*.txt')
		retstr += '<ul>\n'
		for link in glob.glob(img_path):
			strLink = unicodedata.normalize('NFKD', link).encode('ascii', 'ignore')
			subname = strLink.split('/')
			name = subname[len(subname) -1]
			retstr += '<li><a href=/get_spectrum_image?path=' + strLink.replace('+', '%2b') + ' click=display_image link=' + strLink.replace('+', '%2b') + '>' + name + '</a></li>\n'
		retstr += '</ul>\n<ul>\n'
		for link in glob.glob(txt_path):
			strLink = unicodedata.normalize('NFKD', link).encode('ascii', 'ignore')
			subname = strLink.split('/')
			#print subname
			name = subname[len(subname) -1]
			retstr += '<li><a href=/get_spectrum_txt?path=' + strLink.replace('+', '%2b') + ' click=display_image link=' + strLink.replace('+', '%2b') + '>' + name + '</a></li>\n'
		retstr += '</ul>\n</body>\n</html>'
		return retstr

	@cherrypy.expose
	@cherrypy.tools.json_out()
	def get_dataset_dirs_list(self, job_path, depth=0):
		depth = int(depth)
		job_roots_dict = self.settings.getSetting(Settings.SECTION_JOB_DIR_ROOTS)
		path = job_roots_dict[job_path]
		dir_list = [{'id': path, 'parent': '#', 'text': job_path, 'state': {'opened': True}}]
		dir_list += get_dirs(path, depth)
		dd = dict()
		dd['core'] = dict()
		dd['core']['data'] = dir_list
		jenc = json.JSONEncoder()
		return jenc.encode(dd)

	@cherrypy.expose
	@cherrypy.tools.json_out()
	def get_spectrum_image_list(self, job_path):
		data_dict = dict()
		img_path = os.path.join(job_path, 'output_old/*.png')
		txt_path = os.path.join(job_path, 'output_old/*.txt')
		data_dict['images'] = glob.glob(img_path)
		data_dict['txt'] = glob.glob(txt_path)
		jenc = json.JSONEncoder()
		return jenc.encode(data_dict)

	@cherrypy.expose
	@cherrypy.tools.json_out()
	def get_mda_list(self, job_path):
		data_dict = dict()
		mda_path = os.path.join(job_path, 'mda/*.mda')
		data_dict['mda_files'] = glob.glob(mda_path)
		jenc = json.JSONEncoder()
		return jenc.encode(data_dict)

	def check_path(self, path):
		try:
			found = False
			job_roots_dict = self.settings.getSetting(Settings.SECTION_JOB_DIR_ROOTS)
			for job_path in job_roots_dict.values():
				#print job_path, path
				if path.startswith(job_path):
					found = True
					break
			return found
		except:
			return False

	@cherrypy.expose
	def get_spectrum_image(self, path):
		encoded_string = ''
		path = path.replace('..', '')
		if self.check_path(path) == True:
			with open(path, "rb") as image_file:
				encoded_string = base64.b64encode(image_file.read())
			retstr = '<img alt="My Image" src="data:image/png;base64,' + encoded_string + '" />'
			return retstr
		else:
			return "Error: file not file "+path

	@cherrypy.expose
	def get_spectrum_txt(self, path):
		path = path.replace('..', '')
		if self.check_path(path) == True:
			retstr = '<!DOCTYPE html><html><head></head><body><pre>'
			with open(path, "rt") as txt_file:
				retstr += txt_file.read()
			#retstr = '<img alt="My Image" src="data:image/png;base64,'+ encoded_string + '" />'
			retstr += '</pre></body></html>'
			return retstr
		else:
			return "Error: file not file " + path

	@cherrypy.expose
	def get_all_unprocessed_jobs(self, *args, **kwargs):
		data_dict = kwargs
		#data_dict['draw'] = 1
		data_dict['data'] = self.db.get_all_unprocessed_jobs()
		data_dict['recordsTotal'] = len(data_dict['data'])
		data_dict['recordsFiltered'] = len(data_dict['data'])
		#result = self.db.get_all_unprocessed_jobs()
		jenc = json.JSONEncoder()
		return jenc.encode(data_dict)

	@cherrypy.expose
	def get_all_processing_jobs(self, *args, **kwargs):
		data_dict = kwargs
		#data_dict['draw'] = 1
		data_dict['data'] = self.db.get_all_processing_jobs()
		data_dict['recordsTotal'] = len(data_dict['data'])
		data_dict['recordsFiltered'] = len(data_dict['data'])
		#result = self.db.get_all_processing_jobs()
		jenc = json.JSONEncoder()
		return jenc.encode(data_dict)

	@cherrypy.expose
	def get_all_finished_jobs(self, *args, **kwargs):
		data_dict = kwargs
		#data_dict['draw'] = 1
		data_dict['data'] = self.db.get_all_finished_jobs()
		data_dict['recordsTotal'] = len(data_dict['data'])
		data_dict['recordsFiltered'] = len(data_dict['data'])
		#result = self.db.get_all_finished_jobs()
		jenc = json.JSONEncoder()
		return jenc.encode(data_dict)

class SchedulerJobsWebService(object):
	'''
	Scheduler exposed /job
	class for adding, updating, or removing jobs
	'''
	exposed = True
	def __init__(self, db):
		self.db = db

	@cherrypy.tools.accept(media='text/plain')
	#@cherrypy.tools.json_out()
	# return list of jobs in queue
	def GET(self, *args, **kwargs):
		data_dict = kwargs
		#data_dict['draw'] = 1
		data_dict['data'] = self.db.get_all_jobs()
		data_dict['recordsTotal'] = len(data_dict['data'])
		data_dict['recordsFiltered'] = len(data_dict['data'])
		#result = self.db.get_all_jobs()
		jenc = json.JSONEncoder()
		return jenc.encode(data_dict)

	# submit job to queue
	def POST(self):
		cl = cherrypy.request.headers['Content-Length']
		rawbody = cherrypy.request.body.read(int(cl))
		job = json.loads(rawbody)
		if 'DataPath' in job:
			job['DataPath'] = job['DataPath'].replace('\\', '/')
		job['Id'] = self.db.insert_job(job)
		cherrypy.engine.publish("new_job", job)
		return 'inserted job Id:' + str(job['Id'])

	# change job properties (priority, ect...)
	def PUT(self, *args, **kwargs):
		cl = cherrypy.request.headers['Content-Length']
		rawbody = cherrypy.request.body.read(int(cl))
		job = json.loads(rawbody)
		self.db.update_job(job)
		cherrypy.engine.publish("update_job", job)
		return 'updated job Id:' + str(job['Id'])

	# delete job from queue
	def DELETE(self):
		cl = cherrypy.request.headers['Content-Length']
		rawbody = cherrypy.request.body.read(int(cl))
		job = json.loads(rawbody)
		cherrypy.engine.publish("delete_job", job)
		return 'Canceling job Id:' + str(job['Id'])


class SchedulerProcessNodeWebService(object):
	'''
	Scheduler exposed /process_node
	class for adding, updating, or removing process nodes
	'''
	exposed = True

	def __init__(self, db):
		self.db = db

	@cherrypy.tools.accept(media='text/plain')
	#@cherrypy.tools.json_out()
	# get list of computer nodes
	def GET(self, *args, **kwargs):
		data_dict = kwargs
		#data_dict['draw'] = 1
		#if computer_name == None:
		data_dict['data'] = self.db.get_all_process_nodes()
		#else:
		#data_dict['data'] = self.db.get_process_node(computer_name)
		data_dict['recordsTotal'] = len(data_dict['data'])
		data_dict['recordsFiltered'] = len(data_dict['data'])
		jenc = json.JSONEncoder()
		return jenc.encode(data_dict)

	# add process node
	def POST(self):
		cl = cherrypy.request.headers['Content-Length']
		rawbody = cherrypy.request.body.read(int(cl))
		proc_node = json.loads(rawbody)
		self.db.insert_process_node(proc_node)
		cherrypy.engine.publish('process_node_update', proc_node)
		return 'inserted process node'

	# change process node status
	def PUT(self):
		cl = cherrypy.request.headers['Content-Length']
		rawbody = cherrypy.request.body.read(int(cl))
		proc_node = json.loads(rawbody)
		#print proc_node
		self.db.insert_process_node(proc_node)
		cherrypy.engine.publish('process_node_update', proc_node)
		return 'updated process node'

	# computer node went offline, remove from list
	def DELETE(self):
		#cherrypy.session.pop('mystring', None)
		pass
