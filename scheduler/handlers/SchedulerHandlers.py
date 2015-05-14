import string
import json
import cherrypy
import glob
import os
import traceback
import base64

class SchedulerHandler(object):

	def __init__(self, db):
		self.db = db

	@cherrypy.expose
	def index(self):
		return file('public/scheduler_index.html')

	@cherrypy.expose
	def jstree(self):
		return file('public/jstree_test.html')

	@cherrypy.expose
	@cherrypy.tools.json_out()
	def get_spectrum_image_list(self, job_path):
		try:
			img_path = os.path.join(job_path, 'output_old/*.png')
			print img_path
			img_list = glob.glob(img_path)
			jenc = json.JSONEncoder()
			return jenc.encode(img_list)
		except:
			exc_str = traceback.format_exc()
			return exc_str

	@cherrypy.expose
	def get_spectrum_image(self, image_path):
		try:
			encoded_string = ''
			with open(image_path, "rb") as image_file:
				encoded_string = base64.b64encode(image_file.read())
			str = '<img alt="My Image" src="data:image/png;base64,'+ encoded_string + '" />'
			return str
			#return file(image_path)
		except:
			exc_str = traceback.format_exc()
			return exc_str

	@cherrypy.expose
	def get_all_unprocessed_jobs(self):
		result = self.db.get_all_unprocessed_jobs()
		jenc = json.JSONEncoder()
		return jenc.encode(result)

	@cherrypy.expose
	def get_all_processing_jobs(self):
		result = self.db.get_all_processing_jobs()
		jenc = json.JSONEncoder()
		return jenc.encode(result)

	@cherrypy.expose
	def get_all_finished_jobs(self):
		result = self.db.get_all_finished_jobs()
		jenc = json.JSONEncoder()
		return jenc.encode(result)

class SchedulerJobsWebService(object):
	'''
	Scheduler exposed /job
	class for adding, updating, or removing jobs
	'''
	exposed = True
	def __init__(self, db):
		self.db = db

	@cherrypy.tools.accept(media='text/plain')
	@cherrypy.tools.json_out()
	#return list of jobs in queue
	def GET(self):
		result = self.db.get_all_jobs()
		jenc = json.JSONEncoder()
		return jenc.encode(result)

	#submit job to queue
	def POST(self):
		cl = cherrypy.request.headers['Content-Length']
		rawbody = cherrypy.request.body.read(int(cl))
		job = json.loads(rawbody)
		job['Id'] = self.db.insert_job(job)
		cherrypy.engine.publish("new_job", job)
		return 'inserted job Id:'+str(job['Id'])

	#change job properties (priority, ect...)
	def PUT(self):
		cl = cherrypy.request.headers['Content-Length']
		rawbody = cherrypy.request.body.read(int(cl))
		job = json.loads(rawbody)
		self.db.update_job(job)
		#cherrypy.engine.publish("new_job", job)
		return 'updated job Id:'+str(job['Id'])

	#delete job from queue
	def DELETE(self):
		#cherrypy.session.pop('mystring', None)
		pass


class SchedulerProcessNodeWebService(object):
	'''
	Scheduler exposed /process_node
	class for adding, updating, or removing process nodes
	'''
	exposed = True
	def __init__(self, db):
		self.db = db

	@cherrypy.tools.accept(media='text/plain')
	@cherrypy.tools.json_out()
	#get list of computer nodes
	def GET(self, computer_name=None):
		result = None
		if computer_name == None:
			result = self.db.get_all_process_nodes()
		else:
			result = self.db.get_process_node(computer_name)
		jenc = json.JSONEncoder()
		return jenc.encode(result)

	#add process node
	def POST(self):
		cl = cherrypy.request.headers['Content-Length']
		rawbody = cherrypy.request.body.read(int(cl))
		proc_node = json.loads(rawbody)
		self.db.insert_process_node(proc_node)
		return 'inserted process node'

	#change process node status
	def PUT(self):
		cl = cherrypy.request.headers['Content-Length']
		rawbody = cherrypy.request.body.read(int(cl))
		proc_node = json.loads(rawbody)
		print proc_node
		self.db.insert_process_node(proc_node)
		cherrypy.engine.publish('process_node_update', proc_node)
		return 'updated process node'

	#computer node went offline, remove from list
	def DELETE(self):
		#cherrypy.session.pop('mystring', None)
		pass


