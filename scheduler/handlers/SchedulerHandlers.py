import string
import json
import cherrypy

class SchedulerHandler(object):
	
	@cherrypy.expose
	def index(self):
		return file('public/scheduler_index.html')

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
	def GET(self, job_id=None):
		result = None
		if job_id == None:
			result = self.db.get_all_jobs()
		else:
			result = self.db.get_job(job_id)
		jenc = json.JSONEncoder()
		return jenc.encode(result)

	#submit job to queue
	def POST(self):
		cl = cherrypy.request.headers['Content-Length']
		rawbody = cherrypy.request.body.read(int(cl))
		job = json.loads(rawbody)
		self.db.insert_job(job)
		return 'inserted job'

	#change job properties (priority, ect...)
	def PUT(self):
		#cherrypy.session['mystring'] = another_string
		pass

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
		return 'updated process node'

	#computer node went offline, remove from list
	def DELETE(self):
		#cherrypy.session.pop('mystring', None)
		pass


