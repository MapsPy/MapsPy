import string
import json
import cherrypy

class ProcessNodeHandler(object):
	
	@cherrypy.expose
	def index(self):
		return file('public/process_node_index.html')

class ProcessNodeJobsWebService(object):
	'''
	ProcessNode exposed /job_queue
	'''
	exposed = True
	def __init__(self, db):
		self.db = db

	@cherrypy.tools.accept(media='text/plain')
	@cherrypy.tools.json_out()
	#get list of jobs on this nodes queue
	def GET(self, job_id=None):
		result = None
		if computer_name == None:
			result = self.db.get_all_jobs()
		else:
			result = self.db.get_job(job_id)
		jenc = json.JSONEncoder()
		return jenc.encode(result)

	#submit a job
	def POST(self):
		cl = cherrypy.request.headers['Content-Length']
		rawbody = cherrypy.request.body.read(int(cl))
		job = json.loads(rawbody)
		self.db.insert_job(job)
		cherrypy.engine.publish("new_job", job)
		return 'inserted job'

	#update job
	def PUT(self):
		return 'updated process node'

	#remove a job from the queue
	def DELETE(self):
		#cherrypy.session.pop('mystring', None)
		pass

