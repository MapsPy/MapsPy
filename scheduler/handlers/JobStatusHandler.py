import string
import json
import cherrypy

class JobStatusHandler(object):
	
	@cherrypy.expose
	def index(self):
		return file('public/index.html')

class JobsWebService(object):
	exposed = True
	def __init__(self, db):
		self.db = db

	@cherrypy.tools.accept(media='text/plain')
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

