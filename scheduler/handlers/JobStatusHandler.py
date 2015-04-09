import random
import string
import json
import cherrypy

class JobStatusHandler(object):
	
	@cherrypy.expose
	def index(self):
		return file('public/index.html')

class JobsWebService(object):
	exposed = True

	@cherrypy.tools.accept(media='text/plain')
	#return list of jobs in queue
	def GET(self):
		#return cherrypy.session['mystring']
		pass

	#submit job to queue
	def POST(self, job_name):
		some_string = ''.join(random.sample(string.hexdigits, int(length)))
		#cherrypy.session['mystring'] = some_string
		return some_string

	#change job properties (priority, ect...)
	def PUT(self, another_string):
		#cherrypy.session['mystring'] = another_string
		pass

	#delete job from queue
	def DELETE(self):
		#cherrypy.session.pop('mystring', None)
		pass

