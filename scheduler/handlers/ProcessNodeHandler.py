import string
import json
import cherrypy

class ProcessNodeWebService(object):
	exposed = True
	def __init__(self, db):
		self.db = db

	@cherrypy.tools.accept(media='text/plain')
	#@cherrypy.tools.json_out()
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

