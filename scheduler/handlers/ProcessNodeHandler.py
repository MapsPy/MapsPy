import random
import string
import json
import cherrypy

class ProcessNodeWebService(object):
	exposed = True
	def __init__(self, db):
		self.db = db

	@cherrypy.tools.accept(media='text/plain')
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
	def POST(self, ComputerName, NumThreads, Status, Heartbeat):
		proc_node = {'ComputerName':ComputerName,'NumThreads':NumThreads,'Status':Status,'Heartbeat':Heartbeat}
		self.db.insert_process_node(proc_node)
		return 'inserted'

	#change process node status
	def PUT(self, ComputerName, NumThreads, Status, Heartbeat):
		proc_node = {'ComputerName':ComputerName,'NumThreads':NumThreads,'Status':Status,'Heartbeat':Heartbeat}
		self.db.insert_process_node(proc_node)
		return 'updated'

	#computer node went offline, remove from list
	def DELETE(self):
		#cherrypy.session.pop('mystring', None)
		pass

