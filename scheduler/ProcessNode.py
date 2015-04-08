
import cherrypy

class ProcessNode(object):
	def __init__(self, settings):
		self.settings = settings
	def run(self):
		print 'nothing to do, exiting'
