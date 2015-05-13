import cherrypy
from cherrypy.process import wspbus, plugins

CHANNEL_DB_INSERT_PROCESS_NODE = 'ch_insert_process_node'

class DatabasePlugin(plugins.SimplePlugin):
	def __init__(self, bus, db_klass, db_name='MapsPy.db'):
		plugins.SimplePlugin.__init__(self, bus)
		self.db = db_klass(db_name)

	def start(self):
		self.bus.log('Starting up DB access')
		self.bus.subscribe("db-save", self.save_it)
		self.bus.subscribe(CHANNEL_DB_INSERT_PROCESS_NODE, self.insert_process_node)

	def stop(self):
		self.bus.log('Stopping down DB access')
		self.bus.unsubscribe("db-save", self.save_it)

	def create_tables(self):
		self.db.create_tables()

	def insert_process_node(self, entity):
		self.db.insert_process_node(entity)

	def insert_job(self, job_dict):
		return self.db.insert_job(job_dict)

	def insert_job_with_id(self, job_dict):
		self.db.insert_job_with_id(job_dict)

	def get_all_process_nodes(self):
		return self.db.get_all_process_nodes()

	def get_process_node(self, node_name):
		return self.db.get_process_node(node_name)

	def get_all_jobs(self):
		return self.db.get_all_jobs()

	def get_all_unprocessed_jobs(self):
		return self.db.get_all_unprocessed_jobs()

	def get_all_processing_jobs(self):
		return self.db.get_all_processing_jobs()

	def get_all_finished_jobs(self):
		return self.db.get_all_finished_jobs()

	def get_job(self, job_id):
		return self.db.get_job(job_id)

	def save_it(self, entity):
		self.db.save(entity)

	def update_job(self, job_dict):
		self.db.update_job(job_dict)
