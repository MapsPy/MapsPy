'''
Created on May 2015

@author: Arthur Glowacki, Argonne National Laboratory

Copyright (c) 2015, Stefan Vogt, Argonne National Laboratory 
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

	def delete_job_by_id(self, job_id):
		return self.db.delete_job_by_id(job_id)

	def insert_process_node(self, entity):
		self.db.insert_process_node(entity)

	def insert_job(self, job_dict):
		return self.db.insert_job(job_dict)

	def insert_job_with_id(self, job_dict):
		self.db.insert_job_with_id(job_dict)

	def get_all_process_nodes(self):
		return self.db.get_all_process_nodes()

	def get_process_node_by_name(self, node_name):
		return self.db.get_process_node_by_name(node_name)

	def get_process_node_by_id(self, node_id):
		return self.db.get_process_node_by_id(node_id)

	def get_all_jobs(self):
		return self.db.get_all_jobs()

	def get_all_unprocessed_jobs(self):
		return self.db.get_all_unprocessed_jobs()

	def get_all_unprocessed_jobs_for_pn_id(self, pn_id):
		return self.db.get_all_unprocessed_jobs_for_pn_id(pn_id)

	def get_all_unprocessed_jobs_for_any_node(self):
		return self.db.get_all_unprocessed_jobs_for_any_node()

	def get_all_unprocessed_and_processing_jobs(self):
		return self.db.get_all_unprocessed_and_processing_jobs()

	def get_all_processing_jobs(self):
		return self.db.get_all_processing_jobs()

	def get_all_finished_jobs(self, limit=None):
		return self.db.get_all_finished_jobs(limit)

	def get_job(self, job_id):
		return self.db.get_job(job_id)

	def save_it(self, entity):
		self.db.save(entity)

	def reset_process_nodes_status(self):
		self.db.reset_process_nodes_status()

	def update_job(self, job_dict):
		return self.db.update_job(job_dict)

	def update_job_pn(self, job_id, pn_id):
		return self.db.update_job_pn(job_id, pn_id)

