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


import sqlite3 as sql

CREATE_PROCESS_NODES_TABLE_STR = 'CREATE TABLE IF NOT EXISTS ProcessNodes(Id INTEGER PRIMARY KEY, ComputerName TEXT, NumThreads INTEGER, Hostname TEXT, Port INTEGER, Status TEXT, Heartbeat TIMESTAMP);'
CREATE_JOBS_TABLE_STR = 'CREATE TABLE IF NOT EXISTS Jobs(Id INTEGER PRIMARY KEY, DataPath TEXT, ProcMask INTEGER, Version TEXT, DetectorElements INTEGER, MaxFilesToProc INTEGER, MaxLinesToProc INTEGER, QuickAndDirty INTEGER, XRF_Bin INTEGER, NNLS INTEGER, XANES_Scan INTEGER, DetectorToStartWith INTEGER, BeamLine TEXT, Standards TEXT, Priority INTEGER, Status INTEGER, StartProcTime TIMESTAMP, FinishProcTime TIMESTAMP);'
CREATE_JOB_QUEUE_TABLE_STR = 'CREATE TABLE IF NOT EXISTS JobQueue(Id INTEGER PRIMARY KEY, JobId INTEGER, PnId INTEGER, Status INTEGER, StartTime TIMESTAMP, StopStime TIMESTAMP, FOREIGN KEY(JobId) REFERENCES Jobs(Id), FOREIGN KEY(PnId) REFERENCES ProcessNodes(Id));'

DROP_PROCESS_NODES_STR = 'DROP TABLE IF EXISTS ProcessNodes;'
DROP_JOBS_STR = 'DROP TABLE IF EXISTS Jobs;'
DROP_JOB_QUEUE_STR = 'DROP TABLE IF EXISTS JobQueue;'

INSERT_PROCESS_NODE = 'INSERT INTO ProcessNodes (ComputerName, NumThreads, Hostname, Port, Status, Heartbeat) VALUES(:ComputerName, :NumThreads, :Hostname, :Port, :Status, :Heartbeat)'
INSERT_JOB = 'INSERT INTO Jobs (DataPath, ProcMask, Version, DetectorElements, MaxFilesToProc, MaxLinesToProc, QuickAndDirty, XRF_Bin, NNLS, XANES_Scan, DetectorToStartWith, BeamLine, Standards, Priority, Status, StartProcTime, FinishProcTime) VALUES(:DataPath, :ProcMask, :Version, :DetectorElements, :MaxFilesToProc, :MaxLinesToProc, :QuickAndDirty, :XRF_Bin, :NNLS, :XANES_Scan, :DetectorToStartWith, :BeamLine, :Standards, :Priority, :Status, NULL, NULL)'
INSERT_JOB_WITH_ID = 'INSERT INTO Jobs (Id, DataPath, ProcMask, Version, DetectorElements, MaxFilesToProc, MaxLinesToProc, QuickAndDirty, XRF_Bin, NNLS, XANES_Scan, DetectorToStartWith, BeamLine, Standards, Priority, Status, StartProcTime, FinishProcTime) VALUES(:Id, :DataPath, :ProcMask, :Version, :DetectorElements, :MaxFilesToProc, :MaxLinesToProc, :QuickAndDirty, :XRF_Bin, :NNLS, :XANES_Scan, :DetectorToStartWith, :BeamLine, :Standards, :Priority, :Status, NULL, NULL)'


UPDATE_PROCESS_NODE_BY_ID = 'UPDATE ProcessNodes SET ComputerName=:ComputerName NumThreads=:NumThreads Hostname=:Hostname, Port=:Port Status=:Status Heartbeat=:Heartbeat WHERE Id=:Id'
UPDATE_PROCESS_NODE_BY_NAME = 'UPDATE ProcessNodes SET NumThreads=:NumThreads, Hostname=:Hostname, Port=:Port, Status=:Status, Heartbeat=:Heartbeat WHERE ComputerName=:ComputerName'
UPDATE_JOB_BY_ID = 'UPDATE Jobs SET DataPath=:DataPath, ProcMask=:ProcMask, Version=:Version, DetectorElements=:DetectorElements, MaxFilesToProc=:MaxFilesToProc, MaxLinesToProc=:MaxLinesToProc, QuickAndDirty=:QuickAndDirty, XRF_Bin=:XRF_Bin, NNLS=:NNLS, XANES_Scan=:XANES_Scan, DetectorToStartWith=:DetectorToStartWith, BeamLine=:BeamLine, Standards=:Standards, Priority=:Priority, Status=:Status, StartProcTime=:StartProcTime, FinishProcTime=:FinishProcTime WHERE Id=:Id'

SELECT_ALL_PROCESS_NODES = 'SELECT ComputerName, NumThreads, Hostname, Port, Status, Heartbeat FROM ProcessNodes'
SELECT_PROCESS_NODE_BY_NAME = 'SELECT Id, ComputerName, NumThreads, Status, Heartbeat FROM ProcessNodes WHERE ComputerName=:ComputerName'
SELECT_ALL_JOBS = 'SELECT Id, DataPath, ProcMask, Version, DetectorElements, MaxFilesToProc, MaxLinesToProc, QuickAndDirty, XRF_Bin, NNLS, XANES_Scan, DetectorToStartWith, BeamLine, Standards, Priority, Status, StartProcTime, FinishProcTime FROM Jobs'
SELECT_ALL_UNPROCESSED_JOBS = SELECT_ALL_JOBS + ' WHERE Status=0'
SELECT_ALL_PROCESSING_JOBS = SELECT_ALL_JOBS + ' WHERE Status=1'
SELECT_ALL_FINISHED_JOBS = SELECT_ALL_JOBS + ' WHERE Status=2'
SELECT_JOB_BY_ID = SELECT_ALL_JOBS + ' WHERE Id=:Id'
SELECT_JOBS_BY_STATUS = SELECT_ALL_JOBS + ' WHERE Status=:Status ORDER BY Priority DESC'


class SQLiteDB:
	def __init__(self, db_name='MapsPy.db'):
		self.uri = db_name

	def create_tables(self, drop=False):
		con = sql.connect(self.uri)
		cur = con.cursor()
		if drop:
			cur.execute(DROP_PROCESS_NODES_STR)
			cur.execute(DROP_JOBS_STR)
			cur.execute(DROP_JOB_QUEUE_STR)
		cur.execute(CREATE_PROCESS_NODES_TABLE_STR)
		cur.execute(CREATE_JOBS_TABLE_STR)
		cur.execute(CREATE_JOB_QUEUE_TABLE_STR)
		con.commit()

	def insert_process_node(self, proc_node_dict):
		#first check if this process node exists
		con = sql.connect(self.uri)
		cur = con.cursor()
		cur.execute(SELECT_PROCESS_NODE_BY_NAME, proc_node_dict)
		con.commit()
		row = cur.fetchone()
		if row == None:
			print 'insert',proc_node_dict
			cur.execute(INSERT_PROCESS_NODE, proc_node_dict)
		else:
			print 'update', proc_node_dict
			cur.execute(UPDATE_PROCESS_NODE_BY_NAME, proc_node_dict)
		con.commit()

	def insert_job(self, job_dict):
		print 'insert job', job_dict
		con = sql.connect(self.uri)
		cur = con.cursor()
		cur.execute(INSERT_JOB, job_dict)
		con.commit()
		return cur.lastrowid

	def insert_job_with_id(self, job_dict):
		print 'insert job with id', job_dict
		con = sql.connect(self.uri)
		cur = con.cursor()
		cur.execute(INSERT_JOB_WITH_ID, job_dict)
		con.commit()

	def get_process_node(self, proc_node_name):
		con = sql.connect(self.uri)
		cur = con.cursor()
		cur.execute(SELECT_PROCESS_NODE_BY_NAME, {'ComputerName':proc_node_name})
		con.commit()
		return cur.fetchone()

	def get_all_process_nodes(self):
		con = sql.connect(self.uri)
		cur = con.cursor()
		cur.execute(SELECT_ALL_PROCESS_NODES)
		con.commit()
		#return cur.fetchall()
		all_nodes = cur.fetchall()
		ret_list = []
		#SELECT_ALL_PROCESS_NODES = 'SELECT ComputerName, NumThreads, Hostname, Port, Status, Heartbeat FROM ProcessNodes'
		for node in all_nodes:
			ret_list += [ {'ComputerName':node[0], 'NumThreads':node[1], 'Hostname':node[2], 'Port':node[3], 'Status': node[4], 'Heartbeat': node[5]} ]
		return ret_list

	def _get_jobs_(self, sql_statement):
		con = sql.connect(self.uri)
		cur = con.cursor()
		cur.execute(sql_statement)
		con.commit()
		all_nodes = cur.fetchall()
		ret_list = []
		#SELECT_ALL_JOBS = 'SELECT Id, DataPath, ProcMask, Version, DetectorElements, MaxFilesToProc, MaxLinesToProc, QuickAndDirty, XRF_Bin, NNLS, XANES_Scan, DetectorToStartWith, BeamLine, Standards, Status, StartProcTime, FinishProcTime FROM Jobs'
		for node in all_nodes:
			ret_list += [ {'Id':node[0], 'DataPath':node[1], 'ProcMask': node[2], 'Version': node[3], 'DetectorElements':node[4], 'MaxFilesToProc':node[5], 'MaxLinesToProc':node[6], 'QuickAndDirty':node[7], 'XRF_Bin':node[8], 'NNLS':node[9], 'XANES_Scan':node[10], 'DetectorToStartWith':node[11], 'BeamLine':node[12], 'Standards':node[13], 'Priority':node[14], 'Status':node[15], 'StartProcTime':node[16], 'FinishProcTime':node[17]  } ]
		return ret_list

	def get_all_jobs(self):
		return self._get_jobs_(SELECT_ALL_JOBS)

	def get_all_unprocessed_jobs(self):
		return self._get_jobs_(SELECT_ALL_UNPROCESSED_JOBS)

	def get_all_processing_jobs(self):
		return self._get_jobs_(SELECT_ALL_PROCESSING_JOBS)

	def get_all_finished_jobs(self):
		return self._get_jobs_(SELECT_ALL_FINISHED_JOBS)

	def get_job(self, job_id):
		con = sql.connect(self.uri)
		cur = con.cursor()
		cur.execute(SELECT_JOB_BY_ID, {'Id':job_id})
		con.commit()
		return cur.fetchone()

	def get_jobs_by_status(self, status):
		con = sql.connect(self.uri)
		cur = con.cursor()
		cur.execute(SELECT_JOBS_BY_STATUS, {'Status':status})
		con.commit()
		return cur.fetchall()

	def update_job(self, job_dict):
		print 'updating job', job_dict
		con = sql.connect(self.uri)
		cur = con.cursor()
		cur.execute(UPDATE_JOB_BY_ID, job_dict)
		con.commit()

	def save(self, entry):
		print 'saving',entry

if __name__ == '__main__':
	import datetime
	proc_node = { 'ComputerName':'Comp1', 'NumThreads':1, 'Hostname':'127.0.0.2', 'Port':8080, 'Status':'idle', 'Heartbeat':datetime.datetime.now()}
	proc_node2 = { 'ComputerName':'Comp2', 'NumThreads':2, 'Hostname':'127.0.0.3', 'Port':8080, 'Status':'idle', 'Heartbeat':datetime.datetime.now()}
	job1 = { 'DataPath':'/data/mapspy1/', 'ProcMask':1, 'Version':'1.00', 'DetectorElements':1, 'MaxFilesToProc':1, 'MaxLinesToProc':11, 'QuickAndDirty':0, 'XRF_Bin':0, 'NNLS':0, 'XANES_Scan':0, 'DetectorToStartWith':0, 'BeamLine':'2-ID-E', 'Standards':'', 'Priority':5, 'Status':0, 'StartProcTime':0, 'FinishProcTime':0 }
	job2 = { 'DataPath':'/data/mapspy2/', 'ProcMask':4, 'Version':'1.00', 'DetectorElements':1, 'MaxFilesToProc':1, 'MaxLinesToProc':11, 'QuickAndDirty':0, 'XRF_Bin':0, 'NNLS':0, 'XANES_Scan':0, 'DetectorToStartWith':0, 'BeamLine':'2-ID-E', 'Standards':'', 'Priority':10, 'Status':0, 'StartProcTime':0, 'FinishProcTime':0 }
	job3 = { 'DataPath':'/data/mapspy3/', 'ProcMask':8, 'Version':'1.00', 'DetectorElements':1, 'MaxFilesToProc':1, 'MaxLinesToProc':11, 'QuickAndDirty':0, 'XRF_Bin':0, 'NNLS':0, 'XANES_Scan':0, 'DetectorToStartWith':0, 'BeamLine':'2-ID-E', 'Standards':'', 'Priority':7, 'Status':0, 'StartProcTime':0, 'FinishProcTime':0 }
	job4 = {'Status': 1, 'MaxLinesToProc': 11, 'FinishProcTime': None, 'ProcMask': 1, 'XRF_Bin': 0, 'MaxFilesToProc': 1, 'BeamLine': u'2-ID-E', 'DataPath': u'/maps_py', 'DetectorElements': 1, 'Priority': 5, 'XANES_Scan': 0, 'Version': u'1.00', 'StartProcTime': None, 'NNLS': 0, 'QuickAndDirty': 0, 'Standards': u'', 'DetectorToStartWith': 0, 'Id': 1}
	db = SQLiteDB('TestDatabase.db')
	db.create_tables(True)
	db.insert_process_node(proc_node)
	db.insert_process_node(proc_node2)
	proc_node['Status'] = 'Offline'
	proc_node['Heartbeat'] = datetime.datetime.now()
	db.insert_process_node(proc_node)
	import json
	result = db.get_all_process_nodes()
	jenc = json.JSONEncoder()
	print jenc.encode(result)
	#add job
	job1['Id'] = db.insert_job(job1)
	job2['Id'] = db.insert_job(job2)
	job3['Id'] = db.insert_job(job3)
	print db.get_jobs_by_status(0)
	job3['Status'] = 3
	db.update_job(job3)
	print db.get_jobs_by_status(0)
	#print db.get_all_jobs()
	
