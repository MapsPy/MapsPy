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

CREATE_PROCESS_NODES_TABLE_STR = 'CREATE TABLE IF NOT EXISTS ProcessNodes(Id INTEGER PRIMARY KEY, ComputerName TEXT, NumThreads INTEGER, Hostname TEXT, Port INTEGER, Status TEXT, Heartbeat TIMESTAMP, ProcessCpuPercent REAL, ProcessMemPercent REAL, SystemCpuPercent REAL, SystemMemPercent REAL, SystemSwapPercent REAL);'
CREATE_JOBS_TABLE_STR = 'CREATE TABLE IF NOT EXISTS Jobs(Id INTEGER PRIMARY KEY, DataPath TEXT, ProcMask INTEGER, Version TEXT, DetectorElements INTEGER, MaxFilesToProc INTEGER, MaxLinesToProc INTEGER, QuickAndDirty INTEGER, XRF_Bin INTEGER, NNLS INTEGER, XANES_Scan INTEGER, DetectorToStartWith INTEGER, BeamLine TEXT, Standards TEXT, DatasetFilesToProc TEXT, Priority INTEGER, Status INTEGER, StartProcTime TIMESTAMP, FinishProcTime TIMESTAMP, Log_Path TEXT, Process_Node_Id INTEGER, Emails, TEXT, Is_Live_Job INTEGER);'
CREATE_JOB_QUEUE_TABLE_STR = 'CREATE TABLE IF NOT EXISTS JobQueue(Id INTEGER PRIMARY KEY, JobId INTEGER, PnId INTEGER, Status INTEGER, StartTime TIMESTAMP, StopStime TIMESTAMP, FOREIGN KEY(JobId) REFERENCES Jobs(Id), FOREIGN KEY(PnId) REFERENCES ProcessNodes(Id));'

DROP_PROCESS_NODES_STR = 'DROP TABLE IF EXISTS ProcessNodes;'
DROP_JOBS_STR = 'DROP TABLE IF EXISTS Jobs;'
DROP_JOB_QUEUE_STR = 'DROP TABLE IF EXISTS JobQueue;'

INSERT_PROCESS_NODE = 'INSERT INTO ProcessNodes (ComputerName, NumThreads, Hostname, Port, Status, Heartbeat, ProcessCpuPercent, ProcessMemPercent, SystemCpuPercent, SystemMemPercent, SystemSwapPercent) VALUES(:ComputerName, :NumThreads, :Hostname, :Port, :Status, :Heartbeat, :ProcessCpuPercent, :ProcessMemPercent, :SystemCpuPercent, :SystemMemPercent, :SystemSwapPercent)'
INSERT_JOB = 'INSERT INTO Jobs (DataPath, ProcMask, Version, DetectorElements, MaxFilesToProc, MaxLinesToProc, QuickAndDirty, XRF_Bin, NNLS, XANES_Scan, DetectorToStartWith, BeamLine, Standards, DatasetFilesToProc, Priority, Status, StartProcTime, FinishProcTime, Log_Path, Process_Node_Id, Emails, Is_Live_Job) VALUES(:DataPath, :ProcMask, :Version, :DetectorElements, :MaxFilesToProc, :MaxLinesToProc, :QuickAndDirty, :XRF_Bin, :NNLS, :XANES_Scan, :DetectorToStartWith, :BeamLine, :Standards, :DatasetFilesToProc, :Priority, :Status, :StartProcTime, :FinishProcTime, :Log_Path, :Process_Node_Id, :Emails, :Is_Live_Job)'
INSERT_JOB_WITH_ID = 'INSERT INTO Jobs (Id, DataPath, ProcMask, Version, DetectorElements, MaxFilesToProc, MaxLinesToProc, QuickAndDirty, XRF_Bin, NNLS, XANES_Scan, DetectorToStartWith, BeamLine, Standards, DatasetFilesToProc, Priority, Status, StartProcTime, FinishProcTime, Log_Path, Process_Node_Id, Emails, Is_Live_Job) VALUES(:Id, :DataPath, :ProcMask, :Version, :DetectorElements, :MaxFilesToProc, :MaxLinesToProc, :QuickAndDirty, :XRF_Bin, :NNLS, :XANES_Scan, :DetectorToStartWith, :BeamLine, :Standards, :DatasetFilesToProc, :Priority, :Status, :StartProcTime, :FinishProcTime, :Log_Path, :Process_Node_Id, :Emails, :Is_Live_Job)'


UPDATE_PROCESS_NODE_BY_ID = 'UPDATE ProcessNodes SET ComputerName=:ComputerName NumThreads=:NumThreads Hostname=:Hostname, Port=:Port Status=:Status Heartbeat=:Heartbeat, ProcessCpuPercent=:ProcessCpuPercent, ProcessMemPercent=:ProcessMemPercent, SystemCpuPercent=:SystemCpuPercent, SystemMemPercent=:SystemMemPercent, SystemSwapPercent=:SystemSwapPercent WHERE Id=:Id'
UPDATE_PROCESS_NODE_BY_NAME = 'UPDATE ProcessNodes SET NumThreads=:NumThreads, Hostname=:Hostname, Port=:Port, Status=:Status, Heartbeat=:Heartbeat, ProcessCpuPercent=:ProcessCpuPercent, ProcessMemPercent=:ProcessMemPercent, SystemCpuPercent=:SystemCpuPercent, SystemMemPercent=:SystemMemPercent, SystemSwapPercent=:SystemSwapPercent WHERE ComputerName=:ComputerName'
UPDATE_JOB_BY_ID = 'UPDATE Jobs SET DataPath=:DataPath, ProcMask=:ProcMask, Version=:Version, DetectorElements=:DetectorElements, MaxFilesToProc=:MaxFilesToProc, MaxLinesToProc=:MaxLinesToProc, QuickAndDirty=:QuickAndDirty, XRF_Bin=:XRF_Bin, NNLS=:NNLS, XANES_Scan=:XANES_Scan, DetectorToStartWith=:DetectorToStartWith, BeamLine=:BeamLine, Standards=:Standards, DatasetFilesToProc=:DatasetFilesToProc, Priority=:Priority, Status=:Status, StartProcTime=:StartProcTime, FinishProcTime=:FinishProcTime, Log_Path=:Log_Path, Process_Node_Id=:Process_Node_Id, Emails=:Emails, Is_Live_Job=:Is_Live_Job WHERE Id=:Id'

RESET_PN_STATUS = 'UPDATE ProcessNodes SET Status="Offline", ProcessCpuPercent=0.0, ProcessMemPercent=0.0 WHERE Id>0;'

SELECT_ALL_PROCESS_NODES = 'SELECT Id, ComputerName, NumThreads, Hostname, Port, Status, Heartbeat, ProcessCpuPercent, ProcessMemPercent, SystemCpuPercent, SystemMemPercent, SystemSwapPercent FROM ProcessNodes'
SELECT_PROCESS_NODE_BY_NAME = SELECT_ALL_PROCESS_NODES + ' WHERE ComputerName=:ComputerName'
SELECT_PROCESS_NODE_BY_ID = SELECT_ALL_PROCESS_NODES + ' WHERE Id=:Id'
SELECT_ALL_JOBS = 'SELECT Id, DataPath, ProcMask, Version, DetectorElements, MaxFilesToProc, MaxLinesToProc, QuickAndDirty, XRF_Bin, NNLS, XANES_Scan, DetectorToStartWith, BeamLine, Standards, DatasetFilesToProc, Priority, Status, StartProcTime, FinishProcTime, Log_Path, Process_Node_Id, Emails, Is_Live_Job FROM Jobs'
SELECT_ALL_UNPROCESSED_JOBS = SELECT_ALL_JOBS + ' WHERE Status=0'
SELECT_ALL_PROCESSING_JOBS = SELECT_ALL_JOBS + ' WHERE Status=1'
SELECT_ALL_FINISHED_JOBS = SELECT_ALL_JOBS + ' WHERE Status>=2'
SELECT_ALL_UNPROCESSED_AND_PROCESSING_JOBS = SELECT_ALL_JOBS + ' WHERE Status<=1 ORDER BY Status DESC'
SELECT_ALL_UNPROCESSED_JOBS_FOR_PN_ID = SELECT_ALL_JOBS + ' WHERE Status<=1 AND Process_Node_Id=:Process_Node_Id ORDER BY Priority ASC'
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

	def get_process_node_by_name(self, proc_node_name):
		nodes = self._get_proc_node(SELECT_PROCESS_NODE_BY_NAME, {'ComputerName':proc_node_name})
		if len(nodes) > 0:
			return nodes[0]
		return None

	def get_process_node_by_id(self, proc_node_id):
		nodes = self._get_proc_node(SELECT_PROCESS_NODE_BY_ID, {'Id':proc_node_id})
		if len(nodes) > 0:
			return nodes[0]
		return None

	def get_all_process_nodes(self):
		return self._get_proc_node(SELECT_ALL_PROCESS_NODES)

	def _get_proc_node(self, sql_statement, opt_dict=None):
		con = sql.connect(self.uri)
		cur = con.cursor()
		if opt_dict == None:
			cur.execute(sql_statement)
		else:
			cur.execute(sql_statement, opt_dict)
		con.commit()
		all_nodes = cur.fetchall()
		ret_list = []
		#SELECT_ALL_PROCESS_NODES = 'SELECT ComputerName, NumThreads, Hostname, Port, Status, Heartbeat FROM ProcessNodes'
		for node in all_nodes:
			ret_list += [ {'DT_RowId':'row_'+str(node[0]), 'Id':node[0], 'ComputerName':node[1], 'NumThreads':node[2], 'Hostname':node[3], 'Port':node[4], 'Status': node[5], 'Heartbeat': node[6], 'ProcessCpuPercent':node[7], 'ProcessMemPercent':node[8], 'SystemCpuPercent':node[9], 'SystemMemPercent': node[10], 'SystemSwapPercent':node[11] } ]
		return ret_list


	def _get_jobs_(self, sql_statement, opt_dict=None):
		con = sql.connect(self.uri)
		cur = con.cursor()
		if opt_dict == None:
			cur.execute(sql_statement)
		else:
			cur.execute(sql_statement, opt_dict)
		con.commit()
		all_nodes = cur.fetchall()
		ret_list = []
		#SELECT_ALL_JOBS = 'SELECT Id, DataPath, ProcMask, Version, DetectorElements, MaxFilesToProc, MaxLinesToProc, QuickAndDirty, XRF_Bin, NNLS, XANES_Scan, DetectorToStartWith, BeamLine, Standards, DatasetFilesToProc, Status, StartProcTime, FinishProcTime FROM Jobs'
		for node in all_nodes:
			ret_list += [ {'DT_RowId':node[0],  'Id':int(node[0]), 'DataPath':node[1], 'ProcMask': node[2], 'Version': node[3], 'DetectorElements':int(node[4]), 'MaxFilesToProc':int(node[5]), 'MaxLinesToProc':int(node[6]), 'QuickAndDirty':node[7], 'XRF_Bin':node[8], 'NNLS':node[9], 'XANES_Scan':node[10], 'DetectorToStartWith':int(node[11]), 'BeamLine':node[12], 'Standards':node[13], 'DatasetFilesToProc':node[14], 'Priority':int(node[15]), 'Status':int(node[16]), 'StartProcTime':node[17], 'FinishProcTime':node[18], 'Log_Path':node[19], 'Process_Node_Id':int(node[20]), 'Emails':node[21], 'Is_Live_Job':node[22]  } ]
		return ret_list

	def get_all_jobs(self):
		return self._get_jobs_(SELECT_ALL_JOBS)

	def get_all_unprocessed_jobs(self):
		return self._get_jobs_(SELECT_ALL_UNPROCESSED_JOBS)

	def get_all_unprocessed_jobs_for_pn_id(self, pn_id):
		return self._get_jobs_(SELECT_ALL_UNPROCESSED_JOBS_FOR_PN_ID, {'Process_Node_Id': pn_id})

	def get_all_unprocessed_and_processing_jobs(self):
		return self._get_jobs_(SELECT_ALL_UNPROCESSED_AND_PROCESSING_JOBS)

	def get_all_processing_jobs(self):
		return self._get_jobs_(SELECT_ALL_PROCESSING_JOBS)

	def get_all_finished_jobs(self):
		return self._get_jobs_(SELECT_ALL_FINISHED_JOBS)

	def get_job(self, job_id):
		jobs = self._get_jobs_(SELECT_JOB_BY_ID, {'Id': int(job_id)})
		if len(jobs) > 0:
			return jobs[0]
		return None

	def get_jobs_by_status(self, status):
		jobs = self._get_jobs_(SELECT_JOBS_BY_STATUS, {'Status':status})
		if len(jobs) > 0:
			return jobs[0]
		return None

	def update_job(self, job_dict):
		print 'updating job', job_dict
		con = sql.connect(self.uri)
		cur = con.cursor()
		cur.execute(UPDATE_JOB_BY_ID, job_dict)
		con.commit()

	def save(self, entry):
		print 'saving',entry

	def reset_process_nodes_status(self):
		con = sql.connect(self.uri)
		cur = con.cursor()
		cur.execute(RESET_PN_STATUS)
		con.commit()

if __name__ == '__main__':
	from datetime import datetime
	proc_node = { 'ComputerName':'Comp1', 'NumThreads':1, 'Hostname':'127.0.0.2', 'Port':8080, 'Status':'idle', 'Heartbeat':datetime.now(), 'ProcessCpuPercent':0.0, 'ProcessMemPercent':1.0, 'SystemCpuPercent':2.0, 'SystemMemPercent':10.0, 'SystemSwapPercent':0.0}
	proc_node2 = { 'ComputerName':'Comp2', 'NumThreads':2, 'Hostname':'127.0.0.3', 'Port':8080, 'Status':'idle', 'Heartbeat':datetime.now(), 'ProcessCpuPercent':0.0, 'ProcessMemPercent':1.0, 'SystemCpuPercent':2.0, 'SystemMemPercent':10.0, 'SystemSwapPercent':0.0}
	job1 = { 'DataPath':'/data/mapspy1/', 'ProcMask':1, 'Version':'1.00', 'DetectorElements':1, 'MaxFilesToProc':1, 'MaxLinesToProc':11, 'QuickAndDirty':0, 'XRF_Bin':0, 'NNLS':0, 'XANES_Scan':0, 'DetectorToStartWith':0, 'BeamLine':'2-ID-E', 'Standards':'', 'DatasetFilesToProc': 'all', 'Priority':5, 'Status':0, 'StartProcTime':datetime.ctime(datetime.now()), 'FinishProcTime':0, 'Log_Path': '', 'Process_Node_Id': -1, 'Emails':'', 'Is_Live_Job':0 }
	job2 = { 'DataPath':'/data/mapspy2/', 'ProcMask':4, 'Version':'1.00', 'DetectorElements':1, 'MaxFilesToProc':1, 'MaxLinesToProc':11, 'QuickAndDirty':0, 'XRF_Bin':0, 'NNLS':0, 'XANES_Scan':0, 'DetectorToStartWith':0, 'BeamLine':'2-ID-E', 'Standards':'', 'DatasetFilesToProc': 'all', 'Priority':10, 'Status':0, 'StartProcTime':datetime.ctime(datetime.now()), 'FinishProcTime':0, 'Log_Path': '', 'Process_Node_Id': -1, 'Emails':'', 'Is_Live_Job':0 }
	job3 = { 'DataPath':'/data/mapspy3/', 'ProcMask':8, 'Version':'1.00', 'DetectorElements':1, 'MaxFilesToProc':1, 'MaxLinesToProc':11, 'QuickAndDirty':0, 'XRF_Bin':0, 'NNLS':0, 'XANES_Scan':0, 'DetectorToStartWith':0, 'BeamLine':'2-ID-E', 'Standards':'', 'DatasetFilesToProc': 'all', 'Priority':7, 'Status':0, 'StartProcTime':datetime.ctime(datetime.now()), 'FinishProcTime':0, 'Log_Path': '', 'Process_Node_Id': -1, 'Emails':'', 'Is_Live_Job':1 }
	db = SQLiteDB('TestDatabase.db')
	db.create_tables(True)
	db.insert_process_node(proc_node)
	db.insert_process_node(proc_node2)
	proc_node['Status'] = 'Offline'
	proc_node['Heartbeat'] = datetime.now()
	db.insert_process_node(proc_node)
	import json
	result = db.get_all_process_nodes()
	print ' '
	d = datetime.strptime(result[0]['Heartbeat'], '%Y-%m-%d %H:%M:%S.%f')
	print type(d), d
	print ' '
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
