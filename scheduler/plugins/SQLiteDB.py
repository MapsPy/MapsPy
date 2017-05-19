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
CREATE_JOBS_TABLE_STR = 'CREATE TABLE IF NOT EXISTS Jobs(Id INTEGER PRIMARY KEY, DataPath TEXT, DatasetFilesToProc TEXT, Version TEXT, BeamLine TEXT, Priority INTEGER, Status INTEGER, StartProcTime TIMESTAMP, FinishProcTime TIMESTAMP, Log_Path TEXT, Process_Node_Id INTEGER, Emails, TEXT);'
CREATE_JOBS_XRF_ARGS_TABLE_STR = 'CREATE TABLE IF NOT EXISTS JobsXrfArgs(Id INTEGER, ProcMask INTEGER, DetectorElements INTEGER, MaxFilesToProc INTEGER, MaxLinesToProc INTEGER, QuickAndDirty INTEGER, XRF_Bin INTEGER, NNLS INTEGER, XANES_Scan INTEGER, DetectorToStartWith INTEGER, Standards TEXT, Is_Live_Job INTEGER, FOREIGN KEY(Id) REFERENCES Jobs(Id));'
CREATE_JOBS_PTY_ARGS_TABLE_STR = 'CREATE TABLE IF NOT EXISTS JobsPtyArgs(Id INTEGER, CalcSTXM INTEGER, AlgorithmEPIE INTEGER, AlgorithmDM INTEGER, DetectorDistance INTEGER, PixelSize INTEGER, CenterY INTEGER, CenterX INTEGER, DiffractionSize INTEGER, Rotation INTEGER, GPU_ID INTEGER, ProbeSize INTEGER, ProbeModes INTEGER, Threshold INTEGER, Iterations INTEGER,  FOREIGN KEY(Id) REFERENCES Jobs(Id));'

DROP_ALL_TABLES_STR = 'DROP TABLE IF EXISTS ProcessNodes; \
                       DROP TABLE IF EXISTS Jobs; \
                       DROP TABLE IF EXISTS JobsXrfArgs; \
                       DROP TABLE IF EXISTS JobsPtyArgs;'

INSERT_PROCESS_NODE = 'INSERT INTO ProcessNodes (ComputerName, NumThreads, Hostname, Port, Status, Heartbeat, ProcessCpuPercent, ProcessMemPercent, SystemCpuPercent, SystemMemPercent, SystemSwapPercent) VALUES(:ComputerName, :NumThreads, :Hostname, :Port, :Status, :Heartbeat, :ProcessCpuPercent, :ProcessMemPercent, :SystemCpuPercent, :SystemMemPercent, :SystemSwapPercent)'

INSERT_JOB = 'INSERT INTO Jobs (DataPath, Version, BeamLine, DatasetFilesToProc, Priority, Status, StartProcTime, FinishProcTime, Log_Path, Process_Node_Id, Emails) VALUES(:DataPath, :Version, :BeamLine, :DatasetFilesToProc, :Priority, :Status, :StartProcTime, :FinishProcTime, :Log_Path, :Process_Node_Id, :Emails)'
INSERT_XRF_JOB = 'INSERT INTO JobsXrfArgs (Id, ProcMask, DetectorElements, MaxFilesToProc, MaxLinesToProc, QuickAndDirty, XRF_Bin, NNLS, XANES_Scan, DetectorToStartWith, Standards, Is_Live_Job) VALUES(last_insert_rowid(), :ProcMask, :DetectorElements, :MaxFilesToProc, :MaxLinesToProc, :QuickAndDirty, :XRF_Bin, :NNLS, :XANES_Scan, :DetectorToStartWith, :Standards, :Is_Live_Job)'
INSERT_PTY_JOB = 'INSERT INTO JobsPtyArgs (Id, CalcSTXM, AlgorithmEPIE, AlgorithmDM, DetectorDistance, PixelSize, CenterY, CenterX, DiffractionSize, Rotation, GPU_ID, ProbeSize, ProbeModes, Threshold, Iterations) VALUES(last_insert_rowid(), :CalcSTXM, :AlgorithmEPIE, :AlgorithmDM, :DetectorDistance, :PixelSize, :CenterY, :CenterX, :DiffractionSize, :Rotation, :GPU_ID, :ProbeSize, :ProbeModes, :Threshold, :Iterations)'

INSERT_JOB_WTIH_ID = 'INSERT INTO Jobs (Id, DataPath, Version, BeamLine, DatasetFilesToProc, Priority, Status, StartProcTime, FinishProcTime, Log_Path, Process_Node_Id, Emails) VALUES(:Id, :DataPath, :Version, :BeamLine, :DatasetFilesToProc, :Priority, :Status, :StartProcTime, :FinishProcTime, :Log_Path, :Process_Node_Id, :Emails)'
INSERT_XRF_JOB_WTIH_ID = 'INSERT INTO JobsXrfArgs (Id, ProcMask, DetectorElements, MaxFilesToProc, MaxLinesToProc, QuickAndDirty, XRF_Bin, NNLS, XANES_Scan, DetectorToStartWith, Standards, Is_Live_Job) VALUES(:Id, :ProcMask, :DetectorElements, :MaxFilesToProc, :MaxLinesToProc, :QuickAndDirty, :XRF_Bin, :NNLS, :XANES_Scan, :DetectorToStartWith, :Standards, :Is_Live_Job)'
INSERT_PTY_JOB_WITH_ID = 'INSERT INTO JobsPtyArgs (Id, CalcSTXM, AlgorithmEPIE, AlgorithmDM, DetectorDistance, PixelSize, CenterY, CenterX, DiffractionSize, Rotation, GPU_ID, ProbeSize, ProbeModes, Threshold, Iterations) VALUES(:Id, :CalcSTXM, :AlgorithmEPIE, :AlgorithmDM, :DetectorDistance, :PixelSize, :CenterY, :CenterX, :DiffractionSize, :Rotation, :GPU_ID, :ProbeSize, :ProbeModes, :Threshold, :Iterations)'

UPDATE_PROCESS_NODE_BY_ID = 'UPDATE ProcessNodes SET ComputerName=:ComputerName NumThreads=:NumThreads Hostname=:Hostname, Port=:Port Status=:Status Heartbeat=:Heartbeat, ProcessCpuPercent=:ProcessCpuPercent, ProcessMemPercent=:ProcessMemPercent, SystemCpuPercent=:SystemCpuPercent, SystemMemPercent=:SystemMemPercent, SystemSwapPercent=:SystemSwapPercent WHERE Id=:Id'
UPDATE_PROCESS_NODE_BY_NAME = 'UPDATE ProcessNodes SET NumThreads=:NumThreads, Hostname=:Hostname, Port=:Port, Status=:Status, Heartbeat=:Heartbeat, ProcessCpuPercent=:ProcessCpuPercent, ProcessMemPercent=:ProcessMemPercent, SystemCpuPercent=:SystemCpuPercent, SystemMemPercent=:SystemMemPercent, SystemSwapPercent=:SystemSwapPercent WHERE ComputerName=:ComputerName'

UPDATE_JOB_BY_ID = 'UPDATE Jobs SET DataPath=:DataPath, Version=:Version, BeamLine=:BeamLine, DatasetFilesToProc=:DatasetFilesToProc, Priority=:Priority, Status=:Status, StartProcTime=:StartProcTime, FinishProcTime=:FinishProcTime, Log_Path=:Log_Path, Process_Node_Id=:Process_Node_Id, Emails=:Emails WHERE Id=:Id'
UPDATE_JOB_PN = 'UPDATE Jobs SET Process_Node_Id=:Process_Node_Id WHERE Id=:Id'

RESET_PN_STATUS = 'UPDATE ProcessNodes SET Status="Offline", ProcessCpuPercent=0.0, ProcessMemPercent=0.0 WHERE Id>0;'

SELECT_ALL_PROCESS_NODES = 'SELECT Id, ComputerName, NumThreads, Hostname, Port, Status, Heartbeat, ProcessCpuPercent, ProcessMemPercent, SystemCpuPercent, SystemMemPercent, SystemSwapPercent FROM ProcessNodes'
SELECT_PROCESS_NODE_BY_NAME = SELECT_ALL_PROCESS_NODES + ' WHERE ComputerName=:ComputerName'
SELECT_PROCESS_NODE_BY_ID = SELECT_ALL_PROCESS_NODES + ' WHERE Id=:Id'

SELECT_ALL_JOBS = 'SELECT Jobs.Id, Jobs.BeamLine, Jobs.Version, Jobs.DataPath, Jobs.DatasetFilesToProc, Jobs.Priority, Jobs.Status, Jobs.StartProcTime, Jobs.FinishProcTime, Jobs.Log_Path, Jobs.Process_Node_Id, Jobs.Emails, JobsXrfArgs.ProcMask, JobsXrfArgs.DetectorElements, JobsXrfArgs.MaxFilesToProc, JobsXrfArgs.MaxLinesToProc, JobsXrfArgs.QuickAndDirty, JobsXrfArgs.XRF_Bin, JobsXrfArgs.NNLS, JobsXrfArgs.XANES_Scan, JobsXrfArgs.DetectorToStartWith, JobsXrfArgs.Standards, JobsXrfArgs.Is_Live_Job, JobsPtyArgs.CalcSTXM, JobsPtyArgs.AlgorithmEPIE, JobsPtyArgs.AlgorithmDM, JobsPtyArgs.DetectorDistance, JobsPtyArgs.PixelSize, JobsPtyArgs.CenterY, JobsPtyArgs.CenterX, JobsPtyArgs.DiffractionSize, JobsPtyArgs.Rotation, JobsPtyArgs.GPU_ID, JobsPtyArgs.ProbeSize, JobsPtyArgs.ProbeModes, JobsPtyArgs.Threshold, JobsPtyArgs.Iterations FROM Jobs LEFT JOIN JobsXrfArgs ON Jobs.Id == JobsXrfArgs.Id LEFT JOIN JobsPtyArgs ON Jobs.Id == JobsPtyArgs.Id'
#SELECT_ALL_JOBS = 'SELECT Id, DataPath, ProcMask, Version, DetectorElements, MaxFilesToProc, MaxLinesToProc, QuickAndDirty, XRF_Bin, NNLS, XANES_Scan, DetectorToStartWith, BeamLine, Standards, DatasetFilesToProc, Priority, Status, StartProcTime, FinishProcTime, Log_Path, Process_Node_Id, Emails, Is_Live_Job FROM Jobs'
SELECT_ALL_UNPROCESSED_JOBS = SELECT_ALL_JOBS + ' WHERE Jobs.Status=0'
SELECT_ALL_UNPROCESSED_JOBS_ANY_NODE = SELECT_ALL_JOBS + ' WHERE Jobs.Status=0 and Jobs.Process_Node_Id=-1'
SELECT_ALL_PROCESSING_JOBS = SELECT_ALL_JOBS + ' WHERE Jobs.Status=1'
SELECT_ALL_FINISHED_JOBS = SELECT_ALL_JOBS + ' WHERE Jobs.Status>=2'
SELECT_ALL_FINISHED_JOBS_LIMIT = SELECT_ALL_JOBS + ' WHERE Jobs.Status>=2 ORDER BY Jobs.Id DESC LIMIT '
SELECT_ALL_UNPROCESSED_AND_PROCESSING_JOBS = SELECT_ALL_JOBS + ' WHERE Jobs.Status<=1 ORDER BY Jobs.Status DESC'
SELECT_ALL_UNPROCESSED_JOBS_FOR_PN_ID = SELECT_ALL_JOBS + ' WHERE Jobs.Status<=1 AND Jobs.Process_Node_Id=:Process_Node_Id ORDER BY Jobs.Priority ASC'
SELECT_JOB_BY_ID = SELECT_ALL_JOBS + ' WHERE Jobs.Id=:Id'
SELECT_JOBS_BY_STATUS = SELECT_ALL_JOBS + ' WHERE Jobs.Status=:Status ORDER BY Jobs.Priority DESC'

DELETE_JOB_BY_ID = 'DELETE FROM Jobs WHERE Id=:Id'


class SQLiteDB:
	def __init__(self, db_name='MapsPy.db'):
		self.uri = db_name

	def create_tables(self, drop=False):
		con = sql.connect(self.uri)
		cur = con.cursor()
		if drop:
			cur.executescript(DROP_ALL_TABLES_STR)
		cur.execute(CREATE_PROCESS_NODES_TABLE_STR)
		cur.execute(CREATE_JOBS_TABLE_STR)
		cur.execute(CREATE_JOBS_XRF_ARGS_TABLE_STR)
		cur.execute(CREATE_JOBS_PTY_ARGS_TABLE_STR)
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
		INSERT_STR = ''
		if 'BeamLine' in job_dict:
			if job_dict['BeamLine'] == 'XRF':
				INSERT_STR = INSERT_XRF_JOB
			elif job_dict['BeamLine'] == 'PTY':
				INSERT_STR = INSERT_PTY_JOB
		con = sql.connect(self.uri)
		cur = con.cursor()
		cur.execute(INSERT_JOB, job_dict)
		cur.execute(INSERT_STR, job_dict)
		con.commit()
		return cur.lastrowid

	def insert_job_with_id(self, job_dict):
		print 'insert job with id', job_dict
		INSERT_STR = ''
		if 'BeamLine' in job_dict:
			if job_dict['BeamLine'] == 'XRF':
				INSERT_STR = INSERT_XRF_JOB_WTIH_ID
			elif job_dict['BeamLine'] == 'PTY':
				INSERT_STR = INSERT_PTY_JOB_WITH_ID
		con = sql.connect(self.uri)
		cur = con.cursor()
		cur.execute(INSERT_JOB_WTIH_ID, job_dict)
		cur.execute(INSERT_STR, job_dict)
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
			if node[1] == 'XRF':
				ret_list += [ {'DT_RowId':node[0],  'Id':int(node[0]), 'BeamLine':node[1], 'Version': node[2], 'DataPath':node[3], 'DatasetFilesToProc': node[4], 'Priority':int(node[5]), 'Status':int(node[6]), 'StartProcTime':node[7], 'FinishProcTime':node[8], 'Log_Path':node[9], 'Process_Node_Id':node[10], 'Emails':node[11], 'ProcMask':int(node[12]), 'DetectorElements':int(node[13]), 'MaxFilesToProc':int(node[14]), 'MaxLinesToProc':int(node[15]), 'QuickAndDirty':int(node[16]), 'XRF_Bin':int(node[17]), 'NNLS':int(node[18]), 'XANES_Scan':(node[19]), 'DetectorToStartWith':int(node[20]), 'Standards':node[21], 'Is_Live_Job':node[22]  } ]
			elif node[1] == 'PTY':
				ret_list += [ {'DT_RowId':node[0],  'Id':int(node[0]), 'BeamLine':node[1], 'Version': node[2], 'DataPath':node[3], 'DatasetFilesToProc': node[4], 'Priority':int(node[5]), 'Status':int(node[6]), 'StartProcTime':node[7], 'FinishProcTime':node[8], 'Log_Path':node[9], 'Process_Node_Id':node[10], 'Emails':node[11], 'CalcSTXM':int(node[23]), 'AlgorithmEPIE':int(node[24]), 'AlgorithmDM':int(node[25]), 'DetectorDistance':int(node[26]), 'PixelSize':int(node[27]), 'CenterY':int(node[28]), 'CenterX':int(node[29]), 'DiffractionSize':int(node[30]), 'Rotation':int(node[31]), 'GPU_ID':int(node[32]), 'ProbeSize':int(node[33]), 'ProbeModes':int(node[34]), 'Threshold':int(node[35]), 'Iterations':int(node[36])  } ]
		return ret_list

	def get_all_jobs(self):
		return self._get_jobs_(SELECT_ALL_JOBS)

	def get_all_unprocessed_jobs(self):
		return self._get_jobs_(SELECT_ALL_UNPROCESSED_JOBS)

	def get_all_unprocessed_jobs_for_pn_id(self, pn_id):
		return self._get_jobs_(SELECT_ALL_UNPROCESSED_JOBS_FOR_PN_ID, {'Process_Node_Id': pn_id})

	def get_all_unprocessed_jobs_for_any_node(self):
		return self._get_jobs_(SELECT_ALL_UNPROCESSED_JOBS_ANY_NODE)

	def get_all_unprocessed_and_processing_jobs(self):
		return self._get_jobs_(SELECT_ALL_UNPROCESSED_AND_PROCESSING_JOBS)

	def get_all_processing_jobs(self):
		return self._get_jobs_(SELECT_ALL_PROCESSING_JOBS)

	def get_all_finished_jobs(self, limit=None):
		if limit == None:
			return self._get_jobs_(SELECT_ALL_FINISHED_JOBS)
		else:
			return self._get_jobs_(SELECT_ALL_FINISHED_JOBS_LIMIT + str(limit))

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
		try:
			print 'updating job', job_dict
			con = sql.connect(self.uri)
			cur = con.cursor()
			cur.execute(UPDATE_JOB_BY_ID, job_dict)
			con.commit()
			return True
		except:
			return False

	def update_job_pn(self, job_id, pn_id):
		try:
			con = sql.connect(self.uri)
			cur = con.cursor()
			cur.execute(UPDATE_JOB_PN, {'Process_Node_Id':pn_id, 'Id':job_id})
			con.commit()
			return True
		except:
			return False

	def delete_job_by_id(self, job_id):
		try:
			con = sql.connect(self.uri)
			cur = con.cursor()
			cur.execute(DELETE_JOB_BY_ID, {'Id': int(job_id)})
			con.commit()
			return True
		except:
			return False

	def save(self, entry):
		pass

	def reset_process_nodes_status(self):
		con = sql.connect(self.uri)
		cur = con.cursor()
		cur.execute(RESET_PN_STATUS)
		con.commit()

if __name__ == '__main__':
	from datetime import datetime
	proc_node = { 'ComputerName':'Comp1', 'NumThreads':1, 'Hostname':'127.0.0.2', 'Port':8080, 'Status':'idle', 'Heartbeat':datetime.now(), 'ProcessCpuPercent':0.0, 'ProcessMemPercent':1.0, 'SystemCpuPercent':2.0, 'SystemMemPercent':10.0, 'SystemSwapPercent':0.0}
	proc_node2 = { 'ComputerName':'Comp2', 'NumThreads':2, 'Hostname':'127.0.0.3', 'Port':8080, 'Status':'idle', 'Heartbeat':datetime.now(), 'ProcessCpuPercent':0.0, 'ProcessMemPercent':1.0, 'SystemCpuPercent':2.0, 'SystemMemPercent':10.0, 'SystemSwapPercent':0.0}
	xrf_job1 = { 'DataPath':'/data/mapspy1/', 'ProcMask':1, 'Version':'1.00', 'DetectorElements':1, 'MaxFilesToProc':1, 'MaxLinesToProc':11, 'QuickAndDirty':0, 'XRF_Bin':0, 'NNLS':0, 'XANES_Scan':0, 'DetectorToStartWith':0, 'BeamLine':'XRF', 'Standards':'', 'DatasetFilesToProc': 'all', 'Priority':5, 'Status':0, 'StartProcTime':datetime.ctime(datetime.now()), 'FinishProcTime':0, 'Log_Path': '', 'Process_Node_Id': -1, 'Emails':'', 'Is_Live_Job':0 }
	xrf_job2 = { 'DataPath':'/data/mapspy2/', 'ProcMask':4, 'Version':'1.00', 'DetectorElements':1, 'MaxFilesToProc':1, 'MaxLinesToProc':11, 'QuickAndDirty':0, 'XRF_Bin':0, 'NNLS':0, 'XANES_Scan':0, 'DetectorToStartWith':0, 'BeamLine':'XRF', 'Standards':'', 'DatasetFilesToProc': 'all', 'Priority':10, 'Status':0, 'StartProcTime':datetime.ctime(datetime.now()), 'FinishProcTime':0, 'Log_Path': '', 'Process_Node_Id': -1, 'Emails':'', 'Is_Live_Job':0 }
	xrf_job3 = { 'DataPath':'/data/mapspy3/', 'ProcMask':8, 'Version':'1.00', 'DetectorElements':1, 'MaxFilesToProc':1, 'MaxLinesToProc':11, 'QuickAndDirty':0, 'XRF_Bin':0, 'NNLS':0, 'XANES_Scan':0, 'DetectorToStartWith':0, 'BeamLine':'XRF', 'Standards':'', 'DatasetFilesToProc': 'all', 'Priority':7, 'Status':0, 'StartProcTime':datetime.ctime(datetime.now()), 'FinishProcTime':0, 'Log_Path': '', 'Process_Node_Id': -1, 'Emails':'', 'Is_Live_Job':1 }
	pty_job1 = { 'DataPath':'/data/pty1/', 'Version':'1.00', 'BeamLine':'PTY', 'DatasetFilesToProc': 'all', 'Priority':7, 'Status':0, 'StartProcTime':datetime.ctime(datetime.now()), 'FinishProcTime':0, 'Log_Path': '', 'Process_Node_Id': -1, 'Emails':'', 'CalcSTXM':0, 'AlgorithmEPIE':1, 'AlgorithmDM':0, 'DetectorDistance':2, 'PixelSize':3, 'CenterY':4, 'CenterX':5, 'DiffractionSize':6, 'Rotation':0, 'GPU_ID':1, 'ProbeSize':7, 'ProbeModes':8, 'Threshold':9, 'Iterations':10 }
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
	xrf_job1['Id'] = db.insert_job(xrf_job1)
	pty_job1['Id'] = db.insert_job(pty_job1)
	xrf_job2['Id'] = db.insert_job(xrf_job2)
	print db.get_jobs_by_status(0)
	xrf_job3['Id'] = 10
	db.insert_job_with_id(xrf_job3)
	print db.get_jobs_by_status(0)
	print ' '
	print db.get_all_jobs()
	#print db.get_all_jobs()
