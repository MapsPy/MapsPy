import sqlite3 as sql

CREATE_PROCESS_NODES_TABLE_STR = 'CREATE TABLE IF NOT EXISTS ProcessNodes(Id INTEGER PRIMARY KEY, ComputerName TEXT, NumThreads INTEGER, Hostname TEXT, Port INTEGER, Status TEXT, Heartbeat TIMESTAMP);'
CREATE_JOBS_TABLE_STR = 'CREATE TABLE IF NOT EXISTS Jobs(Id INTEGER PRIMARY KEY, DataPath TEXT, ProcMask INTEGER, Version TEXT, DetectorElements INTEGER, MaxFilesToProc INTEGER, MaxLineToProc INTEGER, QuickAndDirty INTEGER, XRF_Bin INTEGER, NNLS INTEGER, XANES_Scan INTEGER, DetectorToStartWith INTEGER, BeamLine TEXT, Standards TEXT);'
CREATE_JOB_QUEUE_TABLE_STR = 'CREATE TABLE IF NOT EXISTS JobQueue(Id INTEGER PRIMARY KEY, JobId INTEGER, PnId INTEGER, StartTime TIMESTAMP, StopStime TIMESTAMP, FOREIGN KEY(JobId) REFERENCES Jobs(Id), FOREIGN KEY(PnId) REFERENCES ProcessNodes(Id));'

DROP_PROCESS_NODES_STR = 'DROP TABLE IF EXISTS ProcessNodes;'
DROP_JOBS_STR = 'DROP TABLE IF EXISTS Jobs;'
DROP_JOB_QUEUE_STR = 'DROP TABLE IF EXISTS JobQueue;'

INSERT_PROCESS_NODE = 'INSERT INTO ProcessNodes (ComputerName, NumThreads, Hostname, Port, Status, Heartbeat) VALUES(:ComputerName, :NumThreads, :Hostname, :Port, :Status, :Heartbeat)'
INSERT_JOB = 'INSERT INTO Jobs (DataPath, ProcMask, Version, DetectorElements, MaxFilesToProc, MaxLineToProc, QuickAndDirty, XRF_Bin, NNLS, XANES_Scan, DetectorToStartWith, BeamLine, Standards) VALUES(:DataPath, :ProcMask, :Version, :DetectorElements, :MaxFilesToProc, :MaxLineToProc, :QuickAndDirty, :XRF_Bin, :NNLS, :XANES_Scan, :DetectorToStartWith, :BeamLine, :Standards)'


UPDATE_PROCESS_NODE_BY_ID = 'UPDATE ProcessNodes SET ComputerName=:ComputerName NumThreads=:NumThreads Hostname=:Hostname, Port=:Port Status=:Status Heartbeat=:Heartbeat WHERE Id=:Id'
UPDATE_PROCESS_NODE_BY_NAME = 'UPDATE ProcessNodes SET NumThreads=:NumThreads, Hostname=:Hostname, Port=:Port, Status=:Status, Heartbeat=:Heartbeat WHERE ComputerName=:ComputerName'

SELECT_ALL_PROCESS_NODES = 'SELECT ComputerName, NumThreads, Status, Heartbeat FROM ProcessNodes'
SELECT_PROCESS_NODE_BY_NAME = 'SELECT Id, ComputerName, NumThreads, Status, Heartbeat FROM ProcessNodes WHERE ComputerName=:ComputerName'
SELECT_ALL_JOBS = 'SELECT Id, DataPath, ProcMask, Version, DetectorElements, MaxFilesToProc, MaxLineToProc, QuickAndDirty, XRF_Bin, NNLS, XANES_Scan, DetectorToStartWith, BeamLine, Standards FROM Jobs'
SELECT_JOB_BY_ID = 'SELECT DataPath, ProcMask, Version, DetectorElements, MaxFilesToProc, MaxLineToProc, QuickAndDirty, XRF_Bin, NNLS, XANES_Scan, DetectorToStartWith, BeamLine, Standards FROM Jobs WHERE Id=:Id'


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
		print 'keys', job_dict.keys()
		con = sql.connect(self.uri)
		cur = con.cursor()
		cur.execute(INSERT_JOB, job_dict)
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
		for node in all_nodes:
			ret_list += [ {'ComputerName':node[0], 'NumThreads':node[1], 'Status': node[2], 'Heartbeat': node[3]} ]
		return ret_list

	def get_all_jobs(self):
		con = sql.connect(self.uri)
		cur = con.cursor()
		cur.execute(SELECT_ALL_JOBS)
		con.commit()
		#return cur.fetchall()
		all_nodes = cur.fetchall()
		ret_list = []
		for node in all_nodes:
			ret_list += [ {'Id':node[0], 'DataPath':node[1], 'ProcMask': node[2], 'Version': node[3]} ]
		return ret_list

	def get_job(self, job_id):
		con = sql.connect(self.uri)
		cur = con.cursor()
		cur.execute(SELECT_JOBS_BY_ID, {'Id':job_id})
		con.commit()
		return cur.fetchone()

	def save(self, entry):
		print 'saving',entry

if __name__ == '__main__':
	import datetime
	proc_node = { 'ComputerName':'Comp1', 'NumThreads':1, 'Hostname':'127.0.0.2', 'Port':8080, 'Status':'idle', 'Heartbeat':datetime.datetime.now()}
	proc_node2 = { 'ComputerName':'Comp2', 'NumThreads':2, 'Hostname':'127.0.0.3', 'Port':8080, 'Status':'idle', 'Heartbeat':datetime.datetime.now()}
	job1 = { 'DataPath':'/data/mapspy/', 'ProcMask':1, 'Version':'1.00', 'DetectorElements':1, 'MaxFilesToProc':1, 'MaxLineToProc':11, 'QuickAndDirty':0, 'XRF_Bin':0, 'NNLS':0, 'XANES_Scan':0, 'DetectorToStartWith':0, 'BeamLine':'2-ID-E', 'Standards':''}
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
	db.insert_job(job1)
	print db.get_all_jobs()
	
