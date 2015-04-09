import sqlite3 as sql

CREATE_PROCESS_NODES_TABLE_STR = 'CREATE TABLE ProcessNodes(Id INTEGER PRIMARY KEY, ComputerName TEXT, NumThreads INTEGER, Status TEXT, Heartbeat TIMESTAMP);'
CREATE_JOBS_TABLE_STR = 'CREATE TABLE Jobs(Id INTEGER PRIMARY KEY, DataPath TEXT, ProcMask INTEGER, Version TEXT, DetectorElements INTEGER, MaxFilesToProc INTEGER, MaxLineToProc INTEGER, QuickAndDirty INTEGER, XRF_Bin INTEGER, NNLS INTEGER, XANES_Scan INTEGER, DetectorToStartWith INTEGER, BeamLine TEXT, Standards TEXT);'
CREATE_JOB_QUEUE_TABLE_STR = 'CREATE TABLE JobQueue(Id INTEGER PRIMARY KEY, JobId INTEGER, PnId INTEGER, StartTime TIMESTAMP, StopStime TIMESTAMP, FOREIGN KEY(JobId) REFERENCES Jobs(Id), FOREIGN KEY(PnId) REFERENCES ProcessNodes(Id));'

DROP_PROCESS_NODES_STR = 'DROP TABLE IF EXISTS ProcessNodes;'
DROP_JOBS_STR = 'DROP TABLE IF EXISTS Jobs;'
DROP_JOB_QUEUE_STR = 'DROP TABLE IF EXISTS JobQueue;'

INSERT_PROCESS_NODE = 'INSERT INTO ProcessNodes (ComputerName, NumThreads, Status, Heartbeat) VALUES(:ComputerName, :NumThreads, :Status, :Heartbeat)'

UPDATE_PROCESS_NODE_BY_ID = 'UPDATE ProcessNodes SET ComputerName=:ComputerName NumThreads=:NumThreads Status=:Status Heartbeat=:Heartbeat WHERE Id=:Id'
UPDATE_PROCESS_NODE_BY_NAME = 'UPDATE ProcessNodes SET NumThreads=:NumThreads, Status=:Status, Heartbeat=:Heartbeat WHERE ComputerName=:ComputerName'

SELECT_ALL_PROCESS_NODES = 'SELECT ComputerName, NumThreads, Status, Heartbeat FROM ProcessNodes'
SELECT_PROCESS_NODE_BY_NAME = 'SELECT Id, ComputerName, NumThreads, Status, Heartbeat FROM ProcessNodes WHERE ComputerName=:ComputerName'


class SQLiteDB:
	def __init__(self, db_name='MapsPy.db'):
		self.uri = db_name

	def create_tables(self):
		con = sql.connect(self.uri)
		cur = con.cursor()
		cur.execute(DROP_PROCESS_NODES_STR)
		cur.execute(DROP_JOBS_STR)
		cur.execute(DROP_JOB_QUEUE_STR)
		cur.execute(CREATE_PROCESS_NODES_TABLE_STR)
		cur.execute(CREATE_JOBS_TABLE_STR)
		cur.execute(CREATE_JOB_QUEUE_TABLE_STR)
		con.commit()

	def insert_process_node(self, proc_node):
		#expected dictionary with (ComputerName, NumThreads, Status, Heartbeat)
		#first check if this process node exists
		con = sql.connect(self.uri)
		cur = con.cursor()
		cur.execute(SELECT_PROCESS_NODE_BY_NAME, proc_node)
		con.commit()
		row = cur.fetchone()
		if row == None:
			print 'insert',proc_node
			cur.execute(INSERT_PROCESS_NODE, proc_node)
		else:
			print 'update', proc_node
			cur.execute(UPDATE_PROCESS_NODE_BY_NAME, proc_node)
		con.commit()

	def get_process_node(self, proc_node_name):
		con = sql.connect(self.uri)
		proc_node = {'ComputerName':proc_node_name}
		cur = con.cursor()
		cur.execute(SELECT_PROCESS_NODE_BY_NAME, proc_node)
		con.commit()
		return cur.fetchone()

	def get_all_process_nodes(self):
		con = sql.connect(self.uri)
		cur = con.cursor()
		cur.execute(SELECT_ALL_PROCESS_NODES)
		con.commit()
		return cur.fetchall()

	def save(self, entry):
		print 'saving',entry

if __name__ == '__main__':
	import datetime
	proc_node = {'Id':123, 'ComputerName':'Comp1', 'NumThreads':1, 'Status':'idle', 'Heartbeat':datetime.datetime.now()}
	proc_node2 = {'Id':13, 'ComputerName':'Comp2', 'NumThreads':2, 'Status':'idle', 'Heartbeat':datetime.datetime.now()}
	db = SQLiteDB('TestDatabase.db')
	db.create_tables()
	db.insert_process_node(proc_node)
	db.insert_process_node(proc_node2)
	proc_node['Status'] = 'Offline'
	proc_node['Heartbeat'] = datetime.datetime.now()
	db.insert_process_node(proc_node)
	#proc_node2 = db.get_process_node(proc_node['ComputerName'])
	#print type(proc_node2)
	#print proc_node2
	import json
	result = db.get_all_process_nodes()
	jenc = json.JSONEncoder()
	print jenc.encode(result)
	
	

