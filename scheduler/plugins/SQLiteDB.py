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
import Constants
import json

DELETE_JOB_BY_ID = 'DELETE FROM Jobs WHERE Id=:Id'

SQL_ORDER_BY_JOB_ID_DESC = ' ORDER BY ' + Constants.TABLE_JOBS + '.' + Constants.JOB_ID + ' DESC'
SQL_ORDER_BY_JOB_PRIORITY_ASC = ' ORDER BY ' + Constants.TABLE_JOBS + '.' + Constants.JOB_PRIORITY + ' ASC'
SQL_ORDER_BY_JOB_STATUS_DESC = ' ORDER BY ' + Constants.TABLE_JOBS + '.' + Constants.JOB_STATUS + ' DESC'

SQL_UNASSIGNED_JOB = ' ' + Constants.TABLE_JOBS + '.' + Constants.JOB_PROCESS_NODE_ID + '=-1'

SQL_WHERE_JOB_ID_IS = ' WHERE ' + Constants.TABLE_JOBS + '.' + Constants.JOB_ID + '='
SQL_WHERE_JOB_STATUS_IS = ' WHERE ' + Constants.TABLE_JOBS + '.' + Constants.JOB_STATUS + '='
SQL_WHERE_JOB_STATUS_NEW = ' WHERE ' + Constants.TABLE_JOBS + '.' + Constants.JOB_STATUS + '=' + str(Constants.JOB_STATUS_NEW)
SQL_WHERE_JOB_STATUS_PROCESSING = ' WHERE ' + Constants.TABLE_JOBS + '.' + Constants.JOB_STATUS + '=' + str(Constants.JOB_STATUS_PROCESSING)
SQL_WHERE_JOB_STATUS_COMPLETED = ' WHERE ' + Constants.TABLE_JOBS + '.' + Constants.JOB_STATUS + '>=' + str(Constants.JOB_STATUS_COMPLETED)
SQL_WHERE_JOB_STATUS_NEW_OR_PROCESSING = ' WHERE ' + Constants.TABLE_JOBS + '.' + Constants.JOB_STATUS + '<=' + str(Constants.JOB_STATUS_PROCESSING)
SQL_WHERE_UNPROCESSED_AND_PROCESSING_JOBS = SQL_WHERE_JOB_STATUS_NEW_OR_PROCESSING + SQL_ORDER_BY_JOB_STATUS_DESC

TABLES = { Constants.TABLE_PROCESS_NODES:{ 'Name':Constants.TABLE_PROCESS_NODES,
										   'Columns':[ {'Key':Constants.PROCESS_NODE_ID, 'Type':'INTEGER', 'Prop':'PRIMARY KEY'},
													  {'Key':Constants.PROCESS_NODE_COMPUTERNAME, 'Type':'TEXT', 'Prop':''},
													  {'Key':Constants.PROCESS_NODE_NUM_THREADS, 'Type':'INTEGER', 'Prop':''},
													  {'Key':Constants.PROCESS_NODE_HOSTNAME, 'Type':'TEXT', 'Prop':''},
													  {'Key':Constants.PROCESS_NODE_PORT, 'Type':'INTEGER', 'Prop':''},
													  {'Key':Constants.PROCESS_NODE_STATUS, 'Type':'TEXT', 'Prop':''},
													  {'Key':Constants.PROCESS_NODE_HEARTBEAT, 'Type':'TIMESTAMP', 'Prop':''},
													  {'Key':Constants.PROCESS_NODE_PROCESS_CPU_PERCENT, 'Type':'REAL', 'Prop':''},
													  {'Key':Constants.PROCESS_NODE_PROCESS_MEM_PERCENT, 'Type':'REAL', 'Prop':''},
													  {'Key':Constants.PROCESS_NODE_SYSTEM_CPU_PERCENT, 'Type':'REAL', 'Prop':''},
													  {'Key':Constants.PROCESS_NODE_SYSTEM_MEM_PERCENT, 'Type':'REAL', 'Prop':''},
													  {'Key':Constants.PROCESS_NODE_SYSTEM_SWAP_PERCENT, 'Type':'REAL', 'Prop':''},
													  {'Key':Constants.PROCESS_NODE_SUPPORTED_SOFTWARE, 'Type':'TEXT', 'Prop':''},
													   ]
		   								},
			Constants.TABLE_JOBS:{ 'Name':Constants.TABLE_JOBS,
									'Columns':[ {'Key':Constants.JOB_ID, 'Type':'INTEGER', 'Prop':'PRIMARY KEY'},
									  {'Key':Constants.JOB_EXPERIMENT, 'Type':'TEXT', 'Prop':''},
									  {'Key':Constants.JOB_DATA_PATH, 'Type':'TEXT', 'Prop':''},
									  {'Key':Constants.JOB_VERSION, 'Type':'TEXT', 'Prop':''},
									  {'Key':Constants.JOB_BEAM_LINE, 'Type':'TEXT', 'Prop':''},
									  {'Key':Constants.JOB_DATASET_FILES_TO_PROC, 'Type':'TEXT', 'Prop':''},
									  {'Key':Constants.JOB_PRIORITY, 'Type':'INTEGER', 'Prop':''},
									  {'Key':Constants.JOB_STATUS, 'Type':'INTEGER', 'Prop':''},
									  {'Key':Constants.JOB_START_PROC_TIME, 'Type':'TIMESTAMP', 'Prop':''},
									  {'Key':Constants.JOB_FINISH_PROC_TIME, 'Type':'TIMESTAMP', 'Prop':''},
									  {'Key':Constants.JOB_LOG_PATH, 'Type':'TEXT', 'Prop':''},
									  {'Key':Constants.JOB_PROCESS_NODE_ID, 'Type':'INTEGER', 'Prop':''},
									  {'Key':Constants.JOB_EMAILS, 'Type':'TEXT', 'Prop':''},
									  {'Key':Constants.JOB_ARGS, 'Type':'TEXT', 'Prop':''} ]
								   }
		   }


def Gen_Create_Table(table):
	ret_str = 'CREATE TABLE IF NOT EXISTS '
	ret_str += table['Name']
	ret_str += '('
	for col in table['Columns']:
		ret_str += col['Key'] + ' ' + col['Type'] + ' ' + col['Prop'] + ', '
	#  remove last ','
	ret_str = ret_str[0:len(ret_str) - 2]
	ret_str += ');'
	return ret_str


def Gen_Drop(table_name):
	return 'DROP TABLE IF EXISTS ' + table_name


def Gen_Insert_Into_Table(table, insert_dict):
	ret_str = 'INSERT INTO '
	ret_str += table['Name']
	ret_str += '('
	values_str = ' )VALUES( '
	for col in table['Columns']:
		if insert_dict.has_key(col['Key']):
			ret_str += col['Key'] + ', '
			if col['Type'] == 'TEXT' or col['Type'] == 'TIMESTAMP':
				values_str += "'" + str(insert_dict[col['Key']]) + "', "
			else:
				values_str += str(insert_dict[col['Key']]) + ', '
	#  remove last ','
	ret_str = ret_str[0:len(ret_str) - 2]
	values_str = values_str[0:len(values_str) - 2]
	ret_str += values_str
	ret_str += ');'
	return ret_str


def Gen_Update_Table(table, update_dict, by_statement):
	ret_str = 'UPDATE '
	ret_str += table['Name']
	ret_str += ' SET '
	cnt = 0
	for col in table['Columns']:
		if update_dict.has_key(col['Key']):
			cnt += 1
			if col['Type'] == 'TEXT' or col['Type'] == 'TIMESTAMP':
				ret_str += col['Key'] + "='" + str(update_dict[col['Key']]) + "', "
			else:
				ret_str += col['Key'] + '=' + str(update_dict[col['Key']]) + ', '
	#  remove last ','
	if cnt > 0:
		ret_str = ret_str[0:len(ret_str) - 2]
	ret_str += ' WHERE ' + by_statement
	return ret_str


def Gen_Select_All_Cols(table):
	node_idxs = {}
	i = 0
	ret_str = 'SELECT '
	for col in table['Columns']:
		node_idxs[i] = col
		i += 1
		ret_str += col['Key'] + ', '
	#  remove last ','
	ret_str = ret_str[0:len(ret_str) - 2]
	ret_str += ' FROM '
	ret_str += table['Name']
	return ret_str, node_idxs


def Gen_Select_Count_By_Id(table, key, value):
	ret_str = 'SELECT Count('
	ret_str += table['Name'] + '.' +str(key) + ')'
	ret_str += ' FROM ' + table['Name'] + ' WHERE ' + table['Name'] + '.' + str(key) + '='
	ret_str += "'" + str(value) + "'"
	return ret_str



class SQLiteDB:

	def __init__(self, db_name='MapsPy.db'):
		self.uri = db_name

	def create_tables(self, drop=False):
		con = sql.connect(self.uri)
		cur = con.cursor()
		for table in TABLES.itervalues():
			if drop:
				drop_str = Gen_Drop(table['Name'])
				cur.executescript(drop_str)
			create_str = Gen_Create_Table(table)
			cur.execute(create_str)
		con.commit()

	def insert_process_node(self, proc_node_dict):
		#first check if this process node exists
		sql_statement = Gen_Select_Count_By_Id(TABLES[Constants.TABLE_PROCESS_NODES], Constants.PROCESS_NODE_COMPUTERNAME, proc_node_dict[Constants.PROCESS_NODE_COMPUTERNAME])
		con = sql.connect(self.uri)
		cur = con.cursor()
		cur.execute(sql_statement)
		con.commit()
		row = cur.fetchone()
		if row[0] == 0:
			print 'insert',proc_node_dict
			insert_str = Gen_Insert_Into_Table(TABLES[Constants.TABLE_PROCESS_NODES], proc_node_dict)
			cur.execute(insert_str, proc_node_dict)
		else:
			print 'update', proc_node_dict
			str_by_statement = Constants.PROCESS_NODE_COMPUTERNAME + '=:' + Constants.PROCESS_NODE_COMPUTERNAME
			update_str = Gen_Update_Table(TABLES[Constants.TABLE_PROCESS_NODES], proc_node_dict, str_by_statement)
			cur.execute(update_str, proc_node_dict)
		con.commit()

	def insert_job(self, job_dict):
		print 'insert job', job_dict
		if Constants.JOB_ARGS in job_dict:
			#jenc = json.JSONEncoder()
			#job_dict[Constants.JOB_ARGS] = jenc.encode(job_dict[Constants.JOB_ARGS])
			if type(job_dict[Constants.JOB_ARGS]) == type({}):
				job_dict[Constants.JOB_ARGS] = json.dumps(job_dict[Constants.JOB_ARGS])
		sql_statement = Gen_Insert_Into_Table(TABLES[Constants.TABLE_JOBS], job_dict)
		con = sql.connect(self.uri)
		cur = con.cursor()
		cur.execute(sql_statement)
		con.commit()
		return cur.lastrowid

	def get_process_node_by_name(self, proc_node_name):
		where_statement = ' WHERE ' + Constants.PROCESS_NODE_COMPUTERNAME + '="' + proc_node_name + '"'
		nodes = self._get_where(TABLES[Constants.TABLE_PROCESS_NODES], where_statement)
		if len(nodes) > 0:
			return nodes[0]
		return None

	def get_process_node_by_id(self, proc_node_id):
		where_statement = ' WHERE ' + Constants.PROCESS_NODE_ID + '=' + str(proc_node_id)
		nodes = self._get_where(TABLES[Constants.TABLE_PROCESS_NODES], where_statement)
		if len(nodes) > 0:
			return nodes[0]
		return None

	def get_all_process_nodes(self):
		return self._get_where(TABLES[Constants.TABLE_PROCESS_NODES], ' ')

	def _get_where(self, table, where_statement, opt_dict=None):
		sql_statement, node_idxs = Gen_Select_All_Cols(table)
		sql_statement += ' ' + where_statement
		con = sql.connect(self.uri)
		cur = con.cursor()
		if opt_dict == None:
			cur.execute(sql_statement)
		else:
			cur.execute(sql_statement, opt_dict)
		con.commit()
		all_nodes = cur.fetchall()
		ret_list = []
		for node in all_nodes:
			row = {}
			for i in range(len(node)):
				key = node_idxs[i]['Key']
				row[key] = node[i]
			row['DT_RowId'] = row['Id']
			if table['Name'] == Constants.TABLE_JOBS and not row[Constants.JOB_ARGS] == None:
				#jdec = json.JSONDecoder()
				#a = jdec.decode(row[Constants.JOB_ARGS])
				#a = json.loads(row[Constants.JOB_ARGS])
				row[Constants.JOB_ARGS] = json.loads(row[Constants.JOB_ARGS])
			ret_list += [row]
		return ret_list

	def get_all_jobs(self):
		return self._get_where(TABLES[Constants.TABLE_JOBS], ' ')

	def get_all_unprocessed_jobs(self):
		return self._get_where(TABLES[Constants.TABLE_JOBS], SQL_WHERE_JOB_STATUS_NEW)

	def get_all_unprocessed_jobs_for_pn_id(self, pn_id):
		where_clause = SQL_WHERE_JOB_STATUS_NEW_OR_PROCESSING + ' AND ' + Constants.TABLE_JOBS + '.' + Constants.JOB_PROCESS_NODE_ID + '=' + str(pn_id) + SQL_ORDER_BY_JOB_PRIORITY_ASC
		return self._get_where(TABLES[Constants.TABLE_JOBS], where_clause)

	def get_all_unprocessed_jobs_for_any_node(self):
		where_clause = SQL_WHERE_JOB_STATUS_NEW + ' AND ' + SQL_UNASSIGNED_JOB
		return self._get_where(TABLES[Constants.TABLE_JOBS], where_clause)

	def get_all_unprocessed_and_processing_jobs(self):
		return self._get_where(TABLES[Constants.TABLE_JOBS], SQL_WHERE_UNPROCESSED_AND_PROCESSING_JOBS)

	def get_all_processing_jobs(self):
		return self._get_where(TABLES[Constants.TABLE_JOBS], SQL_WHERE_JOB_STATUS_PROCESSING)

	def get_all_finished_jobs(self, limit=None):
		where_clause = SQL_WHERE_JOB_STATUS_COMPLETED + SQL_ORDER_BY_JOB_ID_DESC
		if not limit == None:
			where_clause + ' LIMIT ' + str(limit)
		return self._get_where(TABLES[Constants.TABLE_JOBS], where_clause)

	def get_job(self, job_id):
		where_clause = SQL_WHERE_JOB_ID_IS + str(job_id)
		jobs = self._get_where(TABLES[Constants.TABLE_JOBS], where_clause)
		if len(jobs) > 0:
			return jobs[0]
		return None

	def get_jobs_by_status(self, status):
		where_clause = SQL_WHERE_JOB_STATUS_IS + str(status)
		jobs = self._get_where(TABLES[Constants.TABLE_JOBS], where_clause)
		if len(jobs) > 0:
			return jobs[0]
		return None

	def update_job(self, job_dict):
		try:
			print 'updating job', job_dict
			saved_args = {}
			if Constants.JOB_ARGS in job_dict:
				saved_args = job_dict[Constants.JOB_ARGS]
				if type(job_dict[Constants.JOB_ARGS]) == type({}):
					job_dict[Constants.JOB_ARGS] = json.dumps(job_dict[Constants.JOB_ARGS])
			where_clause = Constants.JOB_ID + '=' + str(job_dict[Constants.JOB_ID])
			sql_statement = Gen_Update_Table(TABLES[Constants.TABLE_JOBS], job_dict, where_clause)
			con = sql.connect(self.uri)
			cur = con.cursor()
			cur.execute(sql_statement)
			con.commit()
			if Constants.JOB_ARGS in job_dict:
				job_dict[Constants.JOB_ARGS] = saved_args
			return True
		except:
			return False

	def update_job_pn(self, job_id, pn_id):
		try:
			where_clause = Constants.JOB_ID + '=' + str(job_id)
			sql_statement = Gen_Update_Table(TABLES[Constants.TABLE_JOBS], {Constants.JOB_PROCESS_NODE_ID:pn_id}, where_clause)
			con = sql.connect(self.uri)
			cur = con.cursor()
			cur.execute(sql_statement)
			con.commit()
			return True
		except:
			return False

	def delete_job_by_id(self, job_id):
		try:
			con = sql.connect(self.uri)
			cur = con.cursor()
			cur.execute(DELETE_JOB_BY_ID, {Constants.JOB_ID: int(job_id)})
			con.commit()
			return True
		except:
			return False

	def save(self, entry):
		pass

	def reset_process_nodes_status(self):
		update_dict = {Constants.PROCESS_NODE_STATUS:Constants.PROCESS_NODE_STATUS_OFFLINE,
					 Constants.PROCESS_NODE_PROCESS_CPU_PERCENT:0.0,
					 Constants.PROCESS_NODE_PROCESS_MEM_PERCENT:0.0,
					 Constants.PROCESS_NODE_SYSTEM_CPU_PERCENT:0.0,
					 Constants.PROCESS_NODE_SYSTEM_MEM_PERCENT:0.0,
					 Constants.PROCESS_NODE_SYSTEM_SWAP_PERCENT:0.0
					 }
		where_clause = Constants.PROCESS_NODE_ID + '>0;'
		sql_statement = Gen_Update_Table(TABLES[Constants.TABLE_PROCESS_NODES], update_dict, where_clause)
		con = sql.connect(self.uri)
		cur = con.cursor()
		cur.execute(sql_statement)
		con.commit()

if __name__ == '__main__':
	from datetime import datetime
	proc_node = { 'ComputerName':'Comp1', 'NumThreads':1, 'Hostname':'127.0.0.2', 'Port':8080, 'Status':'idle', 'Heartbeat':datetime.now(), 'ProcessCpuPercent':0.0, 'ProcessMemPercent':1.0, 'SystemCpuPercent':2.0, 'SystemMemPercent':10.0, 'SystemSwapPercent':0.0}
	proc_node2 = { 'ComputerName':'Comp2', 'NumThreads':2, 'Hostname':'127.0.0.3', 'Port':8080, 'Status':'idle', 'Heartbeat':datetime.now(), 'ProcessCpuPercent':0.0, 'ProcessMemPercent':1.0, 'SystemCpuPercent':2.0, 'SystemMemPercent':10.0, 'SystemSwapPercent':0.0}
	xrf_job1 = { 'DataPath':'/data/mapspy1/', 'Experiment':'XRF', 'Version':'9.00', 'BeamLine':'2-ID-E', 'DatasetFilesToProc':'all', 'Priority':5, 'Status':0, 'StartProcTime':datetime.ctime(datetime.now()), 'FinishProcTime':0, 'Log_Path': '', 'Process_Node_Id': -1, 'Emails':'', 'Args':{'ProcMask':1, 'DetectorElements':1, 'MaxFilesToProc':1, 'MaxLinesToProc':11, 'QuickAndDirty':0, 'XRF_Bin':0, 'NNLS':0, 'XANES_Scan':0, 'DetectorToStartWith':0, 'Is_Live_Job':0, 'Standards':''} }
	xrf_job2 = { 'DataPath':'/data/mapspy2/', 'Experiment':'XRF', 'Version':'9.00', 'BeamLine':'2-ID-E', 'DatasetFilesToProc':'all', 'Priority':5, 'Status':1, 'StartProcTime':datetime.ctime(datetime.now()), 'FinishProcTime':0, 'Log_Path': '', 'Process_Node_Id': -1, 'Emails':'', 'Args':{'ProcMask':4, 'DetectorElements':1, 'MaxFilesToProc':1, 'MaxLinesToProc':11, 'QuickAndDirty':0, 'XRF_Bin':0, 'NNLS':0, 'XANES_Scan':0, 'DetectorToStartWith':0, 'Is_Live_Job':0, 'Standards':''} }
	xrf_job3 = { 'DataPath':'/data/mapspy3/', 'Experiment':'XRF', 'Version':'9.00', 'BeamLine':'2-ID-E', 'DatasetFilesToProc':'all', 'Priority':5, 'Status':2, 'StartProcTime':datetime.ctime(datetime.now()), 'FinishProcTime':0, 'Log_Path': '', 'Process_Node_Id': -1, 'Emails':'', 'Args':{'ProcMask':8, 'DetectorElements':1, 'MaxFilesToProc':1, 'MaxLinesToProc':11, 'QuickAndDirty':0, 'XRF_Bin':0, 'NNLS':0, 'XANES_Scan':0, 'DetectorToStartWith':0, 'Is_Live_Job':0, 'Standards':''} }
	pty_job1 = { 'DataPath':'/data/pty1/', 'Experiment':'PTY', 'Version':'1.00', 'BeamLine':'2-ID-D', 'DatasetFilesToProc': 'all', 'Priority':7, 'Status':0, 'StartProcTime':datetime.ctime(datetime.now()), 'FinishProcTime':0, 'Log_Path': '', 'Process_Node_Id': -1, 'Emails':'', 'Args':{'CalcSTXM':0, 'AlgorithmEPIE':1, 'AlgorithmDM':0, 'DetectorDistance':2, 'PixelSize':3, 'CenterY':4, 'CenterX':5, 'DiffractionSize':6, 'Rotation':0, 'GPU_ID':1, 'ProbeSize':7, 'ProbeModes':8, 'Threshold':9, 'Iterations':10 } }
	db = SQLiteDB('TestDatabase.db')
	db.create_tables(True)
	db.insert_process_node(proc_node)
	proc_node = db.get_process_node_by_name(proc_node['ComputerName'])
	db.insert_process_node(proc_node2)
	proc_node2 = db.get_process_node_by_name(proc_node2['ComputerName'])
	print db.get_process_node_by_name(proc_node['ComputerName'])
	print ' '
	print db.get_process_node_by_id(proc_node2['Id'])
	print ' '
	proc_node['Status'] = 'Offline'
	proc_node['Heartbeat'] = datetime.now()
	db.insert_process_node(proc_node)
	print ' '
	print db.get_process_node_by_id(proc_node['Id'])
	db.reset_process_nodes_status()
	print db.get_all_process_nodes()
	print '-------------------------------'
	#add job
	xrf_job1['Id'] = db.insert_job(xrf_job1)
	pty_job1['Id'] = db.insert_job(pty_job1)
	xrf_job2['Id'] = db.insert_job(xrf_job2)
	print db.get_all_jobs()
	db.update_job_pn(xrf_job1['Id'], proc_node['Id'])
	db.update_job_pn(xrf_job2['Id'], proc_node2['Id'])
	print ' '
	print db.get_jobs_by_status(0)
	print ' '
	xrf_job3['Id'] = 10
	db.insert_job(xrf_job3)
	print db.get_jobs_by_status(0)
	print ' '
	print db.get_all_jobs()
	print ' '
	xrf_job3['Status'] = 0
	db.update_job(xrf_job3)
	print db.get_all_unprocessed_jobs()
	db.delete_job_by_id(xrf_job2['Id'])
	db.delete_job_by_id(xrf_job3['Id'])
	print ' '
	print db.get_all_jobs()
	print db.get_job(xrf_job1['Id'])
	print db.get_all_unprocessed_jobs_for_pn_id(proc_node['Id'])
	print db.get_all_unprocessed_jobs_for_any_node()
	print db.get_all_unprocessed_and_processing_jobs()
	print db.get_all_processing_jobs()
