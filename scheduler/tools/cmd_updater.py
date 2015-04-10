import sys
import requests
import json
from datetime import datetime
import time

def call_post(s, url, payload):
	r = s.post( url, data=json.dumps(payload) )
	print r.status_code,'::',r.text

def call_put(s, url, payload):
	r = s.put( url, data=json.dumps(payload) )
	print r.status_code,'::',r.text

def call_get(s, url):
	r = s.get(url)
	print r.status_code,'::',r.text

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print 'python cmd_updater.py <url> <pn/job> <post/put/get> <name> <status>'
		sys.exit(1)
	action = sys.argv[1]
	url = sys.argv[2]
	atype = sys.argv[3]
	print action, url, atype
	if atype == 'pn':
		url = url+'/process_node'
		if action == 'post':
			payload={'ComputerName':sys.argv[4],'NumThreads':2,'Status':'Bootup', 'Heartbeat':str(datetime.now())}
		elif action == 'put':
			payload={'ComputerName':sys.argv[4],'NumThreads':2,'Status':sys.argv[5], 'Heartbeat':str(datetime.now())}
	elif atype == 'job':
		url = url+'/job'
		payload = { 'DataPath':sys.argv[4], 'ProcMask':1, 'Version':'1.00', 'DetectorElements':int(sys.argv[5]), 'MaxFilesToProc':1, 'MaxLineToProc':11, 'QuickAndDirty':0, 'XRF_Bin':0, 'NNLS':0, 'XANES_Scan':0, 'DetectorToStartWith':0, 'BeamLine':'2-ID-E', 'Standards':''}
	s = requests.Session()
	print payload
	if action == 'post':
		call_post(s, url, payload)
	elif action == 'put':
		call_put(s, url, payload)
	elif action == 'get':
		call_get(s, url)

