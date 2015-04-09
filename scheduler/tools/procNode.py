import sys
import requests
from datetime import datetime
import time

def addProcessNode(s, url, name):
	r = s.post( url, params={'ComputerName':name,'NumThreads':2,'Status':'BootUp', 'Heartbeat':datetime.now()} )
	print r.status_code,'::',r.text

def updateNode(s, url, name, status):
	r = s.put( url, params={'ComputerName':name,'NumThreads':2,'Status':status, 'Heartbeat':datetime.now()} )
	print r.status_code,'::',r.text

def getNodes(s, url):
	r = s.get(url)
	print r.status_code,'::',r.text

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print 'python addProcessNode.py <url> <post/put/get> <name> <status>'
		sys.exit(1)
	url = sys.argv[1]
	url = url+'/process_node'	
	action = sys.argv[2]
	s = requests.Session()
	try:
		if action == 'post':
			addProcessNode(s, url, sys.argv[3])
		elif action == 'put':
			updateNode(s, url, sys.argv[3], sys.argv[4])
		elif action == 'get':
			getNodes(s, url)
	except:
		print 'Error connecting to',url


