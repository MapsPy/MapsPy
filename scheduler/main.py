import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import sys
import Settings
from Scheduler import Scheduler
from ProcessNode import ProcessNode

settings_filename = 'settings.ini'

def main():
	global settings_filename
	if len(sys.argv) > 1:
		settings_filename = sys.argv[1]
	settings = Settings.SettingsIO()
	settings.load(settings_filename)
	if settings.checkSectionKeys(Settings.SECTION_SERVER, Settings.SERVER_KEYS) == False:
		print 'Error: Could not find all settings in ',settings_filename
		print 'Please add the following keys to',settings_filename,'under the section',Settings.SECTION_SERVER
		for key in Settings.SERVER_KEYS:
			print key
		sys.exit(1)
	serverSettings = settings.getSetting(Settings.SECTION_SERVER)
	role = str(serverSettings[Settings.SERVER_ROLE])
	print 'Role =',role
	if role == 'scheduler':
		scheduler = Scheduler(serverSettings)
		scheduler.run()
	elif role == 'process_node':
		process_node = ProcessNode(settings)
		process_node.run()
	else:
		print 'Unknown role! exiting!'

if __name__ == '__main__':
	main()
