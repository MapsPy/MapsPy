import ConfigParser


#Keys read from ConfigParser are all lower cased!

'''SECTIONS'''
SECTION_SERVER              = 'Server'
SERVER_ROLE                 = 'role'
SERVER_HOSTNAME             = 'hostname'
SERVER_PORT                 = 'port'
SERVER_SCHEDULER_HOSTNAME   = 'scheduler_hostname'
SERVER_SCHEDULER_PORT       = 'scheduler_port'
SERVER_SMTP_ADDRESS			= 'smtp_address'
SERVER_FROM_ADDRESS			= 'from_address'
SERVER_MAIL_USERNAME		= 'mail_username'
SERVER_MAIL_PASSWORD		= 'mail_password'
SCHEDULE_FILES_PATH			= 'schedule_files_path'
SERVER_KEYS = [SERVER_ROLE, SERVER_HOSTNAME, SERVER_PORT, SERVER_SMTP_ADDRESS, SERVER_FROM_ADDRESS,
				SERVER_MAIL_USERNAME, SERVER_MAIL_PASSWORD, SCHEDULE_FILES_PATH]

'''PROCESS NODE Keys'''
SECTION_PROCESS_NODE        	= 'Process_Node'
PROCESS_NODE_NAME           	= 'computer_name'
PROCESS_NODE_THREADS        	= 'num_threads'
PROCESS_NODE_DATABASE_NAME  	= 'database_name'
PROCESS_NODE_PATH_ALIAS			= 'path_alias'
PROCESS_NODE_SOFTWARE_LIST		= 'software_list'
PROCESS_NODE_KEYS = [PROCESS_NODE_NAME, PROCESS_NODE_THREADS, PROCESS_NODE_DATABASE_NAME, PROCESS_NODE_SOFTWARE_LIST]

'''MONITOR keys'''
SECTION_MONITOR = 'Monitor'
MONITOR_JOBS_PATH           = 'jobs_path'
MONITOR_PROCESSING_PATH     = 'processing_path'
MONITOR_FINISHED_INFO_PATH  = 'finished_info_path'
MONITOR_DONE_PATH           = 'done_path'
MONITOR_COMPUTER_NAME       = 'computer_name'
MONITOR_DIR_ALIAS           = 'dir_alias'
MONITOR_CHECK_INTERVAL      = 'check_interval'
MONITOR_KEYS = [MONITOR_JOBS_PATH, MONITOR_PROCESSING_PATH, MONITOR_FINISHED_INFO_PATH, MONITOR_DONE_PATH,
				MONITOR_COMPUTER_NAME, MONITOR_DIR_ALIAS, MONITOR_CHECK_INTERVAL]

SECTION_JOB_DIR_ROOTS = 'JobDirRoots'

'''LISTS'''
SECTIONS = [SECTION_SERVER, SECTION_PROCESS_NODE, SECTION_MONITOR]


'''CLASSES'''


class SettingsIO:
	def __init__(self):
		self.config = ConfigParser.ConfigParser()
		self.settingDicts = dict()

	def load(self, filename):
		print 'Settings loading file:', filename
		self.config.read(filename)
		for sec in self.config.sections():
			self.settingDicts[sec] = self.__get_sect_dict__(sec)

	def getSetting(self, section):
		if section in self.settingDicts:
			return self.settingDicts[section]
		else:
			return dict()

	def checkSectionKeys(self, section_name, keys):
		for k in keys:
			sect = self.getSetting(section_name)
			if not k in sect:
				print 'Could not find Key:', k, 'in server settings'
				return False
		return True

	def __get_sect_dict__(self, section):
		dict1 = {}
		options = self.config.options(section)
		for option in options:
			try:
				dict1[option] = self.config.get(section, option)
				if dict1[option] == -1:
					print("skip: %s" % option)
			except:
				print("exception on %s!" % option)
				dict1[option] = None
		return dict1

if __name__ == '__main__':
	# test
	s = SettingsIO()
	s.load('settings.ini')
	print 'all sections',s.settingDicts.keys()
	schedDict = s.getSetting('Scheduler')
	print 'scheduler section', schedDict.keys()
