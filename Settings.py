import ConfigParser


#Keys read from ConfigParser are all lower cased!

'''SECTIONS'''
SECTION_SERVER = 'Server'
SERVER_ROLE     = 'role'
SERVER_HOSTNAME = 'hostname'
SERVER_PORT     = 'port'
SERVER_KEYS = [SERVER_ROLE, SERVER_HOSTNAME, SERVER_PORT]


'''MONITOR keys'''
SECTION_MONITOR = 'Monitor'
MONITOR_JOBS_PATH          = 'jobs_path'
MONITOR_PROCESSING_PATH    = 'processing_path'
MONITOR_FINISHED_INFO_PATH = 'finished_info_path'
MONITOR_DONE_PATH          = 'done_path'
MONITOR_COMPUTER_NAME      = 'computer_name'
MONITOR_DIR_ALIAS          = 'dir_alias'
MONITOR_CHECK_INTERVAL     = 'check_interval'
MONITOR_KEYS = [MONITOR_JOBS_PATH, MONITOR_PROCESSING_PATH, MONITOR_FINISHED_INFO_PATH, MONITOR_DONE_PATH, MONITOR_COMPUTER_NAME, MONITOR_DIR_ALIAS, MONITOR_CHECK_INTERVAL]

'''LISTS'''
#SECTIONS = [SECTION_SERVER]
SECTIONS = [SECTION_MONITOR]


'''CLASSES'''
class SettingsIO:
	def __init__(s):
		s.config = ConfigParser.ConfigParser()
		s.settingDicts = dict()

	def load(s, filename):
		print 'Settings loading file:', filename
		s.config.read(filename)
		for sec in s.config.sections():
			s.settingDicts[sec] = s.__get_sect_dict__(sec)

	def getSetting(s, section):
		if s.settingDicts.has_key(section):
			return s.settingDicts[section]
		else:
			return dict()

	def checkSectionKeys(s, section_name, keys):
		for k in keys:
			sect = s.getSetting(section_name)
			if sect.has_key(k) == False:
				print 'Could not find Key:',k,'in server settings'
				return False
		return True


	def __get_sect_dict__(s, section):
	    dict1 = {}
	    options = s.config.options(section)
	    for option in options:
	        try:
	            dict1[option] = s.config.get(section, option)
	            if dict1[option] == -1:
	                DebugPrint("skip: %s" % option)
	        except:
	            print("exception on %s!" % option)
	            dict1[option] = None
	    return dict1


if __name__ == '__main__':
	#test
	s = Settings()
	s.load('settings.ini')
	print 'all sections',s.settingDicts.keys()
	schedDict = s.getSetting('Scheduler')
	print 'scheduler section', schedDict.keys()
