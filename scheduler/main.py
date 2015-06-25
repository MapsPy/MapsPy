'''
Created on May 2015

@author: Arthur Glowacki, Argonne National Laboratory

Copyright (c) 2013, Stefan Vogt, Argonne National Laboratory 
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


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import sys
import Settings
from Scheduler import Scheduler
from ProcessNode import ProcessNode
import cherrypy
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
		scheduler = Scheduler(settings)
		scheduler.run()
	elif role == 'process_node':
		process_node = ProcessNode(settings)
		process_node.run()
	else:
		print 'Unknown role! exiting!'

if __name__ == '__main__':
	main()
