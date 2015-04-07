import os, os.path
import sys

import cherrypy

from StatusStruct import StatusStruct

#import Settings

from handlers.StringGen import StringGenerator, StringGeneratorWebService
from plugins.BaseDatabasePlugin import BaseDatabasePlugin
from plugins.SQLiteDB import SQLiteDB

running = True
settings_filename = 'settings.ini'

def main():
   '''
   settings = Settings.SettingsIO()
   settings.load(settings_filename)
   if settings.checkSectionKeys(Settings.SECTION_SERVER, Settings.SERVER_KEYS) == False:
      print 'Error: Could not find all server settings in ',settings_filename
      print 'Shutting down... '
      sys.exit(1)
   serverSettings = settings.getSetting(Settings.SECTION_SERVER)
   HostName = serverSettings[Settings.SERVER_HOSTNAME]
   Port = int(serverSettings[Settings.SERVER_PORT])
   '''
   #conf = serverSettings[Settings.SERVER_CHERRYPY_CONFIG]
   scheduleStruct = StatusStruct()
   conf = {
      '/': {
         'tools.sessions.on': True,
         'tools.staticdir.root': os.path.abspath(os.getcwd())
      },
      '/generator': {
         'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
         'tools.response_headers.on': True,
         'tools.response_headers.headers': [('Content-Type', 'text/plain')],
      },
      '/static': {
         'tools.staticdir.on': True,
         'tools.staticdir.dir': './public'
      }
   }
   webapp = StringGenerator()
   BaseDatabasePlugin(cherrypy.engine, SQLiteDB).subscribe()
   webapp.generator = StringGeneratorWebService()
   cherrypy.quickstart(webapp, '/', conf)

if __name__ == '__main__':
   main()
