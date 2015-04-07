import cherrypy
from cherrypy.process import wspbus, plugins

class BaseDatabasePlugin(plugins.SimplePlugin):
    def __init__(self, bus, db_klass):
        plugins.SimplePlugin.__init__(self, bus)
        self.db = db_klass()

    def start(self):
        self.bus.log('Starting up DB access')
        self.bus.subscribe("db-save", self.save_it)

    def stop(self):
        self.bus.log('Stopping down DB access')
        self.bus.unsubscribe("db-save", self.save_it)

    def save_it(self, entity):
        self.db.save(entity)

