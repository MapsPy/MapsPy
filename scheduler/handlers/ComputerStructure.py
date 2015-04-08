
from multiprocessing import Queue

class ComputerStructure:
	def __init__(self):
		self.computer_name = ''
		self.job_queue = Queue()
		self.cpu_count = 0
		self.process_state = 'unknown'

