'''
Created on Nov 2015

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

import logging
import logging.handlers

class RestBase:
	def __init__(self):
		pass

	def run(self):
		raise RuntimeError("Need to Implement run method")

	def stop(self):
		raise RuntimeError("Need to Implement stop method")

	def _setup_logging_(self, log, logtype, logname, stream_to_console=True):
		maxBytes = getattr(log, "rot_maxBytes", 20971520) # 20Mb
		backupCount = getattr(log, "rot_backupCount", 10)
		fname = getattr(log, logtype, logname)
		h = logging.handlers.RotatingFileHandler(fname, 'a', maxBytes, backupCount)
		h.setLevel(logging.DEBUG)
		formatter = logging.Formatter('%(asctime)s | %(levelname)s | PID[%(process)d] | %(funcName)s(): %(message)s')
		log.setLevel(logging.DEBUG)
		h.setFormatter(formatter)
		log.addHandler(h)
		if stream_to_console:
			ch = logging.StreamHandler()
			ch.setFormatter(formatter)
			ch.setLevel(logging.WARNING)
			log.addHandler(ch)