'''
Created on 29 May 2015

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

import time


def open_file_with_retry(filename, open_attr, retry_amt=5, retry_sleep=0.1, retry_sleep_inc=1.1):
	file = None
	for i in range(retry_amt):
		try:
			file = open(filename, open_attr)
			break
		except:
			print 'failed to open ',filename, ' Retry in ',retry_sleep,' seconds...'
			time.sleep(retry_sleep)
			retry_sleep += retry_sleep_inc
	return file


def call_function_with_retry(func_ptr, retry_amt = 5, retry_sleep = 0.1, retry_sleep_inc = 1.1, *args):
	retVal = None
	print args
	for i in range(retry_amt):
		try:
			retVal = func_ptr(*args[0])
			break
		except:
			print 'failed to call func_ptr',str(func_ptr), ' Retry in ',retry_sleep,' seconds...'
			time.sleep(retry_sleep)
			retry_sleep += retry_sleep_inc
	return retVal

if __name__ == '__main__':
	#tests
	
	f = open_file_with_retry('abc', 'r')
	if f != None:
		f.close()
	
	#f = call_function_with_retry(h5py.File, 5, 0.1, 1.1,('test.h5','r'))
	#print f.keys()
