'''
Created on Nov 29, 2011

@author: Mirna Lerotic, 2nd Look Consulting
		 http://www.2ndlookconsulting.com/


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

from __future__ import division
from scipy.optimize import leastsq

import numpy as np
from pylab import *

#-----------------------------------------------------------------------------	  
def leastsqbound(func, x0, bounds, args=(), **kw):
	"""
	Constrained multivariant Levenberg-Marquard optimization
	<OWNER> = 
	

	Copyright (c) Jonathan J. Helmus (jjhelmus@gmail.com)
	All rights reserved.
	
	Redistribution and use in source and binary forms, with or without modification, 
	are permitted provided that the following conditions are met:
	
		Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
		Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
		Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
	PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
	SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
	ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

	Minimize the sum of squares of a given function using the 
	Levenberg-Marquard algorithm. Contraints on parameters are inforced using 
	variable transformations as described in the MINUIT User's Guide by
	Fred James and Matthias Winkler.

	Parameters:

	* func		functions to call for optimization.
	* x0		Starting estimate for the minimization.
	* bounds	(min,max) pair for each element of x, defining the bounds on
				that parameter.  Use None for one of min or max when there is
				no bound in that direction.
	* args		Any extra arguments to func are places in this tuple.

	Returns: (x,{cov_x,infodict,mesg},ier)

	Return is described in the scipy.optimize.leastsq function.  x and con_v  
	are corrected to take into account the parameter transformation, infodic 
	is not corrected.

	Additional keyword arguments are passed directly to the 
	scipy.optimize.leastsq algorithm. 

	"""
	# check for full output
	if "full_output" in kw and kw["full_output"]:
		full=True
	else:
		full=False

	# convert x0 to internal variables
	i0 = external2internal(x0,bounds)

	# perfrom unconstrained optimization using internal variables
	r = leastsq(err,i0,args=(bounds,func,args),**kw)

	# unpack return convert to external variables and return
	if full:
		xi,cov_xi,infodic,mesg,ier = r
		xe = internal2external(xi,bounds)
		if not cov_xi == None:
			cov_xe = i2e_cov_x(xi,bounds,cov_xi)
		else:
			cov_xe = None
		# XXX correct infodic 'fjac','ipvt', and 'qtf' 
		return xe,cov_xe,infodic,mesg,ier 

	else:
		xi,ier = r
		xe = internal2external(xi,bounds)
		return xe,ier
	
	
#-----------------------------------------------------------------------------		  
def internal2external_grad(xi, bounds):
	""" 
	Calculate the internal to external gradiant
	Calculates the partial of external over internal
	"""
	
	ge = np.empty_like(xi)

	for i,(v,bound) in enumerate(zip(xi,bounds)):
		
		a = bound[0]	# minimum
		b = bound[1]	# maximum

		if a == None and b == None:    # No constraints
			ge[i] = 1.0

		elif b == None:		 # only min
			ge[i] = v/np.sqrt(v**2+1)

		elif a == None:		 # only max
			ge[i] = -v/np.sqrt(v**2+1)

		else:		# both min and max
			ge[i] = (b-a)*np.cos(v)/2.

	return ge

#-----------------------------------------------------------------------------
def i2e_cov_x(xi, bounds, cov_x):

	grad = internal2external_grad(xi,bounds)
	grad = np.atleast_2d(grad)
	
	return np.dot(grad.T,grad)*cov_x

#-----------------------------------------------------------------------------	  
def internal2external(xi, bounds):
	""" Convert a series of internal variables to external variables"""
	
	xe = np.empty_like(xi)

	for i,(v,bound) in enumerate(zip(xi,bounds)):
		
		a = bound[0]	# minimum
		b = bound[1]	# maximum

		if a == None and b == None:    # No constraints
			xe[i] = v

		elif b == None:		 # only min
			xe[i] = a-1.+np.sqrt(v**2.+1.)

		elif a == None:		 # only max
			xe[i] = b+1.-np.sqrt(v**2.+1.)

		else:		# both min and max
			xe[i] = a+((b-a)/2.)*( np.sin(v)+1.)

	return xe

#-----------------------------------------------------------------------------	  
def external2internal(xe, bounds):
	""" Convert a series of external variables to internal variables"""

	xi = np.empty_like(xe)

	for i,(v,bound) in enumerate(zip(xe,bounds)):
		
		a = bound[0]	# minimum
		b = bound[1]	# maximum

		if a == None and b == None: # No constraints
			xi[i] = v

		elif b == None:		# only min
			xi[i] = np.sqrt( (v-a+1.)**2.-1 )

		elif a == None:		# only max
			xi[i] = np.sqrt( (b-v+1.)**2.-1 )

		else:	# both min and max
			xi[i] = np.arcsin( (2.*(v-a)/(b-a))-1.)

	return xi

#-----------------------------------------------------------------------------	  
def err(p, bounds, efunc, args):
	
	pe = internal2external(p,bounds)	
	return efunc(pe,*args)

#--------------------------------------------------------------------------------
def calc_cov_x(infodic, p):
	"""
	Calculate cov_x from fjac, ipvt and p as is done in leastsq 
	"""

	fjac = infodic['fjac']
	ipvt = infodic['ipvt']
	n = len(p)

	# adapted from leastsq function in scipy/optimize/minpack.py
	perm = np.take(np.eye(n),ipvt-1,0)
	r = np.triu(np.transpose(fjac)[:n,:])
	R = np.dot(r,perm)
	#try:
	cov_x = np.linalg.inv(np.dot(np.transpose(R),R))
	#except LinAlgError:
	#	 cov_x = None
	return cov_x

