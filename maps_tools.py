'''
Created on Apr 23, 2013

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
import os

import numpy as np
import matplotlib as mplot
import logging

	
#----------------------------------------------------------------------   
def plot_spectrum(info_elements, spectra=0, i_spectrum=0, add_plot_spectra=0, add_plot_names=0, ps=0, fitp=0, perpix=0,	filename='', outdir='', logger=logging.getLogger('plot')):
	logger.info('ploting spectrum')
	
	from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
	mplot.rcParams['pdf.fonttype'] = 42
	
	fontsize = 9
	mplot.rcParams['font.size'] = fontsize
	
	colortable = []

	colortable.append((0., 0., 0.))  # ; black
	colortable.append((1., 0., 0.))  # ; red
	colortable.append((0., 1., 0.))  # ; green
	colortable.append((0., 0., 1.))  # ; blue
	colortable.append((0., 1., 1.))  # ; turquois
	colortable.append((1., 0., 1.))  # ; magenta
	colortable.append((1., 1., 0.))  # ; yellow
	colortable.append((0.7, 0.7, 0.7))  # ; light grey
	colortable.append((1., 0.8, 0.75))  # ; flesh
	colortable.append((0.35, 0.35, 0.35))  # ; dark grey
	colortable.append((0., 0.5, 0.5))  # ; sea green
	colortable.append((1., 0., 0.53))  # ; pink-red
	colortable.append((0., 1., 0.68))  # ; bluegreen
	colortable.append((1., 0.5, 0.))  # ; orange
	colortable.append((0., 0.68, 1.))  # ; another blue
	colortable.append((0.5, 0., 1.))  # ; violet
	colortable.append((1., 1., 1.))  # ; white
	
	droplist_scale = 0
	png = 0
	if ps == 0:
		png = 2
	if spectra == 0:
		logger.info('spectra are 0, returning')
		return 

	if filename == '':
		have_name = 0
		for isp in range(len(spectra)):
			if spectra[isp].name != '': 
				have_name=1
	
		if have_name == 0 : return 
		filename = spectra[0].name 
	
	if (png > 0) or (ps > 0):
		if png > 0:

			dpi = 100
			canvas_xsize_in = 900./dpi
			canvas_ysize_in = 700./dpi
			fig = mplot.figure.Figure(figsize=(canvas_xsize_in, canvas_ysize_in), dpi=dpi, edgecolor=None)
			canvas = FigureCanvas(fig)
			fig.add_axes()
			axes = fig.gca()
			for child in axes.get_children():
				if isinstance(child, mplot.spines.Spine):
					child.set_color((0., 0., 0.))
			#axes.set_axis_bgcolor(background_color)
			ya = axes.yaxis					 
			xa = axes.xaxis							 
			ya.set_tick_params(labelcolor=(0., 0., 0.))
			ya.set_tick_params(color=(0., 0., 0.))
			xa.set_tick_params(labelcolor=(0., 0., 0.))
			xa.set_tick_params(color=(0., 0., 0.))

		if ps > 0:
			ps_filename = 'ps_'+filename+'.pdf'
			if ps_filename == '' : return 
			eps_plot_xsize = 8.
			eps_plot_ysize = 6.
								
			fig = mplot.figure.Figure(figsize =(eps_plot_xsize, eps_plot_ysize))
			canvas = FigureCanvas(fig)
			fig.add_axes()
			axes = fig.gca()
				
			file_ps = os.path.join(outdir, ps_filename)

	if spectra[i_spectrum].used_chan > 0 : 
		
		this_axis_calib = i_spectrum

		xaxis = (np.arange(spectra[this_axis_calib].used_chan))**2*spectra[this_axis_calib].calib['quad'] + \
				np.arange(spectra[this_axis_calib].used_chan)*spectra[this_axis_calib].calib['lin'] + \
				spectra[this_axis_calib].calib['off']	  
		xtitle = 'energy [keV]'

		xmin = fitp.g.xmin *0.5
		xmax = fitp.g.xmax + (fitp.g.xmax-fitp.g.xmin)*0.10

		wo_a = np.where(xaxis > xmax)[0]
		if len(wo_a) > 0 :
			wo_xmax = np.amin(wo_a) 
		else:
			wo_xmax = spectra[i_spectrum].used_chan*8./10.
		wo_b = np.where(xaxis < xmin)[0]
		if len(wo_b) >0 :
			wo_xmin = np.amax(wo_b) 
		else:
			wo_xmin = 0

		wo = np.where(spectra[i_spectrum].data[wo_xmin:wo_xmax+1] > 0.)
		if len(wo) > 0 : 
			ymin = np.amin(spectra[i_spectrum].data[wo+wo_xmin])*0.9 
		else:
			ymin = 0.1
		if perpix > 0 : ymin = 0.001
		if len(wo_a) > 0 :	
			ymax = np.amax(spectra[i_spectrum].data[wo+wo_xmin]*1.1) 
		else: 
			ymax = np.amax(spectra[i_spectrum].data)
		# make sure ymax is larger than ymin, so as to avoid a crash during plotting
		if ymax <= ymin : ymax = ymin+0.001

		yanno = (1.01+0.04*(1-droplist_scale)) * ymax
		yanno_beta = (1.07+0.53*(1-droplist_scale)) * ymax
		if droplist_scale == 0 :
			yanno_below = 0.8*ymin 
		else:
			yanno_below = ymin -(ymax-ymin)*.04
		yanno_lowest = (0.8+0.15*(1-(1-droplist_scale)))*ymin
		this_spec = spectra[i_spectrum].data[0:spectra[i_spectrum].used_chan]
		wo = np.where(this_spec <= 0)[0]
		if len(wo) > 0:
			this_spec[wo] = ymin
		norm_font_y_size = 10.
		
		plot1 = axes.semilogy(xaxis, this_spec, linewidth=1.0)
		axes.set_xlabel(xtitle)
		axes.set_ylabel('counts')
		axes.set_xlim((xmin, xmax))
		axes.set_ylim((ymin, ymax))
		
		axes.set_position([0.10,0.18,0.85,0.75])
		
		axes.text( -0.10, -0.12, spectra[i_spectrum].name, transform = axes.transAxes)

		if add_plot_spectra.any(): 
			size = add_plot_spectra.shape
			if len(size) == 2 : 
				#for k = size[2]-1, 0, -1 :
				for k in np.arange(size[1]-1, -1, -1):
					plot2 = axes.semilogy(xaxis, add_plot_spectra[:, k], color = colortable[1+k], linewidth=1.0) 

					if k <= 2 :
						axes.text( -0.10+0.4+0.2*k, -0.12, add_plot_names[k],color = colortable[1+k], transform = axes.transAxes)
					if (k >= 3) and (k <= 6) :
						axes.text( -0.10+0.2*(k-3), -0.15, add_plot_names[k],color = colortable[1+k], transform = axes.transAxes)  
					if (k >= 7) :
						axes.text( -0.10+0.2*(k-7), -0.18, add_plot_names[k],color = colortable[1+k], transform = axes.transAxes)

				# plot background next to last
				plot3 = axes.semilogy(xaxis, add_plot_spectra[:, 2], color = colortable[1+2], linewidth=1.0)
				# plot fit last
				plot4 = axes.semilogy(xaxis, add_plot_spectra[:, 0], color = colortable[1+0], linewidth=1.0)

		# plot xrf ticks   
		element_list = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 25, 26, 27, 28, 29, 30, 32, 33, 35])-1
		x_positions = []
		for i in range(len(info_elements)): x_positions.append(info_elements[i].xrf['ka1'])
		color = 2

		local_ymax = np.array([1.03, 1.15, 1.3])*ymax
		local_ymin = ymax*0.9
		for k in range(len(element_list)): 
			i = element_list[k]
			line=mplot.lines.Line2D([x_positions[i],x_positions[i]], [local_ymin,local_ymax[(i-int(i/3)*3)]] ,color=colortable[color])
			line.set_clip_on(False)
			axes.add_line(line)				   
			axes.text( x_positions[i], local_ymax[(i-int(i/3)*3)] , info_elements[i].name, ha='center', va='bottom', color = colortable[color])

		if (png > 0) or (ps > 0) :

			if png > 0:
				axes.text(0.97, -0.23, 'mapspy', transform = axes.transAxes)
				if (png == 1) or (png == 2) :  
					image_filename = filename+'.png'
					logger.info('saving tools png %s', os.path.join(outdir,image_filename))
					fig.savefig(os.path.join(outdir, image_filename), dpi=dpi, edgecolor=None)
				   
				if ps > 0:
					fig.savefig(file_ps)


# -----------------------------------------------------------------------------

def maps_nnls_single(x, input_data, fitmatrix_reduced, n_mca_channels):
	
	fitmatrix_r_trans = fitmatrix_reduced.T
	sz = fitmatrix_r_trans.shape
	m = sz[1]
	n_nn = sz[0]
	these_counts = np.zeros((1,x.size))
	n_relev_channels = min(x.size, n_mca_channels)
	these_counts[0,0:n_relev_channels] = input_data[0:n_relev_channels,0]

	F_nn = []
	G_nn = np.arange(n_nn).T
	x_nn = np.zeros((1, n_nn))

	Atb = np.dot(these_counts, fitmatrix_reduced)

	y = -Atb[0]
	p = 10
	ninf = n_nn+1
	noready = 1
	while noready:
		xF = x_nn[F_nn]
		yG = y[G_nn]
		# make sure xF and yG are defined
		if xF > 0:
			wo_xF = np.where(xF < 0)[0]
		else :
			wo_xF = []
		if yG.size > 0:
			wo_yG = np.where(yG < 0) [0]
		else:
			wo_yG = []
		if len(wo_xF) == 0 and len(wo_yG) == 0: 
			x_nn = np.zeros((1, n_nn))
			y = np.zeros((1, n_nn))
			x_nn[F_nn] = xF
			break					
		else: 
			H1 = []
			H2 = []
			if xF.size > 0 : 
				wo = np.where(xF < 0)[0]
				if len(wo) > 0 : H1 = F_nn[wo]
			if yG.size > 0 : 
				wo = np.where(yG < 0)[0]
				if len(wo) > 0 : H2 = G_nn[wo]
			
			H = list(set(H1) | set(H2)) 
			if len(H) < ninf : 
				ninf = len(H)
			else:
				if p > 1 :
					p = p-1
				else:	  
					r = np.amax(H)
					if r in H1:
						H1 = [r]	   
						H2 = []		   
					else:
						H1 = []		   
						H2 = [r]	   

			F_nn = list(set(set(F_nn) & set(H1)) | set(H2))
			G_nn = list(set(set(G_nn) & set(H2)) | set(H1)) 
		 
		if (len(F_nn) > 0) : 
			AF = fitmatrix_r_trans[F_nn, :]
		else: 
			AF = []
		AG = fitmatrix_r_trans[G_nn, :]
		if (len(F_nn) > 0) :
			xF = np.linalg.lstsq(AF, these_counts)
		if (len(F_nn) > 0) :
			x_nn[F_nn] = xF 
		if (AG.size == 0) :
			continue
		if (len(F_nn) > 0) :
			#yG = np.dot(AG.T,(np.dot(AF,xF)-these_counts))
			yG = np.dot((np.dot(xF,AF)-these_counts),AG.T)
		else:
			#yG = np.dot(AG.T,(0.-these_counts))
			yG = np.dot((0.-these_counts),AG.T)
		y[G_nn] = yG			   

	result = x_nn[0]
	
	return result
	
	
#-----------------------------------------------------------------------------	  
def maps_nnls_line(data_line, xsize, fitmatrix_reduced, n_mca_channels, elements_to_use, element_lookup_in_reduced, n_rows):
	

	results_line = np.zeros((n_rows, len(elements_to_use)))
	
	fitmatrix_r_trans = fitmatrix_reduced.T
	sz = fitmatrix_r_trans.shape
	n_nn = sz[0]
	
	these_counts = np.zeros((1,xsize))
	n_relev_channels = min(xsize, n_mca_channels)
	
	for j_temp in range(n_rows):

		these_counts[0, 0:n_relev_channels] = data_line[0:n_relev_channels, j_temp]
	
		F_nn = []
		G_nn = np.arange(n_nn).T
		x_nn = np.zeros((1, n_nn))
		H1 = []
		H2 = []
		H = []

		Atb = np.dot(these_counts, fitmatrix_reduced)
		y = -Atb
		p = 3
		ninf = n_nn + 1
		noready = 1

		while noready:
			xF = x_nn[0, F_nn]
			yG = y[0, G_nn]

			# make sure xF and yG are defined
			if xF.size > 0:
				wo_xF = np.where(xF < 0)[0]
			else :
				wo_xF = []
			if yG.size > 0:
				wo_yG = np.where(yG < 0)[0]
			else:
				wo_yG = []
			if len(wo_xF) == 0 and len(wo_yG) == 0: 
				x_nn = np.zeros((1, n_nn))
				y = np.zeros((1, n_nn))
				x_nn[0, F_nn] = xF
				break					
			else: 
				H1 = []
				H2 = []
				if xF.size > 0:
					wo = np.where(xF < 0)
					if len(wo[0]) > 0:
						F_nn = np.array(F_nn)
						H1 = F_nn[wo]
				if yG.size > 0 : 
					wo = np.where(yG < 0)
					if len(wo[0]) > 0:
						G_nn = np.array(G_nn)
						H2 = G_nn[wo]
				
				H = list(set(H1) | set(H2)) 
				if len(H) < ninf:
					ninf = len(H)
				else:
					if p > 1:
						p = p - 1
					else:	  
						r = np.amax(H)
						if r in H1:
							H1 = [r]	   
							H2 = []		   
						else:
							H1 = []		   
							H2 = [r]	   
				F_nn = list(set(set(F_nn) - set(H1)) | set(H2))
				G_nn = list(set(set(G_nn) - set(H2)) | set(H1)) 
			 
			if len(F_nn) > 0:
				AF = fitmatrix_r_trans[F_nn, :]
			else: 
				AF = []
			AG = fitmatrix_r_trans[G_nn, :]
			if len(F_nn) > 0:
				xF = np.linalg.lstsq(AF.T, these_counts[0, :])[0]
			if len(F_nn) > 0:
				x_nn[0, F_nn] = xF
			if AG.size == 0:
				continue
			if len(F_nn) > 0:
				# yG = np.dot(AG.T,(np.dot(AF,xF)-these_counts))
				yG = np.dot((np.dot(xF, AF) - these_counts), AG.T)
			else:
				# yG = np.dot(AG.T,(0.-these_counts))
				yG = np.dot((0. - these_counts), AG.T)

			y[0, G_nn] = yG[0, :]

		# result = x_nn[0]
	
		for mm in range(len(elements_to_use)):				   
			if element_lookup_in_reduced[mm] != -1 :
				results_line[j_temp, mm] = x_nn[0, element_lookup_in_reduced[mm]]

	return results_line



def congrid(a, newdims, logger, method='linear', centre=False, minusone=False):
	'''Arbitrary resampling of source array to new dimension sizes.
	Currently only supports maintaining the same number of dimensions.
	To use 1-D arrays, first promote them to shape (x,1).

	Uses the same parameters and creates the same co-ordinate lookup points
	as IDL''s congrid routine, which apparently originally came from a VAX/VMS
	routine of the same name.

	method:
	neighbour - closest value from original data
	nearest and linear - uses n x 1-D interpolations using
						 scipy.interpolate.interp1d
	(see Numerical Recipes for validity of use of n 1-D interpolations)
	spline - uses ndimage.map_coordinates

	centre:
	True - interpolation points are at the centres of the bins
	False - points are at the front edge of the bin

	minusone:
	For example- inarray.shape = (i,j) & new dimensions = (x,y)
	False - inarray is resampled by factors of (i/x) * (j/y)
	True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
	This prevents extrapolation one element beyond bounds of input array.
	'''
	if not a.dtype in [np.float64, np.float32]:
		a = np.cast[float](a)

	m1 = np.cast[int](minusone)
	ofs = np.cast[int](centre) * 0.5
	old = np.array(a.shape)
	ndims = len(a.shape)
	if len(newdims) != ndims:
		logger.error("[congrid] dimensions error. This routine currently only support rebinning to the same number of dimensions.")
		return None
	newdims = np.asarray( newdims, dtype=float )
	dimlist = []

	if method == 'neighbour':
		for i in range( ndims ):
			base = np.indices(newdims)[i]
			dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
							* (base + ofs) - ofs )
		cd = np.array( dimlist ).round().astype(int)
		newa = a[list( cd )]
		return newa

	elif method in ['nearest','linear']:
		# calculate new dims
		for i in range( ndims ):
			base = np.arange( newdims[i] )
			dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
							* (base + ofs) - ofs )
		# specify old dims
		olddims = [np.arange(i, dtype = np.float) for i in list( a.shape )]

		# first interpolation - for ndims = any
		import scipy.interpolate
		mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
		newa = mint( dimlist[-1] )

		trorder = [ndims - 1] + range( ndims - 1 )
		for i in range( ndims - 2, -1, -1 ):
			newa = newa.transpose( trorder )

			mint = scipy.interpolate.interp1d( olddims[i], newa, kind=method )
			newa = mint( dimlist[i] )

		if ndims > 1:
			# need one more transpose to return to original dimensions
			newa = newa.transpose( trorder )

		return newa
	elif method in ['spline']:
		oslices = [ slice(0,j) for j in old ]
		oldcoords = np.ogrid[oslices]
		nslices = [ slice(0,j) for j in list(newdims) ]
		newcoords = np.mgrid[nslices]

		newcoords_dims = range(np.ndim(newcoords))
		#make first index last
		newcoords_dims.append(newcoords_dims.pop(0))
		newcoords_tr = newcoords.transpose(newcoords_dims)
		# makes a view that affects newcoords

		newcoords_tr += ofs

		deltas = (np.asarray(old) - m1) / (newdims - m1)
		newcoords_tr *= deltas

		newcoords_tr -= ofs
		import scipy.ndimage
		newa = scipy.ndimage.map_coordinates(a, newcoords)
		return newa
	else:
		logger.error("Congrid error: Unrecognized interpolation type.\nCurrently only \'neighbour\', \'nearest\',\'linear\', and \'spline\' are supported.")
		return None



