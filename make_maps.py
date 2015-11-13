'''
Created on Nov 9, 2011

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

import os
import multiprocessing
import numpy as np
import time
import maps_generate_img_dat
import maps_definitions
import maps_elements
import maps_fit_parameters
import maps_analyze
import maps_calibration
from file_io.file_util import open_file_with_retry
import logging

# ------------------------------------------------------------------------------------------------


def mp_make_maps(log_name, info_elements, main_dict, maps_conf, header, mdafilename, this_detector, use_fit, total_number_detectors,
				quick_dirty, nnls, xrf_bin, max_no_processors_lines):
	logger = logging.getLogger(log_name)
	fHandler = logging.FileHandler(log_name)
	logger.setLevel(logging.DEBUG)
	fHandler.setLevel(logging.DEBUG)
	formatter = logging.Formatter('%(asctime)s | %(levelname)s | PID[%(process)d] | %(funcName)s(): %(message)s')
	fHandler.setFormatter(formatter)
	logger.addHandler(fHandler)
	try:
		makemaps = maps_generate_img_dat.analyze(logger, info_elements, main_dict, maps_conf, beamline=main_dict['beamline'], use_fit=use_fit)
		makemaps.generate_img_dat_threaded(header, mdafilename, this_detector, total_number_detectors, quick_dirty, nnls, max_no_processors_lines, xrf_bin)
	except:
		logger.exception('Exception at make_maps.mp_make_maps()')
		return -1

	return 0

# ------------------------------------------------------------------------------------------------


def main(main_dict, logger, force_fit=0, no_fit=False):
	verbose = True

	maps_intermediate_solution_file = 'maps_intermediate_solution.tmp'
	if verbose:
		logger.info('main structure: %s', main_dict)

	if force_fit == 1:
		use_fit = 1

	if no_fit == True:
		use_fit = 0

	me = maps_elements.maps_elements(logger)
	info_elements = me.get_element_info()

	maps_def = maps_definitions.maps_definitions(logger)
	maps_conf = maps_def.set_maps_definitions(main_dict['beamline'], info_elements, version=main_dict['version'])
	max_no_processors_lines = main_dict['max_no_processors_lines']
	if max_no_processors_lines == -1:
		max_no_processors_lines = multiprocessing.cpu_count() - 1
		logger.info('cpu_count() = %d\n' % multiprocessing.cpu_count())
		logger.info('max_no_processors_lines to fit lines %s', max_no_processors_lines)

	total_number_detectors = main_dict['total_number_detectors']
	quick_dirty = main_dict['quick_dirty']
	if total_number_detectors < 2:
		quick_dirty = 0
	if quick_dirty != 0:
		total_number_detectors = 1

	logger.info('total_number_detectors %s', total_number_detectors)
	temp = multiprocessing.cpu_count()
	no_processors_to_use_files = min(main_dict['max_no_processors_files'], temp)
	logger.info('no_processors_to_use for files %s', no_processors_to_use_files)

	logger.info("main_dict['dataset_files_to_proc'] %s", main_dict['dataset_files_to_proc'])
	if main_dict['dataset_files_to_proc'][0] == 'all':
		filenames = []
		dirList=os.listdir(main_dict['mda_dir'])
		for fname in dirList:
			if fname[-4:] == '.mda':
				filenames.append(fname)
		no_files = len(filenames)

		# If no .mda files were found look for .h5
		dirList = os.listdir(main_dict['img_dat_dir'])
		if no_files == 0:
			for fname in dirList:
				if fname[-3:] == '.h5':
					filenames.append(fname)
	else:
		filenames = main_dict['dataset_files_to_proc']

	no_files = len(filenames)
	if no_files == 0:
		logger.warning('Did not find any .mda files in /mda directory.')
		return False
	else:
		logger.info('Processing files: %s', filenames)
	#filenames_orig = filenames[:]
	#basename, scan_ext= os.path.splitext(filenames[0])

	# determine the number of files, try to determine the scan size for each file, and then sort the files such
	# that in the analysis the biggest files are analysed first.

	# Calculate intermediate result

	detector_number_arr = np.zeros((no_files), dtype=int)
	#detector_number_arr_orig = np.zeros((no_files), dtype=int)
	suffix = ''
	for this_detector in range(main_dict['detector_to_start_with'], total_number_detectors):
		# Look for override files in main.master_dir
		if total_number_detectors > 1:
			suffix = str(this_detector)
			logger.info('suff = %s', suffix)
			maps_overridefile = os.path.join(main_dict['master_dir'], 'maps_fit_parameters_override.txt') + suffix
			try:
				f = open_file_with_retry(maps_overridefile, 'rt', 2, 0.4, 0.2)
				if f == None:
					maps_overridefile = os.path.join(main_dict['master_dir'], 'maps_fit_parameters_override.txt')
				else:
					logger.info('maps override file %s exists', maps_overridefile)
					f.close()
			except:
				# if i cannot find an override file specific per detector, assuming
				# there is a single overall file.
				maps_overridefile = os.path.join(main_dict['master_dir'], 'maps_fit_parameters_override.txt')
		else:
			maps_overridefile = os.path.join(main_dict['master_dir'], 'maps_fit_parameters_override.txt')

		# below is the routine for using matrix math to calculate elemental
		# content with overlap removal
		logger.info('now using matrix math for analysis; calculate intermediate solution for speed now')
		kk = 0

		temp_elementsuse = []
		for item in maps_conf.chan: temp_elementsuse.append(item.use)
		elements_to_use = np.where(np.array(temp_elementsuse) == 1)
		elements_to_use = elements_to_use[0]
		if elements_to_use.size == 0:
			return False

		spectra = maps_def.define_spectra(main_dict['max_spec_channels'], main_dict['max_spectra'], main_dict['max_ICs'], mode='plot_spec')

		fp = maps_fit_parameters.maps_fit_parameters(logger)
		fitp = fp.define_fitp(main_dict['beamline'], info_elements)

		element_pos = np.concatenate((fitp.keywords.kele_pos, fitp.keywords.lele_pos, fitp.keywords.mele_pos))

		fitp.s.use[:] = 1
		fitp.s.val[np.amin(fitp.keywords.kele_pos):np.amax(fitp.keywords.mele_pos)] = 1e-10
		for j in range(fitp.keywords.kele_pos[0]):
			fitp.s.use[j] = fitp.s.batch[j, 1]

		pileup_string = ''
		test_string = ''
		det = kk
		try:
			fitp, test_string, pileup_string = fp.read_fitp(maps_overridefile, info_elements, det)
			if fitp == None:
				logger.error('ERROR - could not read override file: %s' + maps_overridefile)
				return False
		except:
			logger.exception('could not read override file')
			return False

		for jj in range(fitp.g.n_fitp):
			if fitp.s.name[jj] in test_string:
				fitp.s.val[jj] = 1.
				fitp.s.use[jj] = 5

		n_pars = fitp.g.n_fitp
		parinfo_value = np.zeros((n_pars))
		parinfo_fixed = np.zeros((n_pars), dtype=np.int)
		parinfo_limited = np.zeros((n_pars, 2), dtype=np.int)
		parinfo_limits = np.zeros((n_pars, 2))
		parinfo_relstep = np.zeros((n_pars))
		parinfo_mpmaxstep = np.zeros((n_pars))
		parinfo_mpminstep = np.zeros((n_pars))

		for i in range(n_pars):
			parinfo_value[i] = float(fitp.s.val[i])
			wo = np.where(fitp.keywords.peaks == i)
			if wo[0].size > 0 :
				if fitp.s.val[i] > 0:
					parinfo_value[i] = np.log10(fitp.s.val[i])
				else:
					parinfo_value[i] = 0
			if fitp.s.use[i] == 1:
				parinfo_fixed[i] = 1

		which_parameters_to_fit = np.where(fitp.s.use != 1)
		logger.info('parameters to fit: %s', fitp.s.name[which_parameters_to_fit])

		x = np.arange(float(main_dict['max_spec_channels']))
		add_matrixfit_pars = np.zeros((6))
		add_matrixfit_pars[0] = fitp.s.val[fitp.keywords.energy_pos[0]]
		add_matrixfit_pars[1] = fitp.s.val[fitp.keywords.energy_pos[1]]
		add_matrixfit_pars[2] = fitp.s.val[fitp.keywords.energy_pos[2]]
		add_matrixfit_pars[3] = fitp.s.val[fitp.keywords.added_params[1]]
		add_matrixfit_pars[4] = fitp.s.val[fitp.keywords.added_params[2]]
		add_matrixfit_pars[5] = fitp.s.val[fitp.keywords.added_params[3]]

		parinfo_prime_val = parinfo_value[np.amin(fitp.keywords.kele_pos):np.amax(fitp.keywords.mele_pos)+1]
		parinfo_prime_val = np.concatenate((parinfo_prime_val, [parinfo_value[fitp.keywords.coherent_pos[1]], parinfo_value[fitp.keywords.compton_pos[2]]],
											parinfo_value[fitp.keywords.added_params[4:13]], parinfo_value[fitp.keywords.added_params[1:4]]), axis=0)
		parinfo_prime_fixed = parinfo_fixed[np.amin(fitp.keywords.kele_pos):np.amax(fitp.keywords.mele_pos)+1]
		parinfo_prime_fixed = np.concatenate((parinfo_prime_fixed, [parinfo_fixed[fitp.keywords.coherent_pos[1]], parinfo_fixed[fitp.keywords.compton_pos[2]]],
											  parinfo_fixed[fitp.keywords.added_params[4:13]], parinfo_fixed[fitp.keywords.added_params[1:4]]), axis=0)

		parinfo_prime_limited = parinfo_limited[np.amin(fitp.keywords.kele_pos):np.amax(fitp.keywords.mele_pos)+1,:]
		parinfo_prime_limited = np.concatenate((parinfo_prime_limited, [parinfo_limited[fitp.keywords.coherent_pos[1],:], parinfo_limited[fitp.keywords.compton_pos[2],:]],
											  parinfo_limited[fitp.keywords.added_params[4:13],:], parinfo_limited[fitp.keywords.added_params[1:4],:]), axis=0)

		parinfo_prime_limits = parinfo_limits[np.amin(fitp.keywords.kele_pos):np.amax(fitp.keywords.mele_pos)+1,:]
		parinfo_prime_limits = np.concatenate((parinfo_prime_limits, [parinfo_limits[fitp.keywords.coherent_pos[1],:], parinfo_limits[fitp.keywords.compton_pos[2],:]],
											  parinfo_limits[fitp.keywords.added_params[4:13],:], parinfo_limits[fitp.keywords.added_params[1:4],:]), axis=0)

		fitp.keywords.use_this_par[:] = 0
		fitp.keywords.use_this_par[np.where(parinfo_prime_fixed != 1)] = 1
		# force the last three to be 0, to make sure they do NOT get fitted as peaks.
		fitp.keywords.use_this_par[parinfo_prime_val.size-3:parinfo_prime_val.size] = 0

		temp_fitp_use = fitp.s.use[np.amin(fitp.keywords.kele_pos):np.amax(fitp.keywords.mele_pos) + 1]
		temp_fitp_name = fitp.s.name[np.amin(fitp.keywords.kele_pos):np.amax(fitp.keywords.mele_pos) + 1]
		which_elements_to_fit = (np.nonzero(temp_fitp_use != 1))[0]
		logger.info('elements to fit: %s', temp_fitp_name[which_elements_to_fit])

		fit = maps_analyze.analyze(logger)

		fitmatrix = fit.generate_fitmatrix(fitp, x, parinfo_value)

		wo_use_this_par = (np.nonzero(fitp.keywords.use_this_par[0:(np.max(fitp.keywords.mele_pos) - np.min(fitp.keywords.kele_pos) + 1)] == 1))[0]

		no_use_pars = wo_use_this_par.size + 2
		fitmatrix_reduced = np.zeros((x.size, no_use_pars))

		for mm in range(wo_use_this_par.size):
			fitmatrix_reduced[:, mm] = fitmatrix[:, wo_use_this_par[mm]]
		mm = wo_use_this_par.size - 1
		fitmatrix_reduced[:, mm] = fitmatrix[:, np.max(fitp.keywords.mele_pos) - np.min(fitp.keywords.kele_pos) + 1] # elastic scatter
		mm = mm + 1
		fitmatrix_reduced[:, mm] = fitmatrix[:, np.max(fitp.keywords.mele_pos) - np.min(fitp.keywords.kele_pos) + 2] # inelastic scatter

		if main_dict['nnls'] == 0:
			logger.info('Calculating nnls. Start time: %s', time.time())
			# Compute the singular value decomposition of A:
			# SVDC, fitmatrix_reduced, W, U, V, /double
			U, w, V = np.linalg.svd(fitmatrix_reduced, full_matrices=False)
			# Create a diagonal array WP of reciprocal singular values from the
			# output vector W. To avoid overflow errors when the reciprocal values
			# are calculated, only elements with absolute values greater than or
			# equal to 1.0 to 10-5 are reciprocated.
			wp = np.zeros((no_use_pars,no_use_pars))
			for kk_temp in range(no_use_pars):
				if np.abs(w[kk_temp]) > 1.0e-5:
					wp[kk_temp, kk_temp] = 1.0 / w[kk_temp]
				# We can now express the solution to the linear system as a
				# array-vector product. (See Section 2.6 of Numerical Recipes for
				# a derivation of this formula.)
				# solution = V ## WP ## TRANSPOSE(U) ## B

			sol_intermediate = np.dot(np.dot(V.T, wp), U.T)
			logger.info('SVD finished. Time: %s', time.time())
		else:
			# make sure that sol_intermediate is defined, even if we do not
			# use it.
			sol_intermediate = np.zeros((no_use_pars, main_dict['max_spec_channels']))

		# Save intermediate solution to a file
		filepath = os.path.join(main_dict['output_dir'], maps_intermediate_solution_file) + suffix
		outfile = open_file_with_retry(filepath, 'wb')

		np.savez(outfile, sol_intermediate=sol_intermediate, fitmatrix_reduced=fitmatrix_reduced)
		outfile.close()

		# Read calibration
		logger.info('Looking for calibration info')
		calibration = maps_calibration.calibration(main_dict, maps_conf, logger)

		# perform calibration
		no_nbs = 1
		calibration.read_generic_calibration(this_detector=this_detector,
											total_number_detectors=total_number_detectors,
											no_nbs=no_nbs,
											fitp=fitp,
											info_elements=info_elements)

		no_files = len(filenames)

		detector_number_arr = map(str, detector_number_arr)

		filepath = os.path.join(main_dict['output_dir'], 'mapsprocessinfo_' + '.txt')
		text_file = open_file_with_retry(filepath, 'w')
		text_file.write(time.strftime("%a, %d %b %Y %H:%M:%S"))
		text_file.close()

		seconds_start = time.time()

		if no_processors_to_use_files >= 2:
			# Need to modify stout to flush prints
			logger.info('use multiple processors for multiple files')
			#pool = multiprocessing.Pool(no_processors_to_use_files)
			procs_to_start = []
			procs_to_join = []
			for pp in range(no_files):
				header, scan_ext = os.path.splitext(filenames[pp])
				mdafilename = os.path.join(main_dict['mda_dir'], filenames[pp])
				logger.info('Multiple processor file version: doing filen #: %s this detector: %s pp: %s', mdafilename, this_detector, pp)
				proc = multiprocessing.Process(target=mp_make_maps, args=(logger.name, info_elements, main_dict, maps_conf, header, mdafilename, this_detector,use_fit, total_number_detectors, quick_dirty, main_dict['nnls'], main_dict['xrf_bin'], max_no_processors_lines))
				procs_to_start += [proc]
			num_procs_running = 0
			while len(procs_to_start) > 0 or len(procs_to_join) > 0:
				for proc in procs_to_start[:]:
					if num_procs_running < no_processors_to_use_files:
						logger.debug('Starting file processs')
						proc.start()
						procs_to_join += [proc]
						procs_to_start.remove(proc)
						num_procs_running += 1
				for proc in procs_to_join[:]:
					if proc.exitcode is None and not proc.is_alive():
						time.sleep(0.1)
					elif proc.exitcode < 0:
						logger.error('Exception thrown for one of the files processing')
						proc.join()
						logger.debug('Process Joined')
						procs_to_join.remove(proc)
						num_procs_running -= 1
					else:
						logger.debug('Finished processing file')
						proc.join()
						logger.debug('Process Joined')
						procs_to_join.remove(proc)
						num_procs_running -= 1
		else:
			#  a single processor machine,	just use the single processor
			makemaps = maps_generate_img_dat.analyze(logger, info_elements, main_dict, maps_conf, beamline=main_dict['beamline'], use_fit=use_fit)
			for pp in range(no_files):
				header, scan_ext = os.path.splitext(filenames[pp])
				mdafilename = os.path.join(main_dict['mda_dir'], header + scan_ext)
				logger.info('Single processor file version: doing filen #: %s this detector %s', mdafilename, this_detector)

				makemaps.generate_img_dat_threaded(header, mdafilename, this_detector, total_number_detectors, quick_dirty, main_dict['nnls'], max_no_processors_lines, main_dict['xrf_bin'])

	seconds_end = time.time()
	delta_hours = int((seconds_end - seconds_start) / 3600.)
	delta_min = (seconds_end - seconds_start) / 60. - 60. * int((seconds_end - seconds_start) / 3600.)
	logger.info('fitting of all scans took a total of %s hours and %s minutes', delta_hours, delta_min)

	logger.info('MAPS are finished.')
	return True
