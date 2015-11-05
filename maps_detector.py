'''
Created on Nov 22, 2011

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
import datetime
import numpy as np
import maps_fit_parameters
import maps_analyze
from file_io.file_util import open_file_with_retry

# ----------------------------------------------------------------------


def get_detector_calibration(maps_conf, beamline, info_elements, scan, fpo_file, logger):

	verbose = False

	if beamline == '2-ID-E':
		crate = '2xfm'
	if beamline == '2-ID-D':
		crate = '2idd'
	if beamline == '2-ID-B':
		crate = '2idb1'
	if beamline == '2-BM':
		crate = '2bmb'
	if beamline == 'Bionanoprobe':
		crate = '2xfm'

	# fpo_file = 'maps_fit_parameters_override.txt'

	try:
		# check to see if the file is there by opening it
		f = open(fpo_file, 'rt')
		f.close()
		if verbose:
			logger.info('reading maps_fit_parameters_override.txt')
		fp = maps_fit_parameters.maps_fit_parameters(logger)
		fitp = fp.define_fitp(beamline, info_elements)
		fitp, test_string, pileup_string = fp.read_fitp(fpo_file, info_elements)

		maps_conf.calibration.offset[0] = fitp.s.val[fitp.keywords.energy_pos[0]]
		maps_conf.calibration.slope[0] = fitp.s.val[fitp.keywords.energy_pos[1]]
		fitp = 0

	except:
		if verbose:
			logger.info('will try to use calibration from mda file')
			logger.debug('maps_conf.calibration.slope[0] %s', maps_conf.calibration.slope[0])
			logger.debug('scan.extra_pv %s', scan.extra_pv)
		if (maps_conf.calibration.slope[0] == 0.0) and (len(scan.extra_pv) > 0):
			if verbose:
				logger.debug('have extra pvs')
			if (beamline == '2-ID-D') or (beamline == '2-ID-B'):
				if crate + ':mca1.CALO' in scan.extra_pv:
					maps_conf.calibration.offset[0] = scan.extra_pv[crate + ':mca1.CALO'][2][0]
				if crate + ':mca1.CALS' in scan.extra_pv:
					maps_conf.calibration.slope[0] = scan.extra_pv[crate + ':mca1.CALS'][2][0]

			if beamline == '2-BM':
				maps_conf.calibration.offset[0] = 0.008
				maps_conf.calibration.slope[0] = 0.0057

			if beamline == '2-ID-E':
				if crate + ':mca1.CALO' in scan.extra_pv:
					maps_conf.calibration.offset[0] = scan.extra_pv[crate + ':mca1.CALO'][2][0]
				if crate + ':mca1.CALS' in scan.extra_pv:
					maps_conf.calibration.slope[0] = scan.extra_pv[crate + ':mca1.CALS'][2][0]

				# legacy: assume presence of three element detector if above not found
				if crate + ':med:mca1.CALO' in scan.extra_pv:
					maps_conf.calibration.offset[0] = scan.extra_pv[crate + ':med:mca1.CALO'][2][0]
				if crate + ':med:mca1.CALS' in scan.extra_pv:
					maps_conf.calibration.slope[0] = scan.extra_pv[crate + ':med:mca1.CALS'][2][0]
				if crate + ':med:mca2.CALO' in scan.extra_pv:
					maps_conf.calibration.offset[1] = scan.extra_pv[crate + ':med:mca2.CALO'][2][0]
				if crate + ':med:mca2.CALS' in scan.extra_pv:
					maps_conf.calibration.slope[1] = scan.extra_pv[crate + ':med:mca2.CALS'][2][0]
				if crate + ':med:mca1.CALO' in scan.extra_pv:
					maps_conf.calibration.offset[0] = scan.extra_pv[crate + ':med:mca1.CALO'][2][0]
				if crate + ':med:mca1.CALS' in scan.extra_pv:
					maps_conf.calibration.slope[0] = scan.extra_pv[crate + ':med:mca1.CALS'][2][0]
				if crate + ':med:mca1.CALO' in scan.extra_pv:
					maps_conf.calibration.offset[0] = scan.extra_pv[crate + ':med:mca1.CALO'][2][0]
				if crate + ':med:mca1.CALS' in scan.extra_pv:
					maps_conf.calibration.slope[0] = scan.extra_pv[crate + ':med:mca1.CALS'][2][0]
				if crate + ':med:mca3.CALO' in scan.extra_pv:
					maps_conf.calibration.offset[2] = scan.extra_pv[crate + ':med:mca3.CALO'][2][0]
				if crate + ':med:mca3.CALS' in scan.extra_pv:
					maps_conf.calibration.slope[2] = scan.extra_pv[crate + ':med:mca3.CALS'][2][0]

# ----------------------------------------------------------------------


def find_detector_name(det_descr, date_number, detector_arr, detector_description_arr, make_maps_conf,
						x_coord_arr, y_coord_arr, beamline,
						n_cols, n_rows, maps_overridefile, logger):

	verbose = False

	srcurrent = None
	us_ic = None
	ds_ic = None
	dpc1_ic = None
	dpc2_ic = None
	cfg_1 = None
	cfg_2 = None
	cfg_3 = None
	cfg_4 = None
	cfg_5 = None
	cfg_6 = None
	cfg_7 = None
	cfg_8 = None
	cfg_9 = None
	cfg_10 = None
	ELT1 = None
	ERT1 = None
	ICR1 = None
	OCR1 = None

	# generate direct maps, such as SR current, ICs, life time
	# 0:srcurrent, 1:us_ic, 2:ds_ic, 3:dpc1_ic, 4:dpc2_ic,
	# 5:cfg_1, 6:cfg_2, 7:cfg_3, 8:cfg_4, 9:cfg_5, 10:cfg_6, 11:cfg_7, 12:cfg_8
	# 13:ELT1, 14:ERT1, 15: ELT2, 16:ERT2, 17:ELT3, 18: ERT3

	have_override_file = False

	d_det = np.zeros((n_cols, n_rows, len(det_descr)))
	det_name = []
	dmaps_set = np.zeros((n_cols, n_rows, make_maps_conf.n_used_dmaps))

	try:
		f = open_file_with_retry(maps_overridefile, 'rt')
		# override file exists.
		have_override_file = True

		for line in f:
			if ':' in line :
				slist = line.split(':')
				tag = slist[0]
				value = ':'.join(slist[1:])

				if	 tag == 'SRCURRENT' : srcurrent = value.strip()
				elif tag == 'US_IC'		: us_ic = value.strip()
				elif tag == 'DS_IC'		: ds_ic = value.strip()
				elif tag == 'DPC1_IC'	: dpc1_ic = value.strip()
				elif tag == 'DPC2_IC'	: dpc2_ic = value.strip()
				elif tag == 'CFG_1'		: cfg_1 = value.strip()
				elif tag == 'CFG_2'		: cfg_2 = value.strip()
				elif tag == 'CFG_3'		: cfg_3 = value.strip()
				elif tag == 'CFG_4'		: cfg_4 = value.strip()
				elif tag == 'CFG_5'		: cfg_5 = value.strip()
				elif tag == 'CFG_6'		: cfg_6 = value.strip()
				elif tag == 'CFG_7'		: cfg_7 = value.strip()
				elif tag == 'CFG_8'		: cfg_8 = value.strip()
				elif tag == 'CFG_9'		: cfg_9 = value.strip()
				elif tag == 'CFG_10'	: cfg_10 = value.strip()
				elif tag == 'ELT1'		: ELT1 = value.strip()
				elif tag == 'ERT1'		: ERT1 = value.strip()
				elif tag == 'ICR1'		: ICR1 = value.strip()
				elif tag == 'OCR1'		: OCR1 = value.strip()

		f.close()

		if srcurrent:
			for ii in det_descr:
				this_det = ii
				this_name = ''
				if ((this_det.lower() == 'srcurrent') and (srcurrent)) : this_name = srcurrent
				if ((this_det.lower() == 'us_ic') and (us_ic)) : this_name = us_ic
				if ((this_det.lower() == 'ds_ic') and (ds_ic)) : this_name = ds_ic
				if ((this_det.lower() == 'dpc1_ic') and (dpc1_ic)) : this_name = dpc1_ic
				if ((this_det.lower() == 'dpc2_ic') and (dpc2_ic)) : this_name = dpc2_ic
				if ((this_det.lower() == 'cfg_1') and (cfg_1)) : this_name = cfg_1
				if ((this_det.lower() == 'cfg_2') and (cfg_2)) : this_name = cfg_2
				if ((this_det.lower() == 'cfg_3') and (cfg_3)) : this_name = cfg_3
				if ((this_det.lower() == 'cfg_4') and (cfg_4)) : this_name = cfg_4
				if ((this_det.lower() == 'cfg_5') and (cfg_5)) : this_name = cfg_5
				if ((this_det.lower() == 'cfg_6') and (cfg_6)) : this_name = cfg_6
				if ((this_det.lower() == 'cfg_7') and (cfg_7)) : this_name = cfg_7
				if ((this_det.lower() == 'cfg_8') and (cfg_8)) : this_name = cfg_8
				if ((this_det.lower() == 'cfg_9') and (cfg_9)) : this_name = cfg_9
				if ((this_det.lower() == 'cfg_10') and (cfg_10)) : this_name = cfg_10
				if ((this_det == 'ELT1') and (ELT1)) : this_name = ELT1
				if ((this_det == 'ERT1') and (ERT1)) : this_name = ERT1
				if ((this_det == 'ICR1') and (ICR1)) : this_name = ICR1
				if ((this_det == 'OCR1') and (OCR1)) : this_name = OCR1

				det_name.append(this_name)
				if this_name in detector_description_arr:
					ind = detector_description_arr.index(this_name)
					d_det[:, :, det_descr.index(ii)] = detector_arr[:, :, ind]

	except :
		# Haven't found override file
		have_override_file = False

	if (have_override_file == False) or (srcurrent == None):
		logger.info('did not fine either override file, or srcurrent, use defaults')
		for ii in det_descr:
			this_det = ii
			this_name = ''
			if this_det.lower() == 'srcurrent' : this_name = 'S:SRcurrentAI'
			if beamline == '2-ID-E' :
				if this_det.lower() == 'us_ic' : this_name = '2xfm:scaler1_cts1.B'
				if this_det.lower() == 'ds_ic' : this_name = '2xfm:scaler1_cts1.C'
				if this_det.lower() == 'dpc1_ic' : this_name = '2xfm:scaler1_cts1.D'

				if this_det == 'ELT1' : this_name = '2xfm:med:mca1.ELTM'
				if this_det == 'ELT2' : this_name = '2xfm:med:mca2.ELTM'
				if this_det == 'ELT3' : this_name = '2xfm:med:mca3.ELTM'

				if this_det == 'ERT1' : this_name = '2xfm:med:mca1.ERTM'
				if this_det == 'ERT2' : this_name = '2xfm:med:mca2.ERTM'
				if this_det == 'ERT3' : this_name = '2xfm:med:mca3.ERTM'

				if date_number > datetime.date(2005, 10, 01):
					if this_det == 'ELT1' : this_name = '2xfm:mca1.ELTM'
					if this_det == 'ERT1' : this_name = '2xfm:mca1.ERTM'

				if date_number > datetime.date(2003, 01, 01):
					if this_det.lower() == 'us_ic' : this_name = '2xfm:scaler1_cts1.B'
					if this_det.lower() == 'ds_ic' : this_name = '2xfm:scaler1_cts1.C'
					if this_det.lower() == 'dpc1_ic' : this_name = '2xfm:scaler1_cts1.D'

				if date_number > datetime.date(2005, 11, 21):
					if this_det.lower() == 'us_ic' : this_name = '2xfm:scaler3_cts1.B'
					if this_det.lower() == 'ds_ic' : this_name = '2xfm:scaler3_cts1.C'
					if this_det.lower() == 'dpc1_ic' : this_name = '2xfm:scaler3_cts2.A'
					if this_det.lower() == 'dpc2_ic' : this_name = '2xfm:scaler3_cts2.B'

					if this_det.lower() == 'cfg_1' : this_name = '2xfm:scaler3_cts3.B'
					if this_det.lower() == 'cfg_2' : this_name = '2xfm:scaler3_cts3.C'
					if this_det.lower() == 'cfg_3' : this_name = '2xfm:scaler3_cts3.D'
					if this_det.lower() == 'cfg_4' : this_name = '2xfm:scaler3_cts4.A'
					if this_det.lower() == 'cfg_5' : this_name = '2xfm:scaler3_cts4.B'
					if this_det.lower() == 'cfg_6' : this_name = '2xfm:scaler3_cts4.C'
					if this_det.lower() == 'cfg_7' : this_name = '2xfm:scaler3_cts4.D'
					if this_det.lower() == 'cfg_8' : this_name = '2xfm:scaler3_cts5.A'

				# the part below is for scans in the protein gel setup
				eltm_name = '2xfm1:dxpSaturn:mca1.ELTM'
				if eltm_name in detector_description_arr:
					ind = detector_description_arr.index(eltm_name)

					if this_det == 'ELT1' : this_name = '2xfm1:dxpSaturn:mca1.ELTM'
					if this_det == 'ERT1' : this_name = '2xfm1:dxpSaturn:mca1.ERTM'
					if this_det.lower() == 'us_ic' :
						if '2xfm:scaler1_cts1.B' in detector_description_arr:
							this_name = '2xfm:scaler1_cts1.B'
						else:
							this_name = '2xfm:scaler3_cts1.B'

						if this_det.lower() == 'ds_ic' :
							if '2xfm:scaler1_cts1.C' in detector_description_arr:
								this_name = '2xfm:scaler1_cts1.C'
							else:
								this_name = '2xfm:scaler3_cts1.C'

			if beamline == '2-ID-D' :
				if this_det.lower() == 'us_ic' : this_name = '2idd:scaler1_cts1.C'
				if this_det.lower() == 'ds_ic' : this_name = '2idd:scaler1_cts1.B'

				# two possibilities, either D uses the old electronics (mca) or the
				# DSP. need to test.

				found = 0
				if 'dxpSaturn2idd1:mca1.ELTM' in detector_description_arr:
					if this_det == 'ELT1' : this_name = 'dxpSaturn2idd1:mca1.ELTM'
					if this_det == 'ERT1' : this_name = 'dxpSaturn2idd1:mca1.ERTM'
					found = 1

				if found != 1 :
					if 'dxpSaturn2idd:mca1.ELTM' in detector_description_arr:
						if this_det == 'ELT1' : this_name = 'dxpSaturn2idd:mca1.ELTM'
						if this_det == 'ERT1' : this_name = 'dxpSaturn2idd:mca1.ERTM'
						found = 1

				if found != 1 :
					if this_det == 'ELT1' : this_name = '2idd:mca1.ELTM'
					if this_det == 'ERT1' : this_name = '2idd:mca1.ERTM'

			if beamline == '2-ID-B' :
				if this_det.lower() == 'us_ic' : this_name = '2idb1:scaler1.S2'
				if this_det.lower() == 'ds_ic' : this_name = '2idb1:scaler1.S4'
				if this_det == 'ELT1' : this_name = '2idb1:mca1.ELTM'
				if this_det == 'ERT1' : this_name = '2idb1:mca1.ERTM'

			if beamline == '2-BM' :
				if this_det.lower() == 'us_ic':
					this_name = '2bmb:scaler1_cts1.D'
				if this_det.lower() == 'ds_ic':
					this_name = '2bmb:scaler1_cts1.C'

				if this_det == 'ELT1':
					this_name = '2bmb:mca1.ELTM'
				if this_det == 'ERT1':
					this_name = '2bmb:mca1.ERTM'

				if '2bmb:aim_adc1.ELTM' in detector_description_arr:
					if this_det == 'ELT1':
						this_name = '2bmb:aim_adc1.ELTM'
					if this_det == 'ERT1':
						this_name = '2bmb:aim_adc1.ERTM'

			if beamline == 'Bio-CAT':
				if this_det == 'ELT1':
					this_name = ' Live time'
				if this_det == 'ERT1':
					this_name = ' Real time'

				if this_det.lower() == 'us_ic':
					this_name = 'us_ic'
				if this_det.lower() == 'ds_ic':
					this_name = 'ds_ic'

			det_name.append(this_name)
			if this_name in detector_description_arr:
				ind = detector_description_arr.index(this_name)
				d_det[:, :, det_descr.index(ii)] = detector_arr[:, :, ind]

	dmaps_use = []
	for item in make_maps_conf.dmaps:
		dmaps_use.append(item.use)
	dmaps_use = np.array(dmaps_use)
	det_id = np.where(dmaps_use == 1)
	det_id = det_id[0]

	for jj in range(make_maps_conf.n_used_dmaps) :
		this_det = det_id[jj]
		make_maps_conf.dmaps[this_det].units = ''

		if make_maps_conf.dmaps[this_det].name == 'SRcurrent':
			if 'srcurrent' in det_descr:
				dmaps_set[:, :, this_det] = d_det[:, :, det_descr.index('srcurrent')]
				make_maps_conf.dmaps[this_det].units = 'mA'

		if make_maps_conf.dmaps[this_det].name == 'us_ic':
			if 'us_ic' in det_descr:
				dmaps_set[:, :, this_det] = d_det[:, :, det_descr.index('us_ic')]
				make_maps_conf.dmaps[this_det].units = 'cts/s'

			if beamline == '2-ID-B':
				if '2idb1:scaler1.T' in detector_description_arr:
					dmaps_set[:, :, this_det] = dmaps_set[:, :, this_det] / detector_arr[:, :, detector_description_arr.index('2idb1:scaler1.T')]
				else:
					make_maps_conf.dmaps[this_det].units = 'cts'

		if make_maps_conf.dmaps[this_det].name == 'ds_ic':
			if 'ds_ic' in det_descr:
				dmaps_set[:, :, this_det] = d_det[:, :, det_descr.index('ds_ic')]
				make_maps_conf.dmaps[this_det].units = 'cts/s'

		if make_maps_conf.dmaps[this_det].name == 'abs_ic':
			if ('us_ic' in det_descr) and ('ds_ic' in det_descr):
				dmaps_set[:, :, this_det] = d_det[:, :, det_descr.index('ds_ic')]/d_det[:, :, det_descr.index('us_ic')]

		if make_maps_conf.dmaps[this_det].name == 'V_dpc_ic':
			if ('ds_ic' in det_descr) and ('dpc1_ic' in det_descr):
				dmaps_set[:, :, this_det] = d_det[:, :, det_descr.index('dpc1_ic')]/d_det[:, :, det_descr.index('ds_ic')]

		if make_maps_conf.dmaps[this_det].name == 'H_dpc_ic':
			if ('dpc1_ic' in det_descr) and ('dpc2_ic' in det_descr):
				dmaps_set[:, :, this_det] = d_det[:, :, det_descr.index('dpc2_ic')]/d_det[:, :, det_descr.index('dpc1_ic')]

		if 'ELT' in make_maps_conf.dmaps[this_det].name:
			if make_maps_conf.dmaps[this_det].name in det_descr:
				dmaps_set[:, :, this_det] = d_det[:, :, det_descr.index(make_maps_conf.dmaps[this_det].name)]
				make_maps_conf.dmaps[this_det].units = 's'
			else:
				logger.warning('Could not find elapsed life time detector (1). Will proceed assuming a ELT=1')

		if 'ERT' in make_maps_conf.dmaps[this_det].name:
			if make_maps_conf.dmaps[this_det].name in det_descr:
				dmaps_set[:, :, this_det] = d_det[:, :, det_descr.index(make_maps_conf.dmaps[this_det].name)]
				make_maps_conf.dmaps[this_det].units = 's'

		if 'ICR' in make_maps_conf.dmaps[this_det].name:
			if make_maps_conf.dmaps[this_det].name in det_descr:
				dmaps_set[:, :, this_det] = d_det[:, :, det_descr.index(make_maps_conf.dmaps[this_det].name)]
				make_maps_conf.dmaps[this_det].units = ''

		if 'OCR' in make_maps_conf.dmaps[this_det].name:
			if make_maps_conf.dmaps[this_det].name in det_descr:
				dmaps_set[:, :, this_det] = d_det[:, :, det_descr.index(make_maps_conf.dmaps[this_det].name)]
				make_maps_conf.dmaps[this_det].units = ''

		if (('cfg' in make_maps_conf.dmaps[this_det].name ) or \
			('norm' in make_maps_conf.dmaps[this_det].name) or \
				('phase' in make_maps_conf.dmaps[this_det].name)) :
			if 'us_ic' in det_descr: wo_a = det_descr.index('us_ic')
			else: continue
			if 'cfg_1' in det_descr: wo_1 = det_descr.index('cfg_1')
			else: continue
			if 'cfg_2' in det_descr: wo_2 = det_descr.index('cfg_2')
			else: continue
			if 'cfg_3' in det_descr: wo_3 = det_descr.index('cfg_3')
			else: continue
			if 'cfg_4' in det_descr: wo_4 = det_descr.index('cfg_4')
			else: continue
			if 'cfg_5' in det_descr: wo_5 = det_descr.index('cfg_5')
			else: continue
			if 'cfg_6' in det_descr: wo_6 = det_descr.index('cfg_6')
			else: continue
			if 'cfg_7' in det_descr: wo_7 = det_descr.index('cfg_7')
			else: continue
			if 'cfg_8' in det_descr: wo_8 = det_descr.index('cfg_8')
			else: continue

			t_1 = d_det[:, :, wo_1]
			t_2 = d_det[:, :, wo_2]
			t_3 = d_det[:, :, wo_3]
			t_4 = d_det[:, :, wo_4]
			t_5 = d_det[:, :, wo_5]
			t_6 = d_det[:, :, wo_6]
			t_7 = d_det[:, :, wo_7]
			t_8 = d_det[:, :, wo_8]
			t_abs = t_2+t_3+t_4+t_5
			if make_maps_conf.dmaps[this_det].name == 'abs_cfg' :
				dmaps_set[:, :, this_det] = t_abs/d_det[:, :, wo_a]
			if make_maps_conf.dmaps[this_det].name == 'H_dpc_cfg' :
				dmaps_set[:, :, this_det] = (t_2-t_3-t_4+t_5)/t_abs
			if make_maps_conf.dmaps[this_det].name == 'V_dpc_cfg' :
				dmaps_set[:, :, this_det] = (t_2+t_3-t_4-t_5)/t_abs
			if make_maps_conf.dmaps[this_det].name == 'dia1_dpc_cfg' :
				dmaps_set[:, :, this_det] = (t_2-t_4)/t_abs
			if make_maps_conf.dmaps[this_det].name == 'dia2_dpc_cfg' :
				dmaps_set[:, :, this_det] = (t_3-t_5)/t_abs

			nrml = (t_2-t_3-t_4+t_5)/t_abs
			sz = nrml.shape
			nx = sz[0]
			ny = sz[1]
			if (nx % 2) == 0:
				nx = nx - 1
			if (ny % 2) == 0:
				ny = ny - 1
			nrml = nrml[0:nx, 0:ny]
			ntmb = (t_2+t_3-t_4-t_5)/t_abs
			ntmb = ntmb[0:nx, 0:ny]

			if make_maps_conf.dmaps[this_det].name == 'phase' : no_int = 1
			else: no_int = 0

			anl = maps_analyze.analyze(logger)
			nrml, ntmb, rdt = anl.maps_simple_dpc_integration(nrml, ntmb, no_int = no_int)
			# notem nrml, ntmb, now normalized (what comes up ust go down)
			if make_maps_conf.dmaps[this_det].name == 'H_dpc_norm':
				dmaps_set[0:nx, 0:ny, this_det] = nrml
			if make_maps_conf.dmaps[this_det].name == 'V_dpc_norm':
				dmaps_set[0:nx, 0:ny, this_det] = ntmb
			if make_maps_conf.dmaps[this_det].name == 'phase':
				dmaps_set[0:nx, 0:ny, this_det] = rdt

		if 'deadT' in make_maps_conf.dmaps[this_det].name:
			if ('OCR1' in det_descr) and ('ICR1' in det_descr):
				wo_1 = det_descr.index('OCR1')
				wo_2 = det_descr.index('ICR1')
			else:
				# could not find ocr / icr, try ELT/ERT:
				if ('ELT1' in det_descr) and ('ERT1' in det_descr):
					wo_1 = det_descr.index('ELT1')
					wo_2 = det_descr.index('ERT1')
				else: continue
			dmaps_set[:, :, this_det] = (1. - (d_det[:, :, wo_1]/d_det[:, :, wo_2])) * 100.
			make_maps_conf.dmaps[this_det].units = '%'

		if 'x_coord' in make_maps_conf.dmaps[this_det].name:
			try:
				for mm in range(y_coord_arr.size) :
					dmaps_set[:, mm, this_det] = x_coord_arr[:]
					make_maps_conf.dmaps[this_det].units = 'mm'
			except:
				logger.warning('Warning could not read x_coord')

		if 'y_coord' in make_maps_conf.dmaps[this_det].name:
			try:
				for mm in range(x_coord_arr.size) :
					dmaps_set[mm, :, this_det] = y_coord_arr[:]
					make_maps_conf.dmaps[this_det].units = 'mm'
			except:
				logger.warning('Warning could not read y_coord')

	return dmaps_set
