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
import os
import datetime 
import numpy as np
import henke


kele = ['Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 
        'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 
        'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 
        'In', 'Sn', 'Sb', 'Te', 'I', 'dummy', 'dummy'] 
        
#lele = ['Mo_L', 'Ag_L', 'Sn_L', 'Cd_L', 'I_L', 'Cs_L', 'Ba_L', 'Eu_L', 'Gd_L', 'W_L', 'Pt_L', 'Au_L', 
#        'Hg_L', 'Pb_L', 'U_L', 'Pu_L', 'Sm_L', 'Y_L', 'Pr_L', 'Ce_L', 'Zr_L', 'Os_L', 'Rb_L', 'Ru_L']

lele = ['Mo_L', 'Tc_L', 'Ru_L', 'Rh_L', 'Pd_L', 'Ag_L', 'Cd_L', 'In_L', 'Sn_L', 'Sb_L', 'Te_L', 'I_L', 'Xe_L,', 'Cs_L', 'Ba_L', 'La_L', 'Ce_L', 'Pr_L', 'Nd_L', 'Pm_L', 'Sm_L',
        'Eu_L', 'Gd_L', 'Tb_L', 'Dy_L', 'Ho_L', 'Er_L', 'Tm_L', 'Yb_L', 'Lu_L', 'Hf_L', 'Ta_L', 'W_L', 'Re_L', 'Os_L', 'Ir_L', 'Pt_L', 'Au_L', 'Hg_L', 'Tl_L', 
        'Pb_L', 'Bi_L', 'Po_L', 'At_L', 'Rn_L', 'Fr_L', 'Ac_L', 'Th_L', 'Pa_L', 'U_L', 'Np_L', 'Pu_L', 'Am_L']



mele = ['Au_M', 'Pb_M', 'U_M', 'noise', 'Pt_M', 'dummy', 'dummy', 'dummy', 'dummy']
 
 
#-----------------------------------------------------------------------------
class Cpar:
    def __init__(self):
        
        self.name='' 
        self.energy=0. 
        self.ratio=0.
        self.Ge_mu=0.
        self.Si_mu=0.
        self.mu_fraction=0.
        self.width_multi=1.
        self.type=0
        self.pileup=0
                
#-----------------------------------------------------------------------------        
class Ckeywords:
    def __init__(self):
        
        
        n_spectra_to_fit = 3

        energy_pos = [0, 1, 2]
        fwhm_pos = [3, 4]
        coherent_pos = [5, 6]
        compton_pos = [7, 8, 9, 10, 11, 12, 13, 14]
        added_params = np.arange(13) + 15
        tail_pos = np.arange(12) + 28
        
        kele_pos = np.arange(len(kele))+np.amax(tail_pos)+1      
        lele_pos = np.arange(len(lele))+np.amax(kele_pos)+1
        mele_pos = np.arange(len(mele))+np.amax(lele_pos)+1
                
        peaks = np.concatenate((added_params[4:13], [coherent_pos[1], compton_pos[2]], kele_pos, lele_pos, mele_pos), axis=0)
                      
        self.tail = 1
        self.fixed_fwhm = 0
        self.twoide = 0
        self.twoidd = 0
        self.twobm = 0
        self.spectral_binning = 0
        self.fwhm_off = np.zeros((10)) 
        self.fwhm_lin = np.zeros((10))
        self.background = np.zeros((4000)) 
        self.spectrum = np.zeros((4000)) 
        self.xmin = 0
        self.xmax = 0
        self.multispectrum = np.zeros((2048*n_spectra_to_fit))
        self.energy_pos = energy_pos 
        self.fwhm_pos = fwhm_pos
        self.coherent_pos = coherent_pos 
        self.compton_pos = compton_pos
        self.added_params = added_params
        self.tail_pos = tail_pos 
        self.kele_pos = kele_pos
        self.lele_pos = lele_pos 
        self.mele_pos = mele_pos
        self.use_this_par = np.zeros((np.amax(mele_pos)+1) , dtype = np.int32)
        self.peaks = peaks
        self.add_filter = np.ones((4000))
        
#-----------------------------------------------------------------------------        
class Cg:
    def __init__(self):
        self.n_fitp = 0
        self.current = 0
        self.xmin = 1.
        self.xmax = 11.
        self.no_iters = 0
        self.det_material = 0
        self.be_window_thick = 0.
        self.det_chip_thick = 0.
        self.GE_dead_layer_thick = 0.

#-----------------------------------------------------------------------------        
class Cs:
    def __init__(self, nfp):   
        st = ''.join([' ']*32)   
        self.name = np.array([st]*nfp)
        self.val = np.zeros((nfp))
        self.fixed = np.zeros((nfp))
        self.batch = np.zeros((nfp,5))
        self.max = np.zeros((nfp))
        self.min = np.zeros((nfp))
        self.maxstep = np.zeros((nfp))
        self.minstep = np.zeros((nfp))
        self.use = np.zeros((nfp), dtype = np.int32)
        self.mmax = np.zeros((nfp))
        self.mmin = np.zeros((nfp))
        self.mmaxstep = np.zeros((nfp))
        self.mminstep = np.zeros((nfp))
        

#-----------------------------------------------------------------------------        
class fitp_info:
    def __init__(self):
        
        self.keywords = Ckeywords()
        
        tail_pos = np.arange(12) + 28
        kele_pos = np.arange(len(kele))+np.amax(tail_pos)+1
        lele_pos = np.arange(len(lele))+np.amax(kele_pos)+1
        mele_pos = np.arange(len(mele))+np.amax(lele_pos)+1
        
        # in add pars, have room for up to 7 pileup peaks
        npars = np.amax(mele_pos)-np.amin(kele_pos)+1+10
        self.add_pars = np.empty( (npars, 12), dtype=object)      
        for i in range(npars):
            for j in range(12):
                self.add_pars[i,j] = Cpar()
        
        n_fitp = np.amax(mele_pos)+1

        self.g = Cg()
        
        self.s = Cs(n_fitp)
        
        
        self.g.n_fitp = n_fitp

#-----------------------------------------------------------------------------
class maps_fit_parameters:
    def __init__(self):   
        pass
    
#-----------------------------------------------------------------------------
    def define_fitp(self, beamline, info_elements): 
        
        fitp = fitp_info()
        
        n_spectra_to_fit = 3
        
        ## ======= definitions for batch ======= 
        ## use: 0=unlimited, 1=fixed, 2=lower and higher limit, 
        ## 3=lower limit, 4=higher limit; 5=automated
        ## 1 batch_a: matrix batch fit
        ## 2 batch_b: batch_fit_wo_tails
        ## 3 batch_c: batch_fit_w_tails
        ## 4 batch fit with free E, everything else fixed
         
        fitp.keywords.twobm = 0

        fitp.keywords.twobm = 0
        if beamline == '2-ID-E' :
            fitp.keywords.twoide = 1 
        else:
            fitp.keywords.twoide = 0
        if beamline == '2-ID-D' :
            fitp.keywords.twoidd = 1 
        else:
            fitp.keywords.twoidd = 0
            fitp.keywords.fixed_fwhm = 0
   
   

        fitp.s.name[fitp.keywords.energy_pos] = ['e_offset', 'e_linear', 'e_quadratic'] # 1-3
        fitp.s.val[fitp.keywords.energy_pos] = [0., 1., 0.]
        fitp.s.use[fitp.keywords.energy_pos] = [0., 0., 1.]
        fitp.s.max[fitp.keywords.energy_pos] = [0.2, 0.1, 0.0001]
        fitp.s.min[fitp.keywords.energy_pos] = [-0.2, 0.001, -0.0001]      
        fitp.s.batch[fitp.keywords.energy_pos, 0:2] = 1.
        fitp.s.batch[fitp.keywords.energy_pos, 2:5] = 2.
        

        fitp.s.name[fitp.keywords.fwhm_pos] = ['fwhm_offset', 'fwhm_fanoprime'] #4-5
        fitp.s.val[fitp.keywords.fwhm_pos] = [0.12, .00012]
        fitp.s.use[fitp.keywords.fwhm_pos] = [2., 2.]
        fitp.s.min[fitp.keywords.fwhm_pos] = [.005, .000001]
        fitp.s.max[fitp.keywords.fwhm_pos] = [.5, .05]
        fitp.s.batch[fitp.keywords.fwhm_pos,0:2] = 1
        fitp.s.batch[fitp.keywords.fwhm_pos,2:4] = 2.
        fitp.s.batch[fitp.keywords.fwhm_pos, 4] = 1.
    
        fitp.s.name[fitp.keywords.coherent_pos] = ['coherent_sct_energy', 'coherent_sct_amplitude'] 
        fitp.s.val[fitp.keywords.coherent_pos] = [9.99, 10.]
        fitp.s.use[fitp.keywords.coherent_pos] = [2, 5]
        fitp.s.min[fitp.keywords.coherent_pos] = [9.4, 0.]
        fitp.s.max[fitp.keywords.coherent_pos] = [10.4, 0.]
        fitp.s.batch[fitp.keywords.coherent_pos,0] = [1, 5]
        fitp.s.batch[fitp.keywords.coherent_pos,1] = [1, 5]
        fitp.s.batch[fitp.keywords.coherent_pos,4] = [1, 5]
        fitp.s.batch[fitp.keywords.coherent_pos,2] = [2, 5]
        fitp.s.batch[fitp.keywords.coherent_pos,3] = [2, 5]
        #to disable coherent UNCOMMENT below
        #fitp.s.batch[fitp.keywords.coherent_pos,*] = [1]
        
        

        fitp.s.name[fitp.keywords.compton_pos] = ['compton_angle', 'compton_fwhm_corr', 'compton_amplitude', 
                                                  'compton_f_step', 'compton_f_tail', 'compton_gamma', 
                                                  'compton_hi_f_tail', 'compton_hi_gamma'] 
        fitp.s.val[fitp.keywords.compton_pos] = [90., 1., 10., 0., 0.0000, 1., 0., 1.]
        fitp.s.use[fitp.keywords.compton_pos] = [1., 1., 5., 1., 1., 1., 1., 1.]
        fitp.s.min[fitp.keywords.compton_pos] = [70., 1., 0., 0., 0.0000, 0.1, 0.000001, 0.1]
        fitp.s.max[fitp.keywords.compton_pos] = [170., 4., 0., 0., 0., 10., 1.0, 3.]
        fitp.s.batch[fitp.keywords.compton_pos,0] = [1., 1., 5, 1, 1, 1, 1, 1]
        fitp.s.batch[fitp.keywords.compton_pos,1] = [1., 1., 5, 1, 1, 1, 1, 1]
        fitp.s.batch[fitp.keywords.compton_pos,4] = [1., 1., 5, 1, 1, 1, 1, 1]
        fitp.s.batch[fitp.keywords.compton_pos,2] = [2., 2., 5, 1, 3, 1, 2, 1]
        fitp.s.batch[fitp.keywords.compton_pos,3] = [2., 2., 5, 1, 3, 2, 2, 2]
        # to disable compton UNCOMMENT below
        # fitp.s.batch[fitp.keywords.compton_pos,*] = [1]

        fitp.s.name[fitp.keywords.added_params] = ['snip_width', 'si_escape', 'ge_escape', 'linear'] + ['pileup']*9
        fitp.s.val[fitp.keywords.added_params] = [0.15, 0., 0., 0.]+ [1e-10]*9
        fitp.s.use[fitp.keywords.added_params] = [1, 1, 1, 1] + [1]*9
        fitp.s.min[fitp.keywords.added_params] = [0.1, 0., 0., 0.] + [ -10]*9
    
        fitp.s.max[fitp.keywords.added_params] = [2.*np.sqrt(2.), 1., 1., 1.]+ [ 1.]*9
        fitp.s.batch[fitp.keywords.added_params,0] = [1, 1, 1, 1]+ [ 1]*9
        fitp.s.batch[fitp.keywords.added_params,1] = [1, 1, 1, 1]+ [ 1]*9
        fitp.s.batch[fitp.keywords.added_params,2] = [1, 1, 1, 1]+ [ 1]*9
        fitp.s.batch[fitp.keywords.added_params,3] = [1, 1, 1, 1]+ [ 1]*9
        fitp.s.batch[fitp.keywords.added_params,4] = [1, 1, 1, 1]+ [ 1]*9


        fitp.s.name[fitp.keywords.tail_pos] = ['f_step_offset', 'f_step_linear', 'f_step_quadratic', 
                                               'f_tail_offset', 'f_tail_linear', 'f_tail_quadratic', 
                                               'gamma_offset', 'gamma_linear', 'gamma_quadratic', 
                                               'kb_f_tail_offset', 'kb_f_tail_linear', 'kb_f_tail_quadratic']
    
        fitp.s.val[fitp.keywords.tail_pos] = [0., 0., 0., 0.04, 0.01, 0., 2., 0., 0., 0., 0., 0.]       
        # mdj tail parameters are below
        #fitp.s.val[fitp.keywords.tail_pos] = [0., 0., 0., 0.00, 0.02, 0., 2., 0., 0. ]       
    
        fitp.s.batch[fitp.keywords.tail_pos,0:3] = 1.
        fitp.s.batch[fitp.keywords.tail_pos,4] = 1.
        fitp.s.batch[fitp.keywords.tail_pos,3] = [1., 1., 1., 2., 2., 1., 1., 1., 1., 
                                               2., 2., 1.]    
        # mdj tail parameters are below
        # fitp.s.batch[fitp.keywords.tail_pos,3] = [1., 1., 1., 1., 2., 1., 2., 1., 1.]    
    
        fitp.s.use[fitp.keywords.tail_pos] = 1.
        fitp.s.min[fitp.keywords.tail_pos] = [0., 0., 0., 
                                              0.0, 0.0, 0., 
                                              0.1, 0.0, 0., 
                                              0., 0., 0.]
        fitp.s.max[fitp.keywords.tail_pos] = [1., 1., 0., 
                                              0.1, 1., 0.01, 
                                              10., 3., 0., 
                                              0.2, 0.02, 0.]

        fitp.s.name[fitp.keywords.kele_pos] = kele
        fitp.s.use[fitp.keywords.kele_pos] = 5
        test = np.array(['Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Ti', 'V', 'Cr', 
                'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn'])


        for i in range(len(fitp.keywords.kele_pos)):
            for j in range(len(info_elements)):
                if info_elements[j].name == kele[i]:
                    wo = np.where(test == kele[i])
                    fitp.s.val[fitp.keywords.kele_pos[i]] = 0.
                    if len(wo[0]) > 0: 
                        fitp.s.use[fitp.keywords.kele_pos[i]] = 5
                    else:   
                        fitp.s.use[fitp.keywords.kele_pos[i]] = 1             
 
                    fitp.add_pars[i, 0].name = kele[i]
                    fitp.add_pars[i, 0].energy = info_elements[j].xrf['ka1']
                    fitp.add_pars[i, 1].energy = info_elements[j].xrf['ka2']
                    fitp.add_pars[i, 2].energy = info_elements[j].xrf['kb1']
                    fitp.add_pars[i, 3].energy = info_elements[j].xrf['kb2']

                    fitp.add_pars[i, 0].ratio = 1.
                    fitp.add_pars[i, 1].ratio = info_elements[j].xrf_abs_yield['ka2']/info_elements[j].xrf_abs_yield['ka1']
                    fitp.add_pars[i, 2].ratio = info_elements[j].xrf_abs_yield['kb1']/info_elements[j].xrf_abs_yield['ka1']
                    fitp.add_pars[i, 3].ratio = info_elements[j].xrf_abs_yield['kb2']/info_elements[j].xrf_abs_yield['ka1'] 

                    fitp.add_pars[i, 0].type = 1
                    fitp.add_pars[i, 1].type = 1
                    fitp.add_pars[i, 2].type = 2
                    fitp.add_pars[i, 3].type = 2

                else:
                    fitp.s.val[fitp.keywords.kele_pos[i]] = 0.
                    fitp.s.use[fitp.keywords.kele_pos[i]] = 1             

        
        fitp.s.name[fitp.keywords.lele_pos] = lele
        fitp.s.use[fitp.keywords.lele_pos] = 1
        for i in range(len(fitp.keywords.lele_pos)):
            ii = i+np.amax(fitp.keywords.kele_pos)-np.amin(fitp.keywords.kele_pos)+1
            for j in range(len(info_elements)):
                elname = lele[i]
                elname = elname[:-2]
                if info_elements[j].name == elname :
                    if fitp.s.use[fitp.keywords.lele_pos[i]] != 1 :
                        fitp.s.val[fitp.keywords.lele_pos[i]] = 10. 
                    else:
                        fitp.s.val[fitp.keywords.lele_pos[i]] = 0.                   
                    fitp.add_pars[ii, 0].name = lele[i]
                    fitp.add_pars[ii, 0].energy = info_elements[j].xrf['la1']
                    fitp.add_pars[ii, 1].energy = info_elements[j].xrf['la2']
                    fitp.add_pars[ii, 2].energy = info_elements[j].xrf['lb1']
                    fitp.add_pars[ii, 3].energy = info_elements[j].xrf['lb2']
                    fitp.add_pars[ii, 4].energy = info_elements[j].xrf['lb3']
                    fitp.add_pars[ii, 5].energy = info_elements[j].xrf['lb4']
                    fitp.add_pars[ii, 6].energy = info_elements[j].xrf['lg1']
                    fitp.add_pars[ii, 7].energy = info_elements[j].xrf['lg2']
                    fitp.add_pars[ii, 8].energy = info_elements[j].xrf['lg3']
                    fitp.add_pars[ii, 9].energy = info_elements[j].xrf['lg4']
                    fitp.add_pars[ii, 10].energy = info_elements[j].xrf['ll']
                    fitp.add_pars[ii, 11].energy = info_elements[j].xrf['ln']
          
                    fitp.add_pars[ii, 0].ratio = 1.
                    fitp.add_pars[ii, 1].ratio = info_elements[j].xrf_abs_yield['la2']/info_elements[j].xrf_abs_yield['la1']
                    fitp.add_pars[ii, 2].ratio = info_elements[j].xrf_abs_yield['lb1']/info_elements[j].xrf_abs_yield['la1']
                    fitp.add_pars[ii, 3].ratio = info_elements[j].xrf_abs_yield['lb2']/info_elements[j].xrf_abs_yield['la1']
                    fitp.add_pars[ii, 4].ratio = info_elements[j].xrf_abs_yield['lb3']/info_elements[j].xrf_abs_yield['la1']
                    fitp.add_pars[ii, 5].ratio = info_elements[j].xrf_abs_yield['lb4']/info_elements[j].xrf_abs_yield['la1']
                    fitp.add_pars[ii, 6].ratio = info_elements[j].xrf_abs_yield['lg1']/info_elements[j].xrf_abs_yield['la1']
                    fitp.add_pars[ii, 7].ratio = info_elements[j].xrf_abs_yield['lg2']/info_elements[j].xrf_abs_yield['la1']
                    fitp.add_pars[ii, 8].ratio = info_elements[j].xrf_abs_yield['lg3']/info_elements[j].xrf_abs_yield['la1']
                    fitp.add_pars[ii, 9].ratio = info_elements[j].xrf_abs_yield['lg4']/info_elements[j].xrf_abs_yield['la1']
                    fitp.add_pars[ii, 10].ratio = info_elements[j].xrf_abs_yield['ll']/info_elements[j].xrf_abs_yield['la1']
                    fitp.add_pars[ii, 11].ratio = info_elements[j].xrf_abs_yield['ln']/info_elements[j].xrf_abs_yield['la1']

                    for t in range(12):
                        fitp.add_pars[ii, t].type = 3
                else: 
                    fitp.s.val[fitp.keywords.lele_pos[i]] = 0.
                    fitp.s.use[fitp.keywords.lele_pos[i]] = 1             

    
        
    
        fitp.s.name[fitp.keywords.mele_pos] = mele
        fitp.s.use[fitp.keywords.mele_pos] = 1
        for i in range(len(fitp.keywords.mele_pos)):
            ii = i+np.amax(fitp.keywords.lele_pos)-np.amin(fitp.keywords.kele_pos)+1
            if fitp.s.use[fitp.keywords.mele_pos[i]] != 1 :
                fitp.s.val[fitp.keywords.mele_pos[i]] = 10.
            else:
                fitp.s.val[fitp.keywords.mele_pos[i]] = 0.
            fitp.add_pars[ii, 0].name = mele[i]
            if mele[i] == 'Pb_M' : 
                fitp.add_pars[ii, 0].energy = 2.383
            if mele[i] == 'Au_M' : 
                fitp.add_pars[ii, 0].energy = 2.123
                fitp.add_pars[ii, 1].energy = 2.2
                fitp.add_pars[ii, 1].ratio = 0.5
                fitp.add_pars[ii, 1].type = 7
 
            if mele[i] == 'Pt_M' :
                fitp.add_pars[ii, 0].energy = 2.0505
            # if mele[i] == 'noise' : fitp.add_pars[ii, 0].energy = 3.42
            if mele[i] == 'noise' : fitp.add_pars[ii, 0].energy = 6.3
            fitp.add_pars[ii, 0].type = 7
       
            if mele[i] == 'U_M' :
                fitp.add_pars[ii, 0].energy = 3.168
                fitp.add_pars[ii, 1].energy = 3.338
                fitp.add_pars[ii, 2].energy = 3.5627
                fitp.add_pars[ii, 3].energy = 2.505

                fitp.add_pars[ii, 1].ratio = 0.7
                fitp.add_pars[ii, 2].ratio = 0.23
                fitp.add_pars[ii, 3].ratio = 0.05
                
                for t in range(4):
                    fitp.add_pars[ii, t].type = 7
      
            fitp.add_pars[ii, 0].ratio = 1.0



        # below is the 1/e absorption length (in microns), 1/mu_l
        # temp = 1.239852/(add_pars[i, k].energy *4.*!pi*beta)
        # now calculate mass attenuation coeff = mu_l/rho
        # add_pars[i, k].Ge_mu = 1./(temp*5.323)
        # to convert mu from cm2*um to cm3, need to mutiply Ge_mu*10000.
        AVOGADRO = 6.02204531e23
        HC_ANGSTROMS = 12398.52
        RE = 2.817938070e-13        # in cm
        henkedata = henke.henke()
        for k in range(2):
            if k == 0 : 
                name = 'Ge'
                density = 5.323
       
            if k == 1 :
                name = 'Si'
                density = 2.3

            z_array, atwt =  henkedata.compound(name, density)    
            wo = np.where(z_array == 1.)
            if 1. not in z_array:
                print  'encountered error, will return'
  
            z = wo[0][0]+1
            if (atwt != 0.0) : 
                molecules_per_cc = density * AVOGADRO / atwt 
            else : 
                molecules_per_cc = 0.0
                
            energies_all, f1_all, f2_all, energies_extra, f1_extra, f2_extra  = henkedata.extra(ielement = z-1)
                         
            for i in range( np.amax(fitp.keywords.mele_pos)-np.amin(fitp.keywords.kele_pos)+1+10) : 
                for j in range(12) : 
                    energy = fitp.add_pars[i, j].energy*1000.
                    if energy == 0. : 
                        continue 
                    wavelength_angstroms = HC_ANGSTROMS/energy
                    # This constant has wavelength in angstroms and then
                    # they are converted to centimeters.
                    constant = RE * (1.0e-16 * wavelength_angstroms * wavelength_angstroms) * molecules_per_cc / (2.0 * np.math.pi)             
                
                    wo = np.where(energies_all > energy)
                    #the first value that is larger than energy must be the closest value
                    if wo[0].size > 0: 
                        hi_e_ind = wo[0][0]
                    else: 
                        hi_e_ind = 0     
                          
                    wo = np.where(energies_all < energy)
                    # the last value that is smaller than energy must be the closest value
                    if wo[0].size > 0: 
                        lo_e_ind = wo[0][-1]  
                    else : 
                        lo_e_ind = len(energies_all)-1        
                    ln_lower_energy = np.log(energies_all[lo_e_ind])
                    ln_higher_energy = np.log(energies_all[hi_e_ind])
                    fraction = (np.log(energy)-ln_lower_energy)/(ln_higher_energy-ln_lower_energy)             
                    ln_f2_lower = np.log(np.abs(f2_all[lo_e_ind]))
                    ln_f2_higher = np.log(np.abs(f2_all[hi_e_ind]))
                    f2 = np.exp(ln_f2_lower + fraction * (ln_f2_higher - ln_f2_lower))                   
                    beta = constant * f2
                    if k == 0 : 
                        fitp.add_pars[i, j].Ge_mu = (energy *4.*np.math.pi*beta)/(5.323* 1.239852)*10000.
                    if k == 1 : 
                        fitp.add_pars[i, j].Si_mu = (energy *4.*np.math.pi*beta)/(2.33* 1.239852)*10000.
      
        # by default use Germanium detector, if not, need to change in override file
        # fraction and width usually are the same. However, have two different
        # parameters, as this allows, e.g., to change the Si tailing specifically
        # via the override file
        parsdims = fitp.add_pars.shape

        for i in range(parsdims[0]):
            for j in range(parsdims[1]):
                fitp.add_pars[i,j].mu_fraction = fitp.add_pars[i,j].Ge_mu
      
        self.fitp = fitp
        

        return fitp
 
#-----------------------------------------------------------------------------
# Read fit parameters from a file
    def read_fitp(self, filename, info_elements, det = 0, string_only = False): 
        
        fitp = self.fitp
        
        
        n_detector_elements = 1 
        
        keywords = fitp.keywords
        add_pars = fitp.add_pars
        
        verbose = False
        
        f = open(filename, 'rt')
        for line in f:
            if ':' in line : 
                slist = line.split(':')
                tag = slist[0]
                value = ''.join(slist[1:])
                
                
                if tag == 'VERSION': version = float(value)
                
                elif tag == 'DATE' : date = value
                
                elif tag == 'IDENTIFYING_NAME_[WHATEVERE_YOU_LIKE]' : string = value
                
                elif tag == 'ELEMENTS_TO_FIT' :  
                    test_string = value.split(',')
                    test_string = [x.strip() for x in test_string]    
                    if string_only : 
                        f.close() 
                        return test_string
                    
                elif tag == 'ELEMENTS_WITH_PILEUP' :  
                    pileup_string = value.split(',')
                    pileup_string = [x.strip() for x in pileup_string]            
         
                elif tag == 'CAL_OFFSET_[E_OFFSET]':  
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.energy_pos[0]] = temp[det]            
 
                elif tag == 'CAL_OFFSET_[E_OFFSET]_MAX':  
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.max[keywords.energy_pos[0]] = temp[det]
        
                elif tag == 'CAL_OFFSET_[E_OFFSET]_MIN':  
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.min[keywords.energy_pos[0]] = temp[det]
                   
                elif tag == 'CAL_SLOPE_[E_LINEAR]' :  
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.energy_pos[1]] = temp[det]
          
                elif tag == 'CAL_SLOPE_[E_LINEAR]_MAX' : 
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.max[keywords.energy_pos[1]] = temp[det]
          
                elif tag == 'CAL_SLOPE_[E_LINEAR]_MIN' :  
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.min[keywords.energy_pos[1]] = temp[det]
          
                elif tag == 'CAL_QUAD_[E_QUADRATIC]'  :  
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.energy_pos[2]] = temp[det]
                    # if this parameter is zero, do NOT fit 
                    if (fitp.s.val[keywords.energy_pos[2]]) == 0. : 
                        fitp.s.batch[keywords.energy_pos[2],:] = 1
                        fitp.s.use[keywords.energy_pos[2]] = 1
                    else: 
                        fitp.s.batch[keywords.energy_pos[2],:] = fitp.s.batch[keywords.energy_pos[1],:] 
                        fitp.s.use[keywords.energy_pos[2]] = fitp.s.use[keywords.energy_pos[1]]

                elif tag == 'CAL_QUAD_[E_QUADRATIC]_MAX' :  
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.max[keywords.energy_pos[2]] = temp[det]
         
                elif tag == 'CAL_QUAD_[E_QUADRATIC]_MIN' :  
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.min[keywords.energy_pos[2]] = temp[det]
 
                elif tag == 'FWHM_OFFSET' :  
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.fwhm_pos[0]] = temp[det]
          
                elif tag == 'FWHM_FANOPRIME'  :  
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.fwhm_pos[1]] = temp[det]
         
                elif tag == 'COHERENT_SCT_ENERGY' :  
                    temp = float(value)
                    if verbose: print tag, temp
                    fitp.s.val[keywords.coherent_pos[0]] = temp
           
                elif tag == 'COHERENT_SCT_ENERGY_MAX' : 
                    temp = float(value)
                    if verbose: print tag, temp
                    fitp.s.max[keywords.coherent_pos[0]] = temp
           
                elif tag == 'COHERENT_SCT_ENERGY_MIN' : 
                    temp = float(value)
                    if verbose: print tag, temp
                    fitp.s.min[keywords.coherent_pos[0]] = temp
 
                elif tag == 'COMPTON_ANGLE' :  
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.compton_pos[0]] = temp[det]
          
                elif tag == 'COMPTON_ANGLE_MAX' :
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.max[keywords.compton_pos[0]] = temp[det]
          
                elif tag == 'COMPTON_ANGLE_MIN' : 
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.min[keywords.compton_pos[0]] = temp[det]
          
                elif tag == 'COMPTON_FWHM_CORR' :  
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.compton_pos[1]] = temp[det]
          
                elif tag == 'COMPTON_STEP' : 
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.compton_pos[3]] = temp[det]
           
                elif tag == 'COMPTON_F_TAIL' :  
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.compton_pos[4]] = temp[det]
          
                elif tag == 'COMPTON_GAMMA' : 
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.compton_pos[5]] = temp[det]
                    
                elif tag == 'COMPTON_HI_F_TAIL' : 
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.compton_pos[6]] = temp[det]
         
                elif tag == 'COMPTON_HI_GAMMA' : 
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.compton_pos[7]] = temp[det]
          
                elif tag == 'STEP_OFFSET' :
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.tail_pos[0]] = temp[det]
           
                elif tag == 'STEP_LINEAR' :  
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.tail_pos[1]] = temp[det]
          
                elif tag == 'STEP_QUADRATIC' :  
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.tail_pos[2]] = temp[det]
         
                elif tag == 'F_TAIL_OFFSET' :  
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.tail_pos[3]] = temp[det]
          
                elif tag == 'F_TAIL_LINEAR' : 
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.tail_pos[4]] = temp[det]
           
                elif tag == 'F_TAIL_QUADRATIC' : 
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.tail_pos[5]] = temp[det]
         
                elif tag == 'KB_F_TAIL_OFFSET' :   
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.tail_pos[9]] = temp[det]
            
                elif tag == 'KB_F_TAIL_LINEAR' :   
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.tail_pos[10]] = temp[det]
            
                elif tag == 'KB_F_TAIL_QUADRATIC' :   
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.tail_pos[11]] = temp[det]
            
                elif tag == 'GAMMA_OFFSET' :   
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.tail_pos[6]] = temp[det]
            
                elif tag == 'GAMMA_LINEAR' :   
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.tail_pos[7]] = temp[det]
            
                elif tag == 'GAMMA_QUADRATIC' :   
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.tail_pos[8]] = temp[det]
            
                elif tag == 'SNIP_WIDTH' :   
                    temp = map(float,value.split(','))
                    if verbose: print tag, temp
                    fitp.s.val[keywords.added_params[0]] = temp[det]
            
                elif tag == 'FIT_SNIP_WIDTH' :   
                    temp = int(value)
                    if verbose: print tag, temp
                    if temp == 1 :  
                        fitp.s.use[keywords.added_params[0]] = 2
                        fitp.s.batch[keywords.added_params[0],1:3] = 2
           
            
                elif tag == 'DETECTOR_MATERIAL' :   
                    temp = int(value)
                    if verbose: print tag, temp
                    fitp.g.det_material = temp
                    parsdims = fitp.add_pars.shape
                    if fitp.g.det_material == 1 : 
                        for i in range(parsdims[0]):
                            for j in range(parsdims[1]):
                                fitp.add_pars[i,j].mu_fraction = fitp.add_pars[i,j].Si_mu
                    else:   
                        for i in range(parsdims[0]):
                            for j in range(parsdims[1]):
                                fitp.add_pars[i,j].mu_fraction = fitp.add_pars[i,j].Ge_mu
            
                elif tag == 'BE_WINDOW_THICKNESS' :   
                    temp = float(value)
                    if verbose: print tag, temp
                    fitp.g.be_window_thick = float(temp)*1000.
            
                elif tag == 'DET_CHIP_THICKNESS' :   
                    temp = float(value)
                    if verbose: print tag, temp
                    fitp.g.det_chip_thick = float(temp)*1000.
            
                elif tag == 'GE_DEAD_LAYER' :   
                    temp = float(value)
                    if verbose: print tag, temp
                    fitp.g.GE_dead_layer_thick = float(temp)*1000.
            
                elif tag == 'MAX_ENERGY_TO_FIT' :   
                    temp = float(value)
                    if verbose: print tag, temp
                    fitp.g.xmax = temp
            
                elif tag == 'MIN_ENERGY_TO_FIT' :   
                    temp = float(value)
                    if verbose: print tag, temp
                    fitp.g.xmin = temp
            
                elif tag == 'BRANCHING_FAMILY_ADJUSTMENT_L' : 
                    temp_string = value.split(',')
                    temp_string = [x.strip() for x in temp_string]            
                    wo = np.where(fitp.s.name == temp_string[0])

                    if verbose: print 'found BRANCHING_FAMILY_ADJUSTMENT_L', temp_string

                    if (wo[0].size > 0) and ( len(temp_string) == 4) : 
                        ii = wo[0][0] - np.amin(keywords.kele_pos) 
                        factor_l1 = float(temp_string[1])
                        factor_l2 = float(temp_string[2])
                        factor_l3 = float(temp_string[3])
                        name = temp_string[0].strip()
                        el_names = [x.name for x in  info_elements]
                        if name[:-2] in el_names:
                            j = el_names.index(name[:-2])
                        else: 
                            j = -1

                        # below, adjust branching ratio of L family, i.e, L1 vs L 2, vs
                        # L3 based lines
                        if j > 0 :                    
                            add_pars[ii, 0].ratio = 1.
                            add_pars[ii, 1].ratio = info_elements[j].xrf_abs_yield['la2']/info_elements[j].xrf_abs_yield['la1']
                            add_pars[ii, 2].ratio = info_elements[j].xrf_abs_yield['lb1']/info_elements[j].xrf_abs_yield['la1']
                            add_pars[ii, 3].ratio = info_elements[j].xrf_abs_yield['lb2']/info_elements[j].xrf_abs_yield['la1']
                            add_pars[ii, 4].ratio = info_elements[j].xrf_abs_yield['lb3']/info_elements[j].xrf_abs_yield['la1']
                            add_pars[ii, 5].ratio = info_elements[j].xrf_abs_yield['lb4']/info_elements[j].xrf_abs_yield['la1']
                            add_pars[ii, 6].ratio = info_elements[j].xrf_abs_yield['lg1']/info_elements[j].xrf_abs_yield['la1']
                            add_pars[ii, 7].ratio = info_elements[j].xrf_abs_yield['lg2']/info_elements[j].xrf_abs_yield['la1']
                            add_pars[ii, 8].ratio = info_elements[j].xrf_abs_yield['lg3']/info_elements[j].xrf_abs_yield['la1']
                            add_pars[ii, 9].ratio = info_elements[j].xrf_abs_yield['lg4']/info_elements[j].xrf_abs_yield['la1']
                            add_pars[ii, 10].ratio = info_elements[j].xrf_abs_yield['ll']/info_elements[j].xrf_abs_yield['la1']
                            add_pars[ii, 11].ratio = info_elements[j].xrf_abs_yield['ln']/info_elements[j].xrf_abs_yield['la1']
                            add_pars[ii, 0].ratio = add_pars[ii, 0].ratio * factor_l3
                            add_pars[ii, 1].ratio = add_pars[ii, 1].ratio * factor_l3
                            add_pars[ii, 3].ratio = add_pars[ii, 3].ratio * factor_l3
                            add_pars[ii, 10].ratio = add_pars[ii, 10].ratio * factor_l3
                            add_pars[ii, 2].ratio = add_pars[ii, 2].ratio * factor_l2
                            add_pars[ii, 6].ratio = add_pars[ii, 6].ratio * factor_l2
                            add_pars[ii, 11].ratio = add_pars[ii, 11].ratio * factor_l2
                            add_pars[ii, 4].ratio = add_pars[ii, 4].ratio * factor_l1
                            add_pars[ii, 5].ratio = add_pars[ii, 5].ratio * factor_l1
                            add_pars[ii, 7].ratio = add_pars[ii, 7].ratio * factor_l1
                            add_pars[ii, 8].ratio = add_pars[ii, 8].ratio * factor_l1
                            add_pars[ii, 9].ratio = add_pars[ii, 9].ratio * factor_l1

                elif tag == 'BRANCHING_RATIO_ADJUSTMENT_L':                    
                    temp_string = value.split(',')
                    temp_string = [x.strip() for x in temp_string]            
                    wo = np.where(fitp.s.name == temp_string[0])
                    if verbose: print tag, temp_string                    

                    if (wo[0].size > 0) and ( len(temp_string) == 13) : 
                        ii = wo[0][0] - np.amin(keywords.kele_pos)
                        name = temp_string[0].strip()
                        el_names = [x.name for x in  info_elements]
                        
                        # adjust branching ratios within families, but all relative to La1
                        if (len(temp_string) >= 13) : 
                            if verbose: print 'found BRANCHING_RATIO_ADJUSTMENT_L', temp_string
                            for jj in range(12): 
                                old_value = add_pars[ii, jj].ratio
                                add_pars[ii, jj].ratio = add_pars[ii, jj].ratio*float(temp_string[(jj+1)])
                                if verbose: print 'old_value: ', old_value, ' new value: ', add_pars[ii, jj].ratio

         
         
                elif tag == 'BRANCHING_RATIO_ADJUSTMENT_K' :  
                    temp_string = value.split(',')
                    temp_string = [x.strip() for x in temp_string]            
                    wo = np.where(fitp.s.name == temp_string[0])
                    if verbose: print tag, temp_string    
                    
                    if (wo[0].size > 0) and ( len(temp_string) == 5) : 
                        ii = wo[0][0] - np.amin(keywords.kele_pos)
                        name = temp_string[0].strip()
                        el_names = [x.name for x in  info_elements]
                        if name[:-2] in el_names:
                            j = el_names.index(name[:-2])
                        else: 
                            j = -1

                        # adjust branching ratios within families, but all relative to Ka1
                        if j > 0 : 
                            add_pars[ii, 0].ratio = 1.
                            add_pars[ii, 1].ratio = info_elements[j].xrf_abs_yield['ka2']/info_elements[j].xrf_abs_yield['ka1']
                            add_pars[ii, 2].ratio = info_elements[j].xrf_abs_yield['kb1']/info_elements[j].xrf_abs_yield['ka1']
                            add_pars[ii, 3].ratio = info_elements[j].xrf_abs_yield['kb2']/info_elements[j].xrf_abs_yield['ka1']

                        if (len(temp_string) >= 5) : 
                            for jj in range(4):     
                                old_value = add_pars[ii, jj].ratio   
                                add_pars[ii, jj].ratio = add_pars[ii, jj].ratio*float(temp_string[(jj+1)])
                                if verbose: print 'old_value: ', old_value, ' new value: ', add_pars[ii, jj].ratio
    
    
                elif tag == 'TAIL_FRACTION_ADJUST_SI' : 
                    temp = float(value)
                    if verbose: print tag, temp
                    wo = np.where(fitp.s.name == 'Si')
                    ii = wo[0][0] - np.amin(keywords.kele_pos)
                    add_pars[ii, :].mu_fraction = value * add_pars[ii, :].mu_fraction 
          
                elif tag == 'TAIL_WIDTH_ADJUST_SI' :
                    temp = float(value)
                    if verbose: print tag, temp
                    wo = np.where(fitp.s.name == 'Si')
                    ii = wo[0][0] - np.amin(keywords.kele_pos)             
                    add_pars[ii, 0:3].width_multi = value
           
                elif tag == 'SI_ESCAPE_FACTOR' : 
                    temp = float(value)
                    if verbose: print tag, temp
                    fitp.s.val[keywords.added_params[1]] = temp
          
                elif tag == 'GE_ESCAPE_FACTOR' : 
                    temp = float(value)
                    if verbose: print tag, temp
                    fitp.s.val[keywords.added_params[2]] = temp
         
                elif tag == 'ESCAPE_LINEAR' :  
                    temp = float(value)
                    if verbose: print tag, temp
                    fitp.s.val[keywords.added_params[3]] = temp
           
                elif tag == 'SI_ESCAPE_ENABLE' : 
                    temp = float(value)
                    if verbose: print tag, temp
                    if temp == 0. :
                        fitp.s.use[keywords.added_params[1]] = 1
                        fitp.s.batch[keywords.added_params[1],1:4] = 1
                        fitp.s.use[keywords.added_params[3]] = 1
                        fitp.s.batch[keywords.added_params[3],1:4] = 1
                    else: 
                        fitp.s.use[keywords.added_params[1]] = 2
                        fitp.s.batch[keywords.added_params[1],1:4] = 2
                        fitp.s.use[keywords.added_params[3]] = 2
                        fitp.s.batch[keywords.added_params[3],1:4] = 2
    
                elif tag == 'GE_ESCAPE_ENABLE' :  
                    temp = float(value)
                    if verbose: print tag, temp
                    if temp == 0. :
                        fitp.s.use[keywords.added_params[2]] = 1
                        fitp.s.batch[keywords.added_params[2],1:4] = 1
                        fitp.s.use[keywords.added_params[3]] = 1
                        fitp.s.batch[keywords.added_params[3],1:4] = 1
                    else:
                        fitp.s[keywords.added_params[2]].use = 2
                        fitp.s.batch[keywords.added_params[2],1:4] = 2
                        fitp.s[keywords.added_params[3]].use = 2
                        fitp.s.batch[keywords.added_params[3],1:4] = 2

        
        f.close()
        
        return fitp, test_string, pileup_string
    
#-----------------------------------------------------------------------------    
    def write_fit_parameters(self, main, fitp, filename, test_string, pileup_string = '', suffix = ''):
        
        
        print 'saving ', filename
        
        n_detector_elements = 1     
 
        srcurrent = -1
        us_ic = -1
        ds_ic = -1
        dpc1_ic = -1
        dpc2_ic = -1
        cfg_1 = -1
        cfg_2 = -1
        cfg_3 = -1
        cfg_4 = -1
        cfg_5 = -1
        cfg_6 = -1
        cfg_7 = -1
        cfg_8 = -1
        cfg_9 = -1
        cfg_10 = -1
        airpath = -1
        ELT1 = -1
        ERT1 = -1
        ICR1 = -1
        OCR1 = -1    
        US_AMP_SENS_NUM = -1
        US_AMP_SENS_UNIT = -1
        DS_AMP_SENS_NUM = -1               
        DS_AMP_SENS_UNIT = -1   
                            
        #Look for override files in main.master_dir
        maps_overridefile = os.path.join(main['master_dir'],'maps_fit_parameters_override.txt')+suffix
        try:
            f = open(maps_overridefile, 'rt')    
            print maps_overridefile, ' exists.'
            f.close()
        except :
            # if i cannot find an override file specific per detector, assuming
            # there is a single overall file.
            maps_overridefile = os.path.join(main['master_dir'],'maps_fit_parameters_override.txt')  
       
        
        try:
            f = open(maps_overridefile, 'rt')    
            #override file exists.
            have_override_file = True
            
            for line in f:
                if ':' in line : 
                    slist = line.split(':')
                    tag = slist[0]
                    value = ':'.join(slist[1:])
  
                    if   tag == 'SRCURRENT' : srcurrent = value.strip()
                    elif tag == 'US_IC'     : us_ic = value.strip()
                    elif tag == 'DS_IC'     : ds_ic = value.strip()
                    elif tag == 'DPC1_IC'   : dpc1_ic = value.strip()
                    elif tag == 'DPC2_IC'   : dpc2_ic = value.strip()
                    elif tag == 'CFG_1'     : cfg_1 = value.strip()
                    elif tag == 'CFG_2'     : cfg_2 = value.strip()
                    elif tag == 'CFG_3'     : cfg_3 = value.strip()
                    elif tag == 'CFG_4'     : cfg_4 = value.strip()
                    elif tag == 'CFG_5'     : cfg_5 = value.strip()
                    elif tag == 'CFG_6'     : cfg_6 = value.strip()
                    elif tag == 'CFG_7'     : cfg_7 = value.strip()
                    elif tag == 'CFG_8'     : cfg_8 = value.strip()
                    elif tag == 'CFG_9'     : cfg_9 = value.strip()
                    elif tag == 'CFG_10'    : cfg_10 = value.strip()
                    elif tag == 'ELT1'      : ELT1 = value.strip()
                    elif tag == 'ERT1'      : ERT1 = value.strip()
                    elif tag == 'ICR1'      : ICR1 = value.strip()
                    elif tag == 'OCR1'      : OCR1 = value.strip() 
                    elif tag == 'AIRPATH'   : airpath = value.strip()              
                    elif tag == 'US_AMP_SENS_NUM'  : US_AMP_SENS_NUM = value
                    elif tag == 'US_AMP_SENS_UNIT' : US_AMP_SENS_UNIT = value
                    elif tag == 'DS_AMP_SENS_NUM' : DS_AMP_SENS_NUM = value                
                    elif tag == 'DS_AMP_SENS_UNIT' : DS_AMP_SENS_UNIT = value              
            f.close()
        except:
            pass
 
        f = open(filename, 'w')
    
        print>>f, '   This file will override default fit settings for the maps program for a 3 element detector remove: removeme_*elementdetector_to make it work. '
        print>>f, '   note, the filename MUST be maps_fit_parameters_override.txt'
    
        print>>f, 'VERSION:', 5.0
        print>>f, 'DATE:', datetime.datetime.utcnow()
        print>>f, '   put below the number of detectors that were used to acquire spectra. IMPORTANT:'
        print>>f, '   this MUST come after VERSION, and before all other options!'
        print>>f, 'DETECTOR_ELEMENTS:', 1
        print>>f, '   give this file an internal name, whatever you like'
        print>>f, 'IDENTIFYING_NAME_[WHATEVERE_YOU_LIKE]:', 'automatic'
    
        print>>f, '   list the elements that you want to be fit. For K lines, just use the element'
        print>>f, '   name, for L lines add _L, e.g., Au_L, for M lines add _M'


        BRANCHING_FAMILY_ADJUSTMENT_L = ''
        BRANCHING_RATIO_ADJUSTMENT_L = ''
        f2 = open(os.path.join(main['master_dir'], 'maps_fit_parameters_override.txt'), 'rt')
        for line in f2:
            if ':' in line : 
                slist = line.split(':')
                tag = slist[0]
                value = ''.join(slist[1:])
                
                if tag == 'ELEMENTS_TO_FIT' :  
                    test_string = value.split(',')
                    
                    test_string = [x.strip() for x in test_string]   
                    print>>f, 'ELEMENTS_TO_FIT: '+', '.join(test_string)
         
                    
                elif tag == 'ELEMENTS_WITH_PILEUP' :  
                    pileup_string = value.split(',')
                    print>>f, '   list the element combinations you want to fit for pileup, e.g., Si_Si, Si_Si_Si, Si_Cl, etc'
                    pileup_string = [x.strip() for x in pileup_string]    
                    print>>f, 'ELEMENTS_WITH_PILEUP: '+', '.join(pileup_string)
        
                elif tag == 'BRANCHING_FAMILY_ADJUSTMENT_L' : 
                    temp_string = value.split(',')
                    temp_string = [x.strip() for x in temp_string]  
                    BRANCHING_FAMILY_ADJUSTMENT_L = ', '.join(temp_string)
                    
                elif tag == 'BRANCHING_RATIO_ADJUSTMENT_L':                    
                    temp_string = value.split(',')
                    temp_string = [x.strip() for x in temp_string] 
                    BRANCHING_RATIO_ADJUSTMENT_L = ', '.join(temp_string)
        f2.close()               

        
        print>>f, '   offset of energy calibration, in kev'
        print>>f, 'CAL_OFFSET_[E_OFFSET]:', fitp.s.val[fitp.keywords.energy_pos[0]]
        print>>f, 'CAL_OFFSET_[E_OFFSET]_MAX:', fitp.s.max[fitp.keywords.energy_pos[0]]
        print>>f, 'CAL_OFFSET_[E_OFFSET]_MIN:', fitp.s.min[fitp.keywords.energy_pos[0]]
    
        print>>f, '   slope of energy calibration, in leV / channel'
        print>>f, 'CAL_SLOPE_[E_LINEAR]:', fitp.s.val[fitp.keywords.energy_pos[1]]
        print>>f, 'CAL_SLOPE_[E_LINEAR]_MAX:', fitp.s.max[fitp.keywords.energy_pos[1]]
        print>>f, 'CAL_SLOPE_[E_LINEAR]_MIN:', fitp.s.min[fitp.keywords.energy_pos[1]]
    
        print>>f, '   quadratic correction for energy calibration, unless you know exactly what you are doing, please leave it at 0.'
        print>>f, 'CAL_QUAD_[E_QUADRATIC]:', fitp.s.val[fitp.keywords.energy_pos[2]]
        print>>f, 'CAL_QUAD_[E_QUADRATIC]_MAX:', fitp.s.max[fitp.keywords.energy_pos[2]]
        print>>f, 'CAL_QUAD_[E_QUADRATIC]_MIN:', fitp.s.min[fitp.keywords.energy_pos[2]]
    
        print>>f, '    energy_resolution at 0keV'
        print>>f, 'FWHM_OFFSET:', fitp.s.val[fitp.keywords.fwhm_pos[0]]
    
        print>>f, '    energy dependence of the energy resolution'
        print>>f, 'FWHM_FANOPRIME:', fitp.s.val[fitp.keywords.fwhm_pos[1]]
    
        print>>f, '    incident energy'
        print>>f, 'COHERENT_SCT_ENERGY:', fitp.s.val[fitp.keywords.coherent_pos[0]]
        print>>f, '    upper constraint for the incident energy'
        print>>f, 'COHERENT_SCT_ENERGY_MAX:', fitp.s.max[fitp.keywords.coherent_pos[0]]
        print>>f, '    lower constraint for the incident energy'
        print>>f, 'COHERENT_SCT_ENERGY_MIN:', fitp.s.min[fitp.keywords.coherent_pos[0]]
        print>>f, '    angle for the compton scatter (in degrees)'
        print>>f, 'COMPTON_ANGLE:', fitp.s.val[fitp.keywords.compton_pos[0]]
        print>>f, 'COMPTON_ANGLE_MAX:', fitp.s.max[fitp.keywords.compton_pos[0]]
        print>>f, 'COMPTON_ANGLE_MIN:', fitp.s.min[fitp.keywords.compton_pos[0]]
        print>>f, '    additional width of the compton'
        print>>f, 'COMPTON_FWHM_CORR:', fitp.s.val[fitp.keywords.compton_pos[1]]
        print>>f, 'COMPTON_STEP:', fitp.s.val[fitp.keywords.compton_pos[3]]
        print>>f, 'COMPTON_F_TAIL:', fitp.s.val[fitp.keywords.compton_pos[4]]
        print>>f, 'COMPTON_GAMMA:', fitp.s.val[fitp.keywords.compton_pos[5]]
        print>>f, 'COMPTON_HI_F_TAIL:', fitp.s.val[fitp.keywords.compton_pos[6]]
        print>>f, 'COMPTON_HI_GAMMA:', fitp.s.val[fitp.keywords.compton_pos[7]]
        
        print>>f, '    tailing parameters, see also Grieken, Markowicz, Handbook of X-ray spectrometry'
        print>>f, '    2nd ed, van Espen spectrum evaluation page 287.  _A corresponds to f_S, _B to'    
        print>>f, '    f_T and _C to gamma'    
        print>>f, 'STEP_OFFSET:', fitp.s.val[fitp.keywords.tail_pos[0]]
        print>>f, 'STEP_LINEAR:', fitp.s.val[fitp.keywords.tail_pos[1]]
        print>>f, 'STEP_QUADRATIC:', fitp.s.val[fitp.keywords.tail_pos[2]]
        print>>f, 'F_TAIL_OFFSET:', fitp.s.val[fitp.keywords.tail_pos[3]]
        print>>f, 'F_TAIL_LINEAR:', fitp.s.val[fitp.keywords.tail_pos[4]]
        print>>f, 'F_TAIL_QUADRATIC:', fitp.s.val[fitp.keywords.tail_pos[5]]
        print>>f, 'KB_F_TAIL_OFFSET:', fitp.s.val[fitp.keywords.tail_pos[9]]
        print>>f, 'KB_F_TAIL_LINEAR:', fitp.s.val[fitp.keywords.tail_pos[10]]
        print>>f, 'KB_F_TAIL_QUADRATIC:', fitp.s.val[fitp.keywords.tail_pos[11]]
        print>>f, 'GAMMA_OFFSET:', fitp.s.val[fitp.keywords.tail_pos[6]]
        print>>f, 'GAMMA_LINEAR:', fitp.s.val[fitp.keywords.tail_pos[7]]
        print>>f, 'GAMMA_QUADRATIC:', fitp.s.val[fitp.keywords.tail_pos[8]]
        print>>f, '    snip width is the width used for estimating background. 0.5 is typically a good start '    
        print>>f, 'SNIP_WIDTH:', fitp.s.val[fitp.keywords.added_params[0]]
        print>>f, '    set FIT_SNIP_WIDTH to 1 to fit the width of the snipping for background estimate, set to 0 not to. Only use if you know what it is doing!'
        if fitp.s.use[fitp.keywords.added_params[0]] == 2 : 
            print>>f, 'FIT_SNIP_WIDTH:', 1 
        else:
            print>>f, 'FIT_SNIP_WIDTH:', 0
        print>>f, '    detector material: 0= Germanium, 1 = Si'
        print>>f, 'DETECTOR_MATERIAL:', fitp.g.det_material
        print>>f, '    beryllium window thickness, in micrometers, typically 8 or 24'
        print>>f, 'BE_WINDOW_THICKNESS:',  fitp.g.be_window_thick/1000.
        print>>f, 'thickness of the detector chip, e.g., 350 microns for an SDD'
        print>>f, 'DET_CHIP_THICKNESS:',  fitp.g.det_chip_thick/1000.
        print>>f, 'thickness of the Germanium detector dead layer, in microns, for the purposes of the NBS calibration'
        print>>f, 'GE_DEAD_LAYER:', fitp.g.GE_dead_layer_thick/1000.
    
        print>>f, '    maximum energy value to fit up to [keV]'
        print>>f, 'MAX_ENERGY_TO_FIT:', fitp.g.xmax
        print>>f, '    minimum energy value [keV]'
        print>>f, 'MIN_ENERGY_TO_FIT:', fitp.g.xmin
    
        print>>f, '    this allows manual adjustment of the branhcing ratios between the different lines of L1, L2, and L3.'
        print>>f, '    note, the numbers that are put in should be RELATIVE modifications, i.e., a 1 will correspond to exactly the literature value,'
        print>>f, '    0.8 will correspond to to 80% of that, etc.'
    
        if BRANCHING_FAMILY_ADJUSTMENT_L != '': 
            print>>f, 'BRANCHING_FAMILY_ADJUSTMENT_L: '+BRANCHING_FAMILY_ADJUSTMENT_L
   
        print>>f, '    this allows manual adjustment of the branhcing ratios between the different L lines, such as La 1, la2, etc.'
        print>>f, '    Please note, these are all RELATIVE RELATIVE modifications, i.e., a 1 will correspond to exactly the literature value, etc.'
        print>>f, '    all will be normalized to the La1 line, and the values need to be in the following order:'
        print>>f, '    La1, La2, Lb1, Lb2, Lb3, Lb4, Lg1, Lg2, Lg3, Lg4, Ll, Ln'
        print>>f, '    please note, the first value (la1) MUST BE A 1. !!!'
    
        if BRANCHING_RATIO_ADJUSTMENT_L != '':
            print>>f, 'BRANCHING_RATIO_ADJUSTMENT_L: '+ BRANCHING_RATIO_ADJUSTMENT_L
 
        print>>f, '    this allows manual adjustment of the branhcing ratios between the different K lines, such as Ka1, Ka2, Kb1, Kb2'
        print>>f, '    Please note, these are all RELATIVE RELATIVE modifications, i.e., a 1 will correspond to exactly the literature value, etc.'
        print>>f, '    all will be normalized to the Ka1 line, and the values need to be in the following order:'
        print>>f, '    Ka1, Ka2, Kb1(+3), Kb2'
        print>>f, '    please note, the first value (Ka1) MUST BE A 1. !!!'


        f2 = open(os.path.join(main['master_dir'], 'maps_fit_parameters_override.txt'), 'rt')
        for line in f2:
            if ':' in line : 
                slist = line.split(':')
                tag = slist[0]
                value = ''.join(slist[1:])
                
                if tag == 'BRANCHING_RATIO_ADJUSTMENT_K' :  
                    test_string = value.split(',')
                    test_string = [x.strip() for x in test_string]    
                    print>>f, 'BRANCHING_RATIO_ADJUSTMENT_K: '+', '.join(test_string)

                elif tag == 'TAIL_FRACTION_ADJUST_SI' : 
                    temp = str(value)
                    print>>f, '    the two parameters below allow manual (multiplikative) adjustment of the Si tailing parameters. A value of one is nominal.  '
                    print>>f, '    only change if you now exactly what this does'
                    print>>f, 'TAIL_FRACTION_ADJUST_SI:'+temp
          
                elif tag == 'TAIL_WIDTH_ADJUST_SI' :
                    temp = str(value)
                    print>>f, 'TAIL_WIDTH_ADJUST_SI:'+temp
                                                            
        f2.close() 
    

    
        print>>f, '    the parameter adds the escape peaks (offset) to the fit if larger than 0. You should not enable Si and Ge at the same time, ie, one of these two values should be zero'
        print>>f, 'SI_ESCAPE_FACTOR:', fitp.s.val[fitp.keywords.added_params[1]]
        print>>f, 'GE_ESCAPE_FACTOR:', fitp.s.val[fitp.keywords.added_params[2]]
        print>>f, '    this parameter adds a component to the escape peak that depends linear on energy'
        print>>f, 'LINEAR_ESCAPE_FACTOR:', fitp.s.val[fitp.keywords.added_params[3]]
        print>>f, '    the parameter enables fitting of the escape peak strengths. set 1 to enable, set to 0 to disable. (in matrix fitting always disabled)'
    
        if fitp.s.use[fitp.keywords.added_params[1]] == 2 : 
            print>>f, 'SI_ESCAPE_ENABLE:', 1 
        else:
            print>>f, 'SI_ESCAPE_ENABLE:', 0
        if fitp.s.use[fitp.keywords.added_params[2]] == 2 : 
            print>>f, 'GE_ESCAPE_ENABLE:', 1 
        else:
            print>>f, 'GE_ESCAPE_ENABLE:', 0
        
        print>>f, '    the lines (if any) below will override the detector names built in to maps. please modify only if you are sure you understand the effect'
    
        if srcurrent != -1 : print>>f, 'SRCURRENT:', srcurrent
        if us_ic != -1 : print>>f, 'US_IC:', us_ic
        if ds_ic != -1 : print>>f, 'DS_IC:', ds_ic
        if dpc1_ic != -1 : print>>f, 'DPC1_IC:', dpc1_ic
        if dpc2_ic != -1 : print>>f, 'DPC2_IC:', dpc2_ic
        if cfg_1 != -1 : print>>f, 'CFG_1:', cfg_1
        if cfg_2 != -1 : print>>f, 'CFG_2:', cfg_2
        if cfg_3 != -1 : print>>f, 'CFG_3:', cfg_3
        if cfg_4 != -1 : print>>f, 'CFG_4:', cfg_4
        if cfg_5 != -1 : print>>f, 'CFG_5:', cfg_5
        if cfg_6 != -1 : print>>f, 'CFG_6:', cfg_6
        if cfg_7 != -1 : print>>f, 'CFG_7:', cfg_7
        if cfg_8 != -1 : print>>f, 'CFG_8:', cfg_8
        if cfg_9 != -1 : print>>f, 'CFG_9:', cfg_9
        if suffix != '' :  
            this_detector_number = int(suffix)+1
            temp = ELT1[:]
            temp.split('mca')
            position = len(temp[0])
            ELT1 = ELT1[0:position+1]+'mca'+str(this_detector_number)+ELT1[position+4: len(ELT1)-(position+4)+1]
            temp = ERT1[:]
            temp.split('mca')
            position = len(temp[0])
            ERT1 = ERT1[0: position]+'mca'+str(this_detector_number)+ERT1[position+4: len(ERT1)-(position+4)+1]
    
        if ELT1 != -1  : print>>f, 'ELT1:', ELT1
        if ERT1 != -1  : print>>f, 'ERT1:', ERT1
        if ICR1 != -1  : print>>f, 'ICR1:', ICR1
        if OCR1 != -1  : print>>f, 'OCR1:', OCR1
        if airpath != -1  : print>>f, 'AIRPATH:', airpath
    
        print>>f, '    the lines below (if any) give backup description of IC amplifier sensitivity, in case it cannot be found in the mda file'
        print>>f, '      for the amps, the _NUM value should be between 0 and 8 where 0=1, 1=2, 2=5, 3=10, 4=20, 5=50, 6=100, 7=200, 8=500'
        print>>f, '      for the amps, the _UNIT value should be between 0 and 3 where 0=pa/v, 1=na/v, 2=ua/v 3=ma/v'

        if US_AMP_SENS_NUM != -1  : print>>f, 'US_AMP_SENS_NUM:', US_AMP_SENS_NUM
        if US_AMP_SENS_UNIT != -1  : print>>f, 'US_AMP_SENS_UNIT:', US_AMP_SENS_UNIT
        if DS_AMP_SENS_NUM != -1  : print>>f, 'DS_AMP_SENS_NUM:', DS_AMP_SENS_NUM
        if DS_AMP_SENS_UNIT != -1  : print>>f, 'DS_AMP_SENS_UNIT:', DS_AMP_SENS_UNIT


        f.close()
        
        return
    
            
#-----------------------------------------------------------------------------
    def parse_pileupdef(self, fitp, pileup_string, info_elements): 

        keywords = fitp.keywords
        add_pars = fitp.add_pars
        
        for jj in range(9) :
            fitp.s.use[keywords.added_params[jj+4]] = 1.
        for jj in range(9):
            add_pars[max(keywords.mele_pos)-min(keywords.kele_pos)+1+jj, 0].energy = 0.
            
        
        if pileup_string:
            for jj in range(np.amin([8, len(pileup_string)])) : 
                pileup_elements = pileup_string[jj].split('_')
                for i in range(len(pileup_elements)): pileup_elements[i].strip()       
                n_pileup = len(pileup_elements)
                print  'pileup elements:', pileup_elements
                if n_pileup < 1 : 
                    message = 'WARNING: could not parse this element of the pileup string:' + pileup_string[jj]+' if you do not see anything wrong with this string or the override file, please contact the supplier of this software'
                    print message
                    continue 

                for ii in range(n_pileup): 
                    temp = pileup_elements[ii]
                    temp = temp.split(':')
                    element = temp[0]
                    
                    ind_pileup = -1
                    for iel in range(len(info_elements)):
                        if info_elements[iel].name == element:
                            ind_pileup = iel
                            #print 'have element', element, iel
            
                        this_energy = info_elements[ind_pileup].xrf['ka1']
                        if len(temp) >= 2 : 
                            # assume the colon is used to indicate another type of element,
                            # for now, K beta
                            if temp[1].lower() == 'kb' :
                                this_energy = info_elements[ind_pileup].xrf['kb1']
                                
                        if ii == 0 :
                            add_pars[max(keywords.mele_pos)-min(keywords.kele_pos)+1+jj, 0].energy = this_energy
                            fitp.s.use[keywords.added_params[jj+4]] = 5.
                            add_pars[max(keywords.mele_pos)-min(keywords.kele_pos)+1+jj, 0].ratio = 1.
                         
                        if ii > 0 :
                            add_pars[max(keywords.mele_pos)-min(keywords.kele_pos)+1+jj, 0].energy = add_pars[max(keywords.mele_pos)-min(keywords.kele_pos)+1+jj, 0].energy + this_energy                          
                  
                    if ind_pileup == -1 : 
                        message = 'WARNING: could not parse this element of the pileup string:' + pileup_string[jj] +' if you do not see anything wrong with this string or the override file, please contact the supplier of this software'
                        print message
                        continue
                    
                    
        return
            
        
