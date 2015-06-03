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
import time as time
import os

import numpy as np
import scipy 
from pylab import *
import matplotlib as mplot

import maps_fit_parameters

from fitting.leastsqbound import leastsqbound

SQRT2 = np.sqrt(2.)

#-----------------------------------------------------------------------------
def erf(x):
    # save the sign of x
    sign = 1
    if x < 0: 
        sign = -1
    x = abs(x)

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math.exp(-x*x)
    return sign*y # erf(-x) = -erf(x)

def erfc(x):
    return 1-erf(x)

#-----------------------------------------------------------------------------
class analyze:
    def __init__(self):   
        pass
    
#-----------------------------------------------------------------------------    
    def set_fit_vals(self, fitp, p, element_pos): 
        
        dims = fitp.add_pars.shape
        apd1 = dims[0]
        apd2 = dims[1]
        
        ap_energy = np.zeros((apd1,apd2))
        ap_mu_fraction = np.zeros((apd1,apd2))
        ap_width_multi = np.zeros((apd1,apd2))
        for i in range(apd1):
            for j in range(apd2):
                ap_energy[i,j] = fitp.add_pars[i,j].energy
                ap_mu_fraction[i,j] = fitp.add_pars[i,j].mu_fraction
                ap_width_multi[i,j] = fitp.add_pars[i,j].width_multi
                
        
        keywords = fitp.keywords
        sigma = np.sqrt( (p[keywords.fwhm_pos[0]]/2.3548)**2 + (ap_energy)*2.96*p[keywords.fwhm_pos[1]]  )
        f_step =  np.abs( ap_mu_fraction*( p[keywords.tail_pos[0]]+p[keywords.tail_pos[1]]*(ap_energy)))
        f_tail = np.abs( p[keywords.tail_pos[3]]+ p[keywords.tail_pos[4]]*ap_mu_fraction)
        kb_f_tail = np.abs( p[keywords.tail_pos[9]]+ p[keywords.tail_pos[10]]*ap_mu_fraction)
        gamma  = np.abs(p[keywords.tail_pos[6]]+p[keywords.tail_pos[7]]*(ap_energy))*ap_width_multi
            
        return sigma, f_step, f_tail, kb_f_tail, gamma
   

#-----------------------------------------------------------------------------      
    def residuals(self, p, y, x, weights):
        err = (y-self.model_spectrum(x,p))*weights
        return err
  
    
#-----------------------------------------------------------------------------    
# CONSTRUCT THE ENERGY AXIS
# p[keywords.energy_pos[0]] : energy of channel 0
# p[keywords.energy_pos[1]] : gain
# p[keywords.energy_pos[2]] : quadratic, cf. v. Grieken etal, pp 226    
# p[keywords.fwhm_pos[0]] : noise, 
# p[keywords.fwhm_pos[1]] : fano
    def model_spectrum(self, energy, p, allpars = False):
                
        if allpars == False:
            pall = self.parameters
            pall[np.where(self.fitp.s.use>1)] = p
        else:
            pall = p.copy()
                 
        keywords = self.fitp.keywords
        add_pars = self.fitp.add_pars
        element_pos = np.concatenate((keywords.kele_pos, keywords.lele_pos, keywords.mele_pos), axis=0)
        gain = pall[keywords.energy_pos[1]]
        ev = pall[keywords.energy_pos[0]] + energy * gain + (energy)**2 * pall[keywords.energy_pos[2]]
        
        sigma, f_step, f_tail, kb_f_tail, gamma = self.set_fit_vals(self.fitp, pall, element_pos)

        background = self.maps_snip(self.fitp, pall) 
#        import matplotlib.pyplot as plt 
#        plt.plot(background)
#        plt.show()
    
        if self.snip_background == 0 : background[:] = 0.
        counts_background = background[keywords.xmin:keywords.xmax+1].copy()
        counts_ka = counts_background.copy()
        counts_ka[:] = 0.
        counts_kb = counts_ka.copy()
        counts_kb[:] = 0.
        counts_l = counts_ka.copy()
        counts_l[:] = 0.
        counts_m = counts_ka.copy()
        counts_elastic = counts_ka.copy()
        counts_compton = counts_ka.copy()
        counts_step = counts_ka.copy()
        counts_tail = counts_ka.copy()
        counts_pileup = counts_ka.copy()
        counts_escape = counts_ka.copy()
        
        
        for i in range(len(element_pos)): 
            # skip calculation for peaks (element + scatter) that are fixed AND
            # close to zero (default 'zero' at 1e-10)

            if (keywords.use_this_par[element_pos[i]] != 1) and (pall[element_pos[i]] <= -9.999999) : continue 
    
            #print i, keywords.use_this_par[element_pos[i]] , pall[element_pos[i]]
    
            pre_faktor = 10.**pall[element_pos[i]]
            
            for j in range(12): 
                # gaussian peak shape          
                if (add_pars[i, j].ratio == 0.) : continue
                if (add_pars[i, j].energy <= 0.) : continue
                      
                
                delta_energy = ev.copy() - (add_pars[i, j].energy)

        
                faktor = float(add_pars[i, j].ratio* pre_faktor)
                if (add_pars[i, j].type == 2) : 
                    faktor = faktor /(1. +kb_f_tail[i, j]+f_step[i, j])
                if (add_pars[i, j].type == 1) or (add_pars[i, j].type == 3) :
                    faktor = faktor / (1. +f_tail[i, j]+f_step[i, j])
                # peak, gauss
                value = faktor * self.model_gauss_peak(gain, sigma[i, j], delta_energy)
                if (add_pars[i, j].type == 1) : counts_ka = counts_ka+value          
                if (add_pars[i, j].type == 2) : counts_kb = counts_kb+value          
                if (add_pars[i, j].type == 3) : counts_l = counts_l+value          
                if (add_pars[i, j].type == 7) : counts_m = counts_m+value          
                #  peak, step
                if f_step[i, j] > 0. : 
                    value = faktor * f_step[i, j]
                    value = value * self.model_gauss_step(gain, sigma[i, j], delta_energy, add_pars[i, j].energy)
                    counts_step = counts_step+value
                    
                 
                #  peak, tail;; use different tail for K beta vs K alpha lines
                if (add_pars[i, j].type == 2) :  
                    value = float(faktor * kb_f_tail[i, j])
                    value = value * self.model_gauss_tail(gain, sigma[i, j], delta_energy, gamma[i, j])
                    counts_tail = counts_tail+value                                   
                
#                if (add_pars[i, j].type == 1) or (add_pars[i, j].type == 3) :  
#                    value = float(faktor * f_tail[i, j])
#                    value = value * self.model_gauss_tail(gain, sigma[i, j], delta_energy, gamma[i, j])
#                    counts_tail = counts_tail + value                                   
  
        
        #scattering: elastic peak
        value[:] = 0.
        value, rayleigh_sigma = self.elastic_peak(self.fitp, value, ev, pall, gain)
        counts_elastic = counts_elastic + value
        
        # compton peak
        value[:] = 0.
        value, compton_sigma, faktor = self.compton_peak(self.fitp, value, ev, pall, gain)
        counts_compton = counts_compton + value
        
        
        # pileup
        temp_element_pos = np.array(keywords.added_params[4:13])
        for ii in range(9): 
            # skip calculation for peaks (element + scatter) that are fixed AND
            # close to zero (default 'zero' at 1e-10)
            if (pall[temp_element_pos[ii]] <= -10) : continue
            j = 0
            i = np.amax(keywords.mele_pos)-np.amin(keywords.kele_pos)+1+ii
            if (add_pars[i, j].energy <= 0.) : continue
            delta_energy = ev.copy() - (add_pars[i, j].energy)
            faktor = add_pars[i, j].ratio* (10.**pall[temp_element_pos[ii]])
            #print ii,  pall[temp_element_pos[ii]], faktor
            # peak, gauss
            value = faktor * self.model_gauss_peak(gain, sigma[i, j], delta_energy)
            counts_pileup = counts_pileup + value
   
        
        counts = counts_ka + counts_kb + counts_l + counts_m + counts_elastic + counts_compton + counts_step +\
             counts_tail + counts_pileup
             

        if np.sum(np.abs(pall[keywords.added_params[1:4]])) >= 0. : 
            # si escape
            if (pall[keywords.added_params[1]] > 0.) : 
                escape_E = 1.73998
                wo = np.where(ev > escape_E+ev[0])
                escape_factor = pall[keywords.added_params[1]] + pall[keywords.added_params[3]] * ev
                if len(wo[0]): 
                    for ii in range(len(wo[0])):
                        counts_escape[ii] = counts[wo[0][ii]]*np.amax(np.append(escape_factor[wo[0][ii]],0.0))
            # Ge escape
            counts = counts + counts_escape
     

        counts = counts + counts_background     
        
#        print 'counts_background', np.sum(counts_background)
#        print 'counts_ka', np.sum(counts_ka)
#        print 'counts_kb', np.sum(counts_kb)
#        print 'counts_l', np.sum(counts_l)
#        print 'counts_m', np.sum(counts_m)
#        print 'counts_elastic', np.sum(counts_elastic)
#        print 'counts_compton', np.sum(counts_compton)
#        print 'counts_step', np.sum(counts_step)
#        print 'counts_tail', np.sum(counts_tail)
#        print 'counts_pileup', np.sum(counts_pileup)
#        print 'counts_escape', np.sum(counts_escape)   

        self.counts_background = counts_background
        self.counts_ka = counts_ka
        self.counts_kb = counts_kb
        self.counts_l = counts_l
        self.counts_m = counts_m
        self.counts_elastic = counts_elastic
        self.counts_compton = counts_compton
        self.counts_step = counts_step
        self.counts_tail = counts_tail
        self.counts_pileup = counts_pileup
        self.counts_escape = counts_escape
        
        
#        import matplotlib.pyplot as plt 
#        plt.semilogy(counts+0.1)       
#        plt.semilogy(counts_background +0.1)
#        plt.semilogy(counts_ka+0.1)
#        plt.semilogy(counts_kb+0.1)
#        plt.semilogy(counts_l+0.1)
#        plt.semilogy(counts_m+0.1)
#        plt.semilogy(counts_elastic+0.1)
#        plt.semilogy(counts_compton+0.1)
#        plt.semilogy(counts_step+0.1)
#        plt.semilogy(counts_tail+0.1)
#        plt.semilogy(counts_pileup+0.1)
#        plt.semilogy(counts_escape+0.1)
#        plt.show()
    
        return counts
                
#-----------------------------------------------------------------------------      
    def residuals_matrix(self, p, y, x, weights):
        err = (y-self.matrixmodel_spectrum(x,p))*weights
        return err

#-----------------------------------------------------------------------------    

    def matrixmodel_spectrum(self, energy, pfit):
        
        keywords = self.fitp.keywords
        
        p = self.parameters.copy()
        p[self.iusepar] = pfit[:]
        
        counts = 0.
        if keywords.spectral_binning > 0:
            ind = energy/keywords.spectral_binning
            counts = keywords.background[ind.astype(int)]
        else:
            counts = keywords.background[energy]
       

        counts_background = counts.copy()
        counts[:] = 0.


        for i in range(len(self.iusepar)): 
            value = (10.**p[self.iusepar[i]])*self.fitmatrix[:, self.iusepar[i]]
            counts = counts + value

        if (np.sum(self.add_matrixfit_pars[3:6]) >= 0.):  
            ev = self.add_matrixfit_pars[keywords.energy_pos[0]] + energy * self.add_matrixfit_pars[keywords.energy_pos[1]] + (energy)**2 * self.add_matrixfit_pars[keywords.energy_pos[2]]
            counts_escape = counts.copy()
            counts_escape[:] = 0.
            if (self.add_matrixfit_pars[3] > 0.) : 
                escape_E = 1.73998
                wo = np.where(ev > escape_E+ev[0])

                escape_factor = np.abs(p[len(p)-3] + p[len(p)-1] * ev)
                if len(wo[0]) > 0: 
                    for ii in range(len(wo[0])):
                        counts_escape[ii] = counts[wo[ii]]*np.amax(np.append(escape_factor[wo[ii]],0.0))

                counts = counts + counts_escape
    
        counts = counts + counts_background

        return counts
    
#-----------------------------------------------------------------------------    
# models a gaussian fluorescence peak, see also van espen, spectrum evaluation,
# in van grieken, handbook of x-ray spectrometry, 2nd ed, page 182 ff
    def model_gauss_peak(self, gain, sigma, delta_energy): 
            
        counts = gain / ( sigma *np.sqrt(2.*np.math.pi)) * np.exp( -0.5* ((delta_energy / sigma)**2) )

        return counts
    
#-----------------------------------------------------------------------------  
# models a gaussian fluorescence peak, see also van espen, spectrum evaluation,
# in van grieken, handbook of x-ray spectrometry, 2nd ed, page 182 ff  
    def model_gauss_step(self, gain, sigma, delta_energy, peak_E): 

        import scipy.special
        counts = gain / 2. /  peak_E * scipy.special.erfc(delta_energy/(SQRT2*sigma))


        return counts
    
#-----------------------------------------------------------------------------    
# models a gaussian fluorescence peak, see also van espen, spectrum evaluation,
# in van grieken, handbook of x-ray spectrometry, 2nd ed, page 182 ff
    def model_gauss_tail(self, gain, sigma, delta_energy, gamma): 
        
        import scipy.special          
        
        delta_energy_neg = delta_energy.copy()
        #wo_neg = np.where(delta_energy_neg > 0.)
        wo_neg = (np.nonzero(delta_energy_neg > 0.))[0]
        if wo_neg.size > 0:
            delta_energy_neg[wo_neg] = 0.                  
        temp_a = np.exp(delta_energy_neg/ (gamma * sigma))
        counts = gain / 2. / gamma / sigma / np.exp(-0.5/(gamma**2)) *  \
            temp_a * scipy.special.erfc( delta_energy  /( SQRT2*sigma) + (1./(gamma*SQRT2) )  )
            
            
        return counts

#-----------------------------------------------------------------------------    
    def elastic_peak(self, fitp, counts, ev, p, gain, matrix = False): 
            
        keywords = fitp.keywords
        sigma = np.sqrt( (p[keywords.fwhm_pos[0]]/2.3548)**2  +  (p[keywords.coherent_pos[0]])*2.96*p[keywords.fwhm_pos[1]]  )
        delta_energy = ev - (p[keywords.coherent_pos[0]])

        # elastic peak, gaussian
        value = 1.
        if matrix == False :
            value = value * 10.**(p[keywords.coherent_pos[1]])
        value = value * self.model_gauss_peak(gain, sigma, delta_energy)
        counts = counts + value
        
        return counts, sigma
    
#-----------------------------------------------------------------------------    
    def compton_peak(self, fitp, counts, ev, p, gain, matrix = False): 
        
        keywords = fitp.keywords        
        compton_E = p[keywords.coherent_pos[0]]/(1. +(p[keywords.coherent_pos[0]]/511.)*(1. -np.cos( p[keywords.compton_pos[0]]*2.*np.math.pi/360. )))
    
        sigma = np.sqrt( (p[keywords.fwhm_pos[0]]/2.3548)**2 + compton_E*2.96*p[keywords.fwhm_pos[1]]  )
    
        local_sigma = sigma*p[14]
    
        delta_energy = ev.copy() - compton_E

        # compton peak, gaussian
        faktor = 1. / (1. +p[keywords.compton_pos[3]]+p[keywords.compton_pos[4]]+p[keywords.compton_pos[6]])
        if matrix == False : 
            faktor = faktor * (10.**p[keywords.compton_pos[2]])
        value = faktor * self.model_gauss_peak(gain, sigma*p[keywords.compton_pos[1]], delta_energy)
        counts = counts + value
            
        # compton peak, step
        if p[keywords.compton_pos[3]] > 0.: 
            value = faktor * p[keywords.compton_pos[3]] 
            value = value * self.model_gauss_step(gain, sigma, delta_energy, compton_E)
            counts = counts + value
    
        # compton peak, tail on the low side
        value = faktor * p[keywords.compton_pos[4]] 
        value = value * self.model_gauss_tail(gain, sigma, delta_energy, p[keywords.compton_pos[5]])       
        counts = counts + value
        
        # compton peak, tail on the high side
        value = faktor * p[keywords.compton_pos[6]] 
        value = value * self.model_gauss_tail(gain, sigma, -1.*delta_energy, p[keywords.compton_pos[7]])
        counts = counts + value
        
        return counts, sigma, faktor
        
                
#-----------------------------------------------------------------------------    
    def generate_fitmatrix(self, fitp, energy, p): 
        

        
        keywords = fitp.keywords
        add_pars = fitp.add_pars
        all = 1
        


        n_pileup = 9
        
        element_pos = np.concatenate((keywords.kele_pos, keywords.lele_pos, keywords.mele_pos), axis=0)
        element_pos_size = len(keywords.kele_pos) + len(keywords.lele_pos) + len(keywords.mele_pos)
        fitmatrix = np.zeros((energy.size, element_pos_size+2+n_pileup))    

        gain = p[keywords.energy_pos[1]]
        ev = p[keywords.energy_pos[0]] + energy * gain + (energy)**2 * p[keywords.energy_pos[2]]
        counts = np.zeros((energy.size))
                          
        sigma, f_step, f_tail, kb_f_tail, gamma = self.set_fit_vals(fitp, p, element_pos)
        
#        print'add_pars'
#        for i in range(element_pos.size): 
#            for j in range(12): 
#                #print i,j
#                print add_pars[i, j].ratio
#                #print add_pars[i, j].energy
            
        for i in range(element_pos.size): 
            if (keywords.use_this_par[element_pos[i]] != 1) and (p[element_pos[i]] <= -9.999999) : continue
            for j in range(12): 
                # gaussian peak shape          
                if (add_pars[i, j].ratio == 0.) : continue
                if (add_pars[i, j].energy <= 0.) : continue
                delta_energy = ev.copy() - (add_pars[i, j].energy)
                faktor = add_pars[i, j].ratio
                if (add_pars[i, j].type == 2) : 
                    faktor = faktor /(1. +kb_f_tail[i, j]+f_step[i, j])
                if (add_pars[i, j].type == 1) or (add_pars[i, j].type == 3) :
                    faktor = faktor / (1. +f_tail[i, j]+f_step[i, j])
               
                # peak, gauss
                value = faktor * self.model_gauss_peak(gain, sigma[i, j], delta_energy)
                if (add_pars[i, j].type == 1) : counts = counts+value          
                if (add_pars[i, j].type == 2) : counts = counts+value          
                if (add_pars[i, j].type == 3) : counts = counts+value          
                if (add_pars[i, j].type == 7) : counts = counts+value  
                        
                #  peak, step
                if f_step[i, j] > 0. :
                    value = float(faktor * f_step[i, j])
                    value = value * self.model_gauss_step(gain, sigma[i, j], delta_energy, add_pars[i, j].energy)
                    counts = counts+value
      
                #  peak, tail;; use different tail for K beta vs K alpha lines
                if (add_pars[i, j].type == 2) : 
                    value = float(faktor * kb_f_tail[i, j])
                    value = value * self.model_gauss_tail(gain, sigma[i, j], delta_energy, gamma[i, j])
                    counts = counts+value                                   
         
                if (add_pars[i, j].type == 1) or (add_pars[i, j].type == 3) :  
                    value = float(faktor * f_tail[i, j])
                    value = value * self.model_gauss_tail(gain, sigma[i, j], delta_energy, gamma[i, j])
                    counts = counts + value                                   
            fitmatrix[:, i] = fitmatrix[:, i]+counts[:]
            counts[:] = 0.

        
        i = len(element_pos)
        #scattering:    
        # elastic peak
        value = 0.
        value, rayleigh_sigma = self.elastic_peak(fitp, value, ev, p, gain, matrix = True)
        counts = counts + value
        fitmatrix[:, i] = fitmatrix[:, i]+counts[:]
        counts[:] = 0.
        
        #compton peak
        value = 0.
        value, compton_sigma, faktor = self.compton_peak(fitp, counts, ev, p, gain, matrix = True)
        counts = counts + value
        fitmatrix[:, i+1] = fitmatrix[:, i+1]+counts[:]
        counts[:] = 0.
    
        this_i = i+2
        # pileup
        for ii in range(9): 
            j = 0
            i = np.amax(keywords.mele_pos)-np.amin(keywords.kele_pos)+1+ii
            if (add_pars[i, j].energy <= 0.) : continue
            delta_energy = ev.copy() - (add_pars[i, j].energy)
            faktor = add_pars[i, j].ratio
            value = faktor * self.model_gauss_peak(gain, sigma[i, j], delta_energy)
            counts = counts + value
            fitmatrix[:, this_i+ii] = fitmatrix[:, this_i+ii]+counts[:]
            counts[:] = 0.
   
        return fitmatrix
    

    
#-----------------------------------------------------------------------------    
    def fit_spectrum(self, fitp, this_spectrum, used_chan, calib, 
                     spectral_binning = 0, first = False, maxiter = 0,
                     matrix = True):
        
        
        keywords = fitp.keywords
        add_pars = fitp.add_pars
        keywords.spectral_binning = spectral_binning
        

        self.snip_background = 1

    
        # PARAMETER INITIALISATION

        status = 999
        perror = 0
        niter = 0  
        verbose = 0
        error_status = 0
        
        
        gen_pars_at_bndry = 0
        ele_pars_at_bndry = 0
        
        x = np.arange(used_chan)
        if spectral_binning > 0 : x = x*spectral_binning
        
        y = this_spectrum.copy()
        keywords.spectrum = this_spectrum.copy()
        xmin = np.argmin(np.abs(x-(fitp.g.xmin-fitp.s.val[keywords.energy_pos[0]])/fitp.s.val[keywords.energy_pos[1]])) 
        xmax = np.argmin(np.abs(x-(fitp.g.xmax-fitp.s.val[keywords.energy_pos[0]])/fitp.s.val[keywords.energy_pos[1]])) 
        if (xmax > used_chan-1) or (xmax <= np.amin([xmin, used_chan/20.])) : xmax = used_chan-1    
        if (xmin < 0) or (xmin > xmax) : xmin = 0
    
        keywords.xmin = xmin
        keywords.xmax = xmax
                           
        background = self.maps_snip(fitp, fitp.s.val)
        if self.snip_background == 0: background[:] = 0.

        
        # SET WEIGHTS
        import scipy.signal
        sm_y = 1./(1.+y)
        s = scipy.signal.boxcar(5)
        A = s.sum()
        sm_y = scipy.signal.convolve(sm_y,s,mode='same')/A
        weights = np.abs(sm_y) 
        weights = weights/np.amax(weights)
    
        n_pars = fitp.g.n_fitp
        parinfo_value = np.zeros((n_pars))
        parinfo_fixed = np.zeros((n_pars), dtype = np.int) 
        parinfo_limited = np.zeros((n_pars, 2), dtype = np.int)
        parinfo_limits = np.zeros((n_pars, 2)) 
        parinfo_relstep = np.zeros((n_pars)) 
        parinfo_mpmaxstep = np.zeros((n_pars)) 
        parinfo_mpminstep = np.zeros((n_pars))
    
        
        for i in range(n_pars): 

            parinfo_value[i] = fitp.s.val[i]
            parinfo_mpmaxstep[i] = fitp.s.maxstep[i]
            parinfo_mpminstep[i] = fitp.s.minstep[i]

            if fitp.s.use[i] == 1 : parinfo_fixed[i] = 1 
            else : parinfo_fixed[i] = 0
            if fitp.s.use[i] == 2 : parinfo_limited[i,:] = [1, 1]
            if fitp.s.use[i] == 3 : parinfo_limited[i,:] = [1, 0]
            if fitp.s.use[i] == 4 : parinfo_limited[i,:] = [0, 1]
            if fitp.s.use[i] == 5 : 
                parinfo_limited[i,:] = [0, 0]          
                if (i == keywords.compton_pos[4]) or (i == keywords.compton_pos[5]) : parinfo_limited[i,:] = [1, 1]  
                if (i == keywords.energy_pos[0]) or (i == keywords.energy_pos[1]) : parinfo_limited[i,:] = [1, 1]  
       

            if (fitp.s.use[i] == 5) and (first > 0):                 
                if (i == keywords.coherent_pos[1]) or (i == keywords.compton_pos[2]) or (i >= keywords.kele_pos[0]) or ((i >= keywords.added_params[4]) and (i <= keywords.added_params[12])) : 
                    if (i == keywords.coherent_pos[1]) : 
                        this_min_e = fitp.s.val[keywords.coherent_pos[0]]-0.4
                        this_max_e = fitp.s.val[keywords.coherent_pos[0]]+0.4
                        this_factor = 8
                  
                    if (i == keywords.compton_pos[2]) : 
                        this_min_e = fitp.s.val[keywords.coherent_pos[0]]-0.8
                        this_max_e = fitp.s.val[keywords.coherent_pos[0]]-0.2
                        this_factor = 8

                    if (i >= keywords.kele_pos[0]) :  
                        ii = i-keywords.kele_pos[0]
                        this_min_e = add_pars[ii, 0].energy-.1
                        this_max_e = add_pars[ii, 0].energy+.1
                        this_factor = 8
                      
                    wo_temp = np.where(keywords.added_params[4:13] == i)
                    if len(wo_temp[0]):  
                        this_min_e  = add_pars[np.amax(keywords.mele_pos)-np.amin(keywords.kele_pos)+1+wo_temp[0][0], 0].energy -0.1
                        this_max_e = add_pars[np.amax(keywords.mele_pos)-np.amin(keywords.kele_pos)+1+wo_temp[0][0], 0].energy +0.1
                        this_factor = 8
                        
                    this_xmin = np.argmin(np.abs(x-(this_min_e-fitp.s.val[keywords.energy_pos[0]])/fitp.s.val[keywords.energy_pos[1]])) 
                    this_xmax = np.argmin(np.abs(x-(this_max_e-fitp.s.val[keywords.energy_pos[0]])/fitp.s.val[keywords.energy_pos[1]])) 

             
                    if (this_xmax > y.size-1) or (this_xmax <= min([this_xmin, used_chan/20.])) : this_xmax = y.size-1    
                    if (this_xmin < 0) or (this_xmin > this_xmax) : this_xmin = 0  
                    fitp.s.val[i] = np.amax([np.mean(y[this_xmin:this_xmax+1], axis=0)*this_factor+0.01, 1.])

            parinfo_value[i] = fitp.s.val[i]
            wo = np.nonzero(keywords.peaks == i)
            if len(wo[0]):
                if fitp.s.val[i] > 0:
                    parinfo_value[i] = np.log10(fitp.s.val[i])
                else:
                    parinfo_value[i] = 0.0
                
            #print i, fitp.s.val[i], parinfo_value[i]
            parinfo_limits[i,0] = fitp.s.min[i]
            parinfo_limits[i,1] = fitp.s.max[i]
            if parinfo_limited[i,0] == 1 : 
                if parinfo_value[i] < parinfo_limits[i,0] :
                    print 'The start value for the parameter ', fitp.s.name[i], ' seems to be set at ', str(parinfo_value[i]), \
                            'which is below the lower limit of ', str(parinfo_limits[i,0]), \
                            'please correct in the file maps_fit_parameters_override.txt', 'for now, I am aborting this action'
                    error_status = 1
                    return None, None, None, None, None, None
       
            if parinfo_limited[i,1] == 1 : 
                if parinfo_value[i] > parinfo_limits[i,1] : 
                    print 'The start value for the parameter ', fitp.s.name[i], ' seems to be set at ', str(parinfo_value[i]), \
                            'which is higher than the top limit of ', str(parinfo_limits[i,1]), \
                            'please correct in the file maps_fit_parameters_override.txt', 'for now, I am aborting this action'
                    error_status = 1
                    return None, None, None, None, None, None
            
       
        no_iters_vec = [0, 5, 10, 20, 50, 100, 500, 2000, 10000, 50000]
        if maxiter == 0: 
            maxiter = no_iters_vec[fitp.g.no_iters]

    
        x = x[xmin:xmax+1]
        y = y[xmin:xmax+1]
        weights = np.abs(weights[xmin:xmax+1])
    
        perror = weights * 0.
        parameters = parinfo_value[:]
        nfree = 0
        keywords.use_this_par[:] = 0
        
        #remove inf from parameters
        indinf = np.where(np.isfinite(parameters)==False)
        parameters[indinf]=0
        
        self.parameters = parameters
        self.fipsuse_nonzero = (np.nonzero(fitp.s.use>1))[0]
        fit_parameters = parameters[self.fipsuse_nonzero]
        fit_weights = weights[self.fipsuse_nonzero]
        
                             
        self.fitp = fitp
        
                              
        if matrix: 

            u = parinfo_value[:]
            offset = 15
            parinfo_prime_val = parinfo_value[np.amin(keywords.kele_pos):np.amax(keywords.mele_pos)+1]
            parinfo_prime_val = np.concatenate((parinfo_prime_val, [parinfo_value[keywords.coherent_pos[1]], parinfo_value[keywords.compton_pos[2]]], 
                                                parinfo_value[keywords.added_params[4:13]], parinfo_value[keywords.added_params[1:4]]), axis=0)
            parinfo_prime_fixed = parinfo_fixed[np.amin(keywords.kele_pos):np.amax(keywords.mele_pos)+1]
            parinfo_prime_fixed = np.concatenate((parinfo_prime_fixed, [parinfo_fixed[keywords.coherent_pos[1]], parinfo_fixed[keywords.compton_pos[2]]], 
                                                  parinfo_fixed[keywords.added_params[4:13]], parinfo_fixed[keywords.added_params[1:4]]), axis=0)

            parinfo_prime_limited = parinfo_limited[np.amin(keywords.kele_pos):np.amax(keywords.mele_pos)+1,:]
            parinfo_prime_limited = np.concatenate((parinfo_prime_limited, [parinfo_limited[keywords.coherent_pos[1],:], parinfo_limited[keywords.compton_pos[2],:]], 
                                                  parinfo_limited[keywords.added_params[4:13],:], parinfo_limited[keywords.added_params[1:4],:]), axis=0)
            
            parinfo_prime_limits = parinfo_limits[np.amin(keywords.kele_pos):np.amax(keywords.mele_pos)+1,:]
            parinfo_prime_limits = np.concatenate((parinfo_prime_limits, [parinfo_limits[keywords.coherent_pos[1],:], parinfo_limits[keywords.compton_pos[2],:]], 
                                                  parinfo_limits[keywords.added_params[4:13],:], parinfo_limits[keywords.added_params[1:4],:]), axis=0) 
            
                       

            keywords.use_this_par[np.where(parinfo_prime_fixed != 1)] = 1
            # force the last three to be 0, to make sure they do NOT get fitted as peaks.
            keywords.use_this_par[parinfo_prime_val.size-3:parinfo_prime_val.size] = 0
            parameters = parinfo_prime_val.copy()
            
            self.parameters = parameters
            self.iusepar = (np.nonzero(keywords.use_this_par != 0))[0]
            fit_parameters = parameters[self.iusepar]
            

            self.add_matrixfit_pars = np.zeros((6))
            self.add_matrixfit_pars[0:3] = fitp.s.val[keywords.energy_pos[0:3]]
            self.add_matrixfit_pars[3:6] = fitp.s.val[keywords.added_params[1:4]]
       
            self.fitmatrix = self.generate_fitmatrix(fitp, x, parinfo_value)
            
            
            p0 = np.array(fit_parameters)
            
#            fit = self.matrixmodel_spectrum(x, p0)            
#            import matplotlib.pyplot as plt 
#            plt.plot(x,fit)
#            plt.plot(x,y)
#            #plt.semilogy(x,y+0.1)
#            #plt.show()
#            #plt.semilogy(x,fit+0.1)
#            plt.show()
            
            bounds=[]
            have_bounds = 0
            for i in range(n_pars):
                if keywords.use_this_par[i] != 0:
                    havelimit = parinfo_prime_limited[i,:]
                    if havelimit[0] == 1:
                        bmin = parinfo_prime_limits[i,0]
                        have_bounds = 1
                    else:
                        bmin = None
                    if havelimit[1] == 1:
                        bmax = parinfo_prime_limits[i,1]
                        have_bounds = 1
                    else:
                        bmax = None
                    bounds.append((bmin,bmax))

        
            if np.sum(y) > 0. : 
                
                if verbose : 
                    time1 = time.time()
                    print 'start fit, maxiter=',maxiter
                if have_bounds:
                    print 'with bounds'
                    p1,cov,infodict,mesg,self.success = leastsqbound(self.residuals_matrix, p0, bounds, args=(y, x, weights), maxfev=maxiter, full_output = True)
                else:
                    import scipy.optimize
                    p1,cov,infodict,mesg,self.success = scipy.optimize.leastsq(self.residuals_matrix, p0, args=(y, x, weights),maxfev=maxiter, full_output = True)
                
                if self.success not in [1,2,3,4]:
                    if verbose : print ' Unable to fit spectrum'

                
                perror1 = self.calc_perror(infodict['fjac'], infodict['ipvt'], len(p1))
                
                if verbose : 
                    time2 = time.time()
                    print 'fit done in ', time2-time1
                    print 'success', self.success, mesg
                    print 'number of function evals', infodict['nfev']
                    
                    
                
                parameters[self.iusepar] = p1
                
                perror = np.zeros((len(parameters)))
                perror[self.iusepar] = perror1
            

                fitted_spec = self.matrixmodel_spectrum(x, p1)

#                import matplotlib.pyplot as plt 
#                plt.plot(x,fitted_spec)
#                plt.plot(x,y)
#                #plt.semilogy(x,y+0.1)
#                #plt.show()
#                #plt.semilogy(x,fit+0.1)
#                plt.show()
  
                 
                # store the uncertainty estimates
                delu = parinfo_value.copy() * 0.
                delu[np.min(keywords.kele_pos):np.max(keywords.mele_pos)+1] = perror[0:len(parameters)-offset+1]
                delu[keywords.coherent_pos[1]] = perror[len(parameters)-offset+1]
                delu[keywords.compton_pos[2]] = perror[len(parameters)-offset+2]
                delu[keywords.added_params[4:13]] = perror[len(parameters)-offset+3:len(parameters)-offset+11+1]
                delu[keywords.added_params[1:4]] = perror[len(parameters)-offset+12:len(parameters)-offset+14+1]
                perror = delu
            else: 

                parameters = parinfo_prime_val.copy()
                parameters[0:len(parameters)-offset+1] =  -10.
                parameters[len(parameters)-offset+1] =  -10.
                parameters[len(parameters)-offset+2] =  -10.
                parameters[len(parameters)-offset+3:len(parameters)-offset+11+1] = -10.
                perror = parinfo_value.copy() * 0.
                
                fitted_spec = self.matrixmodel_spectrum(x, p0)
            

            u[min(keywords.kele_pos):max(keywords.mele_pos)+1] = parameters[0:parameters.size-offset+1]
            u[keywords.coherent_pos[1]] = parameters[parameters.size-offset+1]
            u[keywords.compton_pos[2]] = parameters[parameters.size-offset+2]
            u[keywords.added_params[4:13]] = parameters[parameters.size-offset+3:parameters.size-offset+12]
            u[keywords.added_params[1:4]] = parameters[parameters.size-offset+12:parameters.size-offset+15]

 
            dummy = self.model_spectrum(x, u, allpars = True)
            
#            import matplotlib.pyplot as plt 
#            plt.plot(x,dummy)
#            plt.plot(x,y)
#            #plt.semilogy(x,y+0.1)
#            #plt.show()
#            #plt.semilogy(x,fit+0.1)
#            plt.show()
       
        
        else:
            # matrix == False
            
            wusepar = (np.nonzero(parinfo_fixed != 1))[0]
            keywords.use_this_par[wusepar] = 1
            
            bounds=[]
            for i in range(n_pars):
                if fitp.s.use[i]>1:
                    havelimit = parinfo_limited[i,:]
                    if havelimit[0] == 1:
                        bmin = parinfo_limits[i,0]
                    else:
                        bmin = None
                    if havelimit[1] == 1:
                        bmax = parinfo_limits[i,1]
                    else:
                        bmax = None
                    bounds.append((bmin,bmax))
            
            p0 = np.array(fit_parameters)
            
#            fit = self.model_spectrum(x, p0)
#            import matplotlib.pyplot as plt 
#            plt.plot(x,fit)
#            #plt.plot(x,y)
#            #plt.semilogy(x,y+0.1)
#            #plt.show()
#            #plt.semilogy(x,fit+0.1)
#            plt.show()
            
            
            if verbose : 
                time1 = time.time()
                print 'start fit, maxiter=',maxiter
               
            
            import scipy.optimize
            p1,cov,infodict,mesg,self.success = scipy.optimize.leastsq(self.residuals, p0, args=(y, x, weights), maxfev=maxiter, full_output = True)

            if self.success not in [1,2,3,4]:
                if verbose : print ' Unable to fit spectrum'      
            
            #p1,cov,infodict,mesg, self.success = leastsqbound(self.residuals, p0, bounds, args=(y, x, weights),maxfev=maxiter, full_output=1)
            perror1 = self.calc_perror(infodict['fjac'], infodict['ipvt'], len(p1))
            
            
            if verbose : 
                time2 = time.time()
                print 'fit done in ', time2-time1

                      
            parameters[self.fipsuse_nonzero] = p1
            perror = np.zeros((len(parameters)))
            perror[self.fipsuse_nonzero] = perror1
                
            fitted_spec = self.model_spectrum(x, p1)
            u = parameters
            
            
          
            verbose = 0
            string = ''
            if len(wusepar)>0:
                if verbose : print 'Fit results:', self.success
                for kk in range(len(wusepar)):            
                    if verbose : print kk, '  ', fitp.s.name[kk],' parinfo_value: ', parameters[kk]
                    if parinfo_limited[wusepar[kk],0] : string = string+ ' lower limit: '+ str(parinfo_limits[wusepar[kk],0])
                    if parinfo_limited[wusepar[kk],1] : string = string+ ' upper limit: '+ str(parinfo_limits[wusepar[kk],1])
                    if verbose : print string
                    if parinfo_limited[wusepar[kk],0] == parameters[wusepar[kk]] : 
                        if kk < np.min(keywords.kele_pos) : gen_pars_at_bndry = gen_pars_at_bndry+1
                        if kk >= np.min(keywords.kele_pos) : ele_pars_at_bndry = ele_pars_at_bndry+1

        
#             import matplotlib.pyplot as plt 
#             plt.plot(x,y) 
#             plt.plot(x,fitted_spec)   
#             plt.show()    
            

        self.abs_err = np.sum(np.abs(y-fitted_spec))
    
        if np.sum(y) > 0. : self.rel_err = self.abs_err / np.sum(y) 
        else: self.rel_err = 0.

        chisq = np.sum((y-fitted_spec)**2. * weights)
        chisquare = chisq

        wo_free_pars = np.where(parinfo_fixed != 1) 
        if len(wo_free_pars[0]):
            free_pars = len(wo_free_pars[0])
        else:
            free_pars = 0  
        chisqred = chisq / (len(y) - free_pars)
        
        if np.sum(y) > 0. : 
            self.nfev = infodict['nfev']
        else:
            self.nfev = 0

        if verbose >= 2 :
            print ' ERRMSG:', mesg
            print ' status: ', status
            if np.sum(y) > 0. : 
                print ' niter:', infodict['nfev']
                print 'nfree (parameters) = ', nfree
                print 'ndata = ', len(y)
                print ' chisq:', chisq
                print ' chisqred:', chisqred
                print ' abs_err:', self.abs_err
                print ' rel_err:', self.rel_err
                print ' time + core ana:', time()-time1
                print 'snip width:', u[keywords.added_params[0]]
                
           

            name_len = []
            for i in range(len(fitp.s.name)): 
                name_len.append(len(fitp.s.name[i].strip()))
            max_name_length = np.max(np.asarray(name_len))+1
            print '{0:3}, {1:{2}}, {3:15}, {4:15}, {5:15}, {6:15}, {7:3},{8:3}, {9:15}'.format(
                                'num', 'name', max_name_length, 'fit_value', 'fit_sigma', \
                                'low_limit', 'high_limit', 'llt', 'hlt', '.relstep')

            for i in range(n_pars) :
                if parinfo_fixed[i] != 1 :
                    print '{0:3}, {1:{2}}, {3:15}, {4:15}, {5:15}, {6:15}, {7:3},{8:3}, {9:15}'.format(
                        i, fitp.s.name[i], max_name_length, str(u[i]), str(perror[i]), 
                        str(parinfo_limits[i,0]), str(parinfo_limits[i,1]), 
                        str(parinfo_limited[i,0]), str(parinfo_limited[i,1]), str(parinfo_relstep[i]))
        elif verbose == 1 :
                    pos = np.where(parinfo_fixed == 0)
                    if len(pos[0]) >0:
                        print fitp.s.name[pos]
                        print u[pos]
                        print perror[pos]

        
        return u, fitted_spec, background, xmin, xmax, perror

#-----------------------------------------------------------------------------    
    def fit_line(self,data_line, output_dir, n_rows,  matrix, spectral_binning, elt_line,
                 values_line, bkgnd_line, tfy_line,
                 info_elements, fitp, old_fitp, add_pars, keywords, add_matrixfit_pars, xrf_bin, calib):
        

        n_channels = len(data_line[:, 0])
        fitted_line = np.zeros((n_channels, n_rows))
        ka_line = np.zeros((n_channels, n_rows))
        l_line = np.zeros((n_channels, n_rows))
        bkground_line = np.zeros((n_channels, n_rows))
          
          
        old_fitp.s.val[:]=fitp.s.val[:]
                        
        for j_fit in range(n_rows):
            
        
            if (xrf_bin > 0) : 
        
                if (xrf_bin == 2) and (n_rows > 5) : 
                    if j_fit % 2 != 0: continue  
     
                if (xrf_bin == 4) and (n_rows > 5) : 
                    if j_fit % 3 != 0 : continue 
 
             
        
            data_temp = data_line[:, j_fit]
        
            niter = 0
                         
            fitp.s.val[:]=old_fitp.s.val[:]
            
                                            
            u, fitted_spec, background, xmin, xmax, perror = self.fit_spectrum(fitp, data_temp, data_temp.size, calib, spectral_binning = spectral_binning,
                                                                                  first = True, matrix = True, maxiter = 1000)     


                

            sys.stdout.flush()
          
            #return the number of function evaluations:
            niter = self.nfev
            if niter == 1 : 
                text = 'warning: number of iterations resulting from call of maps_fit_spectrum is 1. Must abort this action. '
                print text
                return

        
            fitp_values = u[:]
            fitp_values[keywords.peaks] = 10.**u[keywords.peaks]
            values_line[j_fit, :] = fitp_values[:]/elt_line[j_fit]
            bkgnd_line[j_fit] = np.sum(background[keywords.xmin:keywords.xmax+1])/elt_line[j_fit]
            tfy_line[j_fit] = np.sum(data_temp[:])/elt_line[j_fit]
        
            fitted_line[xmin:xmax+1, j_fit] = fitted_spec[:]
            ka_line[xmin:xmax+1, j_fit] = self.counts_ka[:]
            l_line[xmin:xmax+1, j_fit] = self.counts_l[:]
            bkground_line[xmin:xmax+1, j_fit] = background[keywords.xmin:keywords.xmax+1]
              
        
        return fitted_line, ka_line, l_line, bkground_line,  values_line, bkgnd_line, tfy_line, xmin, xmax





#-----------------------------------------------------------------------------    
    def get_counts(self):
        
        return self.counts_background, self.counts_ka, self.counts_kb, self.counts_l, self.counts_m, \
            self.counts_elastic, self.counts_compton, self.counts_step, self.counts_tail, \
            self.counts_pileup, self.counts_escape
        
        
#-----------------------------------------------------------------------------    
    def calc_perror(self, fjac, ipvt, npars, tol=1.e-14):
        
        ipvt = ipvt - 1
        
        verbose = 0
        n=npars

        covar = None
        perror = None
        # set the covariance matrix 
        if (n is not None) and (fjac is not None) and (ipvt is not None):
            sz = fjac.shape
            if (n > 0) and (sz[0] >= n) and (sz[1] >= n) \
                and (len(ipvt) >= n):
                
                
                if verbose: print 'computing the covariance matrix'

                cv = self.calc_covar(fjac[0:n,0:n], ipvt[0:n])
                cv.shape = [n, n]
                #nn = len(xall)
                nn = npars
                

                # Fill in actual covariance matrix, accounting for fixed
                # parameters.
                covar = np.zeros([nn, nn], dtype=float)
                for i in range(n):
                    #covar[ifree,ifree[i]] = cv[:,i]
                    covar[:,i] = cv[:,i]

                # Compute errors in parameters
                if verbose: print  'computing parameter errors'
                perror = np.zeros(nn, dtype=float)
                d = np.diagonal(covar)
                wh = (np.nonzero(d >= 0))[0]
                if len(wh) > 0:
                    perror[wh] = np.sqrt(d[wh])
                    
        return perror
            
#-----------------------------------------------------------------------------    
    def calc_covar(self, rr, ipvt, tol=1.e-14):
        
        verbose = 0
        if verbose:
            print 'Entering calc_covar...'
        if np.rank(rr) != 2:
            print 'ERROR: r must be a two-dimensional matrix'
            return -1
        s = rr.shape
        n = s[0]
        if s[0] != s[1]:
            print 'ERROR: r must be a square matrix'
            return -1

        if ipvt is None:
            print 'ipvrt none'
            ipvt = np.arange(n)

        r = rr.copy()
        r.shape = [n,n]

        # For the inverse of r in the full upper triangle of r
        l = -1
        tolr = tol * np.abs(r[0,0])
        for k in range(n):
            if np.abs(r[k,k]) <= tolr:
                break
            r[k,k] = 1./r[k,k]
            for j in range(k):
                temp = r[k,k] * r[j,k]
                r[j,k] = 0.
                r[0:j+1,k] = r[0:j+1,k] - temp*r[0:j+1,j]
            l = k

        # Form the full upper triangle of the inverse of (r transpose)*r
        # in the full upper triangle of r
        if l >= 0:
            for k in range(l+1):
                for j in range(k):
                    temp = r[j,k]
                    r[0:j+1,j] = r[0:j+1,j] + temp*r[0:j+1,k]
                temp = r[k,k]
                r[0:k+1,k] = temp * r[0:k+1,k]

        # For the full lower triangle of the covariance matrix
        # in the strict lower triangle or and in wa
        wa = np.repeat([r[0,0]], n)
        for j in range(n):
            jj = ipvt[j]
            sing = j > l
            for i in range(j+1):
                if sing:
                    r[i,j] = 0.
                ii = ipvt[i]
                if ii > jj:
                    r[ii,jj] = r[i,j]
                if ii < jj:
                    r[jj,ii] = r[i,j]
            wa[jj] = r[j,j]

        # Symmetrize the covariance matrix in r
        for j in range(n):
            r[0:j+1,j] = r[j,0:j+1]
            r[j,j] = wa[j]

        return r

        
#-----------------------------------------------------------------------------    
    def maps_snip(self, fitp, par_values):
        
        v = 1
        
        keywords = fitp.keywords
        
        background = keywords.spectrum.copy()
        n_background = background.size
                     
        # calculate the energy axis from parameter values
        
        e_off = par_values[keywords.energy_pos[0]]
        e_lin = par_values[keywords.energy_pos[1]]
        e_quad = par_values[keywords.energy_pos[2]]
        
   
        energy = np.arange(np.float(n_background))
        if keywords.spectral_binning > 0:
            energy = energy * keywords.spectral_binning
    
        energy = e_off + energy * e_lin + np.power(energy,2) * e_quad
        
        
        tmp = (e_off/2.3548)**2 + energy*2.96*e_lin 
        wind = np.nonzero(tmp < 0)[0]
        tmp[wind] = 0.
        fwhm = 2.35 * np.sqrt(tmp)
        
        
        original_bcgrd = background.copy()
        
        #import matplotlib.pyplot as plt 
        #plt.plot(energy, background)
        
        #smooth the background
        import scipy.signal
        if keywords.spectral_binning > 0 :
            s = scipy.signal.boxcar(3)
        else :
            s = scipy.signal.boxcar(5)
        A = s.sum()
        background = scipy.signal.convolve(background,s,mode='same')/A
        
        #Check smoothing
        #plt.plot(energy, background)
        #plt.show()
        
        # SNIP PARAMETERS
        window_rf = SQRT2
        
        width = par_values[keywords.added_params[0]] 
    
        window_p = width * fwhm / e_lin # in channels
        if keywords.spectral_binning > 0:
            window_p = window_p/2.
        
        background = np.log(np.log(background+1.)+1.)
    
        index = np.arange(np.float(n_background))
        
        #FIRST SNIPPING
    
        if keywords.spectral_binning > 0:
            no_iterations = 3 
        else:
            no_iterations = 2   
            
        for j in range(no_iterations):
            lo_index = index - window_p
            wo = np.where(lo_index < max((keywords.xmin, 0)))
            lo_index[wo] = max((keywords.xmin, 0))
            hi_index = index + window_p
            wo = np.where(hi_index > min((keywords.xmax, n_background-1)))
            hi_index[wo] = min((keywords.xmax, n_background-1))
            
            temp = (background[lo_index.astype(np.int)] + background[hi_index.astype(np.int)]) / 2.
            wo = np.where(background > temp)
            background[wo] = temp[wo] 
            
        #import matplotlib.pyplot as plt
        #plt.plot(energy, np.exp(np.exp(background)-1.)-1.)
        #plt.show()
        
        if keywords.spectral_binning > 0:
            no_iterations = 7 
        else:
            no_iterations = 12
            
        current_width = window_p
        max_current_width = np.amax(current_width)
        
        while max_current_width >= 0.5:
            #print 'max_current_width = ', max_current_width
            lo_index = index - current_width
            wo = np.where(lo_index < max((keywords.xmin, 0)))
            lo_index[wo] = max((keywords.xmin, 0))
            hi_index = index + current_width
            wo = np.where(hi_index > min((keywords.xmax, n_background-1)))
            hi_index[wo] = min((keywords.xmax, n_background-1))       

            temp = (background[lo_index.astype(np.int)] + background[hi_index.astype(np.int)]) / 2.
            wo = np.where(background > temp)
            background[wo] = temp[wo] 
            
            current_width = current_width / window_rf
            max_current_width = np.amax(current_width)
       
        #import matplotlib.pyplot as plt
        #plt.plot(energy, np.exp(np.exp(background)-1.)-1.)
        #plt.show()
       
        background = np.exp(np.exp(background)-1.)-1.
        
        wo = np.where(np.isfinite(background) == False)
        background[wo] = 0.
    
        keywords.background = background.copy()
   
        
        return background
    
    
    
#-----------------------------------------------------------------------------
#  Applies 2-D Fourier integration to reconstruct dpc images,
#  presents fluorescence maps for select elements if present in data directory
# VARIABLE DECLARATIONS / MEANINGS:    
# info                structure of info about illumination etc
# delta, beta         optical parameters
# nrml                right-minus-left normalised to transmission signal
# ntmb                top-minus-bottom normalised to transmission signal
    
# gxdt                 x component of the gradient of the delta.thickness
# gydt                 y component of the gradient of the delta.thickness
    
# ngxdt                x component of the gradient of the delta.thickness, normalised
#                         so as to have zero mean
# ngydt                y component of the gradient of the delta.thickness, normalised
#                         so as to have zero mean
#=============================================
    def maps_simple_dpc_integration(self, nrml, ntmb, no_int = True): 
        
        hc = 0.001239842        # wavelength-energy relationship, microns / keV
    

        if nrml.ndim < 1 : 
            nrml = 0
            ntmb = 0
            rdt = 0
            return 
        
        sz = nrml.shape
        nx = sz[0]
        ny = sz[1]
        
        
        # "what goes up must come down" 
        #     - can be used to remove beam intensity variations AND
        #     - removes first order effect of detector misalignment 
        #            (i.e., removes 'constant gradient')
    
        ylo = 0 
        yhi = ny-1
            
        #find the vertical lines with the smalles
        #spread, which hopefully are the background
        
        for i in range(ylo, yhi): 
            nrml[:, i] = nrml[:, i] - nrml[:, i].mean(axis=0)
            # added this for the other direction, too.
            ntmb[:, i] = ntmb[:, i] - ntmb[:, i].mean(axis=0)
          
        # remove first order effect of detector misalignment in vertical
        #            (i.e., remove 'constant gradient')
        ntmb = ntmb - ntmb.mean(axis=0)
        # added this for the other direction, too.
        nrml = nrml - nrml.mean(axis=0)
        
        rdt = 0
        
        if no_int == False:
            
            cs_d        = 40.0
            zp_d        = 160.0
            zp_dr        = 50.0 / 1000.
            zp_f        = 18.26 * 1000.
            energy    = 10.1
    
            zz        = 82.17
    
            hx         = 0.1
            hy         = 0.1
    
            xlin = np.arange(float(nx)) * hx
            ylin = np.arange(float(ny)) * hy
            ylin = ylin[::-1]
            xax = xlin # (ylin*0+1)
            yax = (xlin*0+1) # ylin
            
            #=============================================
            # calculate as gradient of t
            # gxdt, gydt refers to the gradient of the delta.thickness
    
            # extra factor of 2 comes from use of diameters, not radii...
            ngxdt = (np.math.pi * (zp_d + cs_d)) / ( 8. * zp_f) * nrml
            ngydt = (np.math.pi * (zp_d + cs_d)) / ( 8. * zp_f) * ntmb
            
    
            #=============================================
            # implement FFT reconstruction
            

            dpc = ngxdt + 1j * ngydt

    
            fx = (np.arange(float(nx)) - nx/2) / ((nx-1) * hx)
            fy = (np.arange(float(ny)) - ny/2) / ((ny-1) * hy)
            fy = fy[::-1]
            fxt = np.outer(fx,(fy*0.+1.))
            fy = np.outer((fx*0.+1.), fy )
            fx = fxt
            fxt = 0
            
            xy = 2j * np.math.pi * (fx +1j* fy)
            xy[(nx-1)/2, (ny-1)/2] = 1    # to avoid 0/0 error
            xy = np.fft.fftshift(xy)
            
            Fdpc = np.fft.fft2(dpc)
            
            Fdpc[0,0] = 0 # dc level information not available, leads to numerical error
            
            dt = np.fft.ifft2(Fdpc/xy)
            
            # is dt the magnitude of the complex value or the real part?
            # note that the real part dominates, and the magnitude
            # loses dynamic range due to abs(-1) = 1, i.e. real part is positive only
   
            idt = dt.imag
            rdt = dt.real
            
            
            temp = np.concatenate((rdt[0, 1:ny-1].flatten(), rdt[nx-1, 1:ny-1].flatten(), \
                              rdt[0:nx, 0].flatten(), rdt[0:nx, ny-1].flatten()))
            
            
            # set the average of the perimetric values to be zero
            rdt = rdt - np.mean(temp)
        
        return nrml, ntmb, rdt
#-----------------------------------------------------------------------------
