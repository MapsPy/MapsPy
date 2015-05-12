'''
Created on Nov 21, 2011

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

import csv
        
#-----------------------------------------------------------------------------
class element_info:
    def __init__(self):
        self.z = 0    
        self.name = '' 
        self.xrf = {'Ka1':0., 'Ka2':0., 
                    'Kb1':0., 'Kb2':0., 
                    'La1':0., 'La2':0., 'Lb1':0., 'Lb2':0., 'Lb3':0., 'Lb4':0., 'Lb5':0., 
                    'Lg1':0., 'Lg2':0., 'Lg3':0., 'Lg4':0., 'Ll':0., 'Ln':0., 
                    'Ma1':0., 'Ma2':0., 'Mb':0., 'Mg':0. }
        self.xrf_abs_yield = {'Ka1':0., 'Ka2':0., 
                              'Kb1':0., 'Kb2':0., 
                              'La1':0., 'La2':0., 'Lb1':0., 'Lb2':0., 'Lb3':0., 'Lb4':0., 'Lb5':0., 
                              'Lg1':0., 'Lg2':0., 'Lg3':0., 'Lg4':0., 'Ll':0., 'Ln':0., 
                              'Ma1':0., 'Ma2':0., 'Mb':0., 'Mg':0. }     
        self.yieldD = {'k':0., 'l1':0., 'l2':0., 'l3':0., 'm':0. }
        self.density = 1.
        self.mass = 1.
        self.bindingE = {
                         'K':0., 
                         'L1':0., 'L2':0., 'L3':0., 
                         'M1':0., 'M2':0., 'M3':0., 'M4':0., 'M5':0., 
                         'N1':0., 'N2':0., 'N3':0., 'N4':0., 'N5':0., 'N6':0., 'N7':0., 
                         'O1':0., 'O2':0., 'O3':0., 'O4':0., 'O5':0., 
                         'P1':0., 'P2':0., 'P3':0. }
        self.jump = {
                     'K':0., 
                     'L1':0., 'L2':0., 'L3':0., 
                     'M1':0., 'M2':0., 'M3':0., 'M4':0., 'M5':0., 
                     'N1':0., 'N2':0., 'N3':0., 'N4':0., 'N5':0., 
                     'O1':0., 'O2':0., 'O3':0. }
              
              
              
#-----------------------------------------------------------------------------
class maps_elements:
    def __init__(self):   
        
        pass
  
#-----------------------------------------------------------------------------  
    def get_element_info(self): 
           
        #Number of elements in table
        nels = 100  
        
        els_file = 'reference/xrf_library.csv'
        
        try:
            f = open(els_file, 'r')
            csvf = csv.reader(f, delimiter=',')
        except:
            try:
                els_file = '../reference/xrf_library.csv'
                f = open(els_file, 'r')
                csvf = csv.reader(f, delimiter=',')
            except:
                print 'Error: Could not find xrf_library.csv file! Please get the library file (e.g., from runtime maps at'
                print 'http://www.stefan.vogt.net/downloads.html) and make sure it is in the Python path'
                return None
        
        version = 0.
        for row in csvf:
            if row[0] == 'version:': 
                version = float(row[1])
                break
        if version != 1.2:
            print 'Warning: the only xrf_library.csv file that was found is out of date.  Please use the latest file.'
            print 'A copy can be downloaded, e.g., as part of the runtime maps release available at '
            print 'http://www.stefan.vogt.net/downloads.html'

        element = []
        for i in range(nels):
                element.append(element_info())

                
        rownum = 1 #skip header
        for row in csvf:
            if (row[0]=='version:') or (row[0]=='') or \
                (row[0]=='aprrox intensity') or (row[0]=='transition') or \
                (row[0]=='Z') :
                continue

            i = int(row[0])-1
            
            element[i].z = int(float(row[0]))
            element[i].name = row[1]
            element[i].xrf['ka1'] = float(row[2]) 
            element[i].xrf['ka2'] = float(row[3]) 
            element[i].xrf['kb1'] = float(row[4]) 
            element[i].xrf['kb2'] = float(row[5]) 
            element[i].xrf['la1'] = float(row[6]) 
            element[i].xrf['la2'] = float(row[7]) 
            element[i].xrf['lb1'] = float(row[8]) 
            element[i].xrf['lb2'] = float(row[9]) 
            element[i].xrf['lb3'] = float(row[10]) 
            element[i].xrf['lb4'] = float(row[11]) 
            element[i].xrf['lg1'] = float(row[12]) 
            element[i].xrf['lg2'] = float(row[13]) 
            element[i].xrf['lg3'] = float(row[14]) 
            element[i].xrf['lg4'] = float(row[15]) 
            element[i].xrf['ll'] = float(row[16]) 
            element[i].xrf['ln'] = float(row[17]) 
            element[i].xrf['ma1'] = float(row[18]) 
            element[i].xrf['ma2'] = float(row[19]) 
            element[i].xrf['mb'] = float(row[20]) 
            element[i].xrf['mg'] = float(row[21]) 
            element[i].yieldD['k'] = float(row[22]) 
            element[i].yieldD['l1'] = float(row[23]) 
            element[i].yieldD['l2'] = float(row[24]) 
            element[i].yieldD['l3'] = float(row[25]) 
            element[i].yieldD['m'] = float(row[26]) 
            element[i].xrf_abs_yield['ka1'] = float(row[27]) 
            element[i].xrf_abs_yield['ka2'] = float(row[28]) 
            element[i].xrf_abs_yield['kb1'] = float(row[29]) 
            element[i].xrf_abs_yield['kb2'] = float(row[30]) 
            element[i].xrf_abs_yield['la1'] = float(row[31]) 
            element[i].xrf_abs_yield['la2'] = float(row[32]) 
            element[i].xrf_abs_yield['lb1'] = float(row[33]) 
            element[i].xrf_abs_yield['lb2'] = float(row[34]) 
            element[i].xrf_abs_yield['lb3'] = float(row[35]) 
            element[i].xrf_abs_yield['lb4'] = float(row[36]) 
            element[i].xrf_abs_yield['lg1'] = float(row[37]) 
            element[i].xrf_abs_yield['lg2'] = float(row[38]) 
            element[i].xrf_abs_yield['lg3'] = float(row[39]) 
            element[i].xrf_abs_yield['lg4'] = float(row[40]) 
            element[i].xrf_abs_yield['ll'] = float(row[41]) 
            element[i].xrf_abs_yield['ln'] = float(row[42]) 
            element[i].xrf_abs_yield['ma1'] = float(row[43]) 
            element[i].xrf_abs_yield['ma2'] = float(row[44]) 
            element[i].xrf_abs_yield['mb'] = float(row[45]) 
            element[i].xrf_abs_yield['mg'] = float(row[46]) 
            
            if len(row) > 46 : 
                element[i].density = float(row[47]) 
                element[i].mass = float(row[48]) 

                element[i].bindingE['K'] = float(row[49]) 

                element[i].bindingE['L1'] = float(row[50]) 
                element[i].bindingE['L2'] = float(row[51]) 
                element[i].bindingE['L3'] = float(row[52]) 

                element[i].bindingE['M1'] = float(row[53]) 
                element[i].bindingE['M2'] = float(row[54]) 
                element[i].bindingE['M3'] = float(row[55]) 
                element[i].bindingE['M4'] = float(row[56]) 
                element[i].bindingE['M5'] = float(row[57]) 

                element[i].bindingE['N1'] = float(row[58]) 
                element[i].bindingE['N2'] = float(row[59]) 
                element[i].bindingE['N3'] = float(row[60]) 
                element[i].bindingE['N4'] = float(row[61]) 
                element[i].bindingE['N5'] = float(row[62]) 
                element[i].bindingE['N6'] = float(row[63]) 
                element[i].bindingE['N7'] = float(row[64]) 

                element[i].bindingE['O1'] = float(row[65]) 
                element[i].bindingE['O2'] = float(row[66]) 
                element[i].bindingE['O3'] = float(row[67]) 
                element[i].bindingE['O4'] = float(row[68]) 
                element[i].bindingE['O5'] = float(row[69]) 

                element[i].bindingE['P1'] = float(row[70]) 
                element[i].bindingE['P2'] = float(row[71]) 
                element[i].bindingE['P3'] = float(row[72]) 


                element[i].jump['K'] = float(row[73]) 

                element[i].jump['L1'] = float(row[74]) 
                element[i].jump['L2'] = float(row[75]) 
                element[i].jump['L3'] = float(row[76]) 

                element[i].jump['M1'] = float(row[77]) 
                element[i].jump['M2'] = float(row[78]) 
                element[i].jump['M3'] = float(row[79]) 
                element[i].jump['M4'] = float(row[80]) 
                element[i].jump['M5'] = float(row[81]) 

                element[i].jump['N1'] = float(row[82]) 
                element[i].jump['N2'] = float(row[83]) 
                element[i].jump['N3'] = float(row[84]) 
                element[i].jump['N4'] = float(row[85]) 
                element[i].jump['N5'] = float(row[86]) 

                element[i].jump['O1'] = float(row[87]) 
                element[i].jump['O2'] = float(row[88]) 
                element[i].jump['O3'] = float(row[89]) 
          
          
        f.close()

        return element
        
        
        
        
        
              
