'''
Created on May 2015

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

# ===================================== Job Status =======================================
JOB_STATUS_NEW = 0
JOB_STATUS_PROCESSING = 1
JOB_STATUS_COMPLETED = 2
JOB_STATUS_CANCELING = 3
JOB_STATUS_CANCELED = 4
JOB_STATUS_GENERAL_ERROR = 10

# ===================================== Job Status =======================================
PROCESS_NODE_STATUS_IDLE = 'Idle'
PROCESS_NODE_STATUS_PROCESSING = 'Processing'
PROCESS_NODE_STATUS_OFFLINE = 'Offline'
PROCESS_NODE_STATUS_BOOT_UP = 'BootUp'

# ===================================== Process Node Keys =======================================
PROCESS_NODE_ID = 'Id'  # INTEGER
PROCESS_NODE_COMPUTERNAME = 'ComputerName'  # TEXT
PROCESS_NODE_NUM_THREADS = 'NumThreads'  # INTEGER
PROCESS_NODE_HOSTNAME = 'Hostname'  # TEXT
PROCESS_NODE_PORT = 'Port'  # INTEGER
PROCESS_NODE_STATUS = 'Status'  # TEXT
PROCESS_NODE_HEARTBEAT = 'Heartbeat'  # TIMESTAMP
PROCESS_NODE_PROCESS_CPU_PERCENT = 'ProcessCpuPercent'  # REAL
PROCESS_NODE_PROCESS_MEM_PERCENT = 'ProcessMemPercent'  # REAL
PROCESS_NODE_SYSTEM_CPU_PERCENT = 'SystemCpuPercent'  # REAL
PROCESS_NODE_SYSTEM_MEM_PERCENT = 'SystemMemPercent'  # REAL
PROCESS_NODE_SYSTEM_SWAP_PERCENT = 'SystemSwapPercent'  # REAL

# ===================================== Job Keys =======================================
JOB_ID = 'Id'  # INTEGER
JOB_DATA_PATH = 'DataPath'  # TEXT
JOB_PROC_MASK = 'ProcMask'  # INTEGER
JOB_VERSION = 'Version'  # TEXT
JOB_DETECTOR_ELEMENTS = 'DetectorElements'  # INTEGER
JOB_MAX_FILES_TO_PROC = 'MaxFilesToProc'  # INTEGER
JOB_MAX_LINES_TO_PROC = 'MaxLinesToProc'  # INTEGER
JOB_QUICK_AND_DIRTY = 'QuickAndDirty'  # INTEGER
JOB_XRF_BIN = 'XRF_Bin'  # INTEGER
JOB_NNLS = 'NNLS'  # INTEGER
JOB_XANES_SCAN = 'XANES_Scan'  # INTEGER
JOB_DETECTOR_TO_START_WITH = 'DetectorToStartWith'  # INTEGER
JOB_BEAM_LINE = 'BeamLine'  # TEXT
JOB_STANDARDS = 'Standards'  # TEXT
JOB_DATASET_FILES_TO_PROC = 'DatasetFilesToProc'  # TEXT
JOB_PRIORITY = 'Priority'  # INTEGER
JOB_STATUS = 'Status'  # INTEGER
JOB_START_PROC_TIME = 'StartProcTime'  # TIMESTAMP
JOB_FINISH_PROC_TIME = 'FinishProcTime'  # TIMESTAMP
JOB_LOG_PATH = 'Log_Path'  # TEXT
JOB_PROCESS_NODE_ID = 'Process_Node_Id'  # INTEGER
JOB_EMAILS = 'Emails' # TEXT comma separated

# ===================================== General Keys =======================================
STR_JOB_LOG_DIR_NAME = 'job_logs'

# ===================================== Email Msg =======================================
EMAIL_SUBJECT_ERROR = 'MapsPy Job Failed'
EMAIL_SUBJECT_COMPLETED = 'MapsPy Job Completed'
EMAIL_MESSAGE_ERROR = 'MapsPy Job Failed: \n'
EMAIL_MESSAGE_COMPLETED = 'MapsPy Job Completed: \n'

# ===================================== HDF5 Groups =======================================
HDF5_GRP_MAPS = 'MAPS'
HDF5_GRP_XRF_ROI = 'XRF_roi'
HDF5_GRP_XRF_ROI_PLUS = 'XRF_roi_plus'
HDF5_GRP_XRF_FITS = 'XRF_fits'
HDF5_GRP_CHANNEL_NAMES = 'channel_names'

# ===================================== Directories =======================================
DIR_MDA = 'mda'
DIR_OUTPUT_OLD = 'output_old'
DIR_OUTPUT_FITS = 'output.fits'
DIR_PCA = 'pca.dir'
DIR_IMG_DAT = 'img.dat'
DIR_LINE_DAT = 'line.dat'
DIR_XANES_DAT = 'xanes.dat'
DIR_FLY_DAT = 'fly.dat'