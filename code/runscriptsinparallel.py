from cblparallel import pyfear
from cblparallel.config import *
import cblparallel
from cblparallel.util import mkstemp_safe

import os
import shutil

import scipy.io
import numpy as np

# Move data file to fear after saving to temporary location

for file_name in os.listdir('data'):
    if file_name[-4:] == '.mat':
        data_file = os.path.join('data', file_name)
        break

temp_data_file_name = mkstemp_safe(LOCAL_TEMP_PATH, '.mat')
shutil.move(data_file, temp_data_file_name)

cblparallel.copy_to_remote(temp_data_file_name)

# Load scripts

scripts = []

script_names = sorted(os.listdir('scripts'))
used_script_names = []

for file_name in script_names:
    if file_name[-2:] == '.m':
#       print 'Reading %s' % file_name
        used_script_names.append(file_name)
        with open(os.path.join('scripts', file_name)) as script_file:
            scripts.append(script_file.read() % {'data_file' : os.path.split(temp_data_file_name)[-1], 'output_file' : '%(output_file)s'})

script_names = used_script_names

# Send to cluster

# print 'Found %d scripts' % len(scripts)

output_files = cblparallel.run_batch_on_fear(scripts, language='matlab', max_jobs=1000, verbose=False, zip_files=False, bundle_size=1)

# print '%d output files returned' % len(output_files)

# Move output

for (src, name) in zip(output_files, script_names):
  #  print 'Moving %s output' % name.split('.')[0]
    dest = os.path.join('outputs', name.split('.')[0] + '.mat')
    shutil.move(src, dest)
   # print 'Success'

# Delete local data

#for file_name in os.listdir('data'):
#    if file_name[-4:] == '.mat':
#        os.remove(os.path.join('data', file_name))

os.remove(temp_data_file_name)

# Delete local scripts

for file_name in os.listdir('scripts'):
    if file_name[-2:] == '.m':
        os.remove(os.path.join('scripts', file_name))

# Demo of reading output in python

# output_names = os.listdir('outputs')

# output_values = np.zeros((len(scripts), 2))

# for (i, output) in enumerate(output_names):
#     data = scipy.io.loadmat(os.path.join('outputs', output))
#     output_values[i,0] = data['bicValue'].ravel()[0]
#     output_values[i,1] = data['hypN'].ravel()[0]
