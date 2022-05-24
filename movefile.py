#Import Libraries

import os
import random
import shutil

# Move a set of data randomly to other folders

source = 'new_data\Train\images'
dest = 'new_data\Val\images'
files = os.listdir(source)
no_of_files = 540

for file_name in random.sample(files, no_of_files):
    shutil.move(os.path.join(source, file_name), dest)