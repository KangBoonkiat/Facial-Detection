# Import Libraries

import os

# Move the Label with the same ID to the respectively folder

for folder in ['Train','Test', 'Val']:
    for file in os.listdir(os.path.join('new_data', folder, 'images')):
        filename = file.split('.')[0]+'.json'
        existing_filepath = os.path.join('new_data','annotation', filename)
        if os.path.exists(existing_filepath):
            new_filepath = os.path.join('new_data', folder, 'labels', filename)
            os.replace(existing_filepath, new_filepath)