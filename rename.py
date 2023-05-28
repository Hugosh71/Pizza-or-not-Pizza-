import os

# directory = os.path.join(os.getcwd(), 'pizza')
# for i, filename in enumerate(os.listdir(directory)):
#     new_name = f'pizza{i}.jpg'
#     src = os.path.join(directory, filename)
#     dst = os.path.join(directory, new_name)
#     os.rename(src, dst)
    
# directory = os.path.join(os.getcwd(), 'not_pizza')
# for i, filename in enumerate(os.listdir(directory)):
#     new_name = f'notpizza{i}.jpg'
#     src = os.path.join(directory, filename)
#     dst = os.path.join(directory, new_name)
#     print(f'Renaming {src} to {dst}')
#     os.rename

import os
import shutil

source_folder = os.path.join(os.getcwd(), 'not_pizza')
destination_folder = os.path.join(os.getcwd(), 'not_pizza_full')
i = 1

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

for filename in os.listdir(source_folder):
    if filename.endswith('.jpg'):
        new_filename = f'notpizza{i}.jpg'
        source = os.path.join(source_folder, filename)
        destination = os.path.join(destination_folder, new_filename)
        shutil.copy2(source, destination)
        i += 1