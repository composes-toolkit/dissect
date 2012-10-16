'''
Created on Oct 16, 2012

@author: nghia
'''

import pickle
import os

def save(object_, file_name):
    create_parent_directories(file_name)
    with open(file_name,'w') as f:
        pickle.dump(object_, f)

def load(file_name):
    with open(file_name) as f:
        return pickle.load(f)

def create_directories(directory):
    if (not os.path.exists(directory)):
        os.makedirs(directory)
    
def create_parent_directories(file_name):
    parts = file_name.split("/")
    if (len(parts) > 1):
        parent_dir = "/".join(parts[0:-1])
        if (not os.path.exists(parent_dir)):
            print parent_dir
            os.makedirs(parent_dir)

def print_list(list_, file_name):
    with open(file_name,'w') as f:
        for item in list_:
            f.write(item + "\n")