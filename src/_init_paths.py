import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

paths = ['lib', 'models']
this_dir = os.path.dirname(__file__)
add_path(this_dir)  # src path

proj_root = os.path.abspath(__file__ + "/../../")   # Root path
add_path(os.path.join(proj_root, 'dataset')) 
add_path(os.path.join(proj_root, 'configs')) 
add_path(os.path.join(proj_root, 'results')) 

for dir in paths:
    dir_path = os.path.join(this_dir, dir)
    add_path(dir_path)

