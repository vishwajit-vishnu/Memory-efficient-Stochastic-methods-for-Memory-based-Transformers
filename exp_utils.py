###### exp_utils.py ########
## To save models and log

import os, shutil;
import numpy as np;
import torch;
import functools; 

def logging(s, log_path, print_=False, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f:
            f.write(s + '\n')

def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)

def create_exp_dir(dir_path, scripts_to_save=None):
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # print('Experiment dir : {}'.format(dir_path))
        if scripts_to_save is not None:
            script_path = os.path.join(dir_path, 'scripts')
            if not os.path.exists(script_path):
                os.makedirs(script_path)
            for script in scripts_to_save:
                dst_file = os.path.join(dir_path, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)
    except OSError as err:
        #print("probably eroor due to race condition but dont worry, printing it");
        #print(err);
        pass;

    return get_logger(log_path=os.path.join(dir_path, 'log.txt'));

