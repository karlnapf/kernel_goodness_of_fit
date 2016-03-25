import os
from os.path import expanduser
from time import strftime, gmtime, sleep

import pandas as pd


def store_results(fname = expanduser("~") + os.sep  + "results.txt", **kwargs):
    # add filename if only path is given
    if fname[-1] == os.sep:
        fname += "results.txt"
    
    # create result dir if wanted
    if os.sep in fname:
        try:
            directory = os.sep.join(fname.split(os.sep)[:-1])
            os.makedirs(directory)
        except OSError:
            pass
    
    # use current time as index for the dataframe
    current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    new_df = pd.DataFrame([[kwargs[k] for k in kwargs.keys()]], index=[current_time],columns=kwargs.keys())
    
    if os.path.exists(fname):
        df = pd.read_csv(fname, index_col=0)
        df = df.append(new_df)
    else:
        df = new_df

    # very crude protection against conflicting access from parallel processes
    write_success = False
    while not write_success:
        try:
            df.to_csv(fname)
            write_success = True
        except IOError:
            print("IOError writing to csv ... trying again in 1s.")
            sleep(1)
        
