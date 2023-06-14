import os, json, re, random
import numpy as np
from pprint import pprint
import pandas as pd

def get_action_times(path, need_trans=True):
    is_start, is_in_start = False, False
    usr_acts, sys_acts = ["INFORM", "REQUEST", "INFORM_INTENT", "THANK_YOU", "AFFIRM", "SELECT", "NEGATE", "REQUEST_ALTS", "GOODBYE", "NEGATE_INTENT", "AFFIRM_INTENT", "none"],\
         ["INFORM", "REQUEST", "OFFER", "GOODBYE", "CONFIRM", "INFORM_COUNT", "NOTIFY_SUCCESS", "REQ_MORE", "OFFER_INTENT", "NOTIFY_FAILURE", "none"]
    act_times = {}
    error_files = []
    for split in os.listdir(path):
        for d_file in sorted(os.listdir(os.path.join(path, split))):
            file_name = os.path.join(os.path.join(path, split), d_file)
            dialogs = json.loads(open(file_name, encoding='utf-8').read())
            for dialog in dialogs:
                #pprint(dialog)  
                for turn in dialog["turns"]:
                    for frame in turn["frames"]:  
                        for a in frame["actions"]:
                            if not a['act'] in act_times: act_times[a['act']] = 1
                            else: act_times[a['act']] += 1
    print(act_times)         
    #with open(f'{path}_error.txt', 'w') as f:
    #    for line in sorted(list(set(error_files))):
    #        f.write(f"{line}\n")
    return error_files

if __name__=="__main__":
    path = "processed"
    get_action_times(path, need_trans=False)    
    