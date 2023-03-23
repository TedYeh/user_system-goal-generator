import os, json, re, random
import numpy as np
from pprint import pprint
import pandas as pd

def get_action_times(path, need_trans=True):
    is_start, is_in_start = False, False
    usr_acts, sys_acts = ["INFORM", "REQUEST", "INFORM_INTENT", "THANK_YOU", "AFFIRM", "SELECT", "NEGATE", "REQUEST_ALTS", "GOODBYE", "NEGATE_INTENT", "AFFIRM_INTENT", "none"],\
         ["INFORM", "REQUEST", "OFFER", "GOODBYE", "CONFIRM", "INFORM_COUNT", "NOTIFY_SUCCESS", "REQ_MORE", "OFFER_INTENT", "NOTIFY_FAILURE", "none"]
    
    error_files = []
    for d_file in sorted(os.listdir(path)):
        file_name = os.path.join(path, d_file)
        dialogs = json.loads(open(file_name, encoding='utf-8').read())
        
        for dialog in dialogs:
            #pprint(dialog)  
            for turn in dialog["turns"]:
                for frame in turn["frames"]:  
                    for a in frame["actions"]:
                        if a['act'] in ["INFORM", "CONFIRM", "OFFER"]:
                            for v in a['values']:
                                if v!='無' and v.strip() not in turn["utterance"]:
                                    error_files.append(d_file)
    pprint(sorted(list(set(error_files))))  
    print(len(list(set(error_files))))             
    with open(f'{path}_error.txt', 'w') as f:
        for line in sorted(list(set(error_files))):
            f.write(f"{line}\n")

if __name__=="__main__":
    path = "labeled_dialog_110801006"
    get_action_times(path, need_trans=False)    
    