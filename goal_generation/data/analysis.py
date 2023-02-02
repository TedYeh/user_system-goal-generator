import os, json, re, random
import numpy as np
from pprint import pprint
import pandas as pd

def draw_matrix(matrix, select_acts=[], output_acts=[], labels=[], img_name='default'):    
    fig, ax = plt.subplots()
    plt.title(img_name)
    im = ax.imshow(matrix, cmap='OrRd', aspect='auto')
    # Major ticks
    ax.set_xticks(np.arange(0, len(matrix[0]), 1))
    ax.set_yticks(np.arange(0, len(matrix), 1))    

    # Labels for major ticks
    ax.set_xticklabels(output_acts, rotation=90, color="red" if labels[0]!='Client' else 'blue')
    ax.set_yticklabels(select_acts, rotation=0, color="red" if labels[1]!='Client' else 'blue')    
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False, color="red" if labels[0]!='Client' else 'blue')
    ax.tick_params(axis="y", color="red" if labels[1]!='Client' else 'blue')
    ax.set_xlabel('decide action'+f"({labels[0]})", color="red" if labels[0]!='Client' else 'blue')
    ax.set_ylabel('select action'+f"({labels[1]})", color="red" if labels[1]!='Client' else 'blue')    

    # Minor ticks
    ax.set_xticks(np.arange(-.5, len(matrix[0]), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(matrix), 1), minor=True)  

    fig.colorbar(im)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            c = matrix[i, j]
            if c > 0:
                ax.text(j, i, str(c), va='center', ha='center')
    plt.grid(which='minor', color='k', linewidth=1)
    fig.tight_layout()
    plt.savefig(os.path.join('../transistion matrix', img_name))
    plt.show()

def get_action_times(paths, out_file, need_trans=True):
    is_start, is_in_start = False, False
    usr_acts, sys_acts = ["INFORM", "REQUEST", "INFORM_INTENT", "THANK_YOU", "AFFIRM", "SELECT", "NEGATE", "REQUEST_ALTS", "GOODBYE", "NEGATE_INTENT", "AFFIRM_INTENT", "none"],\
         ["INFORM", "REQUEST", "OFFER", "GOODBYE", "CONFIRM", "INFORM_COUNT", "NOTIFY_SUCCESS", "REQ_MORE", "OFFER_INTENT", "NOTIFY_FAILURE", "none"]
    usr_index, sys_index = 0, 0 
    usr_in_index, sys_in_index = -1, -1 
    usr_matrix, tmp_usr_matrix = np.zeros((len(sys_acts)-1, len(usr_acts)-1)), np.zeros((len(sys_acts)-1, len(usr_acts)-1))
    sys_matrix, tmp_sys_matrix = np.zeros((len(usr_acts)-1, len(sys_acts)-1)), np.zeros((len(usr_acts)-1, len(sys_acts)-1))
    acts = []
    usr_inner_matrix = np.zeros((len(usr_acts), len(usr_acts)))
    sys_inner_matrix = np.zeros((len(sys_acts), len(sys_acts)))
    value_list = analysis_schema('schema.json')
    action_times = {}
    for path in paths:
        file_path = os.path.join('./sgd', path)
        for d_file in os.listdir(file_path)[:-1]:
            file_name = os.path.join(file_path, d_file)
            dialogs = json.loads(open(file_name, encoding='utf-8').read())
            for dialog in dialogs:
                tmp_usr_matrix = np.zeros((len(sys_acts)-1, len(usr_acts)-1))
                tmp_sys_matrix = np.zeros((len(usr_acts)-1, len(sys_acts)-1))
                tran_bool = is_transactional(value_list, dialog["turns"][0]["frames"][0]["actions"][0]["values"][-1]) or \
                     is_transactional(value_list, dialog["turns"][0]["frames"][0]["actions"][-1]["values"][-1])
                if tran_bool ==(not need_trans): continue      
                for turn in dialog["turns"]:
                    acts = []
                    for frame in turn["frames"]:  
                        for a in frame["actions"]:
                            if not a in acts:acts.append(a["act"])
                            #if a["act"] == "INFORM_INTENT":
                            #    tran_bool = is_transactional(value_list, a["values"][-1])
                            #    if tran_bool == need_trans: break 
                        #print(acts) 
                        #input()
                        act = dict(a)
                        if tran_bool ==(not need_trans): continue
                        #print(act['act'], act["values"])  
                        #input()
                        if not is_start: 
                            if act["act"] == "INFORM_INTENT" and is_transactional(value_list, act["values"][-1])==(not need_trans):continue
                            is_start = True
                            if turn['speaker'] == 'USER':usr_index = usr_acts.index(act['act'])
                            else:sys_index = sys_acts.index(act['act'])
                        else:                            
                            if act["act"] == "INFORM_INTENT" and not is_transactional(value_list, act["values"][-1]):continue
                            if turn['speaker'] == 'USER':
                                usr_index = usr_acts.index(act['act'])
                                tmp_usr_matrix[sys_index, usr_index] += 1
                            else:
                                sys_index = sys_acts.index(act['act'])
                                tmp_sys_matrix[usr_index, sys_index] += 1   
                usr_matrix += np.copy(tmp_usr_matrix)
                sys_matrix += np.copy(tmp_sys_matrix)
                is_start = False

    pprint(action_times)
    df = pd.DataFrame({'Action_name':action_times.keys(), "Amount":action_times.values()})
    df.to_csv('times.csv', index=False)
    print("usr")
    print(usr_matrix)
    print("sys")
    print(sys_matrix)    

    print("usr inner")
    print(usr_inner_matrix)
    print("sys inner")
    print(sys_inner_matrix)
    with open(out_file, 'wb') as f:
        np.save(f, usr_matrix)
        np.save(f, sys_matrix)
        np.save(f, usr_inner_matrix)
        np.save(f, sys_inner_matrix)

def get_inner_action_times(paths):
    is_start, is_in_start = False, False
    usr_acts, sys_acts = ["INFORM", "REQUEST", "INFORM_INTENT", "THANK_YOU", "AFFIRM", "SELECT", "NEGATE", "REQUEST_ALTS", "GOODBYE", "NEGATE_INTENT", "AFFIRM_INTENT", "none"],\
         ["INFORM", "REQUEST", "OFFER", "GOODBYE", "CONFIRM", "INFORM_COUNT", "NOTIFY_SUCCESS", "REQ_MORE", "OFFER_INTENT", "NOTIFY_FAILURE", "none"]
    usr_index, sys_index = 0, 0 
    usr_in_index, sys_in_index = -1, -1 
    usr_inner_matrix = np.zeros((len(usr_acts), len(usr_acts)))
    sys_inner_matrix = np.zeros((len(sys_acts), len(sys_acts)))
    for path in paths:
        file_path = os.path.join('./sgd', path)
        for d_file in os.listdir(file_path)[:-1]:
            file_name = os.path.join(file_path, d_file)
            dialogs = json.loads(open(file_name, encoding='utf-8').read())
            for dialog in dialogs:    
                for turn in dialog["turns"]:
                    usr_acts_l, sys_acts_l = [], []
                    for frame in turn["frames"]: 
                        for act_idx in range(len(frame["actions"])):
                            if not is_start: 
                                is_start = True
                                if turn['speaker'] == 'USER':
                                    usr_index = usr_acts.index(frame["actions"][act_idx]['act'])
                                    usr_acts_l.append(frame["actions"][act_idx]['act'])
                                else:
                                    sys_index = sys_acts.index(frame["actions"][act_idx]['act'])
                                    sys_acts_l.append(frame["actions"][act_idx]['act'])
                            else:                            
                                if turn['speaker'] == 'USER':                                    
                                    if act_idx == len(frame["actions"])-1:
                                        if not frame["actions"][act_idx]['act'] in usr_acts_l:
                                            usr_index = usr_acts.index(frame["actions"][act_idx]['act'])
                                            usr_inner_matrix[usr_index, -1] += 1
                                            usr_acts_l.append(frame["actions"][act_idx]['act'])
                                    else:
                                        if not frame["actions"][act_idx]['act'] in usr_acts_l:
                                            usr_inner_matrix[usr_index, usr_acts.index(frame["actions"][act_idx]['act'])] += 1
                                            usr_index = usr_acts.index(frame["actions"][act_idx]['act'])
                                            usr_acts_l.append(frame["actions"][act_idx]['act'])
                                else:
                                    if act_idx == len(frame["actions"])-1:
                                        if not frame["actions"][act_idx]['act'] in sys_acts_l:
                                            sys_index = sys_acts.index(frame["actions"][act_idx]['act'])
                                            sys_inner_matrix[sys_index, -1] += 1
                                            sys_acts_l.append(frame["actions"][act_idx]['act'])
                                    else:
                                        if not frame["actions"][act_idx]['act'] in sys_acts_l:
                                            sys_inner_matrix[sys_index, sys_acts.index(frame["actions"][act_idx]['act'])] += 1
                                            sys_index = sys_acts.index(frame["actions"][act_idx]['act'])
                                            sys_acts_l.append(frame["actions"][act_idx]['act'])
                    is_start = False

    for matrix in [usr_inner_matrix, sys_inner_matrix]:
        for i in range(matrix.shape[0]):
            if not sum(matrix[i])==0:
                matrix[i] /= sum(matrix[i])
            else: matrix[i][-1] = 1.
        print(np.around(matrix, decimals=3))
    #print("usr inner")
    #print(np.around(usr_inner_matrix, decimals=3))
    #print("sys inner")
    #print(np.around(sys_inner_matrix, decimals=3))

def get_weighted_matrix(matrix_file, out_file):   
    with open(matrix_file, 'rb') as f:
        usr_matrix, sys_matrix, usr_inner_matrix, sys_inner_matrix = [np.load(f) for _ in range(4)]
        usr_matrix[3, 2] = 1.

    with open(out_file, 'wb') as f:
        for matrix in [usr_matrix, sys_matrix]:
            for i in range(matrix.shape[0]):
                if not sum(matrix[i])==0:
                    matrix[i] /= sum(matrix[i])
            print(np.around(matrix, decimals=3))
            np.save(f, matrix)
        for matrix in [usr_inner_matrix, sys_inner_matrix]:
            for i in range(matrix.shape[0]):
                if not sum(matrix[i])==0:
                    matrix[i] /= sum(matrix[i])
                else: matrix[i][-1] = 1.
            print(np.around(matrix, decimals=3))
            np.save(f, matrix)

def analysis_schema(schema_file):
    schemas = json.loads(open(schema_file, "r", encoding="utf-8").read())
    values = []
    for schema in schemas:
        for intent in schema["intents"]:
            s = list(intent.values())
            values.append([s[0], s[2]])
    return values

def is_transactional(value_list, intent):
    for value in value_list:
        if intent in value:
            if value[1]: return True
            else: return False
        

def analysis_ptt(path):
    comments_amount = 0
    file_path = os.path.join('./', path)
    for d_file in os.listdir(file_path)[:-1]:
        file_name = os.path.join(file_path, d_file)
        comments = json.loads(open(file_name, encoding='utf-8').read())
        for comment in comments:
            comments_amount += len(comment)
    print(comments_amount)          

EMOJI = re.compile("["
                u"\u00A9" 
                u"\u00AE" 
                u"\u203C" 
                u"\u2049" 
                u"\u20E3" 
                u"\u2122" 
                u"\u2139" 
                u"\u231A"
                u"\u2060" 
                u"\u231B" 
                u"\u2328" 
                u"\u23CF" 
                u"\u24C2" 
                u"\u25AA" 
                u"\u25AB" 
                u"\u25B6" 
                u"\u25C0" 
                u"\u2934" 
                u"\u2935"
                u"\u3030" 
                u"\u303D" 
                u"\u3297" 
                u"\u3299"
                u"\uFFFD"
                u"\u23E9-\u23F3" 
                u"\u23F8-\u23FA"
                u"\u25FB-\u25FE" 
                u"\u2600-\u27EF"  
                u"\u2B00-\u2BFF"                 
                u"\u2194-\u2199" 
                u"\u21A9-\u21AA"  
                u"\U0001F000-\U0001F02F" 
                u"\U0001F0A0-\U0001F0FF" 
                u"\U0001F100-\U0001F64F" 
                u"\U0001F680-\U0001F6FF" 
                u"\U0001F910-\U0001F96B" 
                u"\U0001F980-\U0001F9E0"
                        "]+", re.UNICODE)

def preprocess_line_chat(text_path):
    data=[['group_name', 'contact_name', 'message']]
    with open('user_list.txt', 'r', encoding = 'utf-8') as f:
        usr_list = f.readlines()
    text_list = os.listdir(text_path)
    for text_file in text_list:
        with open(os.path.join(text_path, text_file), 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line: break
                chat_content = line.split('\t')
                chat_content[-1] = EMOJI.sub(r'', chat_content[-1]).replace(',', '').replace('\n', '')                
                if len(chat_content)!=3 or ('/' in chat_content[-1]) or (len(chat_content[-1])<=3) \
                     or ('[Sticker]'in chat_content[-1]) or ('[Photo]'in chat_content[-1]) or ('[Video]'in chat_content[-1]):continue
                user = random.choices(usr_list, k=1)[0].replace('\n', '')
                chat_content[1] = EMOJI.sub(r'', chat_content[1]).replace(',', '').replace(' ', '')
                if random.choices([True, False], weights=[5, 5], k=1)[0]: data.append([text_file.replace('.txt', ''), user, chat_content[-1]])
                else:data.append(['無', user, chat_content[-1]])
                print(data[-1])
                #input()
    list_rows = np.array(data)
    np.savetxt("csv/message_entityies_line.csv", list_rows, delimiter =",",fmt ='% s', encoding='utf-8-sig')

if __name__=="__main__":
    paths = ['train', 'test', 'dev']
    value_list = analysis_schema('schema.json')
    time_matrix = '../matrix/non_matrix.npy'
    time_inner_matrix = '../matrix/non_inner_matrix.npy'
    get_action_times(paths, time_matrix, need_trans=False)    
    get_weighted_matrix(time_matrix, '../matrix/non_matrix_weighted.npy')
    print('inner')
    get_inner_action_times(paths)
    #path = 'ptt\\data\\source_replies\\reply'
    #analysis_ptt(path)
    '''
    tmp = json.loads(open('sgd/train/dialogues_010.json', 'r').read())
    for i in tmp:
        for turn in i['turns']:
            for frames in turn['frames']:
                pprint(frames)
                print(len(frames))
                input()
    '''
    matrix_file = './matrix/matrix_weighted.npy'
    with open(matrix_file, 'rb') as f:
        usr_matrix, sys_matrix, usr_inner_matrix, sys_inner_matrix = [np.load(f) for _ in range(4)]

    non_matrix_file = './matrix/non_matrix_weighted.npy'
    with open(non_matrix_file, 'rb') as f:
        non_usr_matrix, non_sys_matrix, _, _ = [np.load(f) for _ in range(4)]

    usr_matrix = np.around(usr_matrix, decimals=2)    
    draw_matrix(usr_matrix, sys_acts[:-1], usr_acts[:-1], ['Client', 'Assistant'], 'Decide Client (transactional)')

    sys_matrix = np.around(sys_matrix, decimals=2)
    draw_matrix(sys_matrix, usr_acts[:-1], sys_acts[:-1], ['Assistant', 'Client'], 'Decide Assistant (transactional)')

    non_usr_matrix = np.around(non_usr_matrix, decimals=2)    
    draw_matrix(non_usr_matrix, sys_acts[:-1], usr_acts[:-1], ['Client', 'Assistant'], 'Decide Client (non-transactional)')

    non_sys_matrix = np.around(non_sys_matrix, decimals=2)
    draw_matrix(non_sys_matrix, usr_acts[:-1], sys_acts[:-1], ['Assistant', 'Client'], 'Decide Assistant (non-transactional)')
    '''
    usr_inner_matrix = np.around(usr_inner_matrix, decimals=2)
    draw_matrix(usr_inner_matrix, usr_acts, usr_acts, ['Client', 'Client'], 'Continue Decide Client')

    sys_inner_matrix = np.around(sys_inner_matrix, decimals=2)
    draw_matrix(sys_inner_matrix, sys_acts, sys_acts, ['Assistant', 'Assistant'], 'Continue Decide Assistant')
    '''
    #text_path = "./chat_txt"        
    #preprocess_line_chat(text_path)