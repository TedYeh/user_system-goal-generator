import copy
import json
import os
import re
from collections import Counter
from pprint import pprint
from shutil import copy2, rmtree
from zipfile import ZIP_DEFLATED, ZipFile
from copy import deepcopy
from tqdm import tqdm

def time_to_str(times):
    if isinstance(times, str): return times
    elif not times[0]: return ''
    else:
        time_str = ''
        date1, t1 = times[0].split('T')
        d, h, m, amOrpm = date1[5:].replace('-', '/'), t1[:2]+'點', '' if int(t1[3:]) == 0 else t1[3:]+'分', '早上' if int(t1[:2]) < 12 else '下午'
        time_str += d + amOrpm + h + m    
        date2, t2 = times[1].split('T')
        d, h, m, amOrpm = '' if date2[5:].replace('-', '/')==date1[5:].replace('-', '/') else date2[5:].replace('-', '/')\
            , t2[:2]+'點', '' if int(t2[3:]) == 0 else t2[3:]+'分', '早上' if int(t2[:2]) < 12 else '下午'
        time_str += '到'+ d + amOrpm + h + m
        #print(times, time_str)
        return time_str

ontology = {
    "domains": {
        "Message": {
            "description": "傳送訊息",
            "slots": {
                "收件者": {
                    "description": "收件者",
                    "is_categorical": False,
                    "possible_values": []
                },
                "傳送內容": {
                    "description": "傳送內容",
                    "is_categorical": False,
                    "possible_values": []
                },
                "應用程式": {
                    "description": "應用程式",
                    "is_categorical": False,
                    "possible_values": []
                }
            }
        },
        "Gmail": {
            "description": "查找/傳送郵件",
            "slots": {
                "郵件主旨": {
                    "description": "郵件主旨",
                    "is_categorical": False,
                    "possible_values": []
                },
                "寄件者": {
                    "description": "寄件者",
                    "is_categorical": False,
                    "possible_values": []
                },
                "收件者": {
                    "description": "收件者",
                    "is_categorical": False,
                    "possible_values": []
                },
                "副本收件者": {
                    "description": "副本收件者",
                    "is_categorical": False,
                    "possible_values": []
                },
                "密件副本收件者": {
                    "description": "密件副本收件者",
                    "is_categorical": False,
                    "possible_values": []
                },
                "內容": {
                    "description": "信件內容",
                    "is_categorical": False,
                    "possible_values": []
                }
            }
        },
        "Calendar": {
            "description": "查找/建立活動",
            "slots": {
                "活動名稱": {
                    "description": "活動名稱",
                    "is_categorical": False,
                    "possible_values": []
                },
                "活動時間": {
                    "description": "活動時間",
                    "is_categorical": False,
                    "possible_values": []
                },
                "參加者": {
                    "description": "參加者",
                    "is_categorical": False,
                    "possible_values": []
                },
                "是否全天": {
                    "description": "是否全天",
                    "is_categorical": True,
                    "possible_values": ["是", "否"]
                },
                "活動內容": {
                    "description": "活動內容",
                    "is_categorical": False,
                    "possible_values": []
                },
                "活動地點": {
                    "description": "活動地點",
                    "is_categorical": False,
                    "possible_values": []
                }
            }
        },
        "General": {
            "description": "通用領域/社交言語",
            "slots": {}
        }
    },
    "intents": {
        "Inform": {
            "description": "告知相關資訊"
        },
        "Request": {
            "description": "詢問相關資訊"
        },
        "General": {
            "description": "社交言語"
        },
        "Select": {
            "description": "在相關資訊下查詢或執行"
        },
        "NoFound": {
            "description": "未找到符合使用者的要求"
        },
        "bye": {
            "description": "再見"
        },
        "thanks": {
            "description": "感謝"
        },
        "welcome": {
            "description": "不客氣"
        },
        "greet": {
            "description": "打招呼"
        },
        "confirm": {
            "description": "確認"
        },
        "done": {
            "description": "完成任務"
        },
        "reqmore": {
            "description": "需要其他請求"
        },
        "noneed": {
            "description": "不需要"
        }
    },
    "state": {
        "Calendar": {
            "None": "",
            "活動名稱": "",
            "活動時間": "",
            "參加者": "",
            "是否全天": "",
            "活動內容": "",
            "活動地點": ""
        },
        "Gmail": {
            "None": "",
            "郵件主旨": "",
            "寄件者": "",
            "收件者": "",
            "副本收件者": "",
            "密件副本收件者": "",
            "內容": ""
        },
        "Message": {
            "None": "",
            "收件者": "",
            "傳送內容": "",
            "應用程式": ""
        }
    },
    "dialogue_acts": {
        "categorical": {},
        "non-categorical": {},
        "binary": {}
    }
}

cnt_domain_slot = Counter()

def convert_da(da_list, utt):
    '''
    convert dialogue acts to required format
    :param da_dict: list of (intent, domain, slot, value)
    :param utt: user or system utt
    '''
    global ontology, cnt_domain_slot

    converted_da = {
        'categorical': [],
        'non-categorical': [],
        'binary': []
    }

    for intent, domain, slot, value in da_list:
        if slot == "活動時間":
            value = time_to_str(value)
        # if intent in ['Inform', 'Recommend']:
        if intent == 'NoFound':
            assert slot == 'none' and value == 'none'
            converted_da['binary'].append({
                'intent': intent,
                'domain': domain,
                'slot': ''
            })
        elif intent == 'General':
            # intent=General, domain=thank/bye/greet/welcome
            assert slot == 'none' and value == 'none'
            converted_da['binary'].append({
                'intent': domain,
                'domain': intent,
                'slot': ''
            })
        elif intent == 'Request':
            #print(value, slot)
            assert (value == 'none' or value == '') and (slot == 'none'or slot != 'none')
            converted_da['binary'].append({
                'intent': intent,
                'domain': domain,
                'slot': slot
            })
        elif intent == 'Select':
            #assert slot == '源领域'
            converted_da['binary'].append({
                'intent': intent,
                'domain': domain,
                'slot': f"{slot}-{value}"
            })
        else:
            #print(intent)
            assert intent in ['Inform']
            assert slot != 'none' and value != 'none'
            matches = utt.count(value)
            if matches == 1:
                start = utt.index(value)
                end = start + len(value)
                
                converted_da['non-categorical'].append({
                    'intent': intent,
                    'domain': domain,
                    'slot': slot,
                    'value': value,
                    'start': start,
                    'end': end
                })
                cnt_domain_slot['have span'] += 1
            else:
                # can not find span
                converted_da['non-categorical'].append({
                    'intent': intent,
                    'domain': domain,
                    'slot': slot,
                    'value': value
                })
                cnt_domain_slot['no span'] += 1
            # cnt_domain_slot.setdefault(f'{domain}-{slot}', set())
            # cnt_domain_slot[f'{domain}-{slot}'].add(value)
        
    return converted_da

def transform_user_state(user_state):
    goal = []
    for subgoal in user_state:
        gid, domain, slot, value, mentioned = subgoal
        if len(value) != 0:
            t = 'inform'
        else:
            t = 'request'
        if len(goal) < gid:
            goal.append({domain: {'inform': {}, 'request': {}}})
        goal[gid-1][domain][t][slot] = [value, 'mentioned' if mentioned else 'not mentioned']
    return goal


def preprocess():
    k = 1
    original_data_dir = f'./{k}_fold_test'
    new_data_dir = f'data'

    os.makedirs(new_data_dir, exist_ok=True)
    
    for filename in os.listdir(os.path.join(original_data_dir,'database')):
        copy2(f'{original_data_dir}/database/{filename}', new_data_dir)
    

    global ontology

    dataset = 'messagewoz'
    splits = ['train', 'validation', 'test']
    dialogues_by_split = {split: [] for split in splits}
    for split in ['train', 'val', 'test']:
        data = json.load(ZipFile(os.path.join(original_data_dir, f'{split}.json.zip'), 'r').open(f'{split}.json'))
        if split == 'val':
            split = 'validation'
    
        for ori_dialog_id, ori_dialog in data.items():
            dialogue_id = f'{dataset}-{split}-{len(dialogues_by_split[split])}'

            # get user goal and involved domains
            goal = {'inform': {}, 'request': {}}
            #goal["description"] = '\n'.join(ori_dialog["task description"])
            cur_domains = [x[1] for i, x in enumerate(ori_dialog['goal']) if i == 0 or ori_dialog['goal'][i-1][1] != x[1]]

            dialogue = {
                'dataset': dataset,
                'data_split': split,
                'dialogue_id': dialogue_id,
                'original_id': ori_dialog_id,
                'domains': cur_domains,
                'goal': goal,
                'user_state_init': transform_user_state(ori_dialog['goal']),
                #'type': ori_dialog['type'],
                'turns': [],
                'user_state_final': transform_user_state(ori_dialog['final_goal'])
            }
            
            for turn_id, turn in enumerate(ori_dialog['messages'][:]):

                speaker = 'user' if turn['role'] == 'usr' else 'system'
                utt = turn['content']

                das = turn['dialog_act']

                dialogue_acts = convert_da(das, utt)

                dialogue['turns'].append({
                    'speaker': speaker,
                    'utterance': utt,
                    'utt_idx': len(dialogue['turns']),
                    'dialogue_acts': dialogue_acts,
                })

                # add to dialogue_acts dictionary in the ontology
                for da_type in dialogue_acts:
                    das = dialogue_acts[da_type]
                    for da in das:
                        ontology["dialogue_acts"][da_type].setdefault((da['intent'], da['domain'], da['slot']), {})
                        ontology["dialogue_acts"][da_type][(da['intent'], da['domain'], da['slot'])][speaker] = True
                #pprint(dialogue)
                if speaker == 'user':
                    for u_state in turn['user_state']:
                        if u_state[-2]: belief_state[u_state[1]][u_state[2]] = u_state[-2]
                    dialogue['turns'][-1]['user_state'] = transform_user_state(turn['user_state']) 
                    dialogue['turns'][-1]['state'] = deepcopy(dict(belief_state))                   
                else:
                    # add state to last user turn
                    belief_state = turn['sys_state_init']
                    for domain in belief_state:
                        if domain == 'selectedResults':continue
                        #del belief_state['selectedResults']
                        belief_state[domain].pop('selectedResults')
                    dialogue['turns'][-1]['state'] = deepcopy(dict(belief_state))
                    
                    db_query = turn['sys_state']
                    db_results = {}
                    for domain in list(db_query.keys()):
                        if domain == 'selectedResults':db_res = db_query['selectedResults']
                        else: db_res = db_query[domain].pop('selectedResults')
                        if len(db_res) > 0:
                            db_results[domain] = [{'名稱': x} for x in db_res]
                        else:
                            db_query.pop(domain)
                    dialogue['turns'][-1]['db_query'] = db_query
                    dialogue['turns'][-1]['db_results'] = db_results
                pprint(dialogue['turns']) 
                input()   
            dialogues_by_split[split].append(deepcopy(dialogue))
    pprint(cnt_domain_slot.most_common())
    dialogues = []
    for split in splits:
        dialogues += dialogues_by_split[split]
    for da_type in ontology['dialogue_acts']:
        ontology["dialogue_acts"][da_type] = sorted([str(
            {'user': speakers.get('user', False), 'system': speakers.get('system', False), 'intent': da[0],
             'domain': da[1], 'slot': da[2]}) for da, speakers in ontology["dialogue_acts"][da_type].items()])
    json.dump(dialogues[:10], open(f'dummy_data.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(ontology, open(f'{new_data_dir}/ontology.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(dialogues, open(f'{new_data_dir}/dialogues.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    with ZipFile(f'data.zip', 'w', ZIP_DEFLATED) as zf:
        for filename in os.listdir(new_data_dir):
            zf.write(f'{new_data_dir}/{filename}')
    rmtree(new_data_dir)
    return dialogues, ontology

def fix_entity_booked_info(entity_booked_dict, booked):
    for domain in entity_booked_dict:
        if not entity_booked_dict[domain] and booked[domain]:
            entity_booked_dict[domain] = True
            booked[domain] = []
    return entity_booked_dict, booked


if __name__ == '__main__':
    preprocess()