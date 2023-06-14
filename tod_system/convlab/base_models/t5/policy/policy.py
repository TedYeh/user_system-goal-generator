import logging
import torch, os, json
from pprint import pprint
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from convlab.policy.policy import Policy
from convlab.base_models.t5.nlu.serialization import serialize_dialogue_acts, deserialize_dialogue_acts
from convlab.base_models.t5.dst.serialization import serialize_dialogue_state

def analysis_schema(schema_file):
    schemas = json.loads(open(schema_file, "r", encoding="utf-8").read())
    values = []
    for schema in schemas:
        for intent in schema["intents"]:
            s = list(intent.values())
            values.append([s[0], s[2], s[4]])
    return values

def is_transactional(value_list, intent):
    for value in value_list:
        if intent in value:
            if value[1]: return True
            else: return False

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

class T5Policy(Policy):
    def __init__(self, speaker, context_window_size, model_name_or_path, db_path, device='cuda'):
        self.speaker = speaker
        self.opponent = 'user' if speaker == 'system' else 'system'
        self.context_window_size = context_window_size
        self.use_context = context_window_size > 0
        self.result_window = 5
        self.n = 0
        self.value_list = analysis_schema('schema.json')
        self.db_path = db_path
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=self.config)
        self.model.eval()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        logging.info("T5Policy loaded")

    def predict(self, state):
        context_list = list(state['history'][:])
        if self.use_context:
            if len(context_list) > 0 and type(context_list[0]) is list and len(context_list[0]) > 1:
                context_list = [item[1] for item in context_list]
                print(context_list)
            context_list = context_list[-self.context_window_size:]
            utts = context_list
            print('utts', utts)
        else:
            utts = ['']
        context = '\n'.join([f"{self.speaker if (i % 2) == (len(utts) % 2) else self.opponent}: {utt}" for i, utt in enumerate(utts)])
        print('history', context, context_list, state['history'])
        # Define the input and output format for dialogue policy learning
        input_format = "對話決策: 依據對話狀態、對話歷史和資料庫結果來決定系統的對話行為。\n\n Input: [STATE] {dialogue_state} [HISTORY] {context} [DATABASE] {database_results} [QUESTION] 系統對話行為:"
        #input_format = "{dialogue_state}\n\n{context}\n\n{database_results}\n\nSystem Action: "
        dialogue_acts = state['user_action']
        if isinstance(dialogue_acts, dict):
            # da in unified format
            dialogue_acts_seq = serialize_dialogue_acts(dialogue_acts)
        elif isinstance(dialogue_acts[0], dict):
            # da without da type
            dialogue_acts_seq = serialize_dialogue_acts({'categorical': dialogue_acts})
        elif isinstance(dialogue_acts[0], list):
            # da is a list of list (convlab-2 format)
            dialogue_acts_seq = serialize_dialogue_acts(
                    {'categorical': [{'intent': da[0], 'domain': da[1], 'slot': da[2], 'value': da[3]} for da in dialogue_acts]})
        else:
            raise ValueError(f"invalid dialog acts format {dialogue_acts}")

        dialogue_state_text = serialize_dialogue_state(state['belief_state'])

        database_results_text = ""
        domain, database_results = self.db_query(state['belief_state'])
        if database_results and not is_transactional(self.value_list, list(state['belief_state'].values())[0]['intent']):
            for db_result in database_results[self.n * self.result_window:(self.n+1) * self.result_window]:
                slot_values = []
                for slot, results in db_result.items():  
                    slot_values.append(f"[{slot}][{results[:64]}]")
                database_results_text += f"[{domain}]({','.join(slot_values)});"
        else: database_results_text = f"[None]([None][None]);"
        print(database_results_text)
        self.n += 1
        # Format the input and output strings
        input_string = input_format.format(dialogue_state=dialogue_state_text,
                                       context=context,
                                       database_results=database_results_text)
        
        input_seq = self.tokenizer(input_string, return_tensors="pt").to(self.device)
        # print(input_seq)
        output_seq = self.model.generate(**input_seq, max_length=512)
        # print(output_seq)
        output_seq = self.tokenizer.decode(output_seq[0], skip_special_tokens=True)
        print(output_seq)
        das = deserialize_dialogue_acts(output_seq.strip())
        dialog_act = []
        for da in das:
            dialog_act.append([da['intent'], da['domain'], da['slot'], da.get('value','')])
        return dialog_act
    
    def db_query(self, state):        
        db_mapping = {'Calendar_1':'events', 'Messaging_1':'message', 'Mail_1':'mail'}
        db_list = os.listdir(self.db_path)
        db_res = []
        for slot, value in list(state.values())[0].items():            
            if f"{db_mapping[list(state.keys())[0]]}.json" in db_list:
                db = json.loads(open(os.path.join(self.db_path, f"{db_mapping[list(state.keys())[0]]}.json")).read())
                for field in db:                    
                    if slot in list(field.keys()) and value in list(field.values()):
                       db_res.append(field)
        
        return list(state.keys())[0], db_res

if __name__ == '__main__':
    state = {
        'belief_state': {}, 
        'cur_domain': '', 
        'history': [['usr', '我要寄一封信給詩詩，副本收件者是兮兮']], 
        'system_action': [], 
        'user_action': [['Inform', 'Gmail', '副本收件者', '兮兮'], ['Inform', 'Gmail', '收件者', '詩詩']], 
        'terminated': False
    }
    policy = T5Policy(speaker='system', context_window_size=1, db_path='./database', model_name_or_path='output/policy/messagewoz/all/context_1')
    sys_act = policy.predict(state)
    print(sys_act)
