import os
import json
from tqdm import tqdm
import re
from pprint import pprint
from transformers import AutoTokenizer
from convlab.util import load_dataset, load_nlu_data, load_dst_data, load_policy_data, load_nlg_data, load_e2e_data, load_rg_data, retrieve_utterances
from convlab.base_models.t5.nlu.serialization import serialize_dialogue_acts, deserialize_dialogue_acts, equal_da_seq, serialize_dialogue_acts_non_value
from convlab.base_models.t5.dst.serialization import serialize_dialogue_state, deserialize_dialogue_state, equal_state_seq

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

def create_rg_data(dataset, data_dir, args):
    data_by_split = load_rg_data(dataset, speaker=args.speaker)
    data_dir = os.path.join(data_dir, args.speaker)
    os.makedirs(data_dir, exist_ok=True)

    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        for sample in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            if len(sample['context']) == 0:
                continue
            context = '\n'.join([f"{turn['speaker']}: {turn['utterance']}" for turn in sample['context']]+[f'{sample["speaker"]}: '])
            data.append(json.dumps({'context': context, 'response': sample['utterance']}, ensure_ascii=False)+'\n')

        file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
        data_by_split[data_split] = data
    return data_by_split

def create_nlu_data(dataset, data_dir, args):
    data_by_split = load_nlu_data(dataset, speaker=args.speaker, use_context=args.context_window_size>0, context_window_size=args.context_window_size)
    data_dir = os.path.join(data_dir, args.speaker, f'context_{args.context_window_size}')
    os.makedirs(data_dir, exist_ok=True)

    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        for sample in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            response = f"{sample['speaker']}: {sample['utterance']}"
            if args.context_window_size>0:
                context = '\n'.join([f"{turn['speaker']}: {turn['utterance']}" for turn in sample['context']]+[response])
            else:
                context = response
            dialogue_acts_seq = serialize_dialogue_acts(sample['dialogue_acts'])
            assert equal_da_seq(sample['dialogue_acts'], dialogue_acts_seq), print(sample['dialogue_acts'], dialogue_acts_seq, deserialize_dialogue_acts(dialogue_acts_seq))
            #data.append(json.dumps({'context': "對話理解: 依據對話歷史預測對話行為。\n\n 輸入: [HISTORY] "+context+" [QUESTION] 對話行為:", 'dialogue_acts_seq': dialogue_acts_seq}, ensure_ascii=False)+'\n')
            data.append(json.dumps({'context': context, 'dialogue_acts_seq': dialogue_acts_seq}, ensure_ascii=False)+'\n')

        file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
        data_by_split[data_split] = data
    return data_by_split

def create_dst_data(dataset, data_dir, args):
    data_by_split = load_dst_data(dataset, speaker=args.speaker, use_context=args.context_window_size>0, context_window_size=args.context_window_size)
    data_dir = os.path.join(data_dir, args.speaker, f'context_{args.context_window_size}')
    os.makedirs(data_dir, exist_ok=True)
    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        for sample in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            
            response = f"{sample['speaker']}: {sample['utterance']}"
            if args.context_window_size>0:
                context = '\n'.join([f"{turn['speaker']}: {turn['utterance']}" for turn in sample['context']]+[response])
            else:
                context = response
            for domain in sample['state']:
                for slot in sample['state'][domain]:                    
                    if slot == "活動時間":
                        sample['state'][domain][slot] = time_to_str(sample['state'][domain][slot])
                    if isinstance(sample['state'][domain], list): continue
                    #print(sample['state'][domain], slot)
                    vs = sample['state'][domain][slot].split('|')
                    # only the first variation of value
                    sample['state'][domain][slot] = vs[0]
            #del sample['state']['selectedResults']
            state_seq = serialize_dialogue_state(sample['state'])
            dialogue_acts_seq = serialize_dialogue_acts(sample['dialogue_acts'])
            assert equal_state_seq(sample['state'], state_seq), print(sample['state'], state_seq, deserialize_dialogue_state(state_seq))
            data.append(json.dumps({'context': f"對話狀態追蹤: 依據當前對話行為和對話歷史預測對話狀態。\n\n 輸入: [ACTION] {dialogue_acts_seq} [HISTORY] {context} [QUESTION] 對話狀態:", 'state_seq': state_seq}, ensure_ascii=False)+'\n')
            #data.append(json.dumps({'context': context, 'state_seq': state_seq}, ensure_ascii=False)+'\n')

        file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
        data_by_split[data_split] = data
    return data_by_split

def create_policy_data(dataset, data_dir, args):
    # Define the input and output format for dialogue policy learning
    
    #input_format = "對話決策: 依據對話狀態、對話歷史和資料庫結果來決定系統的對話行為。\n\n Input: [STATE] {dialogue_state} [HISTORY] {context} [DATABASE] {database_results} [QUESTION] 系統對話行為:"
    #input_format = "對話決策: 依據對話狀態、對話歷史和資料庫結果來決定系統的對話行為。\n\n 使用者行為: [ACTION] {dialogue_action} 對話狀態: [STATE] {dialogue_state} 對話歷史: [HISTORY] {context} 資料庫結果: [DATABASE] {database_results} [QUESTION] 系統對話行為:"
    input_format = "依據使用者對話行為、對話狀態和資料庫結果來決定系統的對話行為: \n\n 使用者行為: [ACTION] {dialogue_action} 對話狀態: [STATE] {dialogue_state} 資料庫結果: [DATABASE] {database_results} [QUESTION] 系統對話行為: "
    #input_format = "依據對話狀態、對話歷史和資料庫結果來決定系統的對話行為:\n\n對話狀態:{dialogue_state}\n\n對話歷史:{context}\n\n資料庫結果:{database_results}\n\n系統對話行為: "
    #input_format = "{dialogue_state}\n\n{context}\n\n{database_results}\n\nSystem Action: "
    output_format = "{system_action}"
    data_by_split = load_policy_data(dataset, speaker=args.speaker, use_context=args.context_window_size>0, context_window_size=args.context_window_size)
    data_dir = os.path.join(data_dir, args.speaker, f'context_{args.context_window_size}')
    os.makedirs(data_dir, exist_ok=True)    
    
    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        #print(data_split)
        #print(data_by_split[data_split][0])
        #input()
        for sample in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            response = f"{sample['speaker']}: {sample['utterance']}"
            if args.context_window_size>0:
                context = '\n'.join([f"{turn['speaker']}: {turn['utterance']}" for turn in sample['context']])
            else:
                context = response
            if len(serialize_dialogue_acts(sample['dialogue_acts'])) == 0:
                # skip empty dialogue acts
                continue
            
            if sample['speaker'] == 'user':
                user_action_text = serialize_dialogue_acts(sample['dialogue_acts'])
                #user_action_text = serialize_dialogue_acts_non_value(sample['dialogue_acts'])
                #print(sample['state'])
                dialogue_state_text = serialize_dialogue_state(sample['state'])
            else:            
                system_action_text = serialize_dialogue_acts(sample['dialogue_acts']) 
                #dialogue_state_text = serialize_dialogue_state(sample['state'])   
                # Convert the database results into text string
                database_results = sample["db_results"]
                database_results_text = ""
                if database_results and "selectedResults" in database_results:
                    for result in database_results["selectedResults"]:
                        for slot, value in result.items():
                            if slot == "活動時間":value = time_to_str(value)
                            database_results_text += f"[{slot}][{value[:64]}];"
                elif database_results:
                    for domain, results in database_results.items():
                        slot_values = []
                        for result in results:
                            for slot, value in result.items():
                                slot_values.append(f"[{slot}][{value[:64]}]")
                    database_results_text += f"[{domain}]({','.join(slot_values)});"
                else: database_results_text = f"[None]([None][None]);"
                # Format the input and output strings
                input_string = input_format.format(dialogue_action=user_action_text,
                                                dialogue_state=dialogue_state_text,
                                                context=context,
                                                database_results=database_results_text)
                output_string = output_format.format(system_action=system_action_text)

                data.append(json.dumps({'state+da+db': input_string, 'dialogue_acts_seq': output_string}, ensure_ascii=False)+'\n')
                #print({'state+da+db': input_string, 'dialogue_acts_seq': output_string})
        
        file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
        data_by_split[data_split] = data
    return data_by_split

def create_nlg_data(dataset, data_dir, args):
    data_by_split = load_nlu_data(dataset, speaker=args.speaker, use_context=args.context_window_size>0, context_window_size=args.context_window_size)
    data_dir = os.path.join(data_dir, args.speaker, f'context_{args.context_window_size}')
    os.makedirs(data_dir, exist_ok=True)

    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        for sample in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            dialogue_acts_seq = serialize_dialogue_acts(sample['dialogue_acts'])
            if len(dialogue_acts_seq) == 0:
                # skip empty dialogue acts
                continue
            if args.context_window_size>0:
                context = '\n'.join([f"{turn['speaker']}: {turn['utterance']}" for turn in sample['context']]+[f'{sample["speaker"]}: '])
                context = f'{dialogue_acts_seq}\n\n{context}'
                #context = '\n'.join([f"{turn['speaker']}: {turn['utterance']}" for turn in sample['context']]+[f' [QUESTION] {sample["speaker"]}:'])
                #context = f'生成依據對話行為及對話歷史來產生回覆: \n\n輸入: 對話行為: [ACTION] {dialogue_acts_seq} 對話歷史: [HISTORY] {context}'
            else:
                context = f'{dialogue_acts_seq}\n\n{sample["speaker"]}: '
            assert equal_da_seq(sample['dialogue_acts'], dialogue_acts_seq), print(sample['dialogue_acts'], dialogue_acts_seq, deserialize_dialogue_acts(dialogue_acts_seq))
            data.append(json.dumps({'context+da': context, 'response': sample['utterance']}, ensure_ascii=False)+'\n')

        file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
        data_by_split[data_split] = data
    return data_by_split

def create_goal2dialogue_data(dataset, data_dir, args):
    data_by_split = dataset
    os.makedirs(data_dir, exist_ok=True)

    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        for sample in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            goal = re.sub(r'<.*?>', '', sample['goal']['description'])
            dialogue = '\n'.join([f"{turn['speaker']}: {turn['utterance']}" for turn in sample['turns']])
            data.append(json.dumps({'goal': goal, 'dialogue': dialogue}, ensure_ascii=False)+'\n')

        file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
        data_by_split[data_split] = data
    return data_by_split

def create_retnlu_data(dataset, data_dir, args):
    dataset_name = dataset[list(dataset.keys())[0]][0]['dataset']
    data_by_split = load_nlu_data(dataset, speaker=args.speaker, use_context=args.context_window_size>0, context_window_size=args.context_window_size)
    data_dir = os.path.join(data_dir, args.speaker, f'context_{args.context_window_size}', \
        f'in_context_{args.retrieval_in_context}', f'topk_{args.retrieval_topk}')
    os.makedirs(data_dir, exist_ok=True)

    turn_pool = []
    for d in args.retrieval_datasets:
        pool_dataset = load_dataset(d)
        for turn in load_nlu_data(pool_dataset, data_split='train', speaker=args.speaker)['train']:
            if any([len(das) > 0 for da_type, das in turn['dialogue_acts'].items()]):
                turn_pool.append({'dataset': d, **turn})

    data_splits = data_by_split.keys()
    query_turns = []
    for data_split in data_splits:
        query_turns.extend(data_by_split[data_split])
    augmented_dataset = retrieve_utterances(query_turns, turn_pool, args.retrieval_topk, 'all-MiniLM-L6-v2')

    i = 0
    for data_split in data_splits:
        data = []
        for j in tqdm(range(len(data_by_split[data_split])), desc=f'{data_split} sample', leave=False):
            sample = augmented_dataset[i+j]
            response = f"{sample['speaker']}: {sample['utterance']}"
            if args.context_window_size>0:
                context = '\n'.join([f"{turn['speaker']}: {turn['utterance']}" for turn in sample['context']]+[response])
            else:
                context = response
            context = ' '.join([dataset_name, context])
            dialogue_acts_seq = serialize_dialogue_acts(sample['dialogue_acts'])
            assert equal_da_seq(sample['dialogue_acts'], dialogue_acts_seq), print(sample['dialogue_acts'], dialogue_acts_seq, deserialize_dialogue_acts(dialogue_acts_seq))

            retrieved_turns = sample['retrieved_turns']
            for t in retrieved_turns:
                # in-context learning
                retrieved_utterance = f"{t['dataset']} {t['speaker']}: {t['utterance']}"
                retrieved_dialogue_acts_seq = serialize_dialogue_acts(t['dialogue_acts'])
                if args.retrieval_in_context:
                    context = f"{retrieved_utterance} => {retrieved_dialogue_acts_seq}\n\n" + context
                elif data_split != 'test':
                    data.append(json.dumps({'context': retrieved_utterance, 'dialogue_acts_seq': retrieved_dialogue_acts_seq}, ensure_ascii=False)+'\n')        

            data.append(json.dumps({'context': context, 'dialogue_acts_seq': dialogue_acts_seq}, ensure_ascii=False)+'\n')
        i += len(data_by_split[data_split])

        file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
        data_by_split[data_split] = data
    return data_by_split

def get_max_len(data_by_split, tokenizer):
    for data_split in data_by_split.keys():
        seq_len = {}
        for line in data_by_split[data_split]:
            item = json.loads(line.strip())
            for column, seq in item.items():
                seq_len.setdefault(column, [])
                seq_len[column].append(len(tokenizer.tokenize(seq)))
        print(f"data split: {data_split}")
        for column, lens in seq_len.items():
            print(f'\t{column}\tmax_len: {max(lens)}\tmean_len: {round(sum(lens)/len(lens),2)}')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="create data for seq2seq training")
    parser.add_argument('--tasks', '-t', metavar='task_name', nargs='*', choices=['rg', 'nlu', 'dst', 'nlg', 'goal2dialogue', 'retnlu', 'retnlg', 'policy'], help='names of tasks')
    parser.add_argument('--datasets', '-d', metavar='dataset_name', nargs='*', help='names of unified datasets')
    parser.add_argument('--speaker', '-s', type=str, choices=['user', 'system', 'all'], help='speaker(s)')
    parser.add_argument('--context_window_size', '-c', type=int, default=100, help='how many contextual utterances are considered')
    parser.add_argument('--len_tokenizer', '-l', type=str, default=None, help='name or path of tokenizer that used to get seq len')
    parser.add_argument('--ratio', '-r', type=float, default=None, help='how many data is used for training and evaluation')
    parser.add_argument('--dial_ids_order', '-o', type=int, default=None, help='which data order is used for experiments')
    parser.add_argument('--retrieval_datasets', metavar='dataset_name for retrieval augmentation', nargs='*', help='names of unified datasets for retrieval')
    parser.add_argument('--retrieval_topk', type=int, default=3, help='how many utterances to be retrieved')
    parser.add_argument('--retrieval_in_context', action='store_true', default=False, help='whether use the retrieved utterance by in-context learning')
    args = parser.parse_args()
    print(args)
    if args.len_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.len_tokenizer)
    for dataset_name in tqdm(args.datasets, desc='datasets'):
        if args.ratio:
            dataset = load_dataset(dataset_name, dial_ids_order=args.dial_ids_order, split2ratio={'train': args.ratio, 'validation': args.ratio})
        else:
            dataset = load_dataset(dataset_name, args.dial_ids_order)
        for task_name in tqdm(args.tasks, desc='tasks', leave=False):
            data_dir = os.path.join('data', task_name, (dataset_name if not args.ratio else f'{dataset_name}_{args.ratio}_order{args.dial_ids_order}'))
            data_by_split = eval(f"create_{task_name}_data")(dataset, data_dir, args)
            if args.len_tokenizer:
                get_max_len(data_by_split, tokenizer)
