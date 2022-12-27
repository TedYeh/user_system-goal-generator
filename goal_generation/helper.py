from ckiptagger import WS
import random 

ws = WS("./tagger")

def get_query_text(text):    
    ws_results = ws([text])[0]
    s = random.randint(0, len(ws_results))
    e = random.randint(s, len(ws_results)+1)
    return ''.join(ws_results[s:e])