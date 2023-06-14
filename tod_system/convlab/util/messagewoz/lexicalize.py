def delexicalize_da(da):
    delexicalized_da = []
    counter = {}
    for intent, domain, slot, value in da:
        if intent in ['Inform', 'Recommend']:
            key = '+'.join([intent, domain, slot])
            counter.setdefault(key,0)
            counter[key] += 1
            delexicalized_da.append(key+'+'+str(counter[key]))
        else:
            delexicalized_da.append('+'.join([intent, domain, slot, value]))
    
    return delexicalized_da


def lexicalize_da(da, cur_domain, entities):
    lexicalized_da = []
    print('da',da)
    for a in da:
        intent, domain, slot, value = a.split('+')
        if intent in ['General', 'NoFound']:
            lexicalized_da.append([intent, domain, slot, value])
        else:
            if entities:
                entity = entities[0][1]
                value = entity[slot]
            print(lexicalized_da)
            lexicalized_da.append([intent, domain, slot, value])
    return lexicalized_da

def flat_da(delexicalized_da):
    flaten = ['_'.join(x) for x in delexicalized_da]
    return flaten


def deflat_da(meta):
    meta = deepcopy(meta)
    dialog_act = {}
    for da in meta:
        d, i, s, v = da
        k = (d, i)
        if k not in dialog_act:
            dialog_act[k] = []
        dialog_act[k].append([s, v])
    return dialog_act