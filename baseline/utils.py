import os
import json
from tqdm import tqdm

min_count = 2

def get_kb(file_path="../ccks2019_el/kb_data"):
    id2kb = {}
    with open(file_path) as f:
        for line in tqdm(f):
            data = json.loads(line)
            subject_id = data['subject_id']
            subject_alias = list(set([data['subject']] + data.get('alias', [])))
            subject_alias = [alias.lower() for alias in subject_alias]
            subject_desc = '\n'.join('%s：%s' % (i['predicate'], i['object']) for i in data['data'])
            subject_desc = subject_desc.lower()
            if subject_desc:
                id2kb[subject_id] = {'subject_alias': subject_alias, 'subject_desc': subject_desc}

    kb2id = {}
    for i,j in id2kb.items():
        for k in j['subject_alias']:
            if k not in kb2id:
                kb2id[k] = []
            kb2id[k].append(i) 

    return id2kb, kb2id

def get_train_data(file_path='../ccks2019_el/train.json'):
    """
    读取训练数据，返回格式字典数组
    """
    train_data = []
    with open (file_path) as f:
        for line in tqdm(f):
            data = json.loads(line)
            train_data.append({
                'text': data['text'].lower(),
                'mention_data': [(x['mention'].lower(), int(x['offset']), x['kb_id'])
                    for x in data['mention_data'] if x['kb_id'] != 'NIL'
                ]
            })

    return train_data

def get_char_dict(id2kb, train_data):

    if not os.path.exists('../all_chars_me.json'):
        chars = {}
        for d in tqdm(iter(id2kb.values())):
            for c in d['subject_desc']:
                chars[c] = chars.get(c, 0) + 1
        for d in tqdm(iter(train_data)):
            for c in d['text']:
                chars[c] = chars.get(c, 0) + 1

        chars = {i : j for i, j in chars.items() if j >= min_count}

        id2char = {i + 2 : j for i, j in enumerate(chars)} # 0: mask, 1: padding
        char2id = {j : i for i, j in id2char.items()}
        json.dump([id2char, char2id], open('../all_chars_me.json', 'w'))
    else:
        id2char, char2id = json.load(open('../all_chars_me.json'))

    return id2char, char2id

def get_random(n):
    if not os.path.exists('../random_order_train.json'):
        random_order = range(len(train_data))
        np.random.shuffle(random_order)
        json.dump(
            random_order,
            open('../random_order_train.json', 'w'),
            indent=4
        )
    else:
        random_order = json.load(open('../random_order_train.json'))

    return random_order

class DataGenerator(object):
    def __init__(self, data, char2id, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.char2id = char2id
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = range(len(self.data))
            np.random.shuffle(idxs)
            sentence_in, 


