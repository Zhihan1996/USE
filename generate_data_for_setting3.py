
import random
import pickle
from utils import *
import collections
from typing import List
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from numpy import dot
from numpy.linalg import norm
from process_data import read_sequences, generate_subsequence



max_files = 2000
prefix="data/user_event_and_time_sequences_test"

# event_sequences, time_sequences, ghost_user_ids = read_sequences(max_files=max_files,
#                                                                  length_to_keep=-1,
#                                                                  use_time=False,
#                                                                  bucket_name="umap-user-model",
#                                                                  prefix=prefix)
with open("data.pkl", "rb") as f:
    event_sequences, time_sequences, ghost_user_ids = pickle.load(f)

tokenizer = BigBirdTokenizer.from_pretrained("/home/jupyter/research/research/experimental/bic/user_states/artifacts/tokenizers/bigbird_word")
SPECIAL_TOKEN_IDS = [
    k
    for k in tokenizer.special_tokens_map.values()
]


seq_len = 4000+4000+500
min_len = 4000+4000+500
n_events = 686
seqs = []
for e in event_sequences:
    seqs.extend(generate_subsequence(e.split(), seq_len, min_len, time=False, max_per_user=1))

seqs_new = []
for s in seqs:
    if len(set(s.split())) > 40:
        seqs_new.append(s)
        
random.shuffle(seqs_new)
seqs = seqs_new[:30000]


##### user re-id
reid_users = seqs[:5000]
data_reid = []
for u in reid_users:
    u = u.split()
    assert len(u) > 100 + 4096*2
    history = " ".join(u[100:100+4096])
    context = " ".join(u[-4096:])
    data_reid.append((history, context))
    
with open("../test/final_user_reid.pkl", "wb") as f:
    pickle.dump(data_reid, f)


##### fep


seqs_fep = []
for s in seqs:
    start = 4200
    seqs_fep.append(s.split()[start:start+250+4000])
e_count = collections.defaultdict(int)
for seq in seqs_fep:
    unique_events = set(seq[-4000:])
    for e in unique_events:
        e_count[e] += 1

        
min_appearance = 10000
max_appearance = 20000
selected_events = [e for e in e_count if max_appearance > e_count[e] > min_appearance]
tokenizer_ids = [tokenizer.convert_tokens_to_ids("▁"+e) for e in selected_events if e not in SPECIAL_TOKEN_IDS]
e2id = {e: i for i, e in enumerate(selected_events) if e not in SPECIAL_TOKEN_IDS}


# add seq to train
seqs_to_train = [s.split()[:250+4000] for s in seqs]
data_to_train = []

label_freq = 0
for u in seqs_to_train:
    seq_input = u[:4000]
    seq_label = u[4000:]
    label_train = np.zeros([len(selected_events)])
    for e in set(seq_label):
        if e in e2id:
            label_train[e2id[e]] = 1
    label_freq += label_train.sum() / label_train.shape[0] 
    data_to_train.append((" ".join(seq_input), label_train))
label_freq /= len(data_to_train)
print(label_freq)


data_fep = []
label_freq = 0
for idx, u in enumerate(seqs_fep):
    if idx % 100 == 0:
        print(idx)
    
    seq_input = u[:4000]
    # labels = np.zeros([16, len(selected_events)])
    # for i in range(16):
    #     seq_label = set(u[250*(i+1):250*(i+2)])
    #     for e in e2id:
    #         if e in seq_label:
    #             labels[i][e2id[e]] = 1
    labels = np.zeros([4000, len(selected_events)])
    counter = collections.Counter(u[:250])
    for i in range(4000):
        counter[u[i]] -= 1
        counter[u[i + 250]] += 1
        if counter[u[i]] == 0:
            counter.pop(u[i])
        for e in e2id:
            if e in counter:
                labels[i][e2id[e]] = 1
    label_freq += labels.sum() / (labels.shape[0] * labels.shape[1])
    data_fep.append((" ".join(seq_input), labels))
label_freq /= len(seqs_fep)
print(f"Number of data: {len(data_fep)}, label frequency: {label_freq}")

data_fep = [tokenizer_ids] + data_fep

# with open(f"../test/final_fep_250_{min_appearance}_{max_appearance}.pkl", "wb") as f:
    # pickle.dump(data_fep, f)
with open(f"../test/final_fep_{min_appearance}_{max_appearance}_1008_token.pkl", "wb") as f:
    pickle.dump((data_to_train, data_fep), f)