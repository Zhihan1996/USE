# User retrieval

import random
import pickle
from utils import *
import collections
from typing import List
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from numpy import dot
from numpy.linalg import norm

def generate_subsequence(ori: List, 
                         seq_len: int = 512,
                         min_len: int = 128,
                         time: bool = False,
                         max_per_user: int = -1):
    sequences = []
    for i in range(len(ori) // seq_len):
        seq = ori[i : i+seq_len]
        if not time:
            seq = " ".join(seq)
        if len(seq) >= min_len:
            sequences.append(seq)
            if max_per_user > 0 and len(sequences) == max_per_user:
                break
    return sequences

def read_sequences(
    bucket_name: str = "bic-test",
    prefix: str = "data/user_event_and_time_sequences",
    task: str = "train",
    sequence_type: str = "shortlist",
    sample_n: int = 1000,
    features_start_date: str = "20230401",
    features_end_date: str = "20230414",
    labels_date: str = "20230414",
    sample_table_option: str = "sample",
    user_filter_option: str = "unfiltered",
    sc_analytics_tables: str = "all",
    max_files: int = 10000,
    length_to_keep: int = 512,
    use_time: bool = False,
):
    """
    This function reads user sequences stored on GCS bucket.
    You can specify whether you want only event sequences or both event and time sequences.
    """
    assert task in get_available_tasks(scope="all")

    labels_date = features_end_date if task == "train" else labels_date
    prefix = f"{prefix}"

    data = read_avro_file_from_gcs(
        bucket_name=bucket_name,
        prefix=prefix,
        max_files=max_files,
        length_to_keep=length_to_keep,
    )

    assert len(data) > 0

    event_sequences = sorted([item["event_sequence"] for item in data])
    time_sequences = sorted([item["time_sequence"] for item in data])

    assert len(event_sequences) == len(time_sequences)

    ghost_user_ids = [event_sequence[0] for event_sequence in event_sequences]

    time_parser = TimeParser() if use_time else None

    processed_event_sequences = []
    processed_time_sequences = []

    for event_sequence, time_sequence in zip(event_sequences, time_sequences):
        processed_event_sequence, processed_time_sequence = preprocess_user_sequence(
            event_sequence, time_sequence, sequence_type, time_parser, True
        )
        processed_event_sequences.append(processed_event_sequence)
        processed_time_sequences.append(processed_time_sequence)

    return processed_event_sequences, processed_time_sequences, ghost_user_ids

if __name__ == "__main__":
    max_files = 1000
    # seq_len = 1024
    # min_len = 1024

    pred_len = 1000
    seq_len = 4096+1024
    min_len = 4096+1024
    task = "future_event"
    # prefix="data/user_event_and_time_sequences_train_in_all_1000000_from_20230401"
    prefix = "data/user_event_and_time_sequences_test"

    event_sequences, time_sequences, ghost_user_ids = read_sequences(max_files=max_files,
                                                                     length_to_keep=-1,
                                                                     use_time=True,
                                                                     bucket_name="umap-user-model",
                                                                     prefix=prefix)


    # with open("../data/sequences_test_5000.pkl", "rb") as f:
    #     event_sequences, time_sequences = pickle.load(f)
    #     ghost_user_ids = [i for i in range(len(event_sequences))]

    # import pickle

    # with open("/home/jupyter/research/research/experimental/bic/user_states/dataset/event_sequences_2k.pkl", "wb") as f:
    #     pickle.dump(event_sequences, f)
    # with open("/home/jupyter/research/research/experimental/bic/user_states/dataset/time_sequences_2k.pkl", "wb") as f:
    #     pickle.dump(time_sequences, f)
    # with open("/home/jupyter/research/research/experimental/bic/user_states/dataset/ghost_user_ids_2k.pkl", "wb") as f:
    #     pickle.dump(ghost_user_ids, f)




    # import pickle
    # with open("/home/jupyter/research/research/experimental/bic/user_states/dataset/event_sequences_2k.pkl", "rb") as f:
    #     event_sequences = pickle.load(f)
    # with open("/home/jupyter/research/research/experimental/bic/user_states/dataset/time_sequences_2k.pkl", "rb") as f:
    #     time_sequences = pickle.load(f)
    # with open("/home/jupyter/research/research/experimental/bic/user_states/dataset/ghost_user_ids_2k.pkl", "rb") as f:
    #     ghost_user_ids = pickle.load(f)




    if task == "future_event":
        n_events = 686
        ### count event frequency
        seqs = []
        for e in event_sequences:
            # seqs.extend(generate_subsequence(e.split(), seq_len+pred_len, min_len+pred_len, time=False))
            seqs.extend(generate_subsequence(e.split(), seq_len+pred_len, min_len+pred_len, time=False, max_per_user=1))

        tokenizer = BigBirdTokenizer.from_pretrained("/home/jupyter/research/research/experimental/bic/user_states/artifacts/tokenizers/bigbird_word")
        SPECIAL_TOKEN_IDS = [
            tokenizer.convert_tokens_to_ids(k)
            for k in tokenizer.special_tokens_map.values()
        ]

        with open(f"event_of_interests_{n_events}.txt", "r") as f:
            event_lists = f.read().splitlines()
            # events_of_interests = [e.split(",")[0] for e in event_lists]
            selected_events = [e.split(",")[1].strip().strip("▁") for e in event_lists]
            tokenizer_ids = [int(e.split(",")[0]) for e in event_lists]
            e2id = {e: i for i, e in enumerate(selected_events) if int(e) not in SPECIAL_TOKEN_IDS}

    #     count = collections.defaultdict(int)
    #     count_cat = collections.defaultdict(int)
    #     for s in seqs:
    #         s = s.split()[-pred_len:]
    #         cat = [a.split("_")[0] for a in s]
    #         events = set(s)
    #         cat = set(cat)
    #         for w in events:
    #             count[w] += 1
    #         for w in cat:
    #             count_cat[w] += 1
    #     for c in count:
    #         count[c] /= len(seqs)
    #     for c in count_cat:
    #         count_cat[c] /= len(seqs)
    #     count = {k: v for k, v in sorted(count.items(), key=lambda item: item[1])}
    #     count_cat = {k: v for k, v in sorted(count_cat.items(), key=lambda item: item[1])}

    #     selected_events = []
    #     for c in count:
    #         if 0.001 < count[c] < 1:
    #             selected_events.append(c)
    #     print(selected_events)
    #     print(len(selected_events))
    #     e2id = {e: i for i, e in enumerate(selected_events)}




        data = []
        # use the first position to store selected events and its order
        label_freq = 0
        for seq in seqs:
            seq = seq.split()
            label = [0 for _ in selected_events]
            seq_input, seq_label = seq[:-pred_len], seq[-pred_len:]
            for e in set(seq_label):
                if e in e2id:
                    label[e2id[e]] = 1
            data.append((" ".join(seq_input), label))
            label_freq += sum(label) / len(label)
        label_freq /= len(data)
        print(f"Number of data: {len(data)}, label frequency: {label_freq}")

        random.shuffle(data)

        data = [tokenizer_ids] + data

        import pickle
        with open(f"/home/jupyter/research/research/experimental/bic/user_states/test/future_first128_100_20230917_5ktest_{n_events}.pkl", "wb") as f:
            pickle.dump(data, f)



        # use pre-train data
    #     from datasets import Dataset, DatasetDict, load_from_disk
    #     train_data = load_from_disk("/home/jupyter/research/research/experimental/bic/user_states/data/data/user_event_and_time_sequences_train_in_all_1000000_from_20230401_to_20230414_on_20230414_sample_filtered_5000_128_128_False_True_True_event_of_interests_29")
    #     tokenizer = BigBirdTokenizer.from_pretrained("/home/jupyter/research/research/experimental/bic/user_states/artifacts/tokenizers/bigbird_word")

    #     data = []
    #     for i, d in enumerate(train_data):
    #         if i == 25000:
    #             break
    #         text = tokenizer.batch_decode(d["input_ids"])[0][6:]
    #         label = d["fep_labels"][0][-1]
    #         data.append((text, label))

    #     random.shuffle(data)

    #     data = [tokenizer_ids] + data

    #     import pickle
    #     with open(f"/home/jupyter/research/research/experimental/bic/user_states/test/future_128_100_20230914_pretrain_{n_events}.pkl", "wb") as f:
    #         pickle.dump(data, f)
    #     with open(f"/home/jupyter/research/research/experimental/bic/user_states/test/future_128_100_20230914_pretrain_25k_{n_events}.pkl", "wb") as f:
    #         pickle.dump(data[:25001], f)







    elif task == "clm":
        user2e = collections.defaultdict(list)
        user2t = collections.defaultdict(list)

        for i, ghost_id in enumerate(ghost_user_ids):
            e_seq = generate_subsequence(event_sequences[i].split(), seq_len, min_len, time=False)
            t_seq = generate_subsequence(time_sequences[i], seq_len, min_len, time=True)
            if len(e_seq) > 1:
                user2e[ghost_id].extend(e_seq)
                user2t[ghost_id].extend(t_seq)

        sample_per_user = 2
        data = []
        for u in user2e:
            seqs = user2e[u]
            random.shuffle(seqs)
            seqs = seqs[:sample_per_user]
            data.extend(seqs)

        random.shuffle(data)

        # sample format ([seq, pos, neg1, neg2, .., neg99])    
        import pickle
        with open(f"/home/jupyter/research/research/experimental/bic/user_states/test/clm_1024_128_128_20230820_large.pkl", "wb") as f:
            pickle.dump(data, f)


    elif task == "user_retrieval":
        user2e = collections.defaultdict(list)
        user2t = collections.defaultdict(list)

        for i, ghost_id in enumerate(ghost_user_ids):
            e_seq = generate_subsequence(event_sequences[i].split(), seq_len, min_len, time=False)
            t_seq = generate_subsequence(time_sequences[i], seq_len, min_len, time=True)
            if len(e_seq) > 1:
                user2e[ghost_id].extend(e_seq)
                user2t[ghost_id].extend(t_seq)


        def cosine_sim(a, b):
            return dot(a, b)/(norm(a)*norm(b))


        vectorizer = CountVectorizer()
        vectorizer.fit(event_sequences[:1000])

        n_user = len(user2e)
        # n_user = 100
        sample_per_user = 1

        data = []
        selected_users = random.sample(list(user2e.keys()), n_user)
        random.shuffle(selected_users)

        for idx, u in enumerate(selected_users):
            if idx % 10 == 0:
                print(f"{idx} : {len(data)}")
            user_e = user2e[u]
            if len(user_e) < 2:
                continue
            n = min(len(user_e), sample_per_user)
            sampled_e = random.sample(user_e, n)

            negative_users = list(user2e.keys())
            random.shuffle(negative_users)

            if len(data) >= 3000:
                break

            for i in range(n):
                cur_sample = sampled_e[i]
                cur_vec = vectorizer.transform([cur_sample]).toarray()[0]

                # positive sample
                pos = random.choice(list(range(i)) + list(range(i+1, len(user_e))))
                pos_sample = user_e[pos]
                pos_vec = vectorizer.transform([pos_sample]).toarray()[0]

                pos_sim = cosine_sim(cur_vec, pos_vec)
                # if pos_sim > 0.95:
                #     continue

                # negative samples
                neg_idx = random.randint(0, 30)
                neg_samples = []
                count = 0
                for neg in negative_users:
                    idx = neg_idx % len(user2e[neg])
                    neg_sample = user2e[neg][idx]
                    neg_vec = vectorizer.transform([neg_sample]).toarray()[0]
                    # print(cosine_sim(cur_vec, neg_vec))

                    if cosine_sim(cur_vec, neg_vec) > 0.8:
                        neg_samples.append(neg_sample)
                        count += 1
                        if count == 99:
                            break

                if len(neg_samples) < 99:
                    continue

                data.append([cur_sample, pos_sample] + neg_samples)


        random.shuffle(data)
        print(f"Number of data {len(data)}")


        # sample format ([seq, pos, neg1, neg2, .., neg99])    
        import pickle
        with open(f"/home/jupyter/research/research/experimental/bic/user_states/test/final_user_retrieval_4608_20230921_neg080_pos000_3k.pkl", "wb") as f:
            pickle.dump(data, f)



    # elif task == "future_event:

    #     # future event prediction
    #     from transformers import BigBirdTokenizer

    #     tokenizer = BigBirdTokenizer.from_pretrained("/home/jupyter/research/research/experimental/bic/user_states/artifacts/tokenizers/bigbird_word")
    #     words = [w.replace("▁", "") for w in tokenizer.get_vocab().keys() if w.startswith("▁")]

    #     for i, w in enumerate(words):

    #         categories = set([w.split("_")[0] for w in words])


    #     count = collections.defaultdict(int)
    #     for seq in event_sequences:
    #         seq = seq.split()
    #         for w in seq:
    #             count[w] += 1

    #     count = {k: v for k, v in sorted(count.items(), key=lambda item: item[1])}
    #     for w in count:
    #         if "AD" in w:
    #             print(f"{w}_{count[w]}")
    
    

    
# time_l = []
# for time_sequence in time_sequences:
#     if len(time_sequence) == 0:
#         continue
#     start_date = time_sequence[0][1:3]
#     count = 0
    
#     for t in time_sequence[1:]:
#         if t[1:3] == start_date:
#             count += 1
#         else:
#             time_l.append(count)
#             count = 1
#             start_date = t[1:3]
    
#     time_l.append(count)

# print(f"mean {np.mean(time_l)} max {np.max(time_l)} min {np.min(time_l)} median {np.median(time_l)}") 