import argparse
import pickle
import torch.nn.functional as F
import numpy as np
from utils import *
from eval_user import pooling_modes, model_paths


methods = ["bert_final", "retnet_clm_final", "retnet_fep_final", "retnet_sup_final", "retnet_fepsup_final"]
tokenizer_path = "artifacts/tokenizers/bigbird_word"
dataset_dir = "/home/jupyter/research/research/experimental/bic/user_states/test/final_user_reid.pkl"

history_len = 4000
# get data
with open(dataset_dir, "rb") as f:
    raw_data = pickle.load(f)
    print(f"Number of users: {len(raw_data)}")
    
    user_history = [" ".join(d[0].split()[:history_len]) for d in raw_data]
    user_context = [d[1] for d in raw_data]


bucket_name = "umap-user-model"
ori_batch_size = 16
max_model_input_size = 250
recurrent_input_size = 250
split_mode = "split"


for method in methods:
    method_type = method.split("_")[0]
    modes = pooling_modes[method_type]
    batch_size = ori_batch_size // 4 if method_type in ["gpt2"] else ori_batch_size 
    batch_size = ori_batch_size // 2 if method_type in ["retnet"] else ori_batch_size 
    history_embeddings, _, _ = feature_engineering(train_event_sequences = user_history,
                                                    test_event_sequences = [],
                                                    train_time_sequences = [],
                                                     test_time_sequences = [],
                                                     method = method_type,
                                                     modes = modes,
                                                     max_model_input_size = history_len,
                                                     model_path = model_paths[method] if method in model_paths else "",
                                                     tokenizer_path = tokenizer_path,
                                                     bucket_name = bucket_name,
                                                     batch_size = ori_batch_size if method_type in ["bert", "gpt2"] else 4,
                                                     recurrent_input_size = 512 if method_type in ["bert", "gpt2"] else -1,
                                                     split_mode = "split" if method_type in ["bert", "gpt2"] else None,
                                                     reverse_sequence = True,
                                                     last_segment_only = False,
                                                    )
    
    
    for max_model_input_size in range(250, 4001, 250):
        if max_model_input_size > 500:
            batch_size = ori_batch_size // 4 if method_type in ["gpt2"] else ori_batch_size 
            times_l = max_model_input_size // 500
            batch_size = batch_size // times_l

        context_embeddings, _, _ = feature_engineering(train_event_sequences = user_context,
                                                    test_event_sequences = [],
                                                    train_time_sequences = [],
                                                     test_time_sequences = [],
                                                     method = method_type,
                                                     modes = modes,
                                                     max_model_input_size = max_model_input_size,
                                                     model_path = model_paths[method] if method in model_paths else "",
                                                     tokenizer_path = tokenizer_path,
                                                     bucket_name = bucket_name,
                                                     batch_size = batch_size,
                                                     recurrent_input_size = recurrent_input_size,
                                                     split_mode = split_mode,
                                                     reverse_sequence = True,
                                                     last_segment_only = False,
                                                    )

        def get_rank(x):
            vals = x[range(len(x)), range(len(x))]
            return (x > vals[:, None]).long().sum(1) + 1


        for mode in modes:
            embeddings_query = context_embeddings[mode] # [num_sample, feature_dim] 
            embeddings_candidate = history_embeddings[mode]
            embeddings_query = torch.tensor(embeddings_query) if type(embeddings_query) == list else torch.tensor(np.array(embeddings_query))
            embeddings_candidate = torch.tensor(embeddings_candidate) if type(embeddings_candidate) == list else torch.tensor(np.array(embeddings_candidate))
            embeddings_query = F.normalize(embeddings_query.float())
            embeddings_candidate = F.normalize(embeddings_candidate.float())
            feature_dim = embeddings_query.shape[-1]

            similarities = torch.einsum('ih,jh->ij', embeddings_query, embeddings_candidate) # [num_sample, num_sample] 
            ranks = get_rank(similarities)

            mrr = torch.mean(1.0 / ranks)
            top_1_acc = ((ranks <= 1).sum() / len(ranks)).item()
            top_3_acc = ((ranks <= 3).sum() / len(ranks)).item()
            top_5_acc = ((ranks <= 5).sum() / len(ranks)).item()
            top_10_acc = ((ranks <= 10).sum() / len(ranks)).item()

            with open("user_reid_1007_4k.txt", "a") as f:
                message = f"{method}_{mode}_{max_model_input_size}: mrr {mrr}, top1 {top_1_acc}, top3 {top_3_acc}, top5 {top_5_acc}, top10 {top_10_acc}"
                print(message)
                f.write(message + "\n")