import argparse
import collections
import pickle
import json
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn
from utils import *
from eval_user import pooling_modes, model_paths
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report




# methods = ["bert_aT_final", "retnet_fepsup_final", "retnet_clm_final", "retnet_fep_final", "retnet_sup_final"]
methods = ["bert_final", "gpt2_final"]
# methods = ["retnet_fepsup_final", "retnet_clm_final", "retnet_fep_final", "retnet_sup_final"]
tokenizer_path = "artifacts/tokenizers/bigbird_word"
dataset_dir = "/home/jupyter/research/research/experimental/bic/user_states/test/final_fep_10000_20000_1008_token.pkl"

tokenizer = import_tokenizer("artifacts/tokenizers/bigbird_word", 
                             bucket_name="umap-user-model", 
                             max_model_input_size=512)


# with open(dataset_dir, "rb") as f:
#     data_train, data_eval = pickle.load(f)
#     data_train, data_eval = data_train[:5000], data_eval[:5001]
#     # data_train, data_eval = data_train[200:500], data_eval[:201]
#     event_ids = data_eval[0]
#     event_sequences_eval, labels_eval = [], []
#     event_sequences_train, labels_train = [], []
#     for d in data_train:
#         event_sequences_train.append(d[0])
#         labels_train.append(d[1])
#     for d in data_eval[1:]:
#         event_sequences_eval.append(d[0])
#         labels_eval.append(d[1])
#     print(len(labels_train), len(labels_eval))
#     labels_train = np.stack(labels_train)
#     labels_eval = np.stack(labels_eval)
#     # assert len(labels_train) == len(labels_eval) == 30000
#     print(f"Number of samples: {len(labels_eval)}")

with open(dataset_dir, "rb") as f:
    data_train, data_eval = pickle.load(f)
    data_train, data_eval = data_eval[5001:20001], data_eval[:5001]
    event_ids = data_eval[0]
    event_sequences_eval, labels_eval = [], []
    event_sequences_train, labels_train = [], []
    for d in data_train:
        event_sequences_train.append(d[0])
        labels_train.append(d[1])
    for d in data_eval[1:]:
        event_sequences_eval.append(d[0])
        labels_eval.append(d[1])
    print(len(labels_train), len(labels_eval))
    labels_train = np.stack(labels_train)
    labels_eval = np.stack(labels_eval)
    # assert len(labels_train) == len(labels_eval) == 30000
    print(f"Number of samples: {len(labels_eval)}")

bucket_name = "umap-user-model"
ori_batch_size = 8
last_segment_only = False


label_names = [tokenizer.convert_ids_to_tokens(e).strip('▁') for e in event_ids]

seeds = [1]
all_results = {}

max_len = 4000
day_len = 250

# train the MLP first
for seed in seeds:
    for method in methods:
        method_type = method.split("_")[0]
        modes = pooling_modes[method_type]
        # event_sequences_train = [" ".join(e.split()[-day_len:]) for e in event_sequences_train]
        train_embeddings, _, _ = feature_engineering(train_event_sequences = event_sequences_train,
                                                test_event_sequences = [],
                                                train_time_sequences = [],
                                                 test_time_sequences = [],
                                                 method = method_type,
                                                 modes = ["no_pooling"],
                                                 max_model_input_size = max_len,
                                                 # max_model_input_size = day_len,
                                                 model_path = model_paths[method] if method in model_paths else "",
                                                 tokenizer_path = tokenizer_path,
                                                 bucket_name = bucket_name,
                                                 batch_size = ori_batch_size,
                                                 recurrent_input_size = day_len,
                                                 split_mode = "split",
                                                 reverse_sequence = True,
                                                 last_segment_only = False,
                                                )
        
        embeddings = train_embeddings["no_pooling"]
        labels = labels_train
        # X_train, y_train = embeddings, labels_train
        print(f"{embeddings.shape} {labels_train.shape}")
        X_train, y_train = embeddings[:, -250:, :], labels_train[:, -250:, :]
        hidden_size = embeddings.shape[-1]
        num_events = labels_train.shape[-1]
        X_train, y_train = X_train.reshape(-1, hidden_size), y_train.reshape(-1, num_events)
        num_samples = X_train.shape[0]
        num_train = int(num_samples // 250)
        permute = np.random.permutation(num_samples)
        X_train = X_train[permute][:num_train]
        y_train = y_train[permute][:num_train]

        mlpclass = MLPClassifier(
            hidden_layer_sizes = (128,),
            learning_rate = "adaptive",
            alpha = 1,
            max_iter = 2000,
            random_state = seed,
            early_stopping = True,
        )
        mlp = Pipeline([("scaler", StandardScaler()), ("clf", mlpclass)])  #

        mlp.fit(X_train, y_train)
        print("MLP is trained")
        
        y_train_logits = mlp.predict_proba(X_train)
        auc_train = roc_auc_score(y_train, y_train_logits)
        print(f"Training AUC is {auc_train}")

        for split_mode in ["split"]:
            if split_mode == None and method_type == "bert":
                continue

            all_results[f"{split_mode}"] = collections.defaultdict(dict)


            eval_embeddings, _, _ = feature_engineering(train_event_sequences = event_sequences_eval,
                                                    test_event_sequences = [],
                                                    train_time_sequences = [],
                                                     test_time_sequences = [],
                                                     method = method_type,
                                                     modes = ["no_pooling"],
                                                     max_model_input_size = max_len,
                                                     model_path = model_paths[method] if method in model_paths else "",
                                                     tokenizer_path = tokenizer_path,
                                                     bucket_name = bucket_name,
                                                     batch_size = 8,
                                                     recurrent_input_size = day_len,
                                                     split_mode = split_mode,
                                                     reverse_sequence = True,
                                                     last_segment_only = False,
                                                    )
            eval_embeddings = eval_embeddings["no_pooling"] # [num_sample, max_len, hidden_size]

            for i in range(max_len // day_len):
                cur_len = day_len*(i+1)
                print(cur_len)
                X_test = eval_embeddings[:, i*day_len:(i+1)*day_len, :].reshape(-1, hidden_size)
                y_test = labels_eval[:,i*day_len:(i+1)*day_len,:].reshape(-1, num_events)

                # y_pred = mlp.predict(X_test)
                y_logits = mlp.predict_proba(X_test)

                #### filter classes with less than 5 samples
                # min_num_per_class = 5
                # label_count = y_test.sum(axis=0)
                # filtered_classes = np.where(label_count >= min_num_per_class)[0]
                # print(f"Filter {len(filtered_classes)}/{y_logits.shape[1]}")
                # y_test = y_test[:, filtered_classes]
                # y_logits = y_logits[:, filtered_classes]
                # filtered_label_names = [label_names[i] for i in filtered_classes]
                filtered_label_names = label_names


                loss_fct = torch.nn.BCELoss()
                auc = roc_auc_score(y_test, y_logits)
                print(method)
                print(f"\n BCELoss: {loss_fct(torch.Tensor(y_logits), torch.Tensor(y_test))}")
                # print(f"\n AUC: {roc_auc_score(y_test, y_logits, average=None)}")
                print(f"\n AUC: {auc}")

                if method not in all_results[f"{split_mode}"]:
                    all_results[f"{split_mode}"][method] = {}
                all_results[f"{split_mode}"][method][cur_len] = all_results[f"{split_mode}"][method].get(cur_len, 0) + auc * (1/len(seeds))

                with open("user_fep_token_1008_4k_gamma512.txt", "a") as f:
                    f.write(f"{method}_{cur_len}_{split_mode}_{last_segment_only}_mean_{seed}: {auc} \n")


