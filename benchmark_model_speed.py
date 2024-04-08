import argparse
import collections
import pickle
import json
import time
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn
from utils import *
from eval_user import pooling_modes, model_paths
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report



def benchmark_speed(
    event_sequences: List[str],
    time_sequences: List[str],
    method: str,
    modes: List[str],
    max_model_input_size: int,
    model_path: str,
    tokenizer_path: str,
    bucket_name: str,
    batch_size: int,
    # use in recurrent embedding generation. Set as a positive number to activate recurent embedding.
    recurrent_input_size: int = -1,
    split_mode: str = None,  # select from [None, rnn, split]
    last_segment_only: bool = False,
):
    """
    This function takes in a list of user event sequences and
    converts them to sentence embeddings based on a specified
    trained transformer-based model.
    """

    if recurrent_input_size > 0:
        assert max_model_input_size % recurrent_input_size == 0

    model = import_trained_model(
        method=method,
        model_path=model_path,
        bucket_name=bucket_name,
        fill_mask=False,
        tokenizer=None,
    )

    tokenizer = import_tokenizer(
        tokenizer_path=tokenizer_path,
        bucket_name=bucket_name,
        max_model_input_size=max_model_input_size,
        padding_side="left" if method in ["gpt2", "retnet", "rwkv"] else "right",
    )

    n_gpu = torch.cuda.device_count()
    device = "cuda" if n_gpu > 0 else "cpu"
    if n_gpu > 0:
        model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # build data loader to accelerate parallel computation
    use_time = time_sequences and time_sequences[0]
    if use_time:
        data = Dataset.from_dict({"text": event_sequences, "time": time_sequences})
    else:
        data = Dataset.from_dict({"text": event_sequences})
    dataset = DatasetDict()
    dataset["eval"] = data

    def _preprocess_for_masked_language_modeling(raw_sample: dict) -> dict:
        tokenizer_output = tokenizer(
            raw_sample["text"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=max_model_input_size,
        )

        sample = {
            "input_ids": tokenizer_output["input_ids"],
            "attention_mask": tokenizer_output["attention_mask"],
        }

        if use_time:
            sample["time"] = raw_sample["time"]

        return sample

    num_samples = len(event_sequences)
    # Generate a variable "batches" as a list of batch,
    # Each batch is a list of samples, where each sample is a dictionary:
    # e.g., {"event": "E1 E2 E3", "time": "T1 T2 T3"}
    batches = [
        [
            {"event": event_sequences[j], "time": time_sequences[j]}
            if time_sequences
            else {"event": event_sequences[j]}
            for j in range(i, min(i + batch_size, num_samples))
        ]
        for i in range(0, num_samples, batch_size)  # drop the last batch
    ]

    dataset = dataset.map(
        _preprocess_for_masked_language_modeling,
        batched=True,
        remove_columns=["text"],
        num_proc=16,
    )
    dataset.set_format("torch")

    data_loader = DataLoader(dataset["eval"], batch_size=batch_size * n_gpu)

    aggregated_embeddings = {}
    for mode in modes:
        aggregated_embeddings[mode] = torch.empty(0)
    
    time_list = []
    
    with torch.no_grad():
        for i, batch_inputs in enumerate(tqdm(data_loader, desc="Processing batches")):
            if i == 11:
                break
            
            # start_time = time.time()

            if split_mode is None:
                start_time = time.time()
                last_layer_embeddings = (
                    model(**batch_inputs)
                )
                time_list.append(time.time()-start_time)
                

            elif split_mode == "state":
                start_time = time.time()
                model(**batch_inputs, forward_impl="chunkwise")
                time_list.append(time.time()-start_time)
                
            elif split_mode == "rnn":
                
                prev_states = []
                num_segment = max_model_input_size // recurrent_input_size
                all_hidden_states = []
                for i in range(num_segment):
                    start_time = time.time()
                    segment_input_ids = batch_inputs["input_ids"][
                        :, recurrent_input_size * i : recurrent_input_size * (i + 1)
                    ]
                    segment_attention_mask = batch_inputs["attention_mask"][
                        :, recurrent_input_size * i : recurrent_input_size * (i + 1)
                    ]
                    rnn_out = model(
                        input_ids=segment_input_ids,
                        attention_mask=segment_attention_mask,
                        forward_impl="chunkwise",
                        prev_states=prev_states,
                        use_cache=True,
                        sequence_offset=i * recurrent_input_size,
                        output_hidden_states=True,
                    )

                    prev_states = rnn_out.prev_states
                    time_list.append(time.time()-start_time)
            elif split_mode == "split":
                num_segment = max_model_input_size // recurrent_input_size
                all_hidden_states = []
                for i in range(num_segment):
                    if last_segment_only and (i < num_segment - 1):
                        continue

                    segment_input_ids = batch_inputs["input_ids"][
                        :, recurrent_input_size * i : recurrent_input_size * (i + 1)
                    ]
                    segment_attention_mask = batch_inputs["attention_mask"][
                        :, recurrent_input_size * i : recurrent_input_size * (i + 1)
                    ]
                    split_outputs = (
                        model(
                            input_ids=segment_input_ids,
                            attention_mask=segment_attention_mask,
                        )
                    )
            
            
    return time_list
            
            
            




method = "retnet_clm_final"
tokenizer_path = "artifacts/tokenizers/bigbird_word"
dataset_dir = "/home/jupyter/research/research/experimental/bic/user_states/test/final_fep_10000_20000_1002.pkl"

tokenizer = import_tokenizer("artifacts/tokenizers/bigbird_word", 
                             bucket_name="umap-user-model", 
                             max_model_input_size=512)


with open(dataset_dir, "rb") as f:
    data_train, data_eval = pickle.load(f)
    data_train, data_eval = data_train[:2048], data_eval[:100]
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
ori_batch_size = 32
last_segment_only = False


label_names = [tokenizer.convert_ids_to_tokens(e).strip('▁') for e in event_ids]

all_results = {}

max_len = 4000
day_len = 1000
split_mode = None

# train the MLP first
for day_len in [500]:# range(4000,4001,250):
    method_type = method.split("_")[0]
    modes = pooling_modes[method_type]
    event_sequences_train = [" ".join(e.split()[-day_len:]) for e in event_sequences_train]
    time_list = benchmark_speed(event_sequences = event_sequences_train,
                                            time_sequences = [],
                                             method = method_type,
                                             modes = modes,
                                             # max_model_input_size = max_len,
                                             max_model_input_size = day_len,
                                             model_path = model_paths[method] if method in model_paths else "",
                                             tokenizer_path = tokenizer_path,
                                             bucket_name = bucket_name,
                                             batch_size = ori_batch_size,
                                             recurrent_input_size = 250,
                                             split_mode = split_mode,
                                             last_segment_only = False,
                                            )
    print(day_len)
    print(time_list)
        




        
