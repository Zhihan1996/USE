import logging
import os
import random
import subprocess
import time
from collections import Counter
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import List

import fastavro
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from data_collator import (
    DataCollatorForMLMAndSameUserPrediction,
    TimeParser,
    extract_time_from_strings,
    get_time_info,
)
from datasets import Dataset, DatasetDict
from gensim.models import Word2Vec
from google.cloud import storage
from imblearn.over_sampling import RandomOverSampler
from model import EventBertForMaskedLM
from modeling_retnet import (
    RETNET_FFN_RATIO,
    RETNET_QK_RATIO,
    RETNET_V_RATIO,
    RetNetModel,
    RetNetModelWithLMHead,
)
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tabulate import tabulate
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BertForMaskedLM,
    BigBirdConfig,
    BigBirdForMaskedLM,
    BigBirdTokenizer,
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    GPT2LMHeadModel,
    pipeline,
)

logger = logging.getLogger()

POOLING_MODES = ["mean", "max", "weighted_mean", "weighted_max"]
TIME_AWARE_METHODS = ["bert"]
LANGUAGE_MODELS = ["bert", "bigbird", "deberta-v2", "gpt2", "rwkv", "retnet"]


def make_path(path: str):
    """This function makes a path if it doesn't already exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def read_user_table(
    bucket_name: str = "bic-test",
    gcs_folder_name: str = "data",
    task: str = "train",
    sample_n: int = 1000,
    features_start_date: str = "20230313",
    features_end_date: str = "20230313",
    labels_date: str = "20230313",
    sample_table_option: str = "sample",
    user_filter_option: str = "unfiltered",
    sc_analytics_tables: str = "all",
    user_id_filter: list = None,
    max_files: int = 10000,
    length_to_keep: int = 512,
):
    """This function reads user tables stored on bigquery."""

    assert task in get_available_tasks(scope="all")
    assert bucket_name in ["umap-user-model", "bic-test"]

    prefix = f"{gcs_folder_name}/{task}_user_sample_in_{sc_analytics_tables}_{sample_n}_from_{features_start_date}_to_{features_end_date}_on_{labels_date}_{sample_table_option}_{user_filter_option}"

    data = read_avro_file_from_gcs(
        bucket_name=bucket_name,
        prefix=prefix,
        max_files=max_files,
        length_to_keep=length_to_keep,
    )

    df = pd.DataFrame(data)
    df.drop_duplicates(subset="ghost_user_id", keep="first", inplace=True)

    if user_id_filter is not None:
        df = df[df.ghost_user_id.isin(user_id_filter)]
        df = df.set_index("ghost_user_id")
        df = df.loc[user_id_filter].reset_index()

    return df.reset_index(drop=True)


def read_user_sequences(
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
    # use to load sequence in the correct order since the files stored on GCP are in a time-reversed order
    # should set as false when working on datasets when the sequence orders are correct
    reverse_sequence: bool = True,
):
    """
    This function reads user sequences stored on GCS bucket.
    You can specify whether you want only event sequences or both event and time sequences.
    """
    assert task in get_available_tasks(scope="all")

    labels_date = features_end_date if task == "train" else labels_date
    prefix = f"{prefix}_{task}_in_{sc_analytics_tables}_{sample_n}_from_{features_start_date}_to_{features_end_date}_on_{labels_date}_{sample_table_option}_{user_filter_option}"

    data = read_avro_file_from_gcs(
        bucket_name=bucket_name,
        prefix=prefix,
        max_files=max_files,
        length_to_keep=length_to_keep,
    )

    assert len(data) > 0

    data.sort(key=lambda item: item["event_sequence"])
    event_sequences = [item["event_sequence"] for item in data]
    time_sequences = [item["time_sequence"] for item in data]

    ghost_user_ids = [event_sequence[0] for event_sequence in event_sequences]

    time_parser = TimeParser() if use_time else None

    processed_event_sequences = []
    processed_time_sequences = []

    for event_sequence, time_sequence in zip(event_sequences, time_sequences):
        processed_event_sequence, processed_time_sequence = preprocess_user_sequence(
            event_sequence, time_sequence, sequence_type, time_parser, reverse_sequence
        )
        if processed_event_sequence:
            processed_event_sequences.append(processed_event_sequence)
            processed_time_sequences.append(processed_time_sequence)

    return processed_event_sequences, processed_time_sequences, ghost_user_ids


def extract_embeddings(
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

    with torch.no_grad():
        for batch_inputs in tqdm(data_loader, desc="Processing batches"):
            if use_time and method in TIME_AWARE_METHODS:
                batch_inputs["time_ids"] = inputs["time_ids"].to(
                    batch_inputs["input_ids"].device
                )

            if split_mode is None:
                last_layer_embeddings = (
                    model(**batch_inputs).hidden_states[-1].cpu().detach()
                )

            elif split_mode == "state":
                outputs = model(**batch_inputs, forward_impl="chunkwise")
                prev_states = outputs.prev_states[-1]
                last_layer_embeddings = prev_states.mean(dim=1)
                bs = last_layer_embeddings.shape[0]
                last_layer_embeddings = last_layer_embeddings.reshape([bs, 1, -1])

            elif split_mode == "rnn":
                prev_states = []
                num_segment = max_model_input_size // recurrent_input_size
                all_hidden_states = []
                for i in range(num_segment):
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

                    all_hidden_states.append(rnn_out.hidden_states[-1].cpu().detach())
                    prev_states = rnn_out.prev_states
                all_hidden_states = (
                    all_hidden_states[-1] if last_segment_only else all_hidden_states
                )
                last_layer_embeddings = torch.concat(all_hidden_states, dim=1)
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
                        .hidden_states[-1]
                        .cpu()
                        .detach()
                    )
                    all_hidden_states.append(split_outputs)
                last_layer_embeddings = torch.concat(all_hidden_states, dim=1)
            else:
                raise ValueError("Only support split_node of [None, rnn, split]")

            for mode in modes:
                if mode == "mean":
                    embedding = torch.mean(last_layer_embeddings, dim=1)
                elif mode == "max":
                    embedding = torch.max(last_layer_embeddings, dim=1)[0]
                elif mode == "cls":
                    embedding = last_layer_embeddings[:, 0, :]
                elif mode == "no_pooling":
                    embedding = last_layer_embeddings
                elif mode == "last_seg_mean":
                    embedding = torch.mean(
                        last_layer_embeddings[:, -recurrent_input_size:, :], dim=1
                    )
                elif mode == "last":
                    # find the last non-padding token and take its embedding as sequence embedding
                    sequence_lengths = (
                        torch.eq(batch_inputs["input_ids"], tokenizer.pad_token_id)
                        .long()
                        .argmax(-1)
                        - 2
                    ).to(last_layer_embeddings.device)
                    embedding = last_layer_embeddings[
                        torch.arange(
                            batch_inputs["input_ids"].shape[0],
                            device=last_layer_embeddings.device,
                        ),
                        sequence_lengths,
                    ]
                else:
                    weights_tensor = torch.tensor(
                        get_weights(last_layer_embeddings.shape[1])
                    ).to(last_layer_embeddings.device)
                    weighted_last_layer_embeddings = (
                        last_layer_embeddings * weights_tensor.view(1, -1, 1)
                    )
                    if mode == "weighted_mean":
                        embedding = torch.mean(weighted_last_layer_embeddings, dim=1)
                    elif mode == "weighted_max":
                        embedding = torch.max(weighted_last_layer_embeddings, dim=1)[0]
                aggregated_embeddings[mode] = torch.cat(
                    (aggregated_embeddings[mode], embedding.cpu()), dim=0
                )

    return aggregated_embeddings


def get_data_split_indices(
    labels: list, test_size: float = 0.2, random_state: int = 42
):

    """
    This function splits a list into train and test partitions,
    and returns their respective indices.
    """

    numbers = list(range(len(labels)))
    random.seed(random_state)
    test_ind = random.sample(numbers, k=len(labels) * test_size)
    train_ind = list(set(numbers) - set(test_ind))

    return train_ind, test_ind


def mlm_topk_performance(predicted_tokens_list, target_tokens, event_sequences, k):
    """
    This function assesses the performance of masked event prediction,
    returns a dictionary of accuracy@k scores, and a dataframe containing
    the recall and precision scores for each token.
    """
    assert k <= len(predicted_tokens_list[0])

    unique_target_tokens = set(target_tokens)
    unique_predicted_tokens = set(
        [
            d["token_str"]
            for sublist in predicted_tokens_list
            for i, d in enumerate(sublist)
            if i < k
        ]
    )
    top_k_predicted_tokens = [
        [d["token_str"] for d in sublist[:k]] for sublist in predicted_tokens_list
    ]

    correct_ls = []
    acc_dict = {}
    recall_dict = {}
    precision_dict = {}

    for i in range(len(target_tokens)):
        predicted_tokens_dict_list = predicted_tokens_list[i]
        target_token = target_tokens[i]

        # Get the top-k predicted tokens
        topk_predicted_tokens = [
            predicted_tokens_dict_list[idx]["token_str"] for idx in range(k)
        ]

        if target_token in topk_predicted_tokens:
            correct_ls.append(1)
        else:
            correct_ls.append(0)

    # compute top-k accuracy
    acc_dict[f"ALL@{k}"] = sum(correct_ls) / len(target_tokens)

    # compute top-k recall per class
    for token in unique_target_tokens:
        indices = [i for i in range(len(target_tokens)) if target_tokens[i] == token]
        sum_correct = sum([correct_ls[i] for i in indices])
        recall_dict[token] = sum_correct / len(indices)

    recall_df = pd.DataFrame.from_dict(
        recall_dict, orient="index", columns=[f"recall@{k}"]
    ).reset_index(names="event")

    # compute top-k precision per class
    for token in unique_predicted_tokens:
        indices = [
            i for i in range(len(target_tokens)) if token in top_k_predicted_tokens[i]
        ]
        sum_correct = sum([correct_ls[i] for i in indices])
        precision_dict[token] = sum_correct / len(indices)

    prec_df = pd.DataFrame.from_dict(
        precision_dict, orient="index", columns=[f"prec@{k}"]
    ).reset_index(names="event")

    # putting all in a dataframe
    df = (
        pd.Series(
            data=[
                event
                for event_sequence in event_sequences
                for event in event_sequence.split(" ")
            ]
        )
        .value_counts()
        .to_frame(name="freq")
        .reset_index(names="event")
    )
    df = df.merge(recall_df, how="left", on="event")
    df = df.merge(prec_df, how="left", on="event")

    return acc_dict, df


def get_weights(num_tokens):
    """
    This function computes weights for the tokens in an event sequence.
    The weights increase linearly from left to right.
    """
    weights = np.arange(1, num_tokens + 1)
    weights = weights / np.sum(weights)
    return weights


def aggregate_token_embeddings_by_mode(embeddings, mode):
    """
    This function takes in a list of token embeddings, and
    aggregates these embeddings into sentence embeddings,
    based on mean, max, weighted_mean, and weighted_max.
    This is used for the baseline models: sgns and random.
    """
    if len(embeddings) > 0:
        if mode == "mean":
            return np.mean(embeddings, axis=0)
        elif mode == "max":
            return np.max(embeddings, axis=0)
        else:
            weights = get_weights(len(embeddings[0]))
            weighted_embeddings = [
                weight * embedding for weight, embedding in zip(weights, embeddings)
            ]
            if mode == "weighted_mean":
                return np.mean(weighted_embeddings, axis=0)
            elif mode == "weighted_max":
                return np.max(weighted_embeddings, axis=0)
    else:
        return None


def aggregate_sgns_embedding(sequence, sgns_model, mode):
    """
    This function takes in an event sequence, a trained sgns model,
    a specified aggregation mode, and return the corresponding
    aggregated sentence embedding.
    """

    if len(sequence.split(" ")) == 0:
        embeddings = np.zeros(sgns_model.vector_size)
        return embeddings

    else:
        embeddings = []
        for token in sequence.split(" "):
            if token in sgns_model.wv.index_to_key:
                embeddings.append(sgns_model.wv[token])
            else:
                embeddings.append(np.zeros(sgns_model.vector_size))

    return aggregate_token_embeddings_by_mode(embeddings, mode)


def create_random_embeddings_dict(sequences, embedding_size=768):
    """
    This function creates a dictionary of random embeddings, based on
    an input list of event sequences.
    """
    unique_tokens = list(
        set(token for sequence in sequences for token in sequence.split(" "))
    )

    random_embeddings_dict = {}
    np.random.seed(42)
    for token in unique_tokens:
        random_embeddings_dict[token] = np.random.uniform(
            low=-0.05, high=0.05, size=embedding_size
        )

    return random_embeddings_dict


def aggregate_random_embedding(sequence, dictionary, mode, embedding_size):
    """
    This function aggregates token-level random embeddings into
    a sentence-level embedding.
    """
    embedding_size = len(dictionary[next(iter(dictionary))])

    if len(sequence.split(" ")) == 0:
        embeddings = np.zeros(embedding_size)
        return embeddings

    else:
        embeddings = []
        for token in sequence.split(" "):
            if token in dictionary:
                embeddings.append(dictionary[token])
            else:
                embeddings.append(np.zeros(embedding_size))

    return aggregate_token_embeddings_by_mode(embeddings, mode)


def trim_event_sequences(
    event_sequences: list, time_sequences: list, max_model_input_size: int
):
    """
    This function trims event sequences into a specified length.
    """
    event_sequences_split_and_trimmed = [
        " ".join(sequence.split(" ")[max_model_input_size - 1 :: -1])
        for sequence in event_sequences
    ]
    time_sequence_split_and_trimmed = [
        sequence[max_model_input_size - 1 :: -1] for sequence in time_sequences
    ]

    return event_sequences_split_and_trimmed, time_sequence_split_and_trimmed


def feature_engineering(
    train_event_sequences: list,
    test_event_sequences: list,
    train_time_sequences: list,
    test_time_sequences: list,
    modes: List[str],
    method: str = "tf",
    max_model_input_size: int = 512,
    model_path: str = None,
    tokenizer_path: str = None,
    bucket_name: str = None,
    n_pca_dims: int = -1,
    batch_size: int = 32,
    # use in recurrent embedding generation. Set as a positive number to activate recurrent embedding.
    recurrent_input_size: int = -1,
    split_mode: str = None,
    last_segment_only: bool = False,
    **kwargs,
):
    """
    This function turns input event sequences into numerical representations.
    """
    assert method in [
        "tf",
        "tf-l2",
        "tf-idf",
        "n-grams",
        "sgns",
        "random",
        "bert",
        "bigbird",
        "deberta-v2",
        "retnet",
        "rwkv",
        "gpt2",
    ]

    # truncate sequences with max_model_input_size
    train_event_sequences = [
        " ".join(seq.split()[-max_model_input_size:]) for seq in train_event_sequences
    ]
    test_event_sequences = [
        " ".join(seq.split()[-max_model_input_size:]) for seq in test_event_sequences
    ]
    train_time_sequences = [seq[-max_model_input_size:] for seq in train_time_sequences]
    test_time_sequences = [seq[-max_model_input_size:] for seq in test_time_sequences]

    train_features = {}
    test_features = {}

    if method == "tf":
        vectorizer = CountVectorizer()
        train_features["na"] = vectorizer.fit_transform(train_event_sequences).toarray()
        if test_event_sequences:
            test_features["na"] = vectorizer.transform(test_event_sequences).toarray()

    elif method == "n-grams":
        vectorizer = CountVectorizer(ngram_range=(1, 4))
        train_features["na"] = vectorizer.fit_transform(train_event_sequences).toarray()
        if test_event_sequences:
            test_features["na"] = vectorizer.transform(test_event_sequences).toarray()

    elif method == "tf-l2":
        vectorizer = TfidfVectorizer(use_idf=False, norm="l2")
        train_features["na"] = vectorizer.fit_transform(train_event_sequences).toarray()
        if test_event_sequences:
            test_features["na"] = vectorizer.transform(test_event_sequences).toarray()

    elif method == "tf-idf":
        vectorizer = TfidfVectorizer()
        train_features["na"] = vectorizer.fit_transform(train_event_sequences).toarray()
        if test_event_sequences:
            test_features["na"] = vectorizer.transform(test_event_sequences).toarray()

    elif method == "sgns":
        assert all([mode in POOLING_MODES for mode in modes])
        sgns_model = import_sgns_model(model_path=model_path, bucket_name=bucket_name)

        for mode in modes:
            train_features[mode] = [
                aggregate_sgns_embedding(sequence, sgns_model, mode)
                for sequence in tqdm(train_event_sequences)
            ]
            if test_event_sequences:
                test_features[mode] = [
                    aggregate_sgns_embedding(sequence, sgns_model, mode)
                    for sequence in tqdm(test_event_sequences)
                ]

    elif method == "random":
        assert all([mode in POOLING_MODES for mode in modes])
        random_embeddings_dict = create_random_embeddings_dict(train_event_sequences)
        embedding_size = len(random_embeddings_dict[next(iter(random_embeddings_dict))])

        for mode in modes:
            train_features[mode] = [
                aggregate_random_embedding(
                    sequence, random_embeddings_dict, mode, embedding_size
                )
                for sequence in tqdm(train_event_sequences)
            ]
            if test_event_sequences:
                test_features[mode] = [
                    aggregate_random_embedding(
                        sequence, random_embeddings_dict, mode, embedding_size
                    )
                    for sequence in tqdm(test_event_sequences)
                ]

    elif method in LANGUAGE_MODELS:

        train_features = extract_embeddings(
            event_sequences=train_event_sequences,
            time_sequences=train_time_sequences,
            method=method,
            modes=modes,
            max_model_input_size=max_model_input_size,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            bucket_name=bucket_name,
            batch_size=batch_size,
            recurrent_input_size=recurrent_input_size,
            split_mode=split_mode,
        )
        if test_event_sequences:
            test_features = extract_embeddings(
                event_sequences=test_event_sequences,
                time_sequences=test_time_sequences,
                method=method,
                modes=modes,
                max_model_input_size=max_model_input_size,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                bucket_name=bucket_name,
                batch_size=batch_size,
                recurrent_input_size=recurrent_input_size,
                split_mode=split_mode,
            )

    if n_pca_dims > 0:
        # only perform PCA-based dimension reduction when the target dimension is smaller than
        for mode in train_features.keys():
            if n_pca_dims < len(train_features[mode][0]):
                pca = PCA(n_components=n_pca_dims)
                train_features[mode] = pca.fit_transform(train_features[mode])
                test_features[mode] = pca.transform(test_features[mode])

    if method in ["tf", "tf-l2", "tf-idf"]:
        feature_names = vectorizer.get_feature_names_out().tolist()
    else:
        feature_names = []

    return train_features, test_features, feature_names


def classifier_tune_and_fit(
    classifier: str,
    train_features: list,
    test_features: list,
    train_labels: list,
    test_labels: list,
    tune_n_iter: int,
    n_class: int,
    negative_label: str,
    class_list: list,
    task: str,
    feature_names: list,
    save_partial_dependence_results: bool,
    save_feature_importance_results: bool,
    save_predictions: bool,
):
    """
    This function does hyperparameter tuning for a specified classifier
    using random search and k-fold cross validation, fits the best
    hyperparameter values to the complete data, and returns classification
    results (accuracy, f1, precision, recall, auc). In addition,
    for multiclass prediction, a confusion matrix is returned.
    User can also specify the model to save partial dependence results,
    feature importance scores, and individual predictions.
    """
    assert classifier in ["lr", "mlp"]

    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels + test_labels)

    if n_class > 2:
        average_choice = "macro"
        scoring = "f1_macro"
        train_labels = label_encoder.transform(train_labels)
        test_labels = label_encoder.transform(test_labels)
    else:
        average_choice = "binary"
        scoring = "f1"
        train_labels = [0 if label == negative_label else 1 for label in train_labels]
        test_labels_original = test_labels
        test_labels = [0 if label == negative_label else 1 for label in test_labels]
        class_list = [
            class_name for class_name in class_list if class_name != negative_label
        ]

    if classifier == "lr":
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression())]
        )  #

        param_distributions = {
            "clf__penalty": ["l1", "l2", "elasticnet"],
            "clf__C": np.logspace(-4, 4, 20),
            "clf__l1_ratio": np.arange(0.1, 1, 0.1),
            "clf__solver": ["saga"],
            "clf__max_iter": [2000],
            "clf__random_state": [42],
            "clf__class_weight": ["balanced"],
            "clf__n_jobs": [-1],
        }

    else:
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier())])  #

        param_distributions = {
            "clf__hidden_layer_sizes": [(100,), (50, 50), (20, 20, 20)],
            "clf__activation": ["tanh", "relu"],
            "clf__solver": ["adam", "sgd"],
            "clf__alpha": np.logspace(-5, 3, 9),
            "clf__learning_rate": ["constant", "invscaling", "adaptive"],
            "clf__max_iter": [2000],
            "clf__random_state": [42],
            "clf__early_stopping": [True, False],
        }

    #         oversampler = RandomOverSampler(
    #             sampling_strategy="not majority", random_state=42
    #         )

    #         train_features, train_labels = oversampler.fit_resample(
    #             train_features, train_labels
    #         )

    random_search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=tune_n_iter,
        cv=5,
        random_state=42,
        n_jobs=-1,
        scoring=scoring,
    )
    random_search.fit(train_features, train_labels)

    best_estimator = random_search.best_estimator_
    best_estimator.fit(train_features, train_labels)

    if save_partial_dependence_results:
        clf = best_estimator.named_steps.clf
        n_features = len(feature_names)
        partial_dependence_df = pd.DataFrame()
        for feature_index in range(n_features):
            partial_dependence_results = partial_dependence(
                clf,
                features=[feature_index],
                X=train_features,
                percentiles=(0, 1),
                grid_resolution=10,
            )

            df_subset = pd.DataFrame()
            df_subset["y"] = partial_dependence_results["average"][0]
            df_subset["x"] = partial_dependence_results["values"][0]
            df_subset["feature"] = feature_names[feature_index]
            partial_dependence_df = pd.concat(
                [partial_dependence_df, df_subset], axis=0, ignore_index=True
            )

    else:
        partial_dependence_df = pd.DataFrame()

    if save_feature_importance_results:
        clf = best_estimator.named_steps.clf
        feature_importances_df = pd.DataFrame()
        for metric in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
            feature_importances_results = permutation_importance(
                clf,
                test_features,
                test_labels,
                n_repeats=10,
                random_state=42,
                scoring=metric,
                n_jobs=20,
            )
            feature_importances_results = pd.DataFrame(
                {
                    "importances_mean": feature_importances_results["importances_mean"],
                    "importances_std": feature_importances_results["importances_std"],
                    "metric": metric,
                    "feature": feature_names,
                }
            )

            feature_importances_df = pd.concat(
                [feature_importances_df, feature_importances_results],
                axis=0,
                ignore_index=True,
            )

    else:
        feature_importances_df = pd.DataFrame()

    predictions = best_estimator.predict(test_features)
    pred_probs = best_estimator.predict_proba(test_features)

    if save_predictions and n_class == 2:
        predictions_df = pd.DataFrame(
            {
                "prediction": predictions,
                "pred_prob": pred_probs[:, 1],
                "label": test_labels,
            }
        )

    else:
        predictions_df = pd.DataFrame()

    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average=average_choice
    )

    results_per_class = {}

    # compute auc and class- or label_name-level performance (if relevant)
    if n_class > 2:
        class_counts = Counter(label_encoder.inverse_transform(test_labels))

        auc = roc_auc_score(
            test_labels, pred_probs, average=average_choice, multi_class="ovo"
        )

        precisions, recalls, f1s, _ = precision_recall_fscore_support(
            test_labels, predictions, average=None
        )

        aucs = roc_auc_score(test_labels, pred_probs, multi_class="ovr", average=None)

        class_list = label_encoder.classes_.tolist()

        for class_idx, class_name in enumerate(class_list):
            results_per_class[f"precision_{class_name}"] = precisions[class_idx]
            results_per_class[f"recall_{class_name}"] = recalls[class_idx]
            results_per_class[f"f1_{class_name}"] = f1s[class_idx]
            results_per_class[f"auc_{class_name}"] = aucs[class_idx]
            results_per_class[f"n_{class_name}_for_metrics"] = class_counts[class_name]

        cm = confusion_matrix(test_labels, predictions)
        df_cm = pd.DataFrame(cm, index=class_list, columns=class_list)

    else:
        auc = roc_auc_score(test_labels, pred_probs[:, 1])

        if len(class_list) >= 2:
            # if it's a binary prediction task and there are at least two unique label_names
            # that are not the negative label; allows us to look at label_name level performance

            for class_name in class_list:
                indices_positive_class = [
                    index
                    for index, label in enumerate(test_labels_original)
                    if label == class_name
                ]
                indices_negative_class = [
                    index
                    for index, label in enumerate(test_labels_original)
                    if label == negative_label
                ]
                random.seed(42)
                indices_negative_class = random.sample(
                    indices_negative_class, k=len(indices_positive_class)
                )

                class_pred_prob = [
                    pred_probs[:, 1][index] for index in indices_positive_class
                ] + [pred_probs[:, 1][index] for index in indices_negative_class]

                class_predictions = [
                    predictions[index] for index in indices_positive_class
                ] + [predictions[index] for index in indices_negative_class]

                class_test_labels = [1] * len(indices_positive_class) + [0] * len(
                    indices_negative_class
                )
                (
                    class_precision,
                    class_recall,
                    class_f1,
                    _,
                ) = precision_recall_fscore_support(
                    class_test_labels, class_predictions, average=average_choice
                )

                class_auc = roc_auc_score(class_test_labels, class_pred_prob)

                results_per_class[f"precision_{class_name}"] = class_precision
                results_per_class[f"recall_{class_name}"] = class_recall
                results_per_class[f"f1_{class_name}"] = class_f1
                results_per_class[f"auc_{class_name}"] = class_auc
                results_per_class[f"n_{class_name}_for_metrics"] = len(
                    indices_positive_class
                )

        df_cm = pd.DataFrame()

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }

    results.update(results_per_class)

    return results, df_cm, partial_dependence_df, feature_importances_df, predictions_df


def split_sequence(event_sequence, time_sequence, max_len=510):
    """
    This function splits an event sequence into multiple chunks,
    if the length of the sequence is at least twice the max_len.
    """
    n = len(event_sequence)
    assert n >= max_len * 2

    event_seqs = []
    time_seqs = []

    i = 0
    while i < n // max_len:

        if i == 0:
            start = 0
            end = max_len
        else:
            start = end
            end = end + max_len

        event = event_sequence[start:end]
        time = time_sequence[start:end]
        event_seqs.append(" ".join(event))
        time_seqs.append(time)

        i = i + 1

    return event_seqs, time_seqs


def get_split_event_sequences_sample(
    event_sequences, time_sequences, max_len=510, sample_size=100
):
    """
    This function takes in a list of event sequences, reads up to sample_size,
    splits these sequences into two or multiple ones, and returns both the split
    sequences and their respective ghost_user_ids.
    """

    split_event_sequences_sample = []
    split_time_sequences_sample = []
    user_id_ls = []

    for i in range(sample_size):
        try:
            split_event_sequences, split_time_sequences = split_sequence(
                event_sequences[i].split(), time_sequences[i], max_len=max_len
            )
            split_event_sequences_sample += split_event_sequences
            split_time_sequences_sample += split_time_sequences
            user_id_ls += [i] * len(split_event_sequences)
        except:
            pass

    return split_event_sequences_sample, split_time_sequences_sample, user_id_ls


def check_mlm_predictions_dist(mlm_predictions: list, k: int):
    """
    This function checks the distribution of top k predicted tokens.
    This is to make sure that the model isn't just predicting the
    most frequent tokens all the time.
    """
    mlm_predictions_at_k = [sublist[0:k] for sublist in mlm_predictions]
    values_of_key = []

    for sublist in mlm_predictions_at_k:
        for dict_item in sublist:
            if "token_str" in dict_item:
                values_of_key.append(dict_item["token_str"])

    counts = Counter(values_of_key)
    total_count = len(values_of_key)

    ordered_counts = [(key, value) for key, value in counts.most_common()]

    percentages = {}
    for key, value in ordered_counts:
        percentages[key] = np.round((value / total_count) * 100, 2)

    return percentages


def mask_tokens_simple_random(event_sequences: list, random_state=42):
    """
    This function masks one token in an event sequence randomly, for
    every sequence in the input list.
    Returns the masked sequences and the true values of the masked tokens.
    """
    masked_sequences = []
    masked_tokens = []
    random.seed(random_state)
    for sequence in event_sequences:
        tokens = sequence.split(" ")
        mask_index = random.randint(0, len(tokens) - 1)
        masked_token = tokens[mask_index]
        tokens[mask_index] = "[MASK]"
        masked_sequence = " ".join(tokens)
        masked_sequences.append(masked_sequence)
        masked_tokens.append(masked_token)
    return masked_sequences, masked_tokens


def mask_tokens_stratified(event_sequences: list, random_state=42):
    """
    This function masks one token in an event sequence in a stratified fashion, for
    every sequence in the input list. This stratification strategy is to make sure that
    each unique token gets masked equally often. However, because of the skewed distribution
    of unique tokens, this stratification method still can't achieve completely equal masking of
    different tokens (but close to).
    Returns the masked sequences and the true values of the masked tokens.
    """
    # count the frequency of each token
    token_counts_all = Counter(
        [token for sequence in event_sequences for token in sequence.split(" ")]
    )

    masked_tokens = []
    masked_sequences = []

    # mask one token per sequence
    random.seed(random_state)
    for sequence in event_sequences:
        split_sequence = sequence.split(" ")
        unique_tokens = list(set(split_sequence))

        if len(unique_tokens) == 0:
            continue

        while True:
            masked_token = unique_tokens[random.randint(0, len(unique_tokens) - 1)]
            decision_threshold = token_counts_all[masked_token] / sum(
                token_counts_all.values()
            )
            if (
                random.random() > decision_threshold
            ):  # take into account the frequency of the token across all sequences
                masked_token_indices = [
                    i
                    for i in range(len(split_sequence))
                    if split_sequence[i] == masked_token
                ]
                masked_token_index = random.choice(masked_token_indices)
                break  # exit when a token has been chosen to be masked

        masked_tokens.append(masked_token)
        split_sequence[masked_token_index] = "[MASK]"
        masked_sequence = " ".join(split_sequence)
        masked_sequences.append(masked_sequence)

    return masked_sequences, masked_tokens


def preprocess_user_sequence(
    event_sequence: List[str],
    time_sequence: List[str],
    strategy: str,
    time_parser: TimeParser = None,
    # use to load sequence in the correct order since the files stored on GCP are in a time-reversed order
    # should set as false when working on datasets when the sequence orders are correct
    reverse_sequence: bool = True,
):
    """
    This function converts an event sequence into the specified format.
    """
    assert strategy in [
        "shortlist",
        "shortlist_without_page_page_view",
        "shortlist_without_repetition",
        "shortlist_with_counts",
    ]

    split_event_sequence = (
        list(reversed(event_sequence[1:])) if reverse_sequence else event_sequence[1:]
    )
    if time_parser is not None:
        split_time_sequence = (
            list(reversed(time_sequence[1:])) if reverse_sequence else time_sequence[1:]
        )
        # assert len(split_event_sequence) == len(
        #     split_time_sequence
        # ), f"The event sequence has a different length ({len(split_event_sequence)}) as its corresponding time sequence ({len(split_time_sequence)}). Please double-check the dataset."

        split_time_sequence = extract_time_from_strings(
            split_time_sequence, time_parser
        )
    else:
        split_time_sequence = []

    if strategy == "shortlist":
        return " ".join(split_event_sequence), split_time_sequence

    elif strategy == "shortlist_without_page_page_view":
        new_event_sequence, new_time_sequence = zip(
            *[
                (token, ts)
                for token, ts in zip(split_event_sequence, split_time_sequence)
                if token != "PAGE_PAGE_VIEW"
            ]
        )

    elif strategy == "shortlist_without_repetition":
        new_event_sequence = []
        new_time_sequence = []
        for i, token in enumerate(split_event_sequence):
            if len(new_event_sequence) == 0 or token != new_event_sequence[-1]:
                new_event_sequence.append(token)
                new_time_sequence.append(split_time_sequence[i])

    elif strategy == "shortlist_with_counts":

        new_event_sequence = []
        new_time_sequence = []

        current_token = split_event_sequence[0]
        count = 1

        for i in range(1, len(split_event_sequence)):
            if split_event_sequence[i] == current_token:
                count += 1
            else:
                if count > 1:
                    new_event_sequence.extend([current_token, str(fuzzy_int(count))])
                    new_time_sequence.extend(
                        [split_time_sequence[i - count], split_time_sequence[i - 1]]
                    )
                else:
                    new_event_sequence.append(current_token)
                    new_time_sequence.append(split_time_sequence[i - 1])
                    current_token = split_event_sequence[i]
                count = 1

        if count > 1:
            new_event_sequence.extend([current_token, str(fuzzy_int(count))])
            new_time_sequence.extend(
                [split_time_sequence[i - count], split_time_sequence[i]]
            )
        else:
            new_event_sequence.append(current_token)
            new_time_sequence.append(split_time_sequence[i])

    return " ".join(new_event_sequence), ",".join(new_time_sequence)


def fuzzy_int(num: int):
    """
    This function is intended to convert raw counts (e.g., the number of repetitions
    of an event in an event sequence) into bins, to reduce data sparsity.
    """
    if num <= 10:
        return num
    elif num < 100:
        return round(num / 10) * 10
    elif num < 1000:
        return round(num / 100) * 100
    else:
        return 10000


def make_tsne_plot(
    features,
    user_table,
    perplexity,
    group_variable,
    random_state,
    output_path=None,
    file_name=None,
):
    """
    This function takes in user embeddings and user tables, and a grouping variable,
    and returns a t-SNE plot showing the embeddings in 2-dimensional space, colored
    by the groups.
    """
    # Use t-SNE to project the embeddings into a 2D space
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    tsne_embeddings = tsne.fit_transform(np.array(features))

    user_table["tsne_x"] = tsne_embeddings[:, 0]
    user_table["tsne_y"] = tsne_embeddings[:, 1]

    # Create a dictionary that maps each group to a unique color
    unique_groups = set(user_table[group_variable].tolist())
    color_dict = {}
    for group in unique_groups:
        color_dict[group] = sns.color_palette("bright", len(unique_groups))[
            list(unique_groups).index(group)
        ]

    # Plot the scatterplot with the colors depending on gender
    sns.scatterplot(
        x="tsne_x", y="tsne_y", hue=group_variable, palette=color_dict, data=user_table
    )

    # Add a legend to the plot
    plt.legend(title=group_variable, fontsize="small", loc="upper left")

    if output_path and file_name:
        make_path(output_path)
        plt.savefig(f"{output_path}/{file_name}")

    plt.show()


def compute_cosine_similarity_at_group_level(
    ingroup_embeddings: list, outgroup_embeddings: list, n: int, random_state: int = 42
):
    """
    This function computes the between-user cosine similarity for different
    users of the same group, and the between-user cosine similarity for users
    of different groups.
    """
    random.seed(random_state)
    ingroup_embeddings_sample = random.sample(ingroup_embeddings, n)

    random.seed(random_state)
    outgroup_embeddings_sample = random.sample(outgroup_embeddings, n)

    within_group_cos_sim_scores = []
    between_groups_cos_sim_scores = []

    for i in range(n):
        for j in range(i + 1, n):
            within_group_cos_sim_scores.append(
                F.cosine_similarity(
                    torch.tensor(ingroup_embeddings_sample[i]),
                    torch.tensor(ingroup_embeddings_sample[j]),
                    dim=0,
                )
            )

    for i in range(n):
        for j in range(n):
            between_groups_cos_sim_scores.append(
                F.cosine_similarity(
                    torch.tensor(ingroup_embeddings_sample[i]),
                    torch.tensor(outgroup_embeddings_sample[j]),
                    dim=0,
                )
            )

    return np.mean(within_group_cos_sim_scores), np.mean(between_groups_cos_sim_scores)


def compute_cosine_similarity_at_user_level(embeddings, user_ids, n, random_state=42):
    """
    This function computes within-user and between-user cosine similarity.
    """
    within_user_cos_sim_averages = []
    between_users_cos_sim_averages = []

    embeddings = torch.tensor(np.array(embeddings)).float()

    for user_id in set(user_ids):
        same_user_indices = [i for i in range(len(user_ids)) if user_ids[i] == user_id]
        same_user_embeddings = embeddings[same_user_indices]
        other_user_indices = [i for i in range(len(user_ids)) if user_ids[i] != user_id]
        random.seed(random_state)
        other_user_indices = random.sample(other_user_indices, n)
        other_user_embeddings = embeddings[other_user_indices]

        within_user_cos_sim_scores = []
        between_users_cos_sim_scores = []

        for i in range(len(same_user_embeddings)):
            for j in range(i + 1, len(same_user_embeddings)):
                within_user_cos_sim_scores.append(
                    F.cosine_similarity(
                        same_user_embeddings[i], same_user_embeddings[j], dim=0
                    )
                )

        for i in range(len(same_user_embeddings)):
            for j in range(len(other_user_embeddings)):
                between_users_cos_sim_scores.append(
                    F.cosine_similarity(
                        same_user_embeddings[i], other_user_embeddings[j], dim=0
                    )
                )

        within_user_cos_sim_averages.append(np.mean(within_user_cos_sim_scores))
        between_users_cos_sim_averages.append(np.mean(between_users_cos_sim_scores))

    return np.mean(within_user_cos_sim_averages), np.mean(
        between_users_cos_sim_averages
    )


def evaluate_cosine_similarity(
    group_variable,
    user_table,
    event_sequences,
    embeddings,
    model_path,
    tokenizer_path,
    bucket_name,
    max_model_input_size,
    method,
    mode,
    min_sample_size,  # minimum requirement for the number of observations in a group, and for the number of users to sample for the cosine analysis
    user_level=True,  # boolean; whether or not results for the group level analysis should be returned
    group_level=True,  # boolean; whether or not only results for the group level analysis should be returned
):
    """
    If group_level=True, this function computes within- and between-group
    cosine similarity;
    if user_level=True, this function computes within-group-within-user cosine similarity
    and between-group-between-user cosine similarity. This second approach was only used briefly during
    the project.
    """
    value_counts = Counter(user_table[group_variable])
    unique_values = [
        key
        for key, value in value_counts.items()
        if value >= min_sample_size and key is not None
    ]
    unique_values = sorted(unique_values)
    results = {}

    for value_1 in unique_values:

        ingroup_indices = user_table[
            user_table[group_variable] == value_1
        ].index.tolist()
        ingroup_embeddings = [embeddings[i] for i in ingroup_indices]

        if user_level:
            ingroup_event_sequences = [event_sequences[i] for i in ingroup_indices]

            (
                ingroup_split_event_sequences_sample,
                ingroup_user_id_ls,
            ) = get_split_event_sequences_sample(
                ingroup_event_sequences,
                max_len=510,
                sample_size=min(len(ingroup_event_sequences), 1000),
            )

            if method in ["tf", "tf-l2", "tf-idf", "random"]:
                _, ingroup_split_embeddings_sample, _ = feature_engineering(
                    train_sequences=event_sequences,
                    test_sequences=ingroup_split_event_sequences_sample,
                    method=method,
                    mode=mode,
                    max_model_input_size=max_model_input_size,
                    model_path=model_path,
                    bucket_name=bucket_name,
                    tokenizer_path=tokenizer_path,
                )

            else:
                ingroup_split_embeddings_sample, _, _ = feature_engineering(
                    train_sequences=ingroup_split_event_sequences_sample,
                    test_sequences=ingroup_split_event_sequences_sample[0:1],
                    method=method,
                    mode=mode,
                    max_model_input_size=max_model_input_size,
                    model_path=model_path,
                    bucket_name=bucket_name,
                    tokenizer_path=tokenizer_path,
                )

            (
                within_user_cos_sim,
                between_users_cos_sim,
            ) = compute_cosine_similarity_at_user_level(
                ingroup_split_embeddings_sample,
                ingroup_user_id_ls,
                min(min_sample_size, len(ingroup_indices)),
            )

            results[f"{group_variable}: {value_1}"] = [
                round(within_user_cos_sim, 3),
                round(between_users_cos_sim, 3),
                round(within_user_cos_sim - between_users_cos_sim, 3),
            ]

        if group_level:
            for value_2 in unique_values:
                if value_1 == value_2:
                    continue

                outgroup_indices = user_table[
                    user_table[group_variable] == value_2
                ].index.tolist()
                outgroup_embeddings = [embeddings[i] for i in outgroup_indices]
                (
                    within_group_cos_sim,
                    between_groups_cos_sim,
                ) = compute_cosine_similarity_at_group_level(
                    ingroup_embeddings,
                    outgroup_embeddings,
                    min(len(ingroup_indices), len(outgroup_indices), min_sample_size),
                    42,
                )

                results[f"{group_variable}: {value_1} vs {value_2}"] = [
                    round(within_group_cos_sim, 3),
                    round(between_groups_cos_sim, 3),
                    round(within_group_cos_sim - between_groups_cos_sim, 3),
                ]

    return results


def pretty_results(data: dict, headers: list):
    """
    This function prints a dictionary in a nice tabular format.
    """
    data = [[key] + value for key, value in data.items()]
    return tabulate(data, headers)


def report_simple_event_statistics(event_sequences, user_table, group, events):
    """
    This function computes the proportion of each user event within a group.
    """
    event_counts_ls = []
    for i in range(len(event_sequences)):
        event_sequence = event_sequences[i].split(" ")
        event_counts = Counter(event_sequence)
        event_counts_ls.append(event_counts)

    available_events = set(
        [key for dict_item in event_counts_ls for key in dict_item.keys()]
    )
    events = [event for event in events if event in available_events]

    event_counts_df = pd.DataFrame(event_counts_ls).fillna(0)
    event_counts_df["total_n_events"] = event_counts_df.sum(axis=1)
    event_counts_df.loc[
        :, event_counts_df.columns != "total_n_events"
    ] = event_counts_df.loc[:, event_counts_df.columns != "total_n_events"].div(
        event_counts_df["total_n_events"], axis=0
    )

    user_table_with_event_counts = pd.concat([user_table, event_counts_df], axis=1)
    event_proportions_by_group = (
        user_table_with_event_counts.groupby(group)[events].mean().T
    )

    event_proportions_by_group["cv"] = event_proportions_by_group.loc[
        :, event_proportions_by_group.columns != group
    ].std(axis=1) / event_proportions_by_group.loc[
        :, event_proportions_by_group.columns != group
    ].mean(
        axis=1
    )

    return round(event_proportions_by_group.sort_values("cv", ascending=False), 5)


def extract_model_name(string):
    """
    This function extract model name based on a string (e.g., model paths).
    """
    if "deberta-v2" in string:
        return "deberta-v2"
    elif "bigbird" in string:
        return "bigbird"
    elif "gpt2" in string:
        return "gpt2"
    elif "rwkv" in string:
        return "rwkv"
    elif "retnet" in string:
        return "retnet"
    else:
        return "bert"


def read_top_n_predefined_list_of_label_names(filename, n, negative_label=None):
    """
    This function reads the top n predefined list of label names.
    If negative_label is not None, the negative label would also be returned.
    """
    assert n > 0
    with open(filename, "r") as f:
        items = f.read()

    label_names = items.lstrip("\n").rstrip("\n").split("\n")
    assert len(label_names) >= n

    if n > 1:  # for a binary prediction situation
        label_names = label_names[0:n]

    if negative_label is None:
        return label_names
    else:
        return label_names + [negative_label]


def get_n_days_between_two_dates(date1, date2):
    """
    This function computes the number of days betweeen two dates.
    """
    datetime1 = datetime.strptime(date1, "%Y%m%d")
    datetime2 = datetime.strptime(date2, "%Y%m%d")
    difference = (datetime2 - datetime1).days
    return difference


def get_available_tasks(scope: str = "all"):
    """
    This function returns all the downstream tasks we have atm.
    Note that the same function exists in utils_preproccess.py.
    Remember to synchronize the two functions.
    """
    assert scope in ["all", "test", "downstream"]

    all_tasks = [
        "train",
        "test_reported_user_prediction",
        "test_locked_user_prediction",
        "test_account_self_deletion_prediction",
        "test_ad_click_binary_prediction",
        "test_prop_of_ad_clicks_prediction",
        "test_ad_view_time_prediction",
    ]

    if scope == "test":
        all_tasks.remove("train")
    elif scope == "downstream":
        all_tasks.remove("train")
        all_tasks.remove("test_prop_of_ad_clicks_prediction")

    return all_tasks


# Function to upload files to gcs buckets recursively
def upload_files_to_gcs_recursive(bucket_name, local_dir, gcs_dir):
    """
    This function uploads a file or a folder to gcs.
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    for item in os.listdir(local_dir):
        local_path = os.path.join(local_dir, item)
        gcs_path = os.path.join(gcs_dir, item)

        if os.path.isfile(local_path):
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
        elif os.path.isdir(local_path):
            blob = bucket.blob(gcs_path + "/")  # Create a virtual directory in GCS
            blob.upload_from_string("")  # Upload an empty string as a placeholder

            # Recursively upload files inside the subdirectory
            upload_files_to_gcs_recursive(bucket_name, local_path, gcs_path)


def import_tokenizer(
    tokenizer_path: str,
    bucket_name: str,
    max_model_input_size: int,
    padding_side: str = "right",
):
    """
    This function imports a trained tokenizer from gcs.
    """

    local_tokenizer_dir = os.path.join("..", "model_cache", tokenizer_path)

    if not os.path.exists(local_tokenizer_dir):
        logger.warning(
            f"Cache tokenizer to {local_tokenizer_dir}. Please delete it if you are running the code on personal computer."
        )
        download_dir = "/".join(local_tokenizer_dir.strip("/").split("/")[:-1])
        os.makedirs(download_dir, exist_ok=True)
        subprocess.run(
            [
                "gsutil",
                "-m",
                "cp",
                "-r",
                "gs://{}/{}".format(bucket_name, tokenizer_path.rstrip("/")),
                download_dir,
            ],
            check=True,
        )

    if "bigbird" in tokenizer_path:

        tokenizer = BigBirdTokenizer.from_pretrained(
            local_tokenizer_dir,
            model_max_length=max_model_input_size,
            padding_side=padding_side,
            truncation_side="right",
            use_fast=True,
        )

    elif "bert" in tokenizer_path:
        tokenizer = BertTokenizer.from_pretrained(
            local_tokenizer_dir,
            model_max_length=max_model_input_size,
            padding_side="right",
            truncation_side="right",
        )

    else:
        raise Exception(
            "Either bigbird or bert needs to be in the path for the function to interpret what type of tokenizer object needs to be instantiated."
        )

    return tokenizer


def import_trained_model(
    method: str,
    model_path: str,
    bucket_name: str,
    fill_mask: bool = False,
    tokenizer: object = None,
):
    """
    This function imports a trained transformer-based model from gcs.
    """
    # download the model from GCP if not exists locally
    local_model_dir = os.path.join("..", "model_cache", model_path)

    if not os.path.exists(local_model_dir):
        logger.warning(
            f"Cache model to {local_model_dir}. Please delete it if you are running the code on personal computer."
        )
        download_dir = "/".join(local_model_dir.strip("/").split("/")[:-1])
        os.makedirs(download_dir, exist_ok=True)
        subprocess.run(
            [
                "gsutil",
                "-m",
                "cp",
                "-r",
                "gs://{}/{}".format(bucket_name, model_path.rstrip("/")),
                download_dir,
            ],
            check=True,
        )

    if fill_mask:
        fill_mask_model = pipeline(
            task="fill-mask", model=local_model_dir, tokenizer=tokenizer
        )
        return fill_mask_model

    else:
        CLASS_MAP = {
            "bert": EventBertForMaskedLM,
            "bigbird": BigBirdForMaskedLM,
            "deberta-v2": DebertaV2ForMaskedLM,
            "retnet": RetNetModelWithLMHead,
            "gpt2": GPT2LMHeadModel,
        }

        if method in CLASS_MAP:
            model = CLASS_MAP[method].from_pretrained(
                local_model_dir, output_hidden_states=True
            )
        else:
            raise Exception(
                "Currently, only bert, bigbird and deberta-v2 are supported."
            )
        return model


def import_sgns_model(
    model_path: str = "artifacts/sgns/", bucket_name: str = "bic-test"
):
    """
    This function imports a trained sgns model from gcs.
    """

    folder_name = model_path.rstrip("/").split("/")[-1]
    temp_dir = TemporaryDirectory()
    local_model_dir = temp_dir.name
    subprocess.run(
        [
            "gsutil",
            "-m",
            "cp",
            "-r",
            "gs://{}/{}".format(bucket_name, model_path.rstrip("/")),
            local_model_dir,
        ],
        check=True,
    )
    model_path_temp = f"{local_model_dir}/{folder_name}"
    print(f"{model_path_temp}/sgns.model")
    sgns_model = Word2Vec.load(f"{model_path_temp}/sgns.model")
    return sgns_model


def strip_and_split_by_space(string):
    """
    This function strips a string and splits it into a list.
    """
    return string.strip().split(" ")


def read_avro_file_from_gcs(
    bucket_name: str,
    prefix: str,
    max_files: int,
    length_to_keep: int = -1,
    min_length: int = 10,
):
    """
    This function reads avro files (e.g., user tables, event tables, user sequences) stored in gcs.
    """

    storage_client = storage.Client()
    blobs = list(
        storage_client.list_blobs(
            bucket_or_name=bucket_name,
            prefix=prefix,
        )
    )

    avro_files = [blob for blob in blobs if blob.name.endswith(".avro")]

    if len(avro_files) == 0:
        print(f"No avro file matching the specified prefix: {prefix}.")
        return None

    else:
        data = []
        local_file_paths = []
        temp_dir = TemporaryDirectory()

        for blob in tqdm(avro_files[: min(max_files, len(avro_files))]):
            local_file_path = f"{temp_dir.name}/" + blob.name.split("/")[-1]
            local_file_paths.append(local_file_path)
            local_file = blob.download_to_filename(local_file_path)
            find_error = False
            data_blob = []

            with open(local_file_path, "rb") as avro_file:
                avro_reader = fastavro.reader(avro_file)

                # Iterate over the records in the AVRO file
                for i, record in enumerate(avro_reader):
                    # in the pre-training stage, do manual truncation of the inputs to save RAM
                    if "event_sequence" in record:
                        if len(record["time_sequence"]) < min_length:
                            continue

                        if len(record["time_sequence"]) != len(
                            record["event_sequence"]
                        ):
                            logger.warning(
                                f"Find error in file {blob.name}. The time and event sequences have different lengths."
                            )
                            find_error = True
                            break

                        if length_to_keep > 0:
                            record["event_sequence"] = record["event_sequence"][
                                :length_to_keep
                            ]
                            record["time_sequence"] = record["time_sequence"][
                                :length_to_keep
                            ]
                    data_blob.append(record)

            if not find_error:
                data.extend(data_blob)

        return data


def truncate_event_and_time_sequence(
    event_sequences,
    time_sequences,
    get_random_segment: bool = False,
    length_to_keep: int = 512,
):

    processed_events = []
    processed_times = []

    # we always skip the first token, since sometimes there is a false positive [New Session] token
    for e, t in zip(event_sequences, time_sequences):
        e = e.split()
        if get_random_segment and len(e) - length_to_keep > 1:
            start = random.randint(1, len(e) - length_to_keep)
        else:
            start = 1

        processed_events.append(" ".join(e[start : start + length_to_keep]))
        processed_times.append(t[start : start + length_to_keep])

    return processed_events, processed_times
