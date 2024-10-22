import gc
import logging
import os
import pickle
import random
import subprocess
import sys
import traceback
from math import ceil

import numpy as np
import scipy
import torch
import typer
import wandb
from data_collator import (
    DataCollatorForMLMAndSameUserPrediction,
    extract_time_from_strings,
    generate_fep_labels,
)
from datasets import Dataset, DatasetDict, load_from_disk
from google.cloud import storage
from model import EventBertForMaskedLM
from modeling_retnet import (
    RETNET_FFN_RATIO,
    RETNET_QK_RATIO,
    RETNET_V_RATIO,
    RetNetConfig,
    RetNetModelWithLMHead,
)
from tqdm import tqdm
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    BigBirdConfig,
    BigBirdForMaskedLM,
    BigBirdTokenizer,
    DataCollatorForLanguageModeling,
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    EarlyStoppingCallback,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    RwkvConfig,
    RwkvForCausalLM,
    Trainer,
    TrainingArguments,
)
from utils import (
    get_available_tasks,
    import_tokenizer,
    make_path,
    read_user_sequences,
    truncate_event_and_time_sequence,
    upload_files_to_gcs_recursive,
)

assert torch.cuda.is_available()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger()

VALID_MODEL_LIST = [
    "bert",
    "bigbird",
    "deberta-v2",
    "gpt2",
    "eventbert",
    "llama",
    "rwkv",
    "retnet",
]
VALID_OBJECTIVE_LIST = [
    "MLM",
    "SUP",
    "Both",
    "CLM",
    "FEP",
    "FEP_CLM",
    "FEP_CLM_SUP",
    "FEP_SUP",
]

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    wandb_project_name: str = typer.Option(
        "user-model-training-and-tuning", help="keeping track in WANDB"
    ),
    task: str = typer.Option(
        "train", help="use the train data or data from a downstream task"
    ),
    model: str = typer.Option("bert", help=f"choose user model: {VALID_MODEL_LIST}"),
    from_pretrained: str = typer.Option(
        None,
        help="The directory to the pre-trained checkpoint to start from. Set as None to train from scratch.",
    ),
    training_objective: str = typer.Option(
        "MLM", help=f"training objective, choose from {VALID_OBJECTIVE_LIST}"
    ),
    bucket_name: str = typer.Option("bic-test", help="bic-test or umap-user-model."),
    prefix: str = typer.Option(
        "data/user_event_and_time_sequences",
        help="prefix for event sequence file names on GCS",
    ),
    sample_n: int = typer.Option(100, help="training size"),
    features_start_date: str = typer.Option(
        "20230313",
        help="start date of sampling frame for user events, in the format of %Y%m%d",
    ),
    features_end_date: str = typer.Option(
        "20230313",
        help="end date of sampling frame for user events, in the format of %Y%m%d",
    ),
    labels_date: str = typer.Option(
        None, help="date of downstream behavior, in the format of %Y%m%d"
    ),
    sc_analytics_tables: str = typer.Option(
        "app",
        help="""sc-analytics tables to use for generating user events, either a single value string or a string of table suffixes separated by underscore; \
        possible values: app, chat, cognac, ops_feed, ops_map, ops_memories, ops_page, ops_story, page, snap, story, user""",
    ),
    sample_table_option: str = typer.Option(
        "sample",
        help="""whether to use sample event tables or non-sample/population event tables; choose between sample and population""",
    ),
    user_filter_option: str = typer.Option(
        "unfiltered",
        help="""whether to impose any kind of filtering on the users (e.g., demographics, activivity level); choose between filtered and unfiltered""",
    ),
    sequence_type: str = typer.Option(
        "shortlist",
        help="choose among: shortlist, shortlist_without_page_page_view, shortlist_without_repetition, shortlist_with_counts. Check ./event_names.txt for the shortlist.",
    ),
    tokenizer_name: str = typer.Option(
        "bigbird", help="choose between bigbird and bert"
    ),
    tokenizer_model_type: str = typer.Option(
        "word",
        help="choose among: BPE, unigram, char, word; see SentencePiece documentation for details",
    ),
    tokenizer_path: str = typer.Option(
        "artifacts/tokenizers/bigbird_word",
        help="path to self-trained tokenizer in gcs bucket location",
    ),
    model_output_path: str = typer.Option(
        "../artifacts/",
        help="local relative path to saving trained models, which will also be automatically saved to the corresponding gcs bucket using the same relative path",
    ),
    test_size: float = typer.Option(
        0.10, help="the proportion of data reserved for validation"
    ),
    max_model_input_size: int = typer.Option(
        512,
        help="maximum model input size; 512 for BERT, 4096 for BigBird, 24,528 for DeBERTa-v2",
    ),
    min_model_input_size: int = typer.Option(
        50,
        help="minimum model input size",
    ),
    max_files: int = typer.Option(
        1000, help="maximum number of event sequence files (on GCS) to use"
    ),
    n_iter: int = typer.Option(
        1,
        help="number of samples of hyperparameter values; if set to 1, then default values would be used",
    ),
    user_segments_gap: int = typer.Option(
        50,
        help="The minimum number of event between the two extracted user event segments.",
    ),
    contrastive_type: str = typer.Option(
        "Orig",
        help="Two type of SimCLR contrastive loss, namely 'Orig' and 'HardNeg', while the second one put a higher weights on the samples that are hard to classify",
    ),
    contrastive_embedding: str = typer.Option(
        "token",
        help="What to use for contrastive learning. 'token' refers to the average of token embeddings. 'state' refers to the last state (kv_cache) of RetNet",
    ),
    contrastive_temperature: float = typer.Option(
        0.07,
        help="The temperature hyperparameter used in the contrastive loss.",
    ),
    contrastive_head_size: int = typer.Option(
        128,
        help="The dimension of the contrastive head used for same user prediction.",
    ),
    per_device_batch_size: int = typer.Option(16, help="batch size per device"),
    learning_rate: float = typer.Option(
        4e-5,
        help="Learning rate for the optimizer.",
    ),
    warmup_ratio: float = typer.Option(
        0.06,
        help="The percentage of warm up steps. Learning rate will linearly increase to the target in the warm up steps, and linearly decrease to 0 afterwards.",
    ),
    gradient_accumulation_steps: int = typer.Option(
        2,
        help="Accumulate gradient for multiple steps to adjust the global batch size.",
    ),
    eval_steps: int = typer.Option(
        None,
        help="Model will be evaluated every eval_steps. Set as None for a automatically decided value.",
    ),
    logging_steps: int = typer.Option(
        100,
        help="The frequency of print training loss during the training process.",
    ),
    run_name: str = typer.Option(
        "test",
        help="run name used in wandb to track results.",
    ),
    num_train_epochs: int = typer.Option(10, help="Number of training epochs."),
    weight_decay: float = typer.Option(
        0.01,
        help="Weight decay used to limits the scale of each weight parameter. Same as add L2-loss.",
    ),
    use_alibi: bool = typer.Option(
        False,
        help="If use_alibi is activate, we will get rid of the positional embedding and use Attention with Linear Biases (ALiBi) instead.",
    ),
    use_time: bool = typer.Option(
        False,
        help="If use_time is activate, will concatenated time-aware embedding with word embeddings to serve as input representations.",
    ),
    time_embedding_size: str = typer.Option(
        "4 16 16",
        help="The number of dimension used for holiday, month, and weekday embeddings.",
    ),
    alibi_starting_size: int = typer.Option(
        512, help="The initial size used for Attention with Linear Biases (ALiBi)."
    ),
    num_heads: int = typer.Option(
        12, help="The number of attention / retention heads in the model."
    ),
    reverse_sequence: bool = typer.Option(
        False,
        help="Use to reverse the order of event and time sequence to ensure they are in the correct order since the files store on GCS stores files in a time-reversed order.",
    ),
    fep_window_size: int = typer.Option(
        100,
        help="Number of future events used for the Future Event Prediction (FEP) training target.",
    ),
    fep_context_length: int = typer.Option(
        0,
        help="If the number if set as a positive number, then we will only not compute Future Event Prediction (FEP) loss \
                  on the first fep_context_length tokens. This avoid training the model to predict \
                  future event on very beginning tokens where context is limited.",
    ),
    fep_loss_frequency: int = typer.Option(
        1,
        help="It is common that continuous tokens can have the same future event prediction labels. \
              Therefore, instead of calculate FEP loss on every single token, we may also calculate loss \
              every a few tokens. This argument defines the frequency of calculating loss on an input sequence. \
              If set as 1, loss is calculated on every token. \
              If set as k (k > 1), loss is calculated on the k-th 2k-th 3k-th, .., nk-th tokens",
    ),
    fep_event_of_interests: str = typer.Option(
        "event_of_interests_29",
        help="The name of the file contains the list of events of interests.",
    ),
    get_random_segment: bool = typer.Option(
        False,
        help="Whether to generate training sequences from the beginning of each user sequence. \
              If set as True, will randomly select a segment from each user sequence as training data. \
              Otherwise, will always select segment from the beginning of each user sequence",
    ),
):

    assert model in VALID_MODEL_LIST
    assert training_objective in VALID_OBJECTIVE_LIST
    assert task in get_available_tasks(scope="all")
    assert (
        len(time_embedding_size.split()) == 3
    ), "time_embedding_size should be a string consists of 3 integer numbers separated by space, where each value respectively represents the embedding dimension of holiday, month, and weekday"

    if training_objective in ["SUP", "Both"]:
        assert model in ["eventbert", "retnet"]

    if contrastive_embedding == "state":
        assert (
            model == "retnet"
        ), "Only RetNet supports contrastive learning with last state (kv_cache)."

    assert (tokenizer_name == "bigbird" and tokenizer_model_type == "word") or (
        tokenizer_name == "bert" and tokenizer_model_type == "wordpiece"
    )
    assert sample_table_option in ["sample", "population"]
    assert user_filter_option in ["filtered", "unfiltered"]
    sc_analytics_table_list = [
        "all",
        "app",
        "chat",
        "cognac",
        "ops_feed",
        "ops_map",
        "ops_memories",
        "ops_page",
        "ops_story",
        "page",
        "snap",
        "story",
        "user",
    ]
    assert all(
        table_name in sc_analytics_table_list
        for table_name in sc_analytics_tables.split("_")
    )

    assert 0 <= fep_context_length < max_model_input_size
    assert (
        fep_context_length == 0 or fep_loss_frequency == 1
    ), "customize fep_context_length and fep_loss_frequency at the same time is not supported."

    model_output_path = model_output_path + f"{model}/"
    make_path(model_output_path)

    commit_id = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    )

    num_of_gpus = torch.cuda.device_count()

    # set up hyperparameter tuning with wandb
    logger.info("SETTING UP WANBD AND SWEEP.")

    sweep_config = {"method": "random"}

    # note that these hyperparameter values are specific to bert
    # should adapt them when a different model is used
    if n_iter == 1:  # values in the original BERT paper for pretraining
        parameters_dict = {
            "hidden_size": {"values": [768]},
            "num_attention_heads": {"values": [12]},
            "num_hidden_layers": {"values": [12]},
            "weight_decay": {"values": [weight_decay]},  # [1e-2] for BERT default
            "learning_rate": {"values": [learning_rate]},  # [1e-4] for BERT default
            "gradient_accumulation_steps": {
                "values": [gradient_accumulation_steps]
            },  # assuming 8 GPUs
            "num_heads": {"values": [num_heads]},  # [12] for BERT default
        }
    else:
        parameters_dict = {
            "hidden_size": {"values": [192, 384, 768]},
            "num_attention_heads": {"values": [3, 6, 12, 24]},
            "num_hidden_layers": {"values": [3, 6, 12, 24]},
            "weight_decay": {"values": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 0]},
            "learning_rate": {"values": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]},
            "gradient_accumulation_steps": {"values": [1, 2, 4, 8]},  # assuming 8 GPUs
        }

    sweep_config["parameters"] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project=wandb_project_name)

    # read user event sequences
    logger.info("READING USER EVENT SEQUENCES.")
    if labels_date is None:
        labels_date = features_end_date

    if model in ["gpt2", "rwkv", "retnet"]:
        mlm = False
    else:
        mlm = training_objective in ["MLM", "Both"]

    if model in ["bert", "eventbert"]:
        clm = False
    else:
        clm = training_objective in ["CLM", "Both", "FEP_CLM", "FEP_CLM_SUP"]

    # whether each training objective is activated
    # TODO: find a more elegant way to handle different combiantion of training targets
    same_user_prediction = training_objective in [
        "SUP",
        "Both",
        "FEP_SUP",
        "FEP_CLM_SUP",
    ]
    future_event_prediction = training_objective in [
        "FEP",
        "FEP_CLM",
        "FEP_SUP",
        "FEP_CLM_SUP",
    ]

    if same_user_prediction:
        assert model in ["eventbert", "retnet"]
    elif future_event_prediction:
        assert model in ["retnet"]
        assert (
            fep_window_size >= user_segments_gap
        ), "Please make sure the fep_window_size is smaller than user_segments_gap, otherwise the model will be able to observe future events of the first user segment from the second segment."

    logger.info("LOADING TOKENIZER.")
    tokenizer = import_tokenizer(
        tokenizer_path=tokenizer_path,
        bucket_name=bucket_name,
        max_model_input_size=max_model_input_size,
    )

    SPECIAL_TOKEN_IDS = [
        tokenizer.convert_tokens_to_ids(k)
        for k in tokenizer.special_tokens_map.values()
    ]

    logger.info("LOADING EVENTS OF INTERESTS FOR FUTURE EVENT PREDICTION.")
    with open(fep_event_of_interests + ".txt", "r") as f:
        event_lists = f.read().splitlines()
        events_of_interests = [e.split(",")[0] for e in event_lists]
        eid2pos = {
            int(e): i
            for i, e in enumerate(events_of_interests)
            if int(e) not in SPECIAL_TOKEN_IDS
        }
        events_of_interests = set(events_of_interests)

    # load data from load cache or download and process from GCP
    data_save_dir = f"../cache/{prefix}_{task}_in_{sc_analytics_tables}_{sample_n}_from_{features_start_date}_to_{features_end_date}_on_{labels_date}_{sample_table_option}_{user_filter_option}_{max_files}_{min_model_input_size}_{max_model_input_size}_{same_user_prediction}_{reverse_sequence}_{future_event_prediction}_{fep_event_of_interests}_{get_random_segment}"

    # load from cache
    if os.path.exists(data_save_dir):
        dataset = DatasetDict()
        dataset["train"] = load_from_disk(os.path.join(data_save_dir, "train"))
        dataset["test"] = load_from_disk(os.path.join(data_save_dir, "test"))

    # download and process
    else:
        """
        manully truncate the sequences to reduce memory usage and speed up the tokenization
        when same user prediction is activated,
           we take 2 segments with maximum of "max_model_input_size" as training data.
           the two segments should have a gap of "user_segments_gap"
        add one since the first token in the sequence is always the ghost_user_id and we will skip it
        """
        length_to_keep = 20
        if same_user_prediction and future_event_prediction:
            length_to_keep += 2 * max_model_input_size + 2 * fep_window_size
        elif same_user_prediction:
            length_to_keep += 2 * max_model_input_size + user_segments_gap
        elif future_event_prediction:
            length_to_keep += max_model_input_size + fep_window_size
        else:
            length_to_keep += max_model_input_size

        sequence_cache_paths = [
            f"../data/sequences_train_{max_files}.pkl",
            f"../data/sequences_test_{max_files}.pkl",
        ]
        if os.path.exists(sequence_cache_paths[0]) and os.path.exists(
            sequence_cache_paths[1]
        ):
            logger.info(
                f"Load sequence locally from {sequence_cache_paths[0]} and {sequence_cache_paths[1]}"
            )
            with open(sequence_cache_paths[0], "rb") as f:
                event_sequences_train, time_sequences_train = pickle.load(f)
            with open(sequence_cache_paths[1], "rb") as f:
                event_sequences_test, time_sequences_test = pickle.load(f)

            logger.info(
                f"Pre-processing: truncate the event and time sequence for fast tokenization."
            )
            (
                event_sequences_train,
                time_sequences_train,
            ) = truncate_event_and_time_sequence(
                event_sequences_train,
                time_sequences_train,
                get_random_segment,
                length_to_keep,
            )

            (
                event_sequences_test,
                time_sequences_test,
            ) = truncate_event_and_time_sequence(
                event_sequences_test,
                time_sequences_test,
                get_random_segment,
                length_to_keep,
            )

        else:
            logger.info(f"Downloading sequences from GCS.")
            event_sequences, time_sequences, _ = read_user_sequences(
                bucket_name=bucket_name,
                prefix=prefix,
                task=task,
                features_start_date=features_start_date,
                features_end_date=features_end_date,
                labels_date=labels_date,
                sample_n=sample_n,
                sample_table_option=sample_table_option,
                user_filter_option=user_filter_option,
                sc_analytics_tables=sc_analytics_tables,
                sequence_type=sequence_type,
                max_files=max_files,
                length_to_keep=-1,
                use_time=use_time,
                reverse_sequence=reverse_sequence,
            )

            temp = list(zip(event_sequences, time_sequences))
            random.shuffle(temp)
            event_sequences, time_sequences = zip(*temp)
            # res1 and res2 come out as tuples, and so must be converted to lists.
            event_sequences, time_sequences = list(event_sequences), list(
                time_sequences
            )
            num_train = int(len(event_sequences) * 0.95)

            event_sequences_train, time_sequences_train = (
                event_sequences[:num_train],
                time_sequences[:num_train],
            )
            event_sequences_test, time_sequences_test = (
                event_sequences[num_train:],
                time_sequences[num_train:],
            )

            os.makedirs("../data", exist_ok=True)
            with open(sequence_cache_paths[0], "wb") as f:
                pickle.dump((event_sequences_train, time_sequences_train), f)
            with open(sequence_cache_paths[1], "wb") as f:
                pickle.dump((event_sequences_test, time_sequences_test), f)
            logger.info(
                f"Save sequence locally at {sequence_cache_paths[0]} and {sequence_cache_paths[1]}"
            )

        # import custom tokenizer
        logger.info("LOADING TOKENIZER.")

        # preprocessing training data
        logger.info("PREPROCESSING TRAINING DATA.")
        dataset = DatasetDict()
        dataset["train"] = Dataset.from_dict(
            {"text": event_sequences_train, "time": time_sequences_train}
        )
        dataset["test"] = Dataset.from_dict(
            {"text": event_sequences_test, "time": time_sequences_test}
        )

        if same_user_prediction:
            # only activate fep_window_size or user_segments_gap to simplify following calculation
            if not future_event_prediction:
                fep_window_size = 0
            else:
                user_segments_gap = 0
            length_to_filter = (
                2 * min_model_input_size + user_segments_gap + 2 * fep_window_size
            )
            # filter out sequences shorter than "user_segments_gap + 2 * min_model_input_size"
            dataset = dataset.filter(
                lambda example: len(example["text"].split()) > length_to_filter
            )

            def _preprocess_for_same_user_prediction(raw_sample: dict) -> dict:
                text_sequence = raw_sample["text"].split(" ")
                time_sequence = raw_sample["time"]
                seq_len = len(text_sequence)
                # extract two segments from original_sequence
                # set segment length randomly to prevent the model from calculating embedding based on sequence length
                seg1_start = 0
                seg1_len = random.randrange(
                    min_model_input_size - 1,
                    min(
                        max_model_input_size - 2,
                        seq_len
                        - min_model_input_size
                        - user_segments_gap
                        - 2 * fep_window_size,
                    ),
                )
                # max_model_input_size - 2 leave spaces for [CLS] and [SEP] tokens

                seg2_start = seg1_len + user_segments_gap + fep_window_size
                seg2_len = min(
                    max_model_input_size - 2, seq_len - seg2_start - fep_window_size
                )

                text_segment1 = " ".join(
                    text_sequence[seg1_start : seg1_start + seg1_len]
                )
                text_segment2 = " ".join(
                    text_sequence[seg2_start : seg2_start + seg2_len]
                )

                tokenized_texts_1 = (
                    tokenizer(
                        text_segment1,
                        padding="max_length",
                        truncation=True,
                        max_length=max_model_input_size + fep_window_size,
                    )["input_ids"],
                )
                tokenized_texts_2 = (
                    tokenizer(
                        text_segment2,
                        padding="max_length",
                        truncation=True,
                        max_length=max_model_input_size + fep_window_size,
                    )["input_ids"],
                )
                if future_event_prediction:
                    fep_labels_and_ids_1 = [
                        generate_fep_labels(
                            t, eid2pos, fep_window_size, tokenizer.pad_token_id
                        )
                        for t in tokenized_texts_1
                    ]
                    fep_labels_and_ids_2 = [
                        generate_fep_labels(
                            t, eid2pos, fep_window_size, tokenizer.pad_token_id
                        )
                        for t in tokenized_texts_2
                    ]

                    # print(fep_labels_and_ids_1[1][:max_model_input_size].shape)

                    sample = {
                        "fep_labels_1": [l[0] for l in fep_labels_and_ids_1],
                        "fep_labels_2": [l[0] for l in fep_labels_and_ids_2],
                        "input_ids_1": [
                            l[1][:max_model_input_size] for l in fep_labels_and_ids_1
                        ],
                        "input_ids_2": [
                            l[1][:max_model_input_size] for l in fep_labels_and_ids_2
                        ],
                    }
                else:
                    sample = {
                        "input_ids_1": tokenized_texts_1,
                        "input_ids_2": tokenized_texts_2,
                    }

                # extract time information from timestamp strings
                if use_time:
                    sample["time_1"] = time_sequence[seg1_start : seg1_start + seg1_len]
                    sample["time_2"] = time_sequence[seg2_start : seg2_start + seg2_len]

                return sample

            dataset = dataset.map(
                _preprocess_for_same_user_prediction,
                batched=False,  # it runs slow and acts unexpected with this preprocessing function when batched=True
                remove_columns=["text", "time"],
                num_proc=16,
            )

        else:
            if future_event_prediction:
                dataset = dataset.filter(
                    lambda example: len(example["text"].split())
                    > min_model_input_size + fep_window_size
                )

            def _preprocess_for_language_modeling(raw_sample: dict) -> dict:
                tokenized_texts = (
                    tokenizer(
                        raw_sample["text"],
                        padding="max_length",
                        truncation=True,
                        max_length=max_model_input_size + fep_window_size
                        if future_event_prediction
                        else max_model_input_size,
                    )["input_ids"],
                )

                if future_event_prediction:
                    fep_labels_and_ids = [
                        generate_fep_labels(
                            t, eid2pos, fep_window_size, tokenizer.pad_token_id
                        )
                        for t in tokenized_texts
                    ]

                    sample = {
                        "fep_labels": [l[0] for l in fep_labels_and_ids],
                        "input_ids": [
                            l[1][:max_model_input_size] for l in fep_labels_and_ids
                        ],
                    }
                else:
                    sample = {"input_ids": tokenized_texts}

                if use_time:
                    sample["time"] = raw_sample["time"]

                return sample

            dataset = dataset.map(
                _preprocess_for_language_modeling,
                batched=False,
                remove_columns=["text"],
                num_proc=16,
            )

        # cache processed dataset for future usage
        logger.warning(
            f"Cache processed data to {data_save_dir}. Please delete the local cache if you are running this code on personal computer."
        )
        dataset["train"].save_to_disk(os.path.join(data_save_dir, "train"))
        dataset["test"].save_to_disk(os.path.join(data_save_dir, "test"))

    data_collator = DataCollatorForMLMAndSameUserPrediction(
        tokenizer=tokenizer,
        mlm=mlm,
        clm=clm,
        same_user_prediction=same_user_prediction,
        future_event_prediction=future_event_prediction,
        num_fep_events=len(events_of_interests),
        use_time=use_time,
        mlm_probability=0.15,
        pad_to_multiple_of=None,
    )

    dataset.set_format("torch")

    # set up model hyperparameter tuning
    if n_iter == 1:
        logger.info("INITIATE MODEL TRAINING WITHOUT HYPERPARAMETER TUNING.")
    else:
        logger.info("INITIATE MODEL TRAINING WITH HYPERPARAMETER TUNING.")

    def train():
        # global variables: commit_id, training_objective, sample_n, features_start_date,
        # features_end_date, test_size, tokenizer_path, max_model_input_size,
        # max_files, model_output_path,
        torch.cuda.empty_cache()
        gc.collect()

        with wandb.init():
            wandb.run.name = run_name
            wandb.log(
                {
                    "model": model,
                    "commit_id": commit_id,
                    "training_objective": training_objective,
                    "sample_n": sample_n,
                    "features_start_date": features_start_date,
                    "features_end_date": features_end_date,
                    "test_size": test_size,
                    "tokenizer_path": tokenizer_path,
                    "max_model_input_size": max_model_input_size,
                    "max_files": max_files,
                    "n_gpus": num_of_gpus,
                    "per_device_batch_size": per_device_batch_size,
                    "sequence_type": sequence_type,
                    "actual_sample_n": len(dataset["train"]) + len(dataset["test"]),
                }
            )

            run_id = wandb.run.id

            step_size = (
                ceil(
                    len(dataset["train"])
                    / wandb.config.gradient_accumulation_steps
                    / per_device_batch_size
                    / num_of_gpus
                    / 10
                )
                if eval_steps == None
                else eval_steps
            )

            wandb.log({"step_size": step_size})

            def model_init():
                # global variables: tokenizer, max_model_input_size, model
                if model == "bert":
                    config = BertConfig(
                        vocab_size=len(tokenizer),
                        hidden_size=wandb.config.hidden_size,  # default 768
                        num_attention_heads=wandb.config.num_attention_heads,  # default 12
                        num_hidden_layers=wandb.config.num_hidden_layers,  # default 12
                        type_vocab_size=2,  # default 2
                        max_position_embeddings=max_model_input_size,
                    )
                    return BertForMaskedLM(config=config)

                elif model == "rwkv":
                    config = RwkvConfig(
                        vocab_size=len(tokenizer),
                        hidden_size=wandb.config.hidden_size,  # default 768
                        num_attention_heads=wandb.config.num_attention_heads,  # default 12
                        num_hidden_layers=wandb.config.num_hidden_layers,  # default 12
                        intermediate_size=wandb.config.hidden_size * 4,  # default 3072
                        context_length=1024,
                    )
                    return RwkvForCausalLM(config=config)

                elif model == "retnet":
                    assert wandb.config.hidden_size % wandb.config.num_heads == 0
                    retnet_config = {
                        "training_objective": training_objective,
                        "num_fep_events": len(events_of_interests)
                        if future_event_prediction
                        else 0,
                        "fep_context_length": fep_context_length,
                        "fep_loss_frequency": fep_loss_frequency,
                        "contrastive_type": contrastive_type,
                        "contrastive_temperature": contrastive_temperature,
                        "contrastive_head_size": contrastive_head_size,
                        "contrastive_embedding": contrastive_embedding,
                    }

                    config = RetNetConfig(
                        vocab_size=len(tokenizer),
                        hidden_size=wandb.config.hidden_size,  # default 768
                        num_heads=wandb.config.num_heads,  # default 3
                        num_layers=wandb.config.num_hidden_layers,  # default 12
                        qk_dim=RETNET_QK_RATIO
                        * wandb.config.hidden_size,  # default 768
                        v_dim=RETNET_V_RATIO * wandb.config.hidden_size,  # default 768
                        ffn_proj_size=RETNET_FFN_RATIO
                        * wandb.config.hidden_size,  # default 3072
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    config.update(retnet_config)
                    return RetNetModelWithLMHead(config=config)

                elif model == "eventbert":
                    time_embedding_sizes = [
                        int(size) for size in time_embedding_size.split()
                    ]
                    eventbert_config = {
                        "training_objective": training_objective,
                        "use_alibi": use_alibi,
                        "use_time": use_time,
                        "alibi_starting_size": alibi_starting_size,
                        "contrastive_type": contrastive_type,
                        "contrastive_temperature": contrastive_temperature,
                        "contrastive_head_size": contrastive_head_size,
                        "time_embedding_size": {
                            "holiday": time_embedding_sizes[0],
                            "month": time_embedding_sizes[1],
                            "weekday": time_embedding_sizes[2],
                            "hour_minute": 2,  # use cosine and sine embedding
                        },
                    }

                    if from_pretrained is None:
                        config = BertConfig(
                            vocab_size=len(tokenizer),
                            hidden_size=wandb.config.hidden_size,  # default 768
                            num_attention_heads=wandb.config.num_attention_heads,  # default 12
                            num_hidden_layers=wandb.config.num_hidden_layers,  # default 12
                            type_vocab_size=2,  # default 2
                            max_position_embeddings=max_model_input_size,
                        )
                        config.update(eventbert_config)
                        return EventBertForMaskedLM(config=config)
                    else:
                        config = BertConfig.from_pretrained(
                            pretrained_model_name_or_path=from_pretrained
                        )
                        config.update(eventbert_config)
                        return EventBertForMaskedLM.from_pretrained(
                            pretrained_model_name_or_path=from_pretrained, config=config
                        )

                elif model == "bigbird":
                    config = BigBirdConfig(
                        vocab_size=len(tokenizer),
                        hidden_size=192,  # default 768
                        num_attention_heads=3,  # default 12
                        num_hidden_layers=3,  # default 12
                        type_vocab_size=2,  # default 2
                        block_size=64,  # default 64
                        num_random_blocks=3,  # default 3
                        attention_type="block_sparse",  # default block_sparse
                        unk_token_id=0,
                        pad_token_id=tokenizer("<pad>")["input_ids"][1],
                        bos_token_id=tokenizer("")["input_ids"][0],
                        eos_token_id=tokenizer("")["input_ids"][1],
                        sep_token_id=tokenizer("[SEP]")["input_ids"][1],
                        cls_token_id=tokenizer("[CLS]")["input_ids"][1],
                        msk_token_id=tokenizer("[MASK]")["input_ids"][1],
                        max_position_embeddings=max_model_input_size,
                    )
                    return BigBirdForMaskedLM(config=config)

                elif model == "deberta-v2":
                    config = DebertaV2Config(
                        vocab_size=len(tokenizer),
                        hidden_size=wandb.config.hidden_size,  # default 768
                        num_attention_heads=wandb.config.num_attention_heads,  # default 12
                        num_hidden_layers=wandb.config.num_hidden_layers,  # default 12
                        intermediate_size=3072,  # default 3072
                        type_vocab_size=2,  # default 2
                        max_position_embeddings=max_model_input_size,
                        max_relative_positions=max_model_input_size,
                        pad_token_id=tokenizer("<pad>")["input_ids"][1],
                        position_biased_input=True,
                    )
                    return DebertaV2ForMaskedLM(config=config)

                elif model == "gpt2":
                    config = GPT2Config(
                        vocab_size=len(tokenizer),
                        n_positions=max_model_input_size,
                        n_embd=wandb.config.hidden_size,  # default 768
                        n_layer=wandb.config.num_hidden_layers,  # default 12
                        n_head=wandb.config.num_attention_heads,  # default 12
                        bos_token_id=tokenizer("")["input_ids"][0],
                        eos_token_id=tokenizer("")["input_ids"][1],
                    )
                    return GPT2LMHeadModel(config=config)
                
                elif model == "llama":
                    config = LlamaConfig(
                        vocab_size=len(tokenizer),
                        hidden_size=wandb.config.hidden_size,  # default 768
                        intermediate_size=4*wandb.config.hidden_size,  # default 3072
                        num_hidden_layers=wandb.config.num_hidden_layers,  # default 12
                        num_attention_heads=wandb.config.num_attention_heads,  # default 12
                        num_key_value_heads=wandb.config.num_attention_heads,  # default 12
                        bos_token_id=tokenizer("")["input_ids"][0],
                        eos_token_id=tokenizer("")["input_ids"][1],
                        max_position_embeddings=4096,
                    )
                    return LlamaForCausalLM(config=config)

            label_names = None
            if model == "retnet" and not clm:
                label_names = []
            elif model == "eventbert" and not mlm:
                label_names = []

            training_args = TrainingArguments(
                output_dir=f"{model_output_path}{run_id}",
                report_to="wandb",  # Turn on Weights & Biases logging
                per_device_train_batch_size=per_device_batch_size,
                per_device_eval_batch_size=per_device_batch_size,
                gradient_accumulation_steps=wandb.config.gradient_accumulation_steps,
                weight_decay=wandb.config.weight_decay,
                learning_rate=wandb.config.learning_rate,
                # gradient_checkpointing=True,
                dataloader_drop_last=True,
                evaluation_strategy="steps",
                eval_steps=step_size,
                eval_delay=step_size,
                eval_accumulation_steps=wandb.config.gradient_accumulation_steps,
                # the help the Huggingface trainer to correctly report evaluation loss with no language modeling loss is presented
                label_names=label_names,
                logging_strategy="steps",
                logging_steps=logging_steps,
                logging_dir="../artifacts/model" + run_id,
                log_on_each_node=True,
                log_level="info",
                save_total_limit=3,
                save_strategy="steps",
                save_steps=step_size,
                num_train_epochs=num_train_epochs,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                remove_unused_columns=False,
                warmup_ratio=warmup_ratio,
                torch_compile=True,
                disable_tqdm=False,
                fp16=True,
                seed=42,
                data_seed=42,
            )

            # add try-exception for better debugging.
            # when wandb is activated, sometimes there is no traceback when error occurs.
            try:
                trainer = Trainer(
                    model_init=model_init,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=dataset["train"],
                    eval_dataset=dataset["test"],
                )

                trainer.train()
            except:
                # exit gracefully, so wandb logs the problem
                print(traceback.print_exc(), file=sys.stderr)
                exit(1)

            tokenizer.save_pretrained(f"{model_output_path}{run_id}/tokenizer")

            gcs_dir = f"{model_output_path}{run_id}".lstrip("./")
            logger.info(f"UPLOADING ARTIFACTS TO {bucket_name}.{gcs_dir}")

            upload_files_to_gcs_recursive(
                bucket_name=bucket_name,
                local_dir=f"{model_output_path}{run_id}",
                gcs_dir=gcs_dir,
            )

            wandb.log(
                {
                    "best_eval_loss": trainer.state.best_metric,
                    "best_checkpoint_local": os.path.abspath(
                        trainer.state.best_model_checkpoint
                    ),
                    "best_checkpoint_gcs": f"{bucket_name}/"
                    + trainer.state.best_model_checkpoint.lstrip("./"),
                }
            )

    wandb.agent(sweep_id, train, count=n_iter)

    logger.info("TRAINING DONE.")


if __name__ == "__main__":
    app()
