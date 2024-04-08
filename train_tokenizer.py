import logging
from tempfile import TemporaryDirectory

import typer
from google.cloud import storage
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
from transformers import BigBirdTokenizer
from utils import make_path, read_user_sequences, upload_files_to_gcs_recursive

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger()


def main(
    bucket_name: str = typer.Option("bic-test", help="bic-test or umap-user-model."),
    task: str = typer.Option("train", help="train or a downstream task"),
    prefix: str = typer.Option(
        "data/user_event_and_time_sequences",
        help="prefix for event sequence file names on GCS",
    ),
    sample_n: int = typer.Option(100, help="training size"),
    features_start_date: str = typer.Option(
        "20230313", help="start date of sampling frame for user events"
    ),
    features_end_date: str = typer.Option(
        "20230313", help="end date of sampling frame for user events"
    ),
    labels_date: str = typer.Option(None, help="date of downstream behavior"),
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
        help="choose among: shortlist, shortlist_without_page_page_view, shortlist_without_repetition, shortlist_with_counts",
    ),
    tokenizer_output_folder: str = typer.Option(
        "../artifacts/tokenizers",
        help="local path to save the trained tokenizer; a copy will be automatically saved to the corresponding gcs location",
    ),
    tokenizer_name: str = typer.Option("bigbird", help=""),
    tokenizer_model_type: str = typer.Option(
        "word",
        help="choose among: BPE, unigram, char, word; see SentencePiece documentation for details",
    ),
    max_files: int = typer.Option(
        1000, help="maximum number of files to used for tokenizer training"
    ),
):

    assert bucket_name in ["bic-test", "umap-user-model"]
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

    tokenizer_output_path = (
        f"{tokenizer_output_folder}/{tokenizer_name}_{tokenizer_model_type}"
    )
    make_path(tokenizer_output_path)

    if labels_date is None:
        labels_date = features_end_date

    # read user event sequences
    logger.info("READING USER EVENT SEQUENCES.")
    event_sequences, _, _ = read_user_sequences(
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
    )

    with open("./event_names.txt", "r") as f:
        event_names = f.read().lstrip("\n").rstrip("\n").split("\n")
    unique_tokens = set(event_names)

    # we append the event_names.txt to the local_files list, so that we don't miss out
    # on learning to some user events even if they don't exist in the training data.
    local_files = ["./event_names.txt"]
    temp_dir = TemporaryDirectory()
    local_file = f"{temp_dir.name}/event_sequences"
    local_files.append(local_file)

    event_sequences = "\n".join(event_sequences)
    unique_tokens |= set(event_sequences.split())

    with open(local_file, "w") as outfile:
        outfile.write(event_sequences)

    vocab_size = len(unique_tokens)
    print(f"Vocabulary size: {vocab_size}")

    # training custom tokenizer
    logger.info("TRAINING AND SAVING CUSTOM TOKENIZER.")

    if tokenizer_name == "bert":
        tokenizer = BertWordPieceTokenizer(
            clean_text=False,
            handle_chinese_chars=False,
            strip_accents=False,
            lowercase=True,
        )

        tokenizer.train(
            files=local_files[0 : min(max_files, len(local_files))],
            vocab_size=300,
            min_frequency=0,
            show_progress=True,
            limit_alphabet=50,
            initial_alphabet=[],
            wordpieces_prefix="##",
            special_tokens=["[PAD", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        )

        tokenizer.save_model(tokenizer_output_path)

    else:
        SentencePieceTrainer.train(
            input=local_files[0 : min(max_files, len(local_files))],
            model_prefix=f"{tokenizer_output_path}/{tokenizer_name}_{tokenizer_model_type}",
            vocab_size=min(
                vocab_size + 3, 1936
            ),  # +3 for meta piece tokens that are added by the trainer, e.g. '<unk>'. 1936 is the max apparently.
            character_coverage=1.0,
            model_type="word",
            max_sentence_length=1000000,
            num_threads=96,
            minloglevel=1,
        )

        tokenizer = BigBirdTokenizer(
            vocab_file=f"{tokenizer_output_path}/{tokenizer_name}_{tokenizer_model_type}.model",
            pad_token="[PAD]",
        )

        tokenizer.save_pretrained(tokenizer_output_path)

    upload_files_to_gcs_recursive(
        bucket_name=bucket_name,
        local_dir=tokenizer_output_path,
        gcs_dir=tokenizer_output_path.lstrip("./"),
    )


if __name__ == "__main__":
    typer.run(main)
