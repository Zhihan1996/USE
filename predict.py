import logging
import os
import pickle
import random
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import typer
import wandb
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import (
    classifier_tune_and_fit,
    extract_model_name,
    feature_engineering,
    get_available_tasks,
    get_n_days_between_two_dates,
    make_path,
    read_top_n_predefined_list_of_label_names,
    read_user_sequences,
    read_user_table,
    strip_and_split_by_space,
    trim_event_sequences,
    upload_files_to_gcs_recursive,
)

logger = logging.getLogger()

commit_id = (
    subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
)

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    wandb_project_name: str = typer.Option(
        "bad-actor-prediction", help="project name in wandb"
    ),
    bucket_name: str = typer.Option("bic-test", help="bic-test or umap-user-model."),
    task: str = typer.Option(
        "test_reported_user_prediction",
        help="name of downstream task: test_reported_user_prediction, test_locked_user_prediction, test_account_self_deletion_prediction, test_ad_click_binary_prediction, test_ad_view_time_prediction. Note that test_ad_view_time_prediction should be used together with the untagged_user_label parameter",
    ),
    n_class: int = typer.Option(
        2,
        help="number of classes to predict; if n_class=2, the target is whether a user is reported/locked or not; if n_class>2, the prediction target is whether a user is a good actor or belongs to one of the top n-1 classes prespecified by users in  locked_reasons.txt or reported_reasons.txt",
    ),
    prefix: str = typer.Option(
        "data/user_event_and_time_sequences",
        help="prefix for user sequence file names on GCS",
    ),
    sample_n: int = typer.Option(100, help="training size"),
    features_start_dates: str = typer.Option(
        "20230313",
        help="start date(s) of sampling frame, in the format of %Y%m%d, either a single value string or a string of dates separated by empty space",
    ),
    features_end_dates: str = typer.Option(
        "20230315",
        help="end date(s) of sampling frame, in the format of %Y%m%d, either a single value string or a string of dates separated by empty space",
    ),
    labels_dates: str = typer.Option(
        "20230315",
        help="date(s) of downstream behavior, in the format of %Y%m%d, either a single value string or a string of dates separated by empty space",
    ),
    sample_table_option: str = typer.Option(
        "sample",
        help="""whether to use sample event tables or non-sample/population event tables; choose between sample and population""",
    ),
    user_filter_option: str = typer.Option(
        "unfiltered",
        help="""whether to impose any kind of filtering on the users (e.g., demographics, activivity level); choose between filtered and unfiltered""",
    ),
    sc_analytics_tables: str = typer.Option(
        "app",
        help="""sc-analytics tables to use for generating user events, either a single value string or a string of table suffixes separated by underscore; \
        possible values: app, chat, cognac, ops_feed, ops_map, ops_memories, ops_page, ops_story, page, snap, story, user""",
    ),
    sequence_type: str = typer.Option(
        "shortlist",
        help="choose among: shortlist, shortlist_without_page_page_view, shortlist_without_repetition, shortlist_with_counts. Check ./event_names.txt for the shortlist.",
    ),
    baseline_model_names: str = typer.Option(
        None,
        help="baseline model names; separated by empty space",
    ),
    transformer_model_names: str = typer.Option(
        None,
        help="transformer model names; separated by empty space",
    ),
    tokenizer_paths: str = typer.Option(
        None,
        help="paths to self-trained tokenizer in gcs bucket; separated by empty space",
    ),
    transformer_model_paths: str = typer.Option(
        None,
        help="paths to a trained transformer-based model in a gcs bucket; separated by empty space",
    ),
    sgns_model_path: str = typer.Option(
        "artifacts/sgns",
        help="path to a trained sgns model in a gcs bucket",
    ),
    test_size: float = typer.Option(
        0.20, help="the proportion of data reserved for validation"
    ),
    max_model_input_size: int = typer.Option(
        512, help="maximum model input size; 512 for BERT and 4096 for BigBird"
    ),
    max_files_for_training: int = typer.Option(
        100000, help="maximum number of event sequence files (on GCS) to use"
    ),
    n_iter: int = typer.Option(
        10, help="the number of parameter settings that are sampled"
    ),
    negative_label: str = typer.Option(
        "untagged_user",
        help="name for untagged users; for task test_ad_click_binary_prediction, specify as no_ad_click; for task test_ad_view_time_prediction, specify either <=2s or <=15s.",
    ),
    sampling_strategy: str = typer.Option(
        "undersampling",
        help="currently supporting only undersampling to the size of the smallest class",
    ),
    save_partial_dependence_results: bool = typer.Option(
        False,
        help="save a dataframe containing results for making partial dependence plots",
    ),
    save_feature_importance_results: bool = typer.Option(
        False,
        help="save a dataframe containing results about feature importance",
    ),
    save_predictions: bool = typer.Option(
        False,
        help="save a dataframe containing prediction results",
    ),
    output_path_local: str = typer.Option(
        "../artifacts/downstream",
        help="local relative path to saving artifacts, which will also be automatically saved to the corresponding gcs bucket using the same relative path",
    ),
    run_name: str = typer.Option(
        "test",
        help="run name used in wandb to track results.",
    ),
    batch_size: int = typer.Option(
        32,
        help="batch size for extracting transformer-based model embeddings",
    ),
    n_pca_dims: int = typer.Option(
        -1,
        help="whether to perform further dimension reduction with PCA. If set as a positive number, will fit a PCA model with n_pca_dims components to perform dimension reduction.",
    ),
    use_time: bool = typer.Option(
        False,
        help="whether to use time information in model prediction",
    ),
    reverse_sequence: bool = typer.Option(
        False,
        help="Use to reverse the order of event and time sequence to ensure they are in the correct order since the files store on GCS stores files in a time-reversed order.",
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

    assert n_class >= 2
    assert sampling_strategy == "undersampling"
    assert task in [
        "test_reported_user_prediction",
        "test_locked_user_prediction",
        "test_account_self_deletion_prediction",
        "test_ad_click_binary_prediction",
        "test_ad_view_time_prediction",
    ]
    assert task in get_available_tasks(scope="downstream")

    output_path_local = f"{output_path_local}/{task}"

    # convert the *_dates variables from strings to lists
    features_start_dates = strip_and_split_by_space(features_start_dates)
    features_end_dates = strip_and_split_by_space(features_end_dates)
    labels_dates = strip_and_split_by_space(labels_dates)

    # convert user model-related variables from strings to lists
    transformer_model_names = (
        []
        if transformer_model_names == "None"
        else strip_and_split_by_space(transformer_model_names)
    )
    tokenizer_paths = (
        [] if tokenizer_paths == "None" else strip_and_split_by_space(tokenizer_paths)
    )
    transformer_model_paths = (
        []
        if transformer_model_paths == "None"
        else strip_and_split_by_space(transformer_model_paths)
    )

    assert (
        len(transformer_model_names)
        == len(tokenizer_paths)
        == len(transformer_model_paths)
    )

    baseline_model_names = (
        []
        if baseline_model_names == "None"
        else strip_and_split_by_space(baseline_model_names)
    )

    baseline_model_available_options = [
        "tf",
        "tf-l2",
        "tf-idf",
        "sgns",
        "random",
        "n-grams",
    ]

    assert all(
        baseline_model in baseline_model_available_options
        for baseline_model in baseline_model_names
    )

    # get the feature window size
    features_window_size = get_n_days_between_two_dates(
        features_start_dates[0], features_end_dates[0]
    )

    # get the feature-label gap size
    features_labels_gap_size = get_n_days_between_two_dates(
        features_end_dates[0], labels_dates[0]
    )

    # load data from local cache or download and process from GCP
    data_save_dir = f"../cache/{prefix}_{task}_in_{sc_analytics_tables}_{sample_n}_from_{features_start_dates[0]}_to_{features_end_dates[0]}_on_{labels_dates[0]}_{sample_table_option}_{user_filter_option}_{max_model_input_size}"
    # load from cache
    if os.path.exists(os.path.join(data_save_dir, "data.pkl")):
        with open(os.path.join(data_save_dir, "data.pkl"), "rb") as f:
            event_sequences, time_sequences, labels, class_list = pickle.load(f)

    # download and process
    else:
        # read data (event sequences)
        event_sequences = []
        time_sequences = []
        user_table = pd.DataFrame()

        # read user data from different supplied features_* and labels_dates
        # and merge them
        for features_start_date, features_end_date, labels_date in zip(
            features_start_dates, features_end_dates, labels_dates
        ):

            # read user event sequences (features)
            (
                event_sequences_sample,
                time_sequences_sample,
                ghost_user_ids,
            ) = read_user_sequences(
                bucket_name=bucket_name,
                prefix=prefix,
                task=task,
                sequence_type=sequence_type,
                sample_n=sample_n,
                features_start_date=features_start_date,
                features_end_date=features_end_date,
                labels_date=labels_date,
                sample_table_option=sample_table_option,
                user_filter_option=user_filter_option,
                sc_analytics_tables=sc_analytics_tables,
                max_files=max_files_for_training,
                length_to_keep=max_model_input_size,
                use_time=use_time,
                reverse_sequence=reverse_sequence,
            )

            # trim event sequences to the specified max_model_input_size
            event_sequences_sample, time_sequences_sample = trim_event_sequences(
                event_sequences_sample, time_sequences_sample, max_model_input_size
            )
            event_sequences += event_sequences_sample
            time_sequences += time_sequences_sample

            # read labels
            user_table_sample = read_user_table(
                bucket_name=bucket_name,
                task=task,
                features_start_date=features_start_date,
                features_end_date=features_end_date,
                labels_date=labels_date,
                sample_table_option=sample_table_option,
                user_filter_option=user_filter_option,
                sample_n=sample_n,
                sc_analytics_tables=sc_analytics_tables,
                user_id_filter=ghost_user_ids,
            )

            assert ghost_user_ids == user_table_sample.ghost_user_id.tolist()

            if task == "test_ad_view_time_prediction" and negative_label == "<=2s":
                user_table_sample["label_name"] = np.where(
                    user_table_sample["label"] > 2, ">2s", "<=2s"
                )
            if task == "test_ad_view_time_prediction" and negative_label == "<=15s":
                user_table_sample["label_name"] = np.where(
                    user_table_sample["label"] > 15, ">15s", "<=15s"
                )

            user_table = pd.concat(
                [user_table, user_table_sample], ignore_index=True, axis=0
            )

        # get the list of pre-specified label names for test_reported_user_prediction
        if task == "test_reported_user_prediction":
            class_list = read_top_n_predefined_list_of_label_names(
                "reported_label_names.txt", n_class - 1, negative_label
            )
        # get the list of pre-specified label names for test_locked_user_prediction
        elif task == "test_locked_user_prediction":
            class_list = read_top_n_predefined_list_of_label_names(
                "locked_label_names.txt", n_class - 1, negative_label
            )

        else:
            class_list = []

        # for multiclass prediction, we filter the data on only those with labels matching our pre-defined list
        if n_class > 2:
            user_table = user_table[user_table.label_name.isin(class_list)]
        else:
            label_name_counts = user_table["label_name"].value_counts()
            label_names_to_delete = label_name_counts[label_name_counts < 20].index
            user_table = user_table[
                ~user_table["label_name"].isin(label_names_to_delete)
            ]

        row_indices = user_table.index
        event_sequences = [event_sequences[i] for i in row_indices]
        time_sequences = [time_sequences[i] for i in row_indices]
        user_table.reset_index(drop=True, inplace=True)
        labels = user_table.label_name.tolist()

        # undersampling according to the smallest class to avoid data imbalance issues
        if sampling_strategy == "undersampling":
            if n_class == 2:
                index_users_with_negative_labels = user_table[
                    user_table["label_name"] == negative_label
                ].index.tolist()
                index_users_with_positive_labels = user_table[
                    user_table["label_name"] != negative_label
                ].index.tolist()

                if len(index_users_with_negative_labels) >= len(
                    index_users_with_positive_labels
                ):
                    random.seed(42)
                    index_users_with_negative_labels_random_sample = random.sample(
                        index_users_with_negative_labels,
                        k=len(index_users_with_positive_labels),
                    )
                    event_sequences = [
                        event_sequences[i]
                        for i in index_users_with_negative_labels_random_sample
                        + index_users_with_positive_labels
                    ]
                    time_sequences = [
                        time_sequences[i]
                        for i in index_users_with_negative_labels_random_sample
                        + index_users_with_positive_labels
                    ]
                    labels = [
                        user_table.label_name.tolist()[i]
                        for i in index_users_with_negative_labels_random_sample
                        + index_users_with_positive_labels
                    ]
                else:
                    random.seed(42)
                    index_users_with_positive_labels_random_sample = random.sample(
                        index_users_with_positive_labels,
                        k=len(index_users_with_negative_labels),
                    )
                    event_sequences = [
                        event_sequences[i]
                        for i in index_users_with_positive_labels_random_sample
                        + index_users_with_negative_labels
                    ]
                    time_sequences = [
                        time_sequences[i]
                        for i in index_users_with_positive_labels_random_sample
                        + index_users_with_negative_labels
                    ]
                    labels = [
                        user_table.label_name.tolist()[i]
                        for i in index_users_with_positive_labels_random_sample
                        + index_users_with_negative_labels
                    ]

            else:
                raise NotImplementedError("TODO in future PR")

        # get proportion of each class
        label_name_props = dict(user_table.label_name.value_counts(normalize=True))
        label_name_props = {
            "prop_" + key: value for key, value in label_name_props.items()
        }

        # cache the data
        logger.warning(
            f"Cache processed data to {data_save_dir}. Please delete the local cache if you are running this code on personal computer."
        )
        os.makedirs(data_save_dir, exist_ok=True)
        with open(os.path.join(data_save_dir, "data.pkl"), "wb") as f:
            pickle.dump([event_sequences, time_sequences, labels, class_list], f)

    assert len(event_sequences) == len(time_sequences) == len(labels)

    # train/test split for user sequences and user labels
    data = list(zip(event_sequences, time_sequences, labels))
    random.shuffle(data)
    num_test = int(test_size * len(data))
    data_test = data[:num_test]
    data_train = data[num_test:]
    train_event_sequences, train_time_sequences, train_labels = zip(*data_train)
    test_event_sequences, test_time_sequences, test_labels = zip(*data_test)

    # get proportion of the positive label(s) for the total sample and the test set
    total_prop_label = 1 - labels.count(negative_label) / len(labels)
    test_prop_label = 1 - test_labels.count(negative_label) / len(test_labels)

    label_stats = {
        "total_sample_size": len(labels),
        "total_prop_label": total_prop_label,
        "test_sample_size": len(test_labels),
        "test_prop_label": test_prop_label,
    }

    pooling_modes = {
        "tf": ["na"],
        "tf-l2": ["na"],
        "tf-idf": ["na"],
        "n-grams": ["na"],
        "sgns": ["mean", "max", "weighted_mean"],
        "random": ["mean", "max", "weighted_mean"],
        "bert": ["mean", "max", "weighted_mean"],
        "gpt2": ["mean", "weighted_mean"],
        "retnet": ["mean", "last"],
        "rwkv": ["mean", "last"],
        "bigbird": ["mean", "max", "weighted_mean", "weighted_max", "cls"],
        "deberta-v2": ["mean", "max", "weighted_mean", "weighted_max", "cls"],
    }

    # iterate through different feature engineering methods:
    # tf: term frequency (counts)
    # tf: term frequency (l2 normalized)
    # tf-idf
    # sgns: skip-gram with negative sampling
    # random: randomly initiated word embeddings of fixed sizes
    # transformers

    transformer_methods = [
        extract_model_name(transformer_model_path)
        for transformer_model_path in transformer_model_paths
    ]
    methods = baseline_model_names + transformer_methods
    all_model_names = baseline_model_names + transformer_model_names
    assert len(methods) >= 1

    for i, method in enumerate(methods):
        if method in transformer_methods:
            model_path = transformer_model_paths[i - len(baseline_model_names)]
            tokenizer_path = tokenizer_paths[i - len(baseline_model_names)]
        elif method == "sgns":
            model_path = sgns_model_path
            tokenizer_path = ""
        else:
            model_path = ""
            tokenizer_path = ""

        train_features, test_features, feature_names = feature_engineering(
            train_event_sequences=train_event_sequences,
            train_time_sequences=train_time_sequences,
            test_event_sequences=test_event_sequences,
            test_time_sequences=test_time_sequences,
            method=method,
            modes=pooling_modes[method],
            max_model_input_size=max_model_input_size,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            bucket_name=bucket_name,
            batch_size=batch_size,
            n_pca_dims=n_pca_dims,
        )

        for mode in pooling_modes[method]:
            # iterate through different classifiers
            # lr: regularized logigstic regression
            # mlp: multilayer perception
            for classifier in ["mlp"]:  # "lr",

                # model tuning, training and evaluation
                (
                    performance_stats,
                    confusion_matrix,
                    partial_dependence_df,
                    feature_importances_df,
                    predictions_df,
                ) = classifier_tune_and_fit(
                    classifier=classifier,
                    train_features=train_features[mode],
                    test_features=test_features[mode],
                    train_labels=train_labels,
                    test_labels=test_labels,
                    tune_n_iter=n_iter,
                    n_class=n_class,
                    negative_label=negative_label,
                    class_list=class_list,
                    task=task,
                    feature_names=feature_names,
                    save_partial_dependence_results=save_partial_dependence_results,
                    save_feature_importance_results=save_feature_importance_results,
                    save_predictions=save_predictions,
                )

                confusion_matrix.reset_index(inplace=True, names="")

                # log all the performance statistics
                with wandb.init(project=wandb_project_name):
                    wandb.run.name = "_".join([run_name, method, mode])

                    wandb_confusion_matrix = wandb.Table(dataframe=confusion_matrix)
                    wandb_partial_dependence_df = wandb.Table(
                        dataframe=partial_dependence_df
                    )
                    wandb_feature_importances_df = wandb.Table(
                        dataframe=feature_importances_df
                    )
                    wandb_predictions_df = wandb.Table(dataframe=predictions_df)
                    wandb.log(
                        {
                            "commit_id": commit_id,
                            "run_name": wandb.run.name,
                            "model_name": all_model_names[i],
                            "feature": method,
                            "mode": mode,
                            "classifier": classifier,
                            "task": task,
                            "features_start_dates": "&".join(
                                date for date in features_start_dates
                            ),
                            "features_end_dates": "&".join(
                                date for date in features_end_dates
                            ),
                            "labels_dates": "&".join(date for date in labels_dates),
                            "n_class": n_class,
                            "features_window_size": features_window_size,
                            "features_labels_gap_size": features_labels_gap_size,
                            "confusion_matrix": wandb_confusion_matrix,
                            "sc_analytics_tables": sc_analytics_tables,
                            "sample_table_option": sample_table_option,
                            "user_filter_option": user_filter_option,
                            "partial_dependence_table": partial_dependence_df,
                            "feature_importances_table": feature_importances_df,
                            "predictions_table": predictions_df,
                            "negative_label": negative_label,
                            "model_path": model_path,
                            "tokenizer_path": tokenizer_path,
                            "n_iter": n_iter,
                        }
                    )
                    wandb.log(label_stats)
                    wandb.log(performance_stats)
                    wandb.log({"sequence_type": sequence_type})

                    run_id = wandb.run.id
                    make_path(f"{output_path_local}/{run_id}")

                    confusion_matrix.to_csv(
                        f"{output_path_local}/{run_id}/confusion_matrix.csv",
                        index=False,
                    )
                    if save_partial_dependence_results:
                        partial_dependence_df.to_csv(
                            f"{output_path_local}/{run_id}/partial_dependence.csv",
                            index=False,
                        )
                    if save_feature_importance_results:
                        feature_importances_df.to_csv(
                            f"{output_path_local}/{run_id}/feature_importance.csv",
                            index=False,
                        )
                    if save_predictions:
                        predictions_df.to_csv(
                            f"{output_path_local}/{run_id}/predictions.csv", index=False
                        )

                    # write the gcs path based on the specified output_path_local
                    gcs_dir = f"{output_path_local}/{run_id}".lstrip("./")
                    upload_files_to_gcs_recursive(
                        bucket_name=bucket_name,
                        local_dir=f"{output_path_local}/{run_id}",
                        gcs_dir=gcs_dir,
                    )

                    wandb.log({"artifacts_gcs_path": gcs_dir})

                    if n_class > 2:
                        wandb.log(label_name_props)


if __name__ == "__main__":
    app()
