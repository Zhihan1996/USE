import logging
import sys
from datetime import datetime

import apache_beam as beam
import typer
from apache_beam import DoFn, GroupByKey, ParDo
from apache_beam.io import ReadFromBigQuery, WriteToAvro
from apache_beam.options.pipeline_options import (
    GoogleCloudOptions,
    PipelineOptions,
    SetupOptions,
    StandardOptions,
)
from queries import (
    IMPORT_TO_BEAM_QUERY_FOR_TEST_DATA,
    IMPORT_TO_BEAM_QUERY_FOR_TRAIN_DATA,
    TEST_ACCOUNT_SELF_DELETION_PREDICTION_SAMPLE_USERS_QUERY,
    TEST_AD_CLICK_BINARY_PREDICTION_SAMPLE_USERS_QUERY,
    TEST_AD_VIEW_TIME_PREDICTION_SAMPLE_USERS_QUERY,
    TEST_LOCKED_USER_PREDICTION_SAMPLE_USERS_QUERY,
    TEST_PROP_OF_AD_CLICKS_PREDICTION_SAMPLE_USERS_QUERY,
    TEST_REPORTED_USER_PREDICTION_SAMPLE_USERS_QUERY,
    TEST_SAMPLE_TABLE,
    TRAIN_SAMPLE_TABLE,
    TRAIN_SAMPLE_USERS_QUERY,
)
from utils_preprocess import (
    avro_file_in_gcs_exists,
    bq_query,
    generate_query_for_user_events,
    get_available_tasks,
    read_predefined_list_of_items,
    write_bg_data_to_gcs,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger()

test_sample_user_queries = {
    "test_reported_user_prediction": TEST_REPORTED_USER_PREDICTION_SAMPLE_USERS_QUERY,
    "test_locked_user_prediction": TEST_LOCKED_USER_PREDICTION_SAMPLE_USERS_QUERY,
    "test_account_self_deletion_prediction": TEST_ACCOUNT_SELF_DELETION_PREDICTION_SAMPLE_USERS_QUERY,
    "test_ad_click_binary_prediction": TEST_AD_CLICK_BINARY_PREDICTION_SAMPLE_USERS_QUERY,
    "test_prop_of_ad_clicks_prediction": TEST_PROP_OF_AD_CLICKS_PREDICTION_SAMPLE_USERS_QUERY,
    "test_ad_view_time_prediction": TEST_AD_VIEW_TIME_PREDICTION_SAMPLE_USERS_QUERY,
}


class KeyByUser(DoFn):
    def process(self, row):
        yield ((row["ghost_user_id"]), row)


class SortUserActivitiesAndTimestamps(DoFn):
    def process(
        self,
        tup,
        sequence_length,
        task,
        features_end_date,
        labels_date,
        event_ts_col="client_ts",
        event_col="token",
        label_ts_col="min_label_ts",
    ):

        _uid, activities = tup

        # Sort by client_ts
        activities = sorted(activities, key=lambda x: x[event_ts_col])

        if "test" in task and features_end_date == labels_date:
            activities = [
                row for row in activities if row[event_ts_col] < row[label_ts_col]
            ]

        event_sequence = [str(_uid)] + [
            row[event_col].replace(" ", "_") if row[event_col] is not None else "None"
            for row in activities
        ][sequence_length::-1]
        time_sequence = [str(_uid)] + [
            row[event_ts_col].isoformat() if row[event_ts_col] is not None else "None"
            for row in activities
        ][sequence_length::-1]

        assert len(event_sequence) == len(time_sequence)

        yield {
            "event_sequence": event_sequence,
            "time_sequence": time_sequence,
        }


def beam_get_sorted_activities_and_timestamps(
    bq_project: str,
    bq_dataset: str,
    bucket_name: str,
    gcs_folder_name: str,
    task: str,
    sample_n: int,
    sequence_length: int,
    features_start_date: str,
    features_end_date: str,
    labels_date: str,
    sample_table_option: str,
    user_filter_option: str,
    beam_directrunner: bool = False,
    beam_max_num_workers: int = 4,
    sc_analytics_tables: str = "all",
    overwrite: bool = False,
):
    prefix = f"{gcs_folder_name}/user_event_and_time_sequences_{task}_in_{sc_analytics_tables}_{sample_n}_from_{features_start_date}_to_{features_end_date}_on_{labels_date}_{sample_table_option}_{user_filter_option}"
    output_path = f"gs://{bucket_name}/{gcs_folder_name}/user_event_and_time_sequences"

    if avro_file_in_gcs_exists(bucket_name, prefix):
        logger.warning(
            f"Destination file(s) {prefix} already exists in bucket {bucket_name}."
        )
        if not overwrite:
            logger.warning(f"Skipping beam job, overwrite argument set to False")
            return

    logger.info("Sort activities")

    CONFIGS = [
        "--service_account_email=social-content-engagement@research-prototypes.iam.gserviceaccount.com",
        "--project=research-prototypes",
        f"--staging_location={output_path}/staging",
        f"--temp_location={output_path}/temp",
        f"--max_num_workers={beam_max_num_workers}",
        "--worker_machine_type=n2-standard-16",
        "--runner=DirectRunner" if beam_directrunner else "--runner=DataFlowRunner",
        "--job_name=user-activity-stream-" + datetime.now().strftime("%Y%m%d-%H-%M-%S"),
        "--region=us-west1",
    ]

    pipeline_options = PipelineOptions(CONFIGS)
    options = pipeline_options.view_as(StandardOptions)
    setup_options = pipeline_options.view_as(SetupOptions)
    pipeline_options.view_as(GoogleCloudOptions)
    setup_options.save_main_session = True

    p = beam.Pipeline(options=pipeline_options)

    def import_to_beam_query():
        # this function only uses global variables (except query)
        query = IMPORT_TO_BEAM_QUERY_FOR_TRAIN_DATA
        if task != "train":
            query = IMPORT_TO_BEAM_QUERY_FOR_TEST_DATA

        return query.format(
            task=task,
            sample_n=sample_n,
            features_start_date=features_start_date,
            features_end_date=features_end_date,
            labels_date=labels_date,
            project=bq_project,
            dataset=bq_dataset,
            sc_analytics_tables=sc_analytics_tables,
            sample_table_option=sample_table_option,
            user_filter_option=user_filter_option,
        )

    activities = (
        p
        | "Read activities from BQ"
        >> ReadFromBigQuery(
            use_standard_sql=True,
            validate=True,
            query=import_to_beam_query(),
        )
        | "Key session by uid" >> ParDo(KeyByUser())
        | "Group by uid" >> GroupByKey()
        | "Sort activities in sessions"
        >> ParDo(
            SortUserActivitiesAndTimestamps(),
            sequence_length,
            task,
            features_end_date,
            labels_date,
        )
        | "Write to Avro"
        >> WriteToAvro(
            file_path_prefix=f"{output_path}_{task}_in_{sc_analytics_tables}_{sample_n}_from_{features_start_date}_to_{features_end_date}_on_{labels_date}_{sample_table_option}_{user_filter_option}",
            file_name_suffix=".avro",
            schema={
                "type": "record",
                "name": "user_event_and_time_sequences",
                "fields": [
                    {
                        "name": "event_sequence",
                        "type": {"type": "array", "items": "string"},
                        "default": [],
                    },
                    {
                        "name": "time_sequence",
                        "type": {"type": "array", "items": "string"},
                        "default": [],
                    },
                ],
            },
        )
    )

    results = p.run()

    if options.view_as(StandardOptions).runner == "DirectRunner":
        results.wait_until_finish()

    return results


def main(
    bq_project: str = typer.Option("research-prototypes", help="BQ project"),
    bq_dataset: str = typer.Option(
        "bic_test", help="BQ dataset: bic_test or umap_user_model"
    ),
    task: str = typer.Option(
        "train",
        help="train, test_reported_user_prediction, test_locked_user_prediction, test_account_self_deletion_prediction, test_ad_click_binary_prediction, test_prop_of_ad_clicks_prediction, test_ad_view_time_prediction",
    ),
    features_start_date: datetime = typer.Option(
        "2023-03-13",
        help="Start date of data collection",
        formats=["%Y-%m-%d", "%Y%m%d"],
    ),
    features_end_date: datetime = typer.Option(
        "2023-03-13",
        help="End date of data collection",
        formats=["%Y-%m-%d", "%Y%m%d"],
    ),
    labels_date: datetime = typer.Option(
        "2023-03-15",
        help="Date of users being locked",
        formats=["%Y-%m-%d", "%Y%m%d"],
    ),
    sample_n: int = typer.Option(1000, help="Sample size (users)"),
    sequence_length: int = typer.Option(
        10000,
        help="Number of events per user",
    ),
    total_session_time_sec_threshold: int = typer.Option(
        60,
        help="Min number of secs of session time summed between start-date and end-date",
    ),
    bucket_name: str = typer.Option(
        "bic-test",
        help="bic-test or umap-user-model. The former retains data for only three days.",
    ),
    gcs_folder_name: str = typer.Option(
        "data",
        help="folder name for where to store data in a gcs bucket.",
    ),
    overwrite_bq_tables: bool = typer.Option(
        False, help="Overwrite existing BQ tables or not"
    ),
    overwrite_gcs_avro_files: bool = typer.Option(
        False, help="Overwrite existing gcs avro files or not"
    ),
    show_queries: bool = typer.Option(False, help="Show queries during execution"),
    beam_directrunner: bool = typer.Option(
        False, help="Run beam job locally w/ DirectRunner"
    ),
    beam_max_num_workers: int = typer.Option(
        80,
        help="Max number of beam workers",
    ),
    sc_analytics_tables: str = typer.Option(
        "all",
        help="""sc-analytics tables to use for generating user events, either a single value string or a string of table suffixes separated by underscore; \
        possible values: app, chat, cognac, ops_feed, ops_map, ops_memories, ops_page, ops_story, page, snap, story, user. When `use_sample_tables` is set to True, "all" represents all of these values, but note that "all" is not applicable when `use_sample_tables` is set to False""",
    ),
    use_sample_tables: bool = typer.Option(
        True,
        help="""whether to use sample event tables or non-sample/population event tables; True or False""",
    ),
    apply_user_filter_subquery: bool = typer.Option(
        False,
        help="""whether to impose any kind of instert_user_filter_subquery on the users (e.g., demographics, activivity level); True or False""",
    ),
):

    logger.info("START")

    assert task in get_available_tasks(scope="all")

    features_start_date = features_start_date.strftime("%Y%m%d")
    features_end_date = features_end_date.strftime("%Y%m%d")
    labels_date = labels_date.strftime("%Y%m%d")

    assert bucket_name in ["bic-test", "umap-user-model"]
    assert gcs_folder_name is not None
    assert use_sample_tables in [True, False]
    assert apply_user_filter_subquery in [True, False]

    user_filter_option = "filtered" if apply_user_filter_subquery else "unfiltered"
    sample_table_option = "sample" if use_sample_tables else "population"

    if apply_user_filter_subquery:
        user_filter_subquery = f"""l_90_country = 'US'
        AND age >= 18
        AND inferred_age_bucket != '13-17'
        AND is_test_user = FALSE
        AND is_engaged_in_last_30_days = TRUE
        AND ghost_user_id IN (
            SELECT
              ghost_user_id
            FROM (
              SELECT
                ghost_user_id,
                SUM(app_application_open) AS app_application_open_sum,
                SUM(total_session_time_sec) AS total_session_time_sec_sum
              FROM
                `sc-analytics.report_growth.growth_user_engagement_*`
              WHERE
                _TABLE_SUFFIX BETWEEN '{features_start_date}' AND '{features_end_date}'
                AND country = 'US'
              GROUP BY
                ghost_user_id )
            WHERE
              total_session_time_sec_sum >= {total_session_time_sec_threshold}
          )"""
        if task in ["test_ad_click_binary_prediction"]:
            user_filter_subquery = "WHERE " + user_filter_subquery
        else:
            user_filter_subquery = "AND " + user_filter_subquery
    else:
        user_filter_subquery = """"""

    event_names = read_predefined_list_of_items("event_names.txt")

    if task == "train":

        TRAIN_TOKENS_QUERY = TRAIN_SAMPLE_TABLE + generate_query_for_user_events(
            sc_analytics_tables, sample_table_option
        )

        bq_user_table_id = bq_query(
            task=task,
            file_prefix=f"train_user_sample_in_{sc_analytics_tables}",
            query=TRAIN_SAMPLE_USERS_QUERY,
            bq_project=bq_project,
            bq_dataset=bq_dataset,
            sample_n=sample_n,
            features_start_date=features_start_date,
            features_start_date_truncated=features_start_date[2:],
            features_end_date=features_end_date,
            features_end_date_truncated=features_end_date[2:],
            labels_date=features_end_date,
            sample_table_option=sample_table_option,
            user_filter_option=user_filter_option,
            user_filter_subquery=user_filter_subquery,
            sc_analytics_tables=sc_analytics_tables,
            overwrite=overwrite_bq_tables,
            show_query=show_queries,
        )

        bq_event_table_id = bq_query(
            task=task,
            file_prefix=f"train_event_tokens_in_{sc_analytics_tables}",
            query=TRAIN_TOKENS_QUERY,
            bq_project=bq_project,
            bq_dataset=bq_dataset,
            sample_n=sample_n,
            features_start_date=features_start_date,
            features_start_date_truncated=features_start_date[2:],
            features_end_date=features_end_date,
            features_end_date_truncated=features_end_date[2:],
            labels_date=features_end_date,
            sample_table_option=sample_table_option,
            user_filter_option=user_filter_option,
            user_filter_subquery=user_filter_subquery,
            sc_analytics_tables=sc_analytics_tables,
            overwrite=overwrite_bq_tables,
            show_query=show_queries,
            event_names=event_names,
        )

    elif "test" in task:

        TEST_TOKENS_QUERY = TEST_SAMPLE_TABLE + generate_query_for_user_events(
            sc_analytics_tables, sample_table_option
        )

        TEST_SAMPLE_USER_QUERY = test_sample_user_queries[task]

        bq_user_table_id = bq_query(
            task=task,
            file_prefix=task + f"_user_sample_in_{sc_analytics_tables}",
            query=TEST_SAMPLE_USER_QUERY,
            bq_project=bq_project,
            bq_dataset=bq_dataset,
            sample_n=sample_n,
            features_start_date=features_start_date,
            features_start_date_truncated=features_start_date[2:],
            features_end_date=features_end_date,
            features_end_date_truncated=features_end_date[2:],
            labels_date=labels_date,
            sample_table_option=sample_table_option,
            user_filter_option=user_filter_option,
            user_filter_subquery=user_filter_subquery,
            sc_analytics_tables=sc_analytics_tables,
            overwrite=overwrite_bq_tables,
            show_query=show_queries,
        )

        bq_event_table_id = bq_query(
            task=task,
            file_prefix=task + f"_event_tokens_in_{sc_analytics_tables}",
            query=TEST_TOKENS_QUERY,
            bq_project=bq_project,
            bq_dataset=bq_dataset,
            sample_n=sample_n,
            features_start_date=features_start_date,
            features_start_date_truncated=features_start_date[2:],
            features_end_date=features_end_date,
            features_end_date_truncated=features_end_date[2:],
            labels_date=labels_date,
            sample_table_option=sample_table_option,
            user_filter_option=user_filter_option,
            user_filter_subquery=user_filter_subquery,
            sc_analytics_tables=sc_analytics_tables,
            overwrite=overwrite_bq_tables,
            show_query=show_queries,
            event_names=event_names,
        )

    else:

        sys.exit("The task parameter needs to be either train or test.")

    # Save bq tables to gcs as avro files
    write_bg_data_to_gcs(
        bq_project=bq_project,
        bq_dataset=bq_dataset,
        bq_table_id=bq_user_table_id,
        bucket_name=bucket_name,
        gcs_folder_name=gcs_folder_name,
        overwrite=overwrite_gcs_avro_files,
    )

    write_bg_data_to_gcs(
        bq_project=bq_project,
        bq_dataset=bq_dataset,
        bq_table_id=bq_event_table_id,
        bucket_name=bucket_name,
        gcs_folder_name=gcs_folder_name,
        overwrite=overwrite_gcs_avro_files,
    )

    # Start beam job to get user event and time sequences
    beam_get_sorted_activities_and_timestamps(
        bq_project=bq_project,
        bq_dataset=bq_dataset,
        bucket_name=bucket_name,
        gcs_folder_name=gcs_folder_name,
        task=task,
        sample_n=sample_n,
        sequence_length=sequence_length,
        features_start_date=features_start_date,
        features_end_date=features_end_date,
        labels_date=features_end_date if task == "train" else labels_date,
        beam_directrunner=beam_directrunner,
        beam_max_num_workers=beam_max_num_workers,
        sc_analytics_tables=sc_analytics_tables,
        sample_table_option=sample_table_option,
        user_filter_option=user_filter_option,
        overwrite=overwrite_gcs_avro_files,
    )

    logger.info("DONE")


if __name__ == "__main__":
    typer.run(main)
