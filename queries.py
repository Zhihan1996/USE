TRAIN_SAMPLE_USERS_QUERY = """

WITH
  {ghost_user_ids_from_event_tables_subquery}

  users AS (
    SELECT
      ghost_user_id,
      gender,
      age,
      reported_age_bucket,
      inferred_age_bucket,
      creation_time,
      friend_count,
    FROM
      `sc-analytics.report_user.user_demographic_{features_end_date}`
    WHERE
        ghost_user_id IN (
          SELECT ghost_user_id
          FROM ghost_user_ids_from_event_tables
        )
        {user_filter_subquery}
  ),

  sample AS (
    SELECT *
    FROM users
    ORDER BY RAND()
    LIMIT {sample_n}
  ),

  sample_with_sticky_personas AS (
    SELECT *
    FROM sample
    LEFT JOIN (
      SELECT ghost_user_id, sticky_Persona
      FROM `sc-analytics.user_persona_v3.sticky_user_persona_v3_{features_end_date}`
      )
    USING (ghost_user_id)
    ),

  sample_with_sticky_and_nonsticky_personas AS (
    SELECT *
    FROM sample_with_sticky_personas
    LEFT JOIN (
      SELECT ghost_user_id, Persona AS nonsticky_Persona
      FROM `sc-analytics.user_persona_v3.user_persona_v3_{features_end_date}`
      )
    USING (ghost_user_id)
    )

  SELECT *
  FROM sample_with_sticky_and_nonsticky_personas
  ORDER BY ghost_user_id
"""

TEST_SAMPLE_USERS_QUERY_TEMPLATE = """
  {ghost_user_ids_from_event_tables_subquery}

  tagged_users AS (
    SELECT
      ghost_user_id,
      gender,
      age,
      reported_age_bucket,
      inferred_age_bucket,
      creation_time,
      friend_count,
    FROM
      `sc-analytics.report_user.user_demographic_{features_end_date}`
    WHERE
        ghost_user_id IN (
          SELECT DISTINCT ghost_user_id
          FROM tagged_user_ghost_user_ids_and_ts
        )
        AND ghost_user_id IN (
          SELECT ghost_user_id
          FROM ghost_user_ids_from_event_tables
        )
        {user_filter_subquery}
  ),

  tagged_users_sample AS (
    SELECT *,
    1 AS label,
    FROM tagged_users
    ORDER BY RAND() LIMIT {sample_n}
  ),

  tagged_users_sample_with_ts AS (
    SELECT *
    FROM tagged_users_sample
    LEFT JOIN tagged_user_ghost_user_ids_and_ts
    USING (ghost_user_id)
  ),

  untagged_users_sample AS (
    SELECT
      ghost_user_id,
      gender,
      age,
      reported_age_bucket,
      inferred_age_bucket,
      creation_time,
      friend_count,
    FROM
      `sc-analytics.report_user.user_demographic_{features_end_date}`
    WHERE
        ghost_user_id IN (
          SELECT ghost_user_id
          FROM ghost_user_ids_from_event_tables
        )
        AND ghost_user_id NOT IN (
          SELECT DISTINCT ghost_user_id
          FROM tagged_users
        )
        {user_filter_subquery}
    ORDER BY RAND() LIMIT {sample_n}
  ),

  untagged_users_sample_with_ts AS (
    SELECT *,
    0 AS label,
    TIMESTAMP('2030-01-01 00:00:00 UTC') AS min_label_ts,
    "untagged_user" AS label_name,
    FROM untagged_users_sample
  ),

  test_users_sample AS (
    SELECT * FROM tagged_users_sample_with_ts
    UNION ALL SELECT * FROM untagged_users_sample_with_ts
  )

  SELECT *
  FROM test_users_sample
  ORDER BY ghost_user_id
"""

TEST_REPORTED_USER_PREDICTION_SAMPLE_USERS_QUERY = (
    """

WITH

  tagged_user_ids_and_ts AS (
    SELECT
        reportedUserId AS user_id,
        MIN(timestamp) AS min_label_ts,
        reason AS label_name
    FROM
        `snap-report-processor.shepherd_production.report_metadata`
    WHERE
        DATE(timestamp) = EXTRACT(DATE FROM PARSE_TIMESTAMP("%Y%m%d", CAST({labels_date} AS STRING)))
        AND reportedUserId IS NOT NULL
        AND reportedUserId != ''
        AND REGEXP_CONTAINS(reason, r'^[A-Z_]+$')
    GROUP BY user_id, reason
  ),

  tagged_user_ghost_user_ids_and_ts AS (
      SELECT
          * EXCEPT (user_id),
      FROM tagged_user_ids_and_ts
      LEFT JOIN (
          SELECT DISTINCT
              user_id,
              ghost_id AS ghost_user_id
          FROM `sc-mjolnir.enigma.user_map_v2`
          WHERE user_id in (SELECT user_id FROM tagged_user_ids_and_ts)
          )
      USING (user_id)
  ),

"""
    + TEST_SAMPLE_USERS_QUERY_TEMPLATE
)


TEST_LOCKED_USER_PREDICTION_SAMPLE_USERS_QUERY = (
    """
WITH

  tagged_user_ghost_user_ids_and_ts AS (
    SELECT
        ghost_id AS ghost_user_id,
        TIMESTAMP_MILLIS(MIN(timestamp)) AS min_label_ts,
        description AS label_name
    FROM
        `sc-infosec-services.abuse_analytics_data_feed.locked_user_{labels_date}`
    WHERE
        ghost_id IS NOT NULL
        AND ghost_id != ''
        AND punishment = 1
    GROUP BY ghost_id, description
  ),

"""
    + TEST_SAMPLE_USERS_QUERY_TEMPLATE
)

TEST_ACCOUNT_SELF_DELETION_PREDICTION_SAMPLE_USERS_QUERY = (
    """
WITH

  tagged_user_ghost_user_ids_and_ts AS (
    SELECT
        ghost_id AS ghost_user_id,
        TIMESTAMP_MILLIS(MIN(timestamp)) AS min_label_ts,
        description AS label_name
    FROM
        `sc-infosec-services.abuse_analytics_data_feed.locked_user_{labels_date}`
    WHERE
        ghost_id IS NOT NULL
        AND ghost_id != ''
        AND punishment = 3
        AND description = "SELF_SERVICE_DELETE"
    GROUP BY ghost_id, description
  ),

"""
    + TEST_SAMPLE_USERS_QUERY_TEMPLATE
)

TEST_AD_CLICK_BINARY_PREDICTION_SAMPLE_USERS_QUERY = """
WITH

ad_click_users AS (
  SELECT ghost_user_id, MIN(client_ts) AS min_label_ts, 1 as label, "ad_click" as label_name,
  FROM `sc-analytics.prod_analytics_sample.story_daily_events_{labels_date}`
  WHERE event_name = "STORY_AD_VIEW"
    AND exit_event IN ('SWIPE_UP','TAP_CARET')
    AND ad_skippable_type = "FULL"
    AND exit_intent = "OPEN_ATTACHMENT"
    AND video_view_time_sec IS NOT NULL
  GROUP BY ghost_user_id
),

ad_click_users_with_demographics AS (
  SELECT
    t2.ghost_user_id,
    t2.gender,
    t2.age,
    t2.reported_age_bucket,
    t2.inferred_age_bucket,
    t2.creation_time,
    t2.friend_count,
    t1.min_label_ts,
    t1.label,
    t1.label_name
  FROM ad_click_users t1
  LEFT JOIN `sc-analytics.report_user.user_demographic_{features_end_date}` t2
  ON t1.ghost_user_id = t2.ghost_user_id
  {user_filter_subquery}
  ORDER BY RAND()
  LIMIT {sample_n}
),

ad_nonclick_users AS (
  SELECT ghost_user_id, MIN(client_ts) AS min_label_ts, 0 as label, "no_ad_click" as label_name,
  FROM `sc-analytics.prod_analytics_sample.story_daily_events_{labels_date}`
  WHERE event_name = "STORY_AD_VIEW"
    AND exit_event NOT IN ('SWIPE_UP','TAP_CARET')
    AND ad_skippable_type = "FULL"
    AND ghost_user_id NOT IN (
      SELECT ghost_user_id
      FROM ad_click_users
    )
  GROUP BY ghost_user_id
),

ad_nonclick_users_with_demographics AS (
  SELECT
    t2.ghost_user_id,
    t2.gender,
    t2.age,
    t2.reported_age_bucket,
    t2.inferred_age_bucket,
    t2.creation_time,
    t2.friend_count,
    t1.min_label_ts,
    t1.label,
    t1.label_name
  FROM ad_nonclick_users t1
  LEFT JOIN `sc-analytics.report_user.user_demographic_{features_end_date}` t2
  ON t1.ghost_user_id = t2.ghost_user_id
  {user_filter_subquery}
  ORDER BY RAND()
  LIMIT {sample_n}
)

SELECT * FROM ad_click_users_with_demographics
UNION ALL
SELECT * FROM ad_nonclick_users_with_demographics
"""

TEST_PROP_OF_AD_CLICKS_PREDICTION_SAMPLE_USERS_QUERY = """
WITH

test_users AS (
  SELECT ghost_user_id,
    MIN(client_ts) AS min_label_ts,
    AVG(ad_click) AS label,
    "proportion_of_ad_clicks" as label_name,
    COUNT(*) AS counts,
  FROM (
    SELECT DISTINCT ghost_user_id,
      ad_id,
      client_ts,
      CASE
        WHEN exit_event IN ('SWIPE_UP','TAP_CARET') AND exit_intent = "OPEN_ATTACHMENT" THEN 1
        ELSE 0
      END AS ad_click
    FROM `sc-analytics.prod_analytics_sample.story_daily_events_{labels_date}`
    WHERE event_name = "STORY_AD_VIEW"
      AND ad_skippable_type = "FULL"
      AND video_view_time_sec IS NOT NULL
  )
  GROUP BY ghost_user_id
)

SELECT
  t2.ghost_user_id,
  t2.gender,
  t2.age,
  t2.reported_age_bucket,
  t2.inferred_age_bucket,
  t2.creation_time,
  t2.friend_count,
  t1.min_label_ts,
  t1.label,
  t1.label_name,
  t1.counts
FROM test_users t1
LEFT JOIN `sc-analytics.report_user.user_demographic_{features_end_date}` t2
ON t1.ghost_user_id = t2.ghost_user_id
{user_filter_subquery}
ORDER BY RAND()
LIMIT {sample_n}
"""


TEST_AD_VIEW_TIME_PREDICTION_SAMPLE_USERS_QUERY = """
WITH

test_users AS (
  SELECT *,
    "ad_view_time_sec" AS label_name
  FROM (
    SELECT ghost_user_id,
      MIN(client_ts) AS min_label_ts,
      AVG(video_view_time_sec) AS label,
      COUNT(*) AS counts,
    FROM `sc-analytics.prod_analytics_sample.story_daily_events_{labels_date}`
    WHERE event_name = "STORY_AD_VIEW"
      AND ad_skippable_type = "FULL"
      AND video_view_time_sec IS NOT NULL
    GROUP BY ghost_user_id
    )
)

SELECT
  t2.ghost_user_id,
  t2.gender,
  t2.age,
  t2.reported_age_bucket,
  t2.inferred_age_bucket,
  t2.creation_time,
  t2.friend_count,
  t1.min_label_ts,
  t1.label,
  t1.label_name,
  t1.counts
FROM test_users t1
LEFT JOIN `sc-analytics.report_user.user_demographic_{features_end_date}` t2
ON t1.ghost_user_id = t2.ghost_user_id
{user_filter_subquery}
ORDER BY RAND()
LIMIT {sample_n}
"""


TRAIN_SAMPLE_TABLE = """
WITH

  users AS (
    SELECT *
    FROM
      `{project}.{dataset}.train_user_sample_in_{sc_analytics_tables}_{sample_n}_from_{features_start_date}_to_{features_end_date}_on_{labels_date}_{sample_table_option}_{user_filter_option}`
  ),
"""

TEST_SAMPLE_TABLE = """
WITH

  users AS (
    SELECT *
    FROM
      `{project}.{dataset}.{task}_user_sample_in_{sc_analytics_tables}_{sample_n}_from_{features_start_date}_to_{features_end_date}_on_{labels_date}_{sample_table_option}_{user_filter_option}`
  ),
"""

EVENTS_TABLE = """

  events AS (
    SELECT
      ghost_user_id,
      session_id,
      event_name AS token,
      IF (client_ts IS NULL, event_time, client_ts) AS client_ts,
    FROM `sc-analytics.prod_analytics_sample.daily_events_20*`
    WHERE _TABLE_SUFFIX BETWEEN '{features_start_date_truncated}' AND '{features_end_date_truncated}'
      AND ghost_user_id IN (SELECT ghost_user_id FROM users)
      AND NOT((session_id IS NULL))
      AND NOT((event_name IS NULL))
      AND event_name IN ({event_names})
  ),

  new_session_tokens AS (
    SELECT
      ghost_user_id,
      session_id,
      TIMESTAMP_SUB(client_ts, INTERVAL 1 MILLISECOND) AS client_ts,
      "[NEW_SESSION]" AS token,
    FROM (
      SELECT *
      FROM (
        SELECT *,
          ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY client_ts) AS row_num
        FROM events
      )
      WHERE row_num = 1
    )
  )

  SELECT ghost_user_id, session_id, client_ts, token FROM events
  UNION ALL SELECT ghost_user_id, session_id, client_ts, token FROM new_session_tokens
"""

TRAIN_TOKENS_QUERY = TRAIN_SAMPLE_TABLE + EVENTS_TABLE

COMBINED_TOKENS_LOCATION_QUERY = """
WITH

users AS (
  SELECT *
  FROM
    `{project}.{dataset}.user_sample_{sample_n}_from_{features_start_date}_to_{features_end_date}_on_{labels_date}_{sample_table_option}_{user_filter_option}`
),

events AS (
  SELECT ghost_user_id, session_id, token, `client_ts`
  FROM
      `{project}.{dataset}.event_tokens_{sample_n}_from_{features_start_date}_to_{features_end_date}_on_{labels_date}_{sample_table_option}_{user_filter_option}`
),

visits AS (
  SELECT *
  FROM (
      SELECT *
      FROM `sc-targeting-measurement.location_visit.venue_visit_scores_*`
      WHERE _TABLE_SUFFIX BETWEEN '{features_start_date}' AND '{features_end_date}' AND is_real_visit is True
  )
  JOIN (
    SELECT DISTINCT userID AS user_id, ghostID AS ghost_user_id
    FROM `sc-targeting-measurement.user_ids.ghost_id_staging_{features_end_date}`
    WHERE ghostID IN ( SELECT ghost_user_id FROM users)
  )
  USING
    (user_id)
  ),

categories AS (
    SELECT DISTINCT
      ghost_user_id,
      visits.fence_id AS fence_id,
      visits.place_name AS place_name,
      visits.duration AS duration,
      visits.pos_prob AS visit_prob,
      TIMESTAMP_MILLIS(visits.start_time_msec) AS start_time,
      TIMESTAMP_MILLIS(visits.end_time_msec) AS end_time,
      t3.category_name,
      t3.timezone AS timezone,
      t3.region AS region,
      t3.path AS path
    FROM
      visits
    JOIN (
      SELECT
          t1.name AS place_name,
          t1.id AS id,
          t1.address.region AS region,
          t2.name AS category_name,
          t1.timezone AS timezone,
          t2.path AS path
      FROM
          `spectre-1217.verrazano.places` t1
      JOIN
          `spectre-1217.verrazano.place_categories` t2
      ON
          t1.category_id = t2.id) t3
    ON
      visits.fence_id = t3.id
    ),

locations AS (
  SELECT *
  FROM (
      SELECT
        ghost_user_id,
        CAST(NULL AS STRING) as session_id,
        start_time AS location_start_time,
        end_time AS location_end_time,
        visit_prob,
        IF (category IN ({sensitive_categories}), '[Sensitive]', CONCAT('[', category, ']')) AS location_category,
        IF (category_name IN ({sensitive_categories}), '[Sensitive]', CONCAT('[', category_name, ']')) AS location_category_name,
        TRUE as is_visit
      FROM
        categories
      JOIN
        `research-prototypes.user_states_context_factors.location_categories` t2
      ON
        categories.category_name = t2.name)
  WHERE location_start_time != location_end_time
  ),

sessions AS (
  SELECT
    ghost_user_id,
    session_id,
    MIN(client_ts) AS session_start_time,
    MAX(client_ts) AS session_end_time,
  FROM
    events
  GROUP BY
    ghost_user_id, session_id
),

sessions_locations_joined AS (
  SELECT
    s.ghost_user_id,
    s.session_id,
    s.session_start_time,
    s.session_end_time,
    DATETIME_DIFF(s.session_end_time, s.session_start_time, SECOND) AS session_duration,
    l.location_start_time,
    l.location_end_time,
    l.location_category,
    l.location_category_name,
    l.visit_prob,
    l.is_visit
  FROM sessions s
  LEFT JOIN locations l
  ON l.ghost_user_id = s.ghost_user_id AND
    ((ABS(EXTRACT(DAY FROM s.session_start_time) - EXTRACT(DAY FROM l.location_start_time)) < 2) OR
     (ABS(EXTRACT(DAY FROM s.session_start_time) - EXTRACT(DAY FROM l.location_end_time)) < 2) OR
     (ABS(EXTRACT(DAY FROM s.session_end_time) - EXTRACT(DAY FROM l.location_start_time)) < 2) OR
     (ABS(EXTRACT(DAY FROM s.session_end_time) - EXTRACT(DAY FROM l.location_end_time)) < 2))
),

inferred_location_sessions AS (
  SELECT * EXCEPT(is_visit),
  FROM (
    SELECT *,
    CASE
      WHEN session_end_time <= location_start_time THEN DATE_DIFF(session_end_time, location_start_time, SECOND)
      WHEN session_start_time < location_start_time AND session_end_time > location_start_time THEN DATE_DIFF(session_end_time, location_start_time, SECOND)
      WHEN session_start_time >= location_start_time AND session_end_time <= location_end_time THEN DATE_DIFF(session_end_time, session_start_time, SECOND)
      WHEN session_start_time < location_end_time AND session_end_time > location_end_time THEN DATE_DIFF(location_end_time, session_start_time, SECOND)
      WHEN session_start_time >= location_end_time THEN DATE_DIFF(location_end_time, session_start_time, SECOND)
      ELSE NULL
    END AS overlap
    FROM sessions_locations_joined
  )
  WHERE overlap >= 0 AND is_visit = TRUE
),

max_visit_probs AS (
  SELECT
      ghost_user_id, session_id, MAX(visit_prob) as max_visit_prob, location_start_time, location_end_time
  FROM
    inferred_location_sessions
  GROUP BY
    ghost_user_id, session_id, location_start_time, location_end_time
),

inferred_session_locations_filtered_on_max_visit_prob AS (
  SELECT t1.* EXCEPT(visit_prob),
  FROM inferred_location_sessions t1
  INNER JOIN max_visit_probs t2
  ON t1.ghost_user_id = t2.ghost_user_id AND t1.session_id = t2.session_id AND t1.visit_prob = t2.max_visit_prob AND t1.location_start_time = t2.location_start_time AND t1.location_end_time = t2.location_end_time
),

inferred_session_locations_final AS (
  SELECT
    ghost_user_id,
    session_id,
    location_start_time AS start_time,
    location_end_time AS end_time,
    location_category,
    location_category_name
  FROM inferred_session_locations_filtered_on_max_visit_prob
),

inferred_session_locations_long AS (
    SELECT ghost_user_id, session_id, token, client_ts
    FROM (
        SELECT *
        FROM inferred_session_locations_final
        UNPIVOT (client_ts FOR timestamp IN (start_time, end_time))
        )
    UNPIVOT (token FOR location IN (location_category, location_category_name))
)

SELECT ghost_user_id, session_id, token, client_ts
FROM events
UNION ALL
SELECT * FROM inferred_session_locations_long
"""

IMPORT_TO_BEAM_QUERY_FOR_TRAIN_DATA = """
  SELECT
    *
  FROM
    `{project}.{dataset}.{task}_event_tokens_in_{sc_analytics_tables}_{sample_n}_from_{features_start_date}_to_{features_end_date}_on_{labels_date}_{sample_table_option}_{user_filter_option}`
"""

IMPORT_TO_BEAM_QUERY_FOR_TEST_DATA = """
  SELECT
    *
  FROM
    `{project}.{dataset}.{task}_event_tokens_in_{sc_analytics_tables}_{sample_n}_from_{features_start_date}_to_{features_end_date}_on_{labels_date}_{sample_table_option}_{user_filter_option}`
  LEFT JOIN (
    SELECT ghost_user_id, min_label_ts
    FROM `{project}.{dataset}.{task}_user_sample_in_{sc_analytics_tables}_{sample_n}_from_{features_start_date}_to_{features_end_date}_on_{labels_date}_{sample_table_option}_{user_filter_option}`)
  USING (ghost_user_id)
"""
