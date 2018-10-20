last_events_query = """
SELECT
  tbl1.event_name AS event_name,
  tbl1.event_timestamp AS event_timestamp,
  tbl1.user_pseudo_id AS user_pseudo_id,
  tbl1.event_params.key AS event_params_key,
  tbl1.event_params_value_string_value AS event_params_value_string_value,
  tbl1.app_info.version AS app_info_version,
  tbl1.traffic_source.source AS traffic_source_source,
  tbl1.user_properties.key AS user_properties_key,
  tbl1.device.mobile_brand_name AS device_mobile_brand_name,
  tbl1.user_properties.value.string_value AS user_properties_value_string_value,
  event_rank
FROM (
  SELECT
    tbl1.*,
    DENSE_RANK() OVER (PARTITION BY tbl1.user_pseudo_id ORDER BY tbl1.event_timestamp DESC) AS event_rank,
    ROW_NUMBER() OVER (PARTITION BY tbl1.user_pseudo_id, tbl1.event_name, tbl1.event_params_value_string_value, tbl1.event_timestamp) AS row_n
  FROM (
    SELECT
      *
    FROM
      flatten(flatten((
          SELECT
            event_name,
            event_timestamp,
            user_pseudo_id,
            event_params.key,
            REPLACE(event_params.value.string_value,'\\n',' ') AS event_params_value_string_value,
            app_info.version,
            traffic_source.source,
            user_properties.key,
            device.mobile_brand_name,
            user_properties.value.string_value
          FROM
            table_date_range([analytics_153456165.events_],
              TIMESTAMP('{date_events_first}'),
              TIMESTAMP('{date_events_last}'))),
          event_params),
        user_properties.value.string_value)
    WHERE
      event_name IN (
      SELECT
        string_field_0
      FROM
        {target_events_table})) tbl1
  INNER JOIN (
    SELECT
      *
    FROM (
      SELECT
        user_pseudo_id,
        MIN(event_timestamp) AS min_event_timestamp
      FROM
        table_date_range([analytics_153456165.events_],
          TIMESTAMP('{date_target_event_first}'),
          TIMESTAMP('{date_target_event_last}'))
      WHERE
        LOWER(event_name) CONTAINS 'feedback'
        AND LOWER(event_name) CONTAINS 'success'
      GROUP BY
        user_pseudo_id)
    WHERE
      user_pseudo_id IN (
      SELECT
        user_pseudo_id
      FROM
        {target_users_table})) tbl2
  ON
    tbl1.user_pseudo_id = tbl2.user_pseudo_id
  WHERE
    tbl1.event_timestamp <= tbl2.min_event_timestamp)
WHERE 
  row_n = 1 {count_events_filter} 
ORDER BY
  user_pseudo_id,
  event_timestamp
"""

new_users_query = """
SELECT
  tbl1.event_name AS event_name,
  tbl1.event_timestamp AS event_timestamp,
  tbl1.user_pseudo_id AS user_pseudo_id,
  tbl1.event_params.key AS event_params_key,
  tbl1.event_params_value_string_value AS event_params_value_string_value,
  tbl1.app_info.version AS app_info_version,
  tbl1.traffic_source.source AS traffic_source_source,
  tbl1.user_properties.key AS user_properties_key,
  tbl1.device.mobile_brand_name AS device_mobile_brand_name,
  tbl1.user_properties.value.string_value AS user_properties_value_string_value,
  event_rank
FROM (
  SELECT
    tbl1.*,
    DENSE_RANK() OVER (PARTITION BY tbl1.user_pseudo_id ORDER BY tbl1.event_timestamp) AS event_rank,
    ROW_NUMBER() OVER (PARTITION BY tbl1.user_pseudo_id, tbl1.event_name, tbl1.event_params_value_string_value, tbl1.event_timestamp) AS row_n
  FROM (
    SELECT
      *
    FROM
      flatten(flatten((
          SELECT
            event_name,
            event_timestamp,
            user_pseudo_id,
            event_params.key,
            REPLACE(event_params.value.string_value,'\\n',' ') AS event_params_value_string_value,
            app_info.version,
            traffic_source.source,
            user_properties.key,
            device.mobile_brand_name,
            user_properties.value.string_value
          FROM
            table_date_range([analytics_153456165.events_],
              TIMESTAMP('{date_events_first}'),
              TIMESTAMP('{date_events_last}'))),
          event_params),
        user_properties.value.string_value)
    WHERE
      event_name {filter_event_operator} {filter_event_names}) tbl1
  INNER JOIN (
    SELECT
      *
    FROM (
      SELECT
        user_pseudo_id
      FROM
        table_date_range([analytics_153456165.events_],
          TIMESTAMP('{date_target_event_first}'),
          TIMESTAMP('{date_target_event_last}'))
      WHERE
        event_name {reg_event_operator} {reg_event_names}
        AND app_info.version = '{app_version}'
      GROUP BY
        user_pseudo_id)) tbl2
  ON
    tbl1.user_pseudo_id = tbl2.user_pseudo_id)
WHERE
  row_n = 1 {count_events_filter}
ORDER BY
  user_pseudo_id,
  event_timestamp
"""

query_with_params = """
SELECT
  tbl1.event_name AS event_name,
  tbl1.event_timestamp AS event_timestamp,
  tbl1.user_pseudo_id AS user_pseudo_id,
  tbl1.event_params.key AS event_params_key,
  tbl1.event_params_value_string_value AS event_params_value_string_value,
  tbl1.app_info.version AS app_info_version,
  tbl1.traffic_source.source AS traffic_source_source,
  tbl1.user_properties.key AS user_properties_key,
  tbl1.device.mobile_brand_name AS device_mobile_brand_name,
  tbl1.user_properties.value.string_value AS user_properties_value_string_value,
  event_rank
FROM (
  SELECT
    tbl1.*,
    DENSE_RANK() OVER (PARTITION BY tbl1.user_pseudo_id ORDER BY tbl1.event_timestamp {rank_sort_type}) AS event_rank
    {duplicate_event_flag}
  FROM (
    SELECT
      *
    FROM
      flatten(flatten((
          SELECT
            event_name,
            event_timestamp,
            user_pseudo_id,
            event_params.key,
            REPLACE(event_params.value.string_value,'\\n',' ') AS event_params_value_string_value,
            app_info.version,
            traffic_source.source,
            user_properties.key,
            device.mobile_brand_name,
            user_properties.value.string_value
          FROM
            table_date_range([analytics_153456165.events_],
              TIMESTAMP('{date_events_first}'),
              TIMESTAMP('{date_events_last}'))),
          event_params),
        user_properties.value.string_value)
    {events_filter} {events_app_version_filter}) tbl1
  INNER JOIN ({rand_select_start}
      SELECT
        user_pseudo_id {min_event_timestamp}
      FROM
        table_date_range([analytics_153456165.events_],
          TIMESTAMP('{date_users_events_first}'),
          TIMESTAMP('{date_users_events_last}'))
      {users_event_filter} {users_app_version_filter}
      GROUP BY
        user_pseudo_id {rand_select_end}) tbl2
  ON
    tbl1.user_pseudo_id = tbl2.user_pseudo_id {delete_events_after_target})
  {duplicate_event_delete}
  {count_events_filter}
ORDER BY
  user_pseudo_id,
  event_timestamp
"""
