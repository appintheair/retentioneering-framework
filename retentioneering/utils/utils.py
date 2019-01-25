import json
import yaml
from google.cloud import bigquery


class Config(dict):
    def __init__(self, filename, is_json=False):
        with open(filename, 'rb') as f:
            super(Config, self).__init__(json.load(f)) if is_json else super(Config, self).__init__(yaml.load(f))

    def export(self, filename, is_json=False):
        with open(filename, 'wb') as f:
            json.dump(self, f) if is_json else yaml.dump(self, f)


def init_from_file(filename, is_json=False):
    settings = Config(filename, is_json=is_json)
    client = bigquery.Client.from_service_account_json(
        settings['settings']['service_account_path'],
        project=settings['settings']['project'])
    job_config = bigquery.QueryJobConfig()
    job_config.use_legacy_sql = True
    job_config.allow_large_results = True
    job_config.write_disposition = "WRITE_TRUNCATE"
    settings_subset = list(settings['sql'].values())[0]
    job_config.destination = client.dataset(settings_subset['destination_table']['dataset']) \
        .table(settings_subset['destination_table']['table'])

    return client, job_config
