import json
import yaml
from google.cloud import bigquery


class Config(dict):
    """
    Enrichment of dict class with saving option
    """

    def __init__(self, filename, is_json=False):
        """

        :param filename: str
            input file name
        :param is_json: bool
            read in json format (yaml otherwise)
        """
        with open(filename, 'rb') as f:
            super(Config, self).__init__(json.load(f)) if is_json else super(Config, self).__init__(yaml.load(f))

    def export(self, filename, is_json=False):
        """
        Dumps config to file

        :param filename: str
            output file name
        :param is_json: bool
            save in json format (yaml otherwise)
        :return:
        """
        with open(filename, 'wb') as f:
            json.dump(self, f) if is_json else yaml.dump(self, f)


def init_client(service_account_filepath, **kwargs):
    """
    Return the bigquert.Client()

    :param service_account_filepath: path to service account
    :param kwargs: keywords to pass in bigquery.Client.from_service_account_json function

    :type service_account_filepath: path
    :type kwargs: keywords
    :return: bigquert.Client()
    """
    client = bigquery.Client.from_service_account_json(service_account_filepath, **kwargs)
    return client


def init_job_config(project, destination_dataset, destination_table):
    """
    Return the bigquery.QueryJobConfig() with legacy sql and destination table to allow large results

    :param project: project name where destination table is
    :param destination_dataset: dataset id where destination table is
    :param destination_table: destination table id

    :type project: str
    :type destination_dataset: str
    :type destination_table: str
    :return: bigquery.QueryJobConfig()
    """
    job_config = bigquery.QueryJobConfig()
    job_config.use_legacy_sql = True
    job_config.allow_large_results = True
    job_config.write_disposition = "WRITE_TRUNCATE"
    job_config.destination = bigquery.Client(project).dataset(destination_dataset).table(destination_table)
    return job_config


def init_from_file(filename, is_json=False):
    """
    Create bigquert.Client() and bigquery.QueryJobConfig() from json or yaml file

    :param filename: path to file with config
    :param is_json: read file as json if true (read as yaml otherwise)

    :type filename: str
    :type is_json: bool
    :return:
    """
    settings = Config(filename, is_json=is_json)
    client = init_client(settings['settings']['service_account_path'], project=settings['settings']['project'])
    settings_subset = list(settings['sql'].values())[0]
    job_config = init_job_config(
        project=settings['settings']['project'],
        destination_dataset=settings_subset['destination_table']['dataset'],
        destination_table=settings_subset['destination_table']['table']
    )
    return client, job_config
