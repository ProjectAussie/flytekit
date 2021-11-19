import os
import re
import typing
from typing import TypeVar

import pandas
import pandas as pd
from fsspec.core import split_protocol
from google.cloud import bigquery, bigquery_storage
from google.cloud.bigquery import LoadJob, LoadJobConfig
from google.cloud.bigquery_storage_v1 import types

from flytekit.configuration import aws as _aws_config
from flytekit.types.structured.structured_dataset import (
    FLYTE_DATASET_TRANSFORMER,
    DatasetFormat,
    DatasetPersistenceHandler,
    DatasetRetrievalHandler,
)

T = TypeVar("T")


class PandasToParquetPersistenceHandlers(DatasetPersistenceHandler):
    def persist(
        self,
        df: pandas.DataFrame,
        to_file: typing.Union[os.PathLike, str],
        coerce_timestamps: str = "us",
        allow_truncated_timestamps: bool = False,
    ):
        """
        Writes data frame as a chunk to the local directory owned by the Schema object.  Will later be uploaded to s3.
        :param df: data frame to write as parquet
        :param to_file: Sink file to write the dataframe to
        :param coerce_timestamps: format to store timestamp in parquet. 'us', 'ms', 's' are allowed values.
            Note: if your timestamps will lose data due to the coercion, your write will fail!  Nanoseconds are
            problematic in the Parquet format and will not work. See allow_truncated_timestamps.
        :param allow_truncated_timestamps: default False. Allow truncation when coercing timestamps to a coarser
            resolution.
        """

        df.to_parquet(
            to_file,
            coerce_timestamps=coerce_timestamps,
            allow_truncated_timestamps=allow_truncated_timestamps,
            storage_options=_s3_setup_args(),
        )


class PandasToBQPersistenceHandlers(DatasetPersistenceHandler):
    def persist(self, df: pandas.DataFrame, path: str, job_config: typing.Optional[LoadJobConfig] = None) -> LoadJob:
        table_id = path.split("://", 1)[1].replace(":", ".")
        client = bigquery.Client()
        return client.load_table_from_dataframe(df, table_id, job_config=job_config)


class ParquetToPandasRetrievalHandler(DatasetRetrievalHandler):
    def retrieve(self, path) -> pd.DataFrame:
        return pandas.read_parquet(path, storage_options=_s3_setup_args())


class BQToPandasRetrievalHandler(DatasetRetrievalHandler):
    def retrieve(self, path: str, **kwargs) -> pd.DataFrame:
        # path will be like bq://photo-313016:flyte.new_table1
        _, project_id, dataset_id, table_id = re.split("\\.|://|:", path)
        client = bigquery_storage.BigQueryReadClient()
        table = f"projects/{project_id}/datasets/{dataset_id}/tables/{table_id}"
        parent = "projects/{}".format(project_id)

        requested_session = types.ReadSession(
            table=table,
            data_format=types.DataFormat.ARROW,
        )
        read_session = client.create_read_session(parent=parent, read_session=requested_session)

        stream = read_session.streams[0]
        reader = client.read_rows(stream.name)
        frames = []
        for message in reader.rows().pages:
            frames.append(message.to_dataframe())
        return pd.concat(frames)


def _get_storage_optionals(path: str) -> dict:
    protocol = "file"
    if path:
        protocol, _ = split_protocol(path)
        if protocol is None and path.startswith("/"):
            protocol = "file"
    kwargs = {}
    if protocol == "file":
        kwargs = {"auto_mkdir": True}
    if protocol == "s3":
        kwargs = _s3_setup_args()
    return kwargs


def _s3_setup_args():
    kwargs = {}
    if _aws_config.S3_ACCESS_KEY_ID.get() is not None:
        os.environ[_aws_config.S3_ACCESS_KEY_ID_ENV_NAME] = _aws_config.S3_ACCESS_KEY_ID.get()

    if _aws_config.S3_SECRET_ACCESS_KEY.get() is not None:
        os.environ[_aws_config.S3_SECRET_ACCESS_KEY_ENV_NAME] = _aws_config.S3_SECRET_ACCESS_KEY.get()

    # S3fs takes this as a special arg
    if _aws_config.S3_ENDPOINT.get() is not None:
        kwargs["client_kwargs"] = {"endpoint_url": _aws_config.S3_ENDPOINT.get()}

    return kwargs


FLYTE_DATASET_TRANSFORMER.register_handler(pd.DataFrame, DatasetFormat.PARQUET, PandasToParquetPersistenceHandlers())
FLYTE_DATASET_TRANSFORMER.register_handler(pd.DataFrame, DatasetFormat.BIGQUERY, PandasToBQPersistenceHandlers())

FLYTE_DATASET_TRANSFORMER.register_handler(DatasetFormat.PARQUET, pd.DataFrame, ParquetToPandasRetrievalHandler())
FLYTE_DATASET_TRANSFORMER.register_handler(DatasetFormat.BIGQUERY, pd.DataFrame, BQToPandasRetrievalHandler())
