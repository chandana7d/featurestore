# uc_feature_store/serving.py
from abc import ABC, abstractmethod
from databricks.feature_store import FeatureStoreClient
from pyspark.sql import SparkSession, DataFrame
from typing import List, Optional
import logging

class BaseFeatureServing(ABC):
    def __init__(self, catalog: str, schema: str):
        self.spark = SparkSession.builder.getOrCreate()
        self.catalog = catalog
        self.schema = schema
        self.fs = FeatureStoreClient()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def _qualified_name(self, table_name: str) -> str:
        return f"{self.catalog}.{self.schema}.{table_name}"

    @abstractmethod
    def get_features(self, table_name: str, keys: List[str]) -> DataFrame:
        pass


class OfflineFeatureServing(BaseFeatureServing):
    def get_features(self, table_name: str, keys: Optional[List[str]] = None) -> DataFrame:
        qualified_name = self._qualified_name(table_name)
        df = self.spark.table(qualified_name)
        if keys:
            self.logger.info(f"Filtering features for keys: {keys}")
            return df.filter(df[df.columns[0]].isin(keys))
        return df

    def create_training_set(self, feature_tables: List[str], label_df: DataFrame, join_key: str) -> DataFrame:
        self.logger.info("Creating training set by joining features with label data...")
        for table_name in feature_tables:
            features_df = self.get_features(table_name)
            label_df = label_df.join(features_df, on=join_key, how="left")
        return label_df

    def validate_join_keys(self, df: DataFrame, join_key: str) -> bool:
        if join_key not in df.columns:
            raise ValueError(f"Join key '{join_key}' not found in DataFrame columns: {df.columns}")
        self.logger.info(f"Join key '{join_key}' validation passed.")
        return True

    def sample_features(self, table_name: str, n: int = 10) -> DataFrame:
        qualified_name = self._qualified_name(table_name)
        self.logger.info(f"Sampling {n} rows from {qualified_name}")
        return self.spark.table(qualified_name).limit(n)

    def feature_summary(self, table_name: str) -> DataFrame:
        qualified_name = self._qualified_name(table_name)
        df = self.spark.table(qualified_name)
        from pyspark.sql import functions as F
        numeric_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() in ('int', 'bigint', 'double', 'float')]

        agg_exprs = [
            F.count(col).alias(f"count_{col}") for col in numeric_cols
        ] + [
            F.min(col).alias(f"min_{col}") for col in numeric_cols
        ] + [
            F.max(col).alias(f"max_{col}") for col in numeric_cols
        ] + [
            F.avg(col).alias(f"mean_{col}") for col in numeric_cols
        ] + [
            F.stddev(col).alias(f"std_{col}") for col in numeric_cols
        ]

        return df.agg(*agg_exprs)


class OnlineFeatureServing(BaseFeatureServing):
    def get_features(self, table_name: str, keys: List[str]) -> DataFrame:
        qualified_name = self._qualified_name(table_name)
        self.logger.info(f"Fetching features for online use from {qualified_name}")
        return self.fs.read_table(qualified_name).filter(f"{self.fs.read_table(qualified_name).columns[0]} IN ({','.join(keys)})")


if __name__ == "__main__":
    from uc_feature_store.config import GlobalConfig
    cfg = GlobalConfig("configs/fs_config.yaml")

    offline_serving = OfflineFeatureServing(cfg.catalog, cfg.schema)
    spark = SparkSession.builder.getOrCreate()
    label_data = spark.read.parquet("/mnt/labels/labels.parquet")
    training_df = offline_serving.create_training_set(
        feature_tables=["compiled_features_ly"],
        label_df=label_data,
        join_key="StoreNumber"
    )
    training_df.show(5)

    online_serving = OnlineFeatureServing(cfg.catalog, cfg.schema)
    online_features = online_serving.get_features("compiled_features_ly", keys=["123", "456"])
    online_features.show(5)

