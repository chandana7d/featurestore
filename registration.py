# uc_feature_store/config.py
import os
import yaml

class GlobalConfig:
    """
    Centralized configuration management for the Feature Store system.
    Supports YAML-based overrides and environment variable injection.
    """
    def __init__(self, config_file: str = None):
        self.catalog = "feature_catalog"
        self.schema = "prod_features"
        self.mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
        self.redis_host = "localhost"
        self.redis_port = 6379
        self.default_ts_col = "event_ts"

        if config_file:
            self._load_from_yaml(config_file)

    def _load_from_yaml(self, path: str):
        with open(path, "r") as f:
            config_data = yaml.safe_load(f)

        self.catalog = config_data.get("catalog", self.catalog)
        self.schema = config_data.get("schema", self.schema)
        self.mlflow_uri = config_data.get("mlflow_uri", self.mlflow_uri)
        self.redis_host = config_data.get("redis", {}).get("host", self.redis_host)
        self.redis_port = config_data.get("redis", {}).get("port", self.redis_port)
        self.default_ts_col = config_data.get("default_ts_col", self.default_ts_col)

    def to_dict(self):
        return {
            "catalog": self.catalog,
            "schema": self.schema,
            "mlflow_uri": self.mlflow_uri,
            "redis_host": self.redis_host,
            "redis_port": self.redis_port,
            "default_ts_col": self.default_ts_col,
        }


# uc_feature_store/configs/fs_config.yaml
'''
catalog: enterprise_features
schema: prod_v1
mlflow_uri: "databricks"
redis:
  host: "10.128.0.5"
  port: 6380
default_ts_col: event_time
'''


# uc_feature_store/configs/feature_expectations.json
'''
{
  "feature_1": {"not_null": true, "type": "float"},
  "feature_2": {"not_null": true, "type": "int"},
  "feature_3": {"not_null": false, "type": "string"}
}
'''

# uc_feature_store/registry.py
from databricks.feature_store import FeatureStoreClient
from pyspark.sql import SparkSession, DataFrame
from typing import Optional, List, Union
import logging

class FeatureRegistration:
    """
    Handles Unity Catalog + Delta Lake feature registration lifecycle in Databricks.
    Built for scalability and reliability in enterprise ML systems.
    """

    spark: SparkSession
    catalog: str
    schema: str
    fs: FeatureStoreClient
    logger: logging.Logger

    def __init__(self, catalog: str, schema: str) -> None:
        self.spark = SparkSession.builder.getOrCreate()
        self.catalog = catalog
        self.schema = schema
        self.fs = FeatureStoreClient()
        self.logger = logging.getLogger("FeatureRegistration")
        self.logger.setLevel(logging.INFO)

    def _qualified_name(self, table_name: str) -> str:
        return f"{self.catalog}.{self.schema}.{table_name}"

    def register_features(
        self,
        table_name: str,
        df: DataFrame,
        output_path: str,
        pk_column: str,
        partition_by: Optional[str] = None,
        description: Optional[str] = None,
        overwrite: bool = True
    ) -> None:
        qualified_name = self._qualified_name(table_name)
        if overwrite:
            self.logger.info(f"Dropping existing table: {qualified_name}")
            self.spark.sql(f"DROP TABLE IF EXISTS {qualified_name}")

        partition_clause = f"PARTITIONED BY ({partition_by})" if partition_by else ""
        comment_clause = f"COMMENT '{description}'" if description else ""

        ddl = f"""
        CREATE TABLE {qualified_name}
        USING DELTA
        {partition_clause}
        LOCATION '{output_path}'
        {comment_clause}
        """
        self.logger.info(f"Creating table with DDL:\n{ddl}")
        self.spark.sql(ddl)

        self.logger.info(f"Adding NOT NULL and PK constraints on column: {pk_column}")
        self.spark.sql(f"ALTER TABLE {qualified_name} ALTER COLUMN {pk_column} SET NOT NULL")
        self.spark.sql(f"ALTER TABLE {qualified_name} ADD CONSTRAINT {table_name}_pk PRIMARY KEY ({pk_column})")

        self.logger.info("Writing features to Feature Store...")
        clean_df = df.filter(f"{pk_column} IS NOT NULL")
        self.fs.write_table(name=qualified_name, df=clean_df, mode="overwrite" if overwrite else "append")

        self.logger.info(f"✅ Successfully registered feature table: {qualified_name}")

    def drop_table(self, table_name: str) -> None:
        qualified_name = self._qualified_name(table_name)
        self.logger.info(f"Dropping table: {qualified_name}")
        self.spark.sql(f"DROP TABLE IF EXISTS {qualified_name}")

    def describe_table(self, table_name: str) -> List[str]:
        qualified_name = self._qualified_name(table_name)
        return self.spark.sql(f"DESCRIBE TABLE {qualified_name}").collect()

    def list_tables(self) -> List[str]:
        result = self.spark.sql(f"SHOW TABLES IN {self.catalog}.{self.schema}")
        return [row['tableName'] for row in result.collect()]

    def table_exists(self, table_name: str) -> bool:
        return table_name in self.list_tables()


class FeatureTableMetadata:
    def __init__(self, catalog: str, schema: str):
        self.catalog = catalog
        self.schema = schema
        self.spark = SparkSession.builder.getOrCreate()

    def _qualified_name(self, table_name: str) -> str:
        return f"{self.catalog}.{self.schema}.{table_name}"

    def describe_table(self, table_name: str) -> List[str]:
        qualified_name = self._qualified_name(table_name)
        return self.spark.sql(f"DESCRIBE TABLE {qualified_name}").collect()

    def list_tables(self) -> List[str]:
        result = self.spark.sql(f"SHOW TABLES IN {self.catalog}.{self.schema}")
        return [row['tableName'] for row in result.collect()]

    def table_exists(self, table_name: str) -> bool:
        return table_name in self.list_tables()

    def preview_table(self, table_name: str, limit: int = 5) -> DataFrame:
        qualified_name = self._qualified_name(table_name)
        return self.spark.sql(f"SELECT * FROM {qualified_name} LIMIT {limit}")

    def get_table_schema(self, table_name: str) -> str:
        qualified_name = self._qualified_name(table_name)
        df = self.spark.table(qualified_name)
        return df.schema.simpleString()

    def count_rows(self, table_name: str) -> int:
        qualified_name = self._qualified_name(table_name)
        return self.spark.sql(f"SELECT COUNT(*) FROM {qualified_name}").collect()[0][0]

    def get_lineage(self, table_name: str) -> str:
        qualified_name = self._qualified_name(table_name)
        return self.spark.sql(f"SHOW LINEAGE IN {qualified_name}").collect()

    def get_table_tags(self, table_name: str) -> List[str]:
        qualified_name = self._qualified_name(table_name)
        return self.spark.sql(f"SHOW TABLE TAGS {qualified_name}").collect()

    def update_table_tag(self, table_name: str, tag: str, value: str) -> None:
        qualified_name = self._qualified_name(table_name)
        self.spark.sql(f"ALTER TABLE {qualified_name} SET TAGS('{tag}' = '{value}')")

    def remove_table_tag(self, table_name: str, tag: str) -> None:
        qualified_name = self._qualified_name(table_name)
        self.spark.sql(f"ALTER TABLE {qualified_name} UNSET TAGS('{tag}')")

# Usage Example
if __name__ == "__main__":
    from uc_feature_store.config import GlobalConfig
    cfg = GlobalConfig("configs/fs_config.yaml")

    registrar = FeatureRegistration(cfg.catalog, cfg.schema)
    spark = SparkSession.builder.getOrCreate()
    compiled_features_ly: DataFrame = spark.read.parquet("/mnt/delta/compiled_features_ly")

    registrar.register_features(
        table_name="compiled_features_ly",
        df=compiled_features_ly,
        output_path="/mnt/delta/compiled_features_ly",
        pk_column="StoreNumber",
        partition_by="store_region",
        description="Compiled LY features for modeling"
    )
