from typing import List

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


metadata = FeatureTableMetadata(cfg.catalog, cfg.schema)
    print("Schema:", metadata.get_table_schema("compiled_features_ly"))
    print("Row count:", metadata.count_rows("compiled_features_ly"))
    print("Tags:", metadata.get_table_tags("compiled_features_ly"))