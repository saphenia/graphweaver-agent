"""
GraphWeaver Agent - Data Models
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
import uuid


class DatabaseType(str, Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


class ColumnDataType(str, Enum):
    INTEGER = "integer"
    BIGINT = "bigint"
    SMALLINT = "smallint"
    DECIMAL = "decimal"
    FLOAT = "float"
    VARCHAR = "varchar"
    TEXT = "text"
    CHAR = "char"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    TIMESTAMP = "timestamp"
    TIME = "time"
    UUID = "uuid"
    JSON = "json"
    BINARY = "binary"
    UNKNOWN = "unknown"


class RelationshipCardinality(str, Enum):
    ONE_TO_ONE = "1:1"
    ONE_TO_MANY = "1:N"
    MANY_TO_ONE = "N:1"
    MANY_TO_MANY = "N:M"
    UNKNOWN = "unknown"


class DataSourceConfig(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    id: str = Field(default_factory=lambda: f"ds-{uuid.uuid4().hex[:8]}")
    name: str = Field(default="Data Source")
    db_type: DatabaseType = Field(default=DatabaseType.POSTGRESQL)
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    database: str = Field(default="orders")
    username: str = Field(default="saphenia")
    password: str = Field(default="secret")
    schema_name: str = Field(default="public")


class Neo4jConfig(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    uri: str = Field(default="bolt://localhost:7687")
    username: str = Field(default="neo4j")
    password: str = Field(default="password")
    database: str = Field(default="neo4j")


class PipelineConfig(BaseModel):
    min_cardinality_ratio: float = Field(default=0.01)
    min_target_uniqueness: float = Field(default=0.95)
    min_value_overlap: float = Field(default=0.1)
    stage2_min_score: float = Field(default=0.3)
    sample_size: int = Field(default=10000)
    min_match_rate: float = Field(default=0.95)
    min_final_score: float = Field(default=0.7)
    allow_cycles: bool = Field(default=False)
    filter_value_columns: bool = Field(default=True)


class ColumnMetadata(BaseModel):
    table_name: str
    column_name: str
    data_type: ColumnDataType
    is_nullable: bool = True
    is_primary_key: bool = False
    is_unique: bool = False
    is_indexed: bool = False


class ColumnStatistics(BaseModel):
    table_name: str
    column_name: str
    total_count: int = 0
    distinct_count: int = 0
    null_count: int = 0
    uniqueness_ratio: float = 0.0
    null_ratio: float = 0.0
    sample_values: List[Any] = Field(default_factory=list)


class ExistingForeignKey(BaseModel):
    constraint_name: str
    source_table: str
    source_column: str
    target_table: str
    target_column: str


class TableMetadata(BaseModel):
    schema_name: str
    table_name: str
    row_count: int = 0
    columns: List[ColumnMetadata] = Field(default_factory=list)
    primary_key_columns: List[str] = Field(default_factory=list)
    existing_foreign_keys: List[ExistingForeignKey] = Field(default_factory=list)


class FKCandidate(BaseModel):
    id: str = Field(default_factory=lambda: f"fk-{uuid.uuid4().hex[:8]}")
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    score: float = 0.0
    confidence: float = 0.0
    cardinality: RelationshipCardinality = RelationshipCardinality.UNKNOWN
    stage_scores: Dict[str, Any] = Field(default_factory=dict)
    is_valid: bool = False


class DiscoveryResult(BaseModel):
    datasource_id: str
    datasource_name: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    tables_scanned: int = 0
    total_columns: int = 0
    initial_candidates: int = 0
    stage1_passed: int = 0
    stage2_passed: int = 0
    stage3_passed: int = 0
    final_candidates: int = 0
    discovered_fks: List[FKCandidate] = Field(default_factory=list)
    tables: List[str] = Field(default_factory=list)


# =============================================================================
# Aliases for backward compatibility with imports
# =============================================================================

# Alias for DataType (used in __init__.py and other modules)
DataType = ColumnDataType

# Alias for ForeignKeyCandidate (used in __init__.py)
ForeignKeyCandidate = FKCandidate