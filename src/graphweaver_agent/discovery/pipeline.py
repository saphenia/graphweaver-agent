"""
FK Detection Pipeline - Complete multi-stage foreign key discovery.

Stages:
1. Statistical Filtering - Eliminate impossible candidates based on stats
2. Mathematical Scoring - Score candidates using multiple features  
3. Data Sampling - Validate with actual data
4. Graph Validation - Remove cycles, determine cardinality
5. Semantic Filtering - Remove false positives based on names
"""
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime

from graphweaver_agent.models import (
    PipelineConfig,
    DataSourceConfig,
    ColumnMetadata,
    ColumnStatistics,
    ColumnDataType,
    TableMetadata,
    FKCandidate,
    DiscoveryResult,
    ExistingForeignKey,
    RelationshipCardinality,
)
from graphweaver_agent.connectors import create_connector


# =============================================================================
# Utility Functions
# =============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def compute_name_similarity(name1: str, name2: str) -> float:
    """Compute similarity between column names using multiple methods."""
    n1 = name1.lower()
    n2 = name2.lower()
    
    if n1 == n2:
        return 1.0
    
    # Check for common FK patterns: customer_id -> id (in customers table)
    if n1.endswith('_id'):
        entity = n1[:-3]
        if entity == n2 or entity + 's' == n2 or entity == n2.rstrip('s'):
            return 0.95
    
    # Check if target column is 'id' and source matches table pattern
    if n2 == 'id':
        return 0.8
    
    # Check if one contains the other
    if n1 in n2 or n2 in n1:
        return 0.7
    
    # Levenshtein similarity
    distance = levenshtein_distance(n1, n2)
    max_len = max(len(n1), len(n2))
    return 1 - (distance / max_len) if max_len > 0 else 0


def geometric_mean(values: List[float]) -> float:
    """Compute geometric mean, handling zeros."""
    if not values:
        return 0.0
    cleaned = [max(v, 1e-10) for v in values]
    log_sum = sum(math.log(v) for v in cleaned)
    return math.exp(log_sum / len(cleaned))


def types_compatible(source_type: ColumnDataType, target_type: ColumnDataType) -> Tuple[bool, float]:
    """Check if two column types are compatible for FK relationship."""
    if source_type == target_type:
        return True, 1.0
    
    integer_types = {ColumnDataType.INTEGER, ColumnDataType.BIGINT, ColumnDataType.SMALLINT}
    float_types = {ColumnDataType.FLOAT, ColumnDataType.DECIMAL}
    string_types = {ColumnDataType.VARCHAR, ColumnDataType.TEXT, ColumnDataType.CHAR}
    
    for group in [integer_types, float_types, string_types]:
        if source_type in group and target_type in group:
            return True, 0.9
    
    return False, 0.0


def is_value_column(column_name: str) -> bool:
    """Check if column appears to be a value column (not a FK)."""
    value_patterns = [
        'quantity', 'qty', 'amount', 'total', 'price', 'cost',
        'count', 'sum', 'avg', 'rate', 'percentage', 'percent',
        'weight', 'height', 'width', 'length', 'size',
        'balance', 'credit', 'debit', 'fee', 'tax', 'score',
        'rating', 'rank', 'index', 'sequence', 'version',
        'latitude', 'longitude', 'lat', 'lng', 'lon',
    ]
    col_lower = column_name.lower()
    return any(pattern in col_lower for pattern in value_patterns)


def extract_entity_from_column(column_name: str) -> Optional[str]:
    """Extract entity name from FK column (e.g., customer_id -> customer)."""
    col_lower = column_name.lower()
    if col_lower.endswith('_id'):
        return col_lower[:-3]
    if col_lower.endswith('id') and len(col_lower) > 2:
        return col_lower[:-2]
    return None


# =============================================================================
# Stage 1: Statistical Filtering
# =============================================================================

class Stage1StatisticalFilter:
    """
    Filter candidates using statistical properties.
    Eliminates impossible candidates based on type, cardinality, uniqueness.
    """
    
    def __init__(self, connector, config: PipelineConfig):
        self.connector = connector
        self.config = config
        self._stats_cache: Dict[str, ColumnStatistics] = {}
    
    def get_stats(self, table: str, column: str) -> ColumnStatistics:
        key = f"{table}.{column}"
        if key not in self._stats_cache:
            self._stats_cache[key] = self.connector.get_column_statistics(table, column)
        return self._stats_cache[key]
    
    def compute_features(
        self,
        source_col: ColumnMetadata,
        target_col: ColumnMetadata,
        source_stats: ColumnStatistics,
        target_stats: ColumnStatistics,
    ) -> Dict[str, Any]:
        """Compute Stage 1 features for a candidate."""
        
        type_compatible, type_score = types_compatible(source_col.data_type, target_col.data_type)
        
        # Cardinality ratio
        cardinality_ratio = 0.0
        if target_stats.distinct_count > 0:
            cardinality_ratio = source_stats.distinct_count / target_stats.distinct_count
        
        # Value overlap estimation using samples
        source_samples = set(str(v) for v in source_stats.sample_values if v is not None)
        target_samples = set(str(v) for v in target_stats.sample_values if v is not None)
        value_overlap = 0.0
        if source_samples and target_samples:
            intersection = len(source_samples & target_samples)
            value_overlap = intersection / len(source_samples)
        
        # Name similarity
        name_similarity = compute_name_similarity(source_col.column_name, target_col.column_name)
        
        # Also check source column against target table name
        entity = extract_entity_from_column(source_col.column_name)
        if entity:
            table_match = compute_name_similarity(entity, target_col.table_name)
            name_similarity = max(name_similarity, table_match)
        
        features = {
            "type_compatible": type_compatible,
            "type_score": type_score,
            "source_distinct": source_stats.distinct_count,
            "target_distinct": target_stats.distinct_count,
            "cardinality_ratio": cardinality_ratio,
            "source_uniqueness": source_stats.uniqueness_ratio,
            "target_uniqueness": target_stats.uniqueness_ratio,
            "source_null_ratio": source_stats.null_ratio,
            "target_null_ratio": target_stats.null_ratio,
            "value_overlap": value_overlap,
            "name_similarity": name_similarity,
        }
        
        # Determine if passed
        features["passed"] = (
            type_compatible and
            cardinality_ratio >= self.config.min_cardinality_ratio and
            target_stats.uniqueness_ratio >= self.config.min_target_uniqueness
        )
        
        return features
    
    def filter_candidates(
        self,
        candidates: List[FKCandidate],
        columns_by_table: Dict[str, Dict[str, ColumnMetadata]],
    ) -> List[FKCandidate]:
        passed = []
        
        for candidate in candidates:
            source_col = columns_by_table[candidate.source_table].get(candidate.source_column)
            target_col = columns_by_table[candidate.target_table].get(candidate.target_column)
            
            if not source_col or not target_col:
                continue
            
            source_stats = self.get_stats(candidate.source_table, candidate.source_column)
            target_stats = self.get_stats(candidate.target_table, candidate.target_column)
            
            features = self.compute_features(source_col, target_col, source_stats, target_stats)
            candidate.stage_scores["stage1"] = features
            
            if features["passed"]:
                candidate.score = (
                    features["type_score"] * 0.15 +
                    min(features["cardinality_ratio"], 1.0) * 0.15 +
                    features["target_uniqueness"] * 0.25 +
                    features["name_similarity"] * 0.30 +
                    features["value_overlap"] * 0.15
                )
                passed.append(candidate)
        
        return passed


# =============================================================================
# Stage 2: Mathematical Scoring
# =============================================================================

class Stage2MathematicalScorer:
    """Score candidates using geometric mean of features."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def score_candidate(self, candidate: FKCandidate) -> Dict[str, Any]:
        stage1 = candidate.stage_scores.get("stage1", {})
        
        scores = {
            "name_score": stage1.get("name_similarity", 0),
            "type_score": stage1.get("type_score", 0),
            "cardinality_score": min(stage1.get("cardinality_ratio", 0), 1.0),
            "uniqueness_score": stage1.get("target_uniqueness", 0),
            "overlap_score": stage1.get("value_overlap", 0),
        }
        
        # Geometric mean
        all_scores = [v for v in scores.values() if v > 0]
        scores["geometric_mean"] = geometric_mean(all_scores) if all_scores else 0
        
        # Weighted final score
        scores["final_score"] = (
            scores["name_score"] * 0.30 +
            scores["type_score"] * 0.10 +
            scores["cardinality_score"] * 0.15 +
            scores["uniqueness_score"] * 0.20 +
            scores["overlap_score"] * 0.10 +
            scores["geometric_mean"] * 0.15
        )
        
        scores["passed"] = scores["final_score"] >= self.config.stage2_min_score
        
        return scores
    
    def score_candidates(self, candidates: List[FKCandidate]) -> List[FKCandidate]:
        passed = []
        
        for candidate in candidates:
            scores = self.score_candidate(candidate)
            candidate.stage_scores["stage2"] = scores
            candidate.score = scores["final_score"]
            
            if scores["passed"]:
                passed.append(candidate)
        
        passed.sort(key=lambda c: c.score, reverse=True)
        return passed


# =============================================================================
# Stage 3: Data Sampling
# =============================================================================

class Stage3DataSampler:
    """Validate candidates with actual data sampling."""
    
    def __init__(self, connector, config: PipelineConfig):
        self.connector = connector
        self.config = config
    
    def sample_candidate(self, candidate: FKCandidate) -> Dict[str, Any]:
        result = self.connector.check_referential_integrity(
            source_table=candidate.source_table,
            source_column=candidate.source_column,
            target_table=candidate.target_table,
            target_column=candidate.target_column,
            sample_size=self.config.sample_size,
        )
        
        sampling = {
            "sample_size": result["sample_size"],
            "matches": result["matches"],
            "match_rate": result["match_rate"],
            "orphans": result["orphans"],
            "passed": result["match_rate"] >= self.config.min_match_rate,
        }
        
        return sampling
    
    def validate_candidates(self, candidates: List[FKCandidate]) -> List[FKCandidate]:
        passed = []
        
        for candidate in candidates:
            sampling = self.sample_candidate(candidate)
            candidate.stage_scores["stage3"] = sampling
            
            if sampling["passed"]:
                # Heavy weight on data validation
                candidate.score = candidate.score * 0.3 + sampling["match_rate"] * 0.7
                passed.append(candidate)
        
        return passed


# =============================================================================
# Stage 4: Graph Validation
# =============================================================================

class DirectedGraph:
    """Simple directed graph for cycle detection."""
    
    def __init__(self):
        self.edges: Dict[str, Set[str]] = defaultdict(set)
        self.nodes: Set[str] = set()
    
    def add_edge(self, source: str, target: str):
        self.edges[source].add(target)
        self.nodes.add(source)
        self.nodes.add(target)
    
    def has_cycle_with_edge(self, source: str, target: str) -> bool:
        """Check if adding this edge would create a cycle."""
        # Temporarily add edge
        self.edges[source].add(target)
        self.nodes.add(source)
        self.nodes.add(target)
        
        # DFS to detect cycle
        visited = set()
        rec_stack = set()
        
        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.edges.get(node, set()):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        has_cycle = False
        for node in self.nodes:
            if node not in visited:
                if dfs(node):
                    has_cycle = True
                    break
        
        # Remove temporary edge
        self.edges[source].discard(target)
        
        return has_cycle


class Stage4GraphValidator:
    """Validate graph structure - detect cycles, determine cardinality."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def determine_cardinality(
        self, 
        source_uniqueness: float,
        target_uniqueness: float,
    ) -> RelationshipCardinality:
        if target_uniqueness < 0.95:
            return RelationshipCardinality.UNKNOWN
        if source_uniqueness > 0.95:
            return RelationshipCardinality.ONE_TO_ONE
        return RelationshipCardinality.ONE_TO_MANY
    
    def validate_candidates(self, candidates: List[FKCandidate]) -> List[FKCandidate]:
        graph = DirectedGraph()
        passed = []
        
        # Sort by score to prioritize higher scoring candidates
        candidates_sorted = sorted(candidates, key=lambda c: c.score, reverse=True)
        
        for candidate in candidates_sorted:
            source_key = f"{candidate.source_table}.{candidate.source_column}"
            target_key = f"{candidate.target_table}.{candidate.target_column}"
            
            # Check for cycles
            is_cycle = False
            if not self.config.allow_cycles:
                is_cycle = graph.has_cycle_with_edge(source_key, target_key)
            
            stage1 = candidate.stage_scores.get("stage1", {})
            stage3 = candidate.stage_scores.get("stage3", {})
            
            cardinality = self.determine_cardinality(
                stage1.get("source_uniqueness", 0),
                stage1.get("target_uniqueness", 0),
            )
            
            validation = {
                "is_cycle": is_cycle,
                "cardinality": cardinality.value,
                "final_score": candidate.score if not is_cycle else 0.0,
                "passed": not is_cycle and candidate.score >= self.config.min_final_score,
            }
            
            candidate.stage_scores["stage4"] = validation
            candidate.cardinality = cardinality
            
            if validation["passed"]:
                graph.add_edge(source_key, target_key)
                candidate.confidence = candidate.score
                candidate.is_valid = True
                passed.append(candidate)
        
        passed.sort(key=lambda c: c.score, reverse=True)
        return passed


# =============================================================================
# Stage 5: Semantic Filtering
# =============================================================================

class SemanticFilter:
    """Filter based on semantic analysis - remove false positives."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def check_candidate(
        self,
        source_table: str,
        source_column: str,
        target_table: str,
        target_column: str,
    ) -> Tuple[bool, str]:
        """
        FIXED: More strict semantic filtering to prevent false positives.
        - Rejects same-table self-references
        - Requires entity name to match target table
        - Proper plural handling (category → categories)
        """
        # REJECT same-table self-references
        if source_table.lower() == target_table.lower():
            return False, f"Self-reference to same table '{source_table}'"
        
        # Check for value columns
        if self.config.filter_value_columns:
            if is_value_column(source_column):
                return False, f"Source '{source_column}' is a value column"
        
        # Extract entity from source column (e.g., customer_id → customer)
        entity = extract_entity_from_column(source_column)
        target_lower = target_table.lower()
        
        if entity:
            # Proper plural handling
            def pluralize(word):
                """Handle common English pluralization rules."""
                if word.endswith('y'):
                    return word[:-1] + 'ies'  # category → categories
                elif word.endswith(('s', 'x', 'z', 'ch', 'sh')):
                    return word + 'es'
                else:
                    return word + 's'
            
            def singularize(word):
                """Handle common English singularization rules."""
                if word.endswith('ies'):
                    return word[:-3] + 'y'  # categories → category
                elif word.endswith('es'):
                    return word[:-2]
                elif word.endswith('s'):
                    return word[:-1]
                return word
            
            # Check if entity matches target table
            entity_plural = pluralize(entity)
            target_singular = singularize(target_lower)
            
            entity_matches = (
                entity == target_lower or               # exact: customer == customer
                entity == target_singular or            # customer == customer (from customers)
                entity_plural == target_lower           # categories == categories
            )
            
            if entity_matches:
                return True, f"Entity '{entity}' matches table '{target_table}'"
            else:
                return False, f"Entity '{entity}' doesn't match table '{target_table}'"
        
        # No entity extracted (column doesn't end in _id)
        if target_column.lower() == 'id':
            return False, "No entity in source column name"
        
        return True, "OK"
    
    def filter_candidates(self, candidates: List[FKCandidate]) -> List[FKCandidate]:
        passed = []
        
        for candidate in candidates:
            is_valid, reason = self.check_candidate(
                candidate.source_table,
                candidate.source_column,
                candidate.target_table,
                candidate.target_column,
            )
            
            candidate.stage_scores["semantic"] = {
                "passed": is_valid,
                "reason": reason,
            }
            
            if is_valid:
                passed.append(candidate)
        
        return passed


# =============================================================================
# Main Pipeline
# =============================================================================

class FKDetectionPipeline:
    """Complete FK detection pipeline with all 5 stages."""
    
    def __init__(
        self,
        datasource: DataSourceConfig,
        config: Optional[PipelineConfig] = None,
    ):
        self.datasource = datasource
        self.config = config or PipelineConfig()
        self.connector = create_connector(datasource)
    
    def generate_candidates(
        self,
        columns_by_table: Dict[str, Dict[str, ColumnMetadata]],
        pk_columns: Dict[str, List[str]],
    ) -> List[FKCandidate]:
        """Generate initial FK candidates."""
        candidates = []
        
        # Get all PK/unique columns as potential targets
        target_columns = []
        for table, columns in columns_by_table.items():
            pks = set(pk_columns.get(table, []))
            for col_name, col in columns.items():
                if col.is_primary_key or col.is_unique or col_name in pks:
                    target_columns.append((table, col_name, col))
        
        # For each table, find potential FK columns
        for source_table, columns in columns_by_table.items():
            for source_col_name, source_col in columns.items():
                # Skip PKs as source
                if source_col.is_primary_key:
                    continue
                
                # Check against all potential targets
                for target_table, target_col_name, target_col in target_columns:
                    # FIXED: Skip ALL self-table references (not just same column)
                    # A FK should reference a DIFFERENT table
                    if source_table == target_table:
                        continue
                    
                    candidates.append(FKCandidate(
                        source_table=source_table,
                        source_column=source_col_name,
                        target_table=target_table,
                        target_column=target_col_name,
                    ))
        
        return candidates
    
    def run(self, tables: Optional[List[str]] = None) -> DiscoveryResult:
        """Run the complete FK detection pipeline."""
        start_time = time.time()
        
        result = DiscoveryResult(
            datasource_id=self.datasource.id,
            datasource_name=self.datasource.name,
        )
        
        # Get tables
        if tables:
            table_list = tables
        else:
            table_list = self.connector.get_tables()
        
        result.tables = table_list
        result.tables_scanned = len(table_list)
        
        # Get metadata for all tables
        columns_by_table: Dict[str, Dict[str, ColumnMetadata]] = {}
        pk_columns: Dict[str, List[str]] = {}
        
        for table in table_list:
            metadata = self.connector.get_table_metadata(table)
            columns_by_table[table] = {c.column_name: c for c in metadata.columns}
            pk_columns[table] = metadata.primary_key_columns
            result.total_columns += len(metadata.columns)
        
        # Generate candidates
        candidates = self.generate_candidates(columns_by_table, pk_columns)
        result.initial_candidates = len(candidates)
        
        # Stage 1: Statistical filtering
        stage1 = Stage1StatisticalFilter(self.connector, self.config)
        candidates = stage1.filter_candidates(candidates, columns_by_table)
        result.stage1_passed = len(candidates)
        
        # Stage 2: Mathematical scoring
        stage2 = Stage2MathematicalScorer(self.config)
        candidates = stage2.score_candidates(candidates)
        result.stage2_passed = len(candidates)
        
        # Stage 3: Data sampling
        stage3 = Stage3DataSampler(self.connector, self.config)
        candidates = stage3.validate_candidates(candidates)
        result.stage3_passed = len(candidates)
        
        # Stage 4: Graph validation
        stage4 = Stage4GraphValidator(self.config)
        candidates = stage4.validate_candidates(candidates)
        
        # Stage 5: Semantic filtering
        semantic = SemanticFilter(self.config)
        candidates = semantic.filter_candidates(candidates)
        result.final_candidates = len(candidates)
        
        result.discovered_fks = candidates
        result.completed_at = datetime.now()
        result.duration_seconds = time.time() - start_time
        
        return result


def run_discovery(
    host: str = "localhost",
    port: int = 5432,
    database: str = "orders",
    username: str = "saphenia",
    password: str = "secret",
    schema: str = "public",
    min_match_rate: float = 0.95,
    min_score: float = 0.5,
) -> Dict[str, Any]:
    """
    Run FK discovery and return formatted results.
    
    This is the main entry point for the agent tool.
    """
    config = DataSourceConfig(
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
        schema_name=schema,
    )
    
    pipeline_config = PipelineConfig(
        min_match_rate=min_match_rate,
        min_final_score=min_score,
    )
    
    pipeline = FKDetectionPipeline(config, pipeline_config)
    result = pipeline.run()
    
    # Format results for agent
    output = {
        "summary": {
            "tables_scanned": result.tables_scanned,
            "total_columns": result.total_columns,
            "initial_candidates": result.initial_candidates,
            "stage1_statistical_passed": result.stage1_passed,
            "stage2_mathematical_passed": result.stage2_passed,
            "stage3_sampling_passed": result.stage3_passed,
            "final_fks_discovered": result.final_candidates,
            "duration_seconds": round(result.duration_seconds, 2),
        },
        "discovered_fks": [],
    }
    
    for fk in result.discovered_fks:
        stage1 = fk.stage_scores.get("stage1", {})
        stage2 = fk.stage_scores.get("stage2", {})
        stage3 = fk.stage_scores.get("stage3", {})
        
        output["discovered_fks"].append({
            "relationship": f"{fk.source_table}.{fk.source_column} → {fk.target_table}.{fk.target_column}",
            "confidence": round(fk.confidence, 4),
            "cardinality": fk.cardinality.value if fk.cardinality else "unknown",
            "scores": {
                "name_similarity": round(stage1.get("name_similarity", 0), 3),
                "type_score": round(stage1.get("type_score", 0), 3),
                "uniqueness": round(stage1.get("target_uniqueness", 0), 3),
                "geometric_mean": round(stage2.get("geometric_mean", 0), 3),
                "match_rate": round(stage3.get("match_rate", 0), 3),
            },
        })
    
    return output
