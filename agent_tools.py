#!/usr/bin/env python3
"""
GraphWeaver Agent Tools - All tool definitions and global connections.

This module contains:
- Global lazy singleton connections (PostgreSQL, Neo4j, Fuseki, etc.)
- All @tool-decorated functions organized by category
- System prompt
- ALL_TOOLS list
"""
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mcp_servers"))

from langchain.tools import tool

from graphweaver_agent import (
    DataSourceConfig, Neo4jConfig, PostgreSQLConnector,
    Neo4jClient, GraphBuilder, GraphAnalyzer,
)
from graphweaver_agent.discovery.pipeline import run_discovery, FKDetectionPipeline, PipelineConfig
from graphweaver_agent.business_rules import (
    BusinessRulesExecutor, BusinessRulesConfig, BusinessRule, MarquezClient,
    import_lineage_to_neo4j, generate_sample_rules,
)
from graphweaver_agent.rdf import (
    FusekiClient, RDFSyncManager, sync_neo4j_to_rdf,
    GraphWeaverOntology, SPARQLQueryBuilder, PREFIXES_SPARQL
)

try:
    from graphweaver_agent.ltn import (
        LTNRuleLearner, BusinessRuleGenerator, LTNKnowledgeBase,
        LearnedRule, GeneratedRule, RuleLearningConfig,
    )
    LTN_AVAILABLE = True
except ImportError:
    LTN_AVAILABLE = False

try:
    from graphweaver_agent.embeddings.text_embeddings import TextEmbedder, embed_all_metadata
    from graphweaver_agent.embeddings.kg_embeddings import KGEmbedder, generate_all_kg_embeddings
    from graphweaver_agent.embeddings.vector_indexes import VectorIndexManager
    from graphweaver_agent.embeddings.semantic_fk import SemanticFKDiscovery
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


# =============================================================================
# Global Connections (Lazy Singletons)
# =============================================================================

_pg: Optional[PostgreSQLConnector] = None
_neo4j: Optional[Neo4jClient] = None
_pg_config: Optional[DataSourceConfig] = None
_text_embedder = None
_kg_embedder = None
_fuseki: Optional[FusekiClient] = None
_sparql: Optional[SPARQLQueryBuilder] = None
_rule_learner = None
_rule_generator = None
_rules_config: Optional[BusinessRulesConfig] = None
_marquez_client: Optional[MarquezClient] = None
_registry = None


def get_pg() -> PostgreSQLConnector:
    global _pg, _pg_config
    if _pg is None:
        if _pg_config is None:
            _pg_config = DataSourceConfig(
                host=os.environ.get("POSTGRES_HOST", "localhost"),
                port=int(os.environ.get("POSTGRES_PORT", "5432")),
                database=os.environ.get("POSTGRES_DB", "orders"),
                username=os.environ.get("POSTGRES_USER", "saphenia"),
                password=os.environ.get("POSTGRES_PASSWORD", "secret"),
            )
        _pg = PostgreSQLConnector(_pg_config)
    return _pg


def get_pg_config() -> DataSourceConfig:
    global _pg_config
    if _pg_config is None:
        _pg_config = DataSourceConfig(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            database=os.environ.get("POSTGRES_DB", "orders"),
            username=os.environ.get("POSTGRES_USER", "saphenia"),
            password=os.environ.get("POSTGRES_PASSWORD", "secret"),
        )
    return _pg_config


def get_neo4j() -> Neo4jClient:
    global _neo4j
    if _neo4j is None:
        _neo4j = Neo4jClient(Neo4jConfig(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            username=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "password"),
        ))
    return _neo4j


def get_text_embedder():
    global _text_embedder
    if _text_embedder is None:
        if not EMBEDDINGS_AVAILABLE:
            return None
        _text_embedder = TextEmbedder()
    return _text_embedder


def get_kg_embedder():
    global _kg_embedder
    if _kg_embedder is None:
        if not EMBEDDINGS_AVAILABLE:
            return None
        _kg_embedder = KGEmbedder(get_neo4j())
    return _kg_embedder


def get_fuseki() -> FusekiClient:
    global _fuseki
    if _fuseki is None:
        _fuseki = FusekiClient()
    return _fuseki


def get_sparql() -> SPARQLQueryBuilder:
    global _sparql
    if _sparql is None:
        _sparql = SPARQLQueryBuilder(get_fuseki())
    return _sparql


def get_marquez() -> MarquezClient:
    global _marquez_client
    if _marquez_client is None:
        _marquez_client = MarquezClient(
            url=os.environ.get("MARQUEZ_URL", "http://localhost:5000")
        )
    return _marquez_client


def get_registry():
    global _registry
    if _registry is None:
        from graphweaver_agent.dynamic_tools.tool_registry import ToolRegistry
        _registry = ToolRegistry(
            os.environ.get("DYNAMIC_TOOLS_DIR",
                          os.path.join(os.path.dirname(__file__), "dynamic_tools"))
        )
    return _registry


def get_rule_learner():
    global _rule_learner
    if _rule_learner is None:
        if not LTN_AVAILABLE:
            return None
        config = RuleLearningConfig(embedding_dim=384, use_text_embeddings=True, use_kg_embeddings=True)
        _rule_learner = LTNRuleLearner(get_neo4j(), config)
    return _rule_learner


def get_rule_generator():
    global _rule_generator
    if _rule_generator is None:
        if not LTN_AVAILABLE:
            return None
        _rule_generator = BusinessRuleGenerator(get_neo4j())
    return _rule_generator


def get_rules_config():
    return _rules_config


def set_rules_config(config):
    global _rules_config
    _rules_config = config


# =============================================================================
# Tools - Dynamic Tool Management
# =============================================================================

@tool
def check_tool_exists(tool_name: str) -> str:
    """Check if a dynamic tool exists in the registry.
    
    Args:
        tool_name: Name of the tool to check
    Returns:
        Whether the tool exists
    """
    return "✓ EXISTS" if get_registry().tool_exists(tool_name) else "✗ NOT FOUND"


@tool
def list_available_tools() -> str:
    """List all available tools - both builtin and dynamic.
    
    Returns a summary of:
    - Builtin tools (database, FK discovery, graph, embeddings, business rules, RDF, LTN)
    - Dynamic tools created by the user
    """
    dynamic = get_registry().list_tools()
    output = "## Available Tools\n\n"
    output += "### Builtin Tools:\n"
    output += "- **Database**: configure_database, test_database_connection, list_database_tables, get_table_schema, get_column_stats\n"
    output += "- **FK Discovery**: run_fk_discovery, analyze_potential_fk, validate_fk_with_data\n"
    output += "- **Graph**: clear_neo4j_graph, add_fk_to_graph, get_graph_stats, analyze_graph_centrality, find_table_communities, predict_missing_fks, run_cypher, connect_datasets_to_tables\n"
    output += "- **Embeddings**: generate_text_embeddings, generate_kg_embeddings, create_vector_indexes, semantic_search_tables, semantic_search_columns, find_similar_tables, find_similar_columns, predict_fks_from_embeddings, semantic_fk_discovery\n"
    output += "- **Business Rules**: show_sample_business_rules, load_business_rules, load_business_rules_from_file, list_business_rules, execute_business_rule, execute_all_business_rules, get_marquez_lineage, list_marquez_jobs, import_lineage_to_graph, analyze_data_flow, find_impact_analysis\n"
    output += "- **RDF/SPARQL**: test_rdf_connection, sync_graph_to_rdf, run_sparql, sparql_list_tables, sparql_get_foreign_keys, sparql_table_lineage, sparql_downstream_impact, sparql_hub_tables, sparql_orphan_tables, sparql_search, get_rdf_statistics, export_rdf_turtle\n"
    output += "- **LTN**: learn_rules_with_ltn, generate_business_rules_from_ltn, generate_all_validation_rules, export_generated_rules_yaml, export_generated_rules_sql, show_ltn_knowledge_base\n"
    output += "\n### Dynamic Tools:\n"
    if dynamic:
        for t in dynamic:
            output += f"- **{t['name']}**: {t.get('description', 'No description')}\n"
    else:
        output += "- None created yet\n"
    return output


@tool
def create_dynamic_tool(name: str, description: str, code: str) -> str:
    """Create a new dynamic tool that can be called later.
    
    The code must define a `run()` function that will be executed when the tool is called.
    
    Args:
        name: Unique name for the tool
        description: What the tool does
        code: Python code defining a run() function
    Returns:
        Success message or error
    """
    r = get_registry()
    if r.tool_exists(name):
        return f"ERROR: Tool '{name}' already exists. Use update_dynamic_tool to modify it."
    if "def run(" not in code:
        return "ERROR: Code must define a run() function."
    try:
        compile(code, name, "exec")
        path = r.create_tool(name, description, code)
        return f"✓ Created tool '{name}' at {path}\n\nYou can now call it with: run_dynamic_tool(tool_name='{name}')"
    except SyntaxError as e:
        return f"ERROR: Syntax error in code: {e}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def run_dynamic_tool(tool_name: str, **kwargs) -> str:
    """Execute a dynamic tool by name.
    
    Args:
        tool_name: Name of the dynamic tool to execute
        **kwargs: Arguments to pass to the tool's run() function
    Returns:
        The tool's output or error message
    """
    r = get_registry()
    if not r.tool_exists(tool_name):
        available = [t['name'] for t in r.list_tools()]
        return f"ERROR: Tool '{tool_name}' not found. Available dynamic tools: {available}"
    try:
        return str(r.execute_tool(tool_name, **kwargs))
    except Exception as e:
        import traceback
        return f"ERROR executing tool: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def get_tool_source(tool_name: str) -> str:
    """Get the source code of a dynamic tool.
    
    Args:
        tool_name: Name of the tool
    Returns:
        The tool's source code or error
    """
    r = get_registry()
    if not r.tool_exists(tool_name):
        return f"ERROR: Tool '{tool_name}' not found"
    return r.get_tool_source(tool_name)


@tool
def update_dynamic_tool(tool_name: str, code: str, description: str = None) -> str:
    """Update an existing dynamic tool's code and/or description.
    
    Args:
        tool_name: Name of the tool to update
        code: New Python code (must define run() function)
        description: Optional new description
    Returns:
        Success message or error
    """
    r = get_registry()
    if not r.tool_exists(tool_name):
        return f"ERROR: Tool '{tool_name}' not found. Use create_dynamic_tool to create it."
    if "def run(" not in code:
        return "ERROR: Code must define a run() function"
    try:
        compile(code, tool_name, "exec")
        r.update_tool(tool_name, code, description)
        return f"✓ Updated tool '{tool_name}'"
    except SyntaxError as e:
        return f"ERROR: Syntax error in code: {e}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def delete_dynamic_tool(tool_name: str) -> str:
    """Delete a dynamic tool.
    
    Args:
        tool_name: Name of the tool to delete
    Returns:
        Success message or error
    """
    r = get_registry()
    if not r.tool_exists(tool_name):
        return f"ERROR: Tool '{tool_name}' not found"
    r.delete_tool(tool_name)
    return f"✓ Deleted tool '{tool_name}'"


# =============================================================================
# Tools - Database Configuration & Exploration
# =============================================================================

@tool
def configure_database(host: str, port: int, database: str, username: str, password: str) -> str:
    """Configure which PostgreSQL database to connect to.
    
    Args:
        host: Database host
        port: Port number (usually 5432)
        database: Database name
        username: Username
        password: Password
    """
    global _pg, _pg_config
    _pg_config = DataSourceConfig(host=host, port=port, database=database, username=username, password=password)
    _pg = None
    return f"✓ Configured database: {username}@{host}:{port}/{database}"


@tool
def test_database_connection() -> str:
    """Test connection to PostgreSQL database. Call this first."""
    result = get_pg().test_connection()
    if result["success"]:
        return f"✓ Connected to database '{result['database']}' as '{result['user']}'"
    return f"✗ Failed: {result['error']}"


@tool
def list_database_tables() -> str:
    """List all tables with row counts. Use to see what tables exist."""
    tables = get_pg().get_tables_with_info()
    output = "Tables:\n"
    for t in tables:
        output += f"  - {t['table_name']}: {t['column_count']} columns, ~{t['row_estimate']} rows\n"
    return output


@tool
def get_table_schema(table_name: str) -> str:
    """Get columns and primary keys for a table.
    
    Args:
        table_name: Name of table to inspect
    """
    meta = get_pg().get_table_metadata(table_name)
    output = f"Table: {table_name} ({meta.row_count} rows)\n"
    output += f"Primary Key: {', '.join(meta.primary_key_columns) or 'None'}\n"
    output += "Columns:\n"
    for col in meta.columns:
        pk = " [PK]" if col.is_primary_key else ""
        output += f"  - {col.column_name}: {col.data_type.value}{pk}\n"
    return output


@tool
def get_column_stats(table_name: str, column_name: str) -> str:
    """Get statistics for a column - uniqueness, nulls, samples.
    
    Args:
        table_name: Table name
        column_name: Column name
    """
    stats = get_pg().get_column_statistics(table_name, column_name)
    return (f"{table_name}.{column_name}:\n"
            f"  Distinct: {stats.distinct_count}/{stats.total_count} ({stats.uniqueness_ratio:.1%})\n"
            f"  Nulls: {stats.null_count} ({stats.null_ratio:.1%})\n"
            f"  Samples: {stats.sample_values[:5]}")


# =============================================================================
# Tools - FK Discovery
# =============================================================================

@tool
def run_fk_discovery(min_match_rate: float = 0.95, min_score: float = 0.5, auto_embed: bool = True) -> str:
    """Run complete 5-stage FK discovery pipeline AND persist results to Neo4j.
    
    This discovers FK relationships and automatically adds them to the Neo4j graph.
    After running this, you can immediately sync to RDF with sync_graph_to_rdf.
    
    Args:
        min_match_rate: Minimum data match rate to confirm FK (default 0.95 = 95%)
        min_score: Minimum score threshold (default 0.5)
        auto_embed: Automatically generate text embeddings after discovery (default True)
    """
    try:
        global _pg_config
        if _pg_config is None:
            _pg_config = DataSourceConfig(
                host=os.environ.get("POSTGRES_HOST", "localhost"),
                port=int(os.environ.get("POSTGRES_PORT", "5432")),
                database=os.environ.get("POSTGRES_DB", "orders"),
                username=os.environ.get("POSTGRES_USER", "saphenia"),
                password=os.environ.get("POSTGRES_PASSWORD", "secret"),
            )
        result = run_discovery(
            host=_pg_config.host, port=_pg_config.port, database=_pg_config.database,
            username=_pg_config.username, password=_pg_config.password,
            schema=_pg_config.schema_name, min_match_rate=min_match_rate, min_score=min_score,
        )
        summary = result["summary"]
        output = "## FK Discovery Results\n\n"
        output += f"- Tables scanned: {summary['tables_scanned']}\n"
        output += f"- Total columns: {summary['total_columns']}\n"
        output += f"- Initial candidates: {summary['initial_candidates']}\n"
        output += f"- **Final FKs discovered: {summary['final_fks_discovered']}**\n"
        output += f"- Duration: {summary['duration_seconds']}s\n\n"
        try:
            neo4j = get_neo4j()
            pg = get_pg()
            embedder = None
            if auto_embed and EMBEDDINGS_AVAILABLE:
                try:
                    embedder = TextEmbedder.get_shared_instance()
                    print("[run_fk_discovery] Auto-embedding enabled")
                except Exception as emb_err:
                    print(f"[run_fk_discovery] WARNING: Could not load embedder: {emb_err}")
            builder = GraphBuilder(neo4j, embedder=embedder)
            if embedder:
                builder.enable_auto_embedding()
            print("[run_fk_discovery] Clearing schema data (preserving lineage)...")
            neo4j.run_write("MATCH ()-[r:FK_TO]->() DELETE r")
            neo4j.run_write("MATCH (c:Column) DETACH DELETE c")
            neo4j.run_write("MATCH (t:Table) DETACH DELETE t")
            neo4j.run_write("MATCH (ds:DataSource) DETACH DELETE ds")
            print("[run_fk_discovery] ✓ Schema cleared")
            tables_added = set()
            table_columns = {}
            all_tables = []
            try:
                all_tables = pg.get_tables()
                for table in all_tables:
                    meta = pg.get_table_metadata(table)
                    table_columns[table] = [c.column_name for c in meta.columns]
            except Exception as meta_err:
                print(f"[run_fk_discovery] Warning: Could not get column metadata: {meta_err}")
            for table in all_tables:
                if table not in tables_added:
                    builder.add_table(table, column_names=table_columns.get(table, []))
                    for col_name in table_columns.get(table, []):
                        builder.add_column(table, col_name)
                    tables_added.add(table)
            fks_added = 0
            if result["discovered_fks"]:
                for fk in result["discovered_fks"]:
                    rel = fk["relationship"]
                    parts = rel.split(" → ")
                    src_parts = parts[0].split(".")
                    tgt_parts = parts[1].split(".")
                    src_table, src_col = src_parts[0], src_parts[1]
                    tgt_table, tgt_col = tgt_parts[0], tgt_parts[1]
                    builder.add_fk_relationship(src_table, src_col, tgt_table, tgt_col, fk["confidence"], fk["cardinality"])
                    fks_added += 1
            output += f"### ✓ Persisted to Neo4j\n"
            output += f"- Tables added: {len(tables_added)}\n"
            total_columns = sum(len(cols) for cols in table_columns.values())
            output += f"- Columns added: {total_columns}\n"
            output += f"- FK relationships added: {fks_added}\n"
            if embedder:
                try:
                    coverage = builder.ensure_embeddings_exist()
                    output += f"\n### ✓ Embeddings Generated\n"
                    output += f"- Total nodes: {coverage['total']}\n"
                    output += f"- With embeddings: {coverage['with_embedding']}\n"
                    output += f"- Coverage: {coverage['coverage']}\n"
                    if coverage['missing'] > 0:
                        output += f"- ⚠ Missing: {coverage['missing']} nodes\n"
                except Exception as cov_err:
                    output += f"\n⚠ Could not verify embeddings: {cov_err}\n"
            else:
                output += "\n⚠ **Embeddings not generated.** Run `generate_text_embeddings` to enable semantic search.\n"
            output += "\n"
        except Exception as e:
            import traceback
            output += f"### ⚠ Neo4j Error: {e}\n```\n{traceback.format_exc()}\n```\n\n"
        output += "### Discovered Foreign Keys\n\n"
        if result["discovered_fks"]:
            for fk in result["discovered_fks"]:
                scores = fk["scores"]
                output += f"**{fk['relationship']}**\n"
                output += f"  - Confidence: {fk['confidence']:.1%}\n"
                output += f"  - Cardinality: {fk['cardinality']}\n"
                output += f"  - Match rate: {scores['match_rate']:.1%}\n\n"
        else:
            output += "No foreign keys discovered.\n"
        output += "\n**Next:** Run `sync_graph_to_rdf` to sync to Fuseki.\n"
        return output
    except Exception as e:
        import traceback
        return f"ERROR in FK discovery: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def discover_and_sync() -> str:
    """One-stop shop: Discover FKs, build Neo4j graph, and sync to RDF Fuseki."""
    try:
        output = "## Running Complete Pipeline\n\n"
        output += "### Step 1: FK Discovery\n"
        discovery_result = run_fk_discovery.func()
        if "ERROR" in discovery_result:
            return discovery_result
        output += "✓ Discovery complete and persisted to Neo4j\n\n"
        output += "### Step 2: Syncing to RDF\n"
        fuseki = get_fuseki()
        neo4j = get_neo4j()
        conn = fuseki.test_connection()
        if not conn.get("success"):
            output += f"⚠ Fuseki unavailable: {conn.get('error')}\n"
            output += "Graph is still in Neo4j at http://localhost:7474\n"
            return output
        fuseki.ensure_dataset_exists()
        stats = sync_neo4j_to_rdf(neo4j, fuseki)
        output += f"- Tables: {stats.get('tables', 0)}\n"
        output += f"- Columns: {stats.get('columns', 0)}\n"
        output += f"- FKs: {stats.get('fks', 0)}\n"
        output += f"- **Total triples: {stats.get('total_triples', 0)}**\n\n"
        output += "### ✓ Complete!\n- Neo4j: http://localhost:7474\n- Fuseki SPARQL: http://localhost:3030\n"
        return output
    except Exception as e:
        import traceback
        return f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def analyze_potential_fk(source_table: str, source_column: str, target_table: str, target_column: str) -> str:
    """Analyze if a column pair could be a FK. Checks types, names, cardinality.
    
    Args:
        source_table: Table with potential FK
        source_column: Column that might be FK
        target_table: Table being referenced
        target_column: Column being referenced (usually PK)
    """
    pg = get_pg()
    source_stats = pg.get_column_statistics(source_table, source_column)
    target_stats = pg.get_column_statistics(target_table, target_column)
    source_meta = pg.get_table_metadata(source_table)
    target_meta = pg.get_table_metadata(target_table)
    source_col = next((c for c in source_meta.columns if c.column_name == source_column), None)
    target_col = next((c for c in target_meta.columns if c.column_name == target_column), None)
    if not source_col or not target_col:
        return "Error: Column not found"
    type_compatible = source_col.data_type == target_col.data_type
    target_unique = target_stats.uniqueness_ratio > 0.95
    output = f"Analysis: {source_table}.{source_column} → {target_table}.{target_column}\n"
    output += f"  Type compatible: {type_compatible}\n"
    output += f"  Target uniqueness: {target_stats.uniqueness_ratio:.1%}\n"
    output += f"  Source distinct: {source_stats.distinct_count}\n"
    output += f"  Target distinct: {target_stats.distinct_count}\n"
    if type_compatible and target_unique:
        output += f"  Recommendation: LIKELY FK - validate with data\n"
    elif type_compatible:
        output += f"  Recommendation: POSSIBLE - target not unique enough\n"
    else:
        output += f"  Recommendation: UNLIKELY - type mismatch\n"
    return output


@tool
def validate_fk_with_data(source_table: str, source_column: str, target_table: str, target_column: str) -> str:
    """Validate FK by checking if values actually exist. The definitive test.
    
    Args:
        source_table: Table with FK
        source_column: FK column
        target_table: Referenced table
        target_column: Referenced column
    """
    pg = get_pg()
    result = pg.check_referential_integrity(source_table, source_column, target_table, target_column)
    if result["match_rate"] >= 0.95:
        verdict = "✓ CONFIRMED FK"
    elif result["match_rate"] >= 0.8:
        verdict = "⚠ LIKELY FK (some orphans)"
    else:
        verdict = "✗ NOT A FK"
    return (f"Validation: {source_table}.{source_column} → {target_table}.{target_column}\n"
            f"  {verdict}\n"
            f"  Match rate: {result['match_rate']:.1%} ({result['matches']}/{result['sample_size']})")


# =============================================================================
# Tools - Neo4j Graph
# =============================================================================

@tool
def clear_neo4j_graph() -> str:
    """Clear all data from Neo4j graph. Use before rebuilding."""
    try:
        GraphBuilder(get_neo4j()).clear_graph()
        return "✓ Graph cleared"
    except Exception as e:
        return f"ERROR clearing graph: {type(e).__name__}: {e}"


@tool
def add_fk_to_graph(source_table: str, source_column: str, target_table: str, target_column: str,
                    score: float, cardinality: str = "1:N") -> str:
    """Add a confirmed FK relationship to the Neo4j graph.
    
    Args:
        source_table: Table with FK
        source_column: FK column
        target_table: Referenced table
        target_column: Referenced column
        score: Confidence score 0-1
        cardinality: 1:1, 1:N, or N:M
    """
    try:
        builder = GraphBuilder(get_neo4j())
        builder.add_table(source_table)
        builder.add_table(target_table)
        builder.add_fk_relationship(source_table, source_column, target_table, target_column, score, cardinality)
        return f"✓ Added: {source_table}.{source_column} → {target_table}.{target_column}"
    except Exception as e:
        return f"ERROR adding FK: {type(e).__name__}: {e}"


@tool
def get_graph_stats() -> str:
    """Get current graph statistics."""
    try:
        stats = GraphAnalyzer(get_neo4j()).get_statistics()
        return f"Graph: {stats['tables']} tables, {stats['columns']} columns, {stats['fks']} FKs"
    except Exception as e:
        return f"ERROR getting stats: {type(e).__name__}: {e}"


@tool
def analyze_graph_centrality() -> str:
    """Find hub tables (fact tables) and authority tables (dimensions)."""
    try:
        result = GraphAnalyzer(get_neo4j()).centrality_analysis()
        output = "Centrality Analysis:\n"
        output += f"  Hub tables (fact/transaction): {result['hub_tables']}\n"
        output += f"  Authority tables (dimension): {result['authority_tables']}\n"
        output += f"  Isolated tables: {result['isolated_tables']}\n"
        return output
    except Exception as e:
        return f"ERROR analyzing centrality: {type(e).__name__}: {e}"


@tool
def find_table_communities() -> str:
    """Find clusters of related tables."""
    try:
        communities = GraphAnalyzer(get_neo4j()).community_detection()
        if not communities:
            return "No communities found."
        output = "Communities:\n"
        for i, c in enumerate(communities):
            output += f"  {i+1}. {', '.join(c['tables'])}\n"
        return output
    except Exception as e:
        return f"ERROR finding communities: {type(e).__name__}: {e}"


@tool
def predict_missing_fks() -> str:
    """Predict missing FKs based on column naming patterns."""
    try:
        predictions = GraphAnalyzer(get_neo4j()).predict_missing_fks()
        if not predictions:
            return "No predictions - graph appears complete."
        output = "Predicted missing FKs:\n"
        for p in predictions:
            output += f"  - {p['source_table']}.{p['source_column']} → {p['target_table']}\n"
        return output
    except Exception as e:
        return f"ERROR predicting FKs: {type(e).__name__}: {e}"


@tool
def run_cypher(query: str) -> str:
    """Run a custom Cypher query on the Neo4j graph database.
    
    Args:
        query: The Cypher query to execute
    Returns:
        Query results as formatted string
    """
    try:
        neo4j = get_neo4j()
    except Exception as e:
        return f"Error: Not connected to Neo4j: {e}"
    try:
        results = neo4j.run_query(query)
        if results is None:
            results = []
        if not results:
            return "Query executed successfully. No results returned (0 rows)."
        output = f"Results ({len(results)} rows):\n"
        for i, row in enumerate(results[:50]):
            output += f"  {dict(row)}\n"
        if len(results) > 50:
            output += f"  ... and {len(results) - 50} more rows"
        return output
    except Exception as e:
        try:
            neo4j.run_write(query)
            return "Write query executed successfully."
        except Exception as e2:
            return f"Error executing query: {e2}"


@tool
def connect_datasets_to_tables() -> str:
    """Connect Dataset nodes to their matching Table nodes in the graph.
    FIXED: Uses fuzzy matching to handle dataset names like 'ecommerce.orders' matching table 'orders'.
    """
    try:
        neo4j = get_neo4j()
        datasets = neo4j.run_query("MATCH (d:Dataset) RETURN d.name as name")
        tables = neo4j.run_query("MATCH (t:Table) RETURN t.name as name")
        print(f"[connect_datasets] Found {len(datasets) if datasets else 0} datasets")
        print(f"[connect_datasets] Found {len(tables) if tables else 0} tables")
        if datasets:
            print(f"[connect_datasets] Sample datasets: {[d['name'] for d in datasets[:5]]}")
        if tables:
            print(f"[connect_datasets] Sample tables: {[t['name'] for t in tables[:5]]}")
        if not datasets or not tables:
            msg = "## Diagnostics\n\n"
            msg += f"- Datasets found: {len(datasets) if datasets else 0}\n"
            msg += f"- Tables found: {len(tables) if tables else 0}\n\n"
            if not datasets:
                msg += "**No Dataset nodes in Neo4j.** Run `import_lineage_to_graph` first.\n"
            if not tables:
                msg += "**No Table nodes in Neo4j.** Run `run_fk_discovery` first.\n"
            return msg
        table_names = {t['name'].lower(): t['name'] for t in tables if t.get('name')}
        connected = []
        for ds in datasets:
            ds_name = ds.get('name', '')
            if not ds_name:
                continue
            possible_names = [ds_name, ds_name.split('.')[-1], ds_name.split('/')[-1], ds_name.split('.')[-1].split('/')[-1]]
            for possible in possible_names:
                possible_lower = possible.lower()
                if possible_lower in table_names:
                    actual_table = table_names[possible_lower]
                    neo4j.run_write("""
                        MATCH (d:Dataset {name: $ds_name})
                        MATCH (t:Table {name: $table_name})
                        MERGE (d)-[:REPRESENTS]->(t)
                    """, {"ds_name": ds_name, "table_name": actual_table})
                    connected.append({"dataset": ds_name, "table": actual_table})
                    print(f"[connect_datasets] ✓ Connected: {ds_name} → {actual_table}")
                    break
        if not connected:
            msg = "No matching Dataset-Table pairs found.\n\n"
            msg += f"Datasets ({len(datasets)}): {[d['name'] for d in datasets[:10]]}\n"
            msg += f"Tables ({len(tables)}): {[t['name'] for t in tables[:10]]}"
            return msg
        output = f"## Connected {len(connected)} Datasets to Tables\n\n"
        for conn in connected:
            output += f"- `{conn['dataset']}` → `{conn['table']}`\n"
        output += "\nThe FK graph and lineage graph are now connected!"
        return output
    except Exception as e:
        import traceback
        return f"ERROR connecting datasets to tables: {type(e).__name__}: {e}\n{traceback.format_exc()}"


# =============================================================================
# Tools - Embeddings & Semantic Search
# =============================================================================

@tool
def generate_text_embeddings() -> str:
    """Generate text embeddings for all tables, columns, jobs, and datasets in the graph.
    Uses the all-MiniLM-L6-v2 model (384 dimensions).
    """
    if not EMBEDDINGS_AVAILABLE:
        return "ERROR: Embeddings not available. Install sentence-transformers: pip install sentence-transformers"
    try:
        print("[generate_text_embeddings] Starting...")
        neo4j = get_neo4j()
        pg = get_pg()
        embedder = get_text_embedder()
        print("[generate_text_embeddings] Calling embed_all_metadata...")
        result = embed_all_metadata(neo4j_client=neo4j, pg_connector=pg, embedder=embedder)
        print(f"[generate_text_embeddings] Done: {result}")
        if "error" in result:
            return f"ERROR generating embeddings: {result['error']}"
        if "warning" in result:
            return f"WARNING: {result['warning']}"
        stats = result.get("stats", result)
        output = "## Text Embeddings Generated\n\n"
        output += f"- Tables embedded: {stats.get('tables', 0)}\n"
        output += f"- Columns embedded: {stats.get('columns', 0)}\n"
        output += f"- Jobs embedded: {stats.get('jobs', 0)}\n"
        output += f"- Datasets embedded: {stats.get('datasets', 0)}\n"
        if stats.get('error_count', 0) > 0:
            output += f"\n⚠️ {stats['error_count']} errors occurred during embedding.\n"
        output += "\nText embeddings are now stored on Neo4j nodes."
        output += "\nYou can now use semantic_search_tables and semantic_search_columns."
        return output
    except Exception as e:
        import traceback
        return f"ERROR generating text embeddings: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def generate_kg_embeddings() -> str:
    """Generate knowledge graph embeddings using Neo4j GDS FastRP algorithm.
    Creates structural embeddings (128 dimensions).
    """
    if not EMBEDDINGS_AVAILABLE:
        return "ERROR: Embeddings not available."
    try:
        print("[generate_kg_embeddings] Starting...")
        neo4j = get_neo4j()
        result = generate_all_kg_embeddings(neo4j)
        print(f"[generate_kg_embeddings] Done: {result}")
        if "error" in result:
            return f"ERROR: {result['error']}"
        output = "## KG Embeddings Generated\n\n"
        output += f"- Graph projected: {result['projection']['nodeCount']} nodes, {result['projection']['relationshipCount']} relationships\n"
        output += f"- Embeddings written: {result['embeddings']['properties_written']} properties\n"
        output += "\nKG embeddings are now stored on Neo4j nodes."
        output += "\nYou can now use find_similar_tables and predict_fks_from_embeddings."
        return output
    except Exception as e:
        import traceback
        return f"ERROR generating KG embeddings: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def create_vector_indexes() -> str:
    """Create Neo4j vector indexes for fast similarity search."""
    if not EMBEDDINGS_AVAILABLE:
        return "ERROR: Embeddings not available."
    try:
        manager = VectorIndexManager(get_neo4j())
        if not manager.check_vector_support():
            return "WARNING: Your Neo4j version may not support vector indexes."
        result = manager.create_all_indexes()
        output = "## Vector Indexes Created\n\n### Text Embedding Indexes:\n"
        for name, success in result["text_indexes"].items():
            output += f"  {'✓' if success else '✗'} {name}\n"
        output += "\n### KG Embedding Indexes:\n"
        for name, success in result["kg_indexes"].items():
            output += f"  {'✓' if success else '✗'} {name}\n"
        return output
    except Exception as e:
        import traceback
        return f"ERROR creating indexes: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def semantic_search_tables(query: str, top_k: int = 5) -> str:
    """Search for tables using natural language.
    
    Args:
        query: Natural language search query
        top_k: Number of results to return
    """
    if not EMBEDDINGS_AVAILABLE:
        return "ERROR: Embeddings not available."
    try:
        embedder = get_text_embedder()
        neo4j = get_neo4j()
        query_emb = embedder.embed_text(query)
        result = neo4j.run_query("""
            MATCH (t:Table) WHERE t.text_embedding IS NOT NULL
            WITH t, gds.similarity.cosine(t.text_embedding, $embedding) AS score
            WHERE score > 0.3 RETURN t.name AS table_name, score ORDER BY score DESC LIMIT $top_k
        """, {"embedding": query_emb.embedding, "top_k": top_k})
        if not result:
            return f"No tables found matching '{query}'. Make sure text embeddings are generated."
        output = f"## Tables matching '{query}':\n\n"
        for r in result:
            output += f"  - **{r['table_name']}** (similarity: {r['score']:.2f})\n"
        return output
    except Exception as e:
        import traceback
        return f"ERROR in semantic search: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def semantic_search_columns(query: str, top_k: int = 10) -> str:
    """Search for columns using natural language.
    
    Args:
        query: Natural language search query
        top_k: Number of results to return
    """
    if not EMBEDDINGS_AVAILABLE:
        return "ERROR: Embeddings not available."
    try:
        embedder = get_text_embedder()
        neo4j = get_neo4j()
        query_emb = embedder.embed_text(query)
        result = neo4j.run_query("""
            MATCH (c:Column)-[:BELONGS_TO]->(t:Table) WHERE c.text_embedding IS NOT NULL
            WITH t, c, gds.similarity.cosine(c.text_embedding, $embedding) AS score
            WHERE score > 0.3 RETURN t.name AS table_name, c.name AS column_name, score
            ORDER BY score DESC LIMIT $top_k
        """, {"embedding": query_emb.embedding, "top_k": top_k})
        if not result:
            return f"No columns found matching '{query}'. Make sure text embeddings are generated."
        output = f"## Columns matching '{query}':\n\n"
        for r in result:
            output += f"  - **{r['table_name']}.{r['column_name']}** (similarity: {r['score']:.2f})\n"
        return output
    except Exception as e:
        import traceback
        return f"ERROR in semantic search: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def find_similar_tables(table_name: str, top_k: int = 5) -> str:
    """Find tables similar to a given table using both text and KG embeddings.
    
    Args:
        table_name: Name of the source table
        top_k: Number of similar tables to find
    """
    if not EMBEDDINGS_AVAILABLE:
        return "ERROR: Embeddings not available."
    try:
        neo4j = get_neo4j()
        result = neo4j.run_query("""
            MATCH (source:Table {name: $name}) MATCH (other:Table)
            WHERE other <> source AND other.text_embedding IS NOT NULL
            WITH source, other,
                 gds.similarity.cosine(source.text_embedding, other.text_embedding) AS text_sim,
                 CASE WHEN source.kg_embedding IS NOT NULL AND other.kg_embedding IS NOT NULL
                   THEN gds.similarity.cosine(source.kg_embedding, other.kg_embedding) ELSE null END AS kg_sim
            WITH other.name AS table_name, text_sim, kg_sim,
                 CASE WHEN kg_sim IS NOT NULL THEN (text_sim + kg_sim) / 2 ELSE text_sim END AS combined_score
            RETURN table_name, text_sim, kg_sim, combined_score ORDER BY combined_score DESC LIMIT $top_k
        """, {"name": table_name, "top_k": top_k})
        if not result:
            return f"No similar tables found for '{table_name}'. Run generate_text_embeddings first."
        output = f"## Tables similar to '{table_name}':\n\n"
        for r in result:
            output += f"  - **{r['table_name']}**\n"
            if r.get('text_sim'):
                output += f"    Text similarity: {r['text_sim']:.2f}\n"
            if r.get('kg_sim'):
                output += f"    Graph similarity: {r['kg_sim']:.2f}\n"
            output += f"    Combined score: {r['combined_score']:.2f}\n"
        return output
    except Exception as e:
        import traceback
        return f"ERROR finding similar tables: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def find_similar_columns(table_name: str, column_name: str, top_k: int = 10) -> str:
    """Find columns similar to a given column using text embeddings.
    
    Args:
        table_name: Table containing the source column
        column_name: Name of the source column
        top_k: Number of similar columns to find
    """
    if not EMBEDDINGS_AVAILABLE:
        return "ERROR: Embeddings not available."
    try:
        neo4j = get_neo4j()
        result = neo4j.run_query("""
            MATCH (source:Column {name: $column_name})-[:BELONGS_TO]->(st:Table {name: $table_name})
            MATCH (other:Column)-[:BELONGS_TO]->(t:Table)
            WHERE other <> source AND other.text_embedding IS NOT NULL AND t.name <> $table_name
            WITH t.name AS table_name, other.name AS column_name,
                 gds.similarity.cosine(source.text_embedding, other.text_embedding) AS score
            RETURN table_name, column_name, score ORDER BY score DESC LIMIT $top_k
        """, {"table_name": table_name, "column_name": column_name, "top_k": top_k})
        if not result:
            return f"No similar columns found for '{table_name}.{column_name}'. Run generate_text_embeddings first."
        output = f"## Columns similar to '{table_name}.{column_name}':\n\n"
        for r in result:
            output += f"  - **{r['table_name']}.{r['column_name']}** (similarity: {r['score']:.2f})\n"
        return output
    except Exception as e:
        import traceback
        return f"ERROR finding similar columns: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def predict_fks_from_embeddings(threshold: float = 0.7, top_k: int = 20) -> str:
    """Predict potential FK relationships using KG embeddings.
    
    Args:
        threshold: Minimum similarity threshold (0-1)
        top_k: Maximum predictions to return
    """
    if not EMBEDDINGS_AVAILABLE:
        return "ERROR: Embeddings not available."
    try:
        kg_embedder = get_kg_embedder()
        predictions = kg_embedder.predict_missing_links(threshold=threshold, top_k=top_k)
        if not predictions:
            return "No FK predictions found. The graph may be complete or try lowering the threshold."
        output = f"## Predicted FKs from Graph Structure:\n\n"
        for p in predictions:
            output += f"  - **{p['source_table']}.{p['source_column']}** → **{p['target_table']}.{p['target_column']}**\n"
            output += f"    Similarity: {p['similarity']:.2f}\n"
        output += "\nValidate these with validate_fk_with_data before adding to the graph."
        return output
    except Exception as e:
        import traceback
        return f"ERROR predicting FKs: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def semantic_fk_discovery(source_table: str = None, min_score: float = 0.6) -> str:
    """Discover FK relationships using semantic similarity of column names.
    
    Args:
        source_table: Limit search to this table (optional)
        min_score: Minimum combined score threshold
    """
    if not EMBEDDINGS_AVAILABLE:
        return "ERROR: Embeddings not available."
    try:
        discovery = SemanticFKDiscovery(neo4j_client=get_neo4j(), text_embedder=get_text_embedder(), min_combined_score=min_score)
        candidates = discovery.find_semantic_fk_candidates(source_table=source_table, top_k=30)
        if not candidates:
            return "No semantic FK candidates found. Run generate_text_embeddings first."
        output = "## Semantic FK Candidates:\n\n"
        for c in candidates:
            output += f"**{c.source_table}.{c.source_column}** → **{c.target_table}.{c.target_column}**\n"
            output += f"  Semantic similarity: {c.semantic_similarity:.2f}\n"
            if c.kg_similarity:
                output += f"  Graph similarity: {c.kg_similarity:.2f}\n"
            output += f"  Combined score: {c.combined_score:.2f}\n"
            output += f"  Recommendation: {c.recommendation}\n\n"
        return output
    except Exception as e:
        import traceback
        return f"ERROR in semantic FK discovery: {type(e).__name__}: {e}\n{traceback.format_exc()}"


# =============================================================================
# Tools - Business Rules & Lineage
# =============================================================================

@tool
def show_sample_business_rules() -> str:
    """Show a sample business rules YAML file that you can customize."""
    return generate_sample_rules()


@tool
def load_business_rules(yaml_content: str) -> str:
    """Load business rules from YAML content.
    
    Args:
        yaml_content: YAML string defining business rules
    """
    try:
        import yaml
        data = yaml.safe_load(yaml_content)
        rules = []
        for rule_data in data.get('rules', []):
            rules.append(BusinessRule(**rule_data))
        config = BusinessRulesConfig(version=data.get('version', '1.0'), namespace=data.get('namespace', 'default'), rules=rules)
        set_rules_config(config)
        output = f"✓ Loaded {len(config.rules)} business rules:\n"
        for rule in config.rules:
            output += f"  - {rule.name}: {rule.description} [{rule.type.value}]\n"
            output += f"    Inputs: {', '.join(rule.inputs)}\n"
        return output
    except Exception as e:
        return f"ERROR loading rules: {type(e).__name__}: {e}"


@tool
def load_business_rules_from_file(file_path: str = "business_rules.yaml") -> str:
    """Load business rules from a YAML file on disk.
    
    Args:
        file_path: Path to the YAML file (default: business_rules.yaml)
    """
    try:
        import yaml
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        rules = []
        for rule_data in data.get('rules', []):
            rules.append(BusinessRule(**rule_data))
        config = BusinessRulesConfig(version=data.get('version', '1.0'), namespace=data.get('namespace', 'default'), rules=rules)
        set_rules_config(config)
        output = f"✓ Loaded {len(config.rules)} business rules from {file_path}:\n"
        for rule in config.rules:
            output += f"  - {rule.name}: {rule.description} [{rule.type.value}]\n"
        return output
    except FileNotFoundError:
        return f"ERROR: File '{file_path}' not found"
    except Exception as e:
        return f"ERROR loading rules: {type(e).__name__}: {e}"


@tool
def list_business_rules() -> str:
    """List all loaded business rules."""
    rules_config = get_rules_config()
    if rules_config is None or not rules_config.rules:
        return "No business rules loaded. Use load_business_rules() first."
    output = f"## Business Rules (namespace: {rules_config.namespace})\n\n"
    for rule in rules_config.rules:
        output += f"**{rule.name}** [{rule.type.value}]\n"
        output += f"  {rule.description}\n"
        output += f"  Inputs: {', '.join(rule.inputs)}\n"
        output += f"  Outputs: {', '.join(rule.outputs) if rule.outputs else 'query result'}\n"
        if rule.tags:
            output += f"  Tags: {', '.join(rule.tags)}\n"
        output += "\n"
    return output


@tool
def execute_business_rule(rule_name: str, capture_lineage: bool = True) -> str:
    """Execute a single business rule and optionally capture lineage.
    
    Args:
        rule_name: Name of the rule to execute
        capture_lineage: Whether to send lineage to Marquez (default True)
    """
    rules_config = get_rules_config()
    if rules_config is None:
        return "No business rules loaded. Use load_business_rules() first."
    rule = next((r for r in rules_config.rules if r.name == rule_name), None)
    if not rule:
        return f"Rule '{rule_name}' not found. Available: {[r.name for r in rules_config.rules]}"
    try:
        executor = BusinessRulesExecutor(connector=get_pg(), marquez_url=os.environ.get("MARQUEZ_URL", "http://localhost:5000"), namespace=rules_config.namespace)
        result = executor.execute_rule(rule, emit_lineage=capture_lineage)
        output = f"## Executed: {rule_name}\n\n"
        output += f"**Status:** {result['status']}\n**Duration:** {result['duration_seconds']:.2f}s\n**Rows returned:** {result['rows']}\n"
        if result.get('error'):
            output += f"**Error:** {result['error']}\n"
        if capture_lineage:
            output += f"**Lineage captured:** Run ID {result['run_id']}\n"
        if result.get('columns'):
            output += f"**Columns:** {', '.join(result['columns'])}\n"
        if result.get('metrics'):
            output += "\n### Metrics:\n"
            for col, metrics in result['metrics'].items():
                output += f"  {col}: sum={metrics['sum']:.2f}, avg={metrics['avg']:.2f}, min={metrics['min']:.2f}, max={metrics['max']:.2f}\n"
        if result.get('data') and len(result['data']) > 0:
            output += f"\n### Sample Data (first {min(5, len(result['data']))} rows):\n"
            for row in result['data'][:5]:
                output += f"  {row}\n"
        return output
    except Exception as e:
        import traceback
        return f"ERROR executing rule: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def execute_all_business_rules(capture_lineage: bool = True) -> str:
    """Execute all loaded business rules and capture lineage.
    
    Args:
        capture_lineage: Whether to send lineage to Marquez
    """
    rules_config = get_rules_config()
    if rules_config is None or not rules_config.rules:
        return "No business rules loaded. Use load_business_rules() first."
    try:
        executor = BusinessRulesExecutor(connector=get_pg(), marquez_url=os.environ.get("MARQUEZ_URL", "http://localhost:5000"), namespace=rules_config.namespace)
        results = executor.execute_all_rules(rules_config, emit_lineage=capture_lineage)
        output = f"## Executed {len(results)} Business Rules\n\n"
        success = sum(1 for r in results if r['status'] == 'success')
        output += f"**Results:** {success} succeeded, {len(results) - success} failed\n\n"
        for result in results:
            status_icon = "✓" if result['status'] == 'success' else "✗"
            output += f"{status_icon} **{result['rule_name']}**: {result['rows']} rows, {result['duration_seconds']:.2f}s"
            if result.get('error'):
                output += f" - ERROR: {result['error']}"
            output += "\n"
        if capture_lineage:
            output += f"\n**Lineage captured in Marquez** (namespace: {rules_config.namespace})"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def get_marquez_lineage(dataset_name: str, depth: int = 3) -> str:
    """Get lineage graph for a dataset from Marquez.
    
    Args:
        dataset_name: Name of the dataset/table
        depth: How many levels of lineage to retrieve
    """
    try:
        marquez = get_marquez()
        rules_config = get_rules_config()
        namespace = rules_config.namespace if rules_config else "default"
        lineage = marquez.get_dataset_lineage(namespace, dataset_name, depth)
        if not lineage:
            return f"No lineage found for {dataset_name}"
        output = f"## Lineage for {dataset_name}\n\n"
        graph = lineage.get("graph", [])
        for node in graph:
            node_type = node.get("type", "")
            node_id = node.get("id", "")
            if node_type == "DATASET":
                output += f"📊 Dataset: {node.get('data', {}).get('name', node_id)}\n"
            elif node_type == "JOB":
                output += f"⚙️ Job: {node.get('data', {}).get('name', node_id)}\n"
            in_edges = node.get("inEdges", [])
            out_edges = node.get("outEdges", [])
            if in_edges:
                output += f"   ← reads from: {len(in_edges)} sources\n"
            if out_edges:
                output += f"   → writes to: {len(out_edges)} targets\n"
        return output
    except Exception as e:
        return f"ERROR getting lineage: {type(e).__name__}: {e}"


@tool
def list_marquez_jobs() -> str:
    """List all jobs tracked in Marquez."""
    try:
        marquez = get_marquez()
        rules_config = get_rules_config()
        namespace = rules_config.namespace if rules_config else "default"
        jobs = marquez.get_jobs(namespace)
        if not jobs:
            return "No jobs found in Marquez."
        output = f"## Jobs in Marquez (namespace: {namespace})\n\n"
        for job in jobs:
            output += f"**{job.get('name')}**\n"
            output += f"  Inputs: {[i.get('name') for i in job.get('inputs', [])]}\n"
            output += f"  Outputs: {[o.get('name') for o in job.get('outputs', [])]}\n"
            if job.get('latestRun'):
                run = job['latestRun']
                output += f"  Last run: {run.get('state')} at {run.get('startedAt')}\n"
            output += "\n"
        return output
    except Exception as e:
        return f"ERROR listing jobs: {type(e).__name__}: {e}"


@tool
def import_lineage_to_graph() -> str:
    """Import lineage data from Marquez into Neo4j graph."""
    try:
        marquez = get_marquez()
        neo4j = get_neo4j()
        rules_config = get_rules_config()
        namespace = rules_config.namespace if rules_config else "default"
        stats = import_lineage_to_neo4j(marquez, neo4j, namespace)
        output = "## Imported Lineage to Neo4j\n\n"
        output += f"- Jobs created: {stats['jobs']}\n- Datasets linked: {stats['datasets']}\n"
        output += f"- READS relationships: {stats['reads']}\n- WRITES relationships: {stats['writes']}\n"
        output += "\nThe graph now contains both FK relationships AND data lineage!"
        return output
    except Exception as e:
        return f"ERROR importing lineage: {type(e).__name__}: {e}"


@tool
def analyze_data_flow(table_name: str) -> str:
    """Analyze complete data flow for a table - both FKs and lineage.
    
    Args:
        table_name: Name of the table to analyze
    """
    try:
        neo4j = get_neo4j()
        output = f"## Data Flow Analysis: {table_name}\n\n"
        fk_out = neo4j.run_query("MATCH (t:Table {name: $name})<-[:BELONGS_TO]-(c:Column)-[fk:FK_TO]->(tc:Column)-[:BELONGS_TO]->(tt:Table) RETURN c.name as column, tt.name as references_table, tc.name as references_column, fk.score as score", {"name": table_name})
        if fk_out:
            output += "### References (FK →)\n"
            for row in fk_out:
                output += f"  {row['column']} → {row['references_table']}.{row['references_column']}"
                if row.get('score'):
                    output += f" (score: {row['score']:.2f})"
                output += "\n"
            output += "\n"
        fk_in = neo4j.run_query("MATCH (st:Table)<-[:BELONGS_TO]-(sc:Column)-[fk:FK_TO]->(tc:Column)-[:BELONGS_TO]->(t:Table {name: $name}) RETURN st.name as source_table, sc.name as source_column, tc.name as column, fk.score as score", {"name": table_name})
        if fk_in:
            output += "### Referenced By (FK ←)\n"
            for row in fk_in:
                output += f"  {row['source_table']}.{row['source_column']} → {row['column']}"
                if row.get('score'):
                    output += f" (score: {row['score']:.2f})"
                output += "\n"
            output += "\n"
        readers = neo4j.run_query("MATCH (j:Job)-[:READS]->(d:Dataset {name: $name}) RETURN j.name as job_name, j.description as description", {"name": table_name})
        if readers:
            output += "### Jobs Reading This Table\n"
            for row in readers:
                output += f"  ⚙️ {row['job_name']}"
                if row.get('description'):
                    output += f" - {row['description']}"
                output += "\n"
            output += "\n"
        writers = neo4j.run_query("MATCH (j:Job)-[:WRITES]->(d:Dataset {name: $name}) RETURN j.name as job_name, j.description as description", {"name": table_name})
        if writers:
            output += "### Jobs Writing This Table\n"
            for row in writers:
                output += f"  ⚙️ {row['job_name']}"
                if row.get('description'):
                    output += f" - {row['description']}"
                output += "\n"
            output += "\n"
        if not (fk_out or fk_in or readers or writers):
            output += "No relationships found. Run FK discovery and/or execute business rules first."
        return output
    except Exception as e:
        return f"ERROR analyzing data flow: {type(e).__name__}: {e}"


@tool
def find_impact_analysis(table_name: str) -> str:
    """Find all downstream impacts if a table changes.
    
    Args:
        table_name: Table to analyze impact for
    """
    try:
        neo4j = get_neo4j()
        output = f"## Impact Analysis: What breaks if '{table_name}' changes?\n\n"
        dependent_tables = neo4j.run_query("MATCH (t:Table {name: $name})<-[:BELONGS_TO]-(c:Column)<-[:FK_TO]-(fc:Column)-[:BELONGS_TO]->(ft:Table) RETURN DISTINCT ft.name as table_name", {"name": table_name})
        if dependent_tables:
            output += "### Dependent Tables (via FK)\n"
            for row in dependent_tables:
                output += f"  📊 {row['table_name']}\n"
            output += "\n"
        dependent_jobs = neo4j.run_query("MATCH (j:Job)-[:READS]->(d:Dataset {name: $name}) RETURN j.name as job_name", {"name": table_name})
        if dependent_jobs:
            output += "### Jobs That Read This Table\n"
            for row in dependent_jobs:
                output += f"  ⚙️ {row['job_name']}\n"
            output += "\n"
        downstream = neo4j.run_query("MATCH (d1:Dataset {name: $name})<-[:READS]-(j:Job)-[:WRITES]->(d2:Dataset) RETURN DISTINCT j.name as job_name, d2.name as output_dataset", {"name": table_name})
        if downstream:
            output += "### Downstream Datasets (via Jobs)\n"
            for row in downstream:
                output += f"  {row['job_name']} → {row['output_dataset']}\n"
            output += "\n"
        total = len(dependent_tables or []) + len(dependent_jobs or []) + len(downstream or [])
        output += f"**Total potential impacts: {total}**"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# =============================================================================
# Tools - RDF / SPARQL
# =============================================================================

@tool
def debug_rdf_sync() -> str:
    """DEBUG: Test RDF sync step by step with detailed logging."""
    import requests
    output = "## RDF Debug Report\n\n"
    try:
        fuseki = get_fuseki()
        output += f"### Step 1: Connection Test\n- URL: {fuseki.config.url}\n- Dataset: {fuseki.config.dataset}\n- Base URL: {fuseki.base_url}\n"
        try:
            resp = requests.get(f"{fuseki.config.url}/$/ping", timeout=5)
            output += f"- Ping status: {resp.status_code}\n"
        except Exception as e:
            output += f"- Ping FAILED: {e}\n"; return output
        output += "\n### Step 2: Dataset Check\n"
        try:
            resp = requests.get(f"{fuseki.config.url}/$/datasets/{fuseki.config.dataset}", auth=fuseki.auth, timeout=5)
            output += f"- Dataset exists: {resp.status_code == 200}\n"
            if resp.status_code != 200: output += f"- Response: {resp.text[:200]}\n"
        except Exception as e:
            output += f"- Dataset check FAILED: {e}\n"
        output += "\n### Step 3: Insert Test Triple\n"
        test_turtle = '@prefix gw: <http://graphweaver.io/ontology#> .\n@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n<http://graphweaver.io/data#test_table> a gw:Table ; rdfs:label "TEST_TABLE" .\n'
        graph_uri = "http://graphweaver.io/graph/main"
        url = f"{fuseki.base_url}/data?graph={graph_uri}"
        output += f"- Insert URL: {url}\n"
        try:
            resp = requests.post(url, data=test_turtle.encode('utf-8'), headers={"Content-Type": "text/turtle; charset=utf-8"}, auth=fuseki.auth, timeout=10)
            output += f"- Insert status: {resp.status_code}\n"
            if resp.status_code not in [200, 201, 204]: output += f"- Insert response: {resp.text[:300]}\n"
        except Exception as e:
            output += f"- Insert FAILED: {e}\n"
        output += "\n### Step 4: Query Test Triple\n"
        query = f'PREFIX gw: <http://graphweaver.io/ontology#>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\nSELECT ?s ?label WHERE {{ GRAPH <{graph_uri}> {{ ?s a gw:Table ; rdfs:label ?label . }} }}'
        try:
            resp = requests.post(f"{fuseki.base_url}/sparql", data={"query": query}, headers={"Accept": "application/sparql-results+json"}, timeout=10)
            output += f"- Query status: {resp.status_code}\n"
            if resp.status_code == 200:
                bindings = resp.json().get("results", {}).get("bindings", [])
                output += f"- Results found: {len(bindings)}\n"
                for b in bindings[:5]: output += f"  - {b.get('label', {}).get('value', 'N/A')}\n"
            else: output += f"- Query response: {resp.text[:200]}\n"
        except Exception as e:
            output += f"- Query FAILED: {e}\n"
        output += "\n### Step 5: Triple Count\n"
        output += f"- Triples in named graph: {fuseki.get_triple_count(graph_uri)}\n"
        output += f"- Triples in default graph: {fuseki.get_triple_count()}\n"
        output += "\n### Step 6: Neo4j Data Check\n"
        try:
            neo4j = get_neo4j()
            tables = neo4j.run_query("MATCH (t:Table) RETURN count(t) as cnt")
            columns = neo4j.run_query("MATCH (c:Column) RETURN count(c) as cnt")
            fks = neo4j.run_query("MATCH ()-[r:FK_TO]->() RETURN count(r) as cnt")
            output += f"- Tables in Neo4j: {tables[0]['cnt'] if tables else 0}\n"
            output += f"- Columns in Neo4j: {columns[0]['cnt'] if columns else 0}\n"
            output += f"- FK relationships: {fks[0]['cnt'] if fks else 0}\n"
        except Exception as e:
            output += f"- Neo4j check FAILED: {e}\n"
        return output
    except Exception as e:
        import traceback
        return f"DEBUG ERROR: {e}\n{traceback.format_exc()}"


@tool
def test_rdf_connection() -> str:
    """Test connection to the RDF triple store (Apache Jena Fuseki)."""
    try:
        fuseki = get_fuseki()
        result = fuseki.test_connection()
        if result["success"]:
            count = fuseki.get_triple_count()
            return f"✓ Connected to Fuseki RDF store\n  Dataset: {fuseki.config.dataset}\n  Triples: {count}"
        else:
            return f"✗ Connection failed: {result.get('error')}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def sync_graph_to_rdf() -> str:
    """Synchronize the entire Neo4j graph to the RDF triple store.
    FIXED: URL-encode graph URI, inline implementation with proper error logging.
    """
    from urllib.parse import quote
    import requests
    errors = []
    print("=" * 60 + "\n  RDF SYNC - INLINE FIXED VERSION\n" + "=" * 60)
    try:
        fuseki = get_fuseki()
        neo4j = get_neo4j()
        print(f"[sync] Fuseki URL: {fuseki.config.url}, Dataset: {fuseki.config.dataset}")
        print(f"[sync] Fuseki auth user: {fuseki.config.username}")
        fuseki_url = fuseki.config.url
        dataset = fuseki.config.dataset
        base_url = f"{fuseki_url}/{dataset}"
        graph_uri = "http://graphweaver.io/graph/main"
        auth = (fuseki.config.username, fuseki.config.password)
        try:
            resp = requests.get(f"{fuseki_url}/$/ping", timeout=5)
            if resp.status_code != 200: return f"ERROR: Fuseki not responding (status {resp.status_code})"
        except Exception as e:
            return f"ERROR: Cannot connect to Fuseki: {e}"
        fuseki.ensure_dataset_exists()
        requests.post(f"{base_url}/update", data={"update": f"CLEAR GRAPH <{graph_uri}>"}, auth=auth, timeout=30)
        PREFIXES = '@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n@prefix gw: <http://graphweaver.io/ontology#> .\n@prefix gwdata: <http://graphweaver.io/data#> .\n'
        def uri_safe(name):
            return name.replace(" ", "_").replace("-", "_").replace(".", "_")
        def insert_turtle(turtle_content, label="unknown"):
            url = f"{base_url}/data?graph={quote(graph_uri, safe='')}"
            print(f"[sync] INSERT {label}: {len(turtle_content)} bytes to {url}")
            resp = requests.post(url, data=turtle_content.encode('utf-8'), headers={"Content-Type": "text/turtle; charset=utf-8"}, auth=auth, timeout=30)
            ok = resp.status_code in [200, 201, 204]
            if ok:
                print(f"[sync] ✓ {label} insert OK ({resp.status_code})")
            else:
                msg = f"[sync] ✗ {label} insert FAILED: {resp.status_code} - {resp.text[:300]}"
                print(msg)
                errors.append(msg)
            return ok
        stats = {"tables": 0, "columns": 0, "fks": 0, "jobs": 0, "datasets": 0}
        # Tables & columns
        tables_result = neo4j.run_query("MATCH (t:Table) OPTIONAL MATCH (t)<-[:BELONGS_TO]-(c:Column) WITH t, collect({name: c.name}) as columns RETURN t.name as name, columns")
        print(f"[sync] Neo4j tables query returned: {len(tables_result) if tables_result else 'None/empty'}")
        if tables_result:
            turtle_lines = [PREFIXES]
            for table in tables_result:
                tn = table.get("name")
                if not tn: continue
                tu = f"gwdata:table_{uri_safe(tn)}"
                turtle_lines.append(f'{tu} a gw:Table ; rdfs:label "{tn}" .')
                stats["tables"] += 1
                for col in table.get("columns", []):
                    cn = col.get("name")
                    if cn:
                        cu = f"gwdata:column_{uri_safe(tn)}_{uri_safe(cn)}"
                        turtle_lines.append(f'{cu} a gw:Column ; rdfs:label "{cn}" ; gw:belongsToTable {tu} .')
                        turtle_lines.append(f'{tu} gw:hasColumn {cu} .')
                        stats["columns"] += 1
            if not insert_turtle("\n".join(turtle_lines), "tables+columns"): return "ERROR: Failed to insert tables"
        else:
            errors.append("Neo4j tables query returned no results - graph may be empty")
        # FKs
        fks_result = neo4j.run_query("MATCH (sc:Column)-[fk:FK_TO]->(tc:Column) MATCH (sc)-[:BELONGS_TO]->(st:Table) MATCH (tc)-[:BELONGS_TO]->(tt:Table) RETURN st.name as source_table, sc.name as source_column, tt.name as target_table, tc.name as target_column")
        if fks_result:
            turtle_lines = [PREFIXES]
            for fk in fks_result:
                turtle_lines.append(f'gwdata:column_{uri_safe(fk["source_table"])}_{uri_safe(fk["source_column"])} gw:references gwdata:column_{uri_safe(fk["target_table"])}_{uri_safe(fk["target_column"])} .')
                stats["fks"] += 1
            insert_turtle("\n".join(turtle_lines), "foreign_keys")
        # Jobs
        jobs_result = neo4j.run_query("MATCH (j:Job) OPTIONAL MATCH (j)-[:READS]->(input:Dataset) OPTIONAL MATCH (j)-[:WRITES]->(output:Dataset) WITH j, collect(DISTINCT input.name) as inputs, collect(DISTINCT output.name) as outputs RETURN j.name as name, inputs, outputs")
        if jobs_result:
            turtle_lines = [PREFIXES]
            dataset_names = set()
            for job in jobs_result:
                jn = job.get("name")
                if not jn: continue
                ju = f"gwdata:job_{uri_safe(jn)}"
                turtle_lines.append(f'{ju} a gw:Job ; rdfs:label "{jn}" .')
                stats["jobs"] += 1
                for ds in job.get("inputs", []):
                    if ds:
                        du = f"gwdata:dataset_{uri_safe(ds)}"
                        if ds not in dataset_names:
                            turtle_lines.append(f'{du} a gw:Dataset ; rdfs:label "{ds}" .')
                            dataset_names.add(ds)
                        turtle_lines.append(f'{ju} gw:readsFrom {du} .')
                for ds in job.get("outputs", []):
                    if ds:
                        du = f"gwdata:dataset_{uri_safe(ds)}"
                        if ds not in dataset_names:
                            turtle_lines.append(f'{du} a gw:Dataset ; rdfs:label "{ds}" .')
                            dataset_names.add(ds)
                        turtle_lines.append(f'{ju} gw:writesTo {du} .')
            stats["datasets"] = len(dataset_names)
            insert_turtle("\n".join(turtle_lines), "jobs+datasets")
        # REPRESENTS
        represents_result = neo4j.run_query("MATCH (d:Dataset)-[:REPRESENTS]->(t:Table) RETURN d.name as dataset, t.name as table")
        stats["represents"] = 0
        if represents_result:
            turtle_lines = [PREFIXES]
            for rep in represents_result:
                dn, tn = rep.get("dataset"), rep.get("table")
                if dn and tn:
                    turtle_lines.append(f'gwdata:dataset_{uri_safe(dn)} gw:representsTable gwdata:table_{uri_safe(tn)} .')
                    stats["represents"] += 1
            if stats["represents"] > 0: insert_turtle("\n".join(turtle_lines), "represents")
        # Count triples
        count_resp = requests.post(f"{base_url}/sparql", data={"query": f"SELECT (COUNT(*) as ?count) WHERE {{ GRAPH <{graph_uri}> {{ ?s ?p ?o }} }}"}, headers={"Accept": "application/sparql-results+json"}, timeout=30)
        total_triples = 0
        if count_resp.status_code == 200:
            try:
                bindings = count_resp.json().get("results", {}).get("bindings", [])
                if bindings: total_triples = int(bindings[0].get("count", {}).get("value", 0))
            except: pass
        print(f"[sync] Final triple count in named graph: {total_triples}")
        output = "## Synced Graph to RDF\n\n"
        output += f"**Tables:** {stats['tables']} synced\n"
        output += f"**Columns:** {stats['columns']} synced\n"
        output += f"**Foreign Keys:** {stats['fks']} synced\n"
        output += f"**Jobs:** {stats['jobs']} synced\n"
        output += f"**Datasets:** {stats['datasets']} synced\n"
        output += f"**Dataset-Table links:** {stats.get('represents', 0)}\n"
        output += f"\n**Total triples:** {total_triples}\n"
        if total_triples == 0 and stats['tables'] > 0:
            output += "\n⚠️ WARNING: Triples were generated but count is 0 - check Fuseki logs\n"
        if errors:
            output += "\n**Errors:**\n"
            for err in errors:
                output += f"- {err}\n"
        output += f"\nQuery with: GRAPH <{graph_uri}> {{ ?s ?p ?o }}"
        return output
    except Exception as e:
        import traceback
        return f"ERROR syncing to RDF: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def run_sparql(query: str) -> str:
    """Run a custom SPARQL query on the RDF store.
    
    Args:
        query: SPARQL SELECT query
    """
    try:
        sparql = get_sparql()
        results = sparql.custom_query(query)
        if not results: return "Query executed. No results returned."
        output = f"Results ({len(results)} rows):\n"
        for i, row in enumerate(results[:50]):
            output += f"  {row}\n"
        if len(results) > 50: output += f"  ... and {len(results) - 50} more rows"
        return output
    except Exception as e:
        import traceback
        return f"ERROR executing SPARQL: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def sparql_list_tables() -> str:
    """List all tables in the RDF store with column counts."""
    try:
        results = get_sparql().list_tables()
        if not results: return "No tables found. Run sync_graph_to_rdf first."
        output = "## Tables in RDF Store\n\n"
        for r in results:
            output += f"- **{r.get('label', '?')}**: {r.get('columnCount', 0)} columns, {r.get('rowCount', '?')} rows\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def sparql_get_foreign_keys(table_name: str = None) -> str:
    """Get foreign key relationships from RDF store.
    
    Args:
        table_name: Optional - filter by table name
    """
    try:
        results = get_sparql().get_foreign_keys(table_name)
        if not results: return f"No foreign keys found{' for ' + table_name if table_name else ''}."
        output = f"## Foreign Keys{' for ' + table_name if table_name else ''}\n\n"
        for r in results:
            output += f"- **{r.get('sourceTableLabel')}.{r.get('sourceColLabel')}** → **{r.get('targetTableLabel')}.{r.get('targetColLabel')}** (score: {r.get('score', '?')}, cardinality: {r.get('cardinality', '?')})\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def sparql_table_lineage(table_name: str) -> str:
    """Get lineage for a table from RDF store.
    
    Args:
        table_name: Name of the table
    """
    try:
        results = get_sparql().get_table_lineage(table_name)
        if not results: return f"No lineage found for '{table_name}'."
        output = f"## Lineage for {table_name}\n\n"
        reads = [r for r in results if r.get("direction") == "reads"]
        writes = [r for r in results if r.get("direction") == "writes"]
        if reads:
            output += "### Jobs that READ this table:\n"
            for r in reads: output += f"  - {r.get('jobLabel')}\n"
        if writes:
            output += "\n### Jobs that WRITE to this table:\n"
            for r in writes: output += f"  - {r.get('jobLabel')}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def sparql_downstream_impact(table_name: str) -> str:
    """Find downstream impact via RDF/SPARQL.
    
    Args:
        table_name: Name of the table to analyze
    """
    try:
        results = get_sparql().get_downstream_impact(table_name)
        if not results: return f"No downstream dependencies found for '{table_name}'."
        output = f"## Downstream Impact: {table_name}\n\n"
        fk_deps = [r for r in results if r.get("relationshipType") == "FK_REFERENCE"]
        lineage_deps = [r for r in results if r.get("relationshipType") == "LINEAGE"]
        if fk_deps:
            output += "### Tables referencing via FK:\n"
            for r in fk_deps: output += f"  - {r.get('dependentTableLabel')}\n"
        if lineage_deps:
            output += "\n### Downstream via data lineage:\n"
            for r in lineage_deps: output += f"  - {r.get('dependentTableLabel')}\n"
        output += f"\n**Total dependencies: {len(results)}**"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def sparql_hub_tables(min_connections: int = 2) -> str:
    """Find hub tables with many connections (via RDF/SPARQL).
    
    Args:
        min_connections: Minimum connections to be considered a hub
    """
    try:
        results = get_sparql().get_hub_tables(min_connections)
        if not results: return f"No hub tables found with >= {min_connections} connections."
        output = f"## Hub Tables (>= {min_connections} connections)\n\n"
        for r in results:
            output += f"- **{r.get('label')}**: {r.get('totalConnections', 0)} total (in:{r.get('incomingFKs', 0)}, out:{r.get('outgoingFKs', 0)}, reads:{r.get('readByJobs', 0)})\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def sparql_orphan_tables() -> str:
    """Find tables with no FK relationships (via RDF/SPARQL)."""
    try:
        results = get_sparql().find_orphan_tables()
        if not results: return "No orphan tables found - all tables have relationships."
        output = "## Orphan Tables (no FK relationships)\n\n"
        for r in results: output += f"- {r.get('label')}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def sparql_search(search_term: str) -> str:
    """Search the RDF graph by label.
    
    Args:
        search_term: Text to search for in labels
    """
    try:
        results = get_sparql().search_by_label(search_term)
        if not results: return f"No results found for '{search_term}'."
        output = f"## Search results for '{search_term}'\n\n"
        by_type = {}
        for r in results:
            rtype = r.get("type", "").split("#")[-1]
            if rtype not in by_type: by_type[rtype] = []
            by_type[rtype].append(r.get("label"))
        for rtype, labels in by_type.items():
            output += f"### {rtype}s:\n"
            for label in labels: output += f"  - {label}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def get_rdf_statistics() -> str:
    """Get statistics from the RDF store."""
    try:
        fuseki = get_fuseki()
        sparql = get_sparql()
        triple_count = fuseki.get_triple_count("http://graphweaver.io/graph/main")
        stats = sparql.get_statistics()
        output = "## RDF Store Statistics\n\n"
        output += f"- Total triples: {triple_count}\n- Tables: {stats.get('tables', 0)}\n"
        output += f"- Columns: {stats.get('columns', 0)}\n- Foreign Keys: {stats.get('foreignKeys', 0)}\n"
        output += f"- Jobs: {stats.get('jobs', 0)}\n- Datasets: {stats.get('datasets', 0)}\n"
        output += f"\nFuseki UI: http://localhost:3030"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def export_rdf_turtle() -> str:
    """Export the graph ontology in Turtle format."""
    try:
        ontology = GraphWeaverOntology.get_ontology_turtle()
        output = "## GraphWeaver Ontology (Turtle format)\n\n```turtle\n"
        output += ontology[:3000]
        if len(ontology) > 3000: output += "\n... (truncated)"
        output += "\n```"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# =============================================================================
# Tools - LTN (Logic Tensor Networks) Rule Learning
# =============================================================================

@tool
def learn_rules_with_ltn() -> str:
    """Learn logical rules from the knowledge graph using LTN."""
    try:
        if not LTN_AVAILABLE:
            return "LTN not available. Install with: pip install ltn tensorflow"
        learner = get_rule_learner()
        if learner is None:
            return "LTN not available. Install with: pip install ltn tensorflow"
        learned_rules = learner.learn_rules()
        if not learned_rules:
            return "No rules learned. Make sure you have FK discovery results and embeddings generated."
        output = f"## Learned {len(learned_rules)} Rules with LTN\n\n"
        by_type = {}
        for rule in learned_rules:
            rtype = rule.rule_type
            if rtype not in by_type: by_type[rtype] = []
            by_type[rtype].append(rule)
        for rtype, rules in by_type.items():
            output += f"### {rtype.title()} Rules:\n"
            for rule in rules:
                output += f"- **{rule.name}**: `{rule.formula}`\n"
                output += f"  Confidence: {rule.confidence:.2f}, Support: {rule.support}\n"
                if rule.description: output += f"  {rule.description}\n"
            output += "\n"
        output += "\nUse `generate_business_rules_from_ltn` to convert these to executable SQL rules."
        return output
    except Exception as e:
        import traceback
        return f"ERROR learning rules: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def generate_business_rules_from_ltn() -> str:
    """Generate executable business rules from learned LTN patterns."""
    try:
        if not LTN_AVAILABLE:
            return "LTN not available. Install with: pip install ltn tensorflow"
        learner = get_rule_learner()
        generator = get_rule_generator()
        if learner is None or generator is None:
            return "LTN not available. Install with: pip install ltn tensorflow"
        learned_rules = learner.get_learned_rules()
        if not learned_rules:
            learned_rules = learner.learn_rules()
        if not learned_rules:
            return "No learned rules available. Run learn_rules_with_ltn first."
        generated_rules = generator.generate_from_learned_rules(learned_rules)
        if not generated_rules:
            return "No business rules generated."
        output = f"## Generated {len(generated_rules)} Business Rules\n\n"
        by_type = {}
        for rule in generated_rules:
            rtype = rule.rule_type
            if rtype not in by_type: by_type[rtype] = []
            by_type[rtype].append(rule)
        for rtype, rules in by_type.items():
            output += f"### {rtype.title()} Rules ({len(rules)}):\n"
            for rule in rules[:5]:
                output += f"- **{rule.name}**\n  {rule.description}\n  Inputs: {', '.join(rule.inputs)}\n"
            if len(rules) > 5: output += f"  ... and {len(rules) - 5} more\n"
            output += "\n"
        output += "\nUse `export_generated_rules_yaml` to export as YAML file."
        return output
    except Exception as e:
        import traceback
        return f"ERROR generating rules: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def generate_all_validation_rules() -> str:
    """Generate all possible validation rules from the Neo4j graph."""
    try:
        if not LTN_AVAILABLE:
            return "LTN module not available. Install with: pip install ltn tensorflow"
        generator = get_rule_generator()
        if generator is None: return "Rule generator not available."
        all_rules = generator.generate_all_rules()
        if not all_rules: return "No rules generated. Make sure FK discovery has been run."
        output = f"## Generated {len(all_rules)} Validation Rules\n\n"
        by_type = {}
        for rule in all_rules:
            rtype = rule.rule_type
            if rtype not in by_type: by_type[rtype] = []
            by_type[rtype].append(rule)
        for rtype, rules in by_type.items():
            output += f"### {rtype.title()} Rules ({len(rules)}):\n"
            for rule in rules: output += f"- **{rule.name}**: {rule.description}\n"
            output += "\n"
        return output
    except Exception as e:
        import traceback
        return f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def export_generated_rules_yaml() -> str:
    """Export generated business rules as YAML."""
    try:
        if not LTN_AVAILABLE: return "LTN module not available."
        generator = get_rule_generator()
        if generator is None: return "Rule generator not available."
        if not generator.generated_rules: generator.generate_all_rules()
        if not generator.generated_rules: return "No rules to export. Run generate_business_rules_from_ltn first."
        yaml_content = generator.export_yaml()
        filename = "business_rules_generated.yaml"
        with open(filename, "w") as f: f.write(yaml_content)
        output = f"## Exported {len(generator.generated_rules)} Rules to {filename}\n\n```yaml\n"
        output += yaml_content[:2000]
        if len(yaml_content) > 2000: output += "\n... (truncated)"
        output += f"\n```\n\nLoad with: `load_business_rules_from_file business_rules_generated.yaml`"
        return output
    except Exception as e:
        import traceback
        return f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def export_generated_rules_sql() -> str:
    """Export generated rules as SQL script."""
    try:
        if not LTN_AVAILABLE: return "LTN module not available."
        generator = get_rule_generator()
        if generator is None: return "Rule generator not available."
        if not generator.generated_rules: generator.generate_all_rules()
        if not generator.generated_rules: return "No rules to export."
        sql_content = generator.export_sql()
        filename = "validation_rules.sql"
        with open(filename, "w") as f: f.write(sql_content)
        output = f"## Exported SQL to {filename}\n\n```sql\n"
        output += sql_content[:2000]
        if len(sql_content) > 2000: output += "\n... (truncated)"
        output += "\n```"
        return output
    except Exception as e:
        import traceback
        return f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def show_ltn_knowledge_base() -> str:
    """Show the LTN knowledge base with axioms and constraints."""
    try:
        if not LTN_AVAILABLE:
            return "LTN not available. Install with: pip install ltn tensorflow"
        kb = LTNKnowledgeBase.create_default()
        output = "## LTN Knowledge Base\n\n### Axioms (Logical Rules):\n"
        for axiom in kb.get_all_axioms():
            output += f"- **{axiom.name}**: `{axiom.formula}`\n"
            output += f"  Type: {axiom.axiom_type.value}, Weight: {axiom.weight}\n"
            if axiom.description: output += f"  {axiom.description}\n"
        output += "\n### Available Predicates:\n"
        output += "- `IsPK(c)` - Column is primary key\n"
        output += "- `IsFK(c)` - Column is foreign key\n"
        output += "- `FK(c1, c2)` - FK relationship between columns\n"
        output += "- `IsFact(t)` - Table is a fact table\n"
        output += "- `IsDimension(t)` - Table is a dimension table\n"
        output += "- `IsJunction(t)` - Table is a junction table\n"
        output += "- `BelongsTo(c, t)` - Column belongs to table\n"
        output += "- `SameType(c1, c2)` - Columns have same data type\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# =============================================================================
# System Prompt (shared between all modes)
# =============================================================================

SYSTEM_PROMPT = """You are GraphWeaver Agent - an AI assistant that helps users discover foreign key relationships, execute business rules, capture data lineage, and perform semantic search on database metadata.

## Your Capabilities:

### Dynamic Tool Management
- `check_tool_exists` - Check if a dynamic tool exists
- `list_available_tools` - List all builtin and dynamic tools
- `create_dynamic_tool` - Create a new custom tool (code must define run() function)
- `run_dynamic_tool` - Execute a dynamic tool
- `get_tool_source` - View a dynamic tool's source code
- `update_dynamic_tool` - Update a dynamic tool
- `delete_dynamic_tool` - Delete a dynamic tool

### FK Discovery
- `run_fk_discovery` - Complete 5-stage FK discovery pipeline
- `analyze_potential_fk` - Score a single column pair
- `validate_fk_with_data` - Test FK with actual data
- `semantic_fk_discovery` - Find FKs using semantic similarity (even with non-matching names)

### Database Exploration
- `configure_database` - Connect to a PostgreSQL database
- `test_database_connection` - Verify connectivity
- `list_database_tables` - See all tables
- `get_table_schema` - See columns, types, PKs
- `get_column_stats` - Get column statistics

### Knowledge Graph (Neo4j)
- `clear_neo4j_graph` - Reset the graph
- `add_fk_to_graph` - Add FK relationship
- `get_graph_stats` - Graph statistics
- `analyze_graph_centrality` - Hub/authority analysis
- `find_table_communities` - Find related clusters
- `predict_missing_fks` - Suggest missing FKs (name-based)
- `run_cypher` - Execute any custom Cypher query
- `connect_datasets_to_tables` - Link Dataset nodes to Table nodes

### Embeddings & Semantic Search
- `generate_text_embeddings` - Create semantic embeddings for all metadata
- `generate_kg_embeddings` - Create graph structure embeddings (FastRP)
- `create_vector_indexes` - Create Neo4j vector indexes for fast search
- `semantic_search_tables` - Search tables using natural language
- `semantic_search_columns` - Search columns using natural language
- `find_similar_tables` - Find structurally/semantically similar tables
- `find_similar_columns` - Find similar columns across tables
- `predict_fks_from_embeddings` - Predict FKs using graph structure

### Business Rules & Lineage
- `show_sample_business_rules` - Show example YAML format
- `load_business_rules` - Load rules from YAML string
- `load_business_rules_from_file` - Load rules from a YAML file
- `list_business_rules` - Show loaded rules
- `execute_business_rule` - Run single rule with lineage capture
- `execute_all_business_rules` - Run all rules
- `get_marquez_lineage` - Get lineage for a dataset
- `list_marquez_jobs` - Show tracked jobs
- `import_lineage_to_graph` - Import lineage to Neo4j
- `analyze_data_flow` - Full analysis (FKs + lineage)
- `find_impact_analysis` - What breaks if table changes?

### RDF / SPARQL
- `test_rdf_connection` - Test connection to Fuseki triple store
- `sync_graph_to_rdf` - Sync Neo4j graph to RDF store
- `run_sparql` - Execute custom SPARQL queries
- `sparql_list_tables` - List tables via SPARQL
- `sparql_get_foreign_keys` - Get FKs via SPARQL
- `sparql_table_lineage` - Get lineage via SPARQL
- `sparql_downstream_impact` - Find downstream deps via SPARQL
- `sparql_hub_tables` - Find hub tables via SPARQL
- `sparql_orphan_tables` - Find orphan tables
- `sparql_search` - Search graph by label
- `get_rdf_statistics` - RDF store stats
- `export_rdf_turtle` - Export ontology

### LTN / Rule Learning
- `learn_rules_with_ltn` - Learn logical rules from graph using LTN
- `generate_business_rules_from_ltn` - Generate executable rules from learned patterns
- `generate_all_validation_rules` - Generate all validation rules from graph
- `export_generated_rules_yaml` - Export rules as YAML file
- `export_generated_rules_sql` - Export rules as SQL script
- `show_ltn_knowledge_base` - Show LTN axioms and predicates

## Typical Workflow:

1. **Discover FKs**: `run_fk_discovery`
2. **Generate embeddings**: `generate_text_embeddings`, `generate_kg_embeddings`
3. **Create indexes**: `create_vector_indexes`
4. **Semantic FK discovery**: `semantic_fk_discovery` (finds FKs missed by name matching)
5. **Learn rules with LTN**: `learn_rules_with_ltn`
6. **Generate validation rules**: `generate_all_validation_rules`
7. **Export rules**: `export_generated_rules_yaml`
8. **Load business rules**: `load_business_rules_from_file`
9. **Execute rules**: `execute_all_business_rules` (captures lineage)
10. **Import lineage**: `import_lineage_to_graph`
11. **Connect graphs**: `connect_datasets_to_tables`
12. **Sync to RDF**: `sync_graph_to_rdf`
13. **Analyze**: `semantic_search_tables`, `find_similar_tables`, `find_impact_analysis`

Be helpful and thorough!"""


# =============================================================================
# All Tools List
# =============================================================================

ALL_TOOLS = [
    # Dynamic Tool Management
    check_tool_exists, list_available_tools, create_dynamic_tool,
    run_dynamic_tool, get_tool_source, update_dynamic_tool, delete_dynamic_tool,
    # Database
    configure_database, test_database_connection, list_database_tables,
    get_table_schema, get_column_stats,
    # FK Discovery
    run_fk_discovery, discover_and_sync, analyze_potential_fk, validate_fk_with_data,
    # Neo4j Graph
    clear_neo4j_graph, add_fk_to_graph, get_graph_stats, analyze_graph_centrality,
    find_table_communities, predict_missing_fks, run_cypher, connect_datasets_to_tables,
    # Embeddings & Semantic Search
    generate_text_embeddings, generate_kg_embeddings, create_vector_indexes,
    semantic_search_tables, semantic_search_columns, find_similar_tables,
    find_similar_columns, predict_fks_from_embeddings, semantic_fk_discovery,
    # Business Rules & Lineage
    show_sample_business_rules, load_business_rules, load_business_rules_from_file,
    list_business_rules, execute_business_rule, execute_all_business_rules,
    get_marquez_lineage, list_marquez_jobs, import_lineage_to_graph,
    analyze_data_flow, find_impact_analysis,
    # RDF Tools
    test_rdf_connection, sync_graph_to_rdf, debug_rdf_sync, run_sparql,
    sparql_list_tables, sparql_get_foreign_keys, sparql_table_lineage,
    sparql_downstream_impact, sparql_hub_tables, sparql_orphan_tables,
    sparql_search, get_rdf_statistics, export_rdf_turtle,
    # LTN Tools
    learn_rules_with_ltn, generate_business_rules_from_ltn,
    generate_all_validation_rules, export_generated_rules_yaml,
    export_generated_rules_sql, show_ltn_knowledge_base,
]