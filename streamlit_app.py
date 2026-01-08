#!/usr/bin/env python3
# =============================================================================
# FILE: streamlit_app.py
# PATH: /home/gp/Downloads/graphweaver-agent/streamlit_app.py
# =============================================================================
"""
graphweaver-agent/streamlit_app.py

Streamlit Chat Interface for GraphWeaver Agent with Real-Time Streaming

WITH TERMINAL DEBUG LOGGING - Run with: DEBUG=1 streamlit run streamlit_app.py
"""
import os
import sys
import streamlit as st
from typing import Optional, Generator, Dict, Any, List
import anthropic
import time

# =============================================================================
# DEBUG LOGGING SETUP
# =============================================================================
os.environ['PYTHONUNBUFFERED'] = '1'

from debug_logger import DebugLogger, debug, debug_tool, APIStreamLogger, Colors

DEBUG_MODE = os.environ.get("DEBUG", "0").lower() in ("1", "true", "yes")
if DEBUG_MODE:
    DebugLogger.enable(verbose=True, log_file="agent_debug.log")

# =============================================================================
# Original imports
# =============================================================================

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mcp_servers"))

from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

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
        LTNRuleLearner,
        BusinessRuleGenerator,
        LTNKnowledgeBase,
        LearnedRule,
        GeneratedRule,
        RuleLearningConfig,
    )
    LTN_AVAILABLE = True
except (ImportError, AttributeError):
    LTN_AVAILABLE = False

from graphweaver_agent.dynamic_tools.agent_tools import (
    check_tool_exists,
    list_available_tools,
    create_dynamic_tool,
    run_dynamic_tool,
    get_tool_source,
    update_dynamic_tool,
    delete_dynamic_tool,
    DYNAMIC_TOOL_MANAGEMENT_TOOLS,
)


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are GraphWeaver Agent, an intelligent AI assistant for database exploration, FK discovery, knowledge graph building, and data lineage analysis.

You have access to many tools to help users:
- Explore PostgreSQL databases (test connections, list tables, get schemas)
- Discover foreign key relationships using a 5-stage pipeline
- Build and analyze Neo4j knowledge graphs
- Generate and use text/KG embeddings for semantic search
- Execute business rules and capture lineage to Marquez
- Work with RDF/SPARQL via Apache Jena Fuseki
- Learn rules with Logic Tensor Networks (LTN)
- Create and manage dynamic tools at runtime

Be helpful, thorough, and explain what you're doing. When discovering FKs, validate with actual data. When building graphs, explain the structure."""


# =============================================================================
# Streaming Tool Definitions
# =============================================================================

STREAMING_TOOLS = [
    {"name": "check_tool_exists", "description": "Check if a dynamic tool exists in the registry",
     "input_schema": {"type": "object", "properties": {"tool_name": {"type": "string", "description": "Name of the tool to check"}}, "required": ["tool_name"]}},
    {"name": "list_available_tools", "description": "List all available tools - both builtin and dynamic",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "create_dynamic_tool", "description": "Create a new dynamic tool. Code must define a run() function.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "description": {"type": "string"}, "code": {"type": "string"}}, "required": ["name", "description", "code"]}},
    {"name": "run_dynamic_tool", "description": "Execute a dynamic tool by name",
     "input_schema": {"type": "object", "properties": {"tool_name": {"type": "string"}}, "required": ["tool_name"]}},
    {"name": "get_tool_source", "description": "Get the source code of a dynamic tool",
     "input_schema": {"type": "object", "properties": {"tool_name": {"type": "string"}}, "required": ["tool_name"]}},
    {"name": "update_dynamic_tool", "description": "Update a dynamic tool's code",
     "input_schema": {"type": "object", "properties": {"tool_name": {"type": "string"}, "code": {"type": "string"}, "description": {"type": "string"}}, "required": ["tool_name", "code"]}},
    {"name": "delete_dynamic_tool", "description": "Delete a dynamic tool",
     "input_schema": {"type": "object", "properties": {"tool_name": {"type": "string"}}, "required": ["tool_name"]}},
    {"name": "configure_database", "description": "Configure which PostgreSQL database to connect to",
     "input_schema": {"type": "object", "properties": {"host": {"type": "string"}, "port": {"type": "integer"}, "database": {"type": "string"}, "username": {"type": "string"}, "password": {"type": "string"}}, "required": ["host", "port", "database", "username", "password"]}},
    {"name": "test_database_connection", "description": "Test the PostgreSQL database connection",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "list_database_tables", "description": "List all tables in the database with column counts",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_table_schema", "description": "Get schema details for a table (columns, types, PKs)",
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}}, "required": ["table_name"]}},
    {"name": "get_column_stats", "description": "Get statistics for a column (uniqueness, nulls, samples)",
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}, "column_name": {"type": "string"}}, "required": ["table_name", "column_name"]}},
    {"name": "run_fk_discovery", "description": "Run the full 5-stage FK discovery pipeline on the database",
     "input_schema": {"type": "object", "properties": {"min_match_rate": {"type": "number", "default": 0.95}, "min_score": {"type": "number", "default": 0.5}}}},
    {"name": "analyze_potential_fk", "description": "Analyze a potential FK relationship and get a score",
     "input_schema": {"type": "object", "properties": {"source_table": {"type": "string"}, "source_column": {"type": "string"}, "target_table": {"type": "string"}, "target_column": {"type": "string"}}, "required": ["source_table", "source_column", "target_table", "target_column"]}},
    {"name": "validate_fk_with_data", "description": "Validate a FK by checking actual data integrity",
     "input_schema": {"type": "object", "properties": {"source_table": {"type": "string"}, "source_column": {"type": "string"}, "target_table": {"type": "string"}, "target_column": {"type": "string"}}, "required": ["source_table", "source_column", "target_table", "target_column"]}},
    {"name": "clear_neo4j_graph", "description": "Clear all nodes and relationships from Neo4j",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "add_fk_to_graph", "description": "Add a FK relationship to the Neo4j graph",
     "input_schema": {"type": "object", "properties": {"source_table": {"type": "string"}, "source_column": {"type": "string"}, "target_table": {"type": "string"}, "target_column": {"type": "string"}, "score": {"type": "number", "default": 1.0}, "cardinality": {"type": "string", "default": "1:N"}}, "required": ["source_table", "source_column", "target_table", "target_column"]}},
    {"name": "get_graph_stats", "description": "Get statistics about the Neo4j graph",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "analyze_graph_centrality", "description": "Find hub and authority tables in the graph",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "find_table_communities", "description": "Find clusters of related tables",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "predict_missing_fks", "description": "Predict missing FKs based on column naming patterns",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "run_cypher", "description": "Execute a Cypher query on Neo4j",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "connect_datasets_to_tables", "description": "Connect Dataset nodes to their matching Table nodes",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "generate_text_embeddings", "description": "Generate text embeddings for all tables, columns, jobs, and datasets",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "generate_kg_embeddings", "description": "Generate knowledge graph embeddings using Neo4j GDS FastRP",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "create_vector_indexes", "description": "Create Neo4j vector indexes for fast similarity search",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "semantic_search_tables", "description": "Search for tables using natural language",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 5}}, "required": ["query"]}},
    {"name": "semantic_search_columns", "description": "Search for columns using natural language",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 10}}, "required": ["query"]}},
    {"name": "find_similar_tables", "description": "Find tables similar to a given table",
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}, "top_k": {"type": "integer", "default": 5}}, "required": ["table_name"]}},
    {"name": "find_similar_columns", "description": "Find columns similar to a given column",
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}, "column_name": {"type": "string"}, "top_k": {"type": "integer", "default": 10}}, "required": ["table_name", "column_name"]}},
    {"name": "predict_fks_from_embeddings", "description": "Predict FK relationships using embedding similarity",
     "input_schema": {"type": "object", "properties": {"threshold": {"type": "number", "default": 0.7}, "top_k": {"type": "integer", "default": 20}}}},
    {"name": "semantic_fk_discovery", "description": "Discover FKs using semantic similarity",
     "input_schema": {"type": "object", "properties": {"source_table": {"type": "string"}, "min_score": {"type": "number", "default": 0.6}}}},
    {"name": "show_sample_business_rules", "description": "Show sample business rules YAML format",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "load_business_rules", "description": "Load business rules from YAML content",
     "input_schema": {"type": "object", "properties": {"yaml_content": {"type": "string"}}, "required": ["yaml_content"]}},
    {"name": "load_business_rules_from_file", "description": "Load business rules from a YAML file",
     "input_schema": {"type": "object", "properties": {"file_path": {"type": "string", "default": "business_rules.yaml"}}}},
    {"name": "list_business_rules", "description": "List all loaded business rules",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "execute_business_rule", "description": "Execute a specific business rule",
     "input_schema": {"type": "object", "properties": {"rule_name": {"type": "string"}, "capture_lineage": {"type": "boolean", "default": True}}, "required": ["rule_name"]}},
    {"name": "execute_all_business_rules", "description": "Execute all loaded business rules",
     "input_schema": {"type": "object", "properties": {"capture_lineage": {"type": "boolean", "default": True}}}},
    {"name": "get_marquez_lineage", "description": "Get data lineage for a dataset from Marquez",
     "input_schema": {"type": "object", "properties": {"dataset_name": {"type": "string"}, "depth": {"type": "integer", "default": 3}}, "required": ["dataset_name"]}},
    {"name": "list_marquez_jobs", "description": "List all jobs tracked by Marquez",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "import_lineage_to_graph", "description": "Import Marquez lineage data into Neo4j",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "analyze_data_flow", "description": "Analyze data flow for a table",
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}}, "required": ["table_name"]}},
    {"name": "find_impact_analysis", "description": "Find what tables would be impacted by changes",
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}}, "required": ["table_name"]}},
    {"name": "test_rdf_connection", "description": "Test connection to Apache Jena Fuseki",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "sync_graph_to_rdf", "description": "Sync Neo4j graph to RDF triple store",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "run_sparql", "description": "Execute a SPARQL query on Fuseki",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "sparql_list_tables", "description": "List all tables via SPARQL",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "sparql_get_foreign_keys", "description": "Get foreign keys via SPARQL",
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}}}},
    {"name": "sparql_table_lineage", "description": "Get table lineage via SPARQL",
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}}, "required": ["table_name"]}},
    {"name": "sparql_downstream_impact", "description": "Get downstream impact via SPARQL",
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}}, "required": ["table_name"]}},
    {"name": "sparql_hub_tables", "description": "Find hub tables via SPARQL",
     "input_schema": {"type": "object", "properties": {"min_connections": {"type": "integer", "default": 3}}}},
    {"name": "sparql_orphan_tables", "description": "Find orphan tables via SPARQL",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "sparql_search", "description": "Search entities by label via SPARQL",
     "input_schema": {"type": "object", "properties": {"search_term": {"type": "string"}}, "required": ["search_term"]}},
    {"name": "get_rdf_statistics", "description": "Get RDF store statistics",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "export_rdf_turtle", "description": "Export RDF data as Turtle",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "learn_rules_with_ltn", "description": "Learn rules using Logic Tensor Networks",
     "input_schema": {"type": "object", "properties": {"epochs": {"type": "integer", "default": 100}}}},
    {"name": "generate_business_rules_from_ltn", "description": "Generate business rules from LTN learned patterns",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "generate_all_validation_rules", "description": "Generate validation rules from graph structure",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "export_generated_rules_yaml", "description": "Export generated rules to YAML",
     "input_schema": {"type": "object", "properties": {"file_path": {"type": "string", "default": "business_rules_generated.yaml"}}}},
    {"name": "export_generated_rules_sql", "description": "Export generated rules as SQL",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "show_ltn_knowledge_base", "description": "Show the LTN knowledge base",
     "input_schema": {"type": "object", "properties": {}}},
]


# =============================================================================
# Global State Accessors with Debug Logging
# =============================================================================

def get_pg_config() -> DataSourceConfig:
    debug.postgres("Getting PostgreSQL config...")
    from graphweaver_agent.models import DatabaseType
    config = DataSourceConfig(
        host=st.session_state.get("pg_host", os.environ.get("POSTGRES_HOST", "localhost")),
        port=int(st.session_state.get("pg_port", os.environ.get("POSTGRES_PORT", "5432"))),
        database=st.session_state.get("pg_database", os.environ.get("POSTGRES_DB", "orders")),
        username=st.session_state.get("pg_username", os.environ.get("POSTGRES_USER", "saphenia")),
        password=st.session_state.get("pg_password", os.environ.get("POSTGRES_PASSWORD", "secret")),
        db_type=DatabaseType.POSTGRESQL,
        schema_name=st.session_state.get("pg_schema", "public"),
    )
    debug.postgres(f"Config: {config.host}:{config.port}/{config.database}")
    return config


def get_pg() -> PostgreSQLConnector:
    if "pg_connector" not in st.session_state:
        debug.postgres("Creating new PostgreSQL connector...")
        st.session_state.pg_connector = PostgreSQLConnector(get_pg_config())
        debug.postgres("✓ PostgreSQL connector created")
    return st.session_state.pg_connector


def get_neo4j_config() -> Neo4jConfig:
    debug.neo4j("Getting Neo4j config...")
    return Neo4jConfig(
        uri=st.session_state.get("neo4j_uri", os.environ.get("NEO4J_URI", "bolt://localhost:7687")),
        username=st.session_state.get("neo4j_user", os.environ.get("NEO4J_USER", "neo4j")),
        password=st.session_state.get("neo4j_password", os.environ.get("NEO4J_PASSWORD", "password")),
        database=st.session_state.get("neo4j_database", "neo4j"),
    )


def get_neo4j() -> Neo4jClient:
    if "neo4j_client" not in st.session_state:
        config = get_neo4j_config()
        debug.neo4j(f"Creating Neo4j client: {config.uri}")
        try:
            st.session_state.neo4j_client = Neo4jClient(config)
            debug.neo4j("✓ Neo4j client created")
        except Exception as e:
            debug.error(f"Failed to create Neo4j client: {e}", e)
            raise
    return st.session_state.neo4j_client


def get_text_embedder():
    if "text_embedder" not in st.session_state:
        debug.embedding("Loading text embedder model...")
        from graphweaver_agent.embeddings import TextEmbedder
        st.session_state.text_embedder = TextEmbedder()
        debug.embedding("✓ Text embedder loaded")
    return st.session_state.text_embedder


def get_fuseki() -> FusekiClient:
    if "fuseki_client" not in st.session_state:
        debug.api("Creating Fuseki client...")
        st.session_state.fuseki_client = FusekiClient()
    return st.session_state.fuseki_client


def get_sparql() -> SPARQLQueryBuilder:
    if "sparql_builder" not in st.session_state:
        st.session_state.sparql_builder = SPARQLQueryBuilder(get_fuseki())
    return st.session_state.sparql_builder


def get_marquez() -> MarquezClient:
    if "marquez_client" not in st.session_state:
        marquez_url = os.environ.get("MARQUEZ_URL", "http://localhost:5000")
        debug.api(f"Creating Marquez client: {marquez_url}")
        st.session_state.marquez_client = MarquezClient(marquez_url)
    return st.session_state.marquez_client


def get_rule_learner():
    if "rule_learner" not in st.session_state and LTN_AVAILABLE:
        debug.agent("Creating LTN rule learner...")
        st.session_state.rule_learner = LTNRuleLearner(get_neo4j())
    return st.session_state.get("rule_learner")


def get_rule_generator():
    if "rule_generator" not in st.session_state and LTN_AVAILABLE:
        st.session_state.rule_generator = BusinessRuleGenerator(get_neo4j())
    return st.session_state.get("rule_generator")


def get_registry():
    if "tool_registry" not in st.session_state:
        from graphweaver_agent.dynamic_tools import DynamicToolRegistry
        st.session_state.tool_registry = DynamicToolRegistry()
    return st.session_state.tool_registry


# =============================================================================
# Tool Implementations with Debug Logging
# =============================================================================

@debug_tool
def impl_configure_database(host: str, port: int, database: str, username: str, password: str) -> str:
    st.session_state.pg_host = host
    st.session_state.pg_port = port
    st.session_state.pg_database = database
    st.session_state.pg_username = username
    st.session_state.pg_password = password
    if "pg_connector" in st.session_state:
        del st.session_state.pg_connector
    return f"✓ Configured database: {username}@{host}:{port}/{database}"


@debug_tool
def impl_test_database_connection() -> str:
    result = get_pg().test_connection()
    if result["success"]:
        return f"✓ Connected to database '{result['database']}' as '{result['user']}'"
    return f"✗ Failed: {result['error']}"


@debug_tool
def impl_list_database_tables() -> str:
    tables = get_pg().get_tables_with_info()
    output = "Tables:\n"
    for t in tables:
        output += f"  - {t['table_name']}: {t['column_count']} columns, ~{t['row_estimate']} rows\n"
    return output


@debug_tool
def impl_get_table_schema(table_name: str) -> str:
    schema = get_pg().get_table_schema(table_name)
    if not schema:
        return f"Table '{table_name}' not found"
    
    output = f"Table: {table_name} ({schema.get('row_count', 0)} rows)\n"
    output += f"Primary Key: {', '.join(schema.get('primary_keys', []))}\n"
    output += "Columns:\n"
    for col in schema.get('columns', []):
        pk_marker = " [PK]" if col['column_name'] in schema.get('primary_keys', []) else ""
        output += f"  - {col['column_name']}: {col['data_type']}{pk_marker}\n"
    return output


@debug_tool
def impl_get_column_stats(table_name: str, column_name: str) -> str:
    stats = get_pg().get_column_statistics(table_name, column_name)
    output = f"Stats for {table_name}.{column_name}:\n"
    output += f"  Distinct: {stats.distinct_count}\n"
    output += f"  Nulls: {stats.null_count}\n"
    output += f"  Uniqueness: {stats.uniqueness_ratio:.1%}\n"
    output += f"  Samples: {stats.sample_values}\n"
    return output


@debug_tool
def impl_run_fk_discovery(min_match_rate: float = 0.95, min_score: float = 0.5) -> str:
    debug.section("FK DISCOVERY PIPELINE")
    try:
        pg_config = get_pg_config()
        debug.agent(f"Running FK discovery on {pg_config.database}...")
        
        result = run_discovery(
            host=pg_config.host,
            port=pg_config.port,
            database=pg_config.database,
            username=pg_config.username,
            password=pg_config.password,
            schema=pg_config.schema_name,
            min_match_rate=min_match_rate,
            min_score=min_score,
        )
        
        summary = result["summary"]
        debug.agent(f"Discovery complete: {summary['final_fks_discovered']} FKs found")
        
        output = "## FK Discovery Results\n\n"
        output += f"- Tables scanned: {summary['tables_scanned']}\n"
        output += f"- Total columns: {summary['total_columns']}\n"
        output += f"- Initial candidates: {summary['initial_candidates']}\n"
        output += f"- **Final FKs discovered: {summary['final_fks_discovered']}**\n"
        output += f"- Duration: {summary['duration_seconds']}s\n\n"
        
        discovered_fks = result.get("discovered_fks", [])
        
        if discovered_fks:
            debug.subsection("Persisting to Neo4j")
            try:
                neo4j = get_neo4j()
                builder = GraphBuilder(neo4j)
                
                # FIXED: Only clear schema data (Tables/Columns/FKs/DataSource)
                # Preserves lineage data (Jobs/Datasets/READS/WRITES/REPRESENTS)
                debug.neo4j("Clearing schema data (preserving lineage)...")
                
                # Delete FK relationships first
                neo4j.run_write("MATCH ()-[r:FK_TO]->() DELETE r")
                # Delete Columns and their relationships
                neo4j.run_write("MATCH (c:Column) DETACH DELETE c")
                # Delete Tables (but this might delete REPRESENTS links - we'll recreate them)
                neo4j.run_write("MATCH (t:Table) DETACH DELETE t")
                # Delete DataSources
                neo4j.run_write("MATCH (ds:DataSource) DETACH DELETE ds")
                
                debug.neo4j("✓ Schema cleared (Jobs/Datasets preserved)")
                
                tables_added = set()
                fks_added = 0
                
                for fk in discovered_fks:
                    if "relationship" in fk:
                        rel = fk["relationship"]
                        parts = rel.split(" → ")
                        src_parts = parts[0].split(".")
                        tgt_parts = parts[1].split(".")
                        
                        src_table, src_col = src_parts[0], src_parts[1]
                        tgt_table, tgt_col = tgt_parts[0], tgt_parts[1]
                        confidence = fk.get("confidence", 1.0)
                        cardinality = fk.get("cardinality", "1:N")
                    else:
                        src_table = fk.get("source_table")
                        src_col = fk.get("source_column")
                        tgt_table = fk.get("target_table")
                        tgt_col = fk.get("target_column")
                        confidence = fk.get("score", fk.get("confidence", 1.0))
                        cardinality = fk.get("cardinality", "1:N")
                    
                    debug.neo4j(f"Adding FK: {src_table}.{src_col} → {tgt_table}.{tgt_col}")
                    
                    if src_table and src_table not in tables_added:
                        builder.add_table(src_table)
                        tables_added.add(src_table)
                    if tgt_table and tgt_table not in tables_added:
                        builder.add_table(tgt_table)
                        tables_added.add(tgt_table)
                    
                    if src_table and src_col and tgt_table and tgt_col:
                        builder.add_fk_relationship(
                            src_table, src_col, tgt_table, tgt_col,
                            float(confidence), str(cardinality)
                        )
                        fks_added += 1
                
                output += f"### ✓ Persisted to Neo4j\n"
                output += f"- Tables added: {len(tables_added)}\n"
                output += f"- FK relationships added: {fks_added}\n\n"
                
                debug.neo4j(f"✓ Persisted {fks_added} FKs, {len(tables_added)} tables")
                
            except Exception as e:
                debug.error(f"Neo4j persistence failed: {e}", e)
                import traceback
                output += f"### ⚠ Neo4j Error: {e}\n"
                output += f"```\n{traceback.format_exc()}\n```\n\n"
        
        output += "### Discovered Foreign Keys\n\n"
        if discovered_fks:
            for fk in discovered_fks:
                if "relationship" in fk:
                    output += f"**{fk['relationship']}**\n"
                    output += f"  - Confidence: {fk.get('confidence', 1.0):.1%}\n"
                    output += f"  - Cardinality: {fk.get('cardinality', 'unknown')}\n"
                    scores = fk.get("scores", {})
                    if scores:
                        output += f"  - Match rate: {scores.get('match_rate', 0):.1%}\n"
                else:
                    output += f"**{fk.get('source_table')}.{fk.get('source_column')} → {fk.get('target_table')}.{fk.get('target_column')}**\n"
                    output += f"  - Score: {fk.get('score', fk.get('confidence', 1.0)):.2f}\n"
                output += "\n"
        else:
            output += "No foreign keys discovered.\n"
        
        return output
        
    except Exception as e:
        debug.error(f"FK discovery failed: {e}", e)
        import traceback
        return f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@debug_tool
def impl_analyze_potential_fk(source_table: str, source_column: str, target_table: str, target_column: str) -> str:
    pg = get_pg()
    
    try:
        source_schema = pg.get_table_schema(source_table)
        target_schema = pg.get_table_schema(target_table)
        
        source_cols = {c["column_name"]: c for c in source_schema["columns"]}
        target_cols = {c["column_name"]: c for c in target_schema["columns"]}
        
        if source_column not in source_cols:
            return f"Error: Column '{source_column}' not found in '{source_table}'. Available: {list(source_cols.keys())}"
        
        if target_column not in target_cols:
            return f"Error: Column '{target_column}' not found in '{target_table}'. Available: {list(target_cols.keys())}"
        
        source_stats = pg.get_column_statistics(source_table, source_column)
        target_stats = pg.get_column_statistics(target_table, target_column)
        
        source_col_info = source_cols[source_column]
        target_col_info = target_cols[target_column]
        
        type_compatible = source_col_info["data_type"] == target_col_info["data_type"]
        target_unique = target_stats.uniqueness_ratio > 0.95
        
        output = f"Analysis: {source_table}.{source_column} → {target_table}.{target_column}\n"
        output += f"  Source type: {source_col_info['data_type']}\n"
        output += f"  Target type: {target_col_info['data_type']}\n"
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
    except Exception as e:
        import traceback
        return f"Error: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@debug_tool
def impl_validate_fk_with_data(source_table: str, source_column: str, target_table: str, target_column: str) -> str:
    pg = get_pg()
    result = pg.validate_fk(source_table, source_column, target_table, target_column)
    
    output = f"Validation: {source_table}.{source_column} → {target_table}.{target_column}\n"
    if result['is_valid']:
        output += f"  ✓ CONFIRMED FK\n"
        output += f"  Match rate: {result['match_rate']:.1%}\n"
        output += f"  Cardinality: {result.get('cardinality', 'unknown')}\n"
    else:
        output += f"  ✗ NOT A VALID FK\n"
        output += f"  Match rate: {result['match_rate']:.1%}\n"
        if result.get('orphan_count'):
            output += f"  Orphan values: {result['orphan_count']}\n"
    return output


@debug_tool
def impl_clear_neo4j_graph() -> str:
    debug.neo4j("Clearing Neo4j graph...")
    neo4j = get_neo4j()
    neo4j.run_write("MATCH (n) DETACH DELETE n")
    debug.neo4j("✓ Graph cleared")
    return "✓ Neo4j graph cleared"


@debug_tool
def impl_add_fk_to_graph(source_table: str, source_column: str, target_table: str, target_column: str, score: float = 1.0, cardinality: str = "1:N") -> str:
    builder = GraphBuilder(get_neo4j())
    builder.add_table(source_table)
    builder.add_table(target_table)
    builder.add_fk_relationship(source_table, source_column, target_table, target_column, score, cardinality)
    return f"✓ Added: {source_table}.{source_column} → {target_table}.{target_column}"


@debug_tool
def impl_get_graph_stats() -> str:
    """Get comprehensive graph statistics including all node types."""
    debug.neo4j("Getting comprehensive graph statistics...")
    neo4j = get_neo4j()
    
    output = "## Neo4j Graph Statistics\n\n"
    
    # Show connection info
    config = get_neo4j_config()
    output += f"**Connection:** {config.uri}\n"
    output += f"**Database:** {config.database}\n\n"
    
    # Get all labels
    labels_result = neo4j.run_query("CALL db.labels() YIELD label RETURN label")
    labels = [r['label'] for r in (labels_result or [])]
    output += f"**Labels found:** {labels}\n\n"
    
    # Count each label
    output += "### Node Counts:\n"
    for label in labels:
        try:
            count_result = neo4j.run_query(f"MATCH (n:`{label}`) RETURN count(n) as cnt")
            cnt = count_result[0]['cnt'] if count_result else 0
            output += f"- {label}: {cnt}\n"
        except Exception as e:
            output += f"- {label}: ERROR - {e}\n"
    
    # Get all relationship types
    rel_result = neo4j.run_query("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
    rel_types = [r['relationshipType'] for r in (rel_result or [])]
    output += f"\n**Relationship types:** {rel_types}\n\n"
    
    # Count each relationship type
    output += "### Relationship Counts:\n"
    for rel_type in rel_types:
        try:
            count_result = neo4j.run_query(f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) as cnt")
            cnt = count_result[0]['cnt'] if count_result else 0
            output += f"- {rel_type}: {cnt}\n"
        except Exception as e:
            output += f"- {rel_type}: ERROR - {e}\n"
    
    # Sample some data
    output += "\n### Sample Data:\n"
    
    # Sample jobs
    jobs = neo4j.run_query("MATCH (j:Job) RETURN j.name as name LIMIT 5")
    if jobs:
        output += f"- Jobs: {[j['name'] for j in jobs]}\n"
    else:
        output += "- Jobs: (none found)\n"
    
    # Sample datasets
    datasets = neo4j.run_query("MATCH (d:Dataset) RETURN d.name as name LIMIT 5")
    if datasets:
        output += f"- Datasets: {[d['name'] for d in datasets]}\n"
    else:
        output += "- Datasets: (none found)\n"
    
    # Sample tables
    tables = neo4j.run_query("MATCH (t:Table) RETURN t.name as name LIMIT 5")
    if tables:
        output += f"- Tables: {[t['name'] for t in tables]}\n"
    else:
        output += "- Tables: (none found)\n"
    
    return output


@debug_tool
def impl_analyze_graph_centrality() -> str:
    debug.neo4j("Analyzing graph centrality...")
    result = GraphAnalyzer(get_neo4j()).centrality_analysis()
    debug.neo4j(f"Centrality result: {result}")
    output = "Centrality Analysis:\n"
    output += f"  Hub tables (fact/transaction): {result['hub_tables']}\n"
    output += f"  Authority tables (dimension): {result['authority_tables']}\n"
    output += f"  Isolated tables: {result['isolated_tables']}\n"
    return output


@debug_tool
def impl_find_table_communities() -> str:
    communities = GraphAnalyzer(get_neo4j()).community_detection()
    if not communities:
        return "No communities found."
    output = "Communities:\n"
    for i, c in enumerate(communities):
        output += f"  {i+1}. {', '.join(c['tables'])}\n"
    return output


@debug_tool
def impl_predict_missing_fks() -> str:
    predictions = GraphAnalyzer(get_neo4j()).predict_missing_fks()
    if not predictions:
        return "No predictions - graph appears complete."
    output = "Predicted missing FKs:\n"
    for p in predictions:
        output += f"  - {p['source_table']}.{p['source_column']} → {p['target_table']}\n"
    return output


@debug_tool
def impl_run_cypher(query: str) -> str:
    debug.neo4j(f"Executing Cypher query:\n{query}")
    neo4j = get_neo4j()
    try:
        results = neo4j.run_query(query)
        debug.neo4j(f"Query returned {len(results) if results else 0} rows")
        if not results:
            return "Query executed successfully. No results returned."
        output = f"Results ({len(results)} rows):\n"
        for row in results[:50]:
            output += f"  {dict(row)}\n"
        if len(results) > 50:
            output += f"  ... and {len(results) - 50} more rows"
        return output
    except Exception as e:
        debug.error(f"Cypher query failed: {e}", e)
        try:
            neo4j.run_write(query)
            return "Write query executed successfully."
        except Exception as e2:
            return f"Error: {e2}"


@debug_tool
def impl_connect_datasets_to_tables() -> str:
    """Connect Dataset nodes to Table nodes using fuzzy matching.
    
    FIXED: Uses substring matching to handle dataset names like 
    'ecommerce.orders' matching table 'orders'.
    """
    debug.neo4j("Connecting datasets to tables (fuzzy match)...")
    neo4j = get_neo4j()
    
    # Get all datasets and tables
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
    
    # Build table name lookup (lowercase for matching)
    table_names = {t['name'].lower(): t['name'] for t in tables if t.get('name')}
    
    connected = []
    for ds in datasets:
        ds_name = ds.get('name', '')
        if not ds_name:
            continue
        
        # Try to extract table name from dataset name
        # Handle: "namespace.table", "db.schema.table", "postgres://host/db.table"
        possible_names = [
            ds_name,                                    # exact match
            ds_name.split('.')[-1],                     # last part after dot
            ds_name.split('/')[-1],                     # last part after slash
            ds_name.split('.')[-1].split('/')[-1],      # combination
        ]
        
        for possible in possible_names:
            possible_lower = possible.lower()
            if possible_lower in table_names:
                actual_table = table_names[possible_lower]
                # Create relationship
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
    return output


@debug_tool
def impl_generate_text_embeddings() -> str:
    debug.embedding("Generating text embeddings...")
    try:
        from graphweaver_agent.embeddings.text_embeddings import embed_all_metadata
        result = embed_all_metadata(
            neo4j_client=get_neo4j(),
            pg_connector=get_pg(),
            embedder=get_text_embedder(),
        )
        debug.embedding(f"Embedding result: {result}")
        
        # Handle error response
        if "error" in result:
            return f"ERROR: {result['error']}\nStats: {result.get('stats', {})}"
        
        # Handle warning response
        if "warning" in result:
            return f"WARNING: {result['warning']}\nStats: {result.get('stats', {})}"
        
        # Success - extract stats (they're at top level from to_dict())
        tables = result.get('tables', 0)
        columns = result.get('columns', 0)
        jobs = result.get('jobs', 0)
        datasets = result.get('datasets', 0)
        errors = result.get('errors', [])
        
        output = "## Text Embeddings Generated\n\n"
        output += f"### ✓ Table embeddings: {tables}\n"
        output += f"### ✓ Column embeddings: {columns}\n"
        output += f"### ✓ Job embeddings: {jobs}\n"
        output += f"### ✓ Dataset embeddings: {datasets}\n"
        output += f"\n**Total embeddings generated: {tables + columns + jobs + datasets}**\n"
        
        if errors:
            output += f"\n### ⚠ Errors ({len(errors)}):\n"
            for err in errors[:5]:
                output += f"- {err}\n"
            if len(errors) > 5:
                output += f"- ... and {len(errors) - 5} more errors\n"
        
        return output
    except Exception as e:
        debug.error(f"Embedding generation failed: {e}", e)
        import traceback
        return f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@debug_tool
def impl_generate_kg_embeddings() -> str:
    debug.embedding("Generating KG embeddings with Neo4j GDS...")
    debug.neo4j("Creating graph projection for FastRP...")
    try:
        from graphweaver_agent.embeddings.kg_embeddings import generate_all_kg_embeddings
        stats = generate_all_kg_embeddings(get_neo4j())
        debug.embedding(f"KG embedding stats: {stats}")
        output = "## Knowledge Graph Embeddings Generated\n\n"
        output += f"### ✓ Graph projection created: 'graphweaver-kg'\n"
        output += f"- Nodes: {stats.get('node_count', 'unknown')}\n"
        output += f"- Relationships: {stats.get('relationship_count', 'unknown')}\n"
        output += f"\n### ✓ FastRP embeddings computed\n"
        output += f"- Embedding dimension: 128\n"
        output += f"\n### ✓ Embeddings stored as node properties\n"
        output += f"\n**Knowledge graph embeddings ready for similarity analysis!**\n"
        return output
    except Exception as e:
        debug.error(f"KG embedding generation failed: {e}", e)
        import traceback
        return f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@debug_tool
def impl_create_vector_indexes() -> str:
    try:
        from graphweaver_agent.embeddings.vector_indexes import create_all_indexes
        stats = create_all_indexes(get_neo4j())
        return f"Vector indexes created: {stats.get('indexes_created', 0)}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_semantic_search_tables(query: str, top_k: int = 5) -> str:
    try:
        from graphweaver_agent.embeddings.text_embeddings import search_tables
        results = search_tables(get_neo4j(), get_text_embedder(), query, top_k)
        if not results:
            return f"No tables found matching '{query}'"
        output = f"## Tables matching '{query}'\n"
        for r in results:
            output += f"- **{r['name']}** (score: {r['score']:.3f})\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_semantic_search_columns(query: str, top_k: int = 10) -> str:
    try:
        from graphweaver_agent.embeddings.text_embeddings import search_columns
        results = search_columns(get_neo4j(), get_text_embedder(), query, top_k)
        if not results:
            return f"No columns found matching '{query}'"
        output = f"## Columns matching '{query}'\n"
        for r in results:
            output += f"- **{r['table']}.{r['name']}** (score: {r['score']:.3f})\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_find_similar_tables(table_name: str, top_k: int = 5) -> str:
    try:
        from graphweaver_agent.embeddings.text_embeddings import find_similar_tables
        results = find_similar_tables(get_neo4j(), table_name, top_k)
        if not results:
            return f"No similar tables found for '{table_name}'"
        output = f"## Tables similar to '{table_name}'\n"
        for r in results:
            output += f"- **{r['name']}** (similarity: {r['score']:.3f})\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_find_similar_columns(table_name: str, column_name: str, top_k: int = 10) -> str:
    try:
        from graphweaver_agent.embeddings.text_embeddings import find_similar_columns
        results = find_similar_columns(get_neo4j(), table_name, column_name, top_k)
        if not results:
            return f"No similar columns found for '{table_name}.{column_name}'"
        output = f"## Columns similar to '{table_name}.{column_name}'\n"
        for r in results:
            output += f"- **{r['table']}.{r['name']}** (similarity: {r['score']:.3f})\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_predict_fks_from_embeddings(threshold: float = 0.7, top_k: int = 20) -> str:
    try:
        from graphweaver_agent.embeddings.semantic_fk import predict_fks_from_embeddings
        predictions = predict_fks_from_embeddings(get_neo4j(), threshold, top_k)
        if not predictions:
            return "No FK predictions found."
        output = "## FK Predictions from Embeddings\n"
        for p in predictions:
            output += f"- {p['source_table']}.{p['source_column']} → {p['target_table']}.{p['target_column']} (score: {p['score']:.3f})\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_semantic_fk_discovery(source_table: str = None, min_score: float = 0.6) -> str:
    try:
        from graphweaver_agent.embeddings.semantic_fk import SemanticFKDiscovery
        discovery = SemanticFKDiscovery(get_neo4j(), get_text_embedder())
        results = discovery.discover(source_table=source_table, min_score=min_score)
        if not results:
            return "No semantic FK relationships found."
        output = "## Semantic FK Discovery Results\n"
        for r in results:
            output += f"- {r['source_table']}.{r['source_column']} → {r['target_table']}.{r['target_column']} (score: {r['score']:.3f})\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_show_sample_business_rules() -> str:
    return generate_sample_rules()


@debug_tool
def impl_load_business_rules(yaml_content: str) -> str:
    import yaml
    try:
        data = yaml.safe_load(yaml_content)
        if "rules_config" not in st.session_state:
            st.session_state.rules_config = BusinessRulesConfig()
        
        config = st.session_state.rules_config
        config.namespace = data.get("namespace", "default")
        config.rules = []
        
        for r in data.get("rules", []):
            config.rules.append(BusinessRule(
                name=r["name"],
                description=r.get("description", ""),
                sql=r["sql"],
                inputs=r.get("inputs", []),
                outputs=r.get("outputs", []),
                type=r.get("type", "query"),
                tags=r.get("tags", []),
            ))
        
        return f"✓ Loaded {len(config.rules)} business rules in namespace '{config.namespace}'"
    except Exception as e:
        return f"ERROR loading rules: {type(e).__name__}: {e}"


@debug_tool
def impl_load_business_rules_from_file(file_path: str = "business_rules.yaml") -> str:
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return impl_load_business_rules(content)
    except FileNotFoundError:
        return f"ERROR: File not found: {file_path}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_list_business_rules() -> str:
    if "rules_config" not in st.session_state or not st.session_state.rules_config.rules:
        return "No business rules loaded. Use load_business_rules or load_business_rules_from_file first."
    
    config = st.session_state.rules_config
    output = f"## Business Rules (namespace: {config.namespace})\n\n"
    for r in config.rules:
        output += f"- **{r.name}** ({r.type}): {r.description}\n"
    return output


@debug_tool
def impl_execute_business_rule(rule_name: str, capture_lineage: bool = True) -> str:
    if "rules_config" not in st.session_state:
        return "No business rules loaded."
    
    config = st.session_state.rules_config
    rule = next((r for r in config.rules if r.name == rule_name), None)
    
    if not rule:
        return f"Rule '{rule_name}' not found."
    
    try:
        marquez_url = os.environ.get("MARQUEZ_URL", "http://localhost:5000") if capture_lineage else None
        executor = BusinessRulesExecutor(get_pg(), marquez_url)
        result = executor.execute_rule(rule, emit_lineage=capture_lineage)
        
        output = f"## Executed: {rule_name}\n"
        output += f"- Rows returned: {result.get('row_count', 0)}\n"
        if capture_lineage:
            output += f"- Lineage captured: ✓\n"
        return output
    except Exception as e:
        import traceback
        return f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@debug_tool
def impl_execute_all_business_rules(capture_lineage: bool = True) -> str:
    if "rules_config" not in st.session_state or not st.session_state.rules_config.rules:
        return "No business rules loaded."
    
    config = st.session_state.rules_config
    marquez_url = os.environ.get("MARQUEZ_URL", "http://localhost:5000")
    executor = BusinessRulesExecutor(get_pg(), marquez_url)
    
    output = f"## Executing {len(config.rules)} rules\n\n"
    success = 0
    for rule in config.rules:
        try:
            result = executor.execute_rule(rule, emit_lineage=capture_lineage)
            output += f"✓ {rule.name}: {result.get('row_count', 0)} rows\n"
            success += 1
        except Exception as e:
            output += f"✗ {rule.name}: {type(e).__name__}: {e}\n"
    
    output += f"\n**{success}/{len(config.rules)} rules executed successfully**"
    return output


@debug_tool
def impl_get_marquez_lineage(dataset_name: str, depth: int = 3) -> str:
    try:
        lineage = get_marquez().get_lineage(dataset_name, depth)
        return f"Lineage for {dataset_name}:\n{lineage}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_list_marquez_jobs() -> str:
    try:
        jobs = get_marquez().list_jobs()
        if not jobs:
            return "No jobs found in Marquez."
        output = "## Marquez Jobs\n"
        for j in jobs:
            output += f"- {j['name']} ({j['namespace']})\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_import_lineage_to_graph() -> str:
    try:
        stats = import_lineage_to_neo4j(get_marquez(), get_neo4j())
        output = "## Lineage Imported to Neo4j\n\n"
        output += f"- Jobs processed: {stats.get('jobs', 0)}\n"
        output += f"- Datasets processed: {stats.get('datasets', 0)}\n"
        output += f"- READS relationships: {stats.get('reads', 0)}\n"
        output += f"- WRITES relationships: {stats.get('writes', 0)}\n"
        
        # Show verified counts from Neo4j
        if 'actual_jobs' in stats:
            output += "\n### Verified in Neo4j:\n"
            output += f"- Jobs: {stats.get('actual_jobs', 0)}\n"
            output += f"- Datasets: {stats.get('actual_datasets', 0)}\n"
            output += f"- READS edges: {stats.get('actual_reads', 0)}\n"
            output += f"- WRITES edges: {stats.get('actual_writes', 0)}\n"
        
        # Diagnostic if no relationships
        total_rels = stats.get('reads', 0) + stats.get('writes', 0)
        if stats.get('jobs', 0) > 0 and total_rels == 0:
            output += "\n**WARNING:** Jobs imported but no I/O relationships.\n"
            output += "Marquez may not have input/output data for these jobs.\n"
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@debug_tool
def impl_analyze_data_flow(table_name: str) -> str:
    try:
        analyzer = GraphAnalyzer(get_neo4j())
        flow = analyzer.analyze_data_flow(table_name)
        output = f"## Data Flow for '{table_name}'\n\n"
        output += f"### Upstream (data sources):\n"
        for u in flow.get('upstream', []):
            output += f"  - {u}\n"
        output += f"\n### Downstream (data consumers):\n"
        for d in flow.get('downstream', []):
            output += f"  - {d}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_find_impact_analysis(table_name: str) -> str:
    try:
        analyzer = GraphAnalyzer(get_neo4j())
        impact = analyzer.impact_analysis(table_name)
        output = f"## Impact Analysis for '{table_name}'\n\n"
        output += f"If this table changes, the following will be affected:\n\n"
        for item in impact.get('affected', []):
            output += f"  - {item['type']}: {item['name']} (via {item.get('relationship', 'unknown')})\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_test_rdf_connection() -> str:
    try:
        fuseki = get_fuseki()
        if fuseki.test_connection():
            return "✓ Connected to Apache Jena Fuseki"
        return "✗ Failed to connect to Fuseki"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_sync_graph_to_rdf() -> str:
    """
    COMPLETE RDF SYNC - Syncs entire Neo4j graph including:
    - Tables, Columns, Jobs, Datasets, DataSources
    - FK_TO, READS, WRITES, REPRESENTS relationships
    """
    from urllib.parse import quote
    import requests
    
    debug_log = []
    def log(msg):
        print(msg)
        debug_log.append(msg)
    
    log("=" * 70)
    log("  ★★★ COMPLETE RDF SYNC ★★★")
    log("=" * 70)
    
    try:
        # Get clients
        fuseki = get_fuseki()
        neo4j = get_neo4j()
        
        # Config
        fuseki_url = fuseki.config.url
        dataset = fuseki.config.dataset
        base_url = f"{fuseki_url}/{dataset}"
        graph_uri = "http://graphweaver.io/graph/main"
        auth = (fuseki.config.username, fuseki.config.password)
        
        log(f"[CONFIG] Base URL: {base_url}")
        log(f"[CONFIG] Graph URI: {graph_uri}")
        
        # Test Fuseki
        try:
            ping_resp = requests.get(f"{fuseki_url}/$/ping", timeout=5)
            if ping_resp.status_code != 200:
                return f"ERROR: Fuseki ping failed: {ping_resp.status_code}"
            log("[PING] ✓ Fuseki OK")
        except Exception as e:
            return f"ERROR: Cannot reach Fuseki: {e}"
        
        # Clear graph
        try:
            clear_resp = requests.post(
                f"{base_url}/update",
                data={"update": f"CLEAR GRAPH <{graph_uri}>"},
                auth=auth, timeout=30
            )
            log(f"[CLEAR] Response: {clear_resp.status_code}")
        except Exception as e:
            log(f"[CLEAR] Warning: {e}")
        
        # Build complete Turtle
        PREFIXES = """@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix dcat: <http://www.w3.org/ns/dcat#> .
@prefix dct: <http://purl.org/dc/terms/> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix gw: <http://graphweaver.io/ontology#> .
@prefix gwdata: <http://graphweaver.io/data#> .

"""
        
        def uri_safe(name):
            if name is None:
                return "unknown"
            return str(name).replace(" ", "_").replace("-", "_").replace(".", "_").replace("/", "_")
        
        turtle_lines = [PREFIXES]
        stats = {"tables": 0, "columns": 0, "jobs": 0, "datasets": 0, 
                 "datasources": 0, "fks": 0, "reads": 0, "writes": 0, "represents": 0,
                 "table_fks": 0, "col_props": 0, "belongs_to": 0}
        
        # 1. TABLES with columns (enhanced with more properties)
        log("[QUERY] Fetching tables and columns with full metadata...")
        tables = neo4j.run_query("""
            MATCH (t:Table)
            OPTIONAL MATCH (t)-[:BELONGS_TO]->(ds:DataSource)
            OPTIONAL MATCH (t)<-[:BELONGS_TO]-(c:Column)
            WITH t, ds, collect({
                name: c.name, 
                dataType: c.data_type,
                isPK: c.is_primary_key,
                isNullable: c.is_nullable,
                isUnique: c.is_unique
            }) as columns
            RETURN t.name as name, ds.id as datasource, columns
        """)
        
        for table in (tables or []):
            table_name = table.get("name")
            if not table_name:
                continue
            table_uri = f"gwdata:table_{uri_safe(table_name)}"
            ds_id = table.get("datasource") or "default"
            ds_uri = f"gwdata:datasource_{uri_safe(ds_id)}"
            
            # Table triples
            turtle_lines.append(f'{table_uri} a gw:Table, dcat:Dataset ;')
            turtle_lines.append(f'    rdfs:label "{table_name}" ;')
            turtle_lines.append(f'    dct:identifier "{table_name}" ;')
            turtle_lines.append(f'    gw:belongsToDataSource {ds_uri} .')
            stats["tables"] += 1
            stats["belongs_to"] += 1
            
            for col in table.get("columns", []):
                col_name = col.get("name") if col else None
                if col_name:
                    col_uri = f"gwdata:column_{uri_safe(table_name)}_{uri_safe(col_name)}"
                    data_type = col.get("dataType") or "unknown"
                    is_pk = col.get("isPK") or False
                    is_nullable = col.get("isNullable")
                    is_unique = col.get("isUnique") or False
                    
                    # Column triples with more properties
                    turtle_lines.append(f'{col_uri} a gw:Column ;')
                    turtle_lines.append(f'    rdfs:label "{col_name}" ;')
                    turtle_lines.append(f'    gw:columnName "{col_name}" ;')
                    turtle_lines.append(f'    gw:dataType "{data_type}" ;')
                    turtle_lines.append(f'    gw:belongsToTable {table_uri} ;')
                    if is_pk:
                        turtle_lines.append(f'    gw:isPrimaryKey "true"^^xsd:boolean ;')
                        stats["col_props"] += 1
                    if is_unique:
                        turtle_lines.append(f'    gw:isUnique "true"^^xsd:boolean ;')
                        stats["col_props"] += 1
                    if is_nullable is not None:
                        turtle_lines.append(f'    gw:isNullable "{str(is_nullable).lower()}"^^xsd:boolean ;')
                        stats["col_props"] += 1
                    # Remove trailing semicolon and close with period
                    if turtle_lines[-1].endswith(' ;'):
                        turtle_lines[-1] = turtle_lines[-1][:-2] + ' .'
                    else:
                        turtle_lines.append('.')
                    
                    turtle_lines.append(f'{table_uri} gw:hasColumn {col_uri} .')
                    stats["columns"] += 1
        
        log(f"[TABLES] {stats['tables']} tables, {stats['columns']} columns")
        
        # 2. JOBS
        log("[QUERY] Fetching jobs...")
        
        # DIAGNOSTIC: Check all node labels in database
        all_labels = neo4j.run_query("CALL db.labels() YIELD label RETURN label")
        log(f"[DIAGNOSTIC] All labels in Neo4j: {[l['label'] for l in (all_labels or [])]}")
        
        # DIAGNOSTIC: Count all nodes by label
        for label_row in (all_labels or []):
            label = label_row.get('label')
            if label:
                count_result = neo4j.run_query(f"MATCH (n:{label}) RETURN count(n) as cnt")
                cnt = count_result[0]['cnt'] if count_result else 0
                log(f"[DIAGNOSTIC]   {label}: {cnt} nodes")
        
        jobs = neo4j.run_query("""
            MATCH (j:Job)
            RETURN j.name as name, j.namespace as namespace, j.description as description
        """)
        
        for job in (jobs or []):
            job_name = job.get("name")
            if not job_name:
                continue
            job_uri = f"gwdata:job_{uri_safe(job_name)}"
            ns = job.get("namespace") or "default"
            desc = (job.get("description") or "").replace('"', '\\"')
            turtle_lines.append(f'{job_uri} a gw:Job ; rdfs:label "{job_name}" ; gw:namespace "{ns}" ; rdfs:comment "{desc}" .')
            stats["jobs"] += 1
        
        log(f"[JOBS] {stats['jobs']} jobs")
        
        # 3. DATASETS
        log("[QUERY] Fetching datasets...")
        datasets = neo4j.run_query("""
            MATCH (d:Dataset)
            RETURN d.name as name, d.namespace as namespace
        """)
        
        for ds in (datasets or []):
            ds_name = ds.get("name")
            if not ds_name:
                continue
            ds_uri = f"gwdata:dataset_{uri_safe(ds_name)}"
            ns = ds.get("namespace") or "default"
            turtle_lines.append(f'{ds_uri} a gw:Dataset ; rdfs:label "{ds_name}" ; gw:namespace "{ns}" .')
            stats["datasets"] += 1
        
        log(f"[DATASETS] {stats['datasets']} datasets")
        
        # 4. DATASOURCES and Table-DataSource relationships
        log("[QUERY] Fetching datasources...")
        datasources = neo4j.run_query("MATCH (ds:DataSource) RETURN ds.id as id")
        
        for ds in (datasources or []):
            ds_id = ds.get("id")
            if ds_id:
                ds_uri = f"gwdata:datasource_{uri_safe(ds_id)}"
                turtle_lines.append(f'{ds_uri} a gw:DataSource ; rdfs:label "{ds_id}" .')
                stats["datasources"] += 1
        
        log(f"[DATASOURCES] {stats['datasources']} datasources")
        
        # 4b. Table -> DataSource BELONGS_TO relationships
        log("[QUERY] Fetching Table-DataSource relationships...")
        table_ds = neo4j.run_query("""
            MATCH (t:Table)-[:BELONGS_TO]->(ds:DataSource)
            RETURN t.name as table, ds.id as datasource
        """)
        
        stats["table_datasource"] = 0
        for rel in (table_ds or []):
            table_name = rel.get("table")
            ds_id = rel.get("datasource")
            if table_name and ds_id:
                table_uri = f"gwdata:table_{uri_safe(table_name)}"
                ds_uri = f"gwdata:datasource_{uri_safe(ds_id)}"
                turtle_lines.append(f'{table_uri} gw:belongsToDataSource {ds_uri} .')
                stats["table_datasource"] += 1
        
        log(f"[TABLE-DS] {stats['table_datasource']} Table-DataSource relationships")
        
        # 5. FK_TO relationships (with cardinality)
        log("[QUERY] Fetching FK relationships...")
        fks = neo4j.run_query("""
            MATCH (sc:Column)-[fk:FK_TO]->(tc:Column)
            MATCH (sc)-[:BELONGS_TO]->(st:Table)
            MATCH (tc)-[:BELONGS_TO]->(tt:Table)
            RETURN st.name as srcTable, sc.name as srcCol, 
                   tt.name as tgtTable, tc.name as tgtCol,
                   fk.score as confidence, fk.cardinality as cardinality
        """)
        
        for fk in (fks or []):
            src_uri = f"gwdata:column_{uri_safe(fk['srcTable'])}_{uri_safe(fk['srcCol'])}"
            tgt_uri = f"gwdata:column_{uri_safe(fk['tgtTable'])}_{uri_safe(fk['tgtCol'])}"
            src_table_uri = f"gwdata:table_{uri_safe(fk['srcTable'])}"
            tgt_table_uri = f"gwdata:table_{uri_safe(fk['tgtTable'])}"
            conf = fk.get("confidence") or 1.0
            card = fk.get("cardinality") or "unknown"
            
            # Column-to-Column FK
            turtle_lines.append(f'{src_uri} gw:referencesColumn {tgt_uri} .')
            turtle_lines.append(f'{src_uri} gw:fkConfidence "{conf}"^^xsd:float .')
            turtle_lines.append(f'{src_uri} gw:fkCardinality "{card}" .')
            
            # Table-to-Table FK (for easier querying)
            turtle_lines.append(f'{src_table_uri} gw:referencesTable {tgt_table_uri} .')
            turtle_lines.append(f'{tgt_table_uri} gw:referencedByTable {src_table_uri} .')
            stats["table_fks"] += 1
            
            stats["fks"] += 1
        
        log(f"[FKs] {stats['fks']} column FKs, {stats.get('table_fks', 0)} table FKs")
        
        # 6. READS relationships
        log("[QUERY] Fetching READS relationships...")
        reads = neo4j.run_query("""
            MATCH (j:Job)-[:READS]->(d:Dataset)
            RETURN j.name as job, d.name as dataset
        """)
        
        for r in (reads or []):
            job_uri = f"gwdata:job_{uri_safe(r['job'])}"
            ds_uri = f"gwdata:dataset_{uri_safe(r['dataset'])}"
            turtle_lines.append(f'{job_uri} gw:readsDataset {ds_uri} .')
            stats["reads"] += 1
        
        log(f"[READS] {stats['reads']} READS relationships")
        
        # 7. WRITES relationships
        log("[QUERY] Fetching WRITES relationships...")
        writes = neo4j.run_query("""
            MATCH (j:Job)-[:WRITES]->(d:Dataset)
            RETURN j.name as job, d.name as dataset
        """)
        
        for w in (writes or []):
            job_uri = f"gwdata:job_{uri_safe(w['job'])}"
            ds_uri = f"gwdata:dataset_{uri_safe(w['dataset'])}"
            turtle_lines.append(f'{job_uri} gw:writesDataset {ds_uri} .')
            stats["writes"] += 1
        
        log(f"[WRITES] {stats['writes']} WRITES relationships")
        
        # 8. REPRESENTS relationships
        log("[QUERY] Fetching REPRESENTS relationships...")
        represents = neo4j.run_query("""
            MATCH (d:Dataset)-[:REPRESENTS]->(t:Table)
            RETURN d.name as dataset, t.name as table
        """)
        
        for rep in (represents or []):
            ds_uri = f"gwdata:dataset_{uri_safe(rep['dataset'])}"
            table_uri = f"gwdata:table_{uri_safe(rep['table'])}"
            turtle_lines.append(f'{ds_uri} gw:representsTable {table_uri} .')
            stats["represents"] += 1
        
        log(f"[REPRESENTS] {stats['represents']} REPRESENTS relationships")
        
        # Build and insert Turtle
        turtle_content = "\n".join(turtle_lines)
        log(f"[TURTLE] Total size: {len(turtle_content)} bytes")
        
        insert_url = f"{base_url}/data?graph={quote(graph_uri, safe='')}"
        log(f"[INSERT] URL: {insert_url}")
        
        try:
            insert_resp = requests.post(
                insert_url,
                data=turtle_content.encode('utf-8'),
                headers={"Content-Type": "text/turtle; charset=utf-8"},
                auth=auth,
                timeout=60
            )
            log(f"[INSERT] Response: {insert_resp.status_code}")
            
            if insert_resp.status_code not in [200, 201, 204]:
                return f"ERROR: Insert failed: {insert_resp.status_code}\n{insert_resp.text}"
            log("[INSERT] ✓ Success")
        except Exception as e:
            return f"ERROR: Insert failed: {e}"
        
        # Count triples
        try:
            count_resp = requests.post(
                f"{base_url}/sparql",
                data={"query": f"SELECT (COUNT(*) as ?count) WHERE {{ GRAPH <{graph_uri}> {{ ?s ?p ?o }} }}"},
                headers={"Accept": "application/sparql-results+json"},
                timeout=30
            )
            total_triples = 0
            if count_resp.status_code == 200:
                bindings = count_resp.json().get("results", {}).get("bindings", [])
                if bindings:
                    total_triples = int(bindings[0].get("count", {}).get("value", 0))
            log(f"[COUNT] ★ Total triples: {total_triples}")
        except Exception as e:
            log(f"[COUNT] Error: {e}")
            total_triples = sum(stats.values())
        
        log("=" * 70)
        log(f"  SYNC COMPLETE: {total_triples} triples")
        log("=" * 70)
        
        # Build output
        output = "## ★ Graph Synced to RDF ★\n\n"
        output += "### Nodes Synced:\n"
        output += f"- Tables: {stats['tables']}\n"
        output += f"- Columns: {stats['columns']} (with {stats.get('col_props', 0)} extra properties)\n"
        output += f"- Jobs: {stats['jobs']}\n"
        output += f"- Datasets: {stats['datasets']}\n"
        output += f"- DataSources: {stats['datasources']}\n\n"
        output += "### Relationships Synced:\n"
        output += f"- Column FK references: {stats['fks']}\n"
        output += f"- Table FK references: {stats.get('table_fks', 0)}\n"
        output += f"- Table-DataSource: {stats.get('belongs_to', 0)}\n"
        output += f"- READS (lineage): {stats['reads']}\n"
        output += f"- WRITES (lineage): {stats['writes']}\n"
        output += f"- REPRESENTS: {stats['represents']}\n\n"
        output += f"**Total RDF triples: {total_triples}**\n"
        
        return output
        
    except Exception as e:
        import traceback
        return f"ERROR: {e}\n\nDebug:\n" + "\n".join(debug_log) + f"\n\n{traceback.format_exc()}"


@debug_tool
def impl_run_sparql(query: str) -> str:
    try:
        results = get_fuseki().sparql_query(query)
        if not results:
            return "Query executed. No results."
        output = f"Results ({len(results)} rows):\n"
        for row in results[:50]:
            output += f"  {row}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_sparql_list_tables() -> str:
    try:
        results = get_sparql().list_tables()
        if not results:
            return "No tables found in RDF store."
        output = "## Tables in RDF Store\n"
        for r in results:
            output += f"- {r.get('label', 'unknown')}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_sparql_get_foreign_keys(table_name: str = None) -> str:
    try:
        results = get_sparql().get_foreign_keys(table_name)
        if not results:
            return "No foreign keys found."
        output = "## Foreign Keys\n"
        for r in results:
            output += f"- {r.get('sourceTableLabel')}.{r.get('sourceColLabel')} → {r.get('targetTableLabel')}.{r.get('targetColLabel')}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_sparql_table_lineage(table_name: str) -> str:
    try:
        results = get_sparql().get_table_lineage(table_name)
        if not results:
            return f"No lineage found for {table_name}"
        output = f"## Lineage for {table_name}\n"
        for r in results:
            output += f"- {r.get('jobLabel')} {r.get('direction')} {r.get('datasetLabel')}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_sparql_downstream_impact(table_name: str) -> str:
    try:
        results = get_sparql().get_downstream_impact(table_name)
        if not results:
            return f"No downstream impact found for {table_name}"
        output = f"## Downstream Impact for {table_name}\n"
        for r in results:
            output += f"- {r.get('dependentTableLabel')} ({r.get('relationshipType')})\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_sparql_hub_tables(min_connections: int = 3) -> str:
    try:
        results = get_sparql().get_hub_tables(min_connections)
        if not results:
            return f"No hub tables found with {min_connections}+ connections."
        output = f"## Hub Tables ({min_connections}+ connections)\n"
        for r in results:
            output += f"- {r.get('label')}: {r.get('totalConnections')} connections\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_sparql_orphan_tables() -> str:
    try:
        results = get_sparql().find_orphan_tables()
        if not results:
            return "No orphan tables found."
        output = "## Orphan Tables\n"
        for r in results:
            output += f"- {r.get('label')}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_sparql_search(search_term: str) -> str:
    try:
        results = get_sparql().search_by_label(search_term)
        if not results:
            return f"No results for '{search_term}'"
        output = f"## Search Results for '{search_term}'\n"
        for r in results:
            output += f"- {r.get('type', 'unknown')}: {r.get('label')}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_get_rdf_statistics() -> str:
    try:
        stats = get_sparql().get_statistics()
        output = "## RDF Statistics\n"
        for k, v in stats.items():
            output += f"- {k}: {v}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_export_rdf_turtle() -> str:
    try:
        turtle = get_fuseki().export_turtle()
        return f"## RDF Turtle Export\n```turtle\n{turtle[:5000]}{'...' if len(turtle) > 5000 else ''}\n```"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_learn_rules_with_ltn(epochs: int = 100) -> str:
    if not LTN_AVAILABLE:
        return "LTN not available. Please install ltn package."
    try:
        learner = get_rule_learner()
        results = learner.learn_rules()
        output = "## LTN Rule Learning Results\n"
        output += f"- Rules learned: {len(results.get('rules', []))}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_generate_business_rules_from_ltn() -> str:
    if not LTN_AVAILABLE:
        return "LTN not available."
    try:
        generator = get_rule_generator()
        rules = generator.generate_from_learned_rules(get_rule_learner().learned_rules)
        output = f"## Generated {len(rules)} business rules from LTN\n"
        for r in rules[:10]:
            output += f"- {r.name}: {r.description}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_generate_all_validation_rules() -> str:
    if not LTN_AVAILABLE:
        return "LTN not available."
    try:
        generator = get_rule_generator()
        rules = generator.generate_all_from_graph()
        st.session_state.generated_rules = rules
        output = f"## Generated {len(rules)} validation rules\n"
        for r in rules[:10]:
            output += f"- {r.name}: {r.description}\n"
        if len(rules) > 10:
            output += f"... and {len(rules) - 10} more\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_export_generated_rules_yaml(file_path: str = "business_rules_generated.yaml") -> str:
    if "generated_rules" not in st.session_state:
        return "No generated rules. Run generate_all_validation_rules first."
    try:
        generator = get_rule_generator()
        yaml_content = generator.export_yaml(st.session_state.generated_rules)
        with open(file_path, 'w') as f:
            f.write(yaml_content)
        return f"✓ Exported {len(st.session_state.generated_rules)} rules to {file_path}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_export_generated_rules_sql() -> str:
    if "generated_rules" not in st.session_state:
        return "No generated rules."
    try:
        generator = get_rule_generator()
        sql = generator.export_sql(st.session_state.generated_rules)
        return f"## Generated SQL\n```sql\n{sql[:3000]}{'...' if len(sql) > 3000 else ''}\n```"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_show_ltn_knowledge_base() -> str:
    if not LTN_AVAILABLE:
        return "LTN not available."
    try:
        kb = LTNKnowledgeBase.create_default()
        output = "## LTN Knowledge Base\n\n### Axioms:\n"
        for axiom in kb.get_all_axioms():
            output += f"- **{axiom.name}**: {axiom.formula}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_check_tool_exists(tool_name: str) -> str:
    r = get_registry()
    if r.tool_exists(tool_name):
        return f"✓ Tool '{tool_name}' exists"
    return f"✗ Tool '{tool_name}' not found"


@debug_tool
def impl_list_available_tools() -> str:
    r = get_registry()
    tools = r.list_tools()
    output = "## Available Dynamic Tools\n"
    for t in tools:
        output += f"- **{t['name']}**: {t['description']}\n"
    return output


@debug_tool
def impl_create_dynamic_tool(name: str, description: str, code: str) -> str:
    r = get_registry()
    if r.tool_exists(name):
        return f"ERROR: Tool '{name}' already exists."
    if "def run(" not in code:
        return "ERROR: Code must define a run() function."
    try:
        compile(code, name, "exec")
        path = r.create_tool(name, description, code)
        return f"✓ Created tool '{name}'"
    except SyntaxError as e:
        return f"ERROR: Syntax error: {e}"


@debug_tool
def impl_run_dynamic_tool(tool_name: str, **kwargs) -> str:
    r = get_registry()
    if not r.tool_exists(tool_name):
        return f"ERROR: Tool '{tool_name}' not found."
    try:
        return str(r.execute_tool(tool_name, **kwargs))
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_get_tool_source(tool_name: str) -> str:
    r = get_registry()
    if not r.tool_exists(tool_name):
        return f"ERROR: Tool '{tool_name}' not found"
    return r.get_tool_source(tool_name)


@debug_tool
def impl_update_dynamic_tool(tool_name: str, code: str, description: str = None) -> str:
    r = get_registry()
    if not r.tool_exists(tool_name):
        return f"ERROR: Tool '{tool_name}' not found."
    try:
        r.update_tool(tool_name, code, description)
        return f"✓ Updated tool '{tool_name}'"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@debug_tool
def impl_delete_dynamic_tool(tool_name: str) -> str:
    r = get_registry()
    if not r.tool_exists(tool_name):
        return f"ERROR: Tool '{tool_name}' not found"
    r.delete_tool(tool_name)
    return f"✓ Deleted tool '{tool_name}'"


# =============================================================================
# Tool Function Mapping
# =============================================================================

STREAMING_TOOL_FUNCTIONS = {
    "check_tool_exists": lambda **kw: impl_check_tool_exists(**kw),
    "list_available_tools": lambda **kw: impl_list_available_tools(),
    "create_dynamic_tool": lambda **kw: impl_create_dynamic_tool(**kw),
    "run_dynamic_tool": lambda **kw: impl_run_dynamic_tool(**kw),
    "get_tool_source": lambda **kw: impl_get_tool_source(**kw),
    "update_dynamic_tool": lambda **kw: impl_update_dynamic_tool(**kw),
    "delete_dynamic_tool": lambda **kw: impl_delete_dynamic_tool(**kw),
    "configure_database": lambda **kw: impl_configure_database(**kw),
    "test_database_connection": lambda **kw: impl_test_database_connection(),
    "list_database_tables": lambda **kw: impl_list_database_tables(),
    "get_table_schema": lambda **kw: impl_get_table_schema(**kw),
    "get_column_stats": lambda **kw: impl_get_column_stats(**kw),
    "run_fk_discovery": lambda **kw: impl_run_fk_discovery(**kw),
    "analyze_potential_fk": lambda **kw: impl_analyze_potential_fk(**kw),
    "validate_fk_with_data": lambda **kw: impl_validate_fk_with_data(**kw),
    "clear_neo4j_graph": lambda **kw: impl_clear_neo4j_graph(),
    "add_fk_to_graph": lambda **kw: impl_add_fk_to_graph(**kw),
    "get_graph_stats": lambda **kw: impl_get_graph_stats(),
    "analyze_graph_centrality": lambda **kw: impl_analyze_graph_centrality(),
    "find_table_communities": lambda **kw: impl_find_table_communities(),
    "predict_missing_fks": lambda **kw: impl_predict_missing_fks(),
    "run_cypher": lambda **kw: impl_run_cypher(**kw),
    "connect_datasets_to_tables": lambda **kw: impl_connect_datasets_to_tables(),
    "generate_text_embeddings": lambda **kw: impl_generate_text_embeddings(),
    "generate_kg_embeddings": lambda **kw: impl_generate_kg_embeddings(),
    "create_vector_indexes": lambda **kw: impl_create_vector_indexes(),
    "semantic_search_tables": lambda **kw: impl_semantic_search_tables(**kw),
    "semantic_search_columns": lambda **kw: impl_semantic_search_columns(**kw),
    "find_similar_tables": lambda **kw: impl_find_similar_tables(**kw),
    "find_similar_columns": lambda **kw: impl_find_similar_columns(**kw),
    "predict_fks_from_embeddings": lambda **kw: impl_predict_fks_from_embeddings(**kw),
    "semantic_fk_discovery": lambda **kw: impl_semantic_fk_discovery(**kw),
    "show_sample_business_rules": lambda **kw: impl_show_sample_business_rules(),
    "load_business_rules": lambda **kw: impl_load_business_rules(**kw),
    "load_business_rules_from_file": lambda **kw: impl_load_business_rules_from_file(**kw),
    "list_business_rules": lambda **kw: impl_list_business_rules(),
    "execute_business_rule": lambda **kw: impl_execute_business_rule(**kw),
    "execute_all_business_rules": lambda **kw: impl_execute_all_business_rules(**kw),
    "get_marquez_lineage": lambda **kw: impl_get_marquez_lineage(**kw),
    "list_marquez_jobs": lambda **kw: impl_list_marquez_jobs(),
    "import_lineage_to_graph": lambda **kw: impl_import_lineage_to_graph(),
    "analyze_data_flow": lambda **kw: impl_analyze_data_flow(**kw),
    "find_impact_analysis": lambda **kw: impl_find_impact_analysis(**kw),
    "test_rdf_connection": lambda **kw: impl_test_rdf_connection(),
    "sync_graph_to_rdf": lambda **kw: impl_sync_graph_to_rdf(),
    "run_sparql": lambda **kw: impl_run_sparql(**kw),
    "sparql_list_tables": lambda **kw: impl_sparql_list_tables(),
    "sparql_get_foreign_keys": lambda **kw: impl_sparql_get_foreign_keys(**kw),
    "sparql_table_lineage": lambda **kw: impl_sparql_table_lineage(**kw),
    "sparql_downstream_impact": lambda **kw: impl_sparql_downstream_impact(**kw),
    "sparql_hub_tables": lambda **kw: impl_sparql_hub_tables(**kw),
    "sparql_orphan_tables": lambda **kw: impl_sparql_orphan_tables(),
    "sparql_search": lambda **kw: impl_sparql_search(**kw),
    "get_rdf_statistics": lambda **kw: impl_get_rdf_statistics(),
    "export_rdf_turtle": lambda **kw: impl_export_rdf_turtle(),
    "learn_rules_with_ltn": lambda **kw: impl_learn_rules_with_ltn(**kw),
    "generate_business_rules_from_ltn": lambda **kw: impl_generate_business_rules_from_ltn(),
    "generate_all_validation_rules": lambda **kw: impl_generate_all_validation_rules(),
    "export_generated_rules_yaml": lambda **kw: impl_export_generated_rules_yaml(**kw),
    "export_generated_rules_sql": lambda **kw: impl_export_generated_rules_sql(),
    "show_ltn_knowledge_base": lambda **kw: impl_show_ltn_knowledge_base(),
}


# =============================================================================
# Streaming Chat with Full Debug Logging
# =============================================================================

def stream_agent_response(client: anthropic.Anthropic, messages: List[Dict], message_placeholder) -> str:
    """Stream response from Claude with FULL terminal debugging."""
    
    api_logger = APIStreamLogger()
    full_response = ""
    tool_results_log = []
    
    while True:
        tool_inputs = {}
        current_block_id = None
        current_block_type = None
        current_response_content = []
        
        api_logger.on_stream_start("claude-sonnet-4-20250514", messages)
        
        try:
            with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=8192,
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=STREAMING_TOOLS,
            ) as stream:
                for event in stream:
                    if event.type == "content_block_start":
                        current_block_type = event.content_block.type
                        api_logger.on_content_block_start(
                            event.content_block.type,
                            getattr(event.content_block, 'id', None),
                            getattr(event.content_block, 'name', None)
                        )
                        
                        if event.content_block.type == "tool_use":
                            current_block_id = event.content_block.id
                            tool_inputs[current_block_id] = {
                                "name": event.content_block.name,
                                "input": ""
                            }
                            full_response += f"\n\n🔧 **Calling: {event.content_block.name}**\n"
                            message_placeholder.markdown(full_response + "▌")
                            
                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            api_logger.on_text_delta(event.delta.text)
                            full_response += event.delta.text
                            message_placeholder.markdown(full_response + "▌")
                        elif event.delta.type == "input_json_delta":
                            api_logger.on_input_json_delta(event.delta.partial_json)
                            if current_block_id:
                                tool_inputs[current_block_id]["input"] += event.delta.partial_json
                                
                    elif event.type == "content_block_stop":
                        api_logger.on_content_block_stop(current_block_type)
                        current_block_id = None
                        current_block_type = None
                
                response = stream.get_final_message()
                current_response_content = response.content
                api_logger.on_stream_end(response.stop_reason)
                
        except Exception as e:
            api_logger.on_error(e)
            debug.error(f"API streaming error: {e}", e)
            raise
        
        messages.append({"role": "assistant", "content": current_response_content})
        
        if response.stop_reason != "tool_use":
            debug.agent(f"Agent finished with stop_reason: {response.stop_reason}")
            break
        
        tool_results = []
        for block in current_response_content:
            if block.type == "tool_use":
                debug.section(f"EXECUTING TOOL: {block.name}")
                debug.tool(f"Tool inputs:", block.input)
                
                full_response += f"\n⏳ Executing...\n"
                message_placeholder.markdown(full_response + "▌")
                
                fn = STREAMING_TOOL_FUNCTIONS.get(block.name)
                if fn:
                    try:
                        start_time = time.time()
                        result = fn(**block.input)
                        duration = time.time() - start_time
                        
                        debug.tool(f"✓ Tool completed in {duration*1000:.1f}ms")
                        api_logger.on_tool_result(block.name, result)
                        
                    except Exception as e:
                        debug.error(f"Tool execution failed: {e}", e)
                        import traceback
                        traceback.print_exc()
                        result = f"Error: {type(e).__name__}: {e}"
                else:
                    debug.error(f"Unknown tool: {block.name}")
                    result = f"Unknown tool: {block.name}"
                
                full_response += f"\n{result}\n"
                message_placeholder.markdown(full_response + "▌")
                tool_results_log.append({"tool": block.name, "result": result})
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
        
        messages.append({"role": "user", "content": tool_results})
        debug.agent("Sending tool results back to agent, continuing loop...")
    
    message_placeholder.markdown(full_response)
    return full_response


def get_anthropic_client():
    api_key = st.session_state.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


# =============================================================================
# Streamlit UI
# =============================================================================

def main():
    st.set_page_config(
        page_title="GraphWeaver Agent",
        page_icon="🕸️",
        layout="wide",
    )
    
    st.title("🕸️ GraphWeaver Agent")
    st.caption("Chat with Claude to discover FK relationships, build knowledge graphs, and analyze data lineage")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        if DEBUG_MODE:
            st.success("🔍 DEBUG MODE: ON")
            st.caption("Check terminal for logs")
        
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=st.session_state.get("anthropic_api_key", os.environ.get("ANTHROPIC_API_KEY", "")),
        )
        if api_key:
            st.session_state.anthropic_api_key = api_key
        
        st.divider()
        
        st.subheader("🗄️ PostgreSQL")
        pg_host = st.text_input("Host", value=st.session_state.get("pg_host", os.environ.get("POSTGRES_HOST", "localhost")))
        pg_port = st.number_input("Port", value=int(st.session_state.get("pg_port", os.environ.get("POSTGRES_PORT", "5432"))))
        pg_database = st.text_input("Database", value=st.session_state.get("pg_database", os.environ.get("POSTGRES_DB", "orders")))
        pg_username = st.text_input("Username", value=st.session_state.get("pg_username", os.environ.get("POSTGRES_USER", "saphenia")))
        pg_password = st.text_input("Password", type="password", value=st.session_state.get("pg_password", os.environ.get("POSTGRES_PASSWORD", "secret")))
        
        st.divider()
        
        st.subheader("🔵 Neo4j")
        neo4j_uri = st.text_input("URI", value=st.session_state.get("neo4j_uri", os.environ.get("NEO4J_URI", "bolt://localhost:7687")))
        neo4j_user = st.text_input("Neo4j User", value=st.session_state.get("neo4j_user", os.environ.get("NEO4J_USER", "neo4j")))
        neo4j_password = st.text_input("Neo4j Password", type="password", value=st.session_state.get("neo4j_password", os.environ.get("NEO4J_PASSWORD", "password")))
        
        st.divider()
        
        st.subheader("🚀 Quick Actions")
        if st.button("🔍 Discover FKs", use_container_width=True):
            st.session_state.quick_action = "Discover all foreign key relationships in the database"
        if st.button("📊 List Tables", use_container_width=True):
            st.session_state.quick_action = "List all database tables"
        if st.button("🧠 Generate Embeddings", use_container_width=True):
            st.session_state.quick_action = "Generate text embeddings for semantic search"
        if st.button("📈 Analyze Graph", use_container_width=True):
            st.session_state.quick_action = "Analyze graph centrality and find communities"
        
        st.divider()
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.streaming_messages = []
            st.rerun()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "streaming_messages" not in st.session_state:
        st.session_state.streaming_messages = []
    
    if not st.session_state.get("anthropic_api_key") and not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("⚠️ Please enter your Anthropic API key in the sidebar to start chatting.")
        st.stop()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if "quick_action" in st.session_state and st.session_state.quick_action:
        prompt = st.session_state.quick_action
        st.session_state.quick_action = None
    else:
        prompt = st.chat_input("Ask me about your database...")
    
    if prompt:
        debug.section("NEW USER MESSAGE")
        debug.agent(f"User: {prompt}")
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            try:
                client = get_anthropic_client()
                if client is None:
                    st.error("Failed to create Anthropic client. Please check your API key.")
                    st.stop()
                
                st.session_state.streaming_messages.append({"role": "user", "content": prompt})
                message_placeholder = st.empty()
                
                response = stream_agent_response(
                    client, 
                    st.session_state.streaming_messages.copy(),
                    message_placeholder
                )
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.streaming_messages.append({"role": "assistant", "content": response})
                
                if len(st.session_state.streaming_messages) > 20:
                    st.session_state.streaming_messages = st.session_state.streaming_messages[-20:]
                    
            except Exception as e:
                debug.error(f"Fatal error: {e}", e)
                import traceback
                error_msg = f"Error: {type(e).__name__}: {e}\n\n```\n{traceback.format_exc()}\n```"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})


@tool
def test_database_connection_tool() -> str:
    """Test connection to PostgreSQL database."""
    return impl_test_database_connection()

@tool
def list_database_tables_tool() -> str:
    """List all tables with row counts."""
    return impl_list_database_tables()

@tool
def run_fk_discovery_tool(min_match_rate: float = 0.95, min_score: float = 0.5) -> str:
    """Run complete 5-stage FK discovery pipeline."""
    return impl_run_fk_discovery(min_match_rate, min_score)

@tool
def get_graph_stats_tool() -> str:
    """Get current graph statistics."""
    return impl_get_graph_stats()


if __name__ == "__main__":
    main()
