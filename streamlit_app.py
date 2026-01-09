#!/usr/bin/env python3
# =============================================================================
# FILE: streamlit_app.py
# PATH: /home/gp/Downloads/graphweaver-agent/streamlit_app.py
# =============================================================================
"""
graphweaver-agent/streamlit_app.py

Streamlit Chat Interface for GraphWeaver Agent with Real-Time Streaming

UPDATED: Now uses langchain.agents.create_agent API instead of direct Anthropic SDK
UPDATED: Added conversation memory with sidebar history

WITH TERMINAL DEBUG LOGGING - Run with: DEBUG=1 streamlit run streamlit_app.py
"""
import os
import sys
import streamlit as st
from typing import Optional, Generator, Dict, Any, List
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

# =============================================================================
# NEW LangChain Imports - Using create_agent API
# =============================================================================
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import wrap_tool_call
from langchain.tools import tool
from langchain.messages import ToolMessage, HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

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
# NEW: Conversation Memory Imports
# =============================================================================
from conversation_memory import get_memory, Conversation
from conversation_sidebar import (
    render_conversation_sidebar,
    render_conversation_actions,
    add_message,
    get_current_streaming_messages,
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
                
                debug.neo4j("Clearing schema data (preserving lineage)...")
                
                neo4j.run_write("MATCH ()-[r:FK_TO]->() DELETE r")
                neo4j.run_write("MATCH (c:Column) DETACH DELETE c")
                neo4j.run_write("MATCH (t:Table) DETACH DELETE t")
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
    debug.neo4j("Getting comprehensive graph statistics...")
    neo4j = get_neo4j()
    
    output = "## Neo4j Graph Statistics\n\n"
    
    config = get_neo4j_config()
    output += f"**Connection:** {config.uri}\n"
    output += f"**Database:** {config.database}\n\n"
    
    labels_result = neo4j.run_query("CALL db.labels() YIELD label RETURN label")
    labels = [r['label'] for r in (labels_result or [])]
    output += f"**Labels found:** {labels}\n\n"
    
    output += "### Node Counts:\n"
    for label in labels:
        try:
            count_result = neo4j.run_query(f"MATCH (n:`{label}`) RETURN count(n) as cnt")
            cnt = count_result[0]['cnt'] if count_result else 0
            output += f"- {label}: {cnt}\n"
        except Exception as e:
            output += f"- {label}: ERROR - {e}\n"
    
    rel_result = neo4j.run_query("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
    rel_types = [r['relationshipType'] for r in (rel_result or [])]
    output += f"\n**Relationship types:** {rel_types}\n\n"
    
    output += "### Relationship Counts:\n"
    for rel_type in rel_types:
        try:
            count_result = neo4j.run_query(f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) as cnt")
            cnt = count_result[0]['cnt'] if count_result else 0
            output += f"- {rel_type}: {cnt}\n"
        except Exception as e:
            output += f"- {rel_type}: ERROR - {e}\n"
    
    output += "\n### Sample Data:\n"
    
    jobs = neo4j.run_query("MATCH (j:Job) RETURN j.name as name LIMIT 5")
    if jobs:
        output += f"- Jobs: {[j['name'] for j in jobs]}\n"
    else:
        output += "- Jobs: (none found)\n"
    
    datasets = neo4j.run_query("MATCH (d:Dataset) RETURN d.name as name LIMIT 5")
    if datasets:
        output += f"- Datasets: {[d['name'] for d in datasets]}\n"
    else:
        output += "- Datasets: (none found)\n"
    
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
    debug.neo4j("Connecting datasets to tables (fuzzy match)...")
    neo4j = get_neo4j()
    
    datasets = neo4j.run_query("MATCH (d:Dataset) RETURN d.name as name")
    tables = neo4j.run_query("MATCH (t:Table) RETURN t.name as name")
    
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
        
        possible_names = [
            ds_name,
            ds_name.split('.')[-1],
            ds_name.split('/')[-1],
            ds_name.split('.')[-1].split('/')[-1],
        ]
        
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
        
        if "error" in result:
            return f"ERROR: {result['error']}\nStats: {result.get('stats', {})}"
        
        if "warning" in result:
            return f"WARNING: {result['warning']}\nStats: {result.get('stats', {})}"
        
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
        
        if 'actual_jobs' in stats:
            output += "\n### Verified in Neo4j:\n"
            output += f"- Jobs: {stats.get('actual_jobs', 0)}\n"
            output += f"- Datasets: {stats.get('actual_datasets', 0)}\n"
            output += f"- READS edges: {stats.get('actual_reads', 0)}\n"
            output += f"- WRITES edges: {stats.get('actual_writes', 0)}\n"
        
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
        fuseki = get_fuseki()
        neo4j = get_neo4j()
        
        fuseki_url = fuseki.config.url
        dataset = fuseki.config.dataset
        base_url = f"{fuseki_url}/{dataset}"
        graph_uri = "http://graphweaver.io/graph/main"
        auth = (fuseki.config.username, fuseki.config.password)
        
        log(f"[CONFIG] Base URL: {base_url}")
        log(f"[CONFIG] Graph URI: {graph_uri}")
        
        try:
            ping_resp = requests.get(f"{fuseki_url}/$/ping", timeout=5)
            if ping_resp.status_code != 200:
                return f"ERROR: Fuseki ping failed: {ping_resp.status_code}"
            log("[PING] ✓ Fuseki OK")
        except Exception as e:
            return f"ERROR: Cannot reach Fuseki: {e}"
        
        try:
            clear_resp = requests.post(
                f"{base_url}/update",
                data={"update": f"CLEAR GRAPH <{graph_uri}>"},
                auth=auth, timeout=30
            )
            log(f"[CLEAR] Response: {clear_resp.status_code}")
        except Exception as e:
            log(f"[CLEAR] Warning: {e}")
        
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
                    turtle_lines.append(f'{col_uri} a gw:Column ;')
                    turtle_lines.append(f'    rdfs:label "{col_name}" ;')
                    turtle_lines.append(f'    gw:belongsToTable {table_uri} .')
                    stats["columns"] += 1
        
        log(f"[TABLES] {stats['tables']} tables, {stats['columns']} columns")
        
        fks = neo4j.run_query("""
            MATCH (c1:Column)-[fk:FK_TO]->(c2:Column)
            MATCH (c1)-[:BELONGS_TO]->(t1:Table)
            MATCH (c2)-[:BELONGS_TO]->(t2:Table)
            RETURN t1.name as srcTable, c1.name as srcCol, 
                   t2.name as tgtTable, c2.name as tgtCol,
                   fk.score as score
        """)
        
        for fk in (fks or []):
            src_col_uri = f"gwdata:column_{uri_safe(fk.get('srcTable'))}_{uri_safe(fk.get('srcCol'))}"
            tgt_col_uri = f"gwdata:column_{uri_safe(fk.get('tgtTable'))}_{uri_safe(fk.get('tgtCol'))}"
            turtle_lines.append(f'{src_col_uri} gw:foreignKeyTo {tgt_col_uri} .')
            stats["fks"] += 1
        
        log(f"[FKS] {stats['fks']} FK relationships")
        
        jobs = neo4j.run_query("MATCH (j:Job) RETURN j.name as name, j.namespace as ns")
        for job in (jobs or []):
            job_name = job.get("name")
            if job_name:
                job_uri = f"gwdata:job_{uri_safe(job_name)}"
                turtle_lines.append(f'{job_uri} a gw:Job, prov:Activity ;')
                turtle_lines.append(f'    rdfs:label "{job_name}" .')
                stats["jobs"] += 1
        
        datasets_q = neo4j.run_query("MATCH (d:Dataset) RETURN d.name as name, d.namespace as ns")
        for ds in (datasets_q or []):
            ds_name = ds.get("name")
            if ds_name:
                ds_uri = f"gwdata:dataset_{uri_safe(ds_name)}"
                turtle_lines.append(f'{ds_uri} a gw:Dataset, dcat:Dataset ;')
                turtle_lines.append(f'    rdfs:label "{ds_name}" .')
                stats["datasets"] += 1
        
        reads = neo4j.run_query("MATCH (j:Job)-[:READS]->(d:Dataset) RETURN j.name as job, d.name as dataset")
        for r in (reads or []):
            job_uri = f"gwdata:job_{uri_safe(r.get('job'))}"
            ds_uri = f"gwdata:dataset_{uri_safe(r.get('dataset'))}"
            turtle_lines.append(f'{job_uri} gw:reads {ds_uri} .')
            stats["reads"] += 1
        
        writes = neo4j.run_query("MATCH (j:Job)-[:WRITES]->(d:Dataset) RETURN j.name as job, d.name as dataset")
        for w in (writes or []):
            job_uri = f"gwdata:job_{uri_safe(w.get('job'))}"
            ds_uri = f"gwdata:dataset_{uri_safe(w.get('dataset'))}"
            turtle_lines.append(f'{job_uri} gw:writes {ds_uri} .')
            stats["writes"] += 1
        
        represents = neo4j.run_query("MATCH (d:Dataset)-[:REPRESENTS]->(t:Table) RETURN d.name as dataset, t.name as table")
        for rep in (represents or []):
            ds_uri = f"gwdata:dataset_{uri_safe(rep.get('dataset'))}"
            table_uri = f"gwdata:table_{uri_safe(rep.get('table'))}"
            turtle_lines.append(f'{ds_uri} gw:representsTable {table_uri} .')
            stats["represents"] += 1
        
        log(f"[LINEAGE] Jobs: {stats['jobs']}, Datasets: {stats['datasets']}, Reads: {stats['reads']}, Writes: {stats['writes']}")
        
        turtle_content = "\n".join(turtle_lines)
        
        try:
            insert_resp = requests.post(
                f"{base_url}/data?graph={quote(graph_uri, safe='')}",
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
        
        output = "## ★ Graph Synced to RDF ★\n\n"
        output += "### Nodes Synced:\n"
        output += f"- Tables: {stats['tables']}\n"
        output += f"- Columns: {stats['columns']}\n"
        output += f"- Jobs: {stats['jobs']}\n"
        output += f"- Datasets: {stats['datasets']}\n\n"
        output += "### Relationships Synced:\n"
        output += f"- FK references: {stats['fks']}\n"
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
# LangChain Tools (using @tool decorator for create_agent API)
# =============================================================================

@tool
def configure_database_tool(host: str, port: int, database: str, username: str, password: str) -> str:
    """Configure which PostgreSQL database to connect to."""
    return impl_configure_database(host, port, database, username, password)


@tool
def test_database_connection_tool() -> str:
    """Test the PostgreSQL database connection."""
    return impl_test_database_connection()


@tool
def list_database_tables_tool() -> str:
    """List all tables in the database with column counts."""
    return impl_list_database_tables()


@tool
def get_table_schema_tool(table_name: str) -> str:
    """Get schema details for a table (columns, types, PKs)."""
    return impl_get_table_schema(table_name)


@tool
def get_column_stats_tool(table_name: str, column_name: str) -> str:
    """Get statistics for a column (uniqueness, nulls, samples)."""
    return impl_get_column_stats(table_name, column_name)


@tool
def run_fk_discovery_tool(min_match_rate: float = 0.95, min_score: float = 0.5) -> str:
    """Run the full 5-stage FK discovery pipeline on the database."""
    return impl_run_fk_discovery(min_match_rate, min_score)


@tool
def analyze_potential_fk_tool(source_table: str, source_column: str, target_table: str, target_column: str) -> str:
    """Analyze a potential FK relationship and get a score."""
    return impl_analyze_potential_fk(source_table, source_column, target_table, target_column)


@tool
def validate_fk_with_data_tool(source_table: str, source_column: str, target_table: str, target_column: str) -> str:
    """Validate a FK by checking actual data integrity."""
    return impl_validate_fk_with_data(source_table, source_column, target_table, target_column)


@tool
def clear_neo4j_graph_tool() -> str:
    """Clear all nodes and relationships from Neo4j."""
    return impl_clear_neo4j_graph()


@tool
def add_fk_to_graph_tool(source_table: str, source_column: str, target_table: str, target_column: str, score: float = 1.0, cardinality: str = "1:N") -> str:
    """Add a FK relationship to the Neo4j graph."""
    return impl_add_fk_to_graph(source_table, source_column, target_table, target_column, score, cardinality)


@tool
def get_graph_stats_tool() -> str:
    """Get statistics about the Neo4j graph."""
    return impl_get_graph_stats()


@tool
def analyze_graph_centrality_tool() -> str:
    """Find hub and authority tables in the graph."""
    return impl_analyze_graph_centrality()


@tool
def find_table_communities_tool() -> str:
    """Find clusters of related tables."""
    return impl_find_table_communities()


@tool
def predict_missing_fks_tool() -> str:
    """Predict missing FKs based on column naming patterns."""
    return impl_predict_missing_fks()


@tool
def run_cypher_tool(query: str) -> str:
    """Execute a Cypher query on Neo4j."""
    return impl_run_cypher(query)


@tool
def connect_datasets_to_tables_tool() -> str:
    """Connect Dataset nodes to their matching Table nodes."""
    return impl_connect_datasets_to_tables()


@tool
def generate_text_embeddings_tool() -> str:
    """Generate text embeddings for all tables, columns, jobs, and datasets."""
    return impl_generate_text_embeddings()


@tool
def generate_kg_embeddings_tool() -> str:
    """Generate knowledge graph embeddings using Neo4j GDS FastRP."""
    return impl_generate_kg_embeddings()


@tool
def create_vector_indexes_tool() -> str:
    """Create Neo4j vector indexes for fast similarity search."""
    return impl_create_vector_indexes()


@tool
def semantic_search_tables_tool(query: str, top_k: int = 5) -> str:
    """Search for tables using natural language."""
    return impl_semantic_search_tables(query, top_k)


@tool
def semantic_search_columns_tool(query: str, top_k: int = 10) -> str:
    """Search for columns using natural language."""
    return impl_semantic_search_columns(query, top_k)


@tool
def find_similar_tables_tool(table_name: str, top_k: int = 5) -> str:
    """Find tables similar to a given table."""
    return impl_find_similar_tables(table_name, top_k)


@tool
def find_similar_columns_tool(table_name: str, column_name: str, top_k: int = 10) -> str:
    """Find columns similar to a given column."""
    return impl_find_similar_columns(table_name, column_name, top_k)


@tool
def predict_fks_from_embeddings_tool(threshold: float = 0.7, top_k: int = 20) -> str:
    """Predict FK relationships using embedding similarity."""
    return impl_predict_fks_from_embeddings(threshold, top_k)


@tool
def semantic_fk_discovery_tool(source_table: str = None, min_score: float = 0.6) -> str:
    """Discover FKs using semantic similarity."""
    return impl_semantic_fk_discovery(source_table, min_score)


@tool
def show_sample_business_rules_tool() -> str:
    """Show sample business rules YAML format."""
    return impl_show_sample_business_rules()


@tool
def load_business_rules_tool(yaml_content: str) -> str:
    """Load business rules from YAML content."""
    return impl_load_business_rules(yaml_content)


@tool
def load_business_rules_from_file_tool(file_path: str = "business_rules.yaml") -> str:
    """Load business rules from a YAML file."""
    return impl_load_business_rules_from_file(file_path)


@tool
def list_business_rules_tool() -> str:
    """List all loaded business rules."""
    return impl_list_business_rules()


@tool
def execute_business_rule_tool(rule_name: str, capture_lineage: bool = True) -> str:
    """Execute a specific business rule."""
    return impl_execute_business_rule(rule_name, capture_lineage)


@tool
def execute_all_business_rules_tool(capture_lineage: bool = True) -> str:
    """Execute all loaded business rules."""
    return impl_execute_all_business_rules(capture_lineage)


@tool
def get_marquez_lineage_tool(dataset_name: str, depth: int = 3) -> str:
    """Get data lineage for a dataset from Marquez."""
    return impl_get_marquez_lineage(dataset_name, depth)


@tool
def list_marquez_jobs_tool() -> str:
    """List all jobs tracked by Marquez."""
    return impl_list_marquez_jobs()


@tool
def import_lineage_to_graph_tool() -> str:
    """Import Marquez lineage data into Neo4j."""
    return impl_import_lineage_to_graph()


@tool
def analyze_data_flow_tool(table_name: str) -> str:
    """Analyze data flow for a table."""
    return impl_analyze_data_flow(table_name)


@tool
def find_impact_analysis_tool(table_name: str) -> str:
    """Find what tables would be impacted by changes."""
    return impl_find_impact_analysis(table_name)


@tool
def test_rdf_connection_tool() -> str:
    """Test connection to Apache Jena Fuseki."""
    return impl_test_rdf_connection()


@tool
def sync_graph_to_rdf_tool() -> str:
    """Sync Neo4j graph to RDF triple store."""
    return impl_sync_graph_to_rdf()


@tool
def run_sparql_tool(query: str) -> str:
    """Execute a SPARQL query on Fuseki."""
    return impl_run_sparql(query)


@tool
def sparql_list_tables_tool() -> str:
    """List all tables via SPARQL."""
    return impl_sparql_list_tables()


@tool
def sparql_get_foreign_keys_tool(table_name: str = None) -> str:
    """Get foreign keys via SPARQL."""
    return impl_sparql_get_foreign_keys(table_name)


@tool
def sparql_table_lineage_tool(table_name: str) -> str:
    """Get table lineage via SPARQL."""
    return impl_sparql_table_lineage(table_name)


@tool
def sparql_downstream_impact_tool(table_name: str) -> str:
    """Get downstream impact via SPARQL."""
    return impl_sparql_downstream_impact(table_name)


@tool
def sparql_hub_tables_tool(min_connections: int = 3) -> str:
    """Find hub tables via SPARQL."""
    return impl_sparql_hub_tables(min_connections)


@tool
def sparql_orphan_tables_tool() -> str:
    """Find orphan tables via SPARQL."""
    return impl_sparql_orphan_tables()


@tool
def sparql_search_tool(search_term: str) -> str:
    """Search entities by label via SPARQL."""
    return impl_sparql_search(search_term)


@tool
def get_rdf_statistics_tool() -> str:
    """Get RDF store statistics."""
    return impl_get_rdf_statistics()


@tool
def export_rdf_turtle_tool() -> str:
    """Export RDF data as Turtle."""
    return impl_export_rdf_turtle()


@tool
def learn_rules_with_ltn_tool(epochs: int = 100) -> str:
    """Learn rules using Logic Tensor Networks."""
    return impl_learn_rules_with_ltn(epochs)


@tool
def generate_business_rules_from_ltn_tool() -> str:
    """Generate business rules from LTN learned patterns."""
    return impl_generate_business_rules_from_ltn()


@tool
def generate_all_validation_rules_tool() -> str:
    """Generate validation rules from graph structure."""
    return impl_generate_all_validation_rules()


@tool
def export_generated_rules_yaml_tool(file_path: str = "business_rules_generated.yaml") -> str:
    """Export generated rules to YAML."""
    return impl_export_generated_rules_yaml(file_path)


@tool
def export_generated_rules_sql_tool() -> str:
    """Export generated rules as SQL."""
    return impl_export_generated_rules_sql()


@tool
def show_ltn_knowledge_base_tool() -> str:
    """Show the LTN knowledge base."""
    return impl_show_ltn_knowledge_base()


@tool
def check_tool_exists_tool(tool_name: str) -> str:
    """Check if a dynamic tool exists in the registry."""
    return impl_check_tool_exists(tool_name)


@tool
def list_available_tools_tool() -> str:
    """List all available tools - both builtin and dynamic."""
    return impl_list_available_tools()


@tool
def create_dynamic_tool_tool(name: str, description: str, code: str) -> str:
    """Create a new dynamic tool. Code must define a run() function."""
    return impl_create_dynamic_tool(name, description, code)


@tool
def run_dynamic_tool_tool(tool_name: str) -> str:
    """Execute a dynamic tool by name."""
    return impl_run_dynamic_tool(tool_name)


@tool
def get_tool_source_tool(tool_name: str) -> str:
    """Get the source code of a dynamic tool."""
    return impl_get_tool_source(tool_name)


@tool
def update_dynamic_tool_tool(tool_name: str, code: str, description: str = None) -> str:
    """Update a dynamic tool's code."""
    return impl_update_dynamic_tool(tool_name, code, description)


@tool
def delete_dynamic_tool_tool(tool_name: str) -> str:
    """Delete a dynamic tool."""
    return impl_delete_dynamic_tool(tool_name)


# =============================================================================
# All Tools List for create_agent
# =============================================================================

ALL_TOOLS = [
    # Dynamic Tool Management
    check_tool_exists_tool,
    list_available_tools_tool,
    create_dynamic_tool_tool,
    run_dynamic_tool_tool,
    get_tool_source_tool,
    update_dynamic_tool_tool,
    delete_dynamic_tool_tool,
    # Database
    configure_database_tool,
    test_database_connection_tool,
    list_database_tables_tool,
    get_table_schema_tool,
    get_column_stats_tool,
    # FK Discovery
    run_fk_discovery_tool,
    analyze_potential_fk_tool,
    validate_fk_with_data_tool,
    # Neo4j Graph
    clear_neo4j_graph_tool,
    add_fk_to_graph_tool,
    get_graph_stats_tool,
    analyze_graph_centrality_tool,
    find_table_communities_tool,
    predict_missing_fks_tool,
    run_cypher_tool,
    connect_datasets_to_tables_tool,
    # Embeddings
    generate_text_embeddings_tool,
    generate_kg_embeddings_tool,
    create_vector_indexes_tool,
    semantic_search_tables_tool,
    semantic_search_columns_tool,
    find_similar_tables_tool,
    find_similar_columns_tool,
    predict_fks_from_embeddings_tool,
    semantic_fk_discovery_tool,
    # Business Rules
    show_sample_business_rules_tool,
    load_business_rules_tool,
    load_business_rules_from_file_tool,
    list_business_rules_tool,
    execute_business_rule_tool,
    execute_all_business_rules_tool,
    get_marquez_lineage_tool,
    list_marquez_jobs_tool,
    import_lineage_to_graph_tool,
    analyze_data_flow_tool,
    find_impact_analysis_tool,
    # RDF
    test_rdf_connection_tool,
    sync_graph_to_rdf_tool,
    run_sparql_tool,
    sparql_list_tables_tool,
    sparql_get_foreign_keys_tool,
    sparql_table_lineage_tool,
    sparql_downstream_impact_tool,
    sparql_hub_tables_tool,
    sparql_orphan_tables_tool,
    sparql_search_tool,
    get_rdf_statistics_tool,
    export_rdf_turtle_tool,
    # LTN
    learn_rules_with_ltn_tool,
    generate_business_rules_from_ltn_tool,
    generate_all_validation_rules_tool,
    export_generated_rules_yaml_tool,
    export_generated_rules_sql_tool,
    show_ltn_knowledge_base_tool,
]


# =============================================================================
# Middleware for Error Handling
# =============================================================================

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        import traceback
        error_msg = f"Tool error: {type(e).__name__}: {e}"
        debug.error(error_msg, e)
        return ToolMessage(
            content=error_msg,
            tool_call_id=request.tool_call["id"]
        )


# =============================================================================
# Agent Creation using NEW create_agent API
# =============================================================================

def get_agent():
    """Get or create the GraphWeaver agent using create_agent API."""
    if "agent" not in st.session_state:
        debug.agent("Creating GraphWeaver agent with create_agent API...")
        
        api_key = st.session_state.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        
        # Create the model
        model = ChatAnthropic(
            model="claude-opus-4-5-20251101",
            temperature=0.1,
            max_tokens=8192,
            api_key=api_key,
        )
        
        # Create agent with NEW API
        st.session_state.agent = create_agent(
            model=model,
            tools=ALL_TOOLS,
            system_prompt=SYSTEM_PROMPT,
            middleware=[handle_tool_errors],
        )
        
        debug.agent("✓ Agent created successfully")
    
    return st.session_state.agent


# =============================================================================
# Streaming Chat with create_agent API
# =============================================================================

def stream_agent_response(messages: List[Dict], message_placeholder) -> str:
    """Stream response from agent using create_agent API with streaming."""
    
    agent = get_agent()
    if agent is None:
        return "ERROR: Agent not initialized. Check API key."
    
    full_response = ""
    tool_calls_seen = set()
    
    debug.section("AGENT STREAMING")
    debug.agent(f"Input messages: {len(messages)}")
    
    try:
        # Convert messages to LangChain format
        lc_messages = []
        for msg in messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
        
        # Stream response using NEW API
        for chunk in agent.stream(
            {"messages": lc_messages},
            stream_mode="values",
            config={"recursion_limit": 100}
        ):
            if "messages" in chunk and chunk["messages"]:
                latest_message = chunk["messages"][-1]
                
                # Handle text content
                content = getattr(latest_message, 'content', '')
                if isinstance(content, str) and content:
                    new_content = content[len(full_response):]
                    if new_content:
                        full_response = content
                        message_placeholder.markdown(full_response + "▌")
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get('type') == 'text':
                            text = block.get('text', '')
                            new_content = text[len(full_response):]
                            if new_content:
                                full_response = text
                                message_placeholder.markdown(full_response + "▌")
                
                # Handle tool calls
                tool_calls = getattr(latest_message, 'tool_calls', None)
                if tool_calls:
                    for tc in tool_calls:
                        tc_id = tc.get('id', '')
                        if tc_id and tc_id not in tool_calls_seen:
                            tool_calls_seen.add(tc_id)
                            tool_name = tc.get('name', 'unknown')
                            debug.tool(f"Tool call: {tool_name}")
                            full_response += f"\n\n🔧 **{tool_name}**\n"
                            message_placeholder.markdown(full_response + "▌")
        
        debug.agent("Agent finished streaming")
        message_placeholder.markdown(full_response)
        return full_response
        
    except Exception as e:
        debug.error(f"Streaming error: {e}", e)
        import traceback
        error_msg = f"Error: {type(e).__name__}: {e}\n\n```\n{traceback.format_exc()}\n```"
        message_placeholder.markdown(error_msg)
        return error_msg


# =============================================================================
# Streamlit UI - WITH CONVERSATION MEMORY
# =============================================================================

def main():
    st.set_page_config(
        page_title="GraphWeaver Agent",
        page_icon="🕸️",
        layout="wide",
    )
    
    st.title("🕸️ GraphWeaver Agent")
    st.caption("Chat with Claude to discover FK relationships, build knowledge graphs, and analyze data lineage")
    st.caption("**With Conversation Memory** 💾")
    
    with st.sidebar:
        # =============================================
        # NEW: CONVERSATION HISTORY SECTION
        # =============================================
        render_conversation_sidebar()
        render_conversation_actions()
        
        st.divider()
        
        # =============================================
        # CONFIGURATION SECTION
        # =============================================
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
            if "agent" in st.session_state:
                del st.session_state.agent
        
        st.divider()
        
        st.subheader("🗄️ PostgreSQL")
        pg_host = st.text_input("Host", value=st.session_state.get("pg_host", os.environ.get("POSTGRES_HOST", "localhost")))
        pg_port = st.number_input("Port", value=int(st.session_state.get("pg_port", os.environ.get("POSTGRES_PORT", "5432"))))
        pg_database = st.text_input("Database", value=st.session_state.get("pg_database", os.environ.get("POSTGRES_DB", "orders")))
        pg_username = st.text_input("Username", value=st.session_state.get("pg_username", os.environ.get("POSTGRES_USER", "saphenia")))
        pg_password = st.text_input("Password", type="password", value=st.session_state.get("pg_password", os.environ.get("POSTGRES_PASSWORD", "secret")))
        
        st.session_state.pg_host = pg_host
        st.session_state.pg_port = pg_port
        st.session_state.pg_database = pg_database
        st.session_state.pg_username = pg_username
        st.session_state.pg_password = pg_password
        
        st.divider()
        
        st.subheader("🔵 Neo4j")
        neo4j_uri = st.text_input("URI", value=st.session_state.get("neo4j_uri", os.environ.get("NEO4J_URI", "bolt://localhost:7687")))
        neo4j_user = st.text_input("Neo4j User", value=st.session_state.get("neo4j_user", os.environ.get("NEO4J_USER", "neo4j")))
        neo4j_password = st.text_input("Neo4j Password", type="password", value=st.session_state.get("neo4j_password", os.environ.get("NEO4J_PASSWORD", "password")))
        
        st.session_state.neo4j_uri = neo4j_uri
        st.session_state.neo4j_user = neo4j_user
        st.session_state.neo4j_password = neo4j_password
        
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
    
    # Check API key
    if not st.session_state.get("anthropic_api_key") and not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("⚠️ Please enter your Anthropic API key in the sidebar to start chatting.")
        st.stop()
    
    # Display existing messages
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle quick actions or chat input
    if "quick_action" in st.session_state and st.session_state.quick_action:
        prompt = st.session_state.quick_action
        st.session_state.quick_action = None
    else:
        prompt = st.chat_input("Ask me about your database...")
    
    if prompt:
        debug.section("NEW USER MESSAGE")
        debug.agent(f"User: {prompt}")
        
        # Add user message using conversation memory
        add_message("user", prompt)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            try:
                message_placeholder = st.empty()
                
                # Use get_current_streaming_messages() for agent context
                response = stream_agent_response(
                    get_current_streaming_messages(),
                    message_placeholder
                )
                
                # Add assistant response using conversation memory
                add_message("assistant", response)
                
            except Exception as e:
                debug.error(f"Fatal error: {e}", e)
                import traceback
                error_msg = f"Error: {type(e).__name__}: {e}\n\n```\n{traceback.format_exc()}\n```"
                st.error(error_msg)
                add_message("assistant", f"Error: {e}")


if __name__ == "__main__":
    main()
