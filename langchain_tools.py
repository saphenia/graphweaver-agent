#!/usr/bin/env python3
# =============================================================================
# FILE: langchain_tools.py
# DESCRIPTION: LangChain tool definitions wrapping the implementations
# =============================================================================
"""
LangChain Tools for GraphWeaver Agent.

This module contains:
- All @tool decorated LangChain tools
- ALL_TOOLS list for agent creation
- Tool execution helpers for Anthropic SDK
"""
from typing import Dict, List
import inspect

from langchain.tools import tool

# Import all implementations
from tool_implementations import (
    # Database tools
    impl_configure_database,
    impl_test_database_connection,
    impl_list_database_tables,
    impl_get_table_schema,
    impl_get_column_stats,
    # FK Discovery tools
    impl_run_fk_discovery,
    impl_analyze_potential_fk,
    impl_validate_fk_with_data,
    # Neo4j Graph tools
    impl_clear_neo4j_graph,
    impl_add_fk_to_graph,
    impl_get_graph_stats,
    impl_analyze_graph_centrality,
    impl_find_table_communities,
    impl_predict_missing_fks,
    impl_run_cypher,
    impl_connect_datasets_to_tables,
    # Embedding tools
    impl_generate_text_embeddings,
    impl_generate_kg_embeddings,
    impl_create_vector_indexes,
    impl_semantic_search_tables,
    impl_semantic_search_columns,
    impl_find_similar_tables,
    impl_find_similar_columns,
    impl_predict_fks_from_embeddings,
    impl_semantic_fk_discovery,
    # Business Rules tools
    impl_show_sample_business_rules,
    impl_load_business_rules,
    impl_load_business_rules_from_file,
    impl_list_business_rules,
    impl_execute_business_rule,
    impl_execute_all_business_rules,
    impl_get_marquez_lineage,
    impl_list_marquez_jobs,
    impl_import_lineage_to_graph,
    impl_analyze_data_flow,
    impl_find_impact_analysis,
    # RDF tools
    impl_test_rdf_connection,
    impl_sync_graph_to_rdf,
    impl_run_sparql,
    impl_sparql_list_tables,
    impl_sparql_get_foreign_keys,
    impl_sparql_table_lineage,
    impl_sparql_downstream_impact,
    impl_sparql_hub_tables,
    impl_sparql_orphan_tables,
    impl_sparql_search,
    impl_get_rdf_statistics,
    impl_export_rdf_turtle,
    # LTN tools
    impl_learn_rules_with_ltn,
    impl_generate_business_rules_from_ltn,
    impl_generate_all_validation_rules,
    impl_export_generated_rules_yaml,
    impl_export_generated_rules_sql,
    impl_show_ltn_knowledge_base,
    # Dynamic Tool Management tools
    impl_check_tool_exists,
    impl_list_available_tools,
    impl_create_dynamic_tool,
    impl_run_dynamic_tool,
    impl_get_tool_source,
    impl_update_dynamic_tool,
    impl_delete_dynamic_tool,
)


# =============================================================================
# LangChain Tools (using @tool decorator for create_agent API)
# =============================================================================

# -----------------------------------------------------------------------------
# Database Tools
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# FK Discovery Tools
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Neo4j Graph Tools
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Embedding Tools
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Business Rules Tools
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# RDF Tools
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# LTN Tools
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Dynamic Tool Management Tools
# -----------------------------------------------------------------------------

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
# Tool Definitions for Anthropic SDK (for true streaming)
# =============================================================================

def get_sdk_tools() -> List[Dict]:
    """Convert ALL_TOOLS to Anthropic SDK format for true streaming."""
    sdk_tools = []
    for t in ALL_TOOLS:
        # Extract info from LangChain tool
        name = t.name
        description = t.description or ""
        
        # Build input schema from function signature
        sig = inspect.signature(t.func)
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls'):
                continue
            
            # Determine type
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
            
            properties[param_name] = {"type": param_type, "description": f"Parameter: {param_name}"}
            
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        sdk_tools.append({
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        })
    
    # DEBUG: Print tool names to verify embeddings tools are included
    tool_names = [t["name"] for t in sdk_tools]
    print(f"[DEBUG] SDK tools count: {len(sdk_tools)}")
    print(f"[DEBUG] Embedding tools present: generate_text_embeddings_tool={'generate_text_embeddings_tool' in tool_names}, generate_kg_embeddings_tool={'generate_kg_embeddings_tool' in tool_names}")
    
    return sdk_tools


def execute_tool_by_name(tool_name: str, tool_input: Dict) -> str:
    """Execute a tool by name with given input."""
    # Find the tool in ALL_TOOLS
    for t in ALL_TOOLS:
        if t.name == tool_name:
            try:
                return t.func(**tool_input)
            except Exception as e:
                import traceback
                return f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"
    return f"ERROR: Unknown tool: {tool_name}"
