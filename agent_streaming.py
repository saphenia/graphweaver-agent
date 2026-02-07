#!/usr/bin/env python3
"""
GraphWeaver Agent Streaming - Anthropic SDK streaming mode definitions.

This module contains:
- STREAMING_TOOLS: JSON schema tool definitions for Anthropic SDK
- STREAMING_TOOL_FUNCTIONS: Map of tool names to callable functions
- print_token / stream_response / run_interactive_streaming functions
"""
import sys

from agent_tools import (
    SYSTEM_PROMPT,
    # Dynamic tool management
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
    # Embeddings
    generate_text_embeddings, generate_kg_embeddings, create_vector_indexes,
    semantic_search_tables, semantic_search_columns, find_similar_tables,
    find_similar_columns, predict_fks_from_embeddings, semantic_fk_discovery,
    # Business Rules
    show_sample_business_rules, load_business_rules, load_business_rules_from_file,
    list_business_rules, execute_business_rule, execute_all_business_rules,
    get_marquez_lineage, list_marquez_jobs, import_lineage_to_graph,
    analyze_data_flow, find_impact_analysis,
    # RDF
    test_rdf_connection, sync_graph_to_rdf, debug_rdf_sync, run_sparql,
    sparql_list_tables, sparql_get_foreign_keys, sparql_table_lineage,
    sparql_downstream_impact, sparql_hub_tables, sparql_orphan_tables,
    sparql_search, get_rdf_statistics, export_rdf_turtle,
    # LTN
    learn_rules_with_ltn, generate_business_rules_from_ltn,
    generate_all_validation_rules, export_generated_rules_yaml,
    export_generated_rules_sql, show_ltn_knowledge_base,
)


# =============================================================================
# Tool definitions for streaming mode (JSON schema format for Anthropic SDK)
# =============================================================================

STREAMING_TOOLS = [
    # Dynamic tool management
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

    # Database
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

    # FK Discovery
    {"name": "run_fk_discovery", "description": "Run the full 5-stage FK discovery pipeline on the database",
     "input_schema": {"type": "object", "properties": {"min_match_rate": {"type": "number", "default": 0.95}, "min_score": {"type": "number", "default": 0.5}}}},
    {"name": "discover_and_sync", "description": "One-stop shop: Discover FKs, build Neo4j graph, and sync to RDF",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "analyze_potential_fk", "description": "Analyze a potential FK relationship and get a score",
     "input_schema": {"type": "object", "properties": {"source_table": {"type": "string"}, "source_column": {"type": "string"}, "target_table": {"type": "string"}, "target_column": {"type": "string"}}, "required": ["source_table", "source_column", "target_table", "target_column"]}},
    {"name": "validate_fk_with_data", "description": "Validate a FK by checking actual data integrity",
     "input_schema": {"type": "object", "properties": {"source_table": {"type": "string"}, "source_column": {"type": "string"}, "target_table": {"type": "string"}, "target_column": {"type": "string"}}, "required": ["source_table", "source_column", "target_table", "target_column"]}},

    # Graph
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

    # Embeddings
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
    {"name": "predict_fks_from_embeddings", "description": "Predict potential FK relationships using KG embeddings",
     "input_schema": {"type": "object", "properties": {"threshold": {"type": "number", "default": 0.7}, "top_k": {"type": "integer", "default": 20}}}},
    {"name": "semantic_fk_discovery", "description": "Discover FK relationships using semantic similarity",
     "input_schema": {"type": "object", "properties": {"source_table": {"type": "string"}, "min_score": {"type": "number", "default": 0.6}}}},

    # Business Rules
    {"name": "show_sample_business_rules", "description": "Show sample business rules YAML format",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "load_business_rules", "description": "Load business rules from YAML content",
     "input_schema": {"type": "object", "properties": {"yaml_content": {"type": "string"}}, "required": ["yaml_content"]}},
    {"name": "load_business_rules_from_file", "description": "Load business rules from a YAML file",
     "input_schema": {"type": "object", "properties": {"file_path": {"type": "string", "default": "business_rules.yaml"}}}},
    {"name": "list_business_rules", "description": "List all loaded business rules",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "execute_business_rule", "description": "Execute a single business rule",
     "input_schema": {"type": "object", "properties": {"rule_name": {"type": "string"}, "capture_lineage": {"type": "boolean", "default": True}}, "required": ["rule_name"]}},
    {"name": "execute_all_business_rules", "description": "Execute all loaded business rules",
     "input_schema": {"type": "object", "properties": {"capture_lineage": {"type": "boolean", "default": True}}}},
    {"name": "get_marquez_lineage", "description": "Get lineage graph for a dataset from Marquez",
     "input_schema": {"type": "object", "properties": {"dataset_name": {"type": "string"}, "depth": {"type": "integer", "default": 3}}, "required": ["dataset_name"]}},
    {"name": "list_marquez_jobs", "description": "List all jobs tracked in Marquez",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "import_lineage_to_graph", "description": "Import lineage from Marquez into Neo4j",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "analyze_data_flow", "description": "Analyze complete data flow for a table",
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}}, "required": ["table_name"]}},
    {"name": "find_impact_analysis", "description": "Find all downstream impacts if a table changes",
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}}, "required": ["table_name"]}},

    # RDF
    {"name": "test_rdf_connection", "description": "Test connection to Apache Jena Fuseki",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "sync_graph_to_rdf", "description": "Sync Neo4j graph to RDF/Fuseki",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "debug_rdf_sync", "description": "Debug RDF sync step by step",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "run_sparql", "description": "Execute a SPARQL query on Fuseki",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "sparql_list_tables", "description": "List all tables in the RDF store",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "sparql_get_foreign_keys", "description": "Get foreign keys from RDF store",
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}}}},
    {"name": "sparql_table_lineage", "description": "Get lineage for a table from RDF store",
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}}, "required": ["table_name"]}},
    {"name": "sparql_downstream_impact", "description": "Find downstream impact via RDF/SPARQL",
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}}, "required": ["table_name"]}},
    {"name": "sparql_hub_tables", "description": "Find hub tables via RDF/SPARQL",
     "input_schema": {"type": "object", "properties": {"min_connections": {"type": "integer", "default": 2}}}},
    {"name": "sparql_orphan_tables", "description": "Find tables with no FK relationships via RDF",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "sparql_search", "description": "Search the RDF graph by label",
     "input_schema": {"type": "object", "properties": {"search_term": {"type": "string"}}, "required": ["search_term"]}},
    {"name": "get_rdf_statistics", "description": "Get statistics from the RDF store",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "export_rdf_turtle", "description": "Export the graph ontology in Turtle format",
     "input_schema": {"type": "object", "properties": {}}},

    # LTN
    {"name": "learn_rules_with_ltn", "description": "Learn logical rules from the knowledge graph using LTN",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "generate_business_rules_from_ltn", "description": "Generate executable rules from learned LTN patterns",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "generate_all_validation_rules", "description": "Generate all validation rules from the graph",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "export_generated_rules_yaml", "description": "Export generated rules as YAML file",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "export_generated_rules_sql", "description": "Export generated rules as SQL script",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "show_ltn_knowledge_base", "description": "Show the LTN knowledge base with axioms",
     "input_schema": {"type": "object", "properties": {}}},
]


# =============================================================================
# Map tool names to functions for streaming mode
# =============================================================================

STREAMING_TOOL_FUNCTIONS = {
    # Dynamic tool management
    "check_tool_exists": lambda **kw: check_tool_exists.func(**kw),
    "list_available_tools": lambda **kw: list_available_tools.func(**kw),
    "create_dynamic_tool": lambda **kw: create_dynamic_tool.func(**kw),
    "run_dynamic_tool": lambda **kw: run_dynamic_tool.func(**kw),
    "get_tool_source": lambda **kw: get_tool_source.func(**kw),
    "update_dynamic_tool": lambda **kw: update_dynamic_tool.func(**kw),
    "delete_dynamic_tool": lambda **kw: delete_dynamic_tool.func(**kw),
    # Database
    "configure_database": lambda **kw: configure_database.func(**kw),
    "test_database_connection": lambda **kw: test_database_connection.func(**kw),
    "list_database_tables": lambda **kw: list_database_tables.func(**kw),
    "get_table_schema": lambda **kw: get_table_schema.func(**kw),
    "get_column_stats": lambda **kw: get_column_stats.func(**kw),
    # FK Discovery
    "run_fk_discovery": lambda **kw: run_fk_discovery.func(**kw),
    "discover_and_sync": lambda **kw: discover_and_sync.func(**kw),
    "analyze_potential_fk": lambda **kw: analyze_potential_fk.func(**kw),
    "validate_fk_with_data": lambda **kw: validate_fk_with_data.func(**kw),
    # Graph
    "clear_neo4j_graph": lambda **kw: clear_neo4j_graph.func(**kw),
    "add_fk_to_graph": lambda **kw: add_fk_to_graph.func(**kw),
    "get_graph_stats": lambda **kw: get_graph_stats.func(**kw),
    "analyze_graph_centrality": lambda **kw: analyze_graph_centrality.func(**kw),
    "find_table_communities": lambda **kw: find_table_communities.func(**kw),
    "predict_missing_fks": lambda **kw: predict_missing_fks.func(**kw),
    "run_cypher": lambda **kw: run_cypher.func(**kw),
    "connect_datasets_to_tables": lambda **kw: connect_datasets_to_tables.func(**kw),
    # Embeddings
    "generate_text_embeddings": lambda **kw: generate_text_embeddings.func(**kw),
    "generate_kg_embeddings": lambda **kw: generate_kg_embeddings.func(**kw),
    "create_vector_indexes": lambda **kw: create_vector_indexes.func(**kw),
    "semantic_search_tables": lambda **kw: semantic_search_tables.func(**kw),
    "semantic_search_columns": lambda **kw: semantic_search_columns.func(**kw),
    "find_similar_tables": lambda **kw: find_similar_tables.func(**kw),
    "find_similar_columns": lambda **kw: find_similar_columns.func(**kw),
    "predict_fks_from_embeddings": lambda **kw: predict_fks_from_embeddings.func(**kw),
    "semantic_fk_discovery": lambda **kw: semantic_fk_discovery.func(**kw),
    # Business Rules
    "show_sample_business_rules": lambda **kw: show_sample_business_rules.func(**kw),
    "load_business_rules": lambda **kw: load_business_rules.func(**kw),
    "load_business_rules_from_file": lambda **kw: load_business_rules_from_file.func(**kw),
    "list_business_rules": lambda **kw: list_business_rules.func(**kw),
    "execute_business_rule": lambda **kw: execute_business_rule.func(**kw),
    "execute_all_business_rules": lambda **kw: execute_all_business_rules.func(**kw),
    "get_marquez_lineage": lambda **kw: get_marquez_lineage.func(**kw),
    "list_marquez_jobs": lambda **kw: list_marquez_jobs.func(**kw),
    "import_lineage_to_graph": lambda **kw: import_lineage_to_graph.func(**kw),
    "analyze_data_flow": lambda **kw: analyze_data_flow.func(**kw),
    "find_impact_analysis": lambda **kw: find_impact_analysis.func(**kw),
    # RDF
    "test_rdf_connection": lambda **kw: test_rdf_connection.func(**kw),
    "sync_graph_to_rdf": lambda **kw: sync_graph_to_rdf.func(**kw),
    "debug_rdf_sync": lambda **kw: debug_rdf_sync.func(**kw),
    "run_sparql": lambda **kw: run_sparql.func(**kw),
    "sparql_list_tables": lambda **kw: sparql_list_tables.func(**kw),
    "sparql_get_foreign_keys": lambda **kw: sparql_get_foreign_keys.func(**kw),
    "sparql_table_lineage": lambda **kw: sparql_table_lineage.func(**kw),
    "sparql_downstream_impact": lambda **kw: sparql_downstream_impact.func(**kw),
    "sparql_hub_tables": lambda **kw: sparql_hub_tables.func(**kw),
    "sparql_orphan_tables": lambda **kw: sparql_orphan_tables.func(**kw),
    "sparql_search": lambda **kw: sparql_search.func(**kw),
    "get_rdf_statistics": lambda **kw: get_rdf_statistics.func(**kw),
    "export_rdf_turtle": lambda **kw: export_rdf_turtle.func(**kw),
    # LTN
    "learn_rules_with_ltn": lambda **kw: learn_rules_with_ltn.func(**kw),
    "generate_business_rules_from_ltn": lambda **kw: generate_business_rules_from_ltn.func(**kw),
    "generate_all_validation_rules": lambda **kw: generate_all_validation_rules.func(**kw),
    "export_generated_rules_yaml": lambda **kw: export_generated_rules_yaml.func(**kw),
    "export_generated_rules_sql": lambda **kw: export_generated_rules_sql.func(**kw),
    "show_ltn_knowledge_base": lambda **kw: show_ltn_knowledge_base.func(**kw),
}


# =============================================================================
# Streaming Helpers
# =============================================================================

def print_token(text: str) -> None:
    """Print a single token immediately (no buffering)."""
    sys.stdout.write(text)
    sys.stdout.flush()


def stream_response(client, messages: list) -> tuple:
    """Stream response from Claude, printing each token as it arrives."""
    tool_inputs = {}
    current_block_id = None

    with client.messages.stream(
        model="claude-opus-4-5-20251101",
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=messages,
        tools=STREAMING_TOOLS,
    ) as stream:
        for event in stream:
            if event.type == "content_block_start":
                if event.content_block.type == "tool_use":
                    current_block_id = event.content_block.id
                    tool_inputs[current_block_id] = {
                        "name": event.content_block.name,
                        "input": ""
                    }
                    print_token(f"\n\n🔧 **{event.content_block.name}**\n")

            elif event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    print_token(event.delta.text)
                elif event.delta.type == "input_json_delta":
                    if current_block_id:
                        tool_inputs[current_block_id]["input"] += event.delta.partial_json

            elif event.type == "content_block_stop":
                current_block_id = None

        return stream.get_final_message(), tool_inputs


def run_interactive_streaming():
    """Run agent in interactive mode with token-by-token streaming using Anthropic SDK."""
    import anthropic

    print("\n" + "=" * 60)
    print("  🕸️  GraphWeaver Agent - Anthropic SDK Streaming Mode")
    print("=" * 60)
    print("\nI can help you discover FK relationships in your database.")
    print("Try saying:")
    print("  • 'connect and show me the tables'")
    print("  • 'find all foreign keys'")
    print("  • 'generate embeddings and search for customer columns'")
    print("  • 'learn rules with LTN and export as YAML'")
    print("  • 'create a tool that generates an ERD diagram'")
    print("\nType 'quit' to exit.\n")

    client = anthropic.Anthropic()
    messages = []

    while True:
        try:
            user_input = input("\033[92mYou:\033[0m ").strip()
            if not user_input:
                continue
            if user_input.lower() in ('quit', 'exit', 'q'):
                break

            messages.append({"role": "user", "content": user_input})

            print("\n\033[96m🤖 Agent:\033[0m ", end="")
            sys.stdout.flush()

            # Agentic loop
            while True:
                response, tool_inputs = stream_response(client, messages)
                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason != "tool_use":
                    break

                # Execute tool calls
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print_token("\n⏳ Executing...")

                        fn = STREAMING_TOOL_FUNCTIONS.get(block.name)
                        if fn:
                            try:
                                result = fn(**block.input)
                            except Exception as e:
                                result = f"Error: {type(e).__name__}: {e}"
                        else:
                            result = f"Unknown tool: {block.name}"

                        print_token(f"\n```\n{result}\n```\n")

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })

                messages.append({"role": "user", "content": tool_results})

            print("\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted.")
            break
        except Exception as e:
            print(f"\n\033[31mError: {e}\033[0m")
            import traceback
            traceback.print_exc()

    print("\n👋 Goodbye!\n")