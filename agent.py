#!/usr/bin/env python3
"""
GraphWeaver Agent - True Token-by-Token Streaming

Uses the Anthropic SDK directly for real-time streaming output.
Each token is printed immediately as it arrives from the API.
"""
import os
import sys
import json
import anthropic

# Force unbuffered output for true token-by-token streaming
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=False, write_through=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mcp_servers"))

# =============================================================================
# Lazy-loaded Connectors
# =============================================================================

from graphweaver_agent import DataSourceConfig, Neo4jConfig, PostgreSQLConnector, Neo4jClient, GraphBuilder, GraphAnalyzer
from graphweaver_agent.discovery.pipeline import run_discovery
from graphweaver_agent.business_rules import (
    BusinessRulesExecutor, BusinessRulesConfig, BusinessRule, 
    MarquezClient, import_lineage_to_neo4j, generate_sample_rules
)
from graphweaver_agent.rdf import FusekiClient, sync_neo4j_to_rdf, GraphWeaverOntology, SPARQLQueryBuilder

try:
    from graphweaver_agent.ltn import LTNRuleLearner, BusinessRuleGenerator, LTNKnowledgeBase, RuleLearningConfig
    LTN_AVAILABLE = True
except ImportError:
    LTN_AVAILABLE = False

try:
    from graphweaver_agent.embeddings import TextEmbedder, KGEmbedder, VectorIndexes
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# Global lazy singletons
_pg = _pg_config = _neo4j = _text_embedder = _kg_embedder = None
_fuseki = _sparql = _rules_config = _marquez = _registry = None


def get_pg_config():
    global _pg_config
    if _pg_config is None:
        _pg_config = DataSourceConfig(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            database=os.environ.get("POSTGRES_DB", "orders"),
            username=os.environ.get("POSTGRES_USER", "saphenia"),
            password=os.environ.get("POSTGRES_PASSWORD", "secret")
        )
    return _pg_config


def get_pg():
    global _pg
    if _pg is None:
        _pg = PostgreSQLConnector(get_pg_config())
    return _pg


def get_neo4j():
    global _neo4j
    if _neo4j is None:
        _neo4j = Neo4jClient(Neo4jConfig(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "password")
        ))
    return _neo4j


def get_registry():
    global _registry
    if _registry is None:
        from graphweaver_agent.dynamic_tools.tool_registry import ToolRegistry
        _registry = ToolRegistry(
            os.environ.get("DYNAMIC_TOOLS_DIR", 
                          os.path.join(os.path.dirname(__file__), "dynamic_tools"))
        )
    return _registry


def get_marquez():
    global _marquez
    if _marquez is None:
        _marquez = MarquezClient(os.environ.get("MARQUEZ_URL", "http://localhost:5000"))
    return _marquez


def get_fuseki():
    global _fuseki
    if _fuseki is None:
        _fuseki = FusekiClient(
            os.environ.get("FUSEKI_URL", "http://localhost:3030"),
            os.environ.get("FUSEKI_DATASET", "graphweaver")
        )
    return _fuseki


# =============================================================================
# Tool Implementations
# =============================================================================

# --- Dynamic Tool Management ---
def check_tool_exists(tool_name: str) -> str:
    """Check if a dynamic tool exists."""
    return "✓ EXISTS" if get_registry().tool_exists(tool_name) else "✗ NOT FOUND"


def list_available_tools() -> str:
    """List all available tools."""
    dynamic = get_registry().list_tools()
    return (
        "**Builtin**: DB, FK Discovery, Graph, Embeddings, Business Rules, RDF\n"
        f"**Dynamic**: {', '.join(t['name'] for t in dynamic) if dynamic else 'None'}"
    )


def create_dynamic_tool(name: str, description: str, code: str) -> str:
    """Create a new dynamic tool. Code must define a run() function."""
    r = get_registry()
    if r.tool_exists(name):
        return f"ERROR: Tool '{name}' already exists"
    if "def run(" not in code:
        return "ERROR: Code must define a run() function"
    try:
        compile(code, name, "exec")
        path = r.create_tool(name, description, code)
        return f"✓ Created tool '{name}' at {path}"
    except Exception as e:
        return f"ERROR: {e}"


def run_dynamic_tool(tool_name: str, **kwargs) -> str:
    """Execute a dynamic tool."""
    r = get_registry()
    if not r.tool_exists(tool_name):
        return f"ERROR: Tool '{tool_name}' not found"
    try:
        return str(r.execute_tool(tool_name, **kwargs))
    except Exception as e:
        return f"ERROR: {e}"


def get_tool_source(tool_name: str) -> str:
    """Get the source code of a dynamic tool."""
    r = get_registry()
    return r.get_tool_source(tool_name) if r.tool_exists(tool_name) else "Not found"


def update_dynamic_tool(tool_name: str, code: str, description: str = None) -> str:
    """Update a dynamic tool's code."""
    r = get_registry()
    if not r.tool_exists(tool_name):
        return "Not found"
    try:
        compile(code, tool_name, "exec")
        r.update_tool(tool_name, code, description)
        return "✓ Updated"
    except Exception as e:
        return f"ERROR: {e}"


def delete_dynamic_tool(tool_name: str) -> str:
    """Delete a dynamic tool."""
    r = get_registry()
    if not r.tool_exists(tool_name):
        return "Not found"
    r.delete_tool(tool_name)
    return "✓ Deleted"


# --- Database Tools ---
def test_database_connection() -> str:
    """Test the PostgreSQL database connection."""
    try:
        r = get_pg().test_connection()
        return f"✓ Connected to {r['database']}" if r["success"] else f"✗ {r['error']}"
    except Exception as e:
        return f"✗ Connection failed: {e}"


def list_database_tables() -> str:
    """List all tables in the database."""
    tables = get_pg().get_tables_with_info()
    return "\n".join(f"• {t['table_name']}: {t['column_count']} cols, ~{t.get('row_count', '?')} rows" 
                     for t in tables)


def get_table_schema(table_name: str) -> str:
    """Get the schema (columns, types, PKs) for a table."""
    m = get_pg().get_table_metadata(table_name)
    lines = [f"**{table_name}** (~{m.row_count} rows)"]
    for c in m.columns:
        pk = " [PK]" if c.is_primary_key else ""
        nullable = "" if c.is_nullable else " NOT NULL"
        lines.append(f"  • {c.column_name}: {c.data_type}{pk}{nullable}")
    return "\n".join(lines)


def get_column_stats(table_name: str, column_name: str) -> str:
    """Get statistics for a specific column."""
    try:
        stats = get_pg().get_column_stats(table_name, column_name)
        return (
            f"**{table_name}.{column_name}**\n"
            f"  Unique: {stats.get('unique_count', 'N/A')} ({stats.get('uniqueness_pct', 0):.1f}%)\n"
            f"  Nulls: {stats.get('null_count', 0)} ({stats.get('null_pct', 0):.1f}%)\n"
            f"  Sample: {stats.get('sample_values', [])[:5]}"
        )
    except Exception as e:
        return f"ERROR: {e}"


# --- FK Discovery Tools ---
def run_fk_discovery(min_match_rate: float = 0.95, min_score: float = 0.5) -> str:
    """Run the full FK discovery pipeline."""
    cfg = get_pg_config()
    r = run_discovery(
        host=cfg.host, port=cfg.port, database=cfg.database,
        username=cfg.username, password=cfg.password,
        schema=cfg.schema_name,
        min_match_rate=min_match_rate, min_score=min_score
    )
    summary = r.get('summary', {})
    return (
        f"**FK Discovery Complete**\n"
        f"  Candidates evaluated: {summary.get('candidates_evaluated', 0)}\n"
        f"  FKs discovered: {summary.get('final_fks_discovered', 0)}\n"
        f"  Added to graph: {summary.get('added_to_graph', 0)}"
    )


def analyze_potential_fk(source_table: str, source_column: str, 
                         target_table: str, target_column: str) -> str:
    """Analyze a potential FK relationship and return a score."""
    try:
        from graphweaver_agent.discovery.pipeline import analyze_fk_candidate
        score = analyze_fk_candidate(
            get_pg(), source_table, source_column, target_table, target_column
        )
        recommendation = "LIKELY FK" if score > 0.7 else "POSSIBLE" if score > 0.4 else "UNLIKELY"
        return f"Score: {score:.3f} - {recommendation}"
    except Exception as e:
        return f"ERROR: {e}"


def validate_fk_with_data(source_table: str, source_column: str,
                          target_table: str, target_column: str) -> str:
    """Validate FK by checking actual data referential integrity."""
    try:
        from graphweaver_agent.discovery.pipeline import validate_fk
        result = validate_fk(get_pg(), source_table, source_column, target_table, target_column)
        if result['valid']:
            return f"✓ CONFIRMED FK - Match rate: {result['match_rate']*100:.1f}%"
        else:
            return f"✗ INVALID - {result.get('reason', 'Orphaned values found')}"
    except Exception as e:
        return f"ERROR: {e}"


# --- Graph Tools ---
def clear_neo4j_graph() -> str:
    """Clear all nodes and relationships from Neo4j."""
    get_neo4j().run_query("MATCH (n) DETACH DELETE n")
    return "✓ Graph cleared"


def add_fk_to_graph(source_table: str, source_column: str,
                    target_table: str, target_column: str,
                    score: float = 1.0, cardinality: str = "N:1") -> str:
    """Add a FK relationship to the Neo4j graph."""
    try:
        builder = GraphBuilder(get_neo4j())
        builder.add_fk_relationship(source_table, source_column, 
                                   target_table, target_column, score, cardinality)
        return f"✓ Added: {source_table}.{source_column} → {target_table}.{target_column}"
    except Exception as e:
        return f"ERROR: {e}"


def get_graph_stats() -> str:
    """Get statistics about the current graph."""
    try:
        stats = GraphAnalyzer(get_neo4j()).get_statistics()
        return f"Tables: {stats['tables']}, Columns: {stats['columns']}, FKs: {stats['fks']}"
    except Exception as e:
        return f"ERROR: {e}"


def analyze_graph_centrality() -> str:
    """Find hub and authority tables in the graph."""
    try:
        analyzer = GraphAnalyzer(get_neo4j())
        centrality = analyzer.get_centrality()
        
        hubs = sorted(centrality.items(), key=lambda x: x[1].get('out_degree', 0), reverse=True)[:5]
        auths = sorted(centrality.items(), key=lambda x: x[1].get('in_degree', 0), reverse=True)[:5]
        
        lines = ["**Hub Tables** (most outgoing FKs):"]
        lines.extend(f"  • {t}: {s['out_degree']} FKs" for t, s in hubs if s.get('out_degree', 0) > 0)
        lines.append("\n**Authority Tables** (most incoming FKs):")
        lines.extend(f"  • {t}: {s['in_degree']} refs" for t, s in auths if s.get('in_degree', 0) > 0)
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: {e}"


def run_cypher(query: str) -> str:
    """Execute a Cypher query on Neo4j."""
    try:
        results = get_neo4j().run_query(query)
        if not results:
            return "No results"
        return "\n".join(str(dict(r)) for r in results[:20])
    except Exception as e:
        return f"ERROR: {e}"


# --- Business Rules Tools ---
def show_sample_business_rules() -> str:
    """Show sample business rules YAML format."""
    return generate_sample_rules()


def load_business_rules(yaml_content: str) -> str:
    """Load business rules from YAML content."""
    global _rules_config
    try:
        import yaml
        data = yaml.safe_load(yaml_content)
        rules = [BusinessRule(**r) for r in data.get('rules', [])]
        _rules_config = BusinessRulesConfig(
            version=data.get('version', '1.0'),
            namespace=data.get('namespace', 'default'),
            rules=rules
        )
        return f"✓ Loaded {len(rules)} business rules"
    except Exception as e:
        return f"ERROR: {e}"


def execute_all_business_rules() -> str:
    """Execute all loaded business rules and capture lineage."""
    global _rules_config
    if not _rules_config:
        return "ERROR: No rules loaded. Use load_business_rules first."
    try:
        executor = BusinessRulesExecutor(get_pg(), get_marquez())
        results = executor.execute_all(_rules_config)
        success = sum(1 for r in results if r.get('success'))
        return f"✓ Executed {success}/{len(results)} rules successfully"
    except Exception as e:
        return f"ERROR: {e}"


def import_lineage_to_graph() -> str:
    """Import lineage data from Marquez into Neo4j."""
    try:
        count = import_lineage_to_neo4j(get_marquez(), get_neo4j())
        return f"✓ Imported {count} lineage relationships to graph"
    except Exception as e:
        return f"ERROR: {e}"


# --- RDF/SPARQL Tools ---
def test_rdf_connection() -> str:
    """Test connection to Apache Jena Fuseki."""
    try:
        if get_fuseki().test_connection():
            return "✓ Connected to Fuseki"
        return "✗ Fuseki connection failed"
    except Exception as e:
        return f"ERROR: {e}"


def sync_graph_to_rdf() -> str:
    """Sync Neo4j graph to RDF/Fuseki."""
    try:
        count = sync_neo4j_to_rdf(get_neo4j(), get_fuseki())
        return f"✓ Synced {count} triples to RDF"
    except Exception as e:
        return f"ERROR: {e}"


def run_sparql(query: str) -> str:
    """Execute a SPARQL query."""
    try:
        results = get_fuseki().sparql_query(query)
        if not results:
            return "No results"
        return "\n".join(str(r) for r in results[:20])
    except Exception as e:
        return f"ERROR: {e}"


# =============================================================================
# Tool Registry for Claude
# =============================================================================

TOOL_FUNCTIONS = {
    # Dynamic tool management
    "check_tool_exists": check_tool_exists,
    "list_available_tools": list_available_tools,
    "create_dynamic_tool": create_dynamic_tool,
    "run_dynamic_tool": run_dynamic_tool,
    "get_tool_source": get_tool_source,
    "update_dynamic_tool": update_dynamic_tool,
    "delete_dynamic_tool": delete_dynamic_tool,
    # Database
    "test_database_connection": test_database_connection,
    "list_database_tables": list_database_tables,
    "get_table_schema": get_table_schema,
    "get_column_stats": get_column_stats,
    # FK Discovery
    "run_fk_discovery": run_fk_discovery,
    "analyze_potential_fk": analyze_potential_fk,
    "validate_fk_with_data": validate_fk_with_data,
    # Graph
    "clear_neo4j_graph": clear_neo4j_graph,
    "add_fk_to_graph": add_fk_to_graph,
    "get_graph_stats": get_graph_stats,
    "analyze_graph_centrality": analyze_graph_centrality,
    "run_cypher": run_cypher,
    # Business Rules
    "show_sample_business_rules": show_sample_business_rules,
    "load_business_rules": load_business_rules,
    "execute_all_business_rules": execute_all_business_rules,
    "import_lineage_to_graph": import_lineage_to_graph,
    # RDF
    "test_rdf_connection": test_rdf_connection,
    "sync_graph_to_rdf": sync_graph_to_rdf,
    "run_sparql": run_sparql,
}

TOOLS = [
    # Dynamic tool management
    {"name": "check_tool_exists", "description": "Check if a dynamic tool exists", 
     "input_schema": {"type": "object", "properties": {"tool_name": {"type": "string", "description": "Name of the tool"}}, "required": ["tool_name"]}},
    {"name": "list_available_tools", "description": "List all available tools (builtin and dynamic)", 
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
    {"name": "test_database_connection", "description": "Test the PostgreSQL database connection", 
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "list_database_tables", "description": "List all tables in the database with column counts", 
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_table_schema", "description": "Get schema details for a table (columns, types, PKs)", 
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}}, "required": ["table_name"]}},
    {"name": "get_column_stats", "description": "Get statistics for a column (uniqueness, nulls, samples)", 
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}, "column_name": {"type": "string"}}, "required": ["table_name", "column_name"]}},
    
    # FK Discovery
    {"name": "run_fk_discovery", "description": "Run the full FK discovery pipeline on the database", 
     "input_schema": {"type": "object", "properties": {"min_match_rate": {"type": "number", "default": 0.95}, "min_score": {"type": "number", "default": 0.5}}}},
    {"name": "analyze_potential_fk", "description": "Analyze a potential FK relationship and get a score", 
     "input_schema": {"type": "object", "properties": {"source_table": {"type": "string"}, "source_column": {"type": "string"}, "target_table": {"type": "string"}, "target_column": {"type": "string"}}, "required": ["source_table", "source_column", "target_table", "target_column"]}},
    {"name": "validate_fk_with_data", "description": "Validate a FK by checking actual data integrity", 
     "input_schema": {"type": "object", "properties": {"source_table": {"type": "string"}, "source_column": {"type": "string"}, "target_table": {"type": "string"}, "target_column": {"type": "string"}}, "required": ["source_table", "source_column", "target_table", "target_column"]}},
    
    # Graph
    {"name": "clear_neo4j_graph", "description": "Clear all nodes and relationships from Neo4j", 
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "add_fk_to_graph", "description": "Add a FK relationship to the Neo4j graph", 
     "input_schema": {"type": "object", "properties": {"source_table": {"type": "string"}, "source_column": {"type": "string"}, "target_table": {"type": "string"}, "target_column": {"type": "string"}, "score": {"type": "number", "default": 1.0}, "cardinality": {"type": "string", "default": "N:1"}}, "required": ["source_table", "source_column", "target_table", "target_column"]}},
    {"name": "get_graph_stats", "description": "Get statistics about the Neo4j graph", 
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "analyze_graph_centrality", "description": "Find hub and authority tables in the graph", 
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "run_cypher", "description": "Execute a Cypher query on Neo4j", 
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    
    # Business Rules
    {"name": "show_sample_business_rules", "description": "Show sample business rules YAML format", 
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "load_business_rules", "description": "Load business rules from YAML content", 
     "input_schema": {"type": "object", "properties": {"yaml_content": {"type": "string"}}, "required": ["yaml_content"]}},
    {"name": "execute_all_business_rules", "description": "Execute all loaded business rules", 
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "import_lineage_to_graph", "description": "Import lineage from Marquez into Neo4j", 
     "input_schema": {"type": "object", "properties": {}}},
    
    # RDF
    {"name": "test_rdf_connection", "description": "Test connection to Apache Jena Fuseki", 
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "sync_graph_to_rdf", "description": "Sync Neo4j graph to RDF/Fuseki", 
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "run_sparql", "description": "Execute a SPARQL query on Fuseki", 
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
]


# =============================================================================
# Streaming Functions
# =============================================================================

def print_token(text: str) -> None:
    """Print a single token immediately (no buffering)."""
    sys.stdout.write(text)
    sys.stdout.flush()


def stream_response(client: anthropic.Anthropic, messages: list) -> tuple:
    """
    Stream response from Claude, printing each token as it arrives.
    Returns (final_message, tool_calls_dict).
    """
    tool_inputs = {}  # block_id -> {"name": ..., "input": ""}
    current_block_id = None
    
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        system="""You are GraphWeaver, an AI assistant for database foreign key discovery, 
graph analysis, and data lineage tracking.

You can:
- Connect to PostgreSQL and discover FK relationships
- Build and analyze knowledge graphs in Neo4j  
- Execute business rules and track lineage via Marquez
- Sync data to RDF and run SPARQL queries
- Create custom dynamic tools on the fly

Always test connections before running discovery. Be thorough and explain your reasoning.""",
        messages=messages,
        tools=TOOLS,
    ) as stream:
        for event in stream:
            # New content block starting
            if event.type == "content_block_start":
                if event.content_block.type == "tool_use":
                    current_block_id = event.content_block.id
                    tool_inputs[current_block_id] = {
                        "name": event.content_block.name,
                        "input": ""
                    }
                    print_token(f"\n\n🔧 **{event.content_block.name}**\n")
            
            # Delta content arriving (tokens!)
            elif event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    # Print each text token immediately
                    print_token(event.delta.text)
                elif event.delta.type == "input_json_delta":
                    # Accumulate tool input JSON
                    if current_block_id:
                        tool_inputs[current_block_id]["input"] += event.delta.partial_json
            
            # Content block finished
            elif event.type == "content_block_stop":
                current_block_id = None
        
        return stream.get_final_message(), tool_inputs


# =============================================================================
# Main Loop
# =============================================================================

def main():
    """Main interactive loop with token-by-token streaming."""
    
    # Header
    print("\n" + "=" * 60)
    print("  🕸️  GraphWeaver Agent - Token Streaming Edition")
    print("=" * 60)
    print("\nI can help you discover FK relationships in your database.")
    print("Try saying:")
    print("  • 'connect and show me the tables'")
    print("  • 'find all foreign keys'")
    print("  • 'create a tool that generates an ERD diagram'")
    print("\nType 'quit' to exit.\n")
    
    client = anthropic.Anthropic()
    messages = []
    
    while True:
        try:
            # Get user input
            user_input = input("\033[92mYou:\033[0m ").strip()
            if not user_input:
                continue
            if user_input.lower() in ('quit', 'exit', 'q'):
                break
            
            # Add to conversation
            messages.append({"role": "user", "content": user_input})
            
            # Start agent response
            print("\n\033[96m🤖 Agent:\033[0m ", end="")
            sys.stdout.flush()
            
            # Agentic loop - keep going until no more tool calls
            while True:
                response, tool_inputs = stream_response(client, messages)
                messages.append({"role": "assistant", "content": response.content})
                
                # Check if we're done
                if response.stop_reason != "tool_use":
                    break
                
                # Execute tool calls
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print_token("\n⏳ Executing...")
                        
                        fn = TOOL_FUNCTIONS.get(block.name)
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
                
                # Add tool results for next iteration
                messages.append({"role": "user", "content": tool_results})
            
            print("\n")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted.")
            break
        except anthropic.APIError as e:
            print(f"\n\033[31mAPI Error: {e}\033[0m\n")
        except Exception as e:
            print(f"\n\033[31mError: {e}\033[0m")
            import traceback
            traceback.print_exc()
    
    print("\n👋 Goodbye!\n")


if __name__ == "__main__":
    main()
