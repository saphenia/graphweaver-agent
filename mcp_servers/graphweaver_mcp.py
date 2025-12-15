"""
GraphWeaver MCP Server - Tools for Claude to discover FK relationships.

This server exposes database and graph operations as MCP tools that
Claude can call to autonomously discover foreign key relationships.
"""
import os
import sys
import json
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphweaver_agent import (
    DataSourceConfig, Neo4jConfig, PostgreSQLConnector,
    Neo4jClient, GraphBuilder, GraphAnalyzer,
    score_fk_candidate, validate_fk_with_data, determine_cardinality,
)

# =============================================================================
# MCP Server
# =============================================================================

mcp = FastMCP("graphweaver")

# Global state for connections
_pg_connector: Optional[PostgreSQLConnector] = None
_neo4j_client: Optional[Neo4jClient] = None


def get_pg() -> PostgreSQLConnector:
    global _pg_connector
    if _pg_connector is None:
        _pg_connector = PostgreSQLConnector(DataSourceConfig(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            database=os.environ.get("POSTGRES_DB", "orders"),
            username=os.environ.get("POSTGRES_USER", "saphenia"),
            password=os.environ.get("POSTGRES_PASSWORD", "secret"),
        ))
    return _pg_connector


def get_neo4j() -> Neo4jClient:
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient(Neo4jConfig(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            username=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "password"),
        ))
    return _neo4j_client


# =============================================================================
# PostgreSQL Tools
# =============================================================================

@mcp.tool()
async def test_database_connection() -> str:
    """Test connection to the PostgreSQL database. Call this first to verify connectivity."""
    result = get_pg().test_connection()
    if result["success"]:
        return f"✓ Connected to database '{result['database']}' as user '{result['user']}'"
    return f"✗ Connection failed: {result['error']}"


@mcp.tool()
async def list_tables() -> str:
    """List all tables in the database with row counts. Use this to see what tables exist."""
    tables = get_pg().get_tables_with_info()
    if not tables:
        return "No tables found in database."
    
    output = "## Tables in Database\n\n"
    output += "| Table | Columns | Rows |\n|-------|---------|------|\n"
    for t in tables:
        output += f"| {t['table_name']} | {t['column_count']} | {t['row_estimate']} |\n"
    return output


@mcp.tool()
async def get_table_schema(table_name: str) -> str:
    """Get detailed schema for a specific table including columns, types, and primary keys.
    
    Args:
        table_name: Name of the table to inspect
    """
    meta = get_pg().get_table_metadata(table_name)
    
    output = f"## Table: {table_name}\n\n"
    output += f"**Rows:** {meta.row_count}\n"
    output += f"**Primary Key:** {', '.join(meta.primary_key_columns) or 'None'}\n\n"
    output += "### Columns\n\n"
    output += "| Column | Type | Nullable | PK |\n|--------|------|----------|----|\\n"
    for col in meta.columns:
        pk = "✓" if col.is_primary_key else ""
        output += f"| {col.column_name} | {col.data_type.value} | {col.is_nullable} | {pk} |\n"
    return output


@mcp.tool()
async def get_column_statistics(table_name: str, column_name: str) -> str:
    """Get statistics for a column - distinct values, nulls, sample values.
    Use this to understand column characteristics before FK detection.
    
    Args:
        table_name: Table containing the column
        column_name: Column to analyze
    """
    stats = get_pg().get_column_statistics(table_name, column_name)
    
    output = f"## Column: {table_name}.{column_name}\n\n"
    output += f"- **Total rows:** {stats.total_count}\n"
    output += f"- **Distinct values:** {stats.distinct_count}\n"
    output += f"- **Uniqueness:** {stats.uniqueness_ratio:.1%}\n"
    output += f"- **Null count:** {stats.null_count} ({stats.null_ratio:.1%})\n"
    output += f"- **Sample values:** {stats.sample_values[:5]}\n"
    return output


@mcp.tool()
async def analyze_fk_candidate(
    source_table: str,
    source_column: str,
    target_table: str,
    target_column: str
) -> str:
    """Analyze if a column pair could be a foreign key relationship.
    Checks type compatibility, name similarity, and cardinality.
    
    Args:
        source_table: Table with potential FK column
        source_column: Column that might reference another table
        target_table: Table that might be referenced
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
    
    result = score_fk_candidate(source_stats, target_stats, source_col, target_col)
    
    output = f"## FK Analysis: {source_table}.{source_column} → {target_table}.{target_column}\n\n"
    output += f"**Score:** {result['score']}\n"
    output += f"**Recommendation:** {result['recommendation']}\n\n"
    output += "### Details\n"
    output += f"- Type compatible: {result['type_compatible']}\n"
    output += f"- Name similarity: {result['name_similarity']:.2f}\n"
    output += f"- Target uniqueness: {result['target_uniqueness']:.1%}\n"
    output += f"- Cardinality ratio: {result['cardinality_ratio']:.2f}\n"
    output += f"- Is value column: {result['is_value_column']}\n"
    
    return output


@mcp.tool()
async def validate_fk_with_sample(
    source_table: str,
    source_column: str,
    target_table: str,
    target_column: str,
    sample_size: int = 10000
) -> str:
    """Validate FK by checking if source values actually exist in target.
    This is the definitive test - checks real data.
    
    Args:
        source_table: Table with FK column
        source_column: FK column
        target_table: Referenced table
        target_column: Referenced column (usually PK)
        sample_size: Number of values to sample
    """
    result = get_pg().check_referential_integrity(
        source_table, source_column, target_table, target_column, sample_size
    )
    
    validation = validate_fk_with_data(result["match_rate"], result["sample_size"])
    cardinality = determine_cardinality(
        get_pg().get_column_statistics(source_table, source_column).uniqueness_ratio,
        get_pg().get_column_statistics(target_table, target_column).uniqueness_ratio
    )
    
    output = f"## FK Validation: {source_table}.{source_column} → {target_table}.{target_column}\n\n"
    output += f"**Verdict:** {validation['verdict']}\n"
    output += f"**Confidence:** {validation['confidence']:.1%}\n"
    output += f"**Cardinality:** {cardinality}\n\n"
    output += f"### Sample Results\n"
    output += f"- Sampled: {result['sample_size']} distinct values\n"
    output += f"- Matches: {result['matches']}\n"
    output += f"- Orphans: {result['orphans']}\n"
    
    return output


@mcp.tool()
async def run_sql_query(query: str) -> str:
    """Execute a SELECT query on the database. Only SELECT allowed.
    
    Args:
        query: SQL SELECT query
    """
    try:
        results = get_pg().execute_query(query)
        if not results:
            return "Query returned no results."
        
        # Format as markdown table
        cols = list(results[0].keys())
        output = "| " + " | ".join(cols) + " |\n"
        output += "|" + "|".join(["---"] * len(cols)) + "|\n"
        for row in results[:20]:
            output += "| " + " | ".join(str(row[c])[:50] for c in cols) + " |\n"
        
        if len(results) > 20:
            output += f"\n*...and {len(results) - 20} more rows*"
        
        return output
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# Neo4j Graph Tools
# =============================================================================

@mcp.tool()
async def test_neo4j_connection() -> str:
    """Test connection to Neo4j graph database."""
    result = get_neo4j().test_connection()
    if result["success"]:
        return "✓ Connected to Neo4j"
    return f"✗ Neo4j connection failed: {result.get('error')}"


@mcp.tool()
async def clear_graph() -> str:
    """Clear all nodes and relationships from Neo4j. Use before rebuilding."""
    builder = GraphBuilder(get_neo4j())
    builder.clear_graph()
    return "✓ Graph cleared"


@mcp.tool()
async def add_fk_to_graph(
    source_table: str,
    source_column: str,
    target_table: str,
    target_column: str,
    score: float,
    cardinality: str = "1:N"
) -> str:
    """Add a discovered FK relationship to the Neo4j graph.
    
    Args:
        source_table: Table with FK
        source_column: FK column
        target_table: Referenced table
        target_column: Referenced column
        score: Confidence score (0-1)
        cardinality: Relationship type (1:1, 1:N, N:M)
    """
    builder = GraphBuilder(get_neo4j())
    builder.add_table(source_table)
    builder.add_table(target_table)
    builder.add_fk_relationship(source_table, source_column, target_table, target_column, score, cardinality)
    return f"✓ Added FK: {source_table}.{source_column} → {target_table}.{target_column}"


@mcp.tool()
async def get_graph_statistics() -> str:
    """Get statistics about the current graph."""
    stats = GraphAnalyzer(get_neo4j()).get_statistics()
    return f"## Graph Statistics\n\n- Tables: {stats['tables']}\n- Columns: {stats['columns']}\n- FK Relationships: {stats['fks']}"


@mcp.tool()
async def analyze_graph_centrality() -> str:
    """Analyze which tables are hubs (reference many) vs authorities (referenced by many)."""
    result = GraphAnalyzer(get_neo4j()).centrality_analysis()
    
    output = "## Centrality Analysis\n\n"
    output += f"**Hub tables** (reference many others - likely fact tables): {result['hub_tables']}\n\n"
    output += f"**Authority tables** (referenced by many - likely dimension tables): {result['authority_tables']}\n\n"
    output += f"**Isolated tables** (no FK relationships): {result['isolated_tables']}\n\n"
    
    output += "### Degree Centrality\n\n"
    output += "| Table | In-Degree | Out-Degree | Total |\n|-------|-----------|------------|-------|\n"
    for r in result["centrality"][:10]:
        output += f"| {r['table_name']} | {r['in_degree']} | {r['out_degree']} | {r['total_degree']} |\n"
    
    return output


@mcp.tool()
async def detect_communities() -> str:
    """Find clusters of related tables based on FK relationships."""
    communities = GraphAnalyzer(get_neo4j()).community_detection()
    
    if not communities:
        return "No communities detected. Graph may be empty or disconnected."
    
    output = "## Table Communities\n\n"
    for i, comm in enumerate(communities):
        output += f"**Community {i+1}** ({comm['size']} tables): {', '.join(comm['tables'])}\n\n"
    
    return output


@mcp.tool()
async def predict_missing_fks() -> str:
    """Predict potentially missing FK relationships based on column naming patterns."""
    predictions = GraphAnalyzer(get_neo4j()).predict_missing_fks()
    
    if not predictions:
        return "No missing FK predictions. Either graph is complete or naming patterns don't suggest any."
    
    output = "## Predicted Missing FKs\n\n"
    output += "These columns look like FKs based on naming but aren't in the graph:\n\n"
    output += "| Source Table | Column | Suggested Target |\n|--------------|--------|------------------|\n"
    for p in predictions:
        output += f"| {p['source_table']} | {p['source_column']} | {p['target_table']} |\n"
    
    return output


@mcp.tool()
async def run_cypher_query(query: str) -> str:
    """Execute a Cypher query on Neo4j.
    
    Args:
        query: Cypher query to run
    """
    try:
        results = get_neo4j().run_query(query)
        if not results:
            return "Query returned no results."
        return json.dumps(results[:20], indent=2, default=str)
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    mcp.run()