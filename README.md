# GraphWeaver Agent

**Chat with Claude to discover FK relationships, execute business rules, and capture data lineage.**

## Architecture
```
┌─────────────────────────────────────────────────────────────────────┐
│                         Claude Agent                                 │
│    "Discover FKs, run business rules, analyze data lineage..."      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  PostgreSQL   │   │    Neo4j      │   │   Marquez     │
│  (your data)  │   │ (FK + lineage │   │  (OpenLineage │
│               │   │    graph)     │   │    events)    │
└───────────────┘   └───────────────┘   └───────────────┘
```

## Quick Start
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
docker compose up -d postgres neo4j marquez marquez-web
# Wait ~30s for services
docker compose run --rm agent
```

## What You Can Do

### 1. Discover Foreign Keys
```
You: discover all foreign keys
```
Runs 5-stage pipeline (statistical → mathematical → sampling → graph → semantic)

### 2. Execute Business Rules
```
You: show me sample business rules
You: load these business rules: [paste YAML]
You: execute all business rules
```
Runs your SQL queries and captures lineage to Marquez

### 3. Build Knowledge Graph
```
You: import lineage to graph
You: analyze data flow for orders table
You: what breaks if I change customers table?
```
Combines FK relationships + operational lineage

## UIs

| Service | URL | Purpose |
|---------|-----|---------|
| Neo4j Browser | http://localhost:7474 | Visualize FK + lineage graph |
| Marquez Web | http://localhost:3000 | View data lineage |

## Business Rules Format

Create `business_rules.yaml`:
```yaml
version: "1.0"
namespace: "mycompany"

rules:
  - name: daily_revenue
    description: Calculate daily revenue
    type: aggregation
    sql: |
      SELECT date, SUM(amount) as revenue
      FROM orders
      GROUP BY date
    inputs:
      - orders
    outputs:
      - daily_revenue_report
    tags:
      - revenue
```

## Knowledge Graph Queries

In Neo4j Browser:
```cypher
-- See all FK relationships
MATCH (t1:Table)<-[:BELONGS_TO]-(c1)-[fk:FK_TO]->(c2)-[:BELONGS_TO]->(t2:Table)
RETURN t1.name, c1.name, t2.name, c2.name, fk.score

-- See lineage
MATCH (j:Job)-[r]->(d:Dataset)
RETURN j, r, d

-- Combined: FKs + Jobs for a table
MATCH (t:Table {name: 'orders'})
OPTIONAL MATCH (t)<-[:BELONGS_TO]-(c:Column)-[:FK_TO]->(tc)
OPTIONAL MATCH (j:Job)-[:READS|WRITES]->(d:Dataset {name: 'orders'})
RETURN t, c, tc, j, d
```

## Tools Reference

| Tool | Purpose |
|------|---------|
| `run_fk_discovery` | Complete 5-stage FK discovery |
| `execute_all_business_rules` | Run rules + capture lineage |
| `import_lineage_to_graph` | Merge lineage into Neo4j |
| `analyze_data_flow` | Show FKs + lineage for table |
| `find_impact_analysis` | What breaks if table changes |

## Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Claude (via LangChain)                    │
│                                                              │
│  "I should check if orders.customer_id references           │
│   customers.id... let me validate with actual data..."      │
│                                                              │
│         │ thinks │ decides │ calls tools │                   │
└─────────┴────────┴─────────┴─────────────┴──────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
      ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
      │  PostgreSQL │ │   Neo4j     │ │  Analysis   │
      │   Tools     │ │   Tools     │ │   Tools     │
      └─────────────┘ └─────────────┘ └─────────────┘
```

## Tools Available to Claude

### Database Tools
- `test_database_connection` - Verify connectivity
- `list_database_tables` - See all tables
- `get_table_schema` - Columns, types, PKs
- `get_column_stats` - Uniqueness, nulls, samples
- `analyze_potential_fk` - Score FK candidates
- `validate_fk_with_data` - Definitive FK test

### Graph Tools
- `clear_neo4j_graph` - Reset graph
- `add_fk_to_graph` - Add confirmed FK
- `get_graph_stats` - Current graph size
- `analyze_graph_centrality` - Hub/authority tables
- `find_table_communities` - Related table clusters
- `predict_missing_fks` - Suggested FKs

## Example Output
```
Agent: I'll start by testing the database connection.

[Calling: test_database_connection]
✓ Connected to database 'orders' as 'saphenia'

Agent: Good. Now let me see what tables exist.

[Calling: list_database_tables]
Tables:
  - categories: 4 columns, ~4 rows
  - customers: 4 columns, ~5 rows
  - orders: 4 columns, ~6 rows
  - order_items: 5 columns, ~11 rows
  - products: 5 columns, ~10 rows
  - product_suppliers: 4 columns, ~10 rows
  - suppliers: 3 columns, ~4 rows

Agent: I see 7 tables. Let me examine each for potential FK columns.
The orders table likely has a customer_id FK. Let me check...

[Calling: get_table_schema(orders)]
Table: orders (6 rows)
Primary Key: id
Columns:
  - id: integer [PK]
  - customer_id: integer
  - order_date: timestamp
  - total_amount: decimal

Agent: customer_id looks like a FK to customers. Let me analyze it.

[Calling: analyze_potential_fk(orders, customer_id, customers, id)]
Analysis: orders.customer_id → customers.id
  Score: 0.867
  Recommendation: LIKELY FK - Should validate with data
  Type compatible: True
  Name similarity: 0.90
  Target uniqueness: 100.0%

Agent: High score. Let me validate with actual data.

[Calling: validate_fk_with_data(orders, customer_id, customers, id)]
Validation: orders.customer_id → customers.id
  ✓ CONFIRMED FK
  Match rate: 100.0% (5/5)
  Cardinality: 1:N

Agent: Confirmed! Adding to graph...

[Calling: add_fk_to_graph(orders, customer_id, customers, id, 1.0, 1:N)]
✓ Added: orders.customer_id → customers.id

...continues for all tables...

=== FINAL REPORT ===

Discovered 6 FK relationships:
1. orders.customer_id → customers.id (1:N)
2. products.category_id → categories.id (1:N)
3. order_items.order_id → orders.id (1:N)
4. order_items.product_id → products.id (1:N)
5. product_suppliers.product_id → products.id (1:N)
6. product_suppliers.supplier_id → suppliers.id (1:N)

Graph Analysis:
- Hub tables (fact): order_items, product_suppliers
- Authority tables (dimension): customers, categories, products, suppliers
- Communities: 2 clusters identified
```

## MCP Server

The tools are also available as an MCP server:
```bash
python -m mcp_servers.graphweaver_mcp
```

For Claude Desktop, add to config:
```json
{
  "mcpServers": {
    "graphweaver": {
      "command": "python",
      "args": ["-m", "mcp_servers.graphweaver_mcp"],
      "env": {
        "POSTGRES_HOST": "localhost",
        "NEO4J_URI": "bolt://localhost:7687"
      }
    }
  }
}
```

## Interactive Mode
```bash
python agent.py --interactive
```

Then ask questions:
```
You: discover
You: what tables reference customers?
You: predict any missing relationships
```

## Environment Variables
```bash
ANTHROPIC_API_KEY=sk-ant-...   # Required

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=orders
POSTGRES_USER=saphenia
POSTGRES_PASSWORD=secret

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

## Project Structure
```
graphweaver-agent/
├── agent.py              # LangChain agent with Claude
├── mcp_servers/
│   └── graphweaver_mcp.py  # MCP server with tools
├── src/graphweaver_agent/
│   ├── models/           # Pydantic schemas
│   ├── connectors/       # PostgreSQL connector
│   ├── discovery/        # FK scoring functions
│   └── graph/            # Neo4j operations
├── docker-compose.yml    # Full stack
├── Dockerfile
├── init.sql              # Sample database
└── pyproject.toml
```

## Requirements

- Python 3.11+
- Docker (recommended)
- Anthropic API key

## License

MIT