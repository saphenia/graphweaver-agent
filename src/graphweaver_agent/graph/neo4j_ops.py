"""Neo4j Graph Operations - Enhanced with embedding support."""
from typing import Any, Dict, List, Optional
from contextlib import contextmanager
from neo4j import GraphDatabase
from graphweaver_agent.models import Neo4jConfig


class Neo4jClient:
    def __init__(self, config: Neo4jConfig):
        self.config = config
        self._driver = None
    
    def connect(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
            )
        return self._driver
    
    def disconnect(self):
        if self._driver:
            self._driver.close()
            self._driver = None
    
    @contextmanager
    def session(self):
        driver = self.connect()
        session = driver.session(database=self.config.database)
        try:
            yield session
        finally:
            session.close()
    
    def run_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        with self.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]
    
    def run_write(self, query: str, params: Optional[Dict] = None):
        with self.session() as session:
            p = params or {}
            result = session.run(query, p)
            result.consume()

    def test_connection(self) -> Dict[str, Any]:
        try:
            self.run_query("RETURN 1")
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def check_gds_available(self) -> bool:
        """Check if Neo4j GDS plugin is available."""
        try:
            result = self.run_query("RETURN gds.version() as version")
            return bool(result)
        except:
            return False


class GraphBuilder:
    """Build graph with optional automatic embedding generation."""
    
    def __init__(self, client: Neo4jClient, embedder: Optional[Any] = None):
        """
        Initialize GraphBuilder.
        
        Args:
            client: Neo4jClient instance
            embedder: Optional TextEmbedder instance for auto-embedding
        """
        self.client = client
        self._embedder = embedder
        self._auto_embed = embedder is not None
    
    @property
    def embedder(self):
        """Lazy load embedder if needed."""
        if self._embedder is None and self._auto_embed:
            from graphweaver_agent.embeddings.text_embeddings import TextEmbedder
            self._embedder = TextEmbedder()
        return self._embedder
    
    def enable_auto_embedding(self, embedder: Optional[Any] = None):
        """Enable automatic embedding generation for new nodes."""
        self._auto_embed = True
        if embedder:
            self._embedder = embedder
    
    def disable_auto_embedding(self):
        """Disable automatic embedding generation."""
        self._auto_embed = False
    
    def clear_graph(self):
        self.client.run_write("MATCH (n) DETACH DELETE n")
    
    def add_table(self, table_name: str, datasource_id: str = "default", 
                  column_names: Optional[List[str]] = None):
        """
        Add a table node, optionally with embedding.
        
        Args:
            table_name: Name of the table
            datasource_id: Data source identifier
            column_names: Optional list of column names (for embedding context)
        """
        self.client.run_write("""
            MERGE (d:DataSource {id: $ds_id})
            MERGE (t:Table {name: $name})
            MERGE (t)-[:BELONGS_TO]->(d)
        """, {"ds_id": datasource_id, "name": table_name})
        
        # Auto-embed if enabled
        if self._auto_embed and self.embedder:
            try:
                cols = column_names or []
                emb = self.embedder.embed_table_metadata(table_name, cols)
                self.client.run_write("""
                    MATCH (t:Table {name: $name})
                    SET t.text_embedding = $embedding,
                        t.embedding_model = $model
                """, {
                    "name": table_name,
                    "embedding": emb.embedding,
                    "model": emb.model,
                })
            except Exception as e:
                print(f"[GraphBuilder] Warning: Failed to embed table {table_name}: {e}")
    
    def add_column(self, table_name: str, column_name: str, 
                   data_type: Optional[str] = None):
        """
        Add a column node, optionally with embedding.
        
        Args:
            table_name: Name of the parent table
            column_name: Name of the column
            data_type: Optional data type string
        """
        self.client.run_write("""
            MERGE (t:Table {name: $table_name})
            MERGE (c:Column {name: $col_name, table: $table_name})
            MERGE (c)-[:BELONGS_TO]->(t)
        """, {"table_name": table_name, "col_name": column_name})
        
        if data_type:
            self.client.run_write("""
                MATCH (c:Column {name: $col_name, table: $table_name})
                SET c.data_type = $data_type
            """, {"table_name": table_name, "col_name": column_name, "data_type": data_type})
        
        # Auto-embed if enabled
        if self._auto_embed and self.embedder:
            try:
                emb = self.embedder.embed_column_metadata(
                    table_name=table_name,
                    column_name=column_name,
                    data_type=data_type,
                )
                self.client.run_write("""
                    MATCH (c:Column {name: $col_name, table: $table_name})
                    SET c.text_embedding = $embedding,
                        c.embedding_model = $model
                """, {
                    "col_name": column_name,
                    "table_name": table_name,
                    "embedding": emb.embedding,
                    "model": emb.model,
                })
            except Exception as e:
                print(f"[GraphBuilder] Warning: Failed to embed column {table_name}.{column_name}: {e}")
    
    def add_fk_relationship(self, source_table: str, source_col: str,
                           target_table: str, target_col: str,
                           score: float, cardinality: str,
                           auto_embed_columns: bool = True):
        """
        Add FK relationship between columns, creating nodes if needed.
        
        Args:
            source_table: Source table name
            source_col: Source column name
            target_table: Target table name
            target_col: Target column name
            score: Confidence score
            cardinality: Relationship cardinality (e.g., 'many-to-one')
            auto_embed_columns: Whether to auto-embed columns if not already embedded
        """
        # First ensure both tables exist
        self.client.run_write("""
            MERGE (st:Table {name: $src_table})
            MERGE (tt:Table {name: $tgt_table})
        """, {"src_table": source_table, "tgt_table": target_table})
        
        # Then create columns and FK relationship
        self.client.run_write("""
            MATCH (st:Table {name: $src_table})
            MATCH (tt:Table {name: $tgt_table})
            MERGE (sc:Column {name: $src_col, table: $src_table})
            MERGE (tc:Column {name: $tgt_col, table: $tgt_table})
            MERGE (sc)-[:BELONGS_TO]->(st)
            MERGE (tc)-[:BELONGS_TO]->(tt)
            MERGE (sc)-[r:FK_TO]->(tc)
            SET r.score = $score, r.cardinality = $cardinality
        """, {
            "src_table": source_table, "src_col": source_col,
            "tgt_table": target_table, "tgt_col": target_col,
            "score": score, "cardinality": cardinality,
        })
        
        # Auto-embed columns if enabled and requested
        if self._auto_embed and self.embedder and auto_embed_columns:
            self._embed_column_if_missing(source_table, source_col)
            self._embed_column_if_missing(target_table, target_col)
    
    def _embed_column_if_missing(self, table_name: str, column_name: str):
        """Embed a column only if it doesn't already have an embedding."""
        try:
            result = self.client.run_query("""
                MATCH (c:Column {name: $col_name, table: $table_name})
                RETURN c.text_embedding IS NOT NULL as has_embedding
            """, {"col_name": column_name, "table_name": table_name})
            
            if result and not result[0].get("has_embedding"):
                emb = self.embedder.embed_column_metadata(
                    table_name=table_name,
                    column_name=column_name,
                )
                self.client.run_write("""
                    MATCH (c:Column {name: $col_name, table: $table_name})
                    SET c.text_embedding = $embedding,
                        c.embedding_model = $model
                """, {
                    "col_name": column_name,
                    "table_name": table_name,
                    "embedding": emb.embedding,
                    "model": emb.model,
                })
        except Exception as e:
            print(f"[GraphBuilder] Warning: Failed to embed {table_name}.{column_name}: {e}")
    
    def embed_all_existing_nodes(self) -> Dict[str, int]:
        """
        Generate embeddings for all existing nodes that don't have them.
        
        Returns:
            Stats dict with counts of embedded nodes
        """
        if not self.embedder:
            from graphweaver_agent.embeddings.text_embeddings import TextEmbedder
            self._embedder = TextEmbedder()
        
        stats = {"tables": 0, "columns": 0, "errors": []}
        
        # Embed tables without embeddings
        tables = self.client.run_query("""
            MATCH (t:Table)
            WHERE t.text_embedding IS NULL
            OPTIONAL MATCH (t)<-[:BELONGS_TO]-(c:Column)
            RETURN t.name as name, collect(c.name) as columns
        """)
        
        for t in tables:
            try:
                emb = self.embedder.embed_table_metadata(t["name"], t["columns"] or [])
                self.client.run_write("""
                    MATCH (t:Table {name: $name})
                    SET t.text_embedding = $embedding,
                        t.embedding_model = $model
                """, {
                    "name": t["name"],
                    "embedding": emb.embedding,
                    "model": emb.model,
                })
                stats["tables"] += 1
            except Exception as e:
                stats["errors"].append({"node": f"Table:{t['name']}", "error": str(e)})
        
        # Embed columns without embeddings
        columns = self.client.run_query("""
            MATCH (c:Column)-[:BELONGS_TO]->(t:Table)
            WHERE c.text_embedding IS NULL
            RETURN c.name as col_name, c.table as table_name, c.data_type as data_type
        """)
        
        for c in columns:
            try:
                emb = self.embedder.embed_column_metadata(
                    table_name=c["table_name"],
                    column_name=c["col_name"],
                    data_type=c.get("data_type"),
                )
                self.client.run_write("""
                    MATCH (c:Column {name: $col_name, table: $table_name})
                    SET c.text_embedding = $embedding,
                        c.embedding_model = $model
                """, {
                    "col_name": c["col_name"],
                    "table_name": c["table_name"],
                    "embedding": emb.embedding,
                    "model": emb.model,
                })
                stats["columns"] += 1
            except Exception as e:
                stats["errors"].append({
                    "node": f"Column:{c['table_name']}.{c['col_name']}", 
                    "error": str(e)
                })
        
        if stats["errors"]:
            print(f"[GraphBuilder] Embedding complete with {len(stats['errors'])} errors:")
            for err in stats["errors"][:5]:
                print(f"  - {err['node']}: {err['error']}")
            if len(stats["errors"]) > 5:
                print(f"  ... and {len(stats['errors']) - 5} more errors")
        
        print(f"[GraphBuilder] Embedded {stats['tables']} tables, {stats['columns']} columns")
        return stats
    
    def ensure_embeddings_exist(self) -> Dict[str, Any]:
        """
        Check and report on embedding coverage.
        Returns statistics about which nodes have embeddings.
        """
        result = self.client.run_query("""
            MATCH (n)
            WHERE n:Table OR n:Column
            WITH labels(n)[0] AS label,
                 n.text_embedding IS NOT NULL AS has_emb
            RETURN label,
                   count(*) AS total,
                   sum(CASE WHEN has_emb THEN 1 ELSE 0 END) AS with_embedding
            ORDER BY label
        """)
        
        stats = {"by_type": {}, "total": 0, "with_embedding": 0}
        for row in result:
            stats["by_type"][row["label"]] = {
                "total": row["total"],
                "with_embedding": row["with_embedding"],
                "missing": row["total"] - row["with_embedding"],
            }
            stats["total"] += row["total"]
            stats["with_embedding"] += row["with_embedding"]
        
        stats["missing"] = stats["total"] - stats["with_embedding"]
        stats["coverage"] = f"{100 * stats['with_embedding'] / stats['total']:.1f}%" if stats["total"] > 0 else "N/A"
        
        return stats


class GraphAnalyzer:
    def __init__(self, client: Neo4jClient):
        self.client = client
    
    def get_statistics(self) -> Dict[str, int]:
        result = self.client.run_query("""
            MATCH (t:Table) WITH count(t) as tables
            MATCH (c:Column) WITH tables, count(c) as columns
            MATCH ()-[r:FK_TO]->() WITH tables, columns, count(r) as fks
            RETURN tables, columns, fks
        """)
        return result[0] if result else {"tables": 0, "columns": 0, "fks": 0}
    
    def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get statistics about node embeddings."""
        result = self.client.run_query("""
            MATCH (t:Table)
            WITH count(t) as total_tables,
                 count(CASE WHEN t.text_embedding IS NOT NULL THEN 1 END) as tables_with_text_emb,
                 count(CASE WHEN t.kg_embedding IS NOT NULL THEN 1 END) as tables_with_kg_emb
            MATCH (c:Column)
            WITH total_tables, tables_with_text_emb, tables_with_kg_emb,
                 count(c) as total_columns,
                 count(CASE WHEN c.text_embedding IS NOT NULL THEN 1 END) as columns_with_text_emb,
                 count(CASE WHEN c.kg_embedding IS NOT NULL THEN 1 END) as columns_with_kg_emb
            RETURN total_tables, tables_with_text_emb, tables_with_kg_emb,
                   total_columns, columns_with_text_emb, columns_with_kg_emb
        """)
        
        if result:
            r = result[0]
            return {
                "tables": {
                    "total": r["total_tables"],
                    "with_text_embedding": r["tables_with_text_emb"],
                    "with_kg_embedding": r["tables_with_kg_emb"],
                    "missing_text_embedding": r["total_tables"] - r["tables_with_text_emb"],
                },
                "columns": {
                    "total": r["total_columns"],
                    "with_text_embedding": r["columns_with_text_emb"],
                    "with_kg_embedding": r["columns_with_kg_emb"],
                    "missing_text_embedding": r["total_columns"] - r["columns_with_text_emb"],
                },
            }
        return {}
    
    def centrality_analysis(self) -> Dict[str, Any]:
        results = self.client.run_query("""
            MATCH (t:Table)
            OPTIONAL MATCH (t)<-[:BELONGS_TO]-(c:Column)-[:FK_TO]->()
            WITH t, count(DISTINCT c) as out_degree
            OPTIONAL MATCH (t)<-[:BELONGS_TO]-(c:Column)<-[:FK_TO]-()
            WITH t, out_degree, count(DISTINCT c) as in_degree
            RETURN t.name as table_name, in_degree, out_degree,
                   in_degree + out_degree as total_degree
            ORDER BY total_degree DESC
        """)
        return {
            "centrality": results,
            "hub_tables": [r["table_name"] for r in results if r["out_degree"] > 1],
            "authority_tables": [r["table_name"] for r in results if r["in_degree"] > 1],
            "isolated_tables": [r["table_name"] for r in results if r["total_degree"] == 0],
        }
    
    def community_detection(self) -> List[Dict]:
        results = self.client.run_query("""
            MATCH (t1:Table)<-[:BELONGS_TO]-(c1:Column)-[:FK_TO]->(c2:Column)-[:BELONGS_TO]->(t2:Table)
            WITH t1.name as source, collect(DISTINCT t2.name) as targets
            RETURN source, targets ORDER BY size(targets) DESC
        """)
        communities = []
        visited = set()
        for row in results:
            if row["source"] in visited:
                continue
            community = {row["source"]}
            community.update(row["targets"])
            visited.update(community)
            communities.append({"tables": list(community), "size": len(community)})
        return communities
    
    def predict_missing_fks(self) -> List[Dict]:
        return self.client.run_query("""
            MATCH (c:Column)-[:BELONGS_TO]->(t:Table)
            WHERE c.name ENDS WITH '_id' AND NOT (c)-[:FK_TO]->() AND c.name <> 'id'
            WITH c, t
            MATCH (target:Table)
            WHERE toLower(replace(c.name, '_id', '')) = toLower(target.name)
               OR toLower(replace(c.name, '_id', '')) + 's' = toLower(target.name)
            RETURN t.name as source_table, c.name as source_column,
                   target.name as target_table, 'name_pattern' as reason
        """)