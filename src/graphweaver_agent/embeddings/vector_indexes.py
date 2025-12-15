"""
Vector Index Management for Neo4j.

Creates and manages HNSW vector indexes for fast similarity search.
Neo4j 5.11+ required for native vector indexes.
"""
from typing import Dict, List, Any, Optional


class VectorIndexManager:
    """Manage Neo4j vector indexes."""
    
    def __init__(self, neo4j_client):
        """
        Initialize vector index manager.
        
        Args:
            neo4j_client: Neo4j client instance
        """
        self.neo4j = neo4j_client
        
    def check_vector_support(self) -> bool:
        """Check if Neo4j version supports vector indexes."""
        try:
            result = self.neo4j.run_query("""
                CALL dbms.components()
                YIELD name, versions
                WHERE name = 'Neo4j Kernel'
                RETURN versions[0] AS version
            """)
            if result:
                version = result[0]["version"]
                major, minor = version.split(".")[:2]
                # Vector indexes available in 5.11+
                if int(major) >= 5 and int(minor) >= 11:
                    print(f"[VectorIndexManager] Neo4j {version} supports vector indexes")
                    return True
                else:
                    print(f"[VectorIndexManager] Neo4j {version} - vector indexes require 5.11+")
        except Exception as e:
            print(f"[VectorIndexManager] Error checking version: {e}")
        return False
    
    def create_text_embedding_indexes(self, dimensions: int = 384) -> Dict[str, bool]:
        """
        Create vector indexes for text embeddings on all node types.
        
        Args:
            dimensions: Embedding dimensions (default 384 for MiniLM)
            
        Returns:
            Dict of index name -> success status
        """
        indexes = {
            "table_text_embedding": ("Table", "text_embedding"),
            "column_text_embedding": ("Column", "text_embedding"),
            "job_text_embedding": ("Job", "text_embedding"),
            "dataset_text_embedding": ("Dataset", "text_embedding"),
        }
        
        results = {}
        for index_name, (label, property_name) in indexes.items():
            try:
                self.neo4j.run_write(f"""
                    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                    FOR (n:{label})
                    ON (n.{property_name})
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {dimensions},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                """)
                results[index_name] = True
                print(f"[VectorIndexManager] Created index: {index_name}")
            except Exception as e:
                print(f"[VectorIndexManager] Failed to create {index_name}: {e}")
                results[index_name] = False
                
        return results
    
    def create_kg_embedding_indexes(self, dimensions: int = 128) -> Dict[str, bool]:
        """
        Create vector indexes for KG embeddings on all node types.
        
        Args:
            dimensions: Embedding dimensions (default 128 for FastRP)
            
        Returns:
            Dict of index name -> success status
        """
        indexes = {
            "table_kg_embedding": ("Table", "kg_embedding"),
            "column_kg_embedding": ("Column", "kg_embedding"),
            "job_kg_embedding": ("Job", "kg_embedding"),
            "dataset_kg_embedding": ("Dataset", "kg_embedding"),
        }
        
        results = {}
        for index_name, (label, property_name) in indexes.items():
            try:
                self.neo4j.run_write(f"""
                    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                    FOR (n:{label})
                    ON (n.{property_name})
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {dimensions},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                """)
                results[index_name] = True
                print(f"[VectorIndexManager] Created index: {index_name}")
            except Exception as e:
                print(f"[VectorIndexManager] Failed to create {index_name}: {e}")
                results[index_name] = False
                
        return results
    
    def create_all_indexes(
        self, 
        text_dimensions: int = 384, 
        kg_dimensions: int = 128
    ) -> Dict[str, Any]:
        """
        Create all vector indexes.
        
        Args:
            text_dimensions: Dimensions for text embeddings
            kg_dimensions: Dimensions for KG embeddings
            
        Returns:
            Combined results
        """
        return {
            "text_indexes": self.create_text_embedding_indexes(text_dimensions),
            "kg_indexes": self.create_kg_embedding_indexes(kg_dimensions),
        }
    
    def list_indexes(self) -> List[Dict[str, Any]]:
        """List all vector indexes in the database."""
        result = self.neo4j.run_query("""
            SHOW INDEXES
            WHERE type = 'VECTOR'
            RETURN name, labelsOrTypes, properties, state
        """)
        return [dict(r) for r in result] if result else []
    
    def drop_index(self, index_name: str) -> bool:
        """Drop a vector index."""
        try:
            self.neo4j.run_write(f"DROP INDEX {index_name} IF EXISTS")
            print(f"[VectorIndexManager] Dropped index: {index_name}")
            return True
        except Exception as e:
            print(f"[VectorIndexManager] Failed to drop {index_name}: {e}")
            return False
    
    def drop_all_indexes(self) -> int:
        """Drop all vector indexes."""
        indexes = self.list_indexes()
        count = 0
        for idx in indexes:
            if self.drop_index(idx["name"]):
                count += 1
        return count


class SemanticSearch:
    """Semantic search using vector indexes."""
    
    def __init__(self, neo4j_client):
        """
        Initialize semantic search.
        
        Args:
            neo4j_client: Neo4j client instance
        """
        self.neo4j = neo4j_client
        
    def search_tables(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        min_score: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Search tables by semantic similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            min_score: Minimum similarity score
            
        Returns:
            List of matching tables with scores
        """
        result = self.neo4j.run_query("""
            CALL db.index.vector.queryNodes('table_text_embedding', $top_k, $embedding)
            YIELD node, score
            WHERE score >= $min_score
            RETURN node.name AS table_name, score
            ORDER BY score DESC
        """, {"embedding": query_embedding, "top_k": top_k, "min_score": min_score})
        
        return [dict(r) for r in result] if result else []
    
    def search_columns(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        min_score: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Search columns by semantic similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            min_score: Minimum similarity score
            
        Returns:
            List of matching columns with scores
        """
        result = self.neo4j.run_query("""
            CALL db.index.vector.queryNodes('column_text_embedding', $top_k, $embedding)
            YIELD node, score
            WHERE score >= $min_score
            MATCH (node)-[:BELONGS_TO]->(t:Table)
            RETURN t.name AS table_name, node.name AS column_name, score
            ORDER BY score DESC
        """, {"embedding": query_embedding, "top_k": top_k, "min_score": min_score})
        
        return [dict(r) for r in result] if result else []
    
    def search_all(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        min_score: float = 0.5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search all node types by semantic similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results per type
            min_score: Minimum similarity score
            
        Returns:
            Dict with results for each node type
        """
        return {
            "tables": self.search_tables(query_embedding, top_k, min_score),
            "columns": self.search_columns(query_embedding, top_k, min_score),
        }
    
    def find_similar_to_table(
        self,
        table_name: str,
        top_k: int = 5,
        use_text_embedding: bool = True,
        use_kg_embedding: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Find tables similar to a given table using combined embeddings.
        
        Args:
            table_name: Source table name
            top_k: Number of results
            use_text_embedding: Use text embeddings
            use_kg_embedding: Use KG embeddings
            
        Returns:
            List of similar tables with combined scores
        """
        if use_text_embedding and use_kg_embedding:
            # Combined similarity
            result = self.neo4j.run_query("""
                MATCH (source:Table {name: $name})
                MATCH (other:Table)
                WHERE other <> source 
                  AND other.text_embedding IS NOT NULL 
                  AND other.kg_embedding IS NOT NULL
                WITH source, other,
                     gds.similarity.cosine(source.text_embedding, other.text_embedding) AS text_sim,
                     gds.similarity.cosine(source.kg_embedding, other.kg_embedding) AS kg_sim
                WITH other.name AS table_name,
                     text_sim,
                     kg_sim,
                     (text_sim + kg_sim) / 2 AS combined_score
                RETURN table_name, text_sim, kg_sim, combined_score
                ORDER BY combined_score DESC
                LIMIT $top_k
            """, {"name": table_name, "top_k": top_k})
        elif use_text_embedding:
            result = self.neo4j.run_query("""
                MATCH (source:Table {name: $name})
                MATCH (other:Table)
                WHERE other <> source AND other.text_embedding IS NOT NULL
                WITH other.name AS table_name,
                     gds.similarity.cosine(source.text_embedding, other.text_embedding) AS score
                RETURN table_name, score AS text_sim, null AS kg_sim, score AS combined_score
                ORDER BY score DESC
                LIMIT $top_k
            """, {"name": table_name, "top_k": top_k})
        else:
            result = self.neo4j.run_query("""
                MATCH (source:Table {name: $name})
                MATCH (other:Table)
                WHERE other <> source AND other.kg_embedding IS NOT NULL
                WITH other.name AS table_name,
                     gds.similarity.cosine(source.kg_embedding, other.kg_embedding) AS score
                RETURN table_name, null AS text_sim, score AS kg_sim, score AS combined_score
                ORDER BY score DESC
                LIMIT $top_k
            """, {"name": table_name, "top_k": top_k})
            
        return [dict(r) for r in result] if result else []
    
    def find_similar_to_column(
        self,
        table_name: str,
        column_name: str,
        top_k: int = 10,
        same_table: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Find columns similar to a given column.
        
        Args:
            table_name: Source table name
            column_name: Source column name
            top_k: Number of results
            same_table: Include columns from the same table
            
        Returns:
            List of similar columns
        """
        where_clause = "" if same_table else "AND t.name <> $table_name"
        
        result = self.neo4j.run_query(f"""
            MATCH (source:Column {{name: $column_name}})-[:BELONGS_TO]->(st:Table {{name: $table_name}})
            MATCH (other:Column)-[:BELONGS_TO]->(t:Table)
            WHERE other <> source 
              AND other.text_embedding IS NOT NULL
              {where_clause}
            WITH t.name AS table_name,
                 other.name AS column_name,
                 gds.similarity.cosine(source.text_embedding, other.text_embedding) AS score
            RETURN table_name, column_name, score
            ORDER BY score DESC
            LIMIT $top_k
        """, {"table_name": table_name, "column_name": column_name, "top_k": top_k})
        
        return [dict(r) for r in result] if result else []