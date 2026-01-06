"""
Vector Index Management for Neo4j.

Creates and manages HNSW vector indexes for fast similarity search.
Neo4j 5.11+ required for native vector indexes.
"""
from typing import Dict, List, Any, Optional


class VectorIndexManager:
    """Manage Neo4j vector indexes."""
    
    def __init__(self, neo4j_client):
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
                if int(major) >= 5 and int(minor) >= 11:
                    print(f"[VectorIndexManager] Neo4j {version} supports vector indexes")
                    return True
                else:
                    print(f"[VectorIndexManager] Neo4j {version} - vector indexes require 5.11+")
        except Exception as e:
            print(f"[VectorIndexManager] Error checking version: {e}")
        return False
    
    def create_text_embedding_indexes(self, dimensions: int = 384) -> Dict[str, bool]:
        """Create vector indexes for text embeddings on all node types."""
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
        """Create vector indexes for KG embeddings on all node types."""
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
        """Create all vector indexes."""
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
        self.neo4j = neo4j_client
        
    def search_tables(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        min_score: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Search tables by semantic similarity."""
        result = self.neo4j.run_query("""
            MATCH (t:Table)
            WHERE t.text_embedding IS NOT NULL
            WITH t, gds.similarity.cosine(t.text_embedding, $embedding) AS score
            WHERE score >= $min_score
            RETURN t.name AS name, score
            ORDER BY score DESC
            LIMIT $top_k
        """, {"embedding": query_embedding, "top_k": top_k, "min_score": min_score})
        return [dict(r) for r in result] if result else []
    
    def search_columns(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        min_score: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Search columns by semantic similarity."""
        result = self.neo4j.run_query("""
            MATCH (c:Column)-[:BELONGS_TO]->(t:Table)
            WHERE c.text_embedding IS NOT NULL
            WITH t, c, gds.similarity.cosine(c.text_embedding, $embedding) AS score
            WHERE score >= $min_score
            RETURN t.name AS table, c.name AS name, score
            ORDER BY score DESC
            LIMIT $top_k
        """, {"embedding": query_embedding, "top_k": top_k, "min_score": min_score})
        return [dict(r) for r in result] if result else []


# =============================================================================
# Standalone function (used by streamlit_app.py)
# =============================================================================

def create_all_indexes(neo4j_client) -> Dict[str, Any]:
    """Create all vector indexes. Standalone wrapper for VectorIndexManager."""
    manager = VectorIndexManager(neo4j_client)
    
    if not manager.check_vector_support():
        return {"error": "Neo4j version does not support vector indexes (requires 5.11+)"}
    
    results = manager.create_all_indexes()
    
    # Count successes
    text_success = sum(1 for v in results["text_indexes"].values() if v)
    kg_success = sum(1 for v in results["kg_indexes"].values() if v)
    
    return {
        "indexes_created": text_success + kg_success,
        "text_indexes": text_success,
        "kg_indexes": kg_success,
        "details": results,
    }
