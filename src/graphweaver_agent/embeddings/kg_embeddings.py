"""Knowledge Graph Embeddings using Neo4j GDS FastRP.

FIXED: Uses run_write() for GDS mutate operations which require write transactions.
The error "Writing in read access mode not allowed" was caused by using run_query()
for GDS operations that mutate the graph.
"""
from typing import Any, Dict, List, Optional


class KGEmbedder:
    """Generate knowledge graph embeddings using Neo4j GDS FastRP algorithm.
    
    FIXED: All GDS operations that mutate the graph now use run_write()
    instead of run_query() to ensure proper write transaction mode.
    """
    
    GRAPH_NAME = "graphweaver_kg"
    EMBEDDING_DIM = 128
    
    def __init__(self, neo4j_client):
        self.client = neo4j_client
    
    def check_gds_available(self) -> bool:
        """Check if Neo4j GDS plugin is available."""
        try:
            result = self.client.run_query("RETURN gds.version() as version")
            return bool(result)
        except:
            return False
    
    def drop_projection_if_exists(self):
        """Drop existing graph projection if it exists."""
        try:
            # Check if projection exists
            result = self.client.run_query(
                "CALL gds.graph.exists($name) YIELD exists RETURN exists",
                {"name": self.GRAPH_NAME}
            )
            if result and result[0].get("exists"):
                # FIXED: Use run_write for drop operation
                self.client.run_write(
                    "CALL gds.graph.drop($name) YIELD graphName RETURN graphName",
                    {"name": self.GRAPH_NAME}
                )
                print(f"[KGEmbedder] Dropped existing projection: {self.GRAPH_NAME}")
        except Exception as e:
            print(f"[KGEmbedder] Note: {e}")
    
    def create_projection(self) -> Dict[str, Any]:
        """Create a graph projection for embedding generation.
        
        FIXED: Uses run_write() because gds.graph.project mutates the catalog.
        """
        self.drop_projection_if_exists()
        
        # Project all nodes and relationships
        query = """
        CALL gds.graph.project(
            $name,
            ['Table', 'Column', 'Job', 'Dataset', 'DataSource'],
            {
                BELONGS_TO: {orientation: 'UNDIRECTED'},
                FK_TO: {orientation: 'UNDIRECTED'},
                READS: {orientation: 'UNDIRECTED'},
                WRITES: {orientation: 'UNDIRECTED'},
                REPRESENTS: {orientation: 'UNDIRECTED'}
            }
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """
        
        try:
            # FIXED: Use run_write for projection creation
            result = self.client.run_write(query, {"name": self.GRAPH_NAME})
            if result:
                stats = result[0]
                print(f"[KGEmbedder] Graph projected: {stats.get('nodeCount')} nodes, {stats.get('relationshipCount')} relationships")
                return {
                    "graphName": stats.get("graphName"),
                    "nodeCount": stats.get("nodeCount"),
                    "relationshipCount": stats.get("relationshipCount")
                }
        except Exception as e:
            print(f"[KGEmbedder] Projection failed: {e}")
            raise
        
        return {}
    
    def generate_fastrp_embeddings(self) -> Dict[str, Any]:
        """Generate FastRP embeddings and store them on nodes.
        
        FIXED: Uses run_write() because gds.fastRP.mutate writes to the graph.
        """
        query = """
        CALL gds.fastRP.mutate(
            $name,
            {
                embeddingDimension: $dim,
                mutateProperty: 'kg_embedding',
                randomSeed: 42,
                iterationWeights: [0.0, 1.0, 1.0]
            }
        )
        YIELD nodePropertiesWritten, computeMillis
        RETURN nodePropertiesWritten, computeMillis
        """
        
        print(f"[KGEmbedder] Generating FastRP embeddings (dim={self.EMBEDDING_DIM})...")
        
        try:
            # FIXED: Use run_write for FastRP mutate operation
            result = self.client.run_write(query, {
                "name": self.GRAPH_NAME,
                "dim": self.EMBEDDING_DIM
            })
            
            if result:
                stats = result[0]
                print(f"[KGEmbedder] FastRP complete: {stats.get('nodePropertiesWritten')} nodes embedded in {stats.get('computeMillis')}ms")
                return {
                    "nodes_embedded": stats.get("nodePropertiesWritten"),
                    "compute_ms": stats.get("computeMillis")
                }
        except Exception as e:
            print(f"[KGEmbedder] Failed to generate FastRP embeddings: {e}")
            raise RuntimeError(f"Failed to generate FastRP embeddings: {e}")
        
        return {}
    
    def write_embeddings_to_nodes(self) -> int:
        """Write embeddings from projection back to actual nodes.
        
        FIXED: Uses run_write() because this writes properties to nodes.
        """
        query = """
        CALL gds.graph.nodeProperty.stream($name, 'kg_embedding')
        YIELD nodeId, propertyValue
        WITH gds.util.asNode(nodeId) AS node, propertyValue AS embedding
        SET node.kg_embedding = embedding
        RETURN count(*) as written
        """
        
        try:
            # FIXED: Use run_write for writing embeddings to nodes
            result = self.client.run_write(query, {"name": self.GRAPH_NAME})
            if result:
                count = result[0].get("written", 0)
                print(f"[KGEmbedder] Wrote embeddings to {count} nodes")
                return count
        except Exception as e:
            print(f"[KGEmbedder] Failed to write embeddings: {e}")
        
        return 0
    
    def cleanup(self):
        """Drop the graph projection to free memory."""
        try:
            self.client.run_write(
                "CALL gds.graph.drop($name) YIELD graphName RETURN graphName",
                {"name": self.GRAPH_NAME}
            )
            print(f"[KGEmbedder] Graph projection dropped")
        except Exception as e:
            print(f"[KGEmbedder] Cleanup note: {e}")
    
    def generate_embeddings(self) -> Dict[str, Any]:
        """Full pipeline: project graph, generate embeddings, write to nodes.
        
        Returns stats about the embedding generation process.
        """
        stats = {
            "projection": {},
            "fastrp": {},
            "nodes_written": 0,
            "success": False
        }
        
        try:
            # Step 1: Create projection
            stats["projection"] = self.create_projection()
            
            # Step 2: Generate FastRP embeddings
            stats["fastrp"] = self.generate_fastrp_embeddings()
            
            # Step 3: Write embeddings to actual nodes
            stats["nodes_written"] = self.write_embeddings_to_nodes()
            
            stats["success"] = True
            
        except Exception as e:
            stats["error"] = str(e)
            print(f"[KGEmbedder] Embedding generation failed: {e}")
        
        finally:
            # Always cleanup projection
            self.cleanup()
        
        return stats


def generate_all_kg_embeddings(neo4j_client) -> Dict[str, Any]:
    """Generate knowledge graph embeddings for all entities.
    
    Args:
        neo4j_client: Neo4jClient instance
        
    Returns:
        Dictionary with embedding statistics
    """
    embedder = KGEmbedder(neo4j_client)
    
    # Check GDS availability
    if not embedder.check_gds_available():
        return {"error": "Neo4j GDS plugin not available"}
    
    print("[generate_all_kg_embeddings] Starting KG embedding generation...")
    
    # Generate embeddings
    stats = embedder.generate_embeddings()
    
    if stats.get("success"):
        print(f"[generate_all_kg_embeddings] Success: {stats.get('nodes_written', 0)} nodes embedded")
    else:
        print(f"[generate_all_kg_embeddings] Failed: {stats.get('error', 'Unknown error')}")
    
    return stats


def find_similar_nodes(neo4j_client, node_name: str, label: str = None, top_k: int = 5) -> List[Dict]:
    """Find nodes similar to the given node using KG embeddings.
    
    Args:
        neo4j_client: Neo4jClient instance
        node_name: Name of the node to find similar nodes for
        label: Optional label to filter by (Table, Column, Job, Dataset)
        top_k: Number of results to return
        
    Returns:
        List of similar nodes with similarity scores
    """
    # Build query based on whether label filter is provided
    if label:
        query = f"""
        MATCH (n:{label} {{name: $name}})
        WHERE n.kg_embedding IS NOT NULL
        MATCH (other:{label})
        WHERE other.kg_embedding IS NOT NULL AND other.name <> $name
        WITH n, other, gds.similarity.cosine(n.kg_embedding, other.kg_embedding) AS similarity
        ORDER BY similarity DESC
        LIMIT $top_k
        RETURN other.name AS name, labels(other)[0] AS label, similarity
        """
    else:
        query = """
        MATCH (n {name: $name})
        WHERE n.kg_embedding IS NOT NULL
        MATCH (other)
        WHERE other.kg_embedding IS NOT NULL AND other.name <> $name
        WITH n, other, gds.similarity.cosine(n.kg_embedding, other.kg_embedding) AS similarity
        ORDER BY similarity DESC
        LIMIT $top_k
        RETURN other.name AS name, labels(other)[0] AS label, similarity
        """
    
    try:
        results = neo4j_client.run_query(query, {"name": node_name, "top_k": top_k})
        return [
            {
                "name": r.get("name"),
                "label": r.get("label"),
                "similarity": r.get("similarity")
            }
            for r in results
        ]
    except Exception as e:
        print(f"[find_similar_nodes] Error: {e}")
        return []


def predict_fks_from_kg_embeddings(neo4j_client, threshold: float = 0.7, top_k: int = 20) -> List[Dict]:
    """Predict potential FK relationships using KG embedding similarity.
    
    Finds column pairs that are structurally similar in the graph
    but don't have an existing FK relationship.
    
    Args:
        neo4j_client: Neo4jClient instance
        threshold: Minimum similarity score
        top_k: Maximum number of predictions
        
    Returns:
        List of predicted FK relationships
    """
    query = """
    MATCH (c1:Column)-[:BELONGS_TO]->(t1:Table)
    MATCH (c2:Column)-[:BELONGS_TO]->(t2:Table)
    WHERE c1.kg_embedding IS NOT NULL 
      AND c2.kg_embedding IS NOT NULL
      AND t1.name <> t2.name
      AND NOT (c1)-[:FK_TO]-(c2)
      AND (c1.name CONTAINS 'id' OR c1.name CONTAINS '_id')
    WITH c1, t1, c2, t2, 
         gds.similarity.cosine(c1.kg_embedding, c2.kg_embedding) AS similarity
    WHERE similarity > $threshold
    ORDER BY similarity DESC
    LIMIT $top_k
    RETURN t1.name AS source_table, c1.name AS source_column,
           t2.name AS target_table, c2.name AS target_column,
           similarity
    """
    
    try:
        results = neo4j_client.run_query(query, {
            "threshold": threshold,
            "top_k": top_k
        })
        return [
            {
                "source_table": r.get("source_table"),
                "source_column": r.get("source_column"),
                "target_table": r.get("target_table"),
                "target_column": r.get("target_column"),
                "similarity": r.get("similarity")
            }
            for r in results
        ]
    except Exception as e:
        print(f"[predict_fks_from_kg_embeddings] Error: {e}")
        return []
