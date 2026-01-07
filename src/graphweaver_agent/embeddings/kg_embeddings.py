"""
Knowledge Graph Embeddings - Generate structural embeddings using Neo4j GDS.

Uses FastRP (Fast Random Projection) algorithm which is fast and effective
for link prediction and node similarity tasks.

FIXED: Better GDS availability checking, graceful fallback, and error handling.
"""
from typing import Dict, List, Any, Optional
import traceback


class KGEmbedder:
    """Generate knowledge graph embeddings using Neo4j GDS FastRP."""
    
    def __init__(self, neo4j_client, embedding_dimension: int = 128):
        """
        Initialize KG embedder.
        
        Args:
            neo4j_client: Neo4j client instance
            embedding_dimension: Dimension of output embeddings (default 128)
        """
        self.neo4j = neo4j_client
        self.embedding_dimension = embedding_dimension
        self.graph_name = "graphweaver_kg"
        self._gds_available = None
        self._gds_version = None
        
    def check_gds_available(self) -> bool:
        """Check if Neo4j GDS is available."""
        if self._gds_available is not None:
            return self._gds_available
            
        try:
            result = self.neo4j.run_query("RETURN gds.version() as version")
            if result:
                self._gds_version = result[0]['version']
                self._gds_available = True
                print(f"[KGEmbedder] Neo4j GDS version: {self._gds_version}")
                return True
        except Exception as e:
            print(f"[KGEmbedder] GDS not available: {e}")
            self._gds_available = False
        return False
    
    def get_gds_status(self) -> Dict[str, Any]:
        """Get detailed GDS status information."""
        status = {
            "available": self.check_gds_available(),
            "version": self._gds_version,
        }
        
        if not status["available"]:
            status["error"] = "Neo4j GDS plugin not installed or not accessible"
            status["fix"] = (
                "Install Neo4j GDS plugin or use neo4j:5-enterprise Docker image. "
                "See: https://neo4j.com/docs/graph-data-science/current/installation/"
            )
        
        return status
    
    def create_graph_projection(self) -> Dict[str, Any]:
        """
        Create an in-memory graph projection for embedding generation.
        
        Projects all node types and relationship types into memory.
        """
        if not self.check_gds_available():
            return {"error": "GDS not available"}
        
        # Drop existing projection if exists
        try:
            self.neo4j.run_write(f"CALL gds.graph.drop('{self.graph_name}', false)")
            print(f"[KGEmbedder] Dropped existing graph projection")
        except Exception:
            pass  # Graph didn't exist
        
        # First, check what node labels exist
        try:
            existing_labels = self.neo4j.run_query("""
                CALL db.labels() YIELD label
                WHERE label IN ['Table', 'Column', 'Job', 'Dataset']
                RETURN collect(label) as labels
            """)
            available_labels = existing_labels[0]['labels'] if existing_labels else []
            print(f"[KGEmbedder] Available labels: {available_labels}")
            
            if not available_labels:
                return {"error": "No Table, Column, Job, or Dataset nodes found in graph"}
        except Exception as e:
            print(f"[KGEmbedder] Error checking labels: {e}")
            available_labels = ['Table', 'Column', 'Job', 'Dataset']
        
        # Check what relationship types exist
        try:
            existing_rels = self.neo4j.run_query("""
                CALL db.relationshipTypes() YIELD relationshipType
                WHERE relationshipType IN ['BELONGS_TO', 'FK_TO', 'READS', 'WRITES', 'REPRESENTS']
                RETURN collect(relationshipType) as types
            """)
            available_rels = existing_rels[0]['types'] if existing_rels else []
            print(f"[KGEmbedder] Available relationship types: {available_rels}")
            
            if not available_rels:
                return {"error": "No relationships found in graph. Run FK discovery first."}
        except Exception as e:
            print(f"[KGEmbedder] Error checking relationships: {e}")
            available_rels = ['BELONGS_TO', 'FK_TO']
        
        # Build relationship config dynamically
        rel_config = {}
        for rel in available_rels:
            rel_config[rel] = {"orientation": "UNDIRECTED"}
        
        # Create new projection with available labels and relationships
        try:
            result = self.neo4j.run_query(f"""
                CALL gds.graph.project(
                    '{self.graph_name}',
                    {available_labels},
                    $rel_config
                )
                YIELD graphName, nodeCount, relationshipCount
                RETURN graphName, nodeCount, relationshipCount
            """, {"rel_config": rel_config})
            
            if result:
                stats = result[0]
                print(f"[KGEmbedder] Graph projected: {stats['nodeCount']} nodes, {stats['relationshipCount']} relationships")
                return {
                    "graphName": stats["graphName"],
                    "nodeCount": stats["nodeCount"],
                    "relationshipCount": stats["relationshipCount"],
                }
        except Exception as e:
            error_msg = f"Failed to create graph projection: {e}"
            print(f"[KGEmbedder] {error_msg}")
            traceback.print_exc()
            return {"error": error_msg}
        
        return {"error": "Unknown error creating graph projection"}
    
    def generate_fastrp_embeddings(
        self,
        iterations: int = 4,
        normalization_strength: float = 0.0,
        property_name: str = "kg_embedding",
    ) -> Dict[str, Any]:
        """
        Generate FastRP embeddings and write to nodes.
        
        FastRP is a fast algorithm that uses random projection to create
        embeddings that capture graph structure (neighborhood similarity).
        
        Args:
            iterations: Number of iterations (more = captures longer paths)
            normalization_strength: L2 normalization (-1 to 1)
            property_name: Name of the property to store embeddings
            
        Returns:
            Statistics dict
        """
        if not self.check_gds_available():
            return {"error": "GDS not available"}
        
        print(f"[KGEmbedder] Generating FastRP embeddings (dim={self.embedding_dimension})...")
        
        try:
            # Run FastRP and write back to graph
            result = self.neo4j.run_query(f"""
                CALL gds.fastRP.write(
                    '{self.graph_name}',
                    {{
                        embeddingDimension: {self.embedding_dimension},
                        iterationWeights: [0.0, 1.0, 1.0, 1.0],
                        normalizationStrength: {normalization_strength},
                        writeProperty: '{property_name}'
                    }}
                )
                YIELD nodeCount, nodePropertiesWritten
                RETURN nodeCount, nodePropertiesWritten
            """)
            
            if result:
                stats = result[0]
                print(f"[KGEmbedder] Embeddings written: {stats['nodePropertiesWritten']} properties on {stats['nodeCount']} nodes")
                return {
                    "nodes": stats["nodeCount"],
                    "properties_written": stats["nodePropertiesWritten"],
                }
        except Exception as e:
            error_msg = f"Failed to generate FastRP embeddings: {e}"
            print(f"[KGEmbedder] {error_msg}")
            traceback.print_exc()
            return {"error": error_msg}
        
        return {"error": "Unknown error generating embeddings"}
    
    def generate_node2vec_embeddings(
        self,
        walk_length: int = 80,
        walks_per_node: int = 10,
        property_name: str = "kg_embedding_n2v",
    ) -> Dict[str, int]:
        """
        Generate Node2Vec embeddings (alternative to FastRP).
        
        Node2Vec uses biased random walks to learn embeddings.
        Better for some tasks but slower than FastRP.
        
        Args:
            walk_length: Length of random walks
            walks_per_node: Number of walks per node
            property_name: Name of the property to store embeddings
            
        Returns:
            Statistics dict
        """
        if not self.check_gds_available():
            return {"error": "GDS not available"}
        
        print(f"[KGEmbedder] Generating Node2Vec embeddings (dim={self.embedding_dimension})...")
        
        try:
            result = self.neo4j.run_query(f"""
                CALL gds.node2vec.write(
                    '{self.graph_name}',
                    {{
                        embeddingDimension: {self.embedding_dimension},
                        walkLength: {walk_length},
                        walksPerNode: {walks_per_node},
                        writeProperty: '{property_name}'
                    }}
                )
                YIELD nodeCount, nodePropertiesWritten
                RETURN nodeCount, nodePropertiesWritten
            """)
            
            if result:
                stats = result[0]
                print(f"[KGEmbedder] Node2Vec embeddings written: {stats['nodePropertiesWritten']} properties")
                return {
                    "nodes": stats["nodeCount"],
                    "properties_written": stats["nodePropertiesWritten"],
                }
        except Exception as e:
            error_msg = f"Failed to generate Node2Vec embeddings: {e}"
            print(f"[KGEmbedder] {error_msg}")
            return {"error": error_msg}
        
        return {"error": "Unknown error generating Node2Vec embeddings"}
    
    def find_similar_nodes(
        self,
        node_label: str,
        node_name: str,
        property_name: str = "kg_embedding",
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find nodes most similar to a given node using KG embeddings.
        
        Args:
            node_label: Label of the source node (Table, Column, etc.)
            node_name: Name of the source node
            property_name: Embedding property to use
            top_k: Number of results to return
            
        Returns:
            List of similar nodes with similarity scores
        """
        if not self.check_gds_available():
            return []
        
        try:
            result = self.neo4j.run_query(f"""
                MATCH (source:{node_label} {{name: $name}})
                MATCH (other:{node_label})
                WHERE other <> source AND other.{property_name} IS NOT NULL
                WITH source, other, 
                     gds.similarity.cosine(source.{property_name}, other.{property_name}) AS similarity
                RETURN other.name AS name, similarity
                ORDER BY similarity DESC
                LIMIT $top_k
            """, {"name": node_name, "top_k": top_k})
            
            return [dict(r) for r in result] if result else []
        except Exception as e:
            print(f"[KGEmbedder] Error finding similar nodes: {e}")
            return []
    
    def predict_missing_links(
        self,
        source_label: str = "Column",
        target_label: str = "Column",
        property_name: str = "kg_embedding",
        threshold: float = 0.8,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Predict potential missing FK relationships using embedding similarity.
        
        Finds column pairs that are structurally similar (in graph space)
        but don't have a direct FK relationship.
        
        Args:
            source_label: Label for source nodes
            target_label: Label for target nodes
            property_name: Embedding property to use
            threshold: Minimum similarity threshold
            top_k: Max results to return
            
        Returns:
            List of predicted links with scores
        """
        if not self.check_gds_available():
            return []
        
        try:
            result = self.neo4j.run_query(f"""
                // Find column pairs without existing FK
                MATCH (c1:{source_label})-[:BELONGS_TO]->(t1:Table)
                MATCH (c2:{target_label})-[:BELONGS_TO]->(t2:Table)
                WHERE t1 <> t2 
                  AND c1 <> c2
                  AND c1.{property_name} IS NOT NULL 
                  AND c2.{property_name} IS NOT NULL
                  AND NOT (c1)-[:FK_TO]->(c2)
                  AND NOT (c2)-[:FK_TO]->(c1)
                WITH c1, c2, t1, t2,
                     gds.similarity.cosine(c1.{property_name}, c2.{property_name}) AS similarity
                WHERE similarity > $threshold
                RETURN t1.name AS source_table, 
                       c1.name AS source_column,
                       t2.name AS target_table,
                       c2.name AS target_column,
                       similarity
                ORDER BY similarity DESC
                LIMIT $top_k
            """, {"threshold": threshold, "top_k": top_k})
            
            return [dict(r) for r in result] if result else []
        except Exception as e:
            print(f"[KGEmbedder] Error predicting missing links: {e}")
            return []
    
    def get_embedding_stats(self, property_name: str = "kg_embedding") -> Dict[str, Any]:
        """Get statistics about generated embeddings."""
        try:
            result = self.neo4j.run_query(f"""
                MATCH (n)
                WHERE n.{property_name} IS NOT NULL
                WITH labels(n) AS lbls, count(n) AS cnt
                RETURN lbls[0] AS label, cnt AS count
            """)
            
            stats = {}
            total = 0
            if result:
                for r in result:
                    stats[r["label"]] = r["count"]
                    total += r["count"]
            stats["total"] = total
            
            return stats
        except Exception as e:
            print(f"[KGEmbedder] Error getting embedding stats: {e}")
            return {"error": str(e)}
    
    def drop_graph_projection(self):
        """Drop the in-memory graph projection to free memory."""
        try:
            self.neo4j.run_write(f"CALL gds.graph.drop('{self.graph_name}', false)")
            print(f"[KGEmbedder] Graph projection dropped")
        except Exception:
            pass  # Already dropped or didn't exist


def generate_all_kg_embeddings(neo4j_client) -> Dict[str, Any]:
    """
    Convenience function to generate KG embeddings for the entire graph.
    
    FIXED: Better error handling and detailed status reporting.
    
    Args:
        neo4j_client: Neo4j client instance
        
    Returns:
        Statistics dict with detailed status
    """
    embedder = KGEmbedder(neo4j_client)
    
    # Check GDS availability
    gds_status = embedder.get_gds_status()
    if not gds_status["available"]:
        print(f"[generate_all_kg_embeddings] GDS not available")
        return {
            "error": "Neo4j GDS plugin not available",
            "gds_status": gds_status,
            "fix": gds_status.get("fix", "Install Neo4j GDS plugin"),
        }
    
    # Create projection
    print(f"[generate_all_kg_embeddings] Creating graph projection...")
    projection_stats = embedder.create_graph_projection()
    if "error" in projection_stats:
        return {
            "error": f"Failed to create graph projection: {projection_stats['error']}",
            "gds_status": gds_status,
        }
    
    # Check if we have enough nodes
    if projection_stats.get("nodeCount", 0) < 2:
        embedder.drop_graph_projection()
        return {
            "error": "Not enough nodes for embedding generation (need at least 2)",
            "projection": projection_stats,
        }
    
    # Check if we have any relationships
    if projection_stats.get("relationshipCount", 0) == 0:
        embedder.drop_graph_projection()
        return {
            "error": "No relationships found. Run FK discovery first to create relationships.",
            "projection": projection_stats,
        }
    
    # Generate embeddings
    print(f"[generate_all_kg_embeddings] Generating FastRP embeddings...")
    embedding_stats = embedder.generate_fastrp_embeddings()
    
    # Clean up
    embedder.drop_graph_projection()
    
    if "error" in embedding_stats:
        return {
            "error": f"Failed to generate embeddings: {embedding_stats['error']}",
            "projection": projection_stats,
        }
    
    # Get final stats
    final_stats = embedder.get_embedding_stats()
    
    result = {
        "success": True,
        "projection": projection_stats,
        "embeddings": embedding_stats,
        "coverage": final_stats,
    }
    
    print(f"[generate_all_kg_embeddings] Complete: {result}")
    return result


def verify_kg_embeddings(neo4j_client) -> Dict[str, Any]:
    """
    Verify KG embeddings are properly stored.
    
    Returns detailed coverage statistics.
    """
    embedder = KGEmbedder(neo4j_client)
    
    result = {
        "gds_available": embedder.check_gds_available(),
        "gds_version": embedder._gds_version,
    }
    
    # Check embedding coverage
    try:
        coverage = neo4j_client.run_query("""
            MATCH (n)
            WHERE n:Table OR n:Column OR n:Job OR n:Dataset
            WITH labels(n)[0] AS label,
                 n.kg_embedding IS NOT NULL AS has_emb
            RETURN label,
                   count(*) AS total,
                   sum(CASE WHEN has_emb THEN 1 ELSE 0 END) AS with_embedding
            ORDER BY label
        """)
        
        result["coverage"] = {}
        total_nodes = 0
        total_with_emb = 0
        
        for row in coverage:
            label = row["label"]
            result["coverage"][label] = {
                "total": row["total"],
                "with_embedding": row["with_embedding"],
                "percentage": f"{100 * row['with_embedding'] / row['total']:.1f}%" if row["total"] > 0 else "N/A"
            }
            total_nodes += row["total"]
            total_with_emb += row["with_embedding"]
        
        result["summary"] = {
            "total_nodes": total_nodes,
            "with_kg_embedding": total_with_emb,
            "overall_coverage": f"{100 * total_with_emb / total_nodes:.1f}%" if total_nodes > 0 else "N/A"
        }
        
    except Exception as e:
        result["error"] = f"Failed to check coverage: {e}"
    
    return result