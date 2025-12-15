"""
Knowledge Graph Embeddings - Generate structural embeddings using Neo4j GDS.

Uses FastRP (Fast Random Projection) algorithm which is fast and effective
for link prediction and node similarity tasks.
"""
from typing import Dict, List, Any, Optional


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
        
    def check_gds_available(self) -> bool:
        """Check if Neo4j GDS is available."""
        try:
            result = self.neo4j.run_query("RETURN gds.version() as version")
            if result:
                print(f"[KGEmbedder] Neo4j GDS version: {result[0]['version']}")
                return True
        except Exception as e:
            print(f"[KGEmbedder] GDS not available: {e}")
        return False
    
    def create_graph_projection(self) -> Dict[str, Any]:
        """
        Create an in-memory graph projection for embedding generation.
        
        Projects all node types and relationship types into memory.
        """
        # Drop existing projection if exists
        try:
            self.neo4j.run_write(f"CALL gds.graph.drop('{self.graph_name}', false)")
        except:
            pass
        
        # Create new projection with all relationships
        result = self.neo4j.run_query(f"""
            CALL gds.graph.project(
                '{self.graph_name}',
                ['Table', 'Column', 'Job', 'Dataset'],
                {{
                    BELONGS_TO: {{orientation: 'UNDIRECTED'}},
                    FK_TO: {{orientation: 'UNDIRECTED'}},
                    READS: {{orientation: 'UNDIRECTED'}},
                    WRITES: {{orientation: 'UNDIRECTED'}},
                    REPRESENTS: {{orientation: 'UNDIRECTED'}}
                }}
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
        """)
        
        if result:
            stats = result[0]
            print(f"[KGEmbedder] Graph projected: {stats['nodeCount']} nodes, {stats['relationshipCount']} relationships")
            return dict(stats)
        return {}
    
    def generate_fastrp_embeddings(
        self,
        iterations: int = 4,
        normalization_strength: float = 0.0,
        property_name: str = "kg_embedding",
    ) -> Dict[str, int]:
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
        print(f"[KGEmbedder] Generating FastRP embeddings (dim={self.embedding_dimension})...")
        
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
        return {}
    
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
        print(f"[KGEmbedder] Generating Node2Vec embeddings (dim={self.embedding_dimension})...")
        
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
        return {}
    
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
    
    def get_embedding_stats(self, property_name: str = "kg_embedding") -> Dict[str, Any]:
        """Get statistics about generated embeddings."""
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
    
    def drop_graph_projection(self):
        """Drop the in-memory graph projection to free memory."""
        try:
            self.neo4j.run_write(f"CALL gds.graph.drop('{self.graph_name}', false)")
            print(f"[KGEmbedder] Graph projection dropped")
        except:
            pass


def generate_all_kg_embeddings(neo4j_client) -> Dict[str, Any]:
    """
    Convenience function to generate KG embeddings for the entire graph.
    
    Args:
        neo4j_client: Neo4j client instance
        
    Returns:
        Statistics dict
    """
    embedder = KGEmbedder(neo4j_client)
    
    # Check GDS availability
    if not embedder.check_gds_available():
        return {"error": "Neo4j GDS plugin not available"}
    
    # Create projection
    projection_stats = embedder.create_graph_projection()
    if not projection_stats:
        return {"error": "Failed to create graph projection"}
    
    # Generate embeddings
    embedding_stats = embedder.generate_fastrp_embeddings()
    
    # Clean up
    embedder.drop_graph_projection()
    
    return {
        "projection": projection_stats,
        "embeddings": embedding_stats,
    }