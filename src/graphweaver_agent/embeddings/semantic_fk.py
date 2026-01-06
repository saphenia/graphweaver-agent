"""
Semantic FK Discovery - Use embeddings to find FK relationships
even when column names don't match.

This complements the statistical FK discovery by finding semantic matches like:
- customer_id ↔ buyer_identifier
- prod_code ↔ item_sku  
- created_at ↔ timestamp_created
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SemanticFKCandidate:
    """A candidate FK discovered through semantic similarity."""
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    semantic_similarity: float
    kg_similarity: Optional[float] = None
    combined_score: float = 0.0
    name_similarity: float = 0.0
    recommendation: str = ""


class SemanticFKDiscovery:
    """Discover FK relationships using embedding similarity."""
    
    def __init__(
        self,
        neo4j_client,
        text_embedder=None,
        min_semantic_score: float = 0.7,
        min_combined_score: float = 0.6,
    ):
        """
        Initialize semantic FK discovery.
        
        Args:
            neo4j_client: Neo4j client
            text_embedder: TextEmbedder instance (optional)
            min_semantic_score: Minimum text embedding similarity
            min_combined_score: Minimum combined score threshold
        """
        self.neo4j = neo4j_client
        self.text_embedder = text_embedder
        self.min_semantic_score = min_semantic_score
        self.min_combined_score = min_combined_score
        
    def find_semantic_fk_candidates(
        self,
        source_table: Optional[str] = None,
        top_k: int = 50,
    ) -> List[SemanticFKCandidate]:
        """
        Find potential FK relationships using semantic similarity.
        
        Looks for columns that:
        1. Have high text embedding similarity (semantically related names)
        2. Don't already have FK relationships
        3. Target columns are likely PKs (high uniqueness)
        
        Args:
            source_table: Limit search to this source table (optional)
            top_k: Max candidates to return
            
        Returns:
            List of SemanticFKCandidate
        """
        where_clause = ""
        params = {"top_k": top_k, "min_score": self.min_semantic_score}
        
        if source_table:
            where_clause = "AND t1.name = $source_table"
            params["source_table"] = source_table
        
        # Find semantically similar column pairs without FK
        result = self.neo4j.run_query(f"""
            // Find columns with ID-like patterns (potential FKs)
            MATCH (c1:Column)-[:BELONGS_TO]->(t1:Table)
            WHERE (c1.name ENDS WITH '_id' OR c1.name ENDS WITH '_code' OR c1.name ENDS WITH '_key')
              AND c1.text_embedding IS NOT NULL
              {where_clause}
            
            // Find potential target columns (likely PKs - id columns)
            MATCH (c2:Column)-[:BELONGS_TO]->(t2:Table)
            WHERE (c2.name = 'id' OR c2.name ENDS WITH '_id')
              AND c2.text_embedding IS NOT NULL
              AND t1 <> t2
              AND NOT (c1)-[:FK_TO]->(c2)
              AND NOT (c2)-[:FK_TO]->(c1)
            
            // Calculate similarity
            WITH t1, c1, t2, c2,
                 gds.similarity.cosine(c1.text_embedding, c2.text_embedding) AS text_sim
            WHERE text_sim > $min_score
            
            // Add KG similarity if available
            WITH t1, c1, t2, c2, text_sim,
                 CASE 
                   WHEN c1.kg_embedding IS NOT NULL AND c2.kg_embedding IS NOT NULL
                   THEN gds.similarity.cosine(c1.kg_embedding, c2.kg_embedding)
                   ELSE null
                 END AS kg_sim
            
            RETURN t1.name AS source_table,
                   c1.name AS source_column,
                   t2.name AS target_table,
                   c2.name AS target_column,
                   text_sim,
                   kg_sim,
                   CASE 
                     WHEN kg_sim IS NOT NULL 
                     THEN (text_sim * 0.6 + kg_sim * 0.4)
                     ELSE text_sim
                   END AS combined_score
            ORDER BY combined_score DESC
            LIMIT $top_k
        """, params)
        
        candidates = []
        if result:
            for r in result:
                # Calculate name similarity for additional context
                name_sim = self._name_similarity(
                    r["source_column"], 
                    r["target_table"], 
                    r["target_column"]
                )
                
                combined = r["combined_score"]
                if name_sim > 0.5:
                    combined = combined * 0.8 + name_sim * 0.2
                
                recommendation = self._get_recommendation(combined, r["text_sim"], name_sim)
                
                candidates.append(SemanticFKCandidate(
                    source_table=r["source_table"],
                    source_column=r["source_column"],
                    target_table=r["target_table"],
                    target_column=r["target_column"],
                    semantic_similarity=r["text_sim"],
                    kg_similarity=r["kg_sim"],
                    combined_score=combined,
                    name_similarity=name_sim,
                    recommendation=recommendation,
                ))
        
        # Filter by combined score
        candidates = [c for c in candidates if c.combined_score >= self.min_combined_score]
        
        return candidates
    
    def _name_similarity(
        self, 
        source_column: str, 
        target_table: str, 
        target_column: str
    ) -> float:
        """Calculate name-based similarity score."""
        source_lower = source_column.lower()
        target_table_lower = target_table.lower()
        
        # Check if source column contains target table name
        # e.g., customer_id -> customers table
        table_singular = target_table_lower.rstrip('s')
        
        if table_singular in source_lower:
            return 0.9
        if target_table_lower in source_lower:
            return 0.85
            
        # Check common patterns
        source_base = source_lower.replace('_id', '').replace('_code', '').replace('_key', '')
        if source_base == table_singular:
            return 0.95
        if source_base in target_table_lower or target_table_lower in source_base:
            return 0.7
            
        return 0.0
    
    def _get_recommendation(
        self, 
        combined: float, 
        semantic: float, 
        name: float
    ) -> str:
        """Generate recommendation based on scores."""
        if combined >= 0.85:
            return "HIGH CONFIDENCE - Strong semantic and structural match"
        elif combined >= 0.7:
            if name > 0.5:
                return "LIKELY FK - Good semantic match with naming pattern"
            else:
                return "POSSIBLE FK - Strong semantic match, verify with data"
        elif combined >= 0.6:
            return "INVESTIGATE - Moderate match, needs data validation"
        else:
            return "LOW CONFIDENCE - Weak match"
    
    def find_columns_for_concept(
        self,
        concept: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find columns matching a concept using semantic search.
        
        Examples:
        - "customer identifier" -> customer_id, buyer_id, client_code
        - "monetary amount" -> price, cost, total_amount, payment
        - "creation timestamp" -> created_at, timestamp, date_created
        
        Args:
            concept: Natural language concept
            top_k: Number of results
            
        Returns:
            List of matching columns with scores
        """
        if self.text_embedder is None:
            from .text_embeddings import TextEmbedder
            self.text_embedder = TextEmbedder()
        
        # Embed the concept
        concept_emb = self.text_embedder.embed_text(concept)
        
        # Search using vector index
        result = self.neo4j.run_query("""
            CALL db.index.vector.queryNodes('column_text_embedding', $top_k, $embedding)
            YIELD node, score
            MATCH (node)-[:BELONGS_TO]->(t:Table)
            RETURN t.name AS table_name, node.name AS column_name, score
            ORDER BY score DESC
        """, {"embedding": concept_emb.embedding, "top_k": top_k})
        
        return [dict(r) for r in result] if result else []
    
    def find_tables_for_concept(
        self,
        concept: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find tables matching a concept using semantic search.
        
        Examples:
        - "customer data" -> customers, users, accounts
        - "financial transactions" -> orders, payments, invoices
        
        Args:
            concept: Natural language concept
            top_k: Number of results
            
        Returns:
            List of matching tables with scores
        """
        if self.text_embedder is None:
            from .text_embeddings import TextEmbedder
            self.text_embedder = TextEmbedder()
        
        concept_emb = self.text_embedder.embed_text(concept)
        
        result = self.neo4j.run_query("""
            CALL db.index.vector.queryNodes('table_text_embedding', $top_k, $embedding)
            YIELD node, score
            RETURN node.name AS table_name, score
            ORDER BY score DESC
        """, {"embedding": concept_emb.embedding, "top_k": top_k})
        
        return [dict(r) for r in result] if result else []
