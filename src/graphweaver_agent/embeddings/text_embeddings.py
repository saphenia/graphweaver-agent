"""
Text Embeddings - Generate semantic embeddings for database metadata.

Uses sentence-transformers with all-MiniLM-L6-v2 (384 dimensions).
Embeddings are stored directly on Neo4j nodes for combined graph+vector queries.
"""
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    text: str
    embedding: List[float]
    model: str
    dimensions: int


class TextEmbedder:
    """Generate text embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.dimensions = 384 if "MiniLM" in model_name else 768
        
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            print(f"[TextEmbedder] Loading model: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                print(f"[TextEmbedder] Model loaded successfully")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def embed_text(self, text: str) -> EmbeddingResult:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return EmbeddingResult(
            text=text,
            embedding=embedding.tolist(),
            model=self.model_name,
            dimensions=len(embedding),
        )
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[EmbeddingResult]:
        if not texts:
            return []
        
        print(f"[TextEmbedder] Embedding {len(texts)} texts...")
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
        )
        
        results = []
        for text, emb in zip(texts, embeddings):
            results.append(EmbeddingResult(
                text=text,
                embedding=emb.tolist(),
                model=self.model_name,
                dimensions=len(emb),
            ))
        
        print(f"[TextEmbedder] Generated {len(results)} embeddings")
        return results
    
    def embed_column_metadata(
        self,
        table_name: str,
        column_name: str,
        data_type: Optional[str] = None,
        description: Optional[str] = None,
    ) -> EmbeddingResult:
        parts = [f"{table_name}.{column_name}"]
        
        col_lower = column_name.lower()
        if col_lower.endswith("_id") or col_lower == "id":
            parts.append("identifier reference key")
        if "email" in col_lower:
            parts.append("email address contact")
        if "name" in col_lower:
            parts.append("name label title")
        if "date" in col_lower or "time" in col_lower:
            parts.append("temporal datetime timestamp")
        if "price" in col_lower or "amount" in col_lower or "cost" in col_lower:
            parts.append("monetary price currency")
        if "count" in col_lower or "quantity" in col_lower or "qty" in col_lower:
            parts.append("numeric count quantity")
        if "status" in col_lower or "state" in col_lower:
            parts.append("status state enumeration")
        if "address" in col_lower:
            parts.append("location address geographic")
        if "phone" in col_lower or "tel" in col_lower:
            parts.append("phone telephone contact")
        if "url" in col_lower or "link" in col_lower:
            parts.append("url link web")
        if "flag" in col_lower or "is_" in col_lower or "has_" in col_lower:
            parts.append("boolean flag indicator")
            
        if data_type:
            parts.append(f"type:{data_type}")
            
        if description:
            parts.append(description)
            
        text = " ".join(parts)
        return self.embed_text(text)
    
    def embed_table_metadata(
        self,
        table_name: str,
        column_names: List[str],
        description: Optional[str] = None,
    ) -> EmbeddingResult:
        parts = [table_name]
        
        tbl_lower = table_name.lower()
        if "order" in tbl_lower:
            parts.append("transaction purchase sales")
        if "customer" in tbl_lower or "user" in tbl_lower:
            parts.append("person customer user account")
        if "product" in tbl_lower or "item" in tbl_lower:
            parts.append("product item catalog inventory")
        if "payment" in tbl_lower:
            parts.append("payment transaction financial")
        if "log" in tbl_lower or "history" in tbl_lower or "audit" in tbl_lower:
            parts.append("log history audit tracking")
        if "config" in tbl_lower or "setting" in tbl_lower:
            parts.append("configuration settings system")
            
        col_text = " ".join(column_names[:20])
        parts.append(f"columns: {col_text}")
        
        if description:
            parts.append(description)
            
        text = " ".join(parts)
        return self.embed_text(text)
    
    def embed_business_rule(
        self,
        name: str,
        description: str,
        sql: str,
        inputs: List[str],
        outputs: List[str],
    ) -> EmbeddingResult:
        parts = [
            name,
            description,
            f"inputs: {' '.join(inputs)}",
            f"outputs: {' '.join(outputs)}",
            sql[:500] if len(sql) > 500 else sql,
        ]
        text = " ".join(parts)
        return self.embed_text(text)
    
    def similarity(self, emb1: List[float], emb2: List[float]) -> float:
        a = np.array(emb1)
        b = np.array(emb2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed_all_metadata(
    neo4j_client,
    pg_connector,
    embedder: Optional[TextEmbedder] = None,
) -> Dict[str, int]:
    """Embed all database metadata and store in Neo4j nodes."""
    if embedder is None:
        embedder = TextEmbedder()
    
    stats = {"tables": 0, "columns": 0, "jobs": 0, "datasets": 0}
    
    tables = pg_connector.get_tables_with_info()
    print(f"[embed_all_metadata] Processing {len(tables)} tables...")
    
    for table_info in tables:
        table_name = table_info["table_name"]
        
        try:
            meta = pg_connector.get_table_metadata(table_name)
            column_names = [c.column_name for c in meta.columns]
            
            table_emb = embedder.embed_table_metadata(table_name, column_names)
            
            neo4j_client.run_write("""
                MERGE (t:Table {name: $name})
                SET t.text_embedding = $embedding,
                    t.embedding_model = $model
            """, {
                "name": table_name,
                "embedding": table_emb.embedding,
                "model": table_emb.model,
            })
            stats["tables"] += 1
            print(f"[embed_all_metadata] Embedded table: {table_name}")
            
            for col in meta.columns:
                col_emb = embedder.embed_column_metadata(
                    table_name=table_name,
                    column_name=col.column_name,
                    data_type=col.data_type.value if col.data_type else None,
                )
                
                neo4j_client.run_write("""
                    MERGE (t:Table {name: $table_name})
                    MERGE (c:Column {name: $col_name, table: $table_name})
                    MERGE (c)-[:BELONGS_TO]->(t)
                    SET c.text_embedding = $embedding,
                        c.embedding_model = $model
                """, {
                    "col_name": col.column_name,
                    "table_name": table_name,
                    "embedding": col_emb.embedding,
                    "model": col_emb.model,
                })
                stats["columns"] += 1
                
        except Exception as e:
            print(f"[embed_all_metadata] Error processing {table_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    jobs = neo4j_client.run_query("MATCH (j:Job) RETURN j.name as name, j.description as desc")
    if jobs:
        print(f"[embed_all_metadata] Processing {len(jobs)} jobs...")
        for job in jobs:
            text = f"{job['name']} {job.get('desc', '')}"
            job_emb = embedder.embed_text(text)
            
            neo4j_client.run_write("""
                MERGE (j:Job {name: $name})
                SET j.text_embedding = $embedding,
                    j.embedding_model = $model
            """, {
                "name": job["name"],
                "embedding": job_emb.embedding,
                "model": job_emb.model,
            })
            stats["jobs"] += 1
    
    datasets = neo4j_client.run_query("MATCH (d:Dataset) RETURN d.name as name")
    if datasets:
        print(f"[embed_all_metadata] Processing {len(datasets)} datasets...")
        for ds in datasets:
            ds_emb = embedder.embed_text(ds["name"])
            
            neo4j_client.run_write("""
                MERGE (d:Dataset {name: $name})
                SET d.text_embedding = $embedding,
                    d.embedding_model = $model
            """, {
                "name": ds["name"],
                "embedding": ds_emb.embedding,
                "model": ds_emb.model,
            })
            stats["datasets"] += 1
    
    print(f"[embed_all_metadata] Complete: {stats}")
    return stats


# =============================================================================
# Standalone search functions (used by streamlit_app.py)
# =============================================================================

def search_tables(neo4j_client, embedder: TextEmbedder, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search tables by semantic similarity to query."""
    query_emb = embedder.embed_text(query)
    
    results = neo4j_client.run_query("""
        MATCH (t:Table)
        WHERE t.text_embedding IS NOT NULL
        WITH t, gds.similarity.cosine(t.text_embedding, $embedding) AS score
        WHERE score > 0.3
        RETURN t.name AS name, score
        ORDER BY score DESC
        LIMIT $top_k
    """, {"embedding": query_emb.embedding, "top_k": top_k})
    
    return [dict(r) for r in results] if results else []


def search_columns(neo4j_client, embedder: TextEmbedder, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Search columns by semantic similarity to query."""
    query_emb = embedder.embed_text(query)
    
    results = neo4j_client.run_query("""
        MATCH (c:Column)-[:BELONGS_TO]->(t:Table)
        WHERE c.text_embedding IS NOT NULL
        WITH t, c, gds.similarity.cosine(c.text_embedding, $embedding) AS score
        WHERE score > 0.3
        RETURN t.name AS table, c.name AS name, score
        ORDER BY score DESC
        LIMIT $top_k
    """, {"embedding": query_emb.embedding, "top_k": top_k})
    
    return [dict(r) for r in results] if results else []


def find_similar_tables(neo4j_client, table_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Find tables similar to a given table."""
    results = neo4j_client.run_query("""
        MATCH (source:Table {name: $table_name})
        WHERE source.text_embedding IS NOT NULL
        MATCH (other:Table)
        WHERE other.name <> $table_name AND other.text_embedding IS NOT NULL
        WITH other, gds.similarity.cosine(source.text_embedding, other.text_embedding) AS score
        WHERE score > 0.3
        RETURN other.name AS name, score
        ORDER BY score DESC
        LIMIT $top_k
    """, {"table_name": table_name, "top_k": top_k})
    
    return [dict(r) for r in results] if results else []


def find_similar_columns(neo4j_client, table_name: str, column_name: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Find columns similar to a given column."""
    results = neo4j_client.run_query("""
        MATCH (source:Column {name: $column_name, table: $table_name})
        WHERE source.text_embedding IS NOT NULL
        MATCH (other:Column)-[:BELONGS_TO]->(t:Table)
        WHERE (other.name <> $column_name OR other.table <> $table_name) 
          AND other.text_embedding IS NOT NULL
        WITH t, other, gds.similarity.cosine(source.text_embedding, other.text_embedding) AS score
        WHERE score > 0.3
        RETURN t.name AS table, other.name AS name, score
        ORDER BY score DESC
        LIMIT $top_k
    """, {"table_name": table_name, "column_name": column_name, "top_k": top_k})
    
    return [dict(r) for r in results] if results else []
