"""RDF Sync Manager - Synchronize Neo4j graph to RDF triple store.

FIXED: Added proper error handling, logging, and return value checking.
FIXED: URL-encode graph URI in GSP requests.
"""
from typing import Dict, List, Any, Optional
from .fuseki_client import FusekiClient
from .ontology import PREFIXES, GraphWeaverOntology, make_table_uri, make_column_uri, make_job_uri, make_dataset_uri


def _banner(msg: str):
    """Print unmissable banner."""
    print("=" * 60)
    print(f"  {msg}")
    print("=" * 60)


class RDFSyncManager:
    def __init__(self, fuseki_client: FusekiClient, neo4j_client=None):
        self.fuseki = fuseki_client
        self.neo4j = neo4j_client
        self.graph_uri = "http://graphweaver.io/graph/main"

    def initialize_store(self) -> Dict[str, Any]:
        print(f"[RDFSyncManager] Initializing store, graph: {self.graph_uri}")
        if not self.fuseki.ensure_dataset_exists():
            print("[RDFSyncManager] ERROR: Failed to create dataset")
            return {"success": False, "error": "Failed to create dataset"}
        
        print("[RDFSyncManager] Clearing existing graph...")
        self.fuseki.clear_graph(self.graph_uri)
        
        print("[RDFSyncManager] Inserting ontology...")
        ontology_turtle = GraphWeaverOntology.get_ontology_turtle()
        success = self.fuseki.insert_turtle(ontology_turtle, graph=f"{self.graph_uri}/ontology")
        if not success:
            print("[RDFSyncManager] WARNING: Failed to insert ontology")
        
        return {"success": True}

    def sync_tables(self, tables: List[Dict[str, Any]]) -> Dict[str, int]:
        """Sync tables to RDF. Returns dict with 'tables' and 'columns' counts."""
        if not tables:
            print("[RDFSyncManager] No tables to sync")
            return {"tables": 0, "columns": 0}
        
        turtle_lines = [PREFIXES]
        table_count = 0
        column_count = 0
        
        for table in tables:
            table_name = table.get("name") or table.get("table_name")
            if not table_name:
                continue
            
            table_uri = make_table_uri(table_name)
            turtle_lines.append(f'<{table_uri}> a gw:Table ; rdfs:label "{table_name}" .')
            table_count += 1
            
            for col in table.get("columns", []):
                col_name = col.get("name") or col.get("column_name")
                if col_name:
                    col_uri = make_column_uri(table_name, col_name)
                    turtle_lines.append(f'<{col_uri}> a gw:Column ; rdfs:label "{col_name}" ; gw:belongsToTable <{table_uri}> .')
                    turtle_lines.append(f'<{table_uri}> gw:hasColumn <{col_uri}> .')
                    column_count += 1
        
        turtle_content = "\n".join(turtle_lines)
        print(f"[RDFSyncManager] Inserting {table_count} tables, {column_count} columns...")
        print(f"[RDFSyncManager] Turtle content length: {len(turtle_content)} chars")
        
        success = self.fuseki.insert_turtle(turtle_content, graph=self.graph_uri)
        if not success:
            print("[RDFSyncManager] ERROR: insert_turtle failed for tables!")
            return {"tables": 0, "columns": 0, "error": "insert_turtle failed"}
        
        print(f"[RDFSyncManager] ✓ Tables inserted successfully")
        return {"tables": table_count, "columns": column_count}

    def sync_foreign_keys(self, fks: List[Dict[str, Any]]) -> int:
        """Sync FK relationships to RDF."""
        if not fks:
            print("[RDFSyncManager] No FKs to sync")
            return 0
        
        turtle_lines = [PREFIXES]
        for fk in fks:
            src_table = fk.get("source_table", "")
            src_col = fk.get("source_column", "")
            tgt_table = fk.get("target_table", "")
            tgt_col = fk.get("target_column", "")
            
            if not all([src_table, src_col, tgt_table, tgt_col]):
                continue
                
            src_col_uri = make_column_uri(src_table, src_col)
            tgt_col_uri = make_column_uri(tgt_table, tgt_col)
            turtle_lines.append(f'<{src_col_uri}> gw:references <{tgt_col_uri}> .')
        
        turtle_content = "\n".join(turtle_lines)
        print(f"[RDFSyncManager] Inserting {len(fks)} FK relationships...")
        
        success = self.fuseki.insert_turtle(turtle_content, graph=self.graph_uri)
        if not success:
            print("[RDFSyncManager] ERROR: insert_turtle failed for FKs!")
            return 0
        
        print(f"[RDFSyncManager] ✓ FKs inserted successfully")
        return len(fks)

    def sync_jobs(self, jobs: List[Dict[str, Any]]) -> Dict[str, int]:
        """Sync jobs and datasets to RDF."""
        if not jobs:
            print("[RDFSyncManager] No jobs to sync")
            return {"jobs": 0, "datasets": 0}
        
        turtle_lines = [PREFIXES]
        job_count = 0
        dataset_names = set()
        
        for job in jobs:
            job_name = job.get("name")
            if not job_name:
                continue
            
            job_uri = make_job_uri(job_name)
            turtle_lines.append(f'<{job_uri}> a gw:Job ; rdfs:label "{job_name}" .')
            job_count += 1
            
            for inp in job.get("inputs", []):
                ds_name = inp.get("name") if isinstance(inp, dict) else inp
                if ds_name and ds_name not in [None, 'null']:
                    ds_uri = make_dataset_uri(ds_name)
                    if ds_name not in dataset_names:
                        turtle_lines.append(f'<{ds_uri}> a gw:Dataset ; rdfs:label "{ds_name}" .')
                        dataset_names.add(ds_name)
                    turtle_lines.append(f'<{job_uri}> gw:readsFrom <{ds_uri}> .')
            
            for out in job.get("outputs", []):
                ds_name = out.get("name") if isinstance(out, dict) else out
                if ds_name and ds_name not in [None, 'null']:
                    ds_uri = make_dataset_uri(ds_name)
                    if ds_name not in dataset_names:
                        turtle_lines.append(f'<{ds_uri}> a gw:Dataset ; rdfs:label "{ds_name}" .')
                        dataset_names.add(ds_name)
                    turtle_lines.append(f'<{job_uri}> gw:writesTo <{ds_uri}> .')
        
        turtle_content = "\n".join(turtle_lines)
        print(f"[RDFSyncManager] Inserting {job_count} jobs, {len(dataset_names)} datasets...")
        
        success = self.fuseki.insert_turtle(turtle_content, graph=self.graph_uri)
        if not success:
            print("[RDFSyncManager] ERROR: insert_turtle failed for jobs!")
            return {"jobs": 0, "datasets": 0}
        
        print(f"[RDFSyncManager] ✓ Jobs inserted successfully")
        return {"jobs": job_count, "datasets": len(dataset_names)}

    def sync_from_neo4j(self) -> Dict[str, Any]:
        """Sync entire Neo4j graph to RDF."""
        _banner("RDF SYNC STARTING - FIXED VERSION")
        print(f"[RDFSyncManager] Graph URI: {self.graph_uri}")
        print(f"[RDFSyncManager] Fuseki URL: {self.fuseki.base_url}")
        
        if self.neo4j is None:
            print("[RDFSyncManager] ERROR: Neo4j client is None!")
            return {"error": "Neo4j client not configured"}
        
        stats = {"tables": 0, "columns": 0, "fks": 0, "jobs": 0, "datasets": 0, "links": 0}
        
        # Test Fuseki connection first
        print("[RDFSyncManager] Testing Fuseki connection...")
        conn_test = self.fuseki.test_connection()
        if not conn_test.get("success"):
            error_msg = conn_test.get("error", "Unknown error")
            print(f"[RDFSyncManager] ERROR: Fuseki connection failed: {error_msg}")
            return {"error": f"Fuseki connection failed: {error_msg}"}
        print("[RDFSyncManager] ✓ Fuseki connected")
        
        # Initialize
        init_result = self.initialize_store()
        if not init_result.get("success"):
            return {"error": init_result.get("error", "Failed to initialize store")}

        # Sync tables and columns
        print("[RDFSyncManager] Querying Neo4j for tables...")
        tables_result = self.neo4j.run_query("""
            MATCH (t:Table)
            OPTIONAL MATCH (t)<-[:BELONGS_TO]-(c:Column)
            WITH t, collect({name: c.name, data_type: c.data_type}) as columns
            RETURN t.name as name, columns
        """)
        
        if tables_result:
            tables = [dict(r) for r in tables_result]
            print(f"[RDFSyncManager] Found {len(tables)} tables in Neo4j")
            table_stats = self.sync_tables(tables)
            stats["tables"] = table_stats.get("tables", 0)
            stats["columns"] = table_stats.get("columns", 0)
        else:
            print("[RDFSyncManager] WARNING: No tables found in Neo4j!")

        # Sync FK relationships
        print("[RDFSyncManager] Querying Neo4j for FK relationships...")
        fks_result = self.neo4j.run_query("""
            MATCH (sc:Column)-[fk:FK_TO]->(tc:Column)
            MATCH (sc)-[:BELONGS_TO]->(st:Table)
            MATCH (tc)-[:BELONGS_TO]->(tt:Table)
            RETURN st.name as source_table, sc.name as source_column,
                   tt.name as target_table, tc.name as target_column
        """)
        
        if fks_result:
            fks = [dict(r) for r in fks_result]
            print(f"[RDFSyncManager] Found {len(fks)} FK relationships in Neo4j")
            stats["fks"] = self.sync_foreign_keys(fks)
        else:
            print("[RDFSyncManager] No FK relationships found in Neo4j")

        # Sync jobs and datasets
        print("[RDFSyncManager] Querying Neo4j for jobs...")
        jobs_result = self.neo4j.run_query("""
            MATCH (j:Job)
            OPTIONAL MATCH (j)-[:READS]->(input:Dataset)
            OPTIONAL MATCH (j)-[:WRITES]->(output:Dataset)
            WITH j, collect(DISTINCT input.name) as inputs, collect(DISTINCT output.name) as outputs
            RETURN j.name as name, inputs, outputs
        """)
        
        if jobs_result:
            jobs = [dict(r) for r in jobs_result]
            print(f"[RDFSyncManager] Found {len(jobs)} jobs in Neo4j")
            job_stats = self.sync_jobs(jobs)
            stats["jobs"] = job_stats.get("jobs", 0)
            stats["datasets"] = job_stats.get("datasets", 0)
        else:
            print("[RDFSyncManager] No jobs found in Neo4j")

        # Get final triple count
        print("[RDFSyncManager] Getting triple count...")
        stats["total_triples"] = self.fuseki.get_triple_count(self.graph_uri)
        print(f"[RDFSyncManager] Total triples in graph: {stats['total_triples']}")
        
        # If we have tables but 0 triples, something went wrong
        if stats["tables"] > 0 and stats["total_triples"] == 0:
            print("[RDFSyncManager] WARNING: Tables were found but 0 triples in store!")
            print("[RDFSyncManager] Possible causes:")
            print("  - Fuseki insert endpoint not accessible")
            print("  - Authentication failed")
            print("  - Graph URI mismatch")
            
            # Try to get count from default graph as fallback
            default_count = self.fuseki.get_triple_count()
            print(f"[RDFSyncManager] Triples in default graph: {default_count}")
        
        return stats


def sync_neo4j_to_rdf(neo4j_client, fuseki_client: Optional[FusekiClient] = None) -> Dict[str, Any]:
    """Convenience function to sync Neo4j to RDF."""
    if fuseki_client is None:
        fuseki_client = FusekiClient()
    sync_manager = RDFSyncManager(fuseki_client, neo4j_client)
    return sync_manager.sync_from_neo4j()
