"""
Apache Jena Fuseki Client - Connect to RDF triple store.

FIXED: Added detailed logging to debug insert failures.
FIXED: URL-encode graph URI in GSP requests (was causing 0 triples!)
"""
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from urllib.parse import quote, urlencode
import requests


def _fuseki_banner(msg: str):
    """Print unmissable banner."""
    print("*" * 60)
    print(f"  FUSEKI: {msg}")
    print("*" * 60)


@dataclass
class FusekiConfig:
    url: str = "http://localhost:3030"
    dataset: str = "graphweaver"
    username: str = "admin"
    password: str = "admin"


class FusekiClient:
    def __init__(self, config: Optional[FusekiConfig] = None):
        if config is None:
            config = FusekiConfig(
                url=os.environ.get("FUSEKI_URL", "http://localhost:3030"),
                dataset=os.environ.get("FUSEKI_DATASET", "graphweaver"),
                username=os.environ.get("FUSEKI_USER", "admin"),
                password=os.environ.get("FUSEKI_PASSWORD", "admin"),
            )
        self.config = config
        self.base_url = f"{config.url}/{config.dataset}"
        self.auth = (config.username, config.password)
        print(f"[FusekiClient] Initialized: {self.base_url}")

    def test_connection(self) -> Dict[str, Any]:
        _fuseki_banner("TESTING FUSEKI CONNECTION")
        try:
            print(f"[FusekiClient] URL: {self.config.url}")
            print(f"[FusekiClient] Dataset: {self.config.dataset}")
            print(f"[FusekiClient] Auth user: {self.config.username}")
            response = requests.get(f"{self.config.url}/$/ping", timeout=5)
            if response.status_code == 200:
                print("[FusekiClient] ✓ Connection successful")
                return {"success": True, "message": "Connected to Fuseki"}
            print(f"[FusekiClient] Connection failed: status {response.status_code}")
            return {"success": False, "error": f"Status {response.status_code}"}
        except Exception as e:
            print(f"[FusekiClient] Connection error: {e}")
            return {"success": False, "error": str(e)}

    def ensure_dataset_exists(self) -> bool:
        try:
            print(f"[FusekiClient] Checking dataset: {self.config.dataset}")
            response = requests.get(
                f"{self.config.url}/$/datasets/{self.config.dataset}",
                auth=self.auth, timeout=5
            )
            if response.status_code == 200:
                print(f"[FusekiClient] ✓ Dataset exists")
                return True
            
            print(f"[FusekiClient] Creating dataset: {self.config.dataset}")
            response = requests.post(
                f"{self.config.url}/$/datasets",
                auth=self.auth,
                data={"dbName": self.config.dataset, "dbType": "tdb2"},
                timeout=10
            )
            if response.status_code in [200, 201]:
                print(f"[FusekiClient] ✓ Dataset created")
                return True
            print(f"[FusekiClient] Failed to create dataset: {response.status_code}")
            return False
        except Exception as e:
            print(f"[FusekiClient] Dataset error: {e}")
            return False

    def sparql_query(self, query: str) -> List[Dict[str, Any]]:
        try:
            response = requests.post(
                f"{self.base_url}/sparql",
                data={"query": query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=30
            )
            if response.status_code != 200:
                print(f"[FusekiClient] Query failed: {response.status_code}")
                return []
            result = response.json()
            bindings = result.get("results", {}).get("bindings", [])
            simplified = []
            for binding in bindings:
                row = {}
                for var, value in binding.items():
                    row[var] = value.get("value")
                simplified.append(row)
            return simplified
        except Exception as e:
            print(f"[FusekiClient] Query exception: {e}")
            return []

    def sparql_update(self, update: str) -> bool:
        try:
            response = requests.post(
                f"{self.base_url}/update",
                data={"update": update},
                auth=self.auth,
                timeout=30
            )
            success = response.status_code in [200, 204]
            if not success:
                print(f"[FusekiClient] Update failed: {response.status_code} - {response.text[:200]}")
            return success
        except Exception as e:
            print(f"[FusekiClient] Update exception: {e}")
            return False

    def insert_turtle(self, turtle_content: str, graph: Optional[str] = None) -> bool:
        """Insert Turtle RDF data into the store."""
        _fuseki_banner("INSERT_TURTLE CALLED")
        try:
            url = f"{self.base_url}/data"
            if graph:
                # URL-encode the graph URI - THIS WAS THE BUG!
                encoded_graph = quote(graph, safe='')
                url += f"?graph={encoded_graph}"
                print(f"[FusekiClient] Graph URI encoded: {graph} -> {encoded_graph}")
            
            print(f"[FusekiClient] INSERT to: {url}")
            print(f"[FusekiClient] Content length: {len(turtle_content)} bytes")
            
            # Log first few lines for debugging
            lines = turtle_content.split('\n')[:5]
            print(f"[FusekiClient] First lines: {lines}")
            
            response = requests.post(
                url,
                data=turtle_content.encode('utf-8'),
                headers={"Content-Type": "text/turtle; charset=utf-8"},
                auth=self.auth,
                timeout=30
            )
            
            success = response.status_code in [200, 201, 204]
            if success:
                print(f"[FusekiClient] ✓ Insert successful: {response.status_code}")
            else:
                print(f"[FusekiClient] ✗ Insert FAILED: {response.status_code}")
                print(f"[FusekiClient] Response: {response.text[:500]}")
            
            return success
        except requests.exceptions.ConnectionError as e:
            print(f"[FusekiClient] Connection error - is Fuseki running? {e}")
            return False
        except Exception as e:
            print(f"[FusekiClient] Insert exception: {type(e).__name__}: {e}")
            return False

    def clear_graph(self, graph: Optional[str] = None) -> bool:
        if graph:
            update = f"CLEAR GRAPH <{graph}>"
        else:
            update = "CLEAR DEFAULT"
        print(f"[FusekiClient] Executing: {update}")
        return self.sparql_update(update)

    def get_triple_count(self, graph: Optional[str] = None) -> int:
        if graph:
            query = f"SELECT (COUNT(*) as ?count) WHERE {{ GRAPH <{graph}> {{ ?s ?p ?o }} }}"
        else:
            query = "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }"
        result = self.sparql_query(query)
        if result:
            count = int(result[0].get("count", 0))
            print(f"[FusekiClient] Triple count ({graph or 'default'}): {count}")
            return count
        return 0
