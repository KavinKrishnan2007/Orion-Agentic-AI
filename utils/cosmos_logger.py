import os
import uuid
from datetime import datetime
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from dotenv import load_dotenv

load_dotenv()

_client = None
_container = None

def _get_container():
    global _client, _container
    if _container:
        return _container
    conn_str = os.getenv("COSMOS_CONNECTION_STRING")
    db_name = os.getenv("COSMOS_DATABASE", "agenticai")
    container_name = os.getenv("COSMOS_CONTAINER", "audit-logs")
    _client = CosmosClient.from_connection_string(conn_str)
    db = _client.create_database_if_not_exists(id=db_name)
    _container = db.create_container_if_not_exists(
        id=container_name,
        partition_key=PartitionKey(path="/workflow_id")
    )
    return _container

def log_event(workflow_id, agent_name, task, output, critic_score=None, status="success"):
    try:
        container = _get_container()
        doc = {
            "id": str(uuid.uuid4()),
            "workflow_id": workflow_id,
            "agent_name": agent_name,
            "task": task,
            "output": output,
            "critic_score": critic_score,
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        container.create_item(body=doc)
        return doc
    except exceptions.CosmosHttpResponseError as e:
        print(f"[CosmosDB] Logging failed: {e.message}")
        return None

def get_recent_logs(limit=10):
    try:
        container = _get_container()
        query = f"SELECT TOP {limit} * FROM c ORDER BY c.timestamp DESC"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        return items
    except Exception as e:
        print(f"[CosmosDB] Fetch failed: {e}")
        return []
