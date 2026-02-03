import os
import time

from docker import from_env
from rich.status import Status


CONTAINERS_CONFIG = [
    {
        "name": "cogito-vectors",
        "image": "crazywillbear/cogito-vectors:latest",
        "ports": {'6333/tcp': 6333, '6334/tcp': 6334},
        "environment": {"QDRANT__SERVICE__API_KEY": "default"}
    },
    {
        "name": "cogito-postgres",
        "image": "crazywillbear/cogito-filters-postgres:latest",
        "ports": {'5432/tcp': 5432},
        "environment": {
            "POSTGRES_USER": "default",
            "POSTGRES_PASSWORD": "default",
            "POSTGRES_DB": "cogito"
        }
    }
]

def manage_containers(status: Status):
    """Ensure required Docker containers are running."""

    _set_env_variables()

    try:
        client = from_env()

    except Exception as e:
        print(f"Could not connect to Docker. Is the service running? Are you on Python <=3.12.0\n{e}")
        return

    for config in CONTAINERS_CONFIG:
        try:
            # 1. Try to get the container by name
            container = client.containers.get(config["name"])

            if container.status == "running":
                status.update(status=f"{config['name']} is already running")
            else:
                status.update(status=f"{config['name']} exists but is stopped. Starting...")
                container.start()
                if _wait_for_running(container):
                    status.update(status=f"{config['name']} launched successfully.")
                else:
                    status.update(status=f"{config['name']} failed to reach 'running' in time.")

        except Exception:
            # 2. If it doesn't exist, pull and run
            status.update(status=f"{config['name']} not found. Pulling image and starting...")
            container = client.containers.run(
                config["image"],
                name=config["name"],
                detach=True,
                ports=config["ports"],
                environment=config.get("environment")
            )
            if _wait_for_running(container):
                status.update(status=f"{config['name']} launched successfully.")
            else:
                status.update(status=f"{config['name']} failed to reach 'running' in time.")


def _set_env_variables():
    """Ensure required Docker containers are running."""

    os.environ["COGITO_QDRANT_HOST"] = "localhost"
    os.environ["COGITO_QDRANT_PORT"] = "6334"
    os.environ["COGITO_QDRANT_API_KEY"] = "default"
    os.environ["COGITO_QDRANT_COLLECTION"] = "proj_gutenberg_philosophy"

    os.environ["COGITO_POSTGRES_USER"] = "default"
    os.environ["COGITO_POSTGRES_PASSWORD"] = "default"
    os.environ["COGITO_POSTGRES_PORT"] = "5432"
    os.environ["COGITO_POSTGRES_DBNAME"] = "cogito"
    os.environ["COGITO_POSTGRES_HOST"] = "localhost"


def _wait_for_running(container, timeout: int = 60, interval: float = 1.0) -> bool:
    """Poll Docker until the container reports running or we time out."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        container.reload()
        if container.status == "running":
            time.sleep(2)  # Give a bit of extra time for the service to be ready
            return True
        time.sleep(interval)

    return False
