from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from dlrouter.api.middleware import set_api_keys
from dlrouter.api.routes import nodes


def _make_client(node_manager: MagicMock) -> TestClient:
    app = FastAPI()
    nodes.set_node_manager(node_manager)
    set_api_keys(None)
    app.include_router(nodes.router)
    return TestClient(app)


def test_add_node_returns_success_when_manager_adds_node() -> None:
    node_manager = MagicMock()
    node_manager.add.return_value = True
    client = _make_client(node_manager)

    response = client.post('/nodes/add', json={'url': 'http://node:8000', 'status': None})

    assert response.status_code == 200
    assert response.json() == {'message': 'Added successfully.'}


def test_add_node_returns_error_when_manager_rejects_node() -> None:
    node_manager = MagicMock()
    node_manager.add.return_value = False
    client = _make_client(node_manager)

    response = client.post('/nodes/add', json={'url': 'http://node:8000', 'status': None})

    assert response.status_code == 200
    assert response.json() == {'error': 'Failed to add node. Check the URL and try again.'}
