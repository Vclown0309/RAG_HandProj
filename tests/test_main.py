from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_root() -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"message": "Hello World"}


def test_say_hello() -> None:
    resp = client.get("/hello/豆老师")
    assert resp.status_code == 200
    assert resp.json() == {"message": "Hello 豆老师"}
