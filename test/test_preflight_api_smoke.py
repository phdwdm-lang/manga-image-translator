import importlib
import os
import sys


def test_api_import_and_core_routes_exist():
    os.environ.setdefault('MTS_SKIP_MODEL_INIT', '1')
    if 'api' in sys.modules:
        del sys.modules['api']
    api = importlib.import_module("api")
    app = getattr(api, "app", None)
    assert app is not None

    paths = {getattr(r, "path", None) for r in getattr(app, "routes", []) or []}

    # Editor-related endpoints should remain available.
    assert "/render_text_preview" in paths
    assert "/render_page" in paths

    # Extension list is used by SettingsModal.
    assert "/extensions/list" in paths


def test_extensions_list_smoke():
    os.environ.setdefault('MTS_SKIP_MODEL_INIT', '1')
    if 'api' in sys.modules:
        del sys.modules['api']
    api = importlib.import_module("api")
    app = getattr(api, "app", None)

    from fastapi.testclient import TestClient

    client = TestClient(app)
    r = client.get("/extensions/list")
    assert r.status_code == 200

    data = r.json()
    assert isinstance(data, dict)
    assert data.get("status") == "ok"
    assert isinstance(data.get("items"), list)
