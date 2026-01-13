import argparse
import os
import sys
import py_compile


def _compile_file(path: str) -> None:
    py_compile.compile(path, doraise=True)


def _assert_route_exists(app, path: str) -> None:
    for r in getattr(app, "routes", []) or []:
        if getattr(r, "path", None) == path:
            return
    raise AssertionError(f"Missing route: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Manga backend preflight checks")
    parser.add_argument(
        "--skip-import",
        action="store_true",
        help="Only run syntax compile checks (skip importing api/app).",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1) Syntax / indentation / basic import-time issues
    compile_targets = [
        os.path.join(repo_root, "api.py"),
        os.path.join(repo_root, "server", "main.py"),
    ]
    for p in compile_targets:
        if os.path.exists(p):
            _compile_file(p)

    if args.skip_import:
        print("preflight: ok (compile only)")
        return 0

    # 2) Minimal smoke checks (no heavy inference)
    sys.path.insert(0, repo_root)

    # Avoid importing/loading heavy models during preflight.
    os.environ.setdefault('MTS_SKIP_MODEL_INIT', '1')

    import api as api_module  # noqa: F401

    app = getattr(api_module, "app", None)
    if app is None:
        raise AssertionError("api.app not found")

    # Ensure editor-related endpoints are still registered (do not call them).
    _assert_route_exists(app, "/render_text_preview")
    _assert_route_exists(app, "/render_page")

    from fastapi.testclient import TestClient

    client = TestClient(app)

    r = client.get("/openapi.json")
    assert r.status_code == 200

    r = client.get("/extensions/list")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)
    assert data.get("status") == "ok"
    assert isinstance(data.get("items"), list)

    print("preflight: ok")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("preflight: failed")
        print(str(e))
        raise
