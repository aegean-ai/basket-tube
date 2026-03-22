"""Tests for api/src/ package scaffold."""

import pytest


class TestPackageStructure:
    """Verify api/src/ package is importable with expected modules."""

    def test_api_src_is_importable(self):
        import api.src  # noqa: F401

    def test_core_config_importable(self):
        from api.src.core.config import Settings, settings  # noqa: F401

    def test_core_dependencies_importable(self):
        from api.src.core.dependencies import get_settings  # noqa: F401

    def test_main_importable(self):
        from api.src.main import create_app  # noqa: F401

    def test_routers_package_importable(self):
        import api.src.routers  # noqa: F401

    def test_schemas_package_importable(self):
        import api.src.schemas  # noqa: F401

    def test_services_package_importable(self):
        import api.src.services  # noqa: F401


class TestSettings:
    def test_s3_settings_have_defaults(self):
        from api.src.core.config import Settings
        s = Settings()
        assert hasattr(s, "s3_bucket")

    def test_env_prefix_is_fw(self):
        from api.src.core.config import Settings
        assert Settings.model_config.get("env_prefix") == "FW_"

    def test_settings_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("FW_DEBUG", "true")
        from api.src.core.config import Settings
        s = Settings()
        assert s.debug is True

    def test_cors_defaults(self):
        from api.src.core.config import Settings
        s = Settings()
        assert s.cors_enabled is True


class TestDependencies:
    def test_get_settings_returns_settings_instance(self):
        from api.src.core.config import Settings
        from api.src.core.dependencies import get_settings
        result = get_settings()
        assert isinstance(result, Settings)


class TestAppFactory:
    @pytest.fixture()
    def app(self):
        from api.src.main import create_app
        return create_app()

    def test_app_title(self, app):
        assert app.title == "BasketTube API"

    def test_healthz_endpoint(self, app):
        from fastapi.testclient import TestClient
        with TestClient(app) as client:
            resp = client.get("/healthz")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}

    def test_routers_registered(self, app):
        paths = list(app.openapi()["paths"].keys())
        assert any("/api/download" in p for p in paths)
        assert any("/api/transcribe" in p for p in paths)
        assert any("/api/vision/detect" in p for p in paths)
        assert any("/api/captions/timeline" in p for p in paths)
