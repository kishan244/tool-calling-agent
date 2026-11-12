"""The web app must start without errors, with and without evaluation results."""
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from evals.build_questions import OUT
from evals.run import run_config

from .conftest import FakeClient
from .oracle import oracle

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_starts_without_results(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_RESULTS_DIR", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception and "No evaluation results" in app.warning[0].value


def test_app_shows_results(tmp_path, monkeypatch):
    result = run_config("analyst", json.loads(OUT.read_text()), FakeClient([oracle]), model="scripted", log=lambda *_: None)
    (tmp_path / "round1").mkdir()
    (tmp_path / "round1" / "analyst.json").write_text(json.dumps(result))
    monkeypatch.setenv("AGENT_RESULTS_DIR", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].value == "100.0%" and any("switched off" in info.value for info in app.info)
