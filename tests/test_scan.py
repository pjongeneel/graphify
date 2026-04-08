"""Tests for graphify scan - AST-only pipeline (no LLM calls)."""
import json
import shutil
from pathlib import Path

import pytest

from graphify.scan import scan

FIXTURES = Path(__file__).parent / "fixtures"


def _copy_fixtures(tmp_path: Path) -> Path:
    """Copy fixture code files to a temp dir for isolated scanning."""
    target = tmp_path / "project"
    target.mkdir()
    for f in FIXTURES.glob("*.py"):
        shutil.copy(f, target / f.name)
    for f in FIXTURES.glob("*.ts"):
        shutil.copy(f, target / f.name)
    return target


def test_scan_produces_graph_json(tmp_path):
    project = _copy_fixtures(tmp_path)
    scan(project, quiet=True)
    graph_path = project / "graphify-out" / "graph.json"
    assert graph_path.exists()
    data = json.loads(graph_path.read_text())
    assert "nodes" in data
    assert "links" in data
    assert len(data["nodes"]) > 0


def test_scan_produces_report(tmp_path):
    project = _copy_fixtures(tmp_path)
    scan(project, quiet=True)
    report_path = project / "graphify-out" / "GRAPH_REPORT.md"
    assert report_path.exists()
    content = report_path.read_text()
    assert "God Nodes" in content
    assert "Communities" in content


def test_scan_returns_result_dict(tmp_path):
    project = _copy_fixtures(tmp_path)
    result = scan(project, quiet=True)
    assert result["node_count"] > 0
    assert result["edge_count"] > 0
    assert result["community_count"] > 0
    assert result["graph"] is not None
    assert isinstance(result["communities"], dict)


def test_scan_empty_dir_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="No code files found"):
        scan(empty)


def test_scan_quiet_mode(tmp_path, capsys):
    project = _copy_fixtures(tmp_path)
    scan(project, quiet=True)
    captured = capsys.readouterr()
    assert "[graphify scan]" not in captured.out


def test_scan_not_quiet(tmp_path, capsys):
    project = _copy_fixtures(tmp_path)
    scan(project, quiet=False)
    captured = capsys.readouterr()
    assert "[graphify scan] Done:" in captured.out


def test_scan_clears_needs_update_flag(tmp_path):
    project = _copy_fixtures(tmp_path)
    out = project / "graphify-out"
    out.mkdir()
    flag = out / "needs_update"
    flag.write_text("1")
    scan(project, quiet=True)
    assert not flag.exists()


def test_scan_html_flag(tmp_path):
    project = _copy_fixtures(tmp_path)
    scan(project, quiet=True, html=True)
    html_path = project / "graphify-out" / "graph.html"
    assert html_path.exists()
    assert "vis-network" in html_path.read_text()


def test_scan_incremental_same_result(tmp_path):
    """Second run on same files should produce identical node/edge counts."""
    project = _copy_fixtures(tmp_path)
    r1 = scan(project, quiet=True)
    r2 = scan(project, quiet=True)
    assert r1["node_count"] == r2["node_count"]
    assert r1["edge_count"] == r2["edge_count"]
