"""AST-only code scanning pipeline. Zero LLM calls.

Entry point for `graphify scan` CLI and internal callers like watch.py and hooks.
"""
from __future__ import annotations
from pathlib import Path


def scan(
    target: Path,
    *,
    html: bool = False,
    wiki: bool = False,
    obsidian: bool = False,
    graphml: bool = False,
    svg: bool = False,
    cypher: bool = False,
    output_dir: Path | None = None,
    follow_symlinks: bool = False,
    quiet: bool = False,
) -> dict:
    """Run the full AST-only pipeline on a directory.

    Steps: collect_files -> extract -> build -> cluster -> analyze -> report -> export.
    No LLM calls. Returns a dict with graph, communities, and counts.

    Raises RuntimeError if no code files are found.
    """
    from graphify.extract import collect_files, extract
    from graphify.build import build_from_json
    from graphify.cluster import cluster, score_all
    from graphify.analyze import god_nodes, surprising_connections, suggest_questions
    from graphify.report import generate
    from graphify.export import to_json

    out_name = Path(output_dir).name if output_dir else "graphify-out"
    code_files = collect_files(target, follow_symlinks=follow_symlinks)
    code_files = [
        f for f in code_files
        if out_name not in f.parts
        and "graphify-out" not in f.parts
        and "__pycache__" not in f.parts
    ]

    if not code_files:
        raise RuntimeError(f"No code files found in {target.resolve()}")

    if not quiet:
        print(f"[graphify scan] Scanning {len(code_files)} code files...")

    result = extract(code_files)

    detection = {
        "files": {"code": [str(f) for f in code_files], "document": [], "paper": [], "image": []},
        "total_files": len(code_files),
        "total_words": 0,
    }

    G = build_from_json(result)
    communities = cluster(G)
    cohesion = score_all(G, communities)
    gods = god_nodes(G)
    surprises = surprising_connections(G, communities)
    labels = {cid: "Community " + str(cid) for cid in communities}
    questions = suggest_questions(G, communities, labels)

    out = Path(output_dir) if output_dir else target / "graphify-out"
    out.mkdir(parents=True, exist_ok=True)

    report = generate(G, communities, cohesion, labels, gods, surprises, detection,
                      {"input": 0, "output": 0}, str(target), suggested_questions=questions)
    (out / "GRAPH_REPORT.md").write_text(report)
    to_json(G, communities, str(out / "graph.json"))

    # clear stale needs_update flag if present
    flag = out / "needs_update"
    if flag.exists():
        flag.unlink()

    # optional exports
    if html:
        from graphify.export import to_html
        to_html(G, communities, str(out / "graph.html"), community_labels=labels)
    if wiki:
        from graphify.wiki import to_wiki
        to_wiki(G, communities, str(out / "wiki"), community_labels=labels,
                cohesion=cohesion, god_nodes_data=gods)
    if obsidian:
        from graphify.export import to_obsidian
        to_obsidian(G, communities, str(out / "obsidian"), community_labels=labels, cohesion=cohesion)
    if graphml:
        from graphify.export import to_graphml
        to_graphml(G, communities, str(out / "graph.graphml"))
    if svg:
        from graphify.export import to_svg
        to_svg(G, communities, str(out / "graph.svg"), community_labels=labels)
    if cypher:
        from graphify.export import to_cypher
        to_cypher(G, str(out / "cypher.txt"))

    if not quiet:
        print(f"[graphify scan] Done: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges, {len(communities)} communities")
        print(f"[graphify scan] Output: {out}")

    return {
        "graph": G,
        "communities": communities,
        "cohesion": cohesion,
        "gods": gods,
        "surprises": surprises,
        "questions": questions,
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "community_count": len(communities),
    }
