# Generate a compact, LLM-friendly codebase summary from the knowledge graph.
#
# If a DjangoContext is provided (Django project detected), produces a
# structured architecture cheat-sheet organized around Django conventions.
# Otherwise falls back to a generic directory-tree summary.
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import PurePosixPath

import networkx as nx

from graphify.django import (
    AppInfo,
    DjangoContext,
    ModelInfo,
    app_node_labels,
    high_leverage_files,
    service_call_graph,
)

# ---------------------------------------------------------------------------
# Noise filters (shared)
# ---------------------------------------------------------------------------

_SKIP_DIRS = {
    "migrations", "static", "staticfiles", "vendor",
    "node_modules", "dist", "build", "__pycache__",
}

_NOISE_LABEL_RE = re.compile(
    r"^\.|(\.min\.js$)|(\.py$)|(\.js$)|(\.ts$)|"
    r"^(Migration|Command)$|"
    r"^\d{4}_"
)


def _is_noise_label(label: str) -> bool:
    if _NOISE_LABEL_RE.search(label):
        return True
    if label.startswith(".") and label.endswith("()"):
        return True
    return False


def _is_test_dir(d: str) -> bool:
    parts = PurePosixPath(d).parts
    return any(p in ("tests", "test", "e2e", "__tests__") for p in parts)


# ===================================================================
# Public entry point
# ===================================================================

def generate_summary(
    G: nx.Graph,
    god_node_list: list[dict],
    django_ctx: DjangoContext | None = None,
) -> str:
    """Generate a compact codebase summary.

    When *django_ctx* is provided the output is a structured Django
    architecture cheat-sheet.  Otherwise a generic directory-tree summary.
    """
    if django_ctx is not None:
        return _generate_django_summary(G, god_node_list, django_ctx)
    return _generate_generic_summary(G, god_node_list)


# ===================================================================
# Django-aware summary
# ===================================================================

def _generate_django_summary(
    G: nx.Graph, god_node_list: list[dict], ctx: DjangoContext
) -> str:
    lines: list[str] = ["# Codebase Summary", ""]

    _section_system_purpose(lines, ctx)
    _section_runtime_surface(lines, ctx)
    _section_domain_model(lines, ctx, G)
    _section_app_map(lines, ctx, G)
    _section_service_layer(lines, G)
    _section_external_boundaries(lines, ctx)
    _section_cross_app_bridges(lines, ctx, G)
    _section_high_leverage_files(lines, G)
    _section_test_surface(lines, G)

    return "\n".join(lines)


# --- System Purpose ---------------------------------------------------

def _section_system_purpose(lines: list[str], ctx: DjangoContext) -> None:
    if not ctx.project_description:
        return
    lines.append("## System Purpose")
    lines.append("")
    lines.append(ctx.project_description)
    lines.append("")


# --- Runtime Surface ---------------------------------------------------

def _section_runtime_surface(lines: list[str], ctx: DjangoContext) -> None:
    # Build include() prefix map: module_dotpath -> url_prefix
    # e.g. "accounts.urls" -> "/accounts/"
    include_map: dict[str, str] = {}
    for app in ctx.apps:
        for u in app.urls:
            if u.view_ref.startswith("include("):
                module = u.view_ref[len("include("):-1]
                prefix = "/" + u.path if not u.path.startswith("/") else u.path
                include_map[module] = prefix

    # Collect URLs, resolving include prefixes for sub-app routes
    urls_by_source: dict[str, list[tuple[str, str]]] = defaultdict(list)
    all_commands: list[tuple[str, str]] = []

    for app in ctx.apps:
        for u in app.urls:
            # Skip include() lines themselves -- they're navigation, not endpoints
            if u.view_ref.startswith("include("):
                continue

            # Resolve full path using include prefix map
            src_module = _source_to_module(u.source_file)
            prefix = include_map.get(src_module, "")
            raw = u.path
            if not raw.startswith("/"):
                raw = "/" + raw
            full_path = prefix.rstrip("/") + raw if prefix else raw

            urls_by_source[u.source_file].append((full_path, u.view_ref))

        for cmd_file in app.files_by_role.get("commands", []):
            stem = PurePosixPath(cmd_file).stem
            if stem != "__init__":
                all_commands.append((stem, cmd_file))

    if not urls_by_source and not all_commands:
        return

    lines.append("## Runtime Surface")
    lines.append("")

    if urls_by_source:
        lines.append("HTTP routes:")
        for src in sorted(urls_by_source.keys()):
            routes = urls_by_source[src]
            lines.append(f"  # {src}")
            for path, view_ref in routes:
                lines.append(f"  `{path}` → {view_ref}")
        lines.append("")

    if all_commands:
        lines.append("Background jobs / management commands:")
        for cmd_name, src in sorted(all_commands):
            app_prefix = PurePosixPath(src).parts[0] if PurePosixPath(src).parts else ""
            lines.append(f"- `{cmd_name}` ({app_prefix})")
        lines.append("")


def _source_to_module(source_file: str) -> str:
    """Convert 'accounts/urls.py' to 'accounts.urls'."""
    p = PurePosixPath(source_file)
    parts = list(p.parts)
    if parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)


# --- Core Domain Model -------------------------------------------------

def _section_domain_model(
    lines: list[str], ctx: DjangoContext, G: nx.Graph
) -> None:
    # Collect all models with relationships
    has_models = any(app.models for app in ctx.apps)
    if not has_models:
        return

    lines.append("## Core Domain Model")
    lines.append("")

    # Also collect inheritance from the graph
    graph_inherits: dict[str, list[str]] = defaultdict(list)
    for u, v, edata in G.edges(data=True):
        if edata.get("relation") == "inherits":
            src_id = edata.get("_src", u)
            tgt_id = edata.get("_tgt", v)
            src_label = G.nodes[src_id].get("label", "")
            tgt_label = G.nodes[tgt_id].get("label", "")
            if src_label and tgt_label:
                graph_inherits[src_label].append(tgt_label)

    for app in ctx.apps:
        if not app.models:
            continue

        lines.append(f"{app.path}/:")

        for model in app.models:
            # Build inheritance display from graph data
            bases = graph_inherits.get(model.name, [])
            if not bases:
                # Fallback to AST-extracted bases
                bases = [b for b in model.bases if b not in ("object",)]
            base_str = f" ({', '.join(bases)})" if bases else ""
            lines.append(f"  {model.name}{base_str}")

            # Show relationship fields
            for field in model.relationship_fields:
                arrow = "→" if field.field_type == "ForeignKey" else (
                    "—" if field.field_type == "OneToOneField" else "↔"
                )
                lines.append(
                    f"    {field.name} {arrow} {field.related_model}"
                    f"  ({field.field_type})"
                )

        lines.append("")


# --- App Map -----------------------------------------------------------

def _section_app_map(
    lines: list[str], ctx: DjangoContext, G: nx.Graph
) -> None:
    lines.append("## App Map")
    lines.append("")

    # Role display order and labels
    _ROLE_ORDER = [
        ("models", "models"),
        ("views", "views"),
        ("services", "services"),
        ("commands", "commands"),
        ("api", "api"),
        ("admin", "admin"),
        ("signals", "signals"),
        ("forms", "forms"),
    ]

    for app in ctx.apps:
        # Skip apps that only have urls (root config apps)
        has_code = any(
            role != "urls" and role != "tests" and role != "other"
            for role in app.files_by_role
        )
        if not has_code:
            continue

        lines.append(f"### {app.path}/")

        for role_key, role_label in _ROLE_ORDER:
            files = app.files_by_role.get(role_key, [])
            if not files:
                continue

            if role_key == "models":
                labels = [model.name for model in app.models]
            else:
                labels = app_node_labels(G, app.path, role_key, files)
            if not labels:
                continue

            # Show top entities per role, capped
            cap = 6 if role_key == "models" else 4
            display = labels[:cap]
            suffix = f", +{len(labels) - cap}" if len(labels) > cap else ""
            lines.append(f"  {role_label:10s} {', '.join(display)}{suffix}")

        lines.append("")


# --- Service Layer -----------------------------------------------------

def _section_service_layer(lines: list[str], G: nx.Graph) -> None:
    call_map = service_call_graph(G)
    if not call_map:
        return

    lines.append("## Service Layer")
    lines.append("")

    for svc_dir in sorted(call_map.keys()):
        edges = call_map[svc_dir]
        lines.append(f"{svc_dir}/:")

        # Build adjacency: caller -> list of (callee, relation)
        out_edges: dict[str, list[str]] = defaultdict(list)
        in_edges: dict[str, list[str]] = defaultdict(list)

        for caller, callee, rel in edges:
            if rel == "calls":
                out_edges[caller].append(callee)
                in_edges[callee].append(caller)

        # Find "public" service functions: those that are called from outside
        # or have outgoing calls.  Skip private helpers as top-level entries.
        shown: set[str] = set()
        all_funcs = set(out_edges.keys()) | set(in_edges.keys())

        # Show functions with outgoing calls first (orchestrators)
        for func in sorted(all_funcs, key=lambda f: -len(out_edges.get(f, []))):
            if func in shown:
                continue
            callees = out_edges.get(func, [])
            if not callees:
                continue
            # Skip private/helper functions as entry-level items
            if func.startswith("_"):
                continue
            shown.add(func)
            callee_str = ", ".join(sorted(set(callees)))
            lines.append(f"  {func}")
            lines.append(f"    → {callee_str}")

        # Show remaining non-private functions that only receive calls
        for func in sorted(all_funcs):
            if func in shown or func.startswith("_"):
                continue
            shown.add(func)
            lines.append(f"  {func}")

        lines.append("")


# --- External Boundaries -----------------------------------------------

def _section_external_boundaries(lines: list[str], ctx: DjangoContext) -> None:
    if not ctx.external_packages:
        return
    lines.append("## External Boundaries")
    lines.append("")
    for pkg in ctx.external_packages:
        lines.append(f"- {pkg}")
    lines.append("")


# --- Cross-App Bridges -------------------------------------------------

def _section_cross_app_bridges(
    lines: list[str], ctx: DjangoContext, G: nx.Graph
) -> None:
    # Two data sources:
    # 1. Model FK/O2O/M2M relationships pointing across apps
    # 2. Graph cross-module edge counts (existing logic)

    bridges: list[str] = []

    # Source 1: model relationships across apps
    model_to_app: dict[str, str] = {}
    for app in ctx.apps:
        for m in app.models:
            model_to_app[m.name] = app.path

    for app in ctx.apps:
        for model in app.models:
            for field in model.relationship_fields:
                target_app = model_to_app.get(field.related_model)
                if target_app and target_app != app.path:
                    bridges.append(
                        f"- {app.path} → {target_app}: "
                        f"{model.name}.{field.name} → {field.related_model}"
                    )

    # Source 2: graph edge counts between app dirs (for non-model connections)
    dir_connections: dict[tuple[str, str], int] = defaultdict(int)
    app_paths = {a.path for a in ctx.apps}

    for u, v, edata in G.edges(data=True):
        u_src = G.nodes[u].get("source_file", "")
        v_src = G.nodes[v].get("source_file", "")
        if not u_src or not v_src:
            continue

        u_app = _file_to_app(u_src, app_paths)
        v_app = _file_to_app(v_src, app_paths)
        if not u_app or not v_app or u_app == v_app:
            continue
        if _is_test_dir(u_src) or _is_test_dir(v_src):
            continue

        pair = tuple(sorted([u_app, v_app]))
        dir_connections[pair] += 1

    if not bridges and not dir_connections:
        return

    lines.append("## Cross-App Bridges")
    lines.append("")

    # Model-level bridges first (more specific)
    if bridges:
        lines.append("Model relationships:")
        for b in bridges:
            lines.append(b)
        lines.append("")

    # Top edge-count bridges (supplement)
    top = sorted(dir_connections.items(), key=lambda x: -x[1])[:8]
    if top:
        lines.append("Edge counts:")
        for (d1, d2), count in top:
            lines.append(f"- `{d1}` ↔ `{d2}` ({count} connections)")
        lines.append("")


def _file_to_app(source_file: str, app_paths: set[str]) -> str | None:
    """Map a source_file path to the best-matching app directory."""
    # Try longest prefix match
    best: str | None = None
    for ap in app_paths:
        if source_file.startswith(ap + "/") or source_file.startswith(ap + "\\"):
            if best is None or len(ap) > len(best):
                best = ap
    return best


# --- High-Leverage Files -----------------------------------------------

def _section_high_leverage_files(lines: list[str], G: nx.Graph) -> None:
    top = high_leverage_files(G, top_n=10)
    if not top:
        return

    lines.append("## High-Leverage Files")
    lines.append("")
    for src, degree in top:
        lines.append(f"- `{src}` ({degree} connections)")
    lines.append("")


# --- Test Surface -------------------------------------------------------

def _section_test_surface(lines: list[str], G: nx.Graph) -> None:
    test_files: set[str] = set()
    test_classes = 0
    factory_count = 0

    for nid, data in G.nodes(data=True):
        if data.get("file_type") == "rationale":
            continue
        src = data.get("source_file", "")
        label = data.get("label", "")
        if not src:
            continue

        is_test = _is_test_dir(src) or PurePosixPath(src).name in (
            "conftest.py", "factories.py"
        ) or PurePosixPath(src).name.startswith("test_")

        if not is_test:
            continue

        if label.endswith(".py"):
            test_files.add(src)
        elif label.startswith("Test"):
            test_classes += 1
        elif label.endswith("Factory"):
            factory_count += 1

    if not test_files and not test_classes:
        return

    lines.append("## Test Surface")
    lines.append("")
    parts: list[str] = []
    if test_files:
        parts.append(f"{len(test_files)} test files")
    if test_classes:
        parts.append(f"{test_classes} test classes")
    if factory_count:
        parts.append(f"{factory_count} factories in test infrastructure")
    lines.append(", ".join(parts) + ".")
    lines.append("")


# ===================================================================
# Generic (non-Django) summary  -- original logic preserved
# ===================================================================

def _generate_generic_summary(G: nx.Graph, god_node_list: list[dict]) -> str:
    """Fallback summary for non-Django codebases -- directory tree + god nodes."""
    by_dir: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for nid, data in G.nodes(data=True):
        src = data.get("source_file", "")
        if not src:
            continue
        label = data.get("label", "")
        if not label or len(label) > 60 or _is_noise_label(label):
            continue
        directory = str(PurePosixPath(src).parent)
        dir_parts = set(PurePosixPath(directory).parts)
        if dir_parts & _SKIP_DIRS:
            continue
        degree = G.degree(nid)
        by_dir[directory].append((nid, label, degree))

    dir_summaries: dict[str, list[str]] = {}
    for d, nodes in by_dir.items():
        seen_labels: dict[str, int] = {}
        for _, label, degree in nodes:
            if label not in seen_labels or degree > seen_labels[label]:
                seen_labels[label] = degree
        top = sorted(seen_labels.items(), key=lambda x: -x[1])
        key_entities = [label for label, deg in top[:4] if deg >= 2]
        dir_summaries[d] = key_entities

    lines: list[str] = ["# Codebase Summary", ""]

    if god_node_list:
        lines.append("## Core Abstractions")
        lines.append("")
        for node in god_node_list[:10]:
            src = ""
            nid = node.get("id", "")
            if nid and nid in G:
                src = G.nodes[nid].get("source_file", "")
            src_str = f" (`{src}`)" if src else ""
            lines.append(
                f"- **{node['label']}** — {node['edges']} connections{src_str}"
            )
        lines.append("")

    lines.append("## Directory Structure")
    lines.append("")

    tree: dict[str, dict] = {}
    for d in sorted(dir_summaries.keys()):
        parts = PurePosixPath(d).parts
        node = tree
        for part in parts:
            node = node.setdefault(part, {})

    def render_tree(
        node: dict, prefix: str, path_so_far: str, depth: int
    ) -> None:
        items = sorted(node.keys())
        for i, name in enumerate(items):
            is_last = i == len(items) - 1
            connector = "`-- " if is_last else "|-- "
            full_path = f"{path_so_far}/{name}" if path_so_far else name
            lookup = full_path.lstrip("./")
            if not lookup:
                lookup = "."
            entities = dir_summaries.get(lookup, [])
            if not entities:
                entities = dir_summaries.get(f"./{lookup}", [])
            desc = ", ".join(entities) if entities else ""
            desc_str = f"  ({desc})" if desc else ""
            lines.append(f"{prefix}{connector}{name}/{desc_str}")
            child_prefix = prefix + ("    " if is_last else "|   ")
            if depth < 4:
                render_tree(node[name], child_prefix, full_path, depth + 1)

    root_entities = dir_summaries.get(".", [])
    if root_entities:
        lines.append(f"./  ({', '.join(root_entities)})")
    render_tree(tree, "", "", 0)
    lines.append("")

    lines.append("## Key Cross-Module Connections")
    lines.append("")
    dir_connections: dict[tuple[str, str], int] = defaultdict(int)
    for u, v in G.edges():
        u_src = G.nodes[u].get("source_file", "")
        v_src = G.nodes[v].get("source_file", "")
        if not u_src or not v_src:
            continue
        u_dir = str(PurePosixPath(u_src).parent)
        v_dir = str(PurePosixPath(v_src).parent)
        if u_dir == v_dir:
            continue
        if _is_test_dir(u_dir) or _is_test_dir(v_dir):
            continue
        if set(PurePosixPath(u_dir).parts) & _SKIP_DIRS:
            continue
        if set(PurePosixPath(v_dir).parts) & _SKIP_DIRS:
            continue
        if u_dir.startswith(v_dir + "/") or v_dir.startswith(u_dir + "/"):
            continue
        pair = tuple(sorted([u_dir, v_dir]))
        dir_connections[pair] += 1

    top_connections = sorted(dir_connections.items(), key=lambda x: -x[1])[:10]
    for (d1, d2), count in top_connections:
        lines.append(f"- `{d1}` <-> `{d2}` ({count} connections)")
    if not top_connections:
        lines.append("- No significant cross-module connections detected")
    lines.append("")

    return "\n".join(lines)
