"""Django-specific static analysis for codebase indexing.

Detects Django project structure, extracts model relationships (FK/O2O/M2M),
URL patterns, external integrations, and per-app file roles -- all from AST
and file-path heuristics. Zero LLM calls.
"""
from __future__ import annotations

import ast
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import networkx as nx


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelField:
    name: str
    field_type: str  # ForeignKey, OneToOneField, ManyToManyField
    related_model: str


@dataclass
class ModelInfo:
    name: str
    source_file: str  # relative to target
    bases: list[str]
    relationship_fields: list[ModelField]


@dataclass
class URLPattern:
    path: str
    view_ref: str
    name: str | None
    source_file: str  # relative to target


@dataclass
class AppInfo:
    name: str  # last component, e.g. "accounts"
    path: str  # relative dir, e.g. "accounts" or "accounts/stripe"
    models: list[ModelInfo]
    urls: list[URLPattern]
    files_by_role: dict[str, list[str]]  # role -> [relative file paths]


@dataclass
class DjangoContext:
    apps: list[AppInfo]
    external_packages: list[str]  # sorted display names
    project_description: str | None


# ---------------------------------------------------------------------------
# Django project / app detection
# ---------------------------------------------------------------------------

def is_django_project(code_files: list[Path], target: Path) -> bool:
    """Return True if *target* looks like a Django project."""
    for f in code_files:
        if f.name == "manage.py":
            return True
        if f.name == "settings.py" or f.name == "settings_base.py":
            try:
                text = f.read_text(errors="ignore")[:8000]
                if "INSTALLED_APPS" in text:
                    return True
            except Exception:
                pass
    return False


def _detect_app_dirs(code_files: list[Path], target: Path) -> list[str]:
    """Return sorted list of Django app directories (relative to *target*).

    An app directory is one that contains an ``apps.py`` file.
    """
    app_dirs: list[str] = []
    for f in code_files:
        if f.name == "apps.py":
            rel_dir = str(f.parent.relative_to(target))
            # Normalize "." to empty -- but apps.py at root is unlikely
            if rel_dir != ".":
                app_dirs.append(rel_dir)
    return sorted(set(app_dirs))


# ---------------------------------------------------------------------------
# File-role classification
# ---------------------------------------------------------------------------

_ROLE_RULES: list[tuple[str, ...]] = [
    # (role, *match_predicates)  -- checked in order
]


def classify_app_files(
    app_dir: str, code_files: list[Path], target: Path
) -> dict[str, list[str]]:
    """Classify source files inside *app_dir* into Django roles.

    Returns ``{role: [relative_path, ...]}`` where role is one of:
    models, views, urls, services, commands, api, admin, signals, forms, tests, other.
    """
    prefix = app_dir + "/"
    roles: dict[str, list[str]] = defaultdict(list)

    for f in code_files:
        if f.suffix != ".py":
            continue
        try:
            rel = str(f.relative_to(target))
        except ValueError:
            continue
        # Must be inside this app (or *is* the app dir for top-level files)
        if not (rel.startswith(prefix) or rel == app_dir + "/" + f.name):
            continue

        inner = rel[len(prefix):]  # path below the app dir
        name = f.name

        role = _classify_one(inner, name)
        roles[role].append(rel)

    return dict(roles)


def _classify_one(inner_path: str, filename: str) -> str:
    """Return the role string for a single file inside a Django app."""
    parts = PurePosixPath(inner_path).parts

    if filename == "models.py" or (len(parts) >= 2 and parts[0] == "models"):
        return "models"
    if filename == "views.py" or (len(parts) >= 2 and parts[0] == "views"):
        return "views"
    if filename == "urls.py":
        return "urls"
    if filename == "services.py" or (len(parts) >= 2 and parts[0] == "services"):
        return "services"
    if "management" in parts and "commands" in parts:
        return "commands"
    if filename == "admin.py":
        return "admin"
    if filename == "signals.py":
        return "signals"
    if filename == "forms.py" or (len(parts) >= 2 and parts[0] == "forms"):
        return "forms"
    if (len(parts) >= 2 and parts[0] == "api") or "serializer" in filename.lower():
        return "api"
    if (
        "tests" in parts
        or "test" in parts
        or filename.startswith("test_")
        or filename == "conftest.py"
        or filename == "factories.py"
    ):
        return "tests"
    return "other"


# ---------------------------------------------------------------------------
# Model extraction  (uses stdlib ast -- no tree-sitter needed)
# ---------------------------------------------------------------------------

_RELATIONSHIP_FIELDS = {"ForeignKey", "OneToOneField", "ManyToManyField"}


def extract_model_info(source_file: str, file_text: str) -> list[ModelInfo]:
    """Extract Django model classes and relationship fields from a models.py file."""
    try:
        tree = ast.parse(file_text)
    except SyntaxError:
        return []

    models: list[ModelInfo] = []

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        bases = _get_base_names(node)
        fields = _get_relationship_fields(node)

        models.append(ModelInfo(
            name=node.name,
            source_file=source_file,
            bases=bases,
            relationship_fields=fields,
        ))

    return models


def _get_base_names(cls: ast.ClassDef) -> list[str]:
    names: list[str] = []
    for base in cls.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def _get_relationship_fields(cls: ast.ClassDef) -> list[ModelField]:
    fields: list[ModelField] = []

    for item in cls.body:
        # Handle both simple assignment and annotated assignment
        target_name: str | None = None
        value: ast.expr | None = None

        if isinstance(item, ast.Assign) and item.targets:
            t = item.targets[0]
            if isinstance(t, ast.Name):
                target_name = t.id
            value = item.value
        elif isinstance(item, ast.AnnAssign) and item.value:
            if isinstance(item.target, ast.Name):
                target_name = item.target.id
            value = item.value

        if target_name is None or not isinstance(value, ast.Call):
            continue

        func_name = _call_func_name(value)
        if func_name not in _RELATIONSHIP_FIELDS:
            continue

        related = _extract_related_model(value)
        if related:
            fields.append(ModelField(
                name=target_name,
                field_type=func_name,
                related_model=related,
            ))

    return fields


def _call_func_name(call: ast.Call) -> str | None:
    """Return the terminal name of the called function, e.g. 'ForeignKey'."""
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _extract_related_model(call: ast.Call) -> str | None:
    """Return the related model name from a relationship field constructor."""
    if not call.args:
        # Try 'to' keyword
        for kw in call.keywords:
            if kw.arg == "to":
                return _resolve_model_ref(kw.value)
        return None
    return _resolve_model_ref(call.args[0])


def _resolve_model_ref(node: ast.expr) -> str | None:
    """Resolve a model reference node to a name string."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        # 'self', 'app.ModelName', or 'ModelName'
        val = node.value
        return val.split(".")[-1] if "." in val else val
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


# ---------------------------------------------------------------------------
# URL pattern extraction  (stdlib ast)
# ---------------------------------------------------------------------------

def extract_url_patterns(source_file: str, file_text: str) -> list[URLPattern]:
    """Extract ``path()`` / ``re_path()`` calls from a urls.py file."""
    try:
        tree = ast.parse(file_text)
    except SyntaxError:
        return []

    patterns: list[URLPattern] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func_name = _call_func_name(node)
        if func_name not in ("path", "re_path"):
            continue

        url_path = _first_str_arg(node)
        if url_path is None:
            continue

        view_ref = _extract_view_ref(node)
        if not view_ref:
            continue

        name = _extract_kwarg_str(node, "name")

        patterns.append(URLPattern(
            path=url_path,
            view_ref=view_ref,
            name=name,
            source_file=source_file,
        ))

    return patterns


def _first_str_arg(call: ast.Call) -> str | None:
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        return call.args[0].value
    return None


def _extract_view_ref(call: ast.Call) -> str | None:
    """Best-effort extraction of the view reference from a path() call."""
    if len(call.args) < 2:
        return None
    second = call.args[1]

    if isinstance(second, ast.Call):
        fn = second.func
        if isinstance(fn, ast.Name) and fn.id == "include":
            arg = _first_str_arg(second)
            if arg:
                return f"include({arg})"
            # include(module.urls) or include([...])
            if second.args:
                inner = second.args[0]
                if isinstance(inner, ast.Attribute):
                    return f"include({_dotted(inner)})"
            return None
        # Something.as_view(), ViewSet.as_view({...})
        return _dotted(fn) + "()" if _dotted(fn) else None

    if isinstance(second, ast.Attribute):
        return _dotted(second)
    if isinstance(second, ast.Name):
        return second.id
    return None


def _dotted(node: ast.expr) -> str | None:
    """Reconstruct a dotted name like ``views.home`` from an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return None


def _extract_kwarg_str(call: ast.Call, kwarg: str) -> str | None:
    for kw in call.keywords:
        if kw.arg == kwarg and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return None


# ---------------------------------------------------------------------------
# External package detection
# ---------------------------------------------------------------------------

_KNOWN_PACKAGES: dict[str, str] = {
    "stripe": "Stripe",
    "twilio": "Twilio",
    "boto3": "AWS",
    "botocore": "AWS",
    "storages": "Django Storages",
    "allauth": "Django Allauth",
    "rest_framework": "Django REST Framework",
    "celery": "Celery",
    "redis": "Redis",
    "sentry_sdk": "Sentry",
    "corsheaders": "CORS",
    "whitenoise": "WhiteNoise",
    "crispy_forms": "Crispy Forms",
    "channels": "Django Channels",
    "graphene": "GraphQL (Graphene)",
    "dramatiq": "Dramatiq",
    "huey": "Huey",
    "requests": "Requests",
    "httpx": "HTTPX",
    "pydantic": "Pydantic",
    "webpack_loader": "Webpack",
    "silk": "Django Silk",
    "debug_toolbar": "Django Debug Toolbar",
    "django_filters": "Django Filter",
    "drf_spectacular": "DRF Spectacular",
    "drf_yasg": "DRF YASG",
    "guardian": "Django Guardian",
    "oauth2_provider": "Django OAuth Toolkit",
    "social_django": "Python Social Auth",
    "captcha": "Django Captcha",
}

_GOOGLE_CLOUD_PREFIXES: dict[str, str] = {
    "google.cloud.storage": "Google Cloud Storage",
    "google.cloud.secretmanager": "Google Secret Manager",
    "google.cloud.pubsub": "Google Pub/Sub",
    "google.cloud.bigquery": "Google BigQuery",
    "google.cloud.firestore": "Google Firestore",
    "google.cloud.tasks": "Google Cloud Tasks",
    "google.cloud.logging": "Google Cloud Logging",
    "google.auth": "Google Auth",
}

_AWS_PREFIXES: dict[str, str] = {
    "boto3": "AWS",
}


def detect_external_packages(code_files: list[Path]) -> list[str]:
    """Scan Python imports across all files to find external integrations.

    Returns a sorted list of human-readable integration names.
    """
    found: set[str] = set()

    for f in code_files:
        if f.suffix != ".py":
            continue
        try:
            text = f.read_text(errors="ignore")
            tree = ast.parse(text)
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    _check_import(alias.name, found)
            elif isinstance(node, ast.ImportFrom) and node.module:
                _check_import(node.module, found)

    return sorted(found)


def _check_import(module_name: str, found: set[str]) -> None:
    # Google Cloud sub-packages (most specific first)
    for prefix, label in _GOOGLE_CLOUD_PREFIXES.items():
        if module_name == prefix or module_name.startswith(prefix + "."):
            found.add(label)
            return

    top = module_name.split(".")[0]
    if top in _KNOWN_PACKAGES:
        found.add(_KNOWN_PACKAGES[top])


# ---------------------------------------------------------------------------
# Project description
# ---------------------------------------------------------------------------

def get_project_description(target: Path) -> str | None:
    """Try to pull a one-liner project description from pyproject.toml or README."""
    desc = _desc_from_pyproject(target)
    if desc:
        return desc
    return _desc_from_readme(target)


def _desc_from_pyproject(target: Path) -> str | None:
    pyproject = target / "pyproject.toml"
    if not pyproject.exists():
        return None
    try:
        text = pyproject.read_text(errors="ignore")
        m = re.search(r'^description\s*=\s*["\'](.+?)["\']', text, re.MULTILINE)
        return m.group(1) if m else None
    except Exception:
        return None


def _desc_from_readme(target: Path) -> str | None:
    for name in ("README.md", "readme.md", "README.rst", "README.txt"):
        path = target / name
        if not path.exists():
            continue
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except Exception:
            continue

        # Grab first content paragraph after the title
        content: list[str] = []
        past_title = False
        for line in lines:
            stripped = line.strip()
            if not past_title:
                if stripped.startswith("#") or re.match(r"^[=\-]{3,}$", stripped):
                    past_title = True
                continue
            if not stripped:
                if content:
                    break
                continue
            if stripped.startswith("#"):
                break
            content.append(stripped)

        if content:
            desc = " ".join(content)
            return desc[:300].rstrip() + ("..." if len(desc) > 300 else "")
    return None


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

_SKIP_FILE_DIRS = {
    "tests", "test", "e2e", "__tests__",
    "static", "staticfiles", "node_modules", "dist", "build",
    ".claude", ".github", ".vscode",
}

_SKIP_FILE_EXTENSIONS = {".js", ".jsx", ".css", ".scss", ".html", ".svg", ".min.js"}


def high_leverage_files(G: nx.Graph, top_n: int = 10) -> list[tuple[str, int]]:
    """Return the *top_n* source files ranked by total degree of contained nodes.

    Excludes test files, static assets, and tooling/skill files.
    """
    file_degree: dict[str, int] = defaultdict(int)
    for nid, data in G.nodes(data=True):
        src = data.get("source_file", "")
        if not src:
            continue
        if data.get("file_type") == "rationale":
            continue
        p = PurePosixPath(src)
        # Skip test files, static assets, tooling dirs
        if any(part in _SKIP_FILE_DIRS for part in p.parts):
            continue
        if p.name in ("conftest.py", "factories.py"):
            continue
        if p.suffix in _SKIP_FILE_EXTENSIONS:
            continue
        file_degree[src] += G.degree(nid)

    ranked = sorted(file_degree.items(), key=lambda x: -x[1])
    return ranked[:top_n]


def service_call_graph(G: nx.Graph) -> dict[str, list[tuple[str, str, str]]]:
    """Build a per-service-directory map of function call relationships.

    Returns ``{service_dir: [(caller_label, callee_label, relation), ...]}``
    where caller or callee is in a ``services/`` path.
    """
    node_data = {nid: data for nid, data in G.nodes(data=True)}

    calls: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    for u, v, edata in G.edges(data=True):
        rel = edata.get("relation", "")
        if rel not in ("calls", "uses"):
            continue

        u_data = node_data.get(u, {})
        v_data = node_data.get(v, {})
        u_src = u_data.get("source_file", "")
        v_src = v_data.get("source_file", "")

        # At least one side must be in a services/ path
        u_in_svc = "/services/" in u_src or u_src.endswith("/services.py")
        v_in_svc = "/services/" in v_src or v_src.endswith("/services.py")
        if not (u_in_svc or v_in_svc):
            continue

        # Skip test nodes
        u_label = u_data.get("label", "")
        v_label = v_data.get("label", "")
        if u_label.startswith("Test") or v_label.startswith("Test"):
            continue
        if u_label.startswith("test_") or v_label.startswith("test_"):
            continue

        # Determine which service dir this belongs to
        svc_src = u_src if u_in_svc else v_src
        svc_dir = str(PurePosixPath(svc_src).parent)

        calls[svc_dir].append((u_label, v_label, rel))

    return dict(calls)


def app_node_labels(
    G: nx.Graph, app_dir: str, role: str, files: list[str]
) -> list[str]:
    """Return top entity labels from graph nodes matching *files*, ranked by degree.

    Filters out file-level nodes, Meta classes, test entities, and generic names.
    """
    _SKIP = {"Meta", "Command", "Migration", "AppConfig"}
    labels_with_deg: list[tuple[str, int]] = []

    file_set = set(files)
    for nid, data in G.nodes(data=True):
        src = data.get("source_file", "")
        if src not in file_set:
            continue
        if data.get("file_type") == "rationale":
            continue
        label = data.get("label", "")
        if not label or label in _SKIP:
            continue
        # Skip file-level nodes (label matches filename stem or ends with .py)
        if label.endswith(".py") or label.endswith(".js"):
            continue
        # Skip method stubs
        if label.startswith("."):
            continue
        # Skip very long labels (docstring fragments)
        if len(label) > 60:
            continue
        # Skip test classes/functions
        if label.startswith("Test") or label.startswith("test_"):
            continue
        # Skip AppConfig subclasses
        if label.endswith("Config") and role != "models":
            continue

        labels_with_deg.append((label, G.degree(nid)))

    # Deduplicate, keeping highest degree
    best: dict[str, int] = {}
    for label, deg in labels_with_deg:
        if label not in best or deg > best[label]:
            best[label] = deg

    # Sort: public names first (higher visibility), then by degree
    ranked = sorted(best.items(), key=lambda x: (x[0].startswith("_"), -x[1]))
    return [label for label, _ in ranked]


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def analyze_django_project(
    target: Path, code_files: list[Path], G: nx.Graph
) -> DjangoContext | None:
    """Run Django-specific static analysis. Returns None if not a Django project."""
    if not is_django_project(code_files, target):
        return None

    app_dirs = _detect_app_dirs(code_files, target)

    apps: list[AppInfo] = []
    for app_dir in app_dirs:
        files_by_role = classify_app_files(app_dir, code_files, target)

        # Extract models from model files
        models: list[ModelInfo] = []
        for model_file in files_by_role.get("models", []):
            full_path = target / model_file
            try:
                text = full_path.read_text(errors="ignore")
                models.extend(extract_model_info(model_file, text))
            except Exception:
                pass

        # Extract URLs
        urls: list[URLPattern] = []
        for url_file in files_by_role.get("urls", []):
            full_path = target / url_file
            try:
                text = full_path.read_text(errors="ignore")
                urls.extend(extract_url_patterns(url_file, text))
            except Exception:
                pass

        app_name = PurePosixPath(app_dir).name
        apps.append(AppInfo(
            name=app_name,
            path=app_dir,
            models=models,
            urls=urls,
            files_by_role=files_by_role,
        ))

    # Also collect URLs from project-level urls.py (not inside any app)
    # These are handled separately since they're often the root urlconf
    _collect_root_urls(target, code_files, app_dirs, apps)

    external = detect_external_packages(code_files)
    description = get_project_description(target)

    return DjangoContext(
        apps=apps,
        external_packages=external,
        project_description=description,
    )


def _collect_root_urls(
    target: Path,
    code_files: list[Path],
    app_dirs: list[str],
    apps: list[AppInfo],
) -> None:
    """Find urls.py files that live in project-config dirs (not inside detected apps)
    and attach their patterns to the matching app or a synthetic root app."""
    for f in code_files:
        if f.name != "urls.py":
            continue
        rel = str(f.relative_to(target))
        # Skip if already inside a detected app
        if any(rel.startswith(ad + "/") for ad in app_dirs):
            continue

        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue

        patterns = extract_url_patterns(rel, text)
        if not patterns:
            continue

        # Attach to a synthetic "root" app or find the project config app
        rel_dir = str(PurePosixPath(rel).parent)
        # See if there's already an app for this dir
        matched = False
        for app in apps:
            if app.path == rel_dir:
                app.urls.extend(patterns)
                matched = True
                break
        if not matched:
            apps.insert(0, AppInfo(
                name=PurePosixPath(rel_dir).name or "root",
                path=rel_dir,
                models=[],
                urls=patterns,
                files_by_role={"urls": [rel]},
            ))
