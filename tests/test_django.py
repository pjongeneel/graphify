import networkx as nx

from graphify.django import (
    AppInfo,
    DjangoContext,
    ModelInfo,
    classify_app_files,
    detect_external_packages,
    extract_model_info,
    extract_url_patterns,
    get_project_description,
    service_call_graph,
)
from graphify.summary import generate_summary


def test_extract_model_info_filters_non_model_classes():
    source = """
from dataclasses import dataclass
from django.contrib.auth.models import AbstractUser
from django.db import models


class Helper:
    pass


@dataclass
class CreditCard:
    brand: str


class User(models.Model):
    email = models.EmailField()


class Account(AbstractUser):
    pass


class AuditRecord(TimeStampedModel):
    created_by = models.ForeignKey("User", on_delete=models.CASCADE)
"""
    models = extract_model_info("app/models.py", source)
    names = [model.name for model in models]

    assert names == ["User", "Account", "AuditRecord"]


def test_extract_url_patterns_flattens_nested_local_includes():
    source = """
from django.urls import include, path


subscription_patterns = [
    path("cancel/", cancel_view, name="cancel"),
]

verification_patterns = [
    path("email/", email_view, name="email"),
]

urlpatterns = [
    path("accounts/", include([
        path("subscription/", include(subscription_patterns)),
        path("verifications/", include(verification_patterns)),
    ])),
    path("api/", include("myapp.api.urls")),
]
"""
    patterns = extract_url_patterns("app/urls.py", source)
    routes = {(pattern.path, pattern.view_ref) for pattern in patterns}

    assert ("accounts/subscription/cancel/", "cancel_view") in routes
    assert ("accounts/verifications/email/", "email_view") in routes
    assert ("api/", "include(myapp.api.urls)") in routes
    assert ("cancel/", "cancel_view") not in routes


def test_classify_app_files_prefers_api_role(tmp_path):
    app_dir = tmp_path / "app"
    (app_dir / "api").mkdir(parents=True)
    (app_dir / "apps.py").write_text("from django.apps import AppConfig\n")
    (app_dir / "views.py").write_text("def home():\n    pass\n")
    (app_dir / "api" / "views.py").write_text("class ContactListCreateView:\n    pass\n")
    (app_dir / "api" / "serializers.py").write_text("class ContactSerializer:\n    pass\n")
    (app_dir / "api" / "urls.py").write_text("urlpatterns = []\n")

    code_files = sorted(tmp_path.rglob("*.py"))
    roles = classify_app_files("app", code_files, tmp_path)

    assert "app/api/views.py" in roles["api"]
    assert "app/api/serializers.py" in roles["api"]
    assert "app/api/urls.py" in roles["urls"]
    assert "app/views.py" in roles["views"]


def test_detect_external_packages_skips_tests_and_tooling(tmp_path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "services.py").write_text("import stripe\n")

    tooling_dir = tmp_path / ".claude" / "skills"
    tooling_dir.mkdir(parents=True)
    (tooling_dir / "tool.py").write_text("import requests\n")

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_feature.py").write_text("import twilio\n")

    packages = detect_external_packages(sorted(tmp_path.rglob("*.py")))

    assert packages == ["Stripe"]


def test_get_project_description_uses_readme_paragraph(tmp_path):
    (tmp_path / "README.md").write_text(
        "# Project\n\nThis summary should come from the README and remain untrimmed even when it is longer than a short title.\n\n## Details\nMore text.\n"
    )
    (tmp_path / "pyproject.toml").write_text('description = "Ignore this pyproject description"\n')

    description = get_project_description(tmp_path)

    assert description == (
        "This summary should come from the README and remain untrimmed even when it is longer than a short title."
    )


def test_service_call_graph_preserves_edge_direction():
    graph = nx.Graph()
    graph.add_node("caller", label="soft_delete_user()", source_file="accounts/services/account_deletion.py")
    graph.add_node("callee", label="cancel_stripe_subscription()", source_file="accounts/services/account_deletion.py")
    graph.add_edge(
        "caller",
        "callee",
        relation="calls",
        _src="caller",
        _tgt="callee",
    )

    calls = service_call_graph(graph)

    assert calls["accounts/services"] == [
        ("soft_delete_user()", "cancel_stripe_subscription()", "calls")
    ]


def test_generate_summary_uses_inheritance_direction():
    graph = nx.Graph()
    graph.add_node("child", label="ChildModel", source_file="app/models.py")
    graph.add_node("base", label="BaseModel", source_file="common/models.py")
    graph.add_edge(
        "child",
        "base",
        relation="inherits",
        _src="child",
        _tgt="base",
    )

    ctx = DjangoContext(
        apps=[
            AppInfo(
                name="app",
                path="app",
                models=[
                    ModelInfo(
                        name="ChildModel",
                        source_file="app/models.py",
                        bases=["BaseModel"],
                        relationship_fields=[],
                    )
                ],
                urls=[],
                files_by_role={"models": ["app/models.py"]},
            )
        ],
        external_packages=[],
        project_description=None,
    )

    summary = generate_summary(graph, [], django_ctx=ctx)

    assert "ChildModel (BaseModel)" in summary
    assert "BaseModel (ChildModel)" not in summary


def test_generate_summary_app_map_models_only_lists_model_classes():
    graph = nx.Graph()
    graph.add_node("user_class", label="MyUser", source_file="accounts/models.py")
    graph.add_node("schedule_class", label="Schedule", source_file="accounts/models.py")
    graph.add_node("schedule_method", label="schedule()", source_file="accounts/models.py")
    graph.add_node("view_fn", label="dashboard()", source_file="accounts/views/dashboard.py")

    ctx = DjangoContext(
        apps=[
            AppInfo(
                name="accounts",
                path="accounts",
                models=[
                    ModelInfo(
                        name="MyUser",
                        source_file="accounts/models.py",
                        bases=["AbstractUser"],
                        relationship_fields=[],
                    ),
                    ModelInfo(
                        name="Schedule",
                        source_file="accounts/models.py",
                        bases=["TimeStampedModel"],
                        relationship_fields=[],
                    ),
                ],
                urls=[],
                files_by_role={
                    "models": ["accounts/models.py"],
                    "views": ["accounts/views/dashboard.py"],
                },
            )
        ],
        external_packages=[],
        project_description=None,
    )

    summary = generate_summary(graph, [], django_ctx=ctx)

    assert "models     MyUser, Schedule" in summary
    assert "schedule()" not in summary
