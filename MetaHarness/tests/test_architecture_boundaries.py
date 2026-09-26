"""Mechanical guards for the refoundation invariants (``docs/refoundation-v3.md``).

Every guard reads the tracked tree with :mod:`ast`, never with a source regex,
and none of them imports the module it inspects: a re-introduced monolith, a
cross-module private import or a resurrected compatibility branch fails here
instead of surviving until the next review.

The ``FROZEN_*`` tables are ratchets: they record the debt the refoundation
landed with, and their entries may only shrink.  Growing one is a design
decision that must be taken explicitly, by editing the frozen table in the very
same change.
"""

from __future__ import annotations

import ast
import sys
import unittest

from functools import cache
from pathlib import Path
from typing import Iterator, Mapping

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "metaharness"
TESTS = ROOT / "tests"
PIPELINE_TESTS = TESTS / "pipeline"
# (top-level module name, directory that holds it): the scanned universe.
MODULE_ROOTS: tuple[tuple[str, Path], ...] = (
    ("metaharness", PACKAGE),
    ("tests", TESTS),
)

# ---------------------------------------------------------------------------
# invariant 10: orchestrator.py is a façade/composition root, <= 500 lines
# ---------------------------------------------------------------------------

ORCHESTRATOR = PACKAGE / "orchestrator.py"
ORCHESTRATOR_MAX_LINES = 500
# The façade owns no wire protocol and no parser: plan, review and check-repair
# grammar belong to the authority modules below.
ORCHESTRATOR_FORBIDDEN_IMPORTS = (
    "metaharness.planning",
    "metaharness.review",
    "metaharness.orchestration.check_repair",
)
PROTOCOL_MARKERS = (
    "META PLAN", "END META PLAN", "META REVIEW", "CHECK REPAIR", "BEGIN STEP", "END STEP",
)
PARSER_PREFIXES = ("parse_", "_parse_", "read_meta", "_read_meta")

# ---------------------------------------------------------------------------
# invariant 9: no module of the split packages exceeds the 900-line budget
# ---------------------------------------------------------------------------

SPLIT_PACKAGES = (PACKAGE / "orchestration", PACKAGE / "planning")
MODULE_MAX_LINES = 900
# Landed sizes of the modules that predate the refoundation: frozen ceilings,
# never raised by accident.  The two replan entries below were raised in the
# change that made the red-gate rung rewrite a step's contract: the evidence
# belongs to the ladder that produced the failure, the rewind/re-execution
# belongs to the step service that owns their artifacts, and a fresh module
# would have had to reach the shared toolbox through the private imports the
# table below forbids.
# Raised in the check-replan change: the red-gate recovery ladder gained its
# last, autonomous rung (a cycle re-decomposition) inside the modules that
# already own the gate episode, the plan authority and the resume proof, and
# the durable answer itself landed in ``planning/check_replan.py``.
# Raised in the single-correction-budget change: the ladder's cycle rung is
# refused against that budget inside the module that owns the rungs, the plan
# authority learned to read a check-replan's own directory, and the review
# service keeps its defence-in-depth refusal of an unaffordable rung.
FROZEN_MODULE_SIZES: Mapping[str, int] = {
    "orchestration/check_repair.py": 2011,
    "orchestration/resume_validation.py": 1392,
    "orchestration/review_service.py": 1790,
    "orchestration/revision.py": 1072,
}

PIPELINE_TEST_MAX_LINES = 1000

# ---------------------------------------------------------------------------
# invariant 7: no module imports a `_private_name` of another module
# ---------------------------------------------------------------------------

# Every private import the refoundation landed with: the intra-package toolbox
# of `orchestration/shared.py`, the explicit primitives its siblings expose, and
# six module-local names six test modules still reach into.  Frozen: any new
# edge fails, and the table may only shrink.
FROZEN_PRIVATE_IMPORTS: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "metaharness.orchestration.candidate": {
        "metaharness.orchestration.shared": ('_json_text', '_read_json_artifact'),
    },
    "metaharness.orchestration.check_recovery": {
        "metaharness.orchestration.shared": ('_archive_attempt_tree', '_safe_candidate_tree'),
    },
    "metaharness.orchestration.check_repair": {
        "metaharness.orchestration.shared": (
            "_PROMPTS_DIR",
            "_is_object_id",
            "_json_text",
            "_read_json_artifact",
        ),
    },
    "metaharness.orchestration.gates": {
        "metaharness.orchestration.shared": (
            "_CHECK_ATTEMPT_ARTIFACTS",
            "_REVISION_ATTEMPT_ARTIFACTS",
            "_archive_attempt",
            "_archive_attempt_tree",
            "_check_payload",
            "_git_ownership",
            "_is_object_id",
            "_json_text",
            "_read_json_artifact",
            "_record_failure_tree",
            "_safe_candidate_tree",
        ),
        "metaharness.orchestration.check_repair": (
            "_SCOPE_REQUEST_SOURCE",
        ),
    },
    "metaharness.orchestration.publication": {
        "metaharness.orchestration.shared": (
            "_is_object_id",
            "_json_text",
            "_read_json_artifact",
            "_status_has_unstaged_or_untracked",
        ),
        "metaharness.orchestration.candidate": ('_candidate_commit_path', '_commit_web_url'),
        "metaharness.orchestration.resume_validation": ('_accepted_review',),
    },
    "metaharness.orchestration.resume_validation": {
        "metaharness.orchestration.check_repair": ('_hard_failure_items',),
        "metaharness.orchestration.shared": (
            "_MAX_AGENT_REPORT_BYTES",
            "_MAX_STEP_REPORT_BYTES",
            "_PLANNER_CONVERSATION",
            "_is_object_id",
            "_json_text",
            "_read_bounded_text",
            "_read_json_artifact",
            "_read_tree_file",
            "_status_has_unstaged_or_untracked",
        ),
    },
    "metaharness.orchestration.review_service": {
        "metaharness.orchestration.shared": (
            "_REVIEW_ATTEMPT_ARTIFACTS",
            "_REVISION_ATTEMPT_ARTIFACTS",
            "_archive_attempt",
            "_bounded_report",
            "_check_payload",
            "_git_ownership",
            "_is_object_id",
            "_json_text",
            "_read_json_artifact",
            "_record_failure_tree",
            "_repair_checks_payload",
            "_safe_candidate_tree",
        ),
        "metaharness.orchestration.revision": (
            "_bounded_previous_revision_report",
            "_deferred_contract_mismatches",
            "_review_payload",
        ),
        "metaharness.orchestration.check_repair": (
            "_check_repair_prompt",
        ),
        "metaharness.orchestration.scope_repair": ('_build_scope_delta',),
        "metaharness.orchestration.resume_validation": (
            "_accepted_review",
            "_read_planner_conversation",
            "_reusable_pre_checks",
        ),
    },
    "metaharness.orchestration.revision": {
        "metaharness.orchestration.shared": (
            "_PROMPTS_DIR",
            "_bounded_report",
            "_check_payload",
            "_git_ownership",
            "_json_text",
            "_ownership_violations",
            "_read_bounded_text",
            "_record_failure_tree",
        ),
    },
    "metaharness.orchestration.scope_repair": {
        "metaharness.orchestration.shared": ('_create_file_once', '_json_text'),
    },
    "metaharness.orchestration.worker_recovery": {
        "metaharness.orchestration.shared": (
            "_REVISION_ATTEMPT_ARTIFACTS",
            "_archive_attempt",
            "_json_text",
            "_read_json_artifact",
            "_record_failure_tree",
            "_safe_candidate_tree",
            "_safe_status",
        ),
    },
    "tests.test_agent_events": {
        "metaharness.claude.agent": ('_scan_events',),
    },
    "tests.test_codex": {
        "metaharness.agent.runtime": ('_MANAGED_CONFIG',),
    },
    "tests.test_context": {
        "metaharness.context": ('_MAX_EXCERPT_LINES',),
    },
    "tests.test_doctor": {
        "metaharness.cli": ('_usable_secret_value',),
    },
    "tests.test_pipeline_v2_invariants": {
        "metaharness.orchestration.resume_validation": ('_load_completed_step',),
    },
    "tests.test_state": {
        "metaharness.state": ('_exclusive_state_lock',),
    },
}

# ---------------------------------------------------------------------------
# invariants 1, 2, 5, 6, 8: authority layering and the clean v3 break
# ---------------------------------------------------------------------------

# The state vocabulary and its durable store: runtime-neutral by construction.
CORE_STATE_MODULES = ("models.py", "state.py")
CORE_STATE_ALLOWED_INTERNAL_IMPORTS: Mapping[str, frozenset[str]] = {
    "models.py": frozenset({"metaharness.recovery_policy"}),
    "state.py": frozenset({"metaharness.models"}),
}
CORE_STATE_FORBIDDEN_MODULES = frozenset({
    "gitops", "web", "agent", "llm", "remote", "integrations", "orchestration",
    "subprocess", "socket", "urllib", "http",
})

PLANNING_PROTOCOL = PACKAGE / "planning" / "protocol.py"
PROTOCOL_ALLOWED_INTERNAL_IMPORTS = frozenset({
    "metaharness.models", "metaharness.plan_repository_validation", "metaharness.step_ids",
})
PROTOCOL_FORBIDDEN_MODULES = frozenset({
    "git", "gitops", "state", "llm", "agent", "web", "remote", "orchestration",
    "subprocess", "socket", "urllib", "http", "requests", "httpx", "openai", "anthropic",
})

RECOVERY_POLICY = PACKAGE / "recovery_policy.py"
RECOVERY_POLICY_FORBIDDEN_MODULES = frozenset({
    "gitops", "state", "llm", "agent", "web", "orchestration", "remote",
    "subprocess", "socket", "urllib", "http", "requests", "httpx", "openai", "anthropic",
})

# Every symbol the v3 format deleted; none of these names may exist anywhere in
# `src/metaharness`.  `default_implementer_profile` stays a documented *config
# key* of `[ui]` in `config.py` (a string, never an identifier) because
# `docs/providers.md` still accepts it as an input path.
REMOVED_COMPATIBILITY_SYMBOLS = (
    "_TEXT_SECTIONS",
    "_apply_decomposition_policy",
    "_apply_execution_mode_policy",
    "_migrate_historical_step_acceptance",
    "REPAIR_PLANNER_INLINE_TARGET_BYTES",
    "REQUIRE_STAGED_POLICY_TEXT",
    "STEP_ACCEPTANCE_INTEGRITY_OPERATION",
    "build_repair_planner_prompt",
    "default_implementer_profile",
    "historical_check_repair_redaction_crash",
    "legacy_mismatch_sources",
    "legacy_prompt_bug_candidate",
    "optional_recovery_fields",
    "persist_implementation_bundle",
    "persist_planning_artifacts_v2",
    "run_planner_v2",
    "stranded_contract_repair",
    "supersede_legacy_prompt_bug",
)
# `render_profile_catalogue` is live in `recommendation.py` and was a dead alias
# of the old `planning_v2.py`: only the planning package must stay free of it.
REMOVED_SYMBOLS_PER_PACKAGE: Mapping[str, tuple[str, ...]] = {
    "planning": ("render_profile_catalogue",),
}
# Durable migration labels of the deleted resume and recovery paths.
REMOVED_COMPATIBILITY_LITERALS = (
    "auto-bounded failing-test evidence",
    "historical_check_repair_redaction_crash",
    "historical_commit_gate_stale_authority",
    "legacy_prompt_bug_candidate",
    "stranded_contract_repair",
)
# Identifier fragments no runtime module may reintroduce.
REMOVED_IDENTIFIER_TOKENS = ("compat", "legacy", "historical", "deprecated", "backward")

# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


@cache
def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@cache
def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _module_of(path: Path, root: Path) -> tuple[str, bool]:
    """The dotted module of one file relative to its root, and its package flag."""

    parts = list(path.relative_to(root).with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def _dotted(name: str, package: str) -> str:
    return f"{package}.{name}" if name else package


def _absolute_target(module: str, is_package: bool, node: ast.ImportFrom) -> str:
    """Resolve one ``from`` import to the dotted module it names."""

    if not node.level:
        return node.module or ""
    parts = module.split(".")
    depth = len(parts) - node.level + (1 if is_package else 0)
    return ".".join(parts[:depth] + ([node.module] if node.module else []))


@cache
def _resolved_imports(
    path: Path, root: Path, package: str,
) -> list[tuple[int, str, tuple[str, ...]]]:
    """(line, dotted target module, imported names) for every import of one file."""

    name, is_package = _module_of(path, root)
    module = _dotted(name, package)
    imports: list[tuple[int, str, tuple[str, ...]]] = []
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.ImportFrom):
            names = tuple(alias.name for alias in node.names)
            imports.append((node.lineno, _absolute_target(module, is_package, node), names))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name, ()))
    return imports


def _package_of(module: str) -> str:
    return module.rsplit(".", 1)[0] if "." in module else ""


def _root_of(module: str) -> str:
    return module.split(".", 1)[0]


def _private_names(names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in names if name.startswith("_") and not name.startswith("__"))


@cache
def _scanned_modules() -> Mapping[str, Path]:
    """Every importable module of the scanned universe, by dotted name."""

    modules = {}
    for package, root in MODULE_ROOTS:
        for path in sorted(root.rglob("*.py")):
            name, _ = _module_of(path, root)
            modules[_dotted(name, package)] = path
    return modules


@cache
def _private_imports() -> list[tuple[str, str, tuple[str, ...], int]]:
    """Every cross-module import of a `_private_name` in the scanned universe."""

    registry = _scanned_modules()
    found: list[tuple[str, str, tuple[str, ...], int]] = []
    for package, root in MODULE_ROOTS:
        for path in sorted(root.rglob("*.py")):
            name, _ = _module_of(path, root)
            module = _dotted(name, package)
            for lineno, target, names in _resolved_imports(path, root, package):
                private = _private_names(names)
                if private and target != module and target in registry:
                    found.append((module, target, private, lineno))
    return found


def _identifiers(tree: ast.Module) -> Iterator[tuple[int, str]]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.lineno, node.name
        elif isinstance(node, ast.Name):
            yield node.lineno, node.id
        elif isinstance(node, ast.arg):
            yield node.lineno, node.arg
        elif isinstance(node, ast.Attribute):
            yield node.lineno, node.attr
        elif isinstance(node, ast.keyword) and node.arg:
            yield node.lineno, node.arg
        elif isinstance(node, ast.alias):
            yield node.lineno, node.name.rsplit(".", 1)[-1]
            if node.asname:
                yield node.lineno, node.asname


def _string_literals(tree: ast.Module) -> Iterator[tuple[int, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


def _docstring_ids(tree: ast.Module) -> frozenset[int]:
    """The ids of every module, class and function docstring constant."""

    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", [])
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            found.add(id(body[0].value))
    return frozenset(found)


def _code_strings(tree: ast.Module) -> Iterator[tuple[int, str]]:
    """String constants that are not docstrings: docstrings may explain history."""

    docstrings = _docstring_ids(tree)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            yield node.lineno, node.value


def _called_names(tree: ast.Module) -> Iterator[tuple[int, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                yield node.lineno, func.id
            elif isinstance(func, ast.Attribute):
                yield node.lineno, func.attr


@cache
def _import_graph(root: Path, package: str) -> Mapping[str, frozenset[str]]:
    """Dotted module -> the modules of the same package it imports, directly."""

    graph = {}
    for path in sorted(root.rglob("*.py")):
        name, _ = _module_of(path, root)
        module = _dotted(name, package)
        graph[module] = frozenset(
            target
            for _, target, _ in _resolved_imports(path, root, package)
            if _root_of(target) == package
        )
    return graph


def _reachable(graph: Mapping[str, frozenset[str]], start: str) -> frozenset[str]:
    seen: set[str] = set()
    pending = [start]
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        pending.extend(graph.get(node, ()))
    return frozenset(seen)


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------


class OrchestratorFacadeTests(unittest.TestCase):
    """Invariant 10: `orchestrator.py` is a façade, never a phase owner."""

    def test_orchestrator_stays_a_thin_facade(self) -> None:
        size = _line_count(ORCHESTRATOR)
        self.assertLessEqual(
            size, ORCHESTRATOR_MAX_LINES,
            f"orchestrator.py is {size} lines: extract the phase it grew into an "
            f"orchestration service instead of raising the façade budget",
        )

    def test_the_dormant_execute_alias_stays_deleted(self) -> None:
        for node in ast.walk(_parse(ORCHESTRATOR)):
            if isinstance(node, ast.ClassDef) and node.name == "Orchestrator":
                assigned = {
                    target.id
                    for statement in node.body
                    if isinstance(statement, ast.Assign)
                    for target in statement.targets
                    if isinstance(target, ast.Name)
                }
                self.assertNotIn("execute", assigned, "Orchestrator.execute is a deleted alias")

    def test_orchestrator_owns_no_protocol_parsing(self) -> None:
        tree = _parse(ORCHESTRATOR)
        imports = _resolved_imports(ORCHESTRATOR, PACKAGE, "metaharness")
        for lineno, target, _ in imports:
            for forbidden in ORCHESTRATOR_FORBIDDEN_IMPORTS:
                self.assertFalse(
                    target == forbidden or target.startswith(forbidden + "."),
                    f"orchestrator.py:{lineno} imports the parser authority {target}",
                )
        self.assertNotIn(
            "re", {target for _, target, _ in imports},
            "the façade owns no grammar, and therefore no regex engine",
        )
        for lineno, name in _called_names(tree):
            self.assertFalse(
                name.startswith(PARSER_PREFIXES), f"orchestrator.py:{lineno} parses {name} itself",
            )
        for lineno, text in _code_strings(tree):
            for marker in PROTOCOL_MARKERS:
                self.assertNotIn(
                    marker, text.upper(),
                    f"orchestrator.py:{lineno} carries the wire marker {marker!r}",
                )


class ModuleSizeTests(unittest.TestCase):
    """Invariant 9: the split packages stay bounded, smallest first."""

    def _split_modules(self) -> list[Path]:
        return sorted(path for package in SPLIT_PACKAGES for path in package.rglob("*.py"))

    def test_new_modules_fit_the_budget(self) -> None:
        for path in self._split_modules():
            name = path.relative_to(PACKAGE).as_posix()
            if name in FROZEN_MODULE_SIZES:
                continue
            size = _line_count(path)
            self.assertLessEqual(
                size, MODULE_MAX_LINES,
                f"{name} is {size} lines: split it by authority before landing it",
            )

    def test_frozen_modules_only_shrink(self) -> None:
        for name, ceiling in sorted(FROZEN_MODULE_SIZES.items()):
            path = PACKAGE / name
            self.assertTrue(path.is_file(), f"{name} disappeared: update the frozen table")
            size = _line_count(path)
            self.assertLessEqual(
                size, ceiling,
                f"{name} grew from {ceiling} to {size} lines: extract an authority from it, "
                f"or take the conscious decision of raising its frozen ceiling",
            )

    def test_pipeline_test_modules_fit_the_budget(self) -> None:
        for path in sorted(PIPELINE_TESTS.rglob("*.py")):
            size = _line_count(path)
            self.assertLessEqual(
                size, PIPELINE_TEST_MAX_LINES,
                f"{path.relative_to(ROOT)} is {size} lines: a journey test belongs next to "
                f"the module-owned invariants it covers",
            )


class PrivateImportTests(unittest.TestCase):
    """Invariant 7 (and 12): a `_private_symbol` never crosses a package."""

    def test_no_private_symbol_crosses_a_package_boundary(self) -> None:
        violations = [
            (importer, target, name, lineno)
            for importer, target, names, lineno in _private_imports()
            for name in names
            if _root_of(importer) == _root_of(target)
            and _package_of(importer) != _package_of(target)
        ]
        self.assertEqual(
            violations, [],
            "private symbols crossed a package boundary: make them public where they are "
            "defined, or move the consumer next to its authority",
        )

    def test_no_test_reaches_into_a_facade_private_symbol(self) -> None:
        violations = [
            (importer, name, lineno)
            for importer, target, names, lineno in _private_imports()
            if target == "metaharness.orchestrator"
            for name in names
        ]
        self.assertEqual(violations, [], "a test depends on a private symbol of the façade")

    def test_frozen_private_imports_only_shrink(self) -> None:
        frozen = {
            (importer, target, name)
            for importer, targets in FROZEN_PRIVATE_IMPORTS.items()
            for target, names in targets.items()
            for name in names
        }
        current = {
            (importer, target, name)
            for importer, target, names, _ in _private_imports()
            for name in names
        }
        extra = sorted(current - frozen)
        self.assertEqual(
            extra, [],
            "new cross-module private imports: the toolbox debt of the refoundation may "
            "only shrink; expose a public name instead",
        )


class AuthorityLayeringTests(unittest.TestCase):
    """Invariants 5, 6, 8 and 10: each authority keeps its own dependencies."""

    def test_planning_protocol_depends_on_the_frozen_vocabulary_only(self) -> None:
        for lineno, target, _ in _resolved_imports(PLANNING_PROTOCOL, PACKAGE, "metaharness"):
            top, leaf = target.split(".")[0], target.split(".")[-1]
            for name in (top, leaf):
                self.assertNotIn(
                    name, PROTOCOL_FORBIDDEN_MODULES,
                    f"planning/protocol.py:{lineno} imports {target}: the grammar is pure",
                )
            if target.startswith("metaharness."):
                self.assertIn(
                    target, PROTOCOL_ALLOWED_INTERNAL_IMPORTS,
                    f"planning/protocol.py:{lineno} imports {target}: the parser may only "
                    f"read the frozen vocabulary and its step-id policy",
                )

    def test_recovery_policy_is_a_pure_deterministic_leaf(self) -> None:
        for lineno, target, _ in _resolved_imports(RECOVERY_POLICY, PACKAGE, "metaharness"):
            top, leaf = target.split(".")[0], target.split(".")[-1]
            self.assertIn(
                top, sys.stdlib_module_names,
                f"recovery_policy.py:{lineno} imports {target}: the classification is "
                f"stdlib-only, so no executor, model or store can influence it",
            )
            for name in (top, leaf):
                self.assertNotIn(
                    name, RECOVERY_POLICY_FORBIDDEN_MODULES,
                    f"recovery_policy.py:{lineno} imports {target}",
                )

    def test_core_state_code_imports_no_runtime_authority(self) -> None:
        for name in CORE_STATE_MODULES:
            allowed = CORE_STATE_ALLOWED_INTERNAL_IMPORTS[name]
            for lineno, target, _ in _resolved_imports(PACKAGE / name, PACKAGE, "metaharness"):
                leaf = target.split(".")[-1]
                self.assertNotIn(
                    leaf, CORE_STATE_FORBIDDEN_MODULES,
                    f"{name}:{lineno} imports {target}: the state machine and its store "
                    f"own no Git, web or agent runtime",
                )
                if target.startswith("metaharness."):
                    self.assertIn(target, allowed, f"{name}:{lineno} imports {target}")

    def test_pipeline_v2_coordinator_never_reaches_the_orchestrator(self) -> None:
        coordinator = PACKAGE / "orchestration" / "pipeline_v2.py"
        for lineno, target, _ in _resolved_imports(coordinator, PACKAGE, "metaharness"):
            self.assertNotEqual(
                target, "metaharness.orchestrator",
                f"pipeline_v2.py:{lineno} imports the façade: the coordinator sequences "
                f"only the injected PipelineV2Operations",
            )
        closure = _reachable(_import_graph(PACKAGE, "metaharness"), "metaharness.orchestration.pipeline_v2")
        self.assertNotIn(
            "metaharness.orchestrator", closure,
            "the coordinator reaches the façade transitively",
        )
        referenced = {name for _, name in _identifiers(_parse(coordinator))}
        self.assertNotIn("Orchestrator", referenced, "the coordinator names the façade")


class CompatibilityBreakTests(unittest.TestCase):
    """Invariants 1 and 2: one current format, no revived compatibility path."""

    def test_no_runtime_identifier_reintroduces_a_removed_branch(self) -> None:
        for path in sorted(PACKAGE.rglob("*.py")):
            where = path.relative_to(ROOT)
            for lineno, name in _identifiers(_parse(path)):
                lowered = name.lower()
                for token in REMOVED_IDENTIFIER_TOKENS:
                    self.assertNotIn(
                        token, lowered,
                        f"{where}:{lineno} reintroduces a {token!r} branch named {name!r}: "
                        f"old runs stay readable through the current reader only",
                    )

    def test_deleted_compatibility_symbols_stay_deleted(self) -> None:
        for path in sorted(PACKAGE.rglob("*.py")):
            relative = path.relative_to(PACKAGE)
            forbidden = set(REMOVED_COMPATIBILITY_SYMBOLS)
            for package, names in REMOVED_SYMBOLS_PER_PACKAGE.items():
                if relative.parts[0] == package:
                    forbidden.update(names)
            for lineno, name in _identifiers(_parse(path)):
                self.assertNotIn(
                    name, forbidden, f"{relative}:{lineno} revives the deleted symbol {name!r}",
                )

    def test_deleted_migration_labels_stay_deleted(self) -> None:
        for path in sorted(PACKAGE.rglob("*.py")):
            where = path.relative_to(ROOT)
            for lineno, text in _string_literals(_parse(path)):
                for label in REMOVED_COMPATIBILITY_LITERALS:
                    self.assertNotIn(
                        label, text,
                        f"{where}:{lineno} revives the deleted migration label {label!r}",
                    )


class RunStateCommandTests(unittest.TestCase):
    """Invariant 13: the machine state is the only thing a caller commands.

    A status is a projection of ``(phase, disposition, reason)``: no service
    picks one to pilot a run, and the store keeps no command that would let it.
    """

    # The command surface of the store: every one of them either refuses a
    # control field or moves the machine state through ``transition``.
    STATE_COMMANDS = (
        "update", "update_metadata", "set_run_state", "transition_run", "record_failure",
    )
    REMOVED_STATUS_COMMANDS = ("update_if_status", "transition_if")

    def test_the_status_command_api_stays_deleted(self) -> None:
        for path in sorted(PACKAGE.rglob("*.py")):
            where = path.relative_to(ROOT)
            for lineno, name in _identifiers(_parse(path)):
                self.assertNotIn(
                    name, self.REMOVED_STATUS_COMMANDS,
                    f"{where}:{lineno} revives {name}: a claim compares the canonical "
                    f"RunIdentity, never a status spelling",
                )

    def test_no_service_commands_the_store_with_a_status(self) -> None:
        for path in sorted(PACKAGE.rglob("*.py")):
            where = path.relative_to(ROOT)
            for node in ast.walk(_parse(path)):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not isinstance(func, ast.Attribute) or func.attr not in self.STATE_COMMANDS:
                    continue
                receiver = ast.unparse(func.value).casefold()
                if "store" not in receiver and "state" not in receiver:
                    continue
                for keyword in node.keywords:
                    self.assertNotIn(
                        keyword.arg, {"status", "disposition", "phase", "reason"},
                        f"{where}:{node.lineno} passes {keyword.arg}= to {func.attr}: the "
                        f"machine state is written through RunMachineState, not named",
                    )


if __name__ == "__main__":  # pragma: no cover - unittest entry point
    unittest.main()
