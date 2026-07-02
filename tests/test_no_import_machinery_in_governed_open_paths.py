"""Guards against re-entrant imports/file-reads inside governed open() paths.

Two invariants, both learned from a live worker crash + a runaway recursion:

1. Package ``__version__`` must be a static assignment — an
   ``importlib.metadata`` lookup OPENS A FILE, and with traced_open patching
   ``builtins.open``/``io.open`` that read re-enters governance (circular
   import inside the workflow sandbox when eager; unbounded recursion from
   ``build_auth_headers`` when lazy).
2. ``file_governance_hooks`` must not run import machinery inside functions
   that execute during arbitrary ``open()`` calls — a function-local
   ``from .types import …`` re-imports a partially initialized package when
   the triggering open() happens mid-package-import.
"""

import ast
from pathlib import Path

import openbox
import openbox_core

OPENBOX_PKG = Path(openbox.__file__).parent
CORE_PKG = Path(openbox_core.__file__).parent


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


class TestStaticVersion:
    def _assert_no_metadata_import(self, init_path: Path):
        tree = _module_ast(init_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "importlib" not in (node.module or ""), (
                    f"{init_path.name} imports {node.module} — __version__ must "
                    "be a static string (metadata lookups open files)"
                )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "importlib" not in alias.name, init_path.name

    def _assert_static_version(self, init_path: Path):
        tree = _module_ast(init_path)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if "__version__" in targets:
                    assert isinstance(node.value, ast.Constant) and isinstance(
                        node.value.value, str
                    ), "__version__ must be a static string literal"
                    return
        raise AssertionError(f"no static __version__ assignment in {init_path}")

    def test_openbox_version_static_no_metadata(self):
        init = OPENBOX_PKG / "__init__.py"
        self._assert_no_metadata_import(init)
        self._assert_static_version(init)

    def test_openbox_core_version_static_no_metadata(self):
        init = CORE_PKG / "__init__.py"
        self._assert_no_metadata_import(init)
        self._assert_static_version(init)

    def test_versions_match_pyproject(self):
        import tomllib

        for pkg_dir, mod in ((OPENBOX_PKG, openbox), (CORE_PKG, openbox_core)):
            pyproject = pkg_dir.parent / "pyproject.toml"
            declared = tomllib.loads(pyproject.read_text())["project"]["version"]
            assert mod.__version__ == declared, (
                f"{mod.__name__}.__version__ ({mod.__version__}) out of sync "
                f"with pyproject ({declared}) — update the static string on release"
            )


class TestNoFunctionLocalImportsInFileHooks:
    def test_no_relative_imports_inside_functions(self):
        tree = _module_ast(OPENBOX_PKG / "file_governance_hooks.py")
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.ImportFrom) and inner.level > 0:
                        offenders.append(
                            f"line {inner.lineno}: from {'.' * inner.level}"
                            f"{inner.module or ''} import …"
                        )
        assert not offenders, (
            "function-local relative imports in file_governance_hooks run "
            "import machinery inside governed open() calls (re-entrancy bomb "
            "when the open happens mid-package-import): " + "; ".join(offenders)
        )
