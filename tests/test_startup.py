from __future__ import annotations

import os
import subprocess
from importlib import metadata
from pathlib import Path

import pytest

from scripts.check_runtime import requirement_errors

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_requirement_check_covers_missing_and_incompatible_packages() -> None:
    versions = {"streamlit": "1.63.0", "xgboost": "2.0.0", "google-auth": "2.40.0"}

    def lookup(name):
        if name not in versions:
            raise metadata.PackageNotFoundError(name)
        return versions[name]

    errors = requirement_errors(
        "# runtime\nstreamlit>=1.41,<2\nxgboost>=3.1,<4\nearthengine-api>=1.5,<2\ngoogle-auth>=2,<3\n",
        lookup,
    )
    assert len(errors) == 2
    assert any("xgboost 2.0.0" in message for message in errors)
    assert any("Missing: earthengine-api" in message for message in errors)


def test_requirement_check_enforces_upper_bound_and_environment_markers() -> None:
    assert requirement_errors("numpy>=1.26,<3", lambda _: "3.0.0")
    assert requirement_errors("numpy>=1.26,<3", lambda _: "2.5.2") == []
    assert requirement_errors("unavailable>=1; python_version < '2'", lambda _: "0") == []


def _stub_launcher(
    tmp_path,
    *,
    existing_environment=True,
    dependency_failure=False,
    py312_available=True,
    py311_available=True,
    compatible_python=True,
):
    project = tmp_path / "project with spaces"
    project.mkdir()
    (project / "scripts").mkdir()
    (project / "app.py").touch()
    (project / "requirements.txt").touch()
    (project / "scripts" / "check_runtime.py").touch()
    if existing_environment:
        (project / ".venv" / "Scripts").mkdir(parents=True)
        (project / ".venv" / "Scripts" / "python.exe").touch()
    # Exercise the real batch branches while replacing ALL Python commands with
    # harmless command stubs. These tests cannot install packages or run a server.
    launcher = (PROJECT_ROOT / "run_windows.bat").read_text(encoding="utf-8")
    launcher = launcher.replace('\n".venv\\Scripts\\python.exe"', '\ncall "%~dp0fake_python.cmd"')
    launcher = launcher.replace("\npy ", '\ncall "%~dp0fake_py.cmd" ')
    launcher = launcher.replace("\npython ", '\ncall "%~dp0fake_python.cmd" ')
    (project / "run_windows.bat").write_text(launcher, encoding="utf-8")
    check_exit = "1" if dependency_failure else "0"
    install_exit = "1" if dependency_failure else "0"
    python_exit = "0" if compatible_python else "1"
    python_stub = (
        "@echo off\n"
        'echo CWD=%CD% ARGS=%*>>"%~dp0calls.log"\n'
        f'if "%~1"=="-c" exit /b {python_exit}\n'
        f'if "%~1"=="scripts\\check_runtime.py" exit /b {check_exit}\n'
        'if "%~2"=="pip" if "%~3"=="--version" exit /b 0\n'
        f'if "%~2"=="pip" if "%~3"=="install" exit /b {install_exit}\n'
        'if "%~2"=="streamlit" exit /b 0\n'
        'if "%~2"=="venv" exit /b 0\n'
        "exit /b 99\n"
    )
    (project / "fake_python.cmd").write_text(python_stub, encoding="utf-8")
    py_stub = (
        "@echo off\n"
        'echo PY=%*>>"%~dp0calls.log"\n'
        + ('if "%~1"=="-3.12" exit /b 1\n' if not py312_available else "")
        + ('if "%~1"=="-3.11" exit /b 1\n' if not py311_available else "")
        + "exit /b 0\n"
    )
    (project / "fake_py.cmd").write_text(py_stub, encoding="utf-8")
    return project


def _run_stub(project: Path, cwd: Path):
    return subprocess.run(
        [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", str(project / "run_windows.bat")],
        cwd=cwd,
        input="\n",
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows batch launcher")
def test_launcher_uses_project_directory_and_preserves_valid_environment(tmp_path) -> None:
    project = _stub_launcher(tmp_path)
    result = _run_stub(project, tmp_path)
    calls = (project / "calls.log").read_text()
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"CWD={project}" in calls
    assert '-m streamlit run "app.py"' in calls
    assert "pip install" not in calls
    assert "-m venv" not in calls
    assert "KEEP THIS WINDOW OPEN" in result.stdout
    assert (project / ".venv" / "Scripts" / "python.exe").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows batch launcher")
def test_launcher_retains_dependency_failure_and_does_not_start_server(tmp_path) -> None:
    project = _stub_launcher(tmp_path, dependency_failure=True)
    result = _run_stub(project, tmp_path)
    calls = (project / "calls.log").read_text()
    assert result.returncode == 1
    assert "Dependency setup failed" in result.stdout
    assert "Press any key" in result.stdout
    assert "pip install" in calls
    assert "-m streamlit" not in calls
    assert (project / ".venv" / "Scripts" / "python.exe").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows batch launcher")
@pytest.mark.parametrize("py312_available, selected", [(True, "-3.12"), (False, "-3.11")])
def test_launcher_discovers_preferred_python_before_creation(tmp_path, py312_available, selected) -> None:
    project = _stub_launcher(tmp_path, existing_environment=False, py312_available=py312_available)
    result = _run_stub(project, tmp_path)
    calls = (project / "calls.log").read_text()
    assert result.returncode == 0, result.stdout + result.stderr
    assert f'PY={selected} -m venv ".venv"' in calls
    assert "-m streamlit run" in calls


@pytest.mark.skipif(os.name != "nt", reason="Windows batch launcher")
def test_launcher_refuses_to_overwrite_incomplete_environment(tmp_path) -> None:
    project = _stub_launcher(tmp_path, existing_environment=False)
    (project / ".venv").mkdir()
    marker = project / ".venv" / "keep.txt"
    marker.write_text("existing user environment", encoding="utf-8")
    result = _run_stub(project, tmp_path)
    assert result.returncode == 1
    assert "existing .venv was found" in result.stdout
    assert marker.read_text(encoding="utf-8") == "existing user environment"
    assert not (project / "calls.log").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows batch launcher")
def test_launcher_falls_back_to_compatible_python_on_path(tmp_path) -> None:
    project = _stub_launcher(tmp_path, existing_environment=False, py312_available=False, py311_available=False)
    result = _run_stub(project, tmp_path)
    calls = (project / "calls.log").read_text()
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'ARGS=-m venv ".venv"' in calls
    assert "-m streamlit run" in calls


@pytest.mark.skipif(os.name != "nt", reason="Windows batch launcher")
def test_launcher_preserves_incompatible_existing_python_without_installing(tmp_path) -> None:
    project = _stub_launcher(tmp_path, compatible_python=False)
    result = _run_stub(project, tmp_path)
    calls = (project / "calls.log").read_text()
    assert result.returncode == 1
    assert "existing .venv is broken" in result.stdout
    assert "-m venv" not in calls
    assert "pip install" not in calls
    assert "-m streamlit" not in calls
    assert (project / ".venv" / "Scripts" / "python.exe").exists()
