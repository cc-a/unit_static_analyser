from pathlib import Path

import pytest

from unit_static_analyser.mypy_checker.checker import UnitChecker


def test_file(tmp_path: Path):
    """Test discovery logic for a single file."""
    file_path = Path("file1.py")
    full_file_path = tmp_path / file_path
    full_file_path.touch()

    files = UnitChecker._find_py_files_and_modules([full_file_path])
    assert len(files) == 1
    assert files[0] == (full_file_path, file_path.with_suffix("").name)


def test_dir(tmp_path: Path):
    """Test discover logic for a single directory containing a file."""
    file_path = Path("file1.py")
    full_file_path = tmp_path / file_path
    full_file_path.touch()

    files = UnitChecker._find_py_files_and_modules([tmp_path])
    assert len(files) == 1
    assert files[0] == (full_file_path, file_path.with_suffix("").name)


def _path_to_module_string(path: Path) -> str:
    return path.with_suffix("").as_posix().replace("/", ".")


@pytest.fixture
def package_directories(tmp_path: Path) -> dict[str, Path]:
    """Create a package directory structure with sub packages."""
    pkg_dir = Path("pkg")
    subpkg1_dir = pkg_dir / "sub1"
    subpkg2_dir = pkg_dir / "sub2"
    pkg_dirs = {
        _path_to_module_string(dir_path): dir_path
        for dir_path in (pkg_dir, subpkg1_dir, subpkg2_dir)
    }
    for d in pkg_dirs.values():
        full_dir_path = tmp_path / d
        full_dir_path.mkdir()
        (full_dir_path / "__init__.py").touch()
    return pkg_dirs


@pytest.fixture
def package_files(tmp_path: Path, package_directories: dict[str, Path]):
    """Create files within the package directory structure."""
    pkg_file_paths = [
        dir_path / f"file{i}.py"
        for i, dir_path in enumerate(package_directories.values())
    ]
    pkg_files = {
        _path_to_module_string(file_path): file_path for file_path in pkg_file_paths
    }
    for file_path in pkg_files.values():
        (tmp_path / file_path).touch()
    return pkg_files


def test_full_package(tmp_path: Path, package_files: dict[str, Path]):
    """Test file discovery when passed the top level directory of a package."""
    files = UnitChecker._find_py_files_and_modules([tmp_path])
    assert len(files) == 6
    for module_name, file_path in package_files.items():
        assert (tmp_path / file_path, module_name) in files


def test_package_parts(
    tmp_path: Path, package_directories: dict[str, Path], package_files: dict[str, Path]
):
    """Test file discovery when passed a subset of a package's files."""
    dir_module_name = "pkg.sub1"
    file_module_name = "pkg.sub2.file2"
    files = UnitChecker._find_py_files_and_modules(
        [
            tmp_path / package_directories[dir_module_name],
            tmp_path / package_files[file_module_name],
        ]
    )
    assert len(files) == 3
    assert (tmp_path / package_files[file_module_name], file_module_name) in files


def test_init(tmp_path: Path, package_directories: dict[str, Path]):
    """Test file discovery when directly passed an __init__.py file."""
    module_name = "pkg.sub1"
    init_file_path = tmp_path / package_directories[module_name] / "__init__.py"
    files = UnitChecker._find_py_files_and_modules([init_file_path])
    assert len(files) == 1
    assert (init_file_path, module_name) in files


def test_package_dependency(tmp_path: Path, package_files: dict[str, Path]):
    """Test that module discover follows import logic."""
    # Add a dependency on pkg.sub2.file2
    file1_module = "pkg.sub1.file1"
    file1_path = package_files[file1_module]
    file1_path.write_text("from ..sub2.file2 import a\n")

    # Add something to import to pkg2.sub2.file2
    file2_module = "pkg.sub2.file2"
    file2_path = package_files[file2_module]
    file2_path.write_text("a: int = 4\n")

    # run through the file discover and processing steps
    files_and_modules = UnitChecker._find_py_files_and_modules([tmp_path / file1_path])
    requested_modules = [module_name for _, module_name in files_and_modules]
    top_level_modules = UnitChecker._get_top_level_modules(requested_modules)
    build_result = UnitChecker._mypy_build(files_and_modules, top_level_modules)
    module_order = UnitChecker.topological_sort_modules(
        build_result.graph, requested_modules
    )
    modules_to_check = UnitChecker._get_modules_for_unit_analysis(
        module_order, top_level_modules
    )

    # ensure both files are checked with file2 before file1
    assert modules_to_check == [file2_module, file1_module]
