from pathlib import Path

import pytest

from unit_static_analyser.checker.checker import UnitChecker
from unit_static_analyser.units import Unit

m_unit = Unit.from_string("m")


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
    (tmp_path / file1_path).write_text("from ..sub2.file2 import a\n")

    # Add something to import to pkg2.sub2.file2
    file2_module = "pkg.sub2.file2"
    file2_path = package_files[file2_module]
    (tmp_path / file2_path).write_text("a: int = 4\n")

    # run through the file discover and processing steps
    files_and_modules = UnitChecker._find_py_files_and_modules([tmp_path / file1_path])
    requested_modules = [module_name for _, module_name in files_and_modules]
    top_level_modules = UnitChecker._get_top_level_modules(requested_modules)
    build_result = UnitChecker._mypy_build(files_and_modules)
    module_order = UnitChecker.topological_sort_modules(
        build_result.graph, requested_modules
    )
    modules_to_check = UnitChecker._get_modules_for_unit_analysis(
        module_order, top_level_modules
    )

    # ensure both files are checked with file2 before file1
    assert modules_to_check == [file2_module, file1_module]


def check_unit(
    checker: UnitChecker,
    obj_path: str,
    unit: Unit,
):
    """Helper function for checking a unit in the test module."""
    for var, var_unit in checker.units.items():
        if var.fullname == f"{obj_path}":
            assert unit == var_unit
            break
    else:
        raise ValueError(f"Missing unit - {obj_path}")


def test_multi_file_analysis_import_from(
    tmp_path: Path, package_files: dict[str, Path]
):
    """Test that units are correctly propagated across multiple files with imports."""
    file1 = package_files["pkg.sub1.file1"]
    (tmp_path / file1).write_text("""
from typing import Annotated
x: Annotated[int, "unit:m"] = 1
""")

    file2 = package_files["pkg.sub2.file2"]
    (tmp_path / file2).write_text("""
from pkg.sub1.file1 import x
y = x
""")

    # Analyze both files together
    checker = UnitChecker()
    checker.check([tmp_path / file1, tmp_path / file2])

    # Now you can assert units for symbols in both files
    check_unit(checker, "pkg.sub1.file1.x", m_unit)
    check_unit(checker, "pkg.sub2.file2.y", m_unit)


def test_multi_file_analysis_import_from_relative(
    tmp_path: Path, package_files: dict[str, Path]
):
    """Test that units are correctly propagated across multiple files with imports."""
    file1 = package_files["pkg.sub1.file1"]
    (tmp_path / file1).write_text("""
from typing import Annotated
x: Annotated[int, "unit:m"] = 1
""")

    file2 = package_files["pkg.sub2.file2"]
    (tmp_path / file2).write_text("""
from ..sub1.file1 import x
y = x
""")

    # Analyze both files together
    checker = UnitChecker()
    checker.check([tmp_path / file1, tmp_path / file2])

    # Now you can assert units for symbols in both files
    check_unit(checker, "pkg.sub1.file1.x", m_unit)
    check_unit(checker, "pkg.sub2.file2.y", m_unit)


def test_multi_file_analysis_import_from_class(
    tmp_path: Path, package_files: dict[str, Path]
):
    """Test that units are correctly propagated across multiple files with imports."""
    file1 = package_files["pkg.sub1.file1"]
    (tmp_path / file1).write_text("""
from typing import Annotated
class A:
    a: Annotated[int, "unit:m"] = 1
""")

    file2 = package_files["pkg.sub2.file2"]
    (tmp_path / file2).write_text("""
from ..sub1.file1 import A
y = A.a
""")

    # Analyze both files together
    checker = UnitChecker()
    checker.check([tmp_path / file1, tmp_path / file2])

    check_unit(checker, "pkg.sub1.file1.A.a", m_unit)
    check_unit(checker, "pkg.sub2.file2.y", m_unit)


def test_multi_file_analysis_import_from_as(
    tmp_path: Path, package_files: dict[str, Path]
):
    """Test that units are correctly propagated across multiple files with imports."""
    file1 = package_files["pkg.sub1.file1"]
    (tmp_path / file1).write_text("""
from typing import Annotated
x: Annotated[int, "unit:m"] = 1
""")

    file2 = package_files["pkg.sub2.file2"]
    (tmp_path / file2).write_text("""
from pkg.sub1.file1 import x as altx
y = altx
""")

    # Analyze both files together
    checker = UnitChecker()
    checker.check([tmp_path / file1, tmp_path / file2])

    # Now you can assert units for symbols in both files
    check_unit(checker, "pkg.sub1.file1.x", m_unit)
    check_unit(checker, "pkg.sub2.file2.y", m_unit)


def test_multi_file_analysis_import(tmp_path: Path, package_files: dict[str, Path]):
    """Test that units are correctly propagated through imports."""
    file1 = package_files["pkg.sub1.file1"]
    (tmp_path / file1).write_text("""
from typing import Annotated
x: Annotated[int, "unit:m"] = 1
""")

    file2 = package_files["pkg.sub2.file2"]
    (tmp_path / file2).write_text("""
import pkg.sub1.file1
y = pkg.sub1.file1.x
""")

    # Analyze both files together
    checker = UnitChecker()
    checker.check([tmp_path / file1, tmp_path / file2])

    # Now you can assert units for symbols in both files
    check_unit(checker, "pkg.sub1.file1.x", m_unit)
    check_unit(checker, "pkg.sub2.file2.y", m_unit)


def test_multi_file_analysis_import_as(tmp_path: Path, package_files: dict[str, Path]):
    """Test multi-file analysis with import as syntax."""
    # Write first file
    file1 = package_files["pkg.sub1.file1"]
    (tmp_path / file1).write_text("""
from typing import Annotated
x: Annotated[int, "unit:m"] = 1
""")

    # Write second file
    file2 = package_files["pkg.sub2.file2"]
    (tmp_path / file2).write_text("""
import pkg.sub1.file1 as alt
y = alt.x
""")

    # Analyze both files together
    checker = UnitChecker()
    checker.check([tmp_path / file1, tmp_path / file2])

    # Now you can assert units for symbols in both files
    check_unit(checker, "pkg.sub1.file1.x", m_unit)
    check_unit(checker, "pkg.sub2.file2.y", m_unit)


def test_multi_file_type_alias(tmp_path: Path, package_files: dict[str, Path]):
    """Check that type aliases can be imported."""
    file1 = package_files["pkg.sub1.file1"]
    (tmp_path / file1).write_text("""
from typing import TypeAlias, Annotated, TypeVar
T = TypeVar("T")
metres: TypeAlias = Annotated[T, "unit:m"]
""")
    file2 = package_files["pkg.sub2.file2"]
    (tmp_path / file2).write_text("""
from pkg.sub1.file1 import metres
a: metres[int] = 4
""")
    checker = UnitChecker()
    checker.check([tmp_path / file2])

    check_unit(checker, "pkg.sub2.file2.a", m_unit)


def test_multi_file_type_alias_import_as(
    tmp_path: Path, package_files: dict[str, Path]
):
    """Check that type aliases can be imported under another name."""
    file1 = package_files["pkg.sub1.file1"]
    (tmp_path / file1).write_text("""
from typing import TypeAlias, Annotated, TypeVar
T = TypeVar("T")
metres: TypeAlias = Annotated[T, "unit:m"]
""")
    file2 = package_files["pkg.sub2.file2"]
    (tmp_path / file2).write_text("""
from pkg.sub1.file1 import metres as something_else
a: something_else[int] = 4
""")
    checker = UnitChecker()
    checker.check([tmp_path / file2])

    check_unit(checker, "pkg.sub2.file2.a", m_unit)
