"""Builds the C library with cargo and puts it inside the package, where `chassis._ffi` looks for
it first, so a wheel carries everything it needs. Building needs Rust and the repository around
this directory."""

import os
import platform
import shutil
import subprocess
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent.parent
LIBRARY = {"Linux": "libchassis_ffi.so", "Darwin": "libchassis_ffi.dylib", "Windows": "chassis_ffi.dll"}


class BuildWithLibrary(build_py):
    def run(self):
        super().run()
        # An editable install finds the library in ../target/release, as development does.
        if getattr(self, "editable_mode", False):
            return
        subprocess.run(["cargo", "build", "--release", "-p", "chassis-ffi"], cwd=ROOT, check=True)
        target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
        name = LIBRARY[platform.system()]
        package = Path(self.build_lib) / "chassis"
        package.mkdir(parents=True, exist_ok=True)
        shutil.copy(target / "release" / name, package / name)


class WithLibrary(Distribution):
    """A package with a native library: its wheel is for one platform."""

    def has_ext_modules(self):
        return True


class PlatformWheel(bdist_wheel):
    """One wheel per platform, for any Python 3: the library is loaded with ctypes, not built
    against Python."""

    def get_tag(self):
        return "py3", "none", super().get_tag()[2]


setup(distclass=WithLibrary, cmdclass={"build_py": BuildWithLibrary, "bdist_wheel": PlatformWheel})
