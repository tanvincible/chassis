"""The old name of `chassisdb`, kept for one release: `import chassisdb` instead."""

import sys
import warnings

import chassisdb
from chassisdb import *  # noqa: F401,F403
from chassisdb import __all__, __version__  # noqa: F401

warnings.warn(
    "The chassis module is now chassisdb, as the package is named; `import chassis` stops "
    "working in a later release\nhelp: import chassisdb instead",
    DeprecationWarning,
    stacklevel=2,
)

# `from chassis.exceptions import ...` and `chassis.exceptions.ChassisError` keep working too.
for _name in ("index", "exceptions", "_ffi"):
    globals()[_name] = sys.modules[f"{__name__}.{_name}"] = getattr(chassisdb, _name)
