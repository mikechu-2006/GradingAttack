"""Optional baseline exports.

Importing the package should not require every optional attack backend. Some
pipeline runs only need defenses, while GCG/RolePlay/Injection may depend on
extra packages that are not installed in all environments.
"""

for _module in ("gcg", "roleplay", "injection"):
    try:
        exec(f"from .{_module} import *")
    except ModuleNotFoundError:
        pass