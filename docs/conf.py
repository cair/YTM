import dataclasses
import importlib.metadata
import types

project = "YTM"
release = importlib.metadata.version("ytm")
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
    "myst_parser",
    "sphinx_copybutton",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "_pages_root", "README.md"]
templates_path = ["_templates"]

autosummary_generate = True
autodoc_default_options = {"members": True, "inherited-members": True, "show-inheritance": True, "exclude-members": "__init__"}
autodoc_mock_imports = ["cupy", "pynvml"]
autodoc_preserve_defaults = True
numpydoc_show_class_members = False
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "switcher": {"json_url": "https://cair.github.io/YTM/stable/_static/switcher.json", "version_match": version},
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "check_switcher": False,
}


def _drop_dataclass_signature(app, what, name, obj, options, signature, return_annotation):
    if what == "class" and dataclasses.is_dataclass(obj):
        return "", return_annotation


def _skip_module_attrs(app, what, name, obj, skip, options):
    if isinstance(obj, types.ModuleType):
        return True


def setup(app):
    app.connect("autodoc-process-signature", _drop_dataclass_signature)
    app.connect("autodoc-skip-member", _skip_module_attrs)
