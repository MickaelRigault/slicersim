# Sphinx configuration for the slicersim documentation.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import datetime
import os
import sys

# Document the source tree (and not an older installed version).
sys.path.insert(0, os.path.abspath("../src"))

import slicersim  # noqa: E402

# -- Project information -----------------------------------------------------
project = "slicersim"
author = "Mickael Rigault"
copyright = f"2024-{datetime.date.today().year}, {author}"
release = slicersim.__version__
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
    "myst_nb",
    "sphinx_design",
    "sphinx_copybutton",
]

master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "notebooks/extra",
    "notebooks/gemini",
]

# `foo` in docstrings is a cross-reference to the Python object foo.
default_role = "py:obj"

# -- API (autodoc / autosummary / napoleon) ----------------------------------
autosummary_generate = True
autosummary_imported_members = False
autodoc_member_order = "groupwise"
autoclass_content = "class"
autodoc_typehints = "none"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
add_module_names = False
toc_object_entries_show_parents = "hide"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "astropy": ("https://docs.astropy.org/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "sncosmo": ("https://sncosmo.readthedocs.io/en/stable", None),
}

# Figures produced by ``.. plot::`` directives in docstrings.
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 120)]

# -- Notebooks (myst-nb) -----------------------------------------------------
# Notebooks are stored with their outputs and are not re-executed.
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "dollarmath"]

# -- Copy button: strip prompts ----------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_book_theme"
html_title = "slicersim"
html_logo = "_static/slicersim_logo.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "repository_url": "https://github.com/MickaelRigault/slicersim",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "show_toc_level": 2,
    "show_navbar_depth": 1,
    "home_page_in_toc": False,
    "navigation_with_keys": False,
}
