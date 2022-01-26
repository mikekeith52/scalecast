import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'scalecast'
copyright = '2022, Michael Keith'
author = 'Michael Keith'

# The full version, including alpha/beta/rc tags
release = '0.5.6'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.napoleon", "autodocsumm", "nbsphinx", "myst_parser", "m2r2", "sphinxcontrib.confluencebuilder"]
autodoc_default_options = {"autosummary": True}

#source_suffix = ['.rst', '.md', '.pdf']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "haiku"
html_theme_options = {
    "full_logo": "true"
}

html_favicon = '../assets/logo2.png'
html_logo = '../assets/logo2.png'
html_static_path = ['_static']