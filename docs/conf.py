import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

import sphinx_rtd_theme
from scalecast import __version__ as version

# -- Project information -----------------------------------------------------

project = 'scalecast'
copyright = '2022, Michael Keith'
author = 'Michael Keith'

# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.napoleon", "autodocsumm", "nbsphinx",
    "myst_parser", "sphinxcontrib.confluencebuilder"]
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
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': True,
    'navigation_depth': 5,
    'style_nav_header_background':'black',
    'prev_next_buttons_location':'both'
}
html_context = {}

html_favicon = './_static/logo2.png'
html_logo = './_static/logo2.png'
html_static_path = ['_static']