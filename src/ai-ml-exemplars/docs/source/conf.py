# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

HERE_ROOT = os.path.abspath(os.path.dirname(__file__))
print(HERE_ROOT)
REPO_ROOT = os.path.abspath(os.path.join(__file__, "..", "..", "..","..","..",".."))
print(REPO_ROOT)
SRC_DIR = os.path.join(REPO_ROOT, "src","ai-ml-exemplars")
print(SRC_DIR)
sys.path.insert(0, SRC_DIR) # project root

import LDDashRagChatbot

print(LDDashRagChatbot.__file__)

os.environ["SPHINX_BUILD"] = "1"

project = 'LDDashRagChatbot'
copyright = '2026, Yuvi Durairaj'
author = 'Yuvi Durairaj'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",      # supports Google/NumPy docstrings
    "sphinx.ext.viewcode",      # link to source
    "sphinx.ext.autosummary",   # generates summary pages
    "myst_parser",              # optional (Markdown support)
]

autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
