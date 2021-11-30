from setuptools import setup, find_packages
import sys
import re

if sys.version_info < (3,):
    sys.exit("Sorry, Python3 is required.")

with open("README.md", encoding="utf8") as f:
    readme = f.read()

# Dependencies
_deps = [
    'scipy>=1.7.0',
    'numpy>=1.16.2',
    'transformers>=4.11.3',
    'scikit-learn>=1.0.1',
    'xgboost>=1.5.1',
    'datasets>=1.8.0',
    'torch>=1.9.0',
    'tensorflow>=2.7.0'
]

# support both w and w/ version
deps = {b: a for a, b in (re.findall(r"^(([^!=<>]+)(?:[!=<>].*)?$)", x)[0] for x in _deps)}

def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]

install_reqs = deps_list('scipy', 'numpy', 'scikit-learn', 'transformers')

extra_reqs = {}
extra_reqs['dev'] = deps_list('datasets', 'xgboost', 'torch', 'tensorflow')

extra_reqs['all'] = (
    install_reqs
    + extra_reqs['dev']
)

setup(
    name="nlpatl",
    version="0.0.2",
    author="Edward Ma",
    author_email="makcedward@gmail.com",
    url="https://github.com/makcedward/nlpatl",
    license="MIT",
    description="Natural language processing active learning library for deep neural networks",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude="test"),
    include_package_data=True,
    install_requires=install_reqs,
    extras_require=extra_reqs,
    keywords=[
        "deep learning", "neural network", "machine learning",
        "nlp", "natural language processing", "text",
        "active learning", "data labeling", "ai", "ml"]
)
