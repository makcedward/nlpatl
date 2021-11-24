from setuptools import setup, find_packages
import sys

if sys.version_info < (3,):
    sys.exit("Sorry, Python3 is required.")

with open("README.md", encoding="utf8") as f:
    readme = f.read()

with open('requirements.txt') as f:
    install_reqs = f.read().splitlines()

setup(
    name="nlpatl",
    version="0.0.1",
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
    keywords=[
        "deep learning", "neural network", "machine learning",
        "nlp", "natural language processing", "text",
        "active learning", "data labeling", "ai", "ml"]
)
