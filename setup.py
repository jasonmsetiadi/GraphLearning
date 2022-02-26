import setuptools
import numpy

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="graphlearning", 
    version="1.1.0",
    author="Jeff Calder",
    author_email="jwcalder@umn.edu",
    description="Python package for graph-based clustering and semi-supervised learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jwcalder/GraphLearning",
    packages=['graphlearning'],
    classifiers=[
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent"],
    install_requires=[  'numpy', 
                        'scipy', 
                        'sklearn', 
                        'matplotlib'],
    python_requires='>=3.6',
)


