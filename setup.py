#!/usr/bin/env python3

import setuptools
from pathlib import Path

package_name = "carefulmodels"

readme = Path("README.md").read_text()
requirements = Path("requirements.txt").read_text()

setuptools.setup(
    name=package_name,
    version="1",
    author="wwww-wwww",
    author_email="wvvwvvvvwvvw@gmail.com",
    maintainer="wwww-wwww",
    maintainer_email="wvvwvvvvwvvw@gmail.com",
    description="careful models",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/wwww-wwww/carefulmodels",
    install_requires=requirements,
    python_requires='>=3.11',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
