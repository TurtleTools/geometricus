from os import path

from setuptools import setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

DISTNAME = "geometricus"
DESCRIPTION = "Fast, structure-based, alignment-free protein embedding"
LONG_DESCRIPTION = long_description
MAINTAINER = "Janani Durairaj, Mehmet Akdel"
MAINTAINER_EMAIL = "janani.durairaj@unibas.ch"
URL = "https://github.com/TurtleTools/geometricus"
LICENSE = "MIT License"
DOWNLOAD_URL = "https://github.com/TurtleTools/geometricus"
VERSION = "0.3.0"
PYTHON_VERSION = (3, 7)
INST_DEPENDENCIES = ["numpy==1.21.5", "scipy", "numba==0.55.1", "prody"]

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        version=VERSION,
        url=URL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        license=LICENSE,
        classifiers=[
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Topic :: Scientific/Engineering",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
        ],
        packages=["geometricus"],
        package_data={},
        install_requires=INST_DEPENDENCIES,
        long_description_content_type='text/markdown',
    )
