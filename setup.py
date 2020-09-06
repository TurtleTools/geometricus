from setuptools import setup

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    
    
DISTNAME = "geometricus"
DESCRIPTION = "Fast, structure-based, alignment-free protein embedding"
LONG_DESCRIPTION = long_description
MAINTAINER = "Janani Durairaj, Mehmet Akdel"
MAINTAINER_EMAIL = "janani.durairaj@wur.nl"
URL = "https://github.com/TurtleTools/geometricus"
LICENSE = "MIT License"
DOWNLOAD_URL = "https://github.com/TurtleTools/geometricus"
VERSION = "0.1.1"
PYTHON_VERSION = (3, 7)
INST_DEPENDENCIES = ["scipy", "numba", "prody"]

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
    )
