from distutils.core import setup

setup(
    name='geometricus',
    version='1.0',
    packages=["geometricus"],
    install_requires=["scipy", "numba", "prody"],
    url="https://github.com/TurtleTools/geometricus",
    license='MIT License',
    author='Janani Durairaj & Mehmet Akdel',
    description='Protein structure embedding method and library'
)
