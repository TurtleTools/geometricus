# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2022-04-19

- Added chiral invariant moment from https://royalsocietypublishing.org/doi/10.1098/rsif.2010.0297
- Made MomentInvariants object pickle-able
- Froze numpy and numba versions

## [0.2.0] - 2020-10-16

Added more third order moment invariants (phi_{2-13} from [1]). These can be chosen via the `moment_types` argument
in `MomentInvariants`' constructors

[1] Flusser, Jan, Tomas Suk, and Barbara Zitová. 2D and 3D image analysis by moments. John Wiley & Sons, 2016.

## [0.1.2] - 2020-09-06

Fixed PyPy readme, added badge

## [0.1.1] - 2020-09-06

Linked readme to PyPy. Updated `pip install` instructions

## [0.1.0] - 2020-09-06

First pip package release


[Unreleased]: https://github.com/TurtleTools/geometricus/compare/v0.3.0...HEAD

[0.3.0]: https://github.com/TurtleTools/geometricus/compare/v0.2.0...v0.3.0

[0.2.0]: https://github.com/TurtleTools/geometricus/compare/v0.1.2...v0.2.0

[0.1.2]: https://github.com/TurtleTools/geometricus/compare/v0.1.1...v0.1.2

[0.1.1]: https://github.com/TurtleTools/geometricus/compare/v0.1.0...v0.1.1

[0.1.0]: https://github.com/TurtleTools/geometricus/releases/tag/v0.1.0
