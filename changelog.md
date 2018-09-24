# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Raise a label mismatch exception if label kwarg to Corpus constructor is inconsistent with automatically determined labels.
- Test fixtures for Corpus creation
- Test coverage for Corpus and Model creation

### Changed
- Update package dependencies versions.

### Fixed
- `CorpusReader.train_batch_gen` raises StopIteration instead of returning None if no data can be generated.
- Decoding from saved model is now possible for arbitrary Tensorflow model topologies that have the same input and output structure via named arguments that specify where input and output to the model occur.
- RNN CTC model class now accepts `pathlib.Path` for directory argument

## [0.3.2]

### Added
- Changelog

### Fixed
- Fixed bug where batch sizes were not integers
- Corpus construction from Elan regression was fixed

## [0.3.1] - 2018-07-14

### Fixed
- Documentation for tutorial running
- Pathlib handling for parameters

## [0.3.0] - 2018-07-14

### Added
- More mypy type annotations
- More test coverage

### Removed
- Removed `ReadyCorpus` in PR #163 (https://github.com/persephone-tools/persephone/pull/163)

