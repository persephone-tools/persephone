# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.1] - 2019-11-13

- Bugfix for an assertion that checked for equality of floating points.
- Bugfix where reference transcription wasn't written.

## [0.4.0] - 2019-08-31

### Added
- Python 3.7 compatibility
- Raise a label mismatch exception if label kwarg to Corpus constructor is inconsistent with automatically determined labels.
- Test fixtures for Corpus creation
- Test coverage for Corpus and Model creation
- Callback at the end of each training epoch
- JSON output of model information
- Encoding of files is now handled explicitly
- Type annotations have been added to most functions that previously were missing them

### Changed
- Update package dependencies versions.

### Fixed
- `CorpusReader.train_batch_gen` now correctly handles edge case when no data can be generated.
- Decoding from saved model is now possible for arbitrary Tensorflow model topologies that have the same input and output structure via named arguments that specify where input and output to the model occur.
- RNN CTC model class now accepts `pathlib.Path` for directory argument
- Max epochs for model training is now correct. Previously there was an off by one error where one more than the supplied max epochs would be run in the training loop.
- Bug where `untranscribed_prefixes` in corpus was taking an intersection of two sets instead of a union.
- Splitting of test, train and validation data sets will no longer produce empty sets. If no possible split can be made it will report the error via raising an exception.
- Empty wave files no longer crash on attempted feature extraction and are now skipped instead.
- Update nltk dependency to resolve possible security issue

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

