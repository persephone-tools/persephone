from setuptools import setup

setup(name='persephone',
      version='0.0.1',
      description='A tool for developing automatic phoneme transcription models',
      long_description=open('README.md').read(),
      url='https://github.com/oadams/persephone',
      author='Oliver Adams',
      author_email='oliver.adams@gmail.com',
      license='GPLv3',
      packages=['persephone'],
      classifiers = [
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
      ],
      keywords='',
      install_requires=[
           'nltk',
           'numpy',
           'python-speech-features',
           'scipy',
           'tensorflow',
      ],
)
