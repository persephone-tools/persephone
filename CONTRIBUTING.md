
Contributing to Persephone
==================================

The preferred workflow for contributing to Persephone is to fork the
main repository on GitHub, clone, and develop on a branch. Steps:

1. Fork the [project repository](https://github.com/oadams/persephone)
   by clicking on the 'Fork' button near the top right of the page. This creates
   a copy of the code under your GitHub user account.

2. Clone your fork of the Persephone repository from your GitHub account to your local disk:

   ```bash
   $ git clone git@github.com:Your-Username/persephone.git
   $ cd persephone
   ```

3. Create a ``feature`` or ``bugfix`` branch to hold your development changes:

   ```bash
   $ git checkout -b my-feature
   ```

   Always use a branch. It's good practice to never work on the ``master`` branch!
   You don't need to worry about the history, changes can be rebased or squashed as needed.

4. Develop the feature on your feature branch. Add changed files using ``git add`` and then ``git commit`` files:

   ```bash
   $ git add modified_files
   $ git commit
   ```

   to record your changes in Git, then push the changes to your GitHub account with:

   ```bash
   $ git push -u origin my-feature
   ```

5. Go to the GitHub web page of your fork of the Persephone repo.  Click the
'Pull request' button to send your changes to the project's maintainers for
review. This will send an
email to the maintainers. 

(If you are unfamiliar with Git, please look up the
[Git documentation](https://git-scm.com/documentation) on the web, or ask a friend or another contributor for help.)

Pull Request Checklist
----------------------

Please follow the following rules before you submit a pull request:

-  If your pull request addresses an issue, please use the pull request title
   to describe the issue and mention the issue number in the pull request description.
   This will make sure a link back to the original issue is created.

-  All public functions and methods should ideally have informative docstrings.

-  Please prefix the title of your pull request with `[MRG]` (Ready for
   Merge), if the contribution is complete and ready for a detailed review.
   Incomplete contributions should be prefixed `[WIP]` (to indicate a work
   in progress) and changed to `[MRG]` when it matures. WIPs may be useful
   to: indicate you are working on something to avoid duplicated work,
   request broad review of functionality or API, or seek collaborators.
   WIPs often benefit from the inclusion of a
   [task list](https://github.com/blog/1375-task-lists-in-gfm-issues-pulls-comments)
   in the pull request description.

-  Documentation and high-coverage tests are very beneficial for enhancements to be accepted, though this is currently hypocritical on my part.


You can also check for common programming errors with the following tools:

-  Code lint, check with:

  ```bash
  $ pip install pylint
  $ pylint path/to/code
  ```

Pylint errors are indicative of critical issues. This is also currently being worked on.

Filing bugs
-----------
We use Github issues to track all bugs and feature requests; feel free to
open an issue if you have found a bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the
following rules before submitting:

-  Verify that your issue is not being currently addressed by other
   [issues](https://github.com/oadams/persephone/issues?q=)
   or [pull requests](https://github.com/oadams/persephone/pulls?q=).

-  Please ensure all code snippets and error messages are formatted in
   appropriate code blocks.
   See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks).

-  If the bug is system specific please include your operating system type and version number,
   as well as your Python versions. This information
   can be found by runnning the following code snippet:

  ```python
  import platform; print(platform.platform())
  import sys; print("Python", sys.version)
  ```
