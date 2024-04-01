.. This file is heavily inspired by the equivalent in https://github.com/NLESC-JCER/QMCTorch

############################
Contribution guidelines
############################

We welcome any kind of contribution to our software, from a simple comment or question to a full-fledged `pull request <https://help.github.com/articles/about-pull-requests/>`_.

A contribution can be one of the following cases:

#. You have a question;
#. you think you may have found a bug (including unexpected behavior);
#. you want to change the code base (e.g., to fix a bug, add a new feature, or update documentation).

The sections below outline the steps in each case.

You have a question
*******************

#. Use the search functionality `here <https://github.com/Sm00thix/IKPLS/issues>`__ to see if someone already filed the same issue;
#. if your issue search did not yield any relevant results, make a new issue;
#. apply the "Question" label; apply other labels when relevant.

You think you may have found a bug
**********************************

#. Use the search functionality `here <https://github.com/Sm00thix/IKPLS/issues>`__ to see if someone already filed the same issue;
#. if your issue search does not yield any relevant results, make a new issue, providing enough information to the rest of the community to understand the cause and context of the problem. Depending on the issue, you may want to include:
    - the `SHA hashcode <https://help.github.com/articles/autolinked-references-and-urls/#commit-shas>`_ of the commit that is causing your problem;
    - some identifying information (name and version number) for dependencies you're using;
    - information about the operating system;
#. apply relevant labels to the newly created issue.

You want to make some kind of change to the code base
*****************************************************

#. (**important**) announce your plan to the rest of the community *before you start working*. This announcement should be in the form of a (new) issue;
#. (**important**) wait until some kind of consensus is reached about your idea being a good idea;
#. if needed, fork the repository to your own Github profile and create your own feature branch off of the latest main commit. While working on your feature branch, make sure to stay up to date with the main branch by pulling in changes, possibly from the 'upstream' repository (follow the instructions `here <https://help.github.com/articles/configuring-a-remote-for-a-fork/>`__ and `here <https://help.github.com/articles/syncing-a-fork/>`__);
#. make sure the existing tests still work by running ``python3 -m pytest tests --typeguard-packages=ikpls/``;
#. add your own tests (if necessary);
#. update or expand the documentation;
#. check that you can build the package locally and that it passes twine check. See `Build from source <#build_from_source>`_ for more information;
#. `push <http://rogerdudler.github.io/git-guide/>`_ your feature branch to (your fork of) the IKPLS repository on GitHub;
#. create the pull request, e.g. following the instructions `here <https://help.github.com/articles/creating-a-pull-request/>`__.

If you feel like you've made a valuable contribution, but you don't know how to write or run tests for it or generate the documentation, don't let this discourage you from making the pull request; we can help you! Just submit the pull request, but remember that you might be asked to append additional commits to your pull request.

.. _build_from_source:

Build from source
*****************

IKPLS uses `poetry <https://python-poetry.org/>`_ to manage its dependencies and packaging. To build the package from source, follow these steps:

#.  Clone the repository:

    .. code-block::
        :class: nohighlight

        git clone https://github.com/Sm00thix/IKPLS.git

#.  Change to the repository directory:

    .. code-block::
        :class: nohighlight
        
        cd IKPLS

#.  Install poetry and twine:

    .. code-block::
        :class: nohighlight

        pip3 install poetry
        pip3 install twine

#.  Install the dependencies:

    .. code-block::
        :class: nohighlight

        poetry install

#.  Build the package:

    .. code-block::
        :class: nohighlight

        poetry build

#.  Check the package with twine:

    .. code-block::
        :class: nohighlight

        twine check dist/*
