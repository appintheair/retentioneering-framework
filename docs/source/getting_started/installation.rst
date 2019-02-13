Installation
~~~~~~~~~~~~

C++ compiler
============

Firstly, you need to download c++ compiler.

Mac
---

Install `homebrew <https://brew.sh/>`__:

.. code:: bash

    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

Next, install the gcc:

.. code:: bash

    brew install gcc

You also need python if you haven't:

.. code:: bash

    brew install python3

Finally, install git:

.. code:: bash

    brew install git

Windows
-------

Install `Microsoft Visual Studio Build
tools <https://visualstudio.microsoft.com/ru/downloads/>`__. You can
find it in Tools for Visual Studio 2017 > `Build Tools for Visual Studio 2017
<https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15>`__.

If you need python then install it from
`here <https://www.python.org/downloads/release/python-368/>`__.
Please, be sure that you select ``Add python to path`` in the bottom of the installer.

Finally, install git from `here <https://git-scm.com/downloads>`__.

Python package
==============

-  To install Python package from github, you need to clone our repository:

   .. code:: bash

       git clone https://github.com/appintheair/aita-ml-retentioneering-python.git

-  Then, install dependencies from the requirements.txt file from that directory:

   .. code:: bash

       pip install -r requirements.txt --user

   or, if previous command don't work, use

   .. code:: bash

       pip3 install -r requirements.txt --user

-  Then, just run the setup.py file from that directory:

   .. code:: bash

       python setup.py install --user

   or, if previous command don't work, use

   .. code:: bash

       python3 setup.py install --user
