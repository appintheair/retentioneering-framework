---------------------------------------------------------------------

.. inclusion-marker-do-not-remove

Installation
~~~~~~~~~~~~

C++ compiler
============

Fistly you need to download c++ compiler

Mac
---

Install `homebrew <https://brew.sh/>`__

.. code:: bash

    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

Then you should install gcc

.. code:: bash

    brew install gcc

You also need python if you haven't it

.. code:: bash

    brew install python3

And git

.. code:: bash

    brew install git

Windows
-------

Install `Microsoft Visual Studio Build
tools <https://visualstudio.microsoft.com/ru/downloads/>`__. You can
find it at
``Инструменты для Visual Studio 2017 > Build Tools для Visual Studio 2017``.

If you need python then install it from
`here <https://www.python.org/downloads/release/python-368/>`__. Please,
be sure that you select ``Add python to path`` in the bottom of
installer.

And git from `here <https://git-scm.com/downloads>`__.

Python package
==============

-  To install Python package from github, you need to clone that
   repository.

   .. code:: bash

       git clone git@github.com:appintheair/aita-ml-retentioneering-python.git

   or

   .. code:: bash

       git clone https://github.com/appintheair/aita-ml-retentioneering-python.git

-  Install dependencies from requirements.txt file from that directory

   .. code:: bash

       sudo pip install -r requirements.txt

   or if previous command don't work

   .. code:: bash

       sudo pip3 install -r requirements.txt

-  Then just run the setup.py file from that directory

   .. code:: bash

       sudo python setup.py install

   or if previous command don't work

   .. code:: bash

       sudo python3 setup.py install


