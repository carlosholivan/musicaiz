Installation instructions
^^^^^^^^^^^^^^^^^^^^^^^^^

pypi
~~~~
The simplest way to install *musicaiz* is through the Python Package Index (PyPI).
This can be achieved by executing the following command::

    pip install musicaiz

or::

    sudo pip install musicaiz



Source
~~~~~~

If you've downloaded the archive manually from the `releases
<https://github.com/carlosholivan/musicaiz/releases/>`_ page, you can install using the
`setuptools` script::

    tar xzf musicaiz-VERSION.tar.gz
    cd musicaiz-VERSION/
    python setup.py install

If you intend to develop musicaiz or make changes to the source code, you can
install with `pip install -e` to link to your actively developed source tree::

    tar xzf musicaiz-VERSION.tar.gz
    cd musicaiz-VERSION/
    pip install -e .

Alternately, the latest development version can be installed via pip::

    pip install git+https://github.com/carlosholivan/musicaiz
