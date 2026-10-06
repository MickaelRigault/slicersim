Installation
============

``slicersim`` requires Python 3.10 or later.

.. tab-set::

   .. tab-item:: pip

      .. code-block:: bash

         pip install slicersim

   .. tab-item:: From source

      .. code-block:: bash

         git clone https://github.com/MickaelRigault/slicersim.git
         cd slicersim
         pip install -e .

The core dependencies (``numpy``, ``scipy``, ``pandas``, ``astropy``,
``sncosmo``, ...) are installed automatically. Plotting methods (``show_*``)
additionally need ``matplotlib``.

Optional extras
---------------

.. code-block:: bash

   pip install "slicersim[tests]"   # run the test suite with pytest
   pip install "slicersim[docs]"    # build this documentation

Build the documentation locally
-------------------------------

.. code-block:: bash

   cd docs
   make html
   open _build/html/index.html

Check your installation
-----------------------

.. code-block:: python

   import slicersim

   exptime, target = slicersim.lazuli_sn_etc("salt", redshift=0.5, snr=20)
   print(f"{exptime:.0f} s")
