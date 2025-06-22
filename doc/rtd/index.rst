.. MLPro Documentations documentation master file, created by
   sphinx-quickstart on Wed Sep 15 12:06:53 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MLPro-Int-scikit-learn - Integration of scikit-learn into MLPro
===============================================================

Welcome to MLPro-Int-scikit-learn, an extension to MLPro to integrate the scikit-learn package.
MLPro is a middleware framework for standardized machine learning in Python. It is 
developed by the South Westphalia University of Applied Sciences, Germany, and provides 
standards, templates, and processes for hybrid machine learning applications. Scikit-learn, in turn, 
provides numerous state-of-the-art algorithms for a vast amount of machine learning topics.

MLPro-Int-scikit-learn provides wrapper classes that enable the use of scikit-learn algorithms and 
data streams in your MLPro applications. The use of these wrappers is illustrated in various example programs.

.. image:: MLPro-Int-scikit-learn.gif
   :width: 300

**Preparation**

Before running the examples, please install the latest versions of MLPro, scikit-learn, and MLPro-Int-scikit-learn as follows:

.. code-block:: bash

   pip install mlpro-int-scikit-learn[full] --upgrade


**See also**
   - `MLPro - The integrative middleware framework for standardized machine learning in Python <https://mlpro.readthedocs.io>`_ 
   - `MLPro-OA - Sub-framework for online machine learning <https://mlpro.readthedocs.io/en/latest/content/03_machine_learning/mlpro_oa/main.html>`_
   - `Scikit-learn - Machine Learning in Python <https://scikit-learn.org>`_      
   - `MLPro-Int-scikit-learn on GitHub <https://github.com/fhswf/MLPro-Int-Scikit-learn>`_
   - `MLPro-Int-scikit-learn on PyPI <https://pypi.org/project/mlpro-int-scikit-learn>`_
   - `Further MLPro extensions <https://mlpro.readthedocs.io/en/latest/content/04_extensions/main.html>`_


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Home

   self


.. toctree::
   :maxdepth: 2
   :caption: Example Pool
   :glob:

   content/01_example_pool/*


.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :glob:

   content/02_api/*


.. toctree::
   :maxdepth: 2
   :caption: About
   :glob:

   content/03_about/*
