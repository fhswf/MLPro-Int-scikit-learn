from setuptools import setup


setup(name='mlpro-int-scikit-learn',
version='0.1.0',
description='MLPro: Integration Scikit-learn',
author='MLPro Team',
author_mail='mlpro@listen.fh-swf.de',
license='Apache Software License (http://www.apache.org/licenses/LICENSE-2.0)',
packages=['mlpro_int_sklearn'],

# Package dependencies for full installation
extras_require={
    "full": [
        "dill",
        "numpy",
        "matplotlib",
        "multiprocess",
        "mlpro",
        "scikit-learn"
    ],
},

zip_safe=False)