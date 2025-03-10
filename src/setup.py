from setuptools import setup


setup(name='mlpro-int-scikit-learn',
version='0.3.0',
description='MLPro: Integration scikit-learn',
author='MLPro Team',
author_mail='mlpro@listen.fh-swf.de',
license='Apache Software License (http://www.apache.org/licenses/LICENSE-2.0)',
packages=['mlpro_int_sklearn'],

# Package dependencies for full installation
extras_require={
    "full": [
        "mlpro[full]>=1.9.6",
        "scikit-learn>=1.6.1"
    ],
},

zip_safe=False)