## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_sklearn.wrappers
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-02-16  1.0.0     DA       First version
## -- 2024-04-16  1.0.1     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2024-04-16)

This module contains the abstract root class for all scikit-learn wrapper classes.

Learn more:
https://scikit-learn.org

"""

from mlpro.bf.various import ScientificObject
from mlpro.wrappers import Wrapper




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrapperSklearn (Wrapper):
    """
    Root class for all scikit-learn wrapper classes.
    """

    C_TYPE              = 'Wrapper scikit-learn'
    C_WRAPPED_PACKAGE   = 'scikit-learn'
    C_MINIMUM_VERSION   = '1.4.1'

    C_SCIREF_TYPE       = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_AUTHOR     = 'scikit-learn community'
    C_SCIREF_URL        = 'https://scikit-learn.org'
