## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_sklearn.wrappers
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-02-16  1.0.0     DA       First version
## -- 2024-04-18  1.1.0     DA       Alignment ot MLPro 1.4.0
## -- 2025-03-05  1.2.0     DA       Update of minimum release of scikit-learn to 1.6.1
## -- 2025-07-23  1.3.0     DA       Refactoring 
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.0 (2025-07-23)

This module contains the abstract root class for all scikit-learn wrapper classes.

Learn more:
https://scikit-learn.org

"""

from mlpro.bf.various import ScientificObject
from mlpro.wrappers import Wrapper



# Export list for public API
__all__ = [ 'WrapperSklearn' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrapperSklearn (Wrapper):
    """
    Root class for all scikit-learn wrapper classes.
    """

    C_TYPE              = 'Wrapper scikit-learn'
    C_WRAPPED_PACKAGE   = 'scikit-learn'
    C_MINIMUM_VERSION   = '1.6.1'

    C_SCIREF_TYPE       = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_AUTHOR     = 'scikit-learn community'
    C_SCIREF_URL        = 'https://scikit-learn.org'
