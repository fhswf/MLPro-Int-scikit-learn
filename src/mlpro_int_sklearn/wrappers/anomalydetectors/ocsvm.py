## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_sklearn.wrappers.anomalydetectors
## -- Module  : ocsvm.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-23  1.0.0     SK       Creation
## -- 2023-06-23  1.0.0     SK       First version release
## -- 2024-02-16  1.1.0     DA       Refactoring
## -- 2024-02-23  1.1.1     SK       Bug Fix
## -- 2024-04-16  1.1.2     DA       Bugfixes in 
## --                                - WrSklearnOneClassSVM2MLPro._adapt()
## -- 2024-05-07  1.2.0     SK       Separation of particular algorithms into separate modules
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2024-05-07)

This module provides wrapper functionalities to incorporate OneClass SVM algorithm of the 
Scikit-learn ecosystem.

Learn more:
https://scikit-learn.org

"""

from mlpro.bf.streams import Instance
from mlpro.oa.streams.tasks.anomalydetectors import *
from mlpro.oa.streams.tasks.anomalydetectors.anomalies import *
from sklearn.svm import OneClassSVM as OCSVM
from mlpro_int_sklearn.wrappers.anomalydetectors.basics import WrAnomalyDetectorSklearn2MLPro





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrSklearnOneClassSVM2MLPro(WrAnomalyDetectorSklearn2MLPro):
    C_NAME          = 'One Class SVM Anomaly Detector'
    C_TYPE          = 'Anomaly Detector'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_data_buffer = 20,
                 p_delay = 3,
                 p_kernel = 'rbf', #['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
                 p_nu = 0.01,
                 p_degree : int = 3,
                 p_gamma : float = 'scale', #['scale', 'auto'] or float
                 p_coef : float = 0,
                 p_tolerance : float = 0.001,
                 p_shrinking : bool = True,
                 p_cache_size : float = 200,
                 p_verbose : bool = False,
                 p_group_anomaly_det = True,
                 p_name:str = None,
                 p_range_max = StreamTask.C_RANGE_THREAD,
                 p_ada : bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize : bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        super().__init__(p_data_buffer = p_data_buffer,
                         p_delay = p_delay,
                         p_group_anomaly_det=p_group_anomaly_det,
                         p_name = p_name,
                         p_range_max = p_range_max,
                         p_ada = p_ada,
                         p_duplicate_data = p_duplicate_data,
                         p_visualize = p_visualize,
                         p_logging = p_logging,
                         **p_kwargs)
        
        self.kernel = p_kernel
        self.nu = p_nu
        self.degree = p_degree
        self.gamma = p_gamma
        self.coef = p_coef
        self.tol = p_tolerance
        self.shrinking = p_shrinking
        self.cache_size = p_cache_size
        self.verbose = p_verbose

        self.svm = OCSVM(kernel=self.kernel,
                        nu=self.nu,
                        degree=self.degree,
                        gamma=self.gamma,
                        coef0=self.coef,
                        tol=self.tol,
                        shrinking=self.shrinking,
                        cache_size=self.cache_size,
                        verbose=self.verbose)


# -------------------------------------------------------------------------------------------------
    def _adapt(self, p_inst_new: List[Instance]) -> bool:
        adapted = False
        
        if len(self.data_points[0]) >= self.delay:
            for i in range(len(self.inst_value)):
                self.svm.fit(np.array(self.data_points[i]).reshape(-1, 1))
                scores = self.svm.predict(np.array(self.data_points[i]).reshape(-1, 1))
                self.ano_scores.append(scores[-1])
                adapted = True

        return adapted

