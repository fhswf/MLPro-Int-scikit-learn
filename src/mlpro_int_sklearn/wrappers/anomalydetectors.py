## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_sklearn.wrappers
## -- Module  : anomalydetectors.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-23  1.0.0     SP       Creation
## -- 2023-06-23  1.0.0     SP       First version release
## -- 2024-02-16  1.1.0     DA       Refactoring
## -- 2024-02-23  1.1.1     SP       Bug Fix
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.1 (2024-02-23)

This module provides wrapper functionalities to incorporate anomaly detector algorithms of the 
Scikit-learn ecosystem. This module includes three algorithms from Scikit-learn that are embedded to
MLPro, such as:

1) Local Outlier Factor (LOF)
2) One Class SVM
3) Isolation Forest (IF)

Learn more:
https://scikit-learn.org

"""

from mlpro.oa.streams.tasks.anomalydetectors import *
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.svm import OneClassSVM as OCSVM
from sklearn.ensemble import IsolationForest as IF
from mlpro_int_sklearn.wrappers.basics import WrapperSklearn




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrAnomalyDetectorSklearn2MLPro(AnomalyDetectorExtended, WrapperSklearn):
    """
    This is the base class for anomaly detection by anomaly detection algorithms which are wrapped
    from Scikit-Learn ecosystem.
    
    """
    C_TYPE = 'ScikitLearn Anomaly Detector'
    C_NAME = 'ScikitLearn Anomlay Detector'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_data_buffer = 20,
                 p_delay = 3,
                 p_name:str = None,
                 p_range_max = StreamTask.C_RANGE_THREAD,
                 p_ada : bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize : bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        super().__init__(p_name = p_name,
                         p_range_max = p_range_max,
                         p_ada = p_ada,
                         p_duplicate_data = p_duplicate_data,
                         p_visualize = p_visualize,
                         p_logging = p_logging,
                         **p_kwargs)
        
        self.data_buffer = p_data_buffer
        self.delay = p_delay
        self.ano_scores = []
        self.data_points = []
        self.inst_value = 0


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, p_inst_del: list):

        for inst in p_inst_new:
            if isinstance(inst, Instance):
                feature_data = inst.get_feature_data()
            else:
                feature_data = inst

        self.inst_value = feature_data.get_values()
        self.ano_scores = []

        if len(self.data_points) == 0:
            for i in range(len(self.inst_value)):
                self.data_points.append([])

        i=0
        for value in self.inst_value:
            self.data_points[i].append(value)
            i+=1

        if len(self.data_points[0]) > self.data_buffer:
            for i in range(len(self.inst_value)):
                self.data_points[i].pop(0)

        self.adapt(p_inst_new, p_inst_del)

        if -1 in self.ano_scores:
            self.raise_anomaly_event(p_inst_new)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrSklearnLOF2MLPro(WrAnomalyDetectorSklearn2MLPro):
    C_NAME          = 'LOF Anomaly Detector'
    C_TYPE          = 'Anomaly Detector'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_neighbours = 10,
                 p_delay = 3,
                 p_data_buffer = 20,
                 p_name:str = None,
                 p_range_max = StreamTask.C_RANGE_THREAD,
                 p_ada : bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize : bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        super().__init__(p_data_buffer = p_data_buffer,
                         p_delay = p_delay,
                         p_name = p_name,
                         p_range_max = p_range_max,
                         p_ada = p_ada,
                         p_duplicate_data = p_duplicate_data,
                         p_visualize = p_visualize,
                         p_logging = p_logging,
                         **p_kwargs)
        
        self.num_neighbours = p_neighbours
        self.lof = LOF(self.num_neighbours)


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_inst_new):
        if len(self.data_points[0]) >= self.delay:
            for i in range(len(self.inst_value)):
                scores = self.lof.fit_predict(np.array(self.data_points[i]).reshape(-1, 1))
                self.ano_scores.append(scores[-1])





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
                 p_name:str = None,
                 p_range_max = StreamTask.C_RANGE_THREAD,
                 p_ada : bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize : bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        super().__init__(p_data_buffer = p_data_buffer,
                         p_delay = p_delay,
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
    def _adapt(self, p_inst_new):
        if len(self.data_points[0]) >= self.delay:
            for i in range(len(self.inst_value)):
                self.svm.fit(np.array(self.data_points[i]).reshape(-1, 1))
                scores = self.svm.predict(np.array(self.data_points[i]).reshape(-1, 1))
                self.ano_scores.append(scores[-1])
     




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrSklearnIsolationForest2MLPro(WrAnomalyDetectorSklearn2MLPro):
    C_NAME          = 'Isolation Forest Anomaly Detector'
    C_TYPE          = 'Anomaly Detector'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_data_buffer = 20,
                 p_delay = 3,
                 p_estimators : int = 100,
                 p_max_samples : int = "auto",
                 p_contamination : float = "auto",
                 p_max_features : int = 1,
                 p_bootstrap : bool = True,
                 p_no_jobs : int = None,
                 p_random_state : int = None,
                 p_verbose : int = 0,
                 p_warm_start : bool = False,
                 p_name:str = None,
                 p_range_max = StreamTask.C_RANGE_THREAD,
                 p_ada : bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize : bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        super().__init__(p_data_buffer = p_data_buffer,
                         p_delay = p_delay,
                         p_name = p_name,
                         p_range_max = p_range_max,
                         p_ada = p_ada,
                         p_duplicate_data = p_duplicate_data,
                         p_visualize = p_visualize,
                         p_logging = p_logging,
                         **p_kwargs)
        
        self.num_estimators = p_estimators
        self.contamination = p_contamination
        self.max_samples = p_max_samples
        self.max_features = p_max_features
        self.bootstrap = p_bootstrap
        self.no_jobs = p_no_jobs
        self.random_state = p_random_state
        self.verbose = p_verbose
        self.warm_start = p_warm_start

        self.iso_f = IF(n_estimators=self.num_estimators,
                        contamination=self.contamination,
                        max_samples=self.max_samples,
                        max_features=self.max_features,
                        bootstrap=self.bootstrap,
                        n_jobs=self.no_jobs,
                        random_state=self.random_state,
                        verbose=self.verbose,
                        warm_start=self.warm_start)
  

## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_inst_new):
        if len(self.data_points[0]) >= self.delay:
            for i in range(len(self.inst_value)):
                self.iso_f.fit(np.array(self.data_points[i]).reshape(-1, 1))
                scores = self.iso_f.predict(np.array(self.data_points[i]).reshape(-1, 1))
                self.ano_scores.append(scores[-1])

