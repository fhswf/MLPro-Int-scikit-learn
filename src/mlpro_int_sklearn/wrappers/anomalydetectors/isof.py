## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_sklearn.wrappers.anomalydetectors
## -- Module  : isof.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-23  1.0.0     SK       Creation
## -- 2023-06-23  1.0.0     SK       First version release
## -- 2024-02-16  1.1.0     DA       Refactoring
## -- 2024-02-23  1.1.1     SK       Bug Fix
## -- 2024-04-16  1.1.2     DA       Bugfixes in 
## --                                - WrSklearnIsolationForest2MLPro._adapt()
## -- 2024-05-07  1.2.0     SK       Separation of particular algorithms into separate modules
## -- 2024-05-24  1.3.0     DA       Refactoring
## -- 2024-11-27  1.4.0     DA       Alignment with MLPro 1.9.2
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.4.0 (2024-11-27)

This module provides wrapper functionalities to incorporate Isolation Forest algorithm of the 
Scikit-learn ecosystem.

Learn more:
https://scikit-learn.org

"""

import numpy as np

from mlpro.bf.various import Log
from mlpro.bf.streams import Instance, StreamTask
from mlpro.oa.streams.tasks.anomalydetectors import *
from mlpro.oa.streams.tasks.anomalydetectors.anomalies import *

from sklearn.ensemble import IsolationForest as IF

from mlpro_int_sklearn.wrappers.anomalydetectors.basics import WrAnomalyDetectorSklearn2MLPro




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
    def _adapt(self, p_inst_new: Instance) -> bool:
        adapted = False
        
        if len(self.data_points[0]) >= self.delay:
            for i in range(len(self.inst_value)):
                self.iso_f.fit(np.array(self.data_points[i]).reshape(-1, 1))
                scores = self.iso_f.predict(np.array(self.data_points[i]).reshape(-1, 1))
                self.ano_scores.append(scores[-1])
                adapted = True

        return adapted

