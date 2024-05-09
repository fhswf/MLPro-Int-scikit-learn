## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_sklearn.wrappers.anomalydetectors
## -- Module  : lof.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-23  1.0.0     SK       Creation
## -- 2023-06-23  1.0.0     SK       First version release
## -- 2024-02-16  1.1.0     DA       Refactoring
## -- 2024-02-23  1.1.1     SK       Bug Fix
## -- 2024-04-16  1.1.2     DA       Bugfixes in 
## --                                - WrSklearnLOF2MLPro._adapt()
## -- 2024-05-07  1.2.0     SK       Separation of particular algorithms into separate modules
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2024-05-07)

This module provides wrapper functionalities to incorporate Local Outlier Factor algorithm of the 
Scikit-learn ecosystem.

Learn more:
https://scikit-learn.org

"""

from mlpro.bf.streams import Instance
from mlpro.oa.streams.tasks.anomalydetectors import *
from mlpro.oa.streams.tasks.anomalydetectors.anomalies import *
from sklearn.neighbors import LocalOutlierFactor as LOF
from mlpro_int_sklearn.wrappers.anomalydetectors.basics import WrAnomalyDetectorSklearn2MLPro





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
        
        self.num_neighbours = p_neighbours
        self.lof = LOF(self.num_neighbours)


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_inst_new: List[Instance]) -> bool:
        adapted = False

        if len(self.data_points[0]) >= self.delay:
            for i in range(len(self.inst_value)):
                scores = self.lof.fit_predict(np.array(self.data_points[i]).reshape(-1, 1))
                self.ano_scores.append(scores[-1])
                adapted = True

        return adapted


