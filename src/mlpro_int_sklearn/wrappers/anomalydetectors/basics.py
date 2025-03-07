## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_sklearn.wrappers.anomalydetectors
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-23  1.0.0     SK       Creation
## -- 2023-06-23  1.0.0     SK       First version release
## -- 2024-02-16  1.1.0     DA       Refactoring
## -- 2024-02-23  1.1.1     SK       Bug Fix
## -- 2024-04-16  1.1.2     DA       Bugfixes in 
## --                                - WrAnomalyDetectorSklearn2MLPro._run()
## -- 2024-05-07  1.2.0     SK       Separation of particular algorithms into separate modules
## -- 2024-05-24  1.3.0     DA       Refactoring
## -- 2024-11-27  1.4.0     DA       Alignment with MLPro 1.9.2
## -- 2025-03-05  2.0.0     DA       Alignment with MLPro 1.9.5 and generalization
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.0.0 (2025-03-05)

This module provides wrapper root classes from Scikit-learn to MLPro, specifically for anomaly detectors. 

Learn more:
https://scikit-learn.org/stable/modules/outlier_detection.html

"""

import numpy as np
from sklearn.base import OutlierMixin

from mlpro.bf.various import Log
from mlpro.bf.streams import StreamTask, Instance, InstDict, InstTypeNew
from mlpro.oa.streams.tasks.anomalydetectors.instancebased import AnomalyDetectorIBPG
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.instancebased import PointAnomaly

from mlpro_int_sklearn.wrappers import WrapperSklearn



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrAnomalyDetectorSklearn2MLPro (AnomalyDetectorIBPG, WrapperSklearn):
    """
    MLPro's wrapper for anomaly detectors of the scikit-learn project. The wrapper raises anomalies
    of type PointAnomaly and GroupAnomaly.

    Parameters
    ----------
    p_algo_scikit_learn : OutlierMixin
        Outlier algorithm from the scikit-learn framework to be wrapped
    p_delay : int
        Number of instances before the detection starts. Default = 3.
    p_instance_buffer_size : int
        Number of instances to be buffered internally as the basis for anomaly detection. Default = 20.
    p_group_anomaly_det : bool
        Paramter to activate group anomaly detection. Default is True.
    p_range_max : int
        Maximum range of asynchonicity. See class Range. Default is Range.C_RANGE_PROCESS.
    p_ada : bool
        Boolean switch for adaptivitiy. Default = True.
    p_duplicate_data : bool
        If True, instances will be duplicated before processing. Default = False.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_anomaly_buffer_size : int = 100
        Size of the internal anomaly buffer self.anomalies. Default = 100.
    """

    C_TYPE = 'Anomaly Detector (scikit-learn)'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_algo_scikit_learn : OutlierMixin,
                  p_delay : int = 3,
                  p_instance_buffer_size : int = 20,
                  p_group_anomaly_det : bool = True, 
                  p_range_max = StreamTask.C_RANGE_THREAD, 
                  p_ada = True, 
                  p_duplicate_data = False, 
                  p_visualize = False, 
                  p_logging=Log.C_LOG_ALL, 
                  p_anomaly_buffer_size = 100 ):
        
        WrapperSklearn.__init__( self, p_logging = p_logging )

        AnomalyDetectorIBPG.__init__( self, 
                                      p_group_anomaly_det = p_group_anomaly_det, 
                                      p_name = type(p_algo_scikit_learn).__name__, 
                                      p_range_max = p_range_max, 
                                      p_ada = p_ada, 
                                      p_duplicate_data = p_duplicate_data, 
                                      p_visualize = p_visualize, 
                                      p_logging = p_logging, 
                                      p_anomaly_buffer_size = p_anomaly_buffer_size )
        
        self._algo_scikitlearn  = p_algo_scikit_learn
        self._inst_buffer_size  = p_instance_buffer_size
        self._delay             = p_delay
        self._inst_buffer       = []
        self._ano_scores        = None


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst: InstDict ):

        for inst_id, (inst_type, inst) in sorted(p_inst.items()):

            if inst_type != InstTypeNew: continue

            self.adapt( p_inst = { inst_id : ( inst_type, inst ) } )

            if -1 in self._ano_scores:
                anomaly = PointAnomaly( p_id = self._get_next_anomaly_id(), 
                                        p_instances = [inst], 
                                        p_ano_scores = self._ano_scores,
                                        p_visualize = self.get_visualization(), 
                                        p_raising_object = self,
                                        p_tstamp = inst.tstamp )
                
                self._raise_anomaly_event( p_anomaly = anomaly )


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_inst_new: Instance) -> bool:

        # 1 Intr0
        adapted          = False
        feature_data     = p_inst_new.get_feature_data()
        feature_values   = feature_data.get_values()
        self._ano_scores = []


        # 2 Preparation of instance data buffer
        if len(self._inst_buffer) == 0:
            for i in range(len(feature_values)):
                self._inst_buffer.append([])


        # 3 Update of instance data buffer
        if len(self._inst_buffer[0]) >= self._inst_buffer_size:
            for i in range(len(feature_values)):
                self._inst_buffer[i].pop(0)

        for i, value in enumerate(feature_values):
            self._inst_buffer[i].append(value)


        # 4 Anomaly detection
        if len(self._inst_buffer[0]) >= self._delay:
            for i in range(len(feature_values)):
                try:
                    # Algorithms like Isolation Forest and One Class SVM provide methods fit() and predict()
                    self._algo_scikitlearn.fit(np.array(self._inst_buffer[i]).reshape(-1, 1))
                    scores = self._algo_scikitlearn.predict(np.array(self._inst_buffer[i]).reshape(-1, 1))
                except:
                    # Algorithms implementing method OutlierMixin.fit_predict()
                    scores = self._algo_scikitlearn.fit_predict(np.array(self._inst_buffer[i]).reshape(-1, 1))

                self._ano_scores.append(scores[-1])

            adapted = True


        # 5 Outro
        return adapted
