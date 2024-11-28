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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.4.0 (2024-11-27)

This module provides wrapper root classes from Scikit-learn to MLPro, specifically for anomaly detectors. 

Learn more:
https://scikit-learn.org

"""

from mlpro.bf.various import Log
from mlpro.bf.streams import StreamTask, InstDict, InstTypeNew
from mlpro.oa.streams.tasks.anomalydetectors import *
from mlpro.oa.streams.tasks.anomalydetectors.anomalies import *

from mlpro_int_sklearn.wrappers.basics import WrapperSklearn





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrAnomalyDetectorSklearn2MLPro(AnomalyDetectorPAGA, WrapperSklearn):
    """
    This is the base class for anomaly detection FOR anomaly detection algorithms which are wrapped
    from Scikit-Learn ecosystem.
    
    """
    C_TYPE = 'ScikitLearn Anomaly Detector'
    C_NAME = 'ScikitLearn Anomlay Detector'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_data_buffer = 20,
                 p_delay = 3,
                 p_group_anomaly_det = True,
                 p_name:str = None,
                 p_range_max = StreamTask.C_RANGE_THREAD,
                 p_ada : bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize : bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        super().__init__(p_group_anomaly_det=p_group_anomaly_det,
                         p_name = p_name,
                         p_range_max = p_range_max,
                         p_ada = p_ada,
                         p_duplicate_data = p_duplicate_data,
                         p_visualize = p_visualize,
                         p_logging = p_logging,
                         **p_kwargs)
        
        self.data_buffer = p_data_buffer
        self.delay = p_delay
        self.data_points = []
        self.inst_value = 0
        self._visualize = p_visualize


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst: InstDict ):

        for inst_id, (inst_type, inst) in sorted(p_inst.items()):

            if inst_type != InstTypeNew: continue

            feature_data = inst.get_feature_data()

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

            self.adapt(p_inst = { inst_id : ( inst_type, inst ) } )

            if -1 in self.ano_scores:
                anomaly = PointAnomaly( p_id=self._get_next_anomaly_id, 
                                        p_instances=[inst], 
                                        p_ano_scores=self.ano_scores,
                                        p_visualize=self._visualize, 
                                        p_raising_object=self,
                                        p_det_time=str(inst.tstamp) )
                self._raise_anomaly_event(anomaly)

