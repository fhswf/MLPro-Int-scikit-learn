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
## -- 2025-05-07  2.1.0     DA       Alignment with MLPro 2.0.0
## -- 2025-05-30  2.2.0     DA/DS    Alignment with MLPro 2.0.2
## -- 2025-06-12  2.3.0     DA/DS    - Alignment with MLPro 2.0.2
## --                                - Rework and optimization
## -- 2025-07-23  2.4.0     DA       Refactoring 
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.4.0 (2025-07-23)

This module provides wrapper root classes from Scikit-learn to MLPro, specifically for anomaly detectors. 

Learn more:
https://scikit-learn.org/stable/modules/outlier_detection.html

"""

import numpy as np
from sklearn.base import OutlierMixin

from mlpro.bf import Log, ParamError
from mlpro.bf.streams import StreamTask, Instance
from mlpro.oa.streams.tasks.changedetectors.anomalydetectors.instancebased import AnomalyDetectorIBPG
from mlpro.oa.streams.tasks.changedetectors.anomalydetectors.anomalies.instancebased import PointAnomaly

from mlpro_int_sklearn.wrappers import WrapperSklearn



# Export list for public API
__all__ = [ 'WrAnomalyDetectorSklearn2MLPro' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrAnomalyDetectorSklearn2MLPro (AnomalyDetectorIBPG, WrapperSklearn):
    """
    MLPro's wrapper for anomaly detectors of the scikit-learn project. The wrapper is limited to 
    detectors of type 'OutlierMixin' providing the method fit_predict() for a mixed data 
    training/prediction. 

    Parameters
    ----------
    p_algo_scikit_learn : OutlierMixin
        Outlier algorithm from the scikit-learn framework to be wrapped
    p_range_max : int
        Maximum range of asynchonicity. See class Range. Default is Range.C_RANGE_PROCESS.
    p_duplicate_data : bool
        If True, instances will be duplicated before processing. Default = False.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging : int = Log.C_LOG_ALL
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_anomaly_buffer_size : int = 100
        Size of the internal anomaly buffer self.anomalies. Default = 100.
    p_instance_buffer_size : int = 20
        Number of instances to be buffered internally as the basis for anomaly detection. Default = 20.
    p_detection_steprate : int = 1
        Detection steprate in the interval [1,p_instance_buffer_size].
    p_group_anomaly_det : bool
        Paramter to activate group anomaly detection. Default is True.

    Notes
    -----
    Supported algorithms
        - LOF - Local Outlier Factor
        - IF - Isolation Forest
        - Elliptic Envelope
        - Further ones inherited from 'OutlierMixin'

    Additional features
        - Optional group anomaly detection
        - 2D/3D anomaly visualization

    Supported types of anomalies
        - PointAnomaly
        - GroupAnomaly
    """

    C_TYPE = 'Anomaly Detector (scikit-learn)'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_algo_scikit_learn : OutlierMixin,
                  p_range_max = StreamTask.C_RANGE_THREAD, 
                  p_duplicate_data = False, 
                  p_visualize = False, 
                  p_logging=Log.C_LOG_ALL, 
                  p_anomaly_buffer_size = 100,
                  p_instance_buffer_size : int = 20,
                  p_detection_steprate : int = 1,
                  p_group_anomaly_det : bool = True, 
                  **p_kwargs ):
        
        WrapperSklearn.__init__( self, p_logging = p_logging )

        AnomalyDetectorIBPG.__init__( self, 
                                      p_group_anomaly_det = p_group_anomaly_det, 
                                      p_name = type(p_algo_scikit_learn).__name__, 
                                      p_range_max = p_range_max, 
                                      p_ada = True, 
                                      p_duplicate_data = p_duplicate_data, 
                                      p_visualize = p_visualize, 
                                      p_logging = p_logging, 
                                      p_anomaly_buffer_size = p_anomaly_buffer_size,
                                      p_thrs_inst = 1,
                                      **p_kwargs )
        
        if ( p_detection_steprate > p_instance_buffer_size ) or ( p_detection_steprate < 1 ):
            raise ParamError('Please set the parameter "p_detection_steprate" >= 1 and <= "p_instance_buffer_size"')
        
        self._algo_scikitlearn          = p_algo_scikit_learn
        self._inst_buffer_size          = p_instance_buffer_size
        self._detection_steprate        = p_detection_steprate
        self._inst_counter              = 0

        self._inst_data_buffer : np.ndarray  = None
        self._inst_data_buffer_full : bool   = False
        self._inst_ref_buffer : np.ndarray   = np.empty(self._inst_buffer_size, dtype = object)

        self._inst_buffer_pos : int          = 0

        self._block_mode = ( self._detection_steprate == self._inst_buffer_size )
        

## -------------------------------------------------------------------------------------------------
    def _detect(self, p_instance : Instance, **p_kwargs):

        # 1 Intro
        feature_data   = p_instance.get_feature_data()
        feature_values = feature_data.get_values()
        num_features   = feature_data.get_related_set().get_num_dim()


        # 2 Preparation of instance data buffer
        if self._inst_data_buffer is None:
            self._inst_data_buffer = np.empty((self._inst_buffer_size, num_features))


        # 3 Update the instance buffer
        if self._block_mode:
            # 3.1 Here, the buffer is used as an inplace ring buffer. Anomaly detection takes place, 
            #     whenever the buffer is overwritten completely.
            self._inst_data_buffer[self._inst_buffer_pos] = feature_values
            self._inst_ref_buffer[self._inst_buffer_pos]  = p_instance
            self._inst_buffer_pos = ( self._inst_buffer_pos + 1 ) % self._inst_buffer_size
            if self._inst_buffer_pos != 0: return

        else:
            # 3.2 Here, the buffer is used as a sliding window. Anomaly detection takes place,
            #     once the buffer is filled and the given step rate has been reached.
            if self._inst_data_buffer_full:
                # 3.2.1 Buffer full -> inplace shift left and new entry on the right
                self._inst_data_buffer[:-1] = self._inst_data_buffer[1:] 
                self._inst_data_buffer[self._inst_buffer_size -1] = feature_values
                self._inst_ref_buffer[:-1] = self._inst_ref_buffer[1:] 
                self._inst_ref_buffer[self._inst_buffer_size -1] = p_instance
                self._inst_counter = ( self._inst_counter + 1 ) % self._detection_steprate

                if self._inst_counter != 0: return

            else:
                # 3.2.2 Buffer to be filled
                self._inst_data_buffer[self._inst_buffer_pos] = feature_values
                self._inst_ref_buffer[self._inst_buffer_pos] = p_instance
                self._inst_buffer_pos = ( self._inst_buffer_pos + 1 ) % self._inst_buffer_size

                if self._inst_buffer_pos != 0: return

                self._inst_data_buffer_full = True


        # 4 Anomaly detection
        scores = self._algo_scikitlearn.fit_predict(self._inst_data_buffer)
        
        # 4.1 Check for anomalies in scores
        for i in np.where(scores == -1)[0]:
            related_instance = self._inst_ref_buffer[i]

            if not self._block_mode:
                # 4.1.1 In case of sliding window, multiple raise of anomalies for the same instance
                #       needs to be avoided.
                if related_instance is np.nan: continue
                self._inst_ref_buffer[i] = np.nan

            anomaly = PointAnomaly( p_status = True,
                                    p_tstamp = related_instance.tstamp,
                                    p_visualize = self.get_visualization(), 
                                    p_raising_object = self,
                                    p_instances = [related_instance] )
            
            self._raise_anomaly_event( p_anomaly = anomaly, p_instance = p_instance )