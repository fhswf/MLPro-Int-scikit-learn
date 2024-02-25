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
Scikit-learn ecosystem. This module includes three algorithms from Scikit-learn that are embedded to MLPro, such as:

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
from datetime import datetime





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class LocalOutlierFactor(AnomalyDetector):
    C_NAME          = 'LOF Anomaly Detector'
    C_TYPE          = 'Anomaly Detector'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_neighbours = 10,
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
        
        self.num_neighbours = p_neighbours
        self.lof = LOF(self.num_neighbours)
        self.delay = p_delay


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, p_inst_del: list):

        self.adapt(p_inst_new, p_inst_del)

        # Determine if data point is an anomaly based on its outlier score
        if -1 in self.ano_scores:
            self.ano_counter += 1
            self.def_anomalies()
            print(self.ano_type)

        
        
        """event_obj = AnomalyEvent(p_raising_object=self, p_det_time=det_time,
                                     p_instance=str(self.data_points[-1]))
            handler = self.event_handler
            self.register_event_handler(event_obj.C_NAME, handler)
            self._raise_event(event_obj.C_NAME, event_obj)"""


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_inst_new):
        for inst in p_inst_new:
            if isinstance(inst, Instance):
                feature_data = inst.get_feature_data()
            else:
                feature_data = inst

        self.inst_value = feature_data.get_values()
        self.inst_id = inst.get_id()
        print(self.inst_id)
        
        self.ano_scores = []
        if len(self.data_points) == 0:
            for i in range(len(self.inst_value)):
                self.data_points.append([])

        i=0
        for value in self.inst_value:
            self.data_points[i].append(value)
            i=i+1

        if len(self.data_points[0]) > 100:
            for i in range(len(self.inst_value)):
                self.data_points[i].pop(0)

        if len(self.data_points[0]) >= self.delay:
            for i in range(len(self.inst_value)):
                scores = self.lof.fit_predict(np.array(self.data_points[i]).reshape(-1, 1))
                self.ano_scores.append(scores[-1])


## -------------------------------------------------------------------------------------------------
    def event_handler(self, p_event_id, p_event_object:Event):
        self.log(Log.C_LOG_TYPE_I, 'Received event id', p_event_id)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OneClassSVM(AnomalyDetector):

    C_NAME          = 'One Class SVM Anomaly Detector'
    C_TYPE          = 'Anomaly Detector'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_kernel = 'rbf',
                 p_nu = 0.01,
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
        
        self.kernel = p_kernel
        self.nu = p_nu
        # Instance of the LOF algorithm
        self.svm = OCSVM(kernel=self.kernel, gamma='auto', nu=self.nu)
        self.count = 0


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, p_inst_del: list):

        det_time = datetime.now()
        det_time = det_time.strftime("%Y-%m-%d %H:%M:%S")

        #Adaptation
        self.adapt(p_inst_new, p_inst_del)

        #self.anomaly_scores = self.svm.decision_function(p_inst_new, p_inst_del)
                
        # Determine if the data point is an anomaly based on its outlier score
        if -1 in self.anomaly_scores:
            self.count = self.count + 1
            print("Anomaly detected", self.count)  
            
            
            """event_obj = AnomalyEvent(p_raising_object=self, p_det_time=det_time,
                                     p_instance=str(self.data_points[-1]))
            handler = self.event_handler
            self.register_event_handler(event_obj.C_NAME, handler)
            self._raise_event(event_obj.C_NAME, event_obj)"""


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_inst_new):
        
        for inst in p_inst_new:
            if isinstance(inst, Instance):
                feature_data = inst.get_feature_data()
            else:
                feature_data = inst

        values = feature_data.get_values()

        self.anomaly_scores = []
        if len(self.data_points) == 0:
            for i in range(len(values)):
                self.data_points.append([])


        i=0
        for value in values:
            self.data_points[i].append(value)
            i=i+1

        if len(self.data_points[0]) > 100:
            for i in range(len(values)):
                self.data_points[i].pop(0)

        if len(self.data_points[0]) >= 20:
            for i in range(len(values)):
                scores = self.svm.fit_predict(np.array(self.data_points[i]).reshape(-1, 1))
                self.anomaly_scores.append(scores[-1])



## -------------------------------------------------------------------------------------------------
    def event_handler(self, p_event_id, p_event_object:Event):
        self.log(Log.C_LOG_TYPE_I, 'Received event id', p_event_id)
        




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class IsolationForest(AnomalyDetector):

    C_NAME          = 'Isolation Forest Anomaly Detector'
    C_TYPE          = 'Anomaly Detector'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_estimators = 100,
                 p_contamination = 0.01,
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
        
        self.num_estimators = p_estimators
        self.contamination = p_contamination
        # Instance of the LOF algorithm
        self.iso_f = IF(n_estimators=self.num_estimators,
                                                contamination=self.contamination)
        self.count = 0
  

## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, p_inst_del: list):

        det_time = datetime.now()
        det_time = det_time.strftime("%Y-%m-%d %H:%M:%S")

        # Adaption
        self.adapt(p_inst_new, p_inst_del)

        # Perform anomaly detection
                
        # Determine if data point is an anomaly based on its outlier score
        ##if len(self.anomaly_scores) != 0 and self.anomaly_scores[-1] == -1:
        if -1 in self.anomaly_scores:
            self.count = self.count + 1
            print("Anomaly detected", self.count)  
            
            
            """event_obj = AnomalyEvent(p_raising_object=self, p_det_time=det_time,
                                     p_instance=str(self.data_points[-1]))
            handler = self.event_handler
            self.register_event_handler(event_obj.C_NAME, handler)
            self._raise_event(event_obj.C_NAME, event_obj)"""


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_inst_new):
        
        for inst in p_inst_new:
            if isinstance(inst, Instance):
                feature_data = inst.get_feature_data()
            else:
                feature_data = inst

        values = feature_data.get_values()

        self.anomaly_scores = []
        if len(self.data_points) == 0:
            for i in range(len(values)):
                self.data_points.append([])


        i=0
        for value in values:
            self.data_points[i].append(value)
            i=i+1

        if len(self.data_points[0]) > 100:
            for i in range(len(values)):
                self.data_points[i].pop(0)

        if len(self.data_points[0]) >= 20:
            for i in range(len(values)):
                self.iso_f.fit(np.array(self.data_points[i]).reshape(-1, 1))
                scores = self.iso_f.predict(np.array(self.data_points[i]).reshape(-1, 1))
                self.anomaly_scores.append(scores[-1])



## -------------------------------------------------------------------------------------------------
    def event_handler(self, p_event_id, p_event_object:Event):
        self.log(Log.C_LOG_TYPE_I, 'Received event id', p_event_id)

