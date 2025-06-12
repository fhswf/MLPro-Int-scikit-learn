## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_scikit_learn
## -- Module  : howto_oa_ad_007_oneclasssvm_paga_3d.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-04-01  0.0.0     SK       Creation
## -- 2023-04-01  1.0.0     SK       First version release
## -- 2024-04-06  1.0.1     DA       Set 3D mode
## -- 2024-05-07  1.0.2     SK       Change in parameter p_outlier_rate
## -- 2024-11-27  1.0.3     DA       Correction for unit testing
## -- 2025-03-05  1.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-03-05)

This module demonstrates the use of anomaly detector based on one class svm algorithm with MLPro.
To this regard, a stream of a stream provider is combined with a stream workflow to a stream scenario.
The workflow consists of a standard task 'Aanomaly Detector'.

You will learn:

1) How to set up a stream workflow based on stream tasks.

2) How to set up a stream scenario based on a stream and a processing stream workflow.

3) How to add a task anomalydetector.

4) How to reuse an anomaly detector algorithm from scikitlearn (https://scikit-learn.org/), specifically
One Class SVM

"""

from sklearn.svm import OneClassSVM as OCSVM

from mlpro.bf.streams.streams import *
from mlpro.bf.various import Log
from mlpro.oa.streams import *

from mlpro_int_sklearn.wrappers.anomalydetectors import WrAnomalyDetectorSklearn2MLPro



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ADScenarioOCSVM (OAStreamScenario):

    C_NAME = 'Scikit-learn One Class SVM'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging):

        # 1 Get the native stream from MLPro stream provider
        mystream = StreamMLProPOutliers( p_functions = ['sin', 'cos', 'const'],
                                         p_outlier_rate=0.1,
                                         p_visualize=p_visualize, 
                                         p_logging=p_logging )

        # 2 Creation of a workflow
        workflow = OAStreamWorkflow( p_name='wf1',
                                     p_range_max=OAStreamWorkflow.C_RANGE_NONE,
                                     p_ada=p_ada,
                                     p_visualize=p_visualize, 
                                     p_logging=p_logging )

        scikit_learn_ocsvm = OCSVM( kernel = 'poly',
                                    degree = 4,
                                    gamma = 'scale',
                                    coef0 = 0,
                                    nu = 0.01 )

        # 4 Wrapping of the Scikit-learn algorithm and integration into the stream workflow
        anomalydetector = WrAnomalyDetectorSklearn2MLPro( p_algo_scikit_learn = scikit_learn_ocsvm,
                                                          p_delay = 3,
                                                          p_instance_buffer_size = 25,
                                                          p_group_anomaly_det = True,
                                                          p_visualize = p_visualize,
                                                          p_logging = p_logging )

        workflow.add_task( p_task=anomalydetector )

        # 5 Return stream and workflow
        return mystream, workflow





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
# 1 Preparation of demo/unit test mode
if __name__ == "__main__":
    # 1.1 Parameters for demo mode
    cycle_limit = 360
    logging     = Log.C_LOG_ALL
    visualize   = True
    step_rate   = 2

else:
    # 1.2 Parameters for internal unit test
    cycle_limit = 2
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    step_rate   = 1


# 2 Instantiate the stream scenario
myscenario = ADScenarioOCSVM( p_mode = Mode.C_MODE_REAL,
                              p_cycle_limit = cycle_limit,
                              p_visualize = visualize,
                              p_logging = logging )

if visualize:
    myscenario.init_plot( p_plot_settings=PlotSettings( p_view = PlotSettings.C_VIEW_3D,
                                                        p_view_autoselect = False,
                                                        p_step_rate = step_rate ) )


# 3 Reset and run own stream scenario
myscenario.reset()

if __name__ == '__main__':
    input('Press ENTER to start stream processing...')

myscenario.run()

if __name__ == '__main__':
    input('Press ENTER to exit...')

