## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_scikit_learn
## -- Module  : howto_oa_ad_015_lof_paga_1d.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-04-01  0.0.0     SK       Creation
## -- 2024-04-01  1.0.0     SK       First version release
## -- 2024-05-07  1.0.1     SK       Change in parameter p_outlier_rate
## -- 2024-11-27  1.0.2     DA       Correction for unit testing
## -- 2025-03-05  1.1.0     DA       Refactoring
## -- 2025-06-21  1.2.0     DS       Refactoring and generalization by user input parameters
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2025-06-21)

This module demonstrates the use of anomaly detector based on local outlier factor algorithm with MLPro.
To this regard, a stream of a stream provider is combined with a stream workflow to a stream scenario.
The workflow consists of a standard task 'Anomaly Detector'.

You will learn:

1) How to set up a stream workflow based on stream tasks.

2) How to set up a stream scenario based on a stream and a processing stream workflow.

3) How to add a task anomalydetector.

4) How to reuse an anomaly detector algorithm from scikitlearn (https://scikit-learn.org/), specifically
Isolation Forest

"""

from sklearn.neighbors import LocalOutlierFactor as LOF

from mlpro.bf.various import Log
from mlpro.bf.ops import Mode
from mlpro.bf.plot import PlotSettings
from mlpro.bf.streams.streams import StreamMLProPOutliers
from mlpro.oa.streams import OAStreamScenario, OAStreamWorkflow

from mlpro_int_sklearn.wrappers.anomalydetectors import WrAnomalyDetectorSklearn2MLPro




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ADScenarioLOF (OAStreamScenario):

    C_NAME = 'Scikit-learn Local Outlier Factor'

## -------------------------------------------------------------------------------------------------
    def _setup( self, 
                p_mode, 
                p_ada: bool, 
                p_visualize: bool, 
                p_logging,
                p_n_neighbors: int = 20,
                p_contamination: float = 0.01,
                p_novelty: bool = False,
                p_metric: str = 'minkowski',
                p_p : int = 2,
                p_anomaly_buffer_size: int = 100,
                p_instance_buffer_size: int = 50,
                p_detection_steprate: int = 50 ):

        # 1 Get the native stream from MLPro stream provider
        mystream = StreamMLProPOutliers( p_functions = ['sin' ], #, 'cos', 'const'],
                                         p_outlier_rate=0.02,
                                         p_visualize=p_visualize, 
                                         p_logging=p_logging )

        # 2 Creation of a workflow
        workflow = OAStreamWorkflow( p_name='wf1',
                                     p_range_max=OAStreamWorkflow.C_RANGE_NONE,
                                     p_ada=p_ada,
                                     p_visualize=p_visualize, 
                                     p_logging=p_logging )
        
        # 3 Instantiation of Scikit-learn 'Local Outlier Factor' anomaly detector
        scikit_learn_lof = LOF( n_neighbors= p_n_neighbors,
                                contamination= p_contamination,
                                novelty= p_novelty,
                                metric= p_metric,
                                p = p_p)
        
        # 4 Wrapping of the Scikit-learn algorithm and integration into the stream workflow
        anomalydetector = WrAnomalyDetectorSklearn2MLPro( p_algo_scikit_learn = scikit_learn_lof,
                                                          p_anomaly_buffer_size = p_anomaly_buffer_size,
                                                          p_instance_buffer_size = p_instance_buffer_size,
                                                          p_detection_steprate = p_detection_steprate,
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
    cycle_limit             = 500
    logging                 = Log.C_LOG_ALL
    step_rate               = 1
    n_neighbours            = 20
    contamination           = 0.01
    novelty                 = False
    metric                  = 'minkowski'
    p                       = 2
    anomaly_buffer_size     = 100
    instance_buffer_size    = 50
    detection_steprate      = 50

    cycle_limit             = int(input(f'\nCycle limit (press ENTER for {cycle_limit}): ') or cycle_limit)
    visualize               = input('Visualization Y/N (press ENTER for Y): ').upper() != 'N'
    if visualize:
        i = input(f'Visualization step rate (press ENTER for {step_rate}): ')
        if i != '': step_rate = int(i)

        i = input('Log level: "A"=All, "W"=Warnings only, "N"=Nothing (press ENTER for "W"): ').upper() 
        if i == 'A': logging = Log.C_LOG_ALL
        elif i == 'N': logging = Log.C_LOG_NOTHING

    n_neighbours            = int(input(f'Algo LOF: Number of neighbours (press ENTER for {n_neighbours}): ') or n_neighbours)
    contamination           = float(input(f'Algo LOF: Contamination (press ENTER for {contamination}): ') or contamination)
    novelty                 = bool(input(f'Algo LOF: Novelty detection Y/N (press ENTER for {novelty}): ')or novelty)
    metric                  = str(input(f'Algo LOF: Distance metric (press ENTER for {metric})') or metric)
    p                       = int(input(f'Algo LOF: Power parameter for Minkowski distance (press ENTER for {p})') or p)
    anomaly_buffer_size     = int(input(f'MLPro Wrapper: Anomaly buffer size (press ENTER for {anomaly_buffer_size}): ') or anomaly_buffer_size)
    instance_buffer_size    = int(input(f'MLPro Wrapper: Instance buffer size (press ENTER for {instance_buffer_size}): ') or instance_buffer_size)
    detection_steprate      = int(input(f'MLPro Wrapper: Detection steprate (press ENTER for {detection_steprate}): ') or detection_steprate)

else:
    # 1.2 Parameters for internal unit test
    cycle_limit             = 20
    logging                 = Log.C_LOG_NOTHING
    visualize               = False
    step_rate               = 1
    n_neighbours            = 20
    contamination           = 0.01
    novelty                 = False
    metric                  = 'minkowski'
    p                       = 2
    anomaly_buffer_size     = 100
    instance_buffer_size    = 10
    detection_steprate      = 10


# 2 Instantiate the stream scenario
myscenario = ADScenarioLOF( p_mode = Mode.C_MODE_REAL,
                            p_cycle_limit = cycle_limit,
                            p_visualize = visualize,
                            p_logging = logging,
                            p_n_neighbors = n_neighbours,
                            p_contamination = contamination,
                            p_novelty = novelty,
                            p_metric = metric,
                            p_p = p,
                            p_anomaly_buffer_size = anomaly_buffer_size,
                            p_instance_buffer_size = instance_buffer_size,
                            p_detection_steprate = detection_steprate )

if visualize:
    myscenario.init_plot( p_plot_settings=PlotSettings( p_view = PlotSettings.C_VIEW_ND,
                                                        p_view_autoselect = True,
                                                        p_step_rate = step_rate ) )


# 3 Reset and run own stream scenario
myscenario.reset()

if __name__ == '__main__':
    input('Press ENTER to start stream processing...')

myscenario.run()

if __name__ == '__main__':
    input('Press ENTER to exit...')