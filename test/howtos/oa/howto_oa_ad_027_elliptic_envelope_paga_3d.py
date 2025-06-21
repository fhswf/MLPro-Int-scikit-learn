## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_scikit_learn
## -- Module  : howto_oa_ad_027_elliptic_envelope_paga_3d.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-06-21  0.1.0     DS       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2025-06-15)

This module demonstrates the use of anomaly detector based on elliptic envelope algorithm with MLPro.
To this regard, a stream of a stream provider is combined with a stream workflow to a stream scenario.
The workflow consists of a standard task 'Aanomaly Detector'.

You will learn:

1) How to set up a stream workflow based on stream tasks.

2) How to set up a stream scenario based on a stream and a processing stream workflow.

3) How to add a task anomalydetector.

4) How to reuse an anomaly detector algorithm from scikitlearn (https://scikit-learn.org/), specifically
Isolation Forest

"""

from sklearn.covariance import EllipticEnvelope as EE

from mlpro.bf.various import Log
from mlpro.bf.ops import Mode
from mlpro.bf.plot import PlotSettings
from mlpro.bf.streams.streams import StreamMLProPOutliers
from mlpro.oa.streams import OAStreamScenario, OAStreamWorkflow

from mlpro_int_sklearn.wrappers.anomalydetectors import WrAnomalyDetectorSklearn2MLPro




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ADScenarioEE (OAStreamScenario):

    C_NAME = 'Scikit-learn Isolation Forest'

## -------------------------------------------------------------------------------------------------
    def _setup( self, 
                p_mode, 
                p_ada: bool, 
                p_visualize: bool, 
                p_logging,
                p_contamination: float = 0.01,
                p_support_fraction: float = 0.5,
                p_assume_centered: bool = False,
                p_anomaly_buffer_size: int = 100,
                p_instance_buffer_size: int = 50,
                p_detection_steprate: int = 50 ):

        # 1 Get the native stream from MLPro stream provider
        mystream = StreamMLProPOutliers( p_functions = ['sin' , 'cos', 'const'],
                                         p_outlier_rate=0.02,
                                         p_visualize=p_visualize, 
                                         p_logging=p_logging )

        # 2 Creation of a workflow
        workflow = OAStreamWorkflow( p_name='wf1',
                                     p_range_max=OAStreamWorkflow.C_RANGE_NONE,
                                     p_ada=p_ada,
                                     p_visualize=p_visualize, 
                                     p_logging=p_logging )
        
        # 3 Instantiation of Scikit-learn 'Elliptic Envelope' anomaly detector
        scikit_learn_ee = EE( contamination = p_contamination,
                              support_fraction = p_support_fraction,
                              assume_centered = p_assume_centered )
        
        # 4 Wrapping of the Scikit-learn algorithm and integration into the stream workflow
        anomalydetector = WrAnomalyDetectorSklearn2MLPro( p_algo_scikit_learn = scikit_learn_ee,
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
    contamination           = 0.01
    support_fraction        = 0.5
    assume_centered         = False
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

    contamination           = float(input(f'Algo EE: Contamination (press ENTER for {contamination}): ') or contamination)
    support_fraction        = float(input(f'Algo EE: Support fraction (press ENTER for {support_fraction}): ') or support_fraction)
    assume_centered         = bool(input(f'Algo EE: Assume centered (True/False, press ENTER for {assume_centered}): ') or assume_centered)
    anomaly_buffer_size     = int(input(f'MLPro Wrapper: Anomaly buffer size (press ENTER for {anomaly_buffer_size}): ') or anomaly_buffer_size)
    instance_buffer_size    = int(input(f'MLPro Wrapper: Instance buffer size (press ENTER for {instance_buffer_size}): ') or instance_buffer_size)
    detection_steprate      = int(input(f'MLPro Wrapper: Detection steprate (press ENTER for {detection_steprate}): ') or detection_steprate)

else:
    # 1.2 Parameters for internal unit test
    cycle_limit             = 20
    logging                 = Log.C_LOG_NOTHING
    visualize               = False
    step_rate               = 1
    contamination           = 0.01
    support_fraction        = 0.5
    assume_centered         = False
    anomaly_buffer_size     = 100
    instance_buffer_size    = 10
    detection_steprate      = 10


# 2 Instantiate the stream scenario
myscenario = ADScenarioEE( p_mode = Mode.C_MODE_REAL,
                           p_cycle_limit = cycle_limit,
                           p_visualize = visualize,
                           p_logging = logging,
                           p_contamination = contamination,
                           p_support_fraction = support_fraction,
                           p_assume_centered = assume_centered,
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