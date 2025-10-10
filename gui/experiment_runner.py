
from common.modules import os, sys, time, json, path, Dict, List, Any, Optional, dt
from src.utils.configuration_utils import load_experiment_config, validate_config
from src.utils.configuration_utils import get_available_experiments
from src.utils.system_logging import get_logger
from src.TemporalCrossValidation import TemporalCvManager

config_dir = "experiment-configs"

class ExperimentRunner:
    
    """
    Utility class for ExperimentLauncher. Handles fetching and execution 
    of different experiments and configurations. 
    """
    
    def __init__(self):
        self.available_experiments = get_available_experiments()

    def run_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """Run a specific experiment with a specific configuration"""
        
        try:
            
            logger = get_logger(experiment_name)
            logger.info(f"Starting experiment: {experiment_name}")
            
            # validate experiment exists
            if experiment_name not in self.available_experiments:
                raise ValueError(f"Unknown experiment: {experiment_name}")
            
            # load and validate configuration
            config = load_experiment_config(experiment_name)
            if not validate_config(experiment_name, config):
                raise ValueError(f"Invalid configuration: {experiment_name}")
            
            # run the experiment; save results
            run = TemporalCvManager(experiment_name, config)
            results = run.exe_experiment()
            self.save_results(results)
            
            logger.info(f"Experiment completed: {experiment_name}")
            return results
            
        except Exception as e:
            logger.error(f"Error running experiment {experiment_name}: {e}")
            return {
                'experiment_name': experiment_name,
                'success': False,
                'error': str(e),
            }
    
    def save_results(self, experiment_name, results: Dict[str, Any]) -> None:
        """Save experiment results to file"""
        
        logger = get_logger(experiment_name)
        
        try:
            # create results directory
            results_dir = os.path.join(config_dir, experiment_name)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # generate filename
            timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{timestamp}-results.json"
            filepath = results_dir / filename
            
            # save
            save_results = results.copy()
            if 'config' in save_results:
                del save_results['config']
            
            with open(filepath, 'w') as f:
                json.dump(save_results, f, indent=2, default=str)
            
            logger.info(f"Results saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def get_experiment_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate a summary of experiment results.

        """
        try:
            summary = f"""
            Experiment Summary
            ==================
            Experiment: {results['experiment_name']}
            Configuration: {results['config_name']}
            Status: {'SUCCESS' if results['success'] else 'FAILED'}
            Duration: {results['duration_seconds']:.2f} seconds
            Start Time: {results['start_time']}
            End Time: {results['end_time']}
            Data Directory: {results['data_dir']}
            """
            
            if not results['success'] and 'error' in results:
                summary += f"Error: {results['error']}\n"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Error generating summary: {e}"
