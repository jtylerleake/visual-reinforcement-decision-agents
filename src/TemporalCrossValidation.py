
from common.modules import np, pd, List, Dict, random
from src.utils.configuration_utils import load_experiment_config, validate_config
from src.utils.system_logging import ExperimentLogger, log_function_call, get_logger
from src.utils.metric_utils import *
from src.environment.extraction_pipeline import ExtractionPipeline
from src.environment.environment_pipeline import EnvironmentPipeline
from src.agents.reinforcement_agent import ReinforcementAgent


class TemporalCvManager:
    
    """
    Framework class for K-fold cross validation with temporal walk through 
    experiment. Model evaluation metrics are aggregated across folds/windows. 
    """
    
    def __init__(self, experiment_name: str, config: Dict):
        
        try:
            
            global logger
            logger = get_logger(experiment_name)
            
            self.experiment_name = experiment_name
            self.config = config
            self.num_folds = self.config['K folds']
            self.num_time_windows = self.config['Time splits']
            self.stratification_type = self.config['Stratification type']
            self.random_seed = self.config['Random seed']
            
            # random seed for reproducibility
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            
        except Exception as e: 
            logger.error(f"""Could not initialize ExperimentMaanger. Error 
                         with configuation file parameters: {e}""")
            
        # state tracking
        self.fold_assignments = {}
        self.windowed_data = {}
        self.cross_validation_state = {}
        
        logger.info("Experiment Manager initialized")
    
    @log_function_call
    def create_folds(self, stocks: List[str]) -> Dict[int, List[str]]:
        """Partitions stocks into folds based on stratification type"""
        
        try:
            if not stocks:
                logger.error("No stocks provided for fold assignment")
                return {}
            
            if self.num_folds <= 1:
                logger.error("Number of folds must be greater than 1")
                return {}
            
            if len(stocks) < self.num_folds:
                logger.error(f"""Number of stocks ({len(stocks)}) is less than
                             number of folds ({self.num_folds})""")
                return {}

            # Partition using Stratification Type
            if self.stratification_type == 'random':
                shuffled_stocks = stocks.copy()
                random.shuffle(shuffled_stocks)
                fold_assignments = {}
                stocks_per_fold = len(stocks) // self.num_folds
                remainder = len(stocks) % self.num_folds
                
                start_idx = 0
                for fold in range(self.num_folds):
                    # add one extra stock to first 'remainder' folds
                    fold_size = stocks_per_fold + (1 if fold < remainder else 0)
                    end_idx = start_idx + fold_size
                    fold_assignments[fold] = shuffled_stocks[start_idx:end_idx]
                    start_idx = end_idx
                
                logger.info(f"""Randomly assigned {len(stocks)} stocks to 
                            {self.num_folds} folds""")
                
            elif self.stratification_type == 'sector balanced':
                return
            
            else:
                logger.error(f"""Unknown stratification strategy: 
                             {self.stratification_strategy}""")
                return {}
            
            self.fold_assignments = fold_assignments
            return fold_assignments
            
        except Exception as e:
            logger.error(f"Error assigning stocks to folds: {e}")
            return {}
    
    @log_function_call
    def create_temporal_windows(self, stock_data: Dict[str, pd.DataFrame]) \
                                -> Dict[str, Dict[int, pd.DataFrame]]:
        """Splits stock data into temporal windows based on time windows spec"""

        try:

            logger.info(f"Creating temporal windows for {len(stock_data)} stocks")
            logger.info(f"Target windows per stock: {self.num_time_windows}")
            
            windowed_data = {}
            
            for ticker, data in stock_data.items():
                if data.empty:
                    logger.warning(f"Empty data for {ticker}, skipping")
                    continue
                
                # calculate window size
                total_points = len(data)
                window_size = total_points // self.num_time_windows
                
                if window_size < 1:
                    logger.warning(f"""Insufficient data for {ticker}: 
                                   {total_points} points (need at least 
                                   {self.num_time_windows})""")
                    continue
                
                # create windows
                windows = {}
                for window in range(self.num_time_windows):
                    start_idx = window * window_size
                    # last window gets any remaining data points
                    end_idx = start_idx + window_size if window \
                        < self.num_time_windows - 1 else total_points
                    
                    window_data = data.iloc[start_idx:end_idx].copy()
                    windows[window] = window_data
                    
                    logger.debug(f"""Window {window} for {ticker}: 
                                 {len(window_data)} data points""")
                
                windowed_data[ticker] = windows
                logger.info(f"""Created {len(windows)} windows for {ticker} 
                            (avg {window_size} points per window)""")
            
            self.windowed_data = windowed_data
            logger.info(f"Created temporal windows for {len(windowed_data)} stocks")
            return windowed_data
            
        except Exception as e:
            logger.error(f"Error creating temporal windows: {e}")
            return {}
    
    @log_function_call
    def exe_experiment(self) -> Dict:
        """Execute the cross-validation with temporal walk-through experiment"""
        
        # logger initialization
        logger = get_logger("Validation Experiment")
        logger.h1("EXECUTING REINFORCEMENT LEARNING VALIDATION EXPERIMENT")
        
        try:

            logger.h2("STARTING EXPERIMENT")
            
            # Step 1: Prepare data
            logger.info("Step 1: Preparing timeseries data pipeline")
            timeseries_pipeline = ExtractionPipeline(self.experiment_name)
            timeseries_data = timeseries_pipeline.exe_data_pipeline(self.config)
            if not timeseries_data:
                logger.error("Failed to prepare stock data")
                return {}
            
            # Step 2: Create fold assignments
            logger.info("Step 2: Creating fold assignments...")
            stocks = list(timeseries_data.keys())
            fold_assignments = self.create_folds(stocks)
            if not fold_assignments:
                logger.error("Failed to create fold assignments")
                return {}
            
            # Step 3: Create temporal windows
            logger.info("Step 3: Creating temporal windows...")
            windowed_data = self.create_temporal_windows(timeseries_data)
            if not windowed_data:
                logger.error("Failed to create temporal windows")
                return {}
            
            # Step 4: Execute cross-validation
            logger.info("Step 4: Executing cross-validation...")
            cv_results = self.exe_cross_validation(fold_assignments, windowed_data)
            
            # Step 5: Aggregate results
            logger.info("Step 5: Aggregating results...")
            aggregated_results = self.aggregate_cv_results(cv_results)
            
            logger.h1("CROSS-VALIDATION COMPLETED SUCCESSFULLY")
            
            return aggregated_results
            
        except Exception as e:
            logger.error(f"Error executing cross-validation: {e}")
            return {}
    
    @log_function_call
    def exe_cross_validation(self, fold_assignments: Dict[int, List[str]], 
                             windowed_data: Dict[str, Dict[int, pd.DataFrame]]) -> Dict:
        """Execute validation for each fold."""
        
        try:
            
            # K-Fold Cross Validation Loopr
            cv_results = {}
            for fold in range(self.num_folds):
                
                logger.info(f"\n--- Processing Fold {fold+1} ---")
                fold_results = {}
                
                # get the max index range for training windows; max training
                # index must be two less (0 indexed) than the total number of 
                # windows in order to maintain space for val/test windows 
                max_training_window = self.num_time_windows-3
                
                # Temporal-Window Walk-Through Loop
                for window in range(max_training_window+1):
                    logger.info(f"Training window {window+1} in fold {fold+1}")
                    
                    # create train/val/test splits
                    train_set, val_set, test_set = self.get_fold_window_splits(
                        fold, window, fold_assignments, windowed_data
                    )
                    
                    if not train_set or not val_set or not test_set:
                        logger.warning(f"""Insufficient data for fold {fold}, 
                                       window {window}""")
                        continue
                    
                    # train, validate, and test model
                    window_results = self.train_validate_test(
                        train_set, val_set, test_set, fold, window
                    )
                    
                    if window_results:
                        fold_results[window] = window_results
                        logger.info(f"Completed fold {fold+1}, window {window+1}")
                    else:
                        logger.warning(f"Failed fold {fold+1}, window {window+1}")
                
                cv_results[fold] = fold_results
                logger.info(f"""Completed fold {fold+1}: {len(fold_results)} 
                            successful windows""")
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Error executing fold validation: {e}")
            return {}
    
    @log_function_call
    def get_fold_window_splits(self, fold: int, window: int, 
                               fold_assignments: Dict[int, List[str]], 
                               windowed_data: Dict[str, Dict[int, pd.DataFrame]]
                               ) -> Dict[str, pd.DataFrame]:
        """Get train, validation, and test data for fold/window combinations"""
        
        try: 
            
            # terminate early
            if fold not in fold_assignments: return {},{},{}

            # isolate test stocks
            test_stocks = fold_assignments[fold]
            train_val_stocks = [
                stock for stocks in fold_assignments.values() for stock in stocks
                if stock not in test_stocks
            ]
            
            # data extraction
            train_set = {
                stock: windowed_data[stock][window] 
                for stock in train_val_stocks}
            
            validation_set = {
                stock: windowed_data[stock][window+1] 
                for stock in train_val_stocks}
            
            test_set = {
                stock: windowed_data[stock][window+2] 
                for stock in test_stocks}
            
            return train_set, validation_set, test_set
        
        except Exception as e:
            logger.error(f"Error getting fold window data: {e}")
            return {}
    
    @log_function_call
    def train_validate_test(self, train_set: Dict, val_set: Dict, test_set: Dict,
                            fold: int, window: int) -> Dict:
        """Train, validate, and test a model for a fold/window combination"""
        
        try:
            
            logger.info(f"Training model for fold {fold+1}, window {window+1}")
            
            # Create GAF environment for training data
            train_env = EnvironmentPipeline(
                experiment_name = self.experiment_name,
                tickers = train_set.keys(),
                gaf_timeseries_periods = self.config['GAF timeseries periods'],
                gaf_features = self.config['GAF features'],
                gaf_target = self.config['GAF target'],
                lookback_window = self.config['Lookback window']
            )
            
            train_env.timeseries_data = train_set
            
            if not train_env.exe_gaf_pipeline():
                logger.error("Failed to build GAF environment for training")
                return {}
            
            logger.info(f"""Successfully created training environment with 
                        {train_env.num_vec_environments} environments""")
            
            # >>> MODEL TRAINING
            logger.info("Training Agent...")
            agent = ReinforcementAgent(train_env, self.config)
            agent.train()
            
            # Create GAF environment for validation data
            val_env = EnvironmentPipeline(
                experiment_name = self.experiment_name,
                tickers = val_set.keys(),
                gaf_timeseries_periods = self.config['GAF timeseries periods'],
                gaf_features = self.config['GAF features'],
                gaf_target = self.config['GAF target'],
                lookback_window = self.config['Lookback window']
            )
            
            val_env.timeseries_data = val_set
            if not val_env.exe_gaf_pipeline():
                logger.error("Failed to build GAF environment for validation")
                return {}
            
            logger.info(f"""Successfully created validation environment with 
                        {val_env.num_vec_environments} environments""")
            
            # >>> MODEL VALIDATION
            logger.info("Validating model...")
            validation_results = self.evaluate_rl_model(agent, val_env)

            # Create GAF environment for test data
            test_env = EnvironmentPipeline(
                experiment_name = self.experiment_name,
                tickers = test_set.keys(),
                gaf_timeseries_periods = self.config['GAF timeseries periods'],
                gaf_features = self.config['GAF features'],
                gaf_target = self.config['GAF target'],
                lookback_window = self.config['Lookback window']
            )
            
            test_env.timeseries_data = test_set
            if not test_env.exe_gaf_pipeline():
                logger.error("Failed to build GAF environment for testing")
                return {}
            
            logger.info(f"""Successfully created testing environment with 
                        {test_env.num_vec_environments} environments""")
            
            # >>> MODEL TESTING
            logger.info("Testing model...")
            test_results = self.evaluate_rl_model(agent, val_env)
            
            # Aggregate results and return 
            results = {
                'fold': fold+1,
                'window': window+1,
                'training_stocks': list(train_set.keys()),
                'validation_stocks': list(val_set.keys()),
                'test_stocks': list(test_set.keys()),
                'validation_results': validation_results,
                'test_results': test_results,
                }
            
            logger.info(f"""Completed training/validation/test for 
                        fold {fold+1}, window {window+1}""")
                        
            return results
            
        except Exception as e:
            logger.error(f"Error during Train/Validate/Test: {e}")
            return {}
    
    @log_function_call
    def evaluate_rl_model(self, agent, environment) -> Dict:
        """Evaluate a trained model on a validation or test environment. Return
        individual and aggregated performance metrics by environment"""
        
        try:
            evaluation_results = {}
            total_reward = 0
            
            # Environment-by-Environment Evaluation
            for ticker, monitor in environment.environments.items():
                
                # reset the environment; return first observation
                env = monitor.env
                obs = env.reset()
                if isinstance(obs, tuple): obs = obs[0]
                
                # metrics to collect
                cumulative_episode_rewards = []
                episode_reward = 0
                actions = []
                targets = []
                
                # Single Environment Episode
                done = False
                while not done:
                    
                    # get an action from trained model
                    action, _ = agent.model.predict(obs, deterministic = True)
                    
                    # handle different action formats
                    if isinstance(action, (list, np.ndarray)):
                        if isinstance(action, np.ndarray) and action.ndim > 0:
                            action_val = action[0]
                        else:
                            action_val = action
                    else:
                        action_val = action
                    
                    actions.append(action_val)
                    
                    # step in environment
                    step_result = env.step(action)
                    
                    # handle different step result formats
                    if len(step_result) == 5:
                        obs, reward, terminated, truncated, _ = step_result
                        done = terminated or truncated
                    elif len(step_result) == 4:
                        obs, reward, done, _ = step_result
                    
                    if isinstance(obs, tuple):
                        obs = obs[0]
                    
                    if isinstance(reward, (list, np.ndarray)):
                        if isinstance(reward, np.ndarray) and reward.ndim > 0:
                            reward = reward[0]
                        else:
                            reward = reward
                    
                    episode_reward += reward
                    cumulative_episode_rewards.append(episode_reward)
                    
                    # get price for this step
                    target = env.prices[env._current_tick]
                    targets.append(target)
                    
                evaluation_results[ticker] = {
                    'stock': ticker,
                    'cumulative_episode_rewards' : cumulative_episode_rewards,
                    'episode_reward': episode_reward,
                    'num_actions': len(actions),
                    'actions': actions,
                    'targets': targets
                }
                
                total_reward += episode_reward

            # Calculate Summary Metrics
            avg_reward = total_reward / len(evaluation_results) if \
                evaluation_results else 0
            
            summary_results = {
                'num_stocks': len(evaluation_results),
                'total_reward': total_reward,
                'avg_reward': avg_reward,
                'individual_results': evaluation_results
            }
            
            logger.info(f"Evaluation complete. Avg reward: {avg_reward:.2f}")
            return summary_results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    
    @log_function_call
    def aggregate_cv_results(self, cv_results: Dict) -> Dict:
        """Aggregate results across all folds and windows"""
        
        try:
            logger.info("Aggregating cross-validation results...")
            
            aggregated = {
                'summary': {
                    'total_folds': len(cv_results),
                    'total_windows': sum(len(fold_results) for fold_results \
                                         in cv_results.values()),
                    'successful_runs': 0,
                    'failed_runs': 0
                },
                'fold_results': cv_results,
                'performance_metrics': {}
            }
            
            # Collect all validation and test results
            all_val_results = []
            all_test_results = []
            
            for fold_id, fold_results in cv_results.items():
                for window_id, window_data in fold_results.items():
                    if 'validation_results' in window_data:
                        all_val_results.append(window_data['validation_results'])
                    if 'test_results' in window_data:
                        all_test_results.append(window_data['test_results'])
                    aggregated['summary']['successful_runs'] += 1
            
            # Calculate aggregate metrics
            if all_val_results:
                val_rewards = [result['avg_reward'] for result in all_val_results]
                aggregated['performance_metrics']['validation'] = {
                    'mean_reward': np.mean(val_rewards),
                    'std_reward': np.std(val_rewards),
                    'min_reward': np.min(val_rewards),
                    'max_reward': np.max(val_rewards),
                    'num_runs': len(val_rewards)
                }
            
            if all_test_results:
                test_rewards = [result['avg_reward'] for result in all_test_results]
                aggregated['performance_metrics']['test'] = {
                    'mean_reward': np.mean(test_rewards),
                    'std_reward': np.std(test_rewards),
                    'min_reward': np.min(test_rewards),
                    'max_reward': np.max(test_rewards),
                    'num_runs': len(test_rewards)
                }
            
            logger.info(f"Aggregation complete: {aggregated['summary']['successful_runs']} successful runs")
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating CV results: {e}")
            return {}

