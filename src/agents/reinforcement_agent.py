
from common.modules import EvalCallback, DQN, A2C, PPO

models = {'PPO': PPO, 'A2C': A2C, 'DQN': DQN}

class ReinforcementAgent:
    
    """
    Overhead class for instantiating and training the reinforcement learning 
    model. Requires a pre-built training environment and an experiment config
    """
        
    def __init__(self, training_environment, config): 
        
        self.config = config
        self.env = training_environment.vec_environment
        
        # model initialization
        model = models[config['RL model']]
        policy = config['RL policy']
        lr = config['Learning rate']
        self.model = model(
            policy, self.env, verbose = 0, learning_rate = lr, device = "auto"
        )
    
    def train(self, do_callback = False) -> bool: 
        """Training method for the reinforcement agent"""
        callback = None
        epochs = self.config['Training epochs']
        if do_callback:
            callback = EvalCallback(
                self.env, best_model_save_path = ".\\models",
                log_path =".\\logs", eval_freq = 1000, 
                deterministic = True, render = False
            )    
        self.model.learn(total_timesteps = epochs, callback = callback)
   