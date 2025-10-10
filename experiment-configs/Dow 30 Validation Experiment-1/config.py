
EXPERIMENT_CONFIG = {
    
    # Name and Description
    'Experiment name': 'Development Experiment',
    'Experiment description': 'Development Experiment Config',   
    
    # Assets
    'Tickers': [
        'KO',
        'SHW',
        'AMGN',
        'AMZN',
        'AXP',
        'BA',
        'CAT',
        'CRM',
        'CSCO',
        'AAPL',
        'HON',
        'MSFT',
        'NVDA',
        'TRV',
        'UNH',
        'VZ',
        'WMT',
        'V',
        'MRK',
        'NKE',
        'PG',
        'CVX',
        'DIS',
        'GS',
        'HD',
        'IBM',
        'JNJ',
        'JPM',
        'MCD',
        'MMM'
    ],
    
    # Date and Frequency Attributes
    'Start date': '2015-01-01',
    'End date': '2020-12-31',
    'Update frequency': '1d',
    
    # Technical Indicators
    'SMA periods': 30,
    'RSI periods': 30,
    
    # GAF Parameters
    'GAF features': ['Close', 'High', 'Low', 'Open', 'SMA', 'OBV'],
    'GAF target': 'Close',
    'GAF image size': 14,
    'GAF timeseries periods': 14,
    
    # Training Parameters
    'Training epochs': 1000,
    'Learning rate': 0.001,
    
    # Reinforcement Learning Parameters
    'RL model': 'PPO', 
    'RL policy': 'MlpPolicy',
    
    # Environment Parameters
    'Lookback window': 10,
    
    # Experiment Design Parameters
    'K folds': 5,
    'Time splits': 5,
    'Stratification type': 'random',
    'Random seed': 42,
    
    # Performance Metrics
    'Calculate sharpe ratio': True,
    'Calculate max drawdown': True,
    'Calculate win rate': True,
    'Calculate profit factor': True,

    # Model Storage
    'Model save path': './models',
    'Checkpoint frequency': 100,
    
}
