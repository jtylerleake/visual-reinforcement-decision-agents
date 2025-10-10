
from common.modules import np, pd, plt, sns, List, Dict
from system_logging import get_logger, log_function_call


def plot_temporal_test_performance(cv_results: Dict, is_test: bool = True):
    """Plot K lines, one for the test/validation performance of 
    each k fold from temporal cross validation"""

    # gather performance per fold
    fold_data = []
    for fold_idx, fold_results in cv_results.items(): 
        # individual_window_results = []
        for window_idx, window_results in fold_results.items(): 
            metrics = fold_results[window_idx]
            metrics = metrics['test_results'] if is_test else metrics['validation_results']
            stock_results = metrics['individual_results']
            stock_results = []
            for stock, rewards in stock_results.items():
                reward_seq = np.array(rewards['cumulative_episode_rewards'])
                stock_results.append(reward_seq)
            fold_data.append(stock_results)
            # individual_window_results.append(stock_results)
        # fold_data.append(individual_window_results)
        
    # compute per fold averages
    fold_avgs = []
    for fold_list in fold_data:
        stacked_results = np.stack(fold_list)
        avgs = np.mean(stacked_results, axis=0)
        fold_avgs.append(avgs)
    
    # plot temporal fold averages
    for i in range(len(fold_avgs)):
        sns.lineplot(
            x = list(range(1, len(fold_avgs[i])+1)),
            y = fold_avgs[i],
            label = f"Fold {i}"
        )

    plt.title("Temporal Cross-Validation Test Performance (5 folds)")
    plt.xlabel("Test Window")
    plt.ylabel("Test Performance (return/reward)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    







class PerformanceMetrics:
    
    """
    Calculate and analyze trading performance metrics.
    """
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        """
        Initialize performance metrics calculator.
        
        Args:
            initial_capital: Starting capital
            commission: Commission per trade (as fraction)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        logger.info(f"Performance metrics initialized with capital: ${initial_capital:,.2f}")
    
    @log_function_call
    def calculate_returns(self, prices: List[float], actions: List[int]) -> Dict[str, float]:
        """
        Calculate returns from price series and actions.
        
        Args:
            prices: List of prices
            actions: List of actions (0=sell, 1=buy, 2=hold)
        
        Returns:
            Dictionary with return metrics
        """
        try:
            if len(prices) != len(actions):
                logger.error("Prices and actions must have same length")
                return {}
            
            # Initialize variables
            position = 0  # 0: no position, 1: long position
            entry_price = 0
            trades = []
            portfolio_values = [self.initial_capital]
            current_capital = self.initial_capital
            
            for i, (price, action) in enumerate(zip(prices, actions)):
                if action == 1 and position == 0:  # Buy
                    position = 1
                    entry_price = price
                    # Apply commission
                    current_capital -= current_capital * self.commission
                    trades.append({
                        'type': 'buy',
                        'price': price,
                        'timestamp': i,
                        'capital': current_capital
                    })
                
                elif action == 0 and position == 1:  # Sell
                    position = 0
                    # Calculate profit/loss
                    pnl = (price - entry_price) / entry_price
                    current_capital *= (1 + pnl)
                    # Apply commission
                    current_capital -= current_capital * self.commission
                    
                    trades.append({
                        'type': 'sell',
                        'price': price,
                        'timestamp': i,
                        'pnl': pnl,
                        'capital': current_capital
                    })
                
                # Update portfolio value
                if position == 1:
                    # Mark to market
                    pnl = (price - entry_price) / entry_price
                    portfolio_value = current_capital * (1 + pnl)
                else:
                    portfolio_value = current_capital
                
                portfolio_values.append(portfolio_value)
            
            # Calculate metrics
            total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
            buy_hold_return = (prices[-1] - prices[0]) / prices[0]
            
            # Calculate daily returns
            daily_returns = []
            for i in range(1, len(portfolio_values)):
                daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                daily_returns.append(daily_return)
            
            # Risk metrics
            volatility = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0
            sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
            
            # Trading metrics
            num_trades = len([t for t in trades if t['type'] == 'sell'])
            win_rate = self._calculate_win_rate(trades)
            profit_factor = self._calculate_profit_factor(trades)
            
            metrics = {
                'total_return': total_return,
                'buy_hold_return': buy_hold_return,
                'excess_return': total_return - buy_hold_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'num_trades': num_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'final_capital': portfolio_values[-1],
                'total_pnl': portfolio_values[-1] - self.initial_capital
            }
            
            logger.info(f"Performance metrics calculated: {num_trades} trades, "
                       f"Return: {total_return:.2%}, Sharpe: {sharpe_ratio:.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return {}
    
    def _calculate_sharpe_ratio(self, returns: List[float], 
                                risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if not returns:
            return 0.0
        
        excess_returns = np.array(returns) - risk_free_rate / 252
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not portfolio_values:
            return 0.0
        
        peak = portfolio_values[0]
        max_dd = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate from trades."""
        sell_trades = [t for t in trades if t['type'] == 'sell']
        if not sell_trades:
            return 0.0
        
        winning_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
        return len(winning_trades) / len(sell_trades)
    
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor from trades."""
        sell_trades = [t for t in trades if t['type'] == 'sell']
        if not sell_trades:
            return 0.0
        
        gross_profit = sum([t.get('pnl', 0) for t in sell_trades if t.get('pnl', 0) > 0])
        gross_loss = abs(sum([t.get('pnl', 0) for t in sell_trades if t.get('pnl', 0) < 0]))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    @log_function_call
    def calculate_rolling_metrics(self, prices: List[float], actions: List[int], 
                                 window: int = 20) -> Dict[str, List[float]]:
        """
        Calculate rolling performance metrics.
        
        Args:
            prices: List of prices
            actions: List of actions
            window: Rolling window size
        
        Returns:
            Dictionary with rolling metrics
        """
        try:
            if len(prices) < window:
                logger.warning(f"Data length {len(prices)} < window {window}")
                return {}
            
            rolling_returns = []
            rolling_sharpe = []
            rolling_volatility = []
            
            for i in range(window, len(prices)):
                window_prices = prices[i-window:i+1]
                window_actions = actions[i-window:i+1]
                
                # Calculate metrics for this window
                window_metrics = self.calculate_returns(window_prices, window_actions)
                
                rolling_returns.append(window_metrics.get('total_return', 0))
                rolling_sharpe.append(window_metrics.get('sharpe_ratio', 0))
                rolling_volatility.append(window_metrics.get('volatility', 0))
            
            return {
                'rolling_returns': rolling_returns,
                'rolling_sharpe': rolling_sharpe,
                'rolling_volatility': rolling_volatility
            }
            
        except Exception as e:
            logger.error(f"Error calculating rolling metrics: {e}")
            return {}
    
    @log_function_call
    def compare_strategies(self, strategy_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple trading strategies.
        
        Args:
            strategy_results: Dictionary mapping strategy names to metrics
        
        Returns:
            DataFrame with strategy comparison
        """
        try:
            comparison_data = []
            
            for strategy_name, metrics in strategy_results.items():
                comparison_data.append({
                    'Strategy': strategy_name,
                    'Total Return (%)': metrics.get('total_return', 0) * 100,
                    'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                    'Max Drawdown (%)': metrics.get('max_drawdown', 0) * 100,
                    'Volatility (%)': metrics.get('volatility', 0) * 100,
                    'Win Rate (%)': metrics.get('win_rate', 0) * 100,
                    'Num Trades': metrics.get('num_trades', 0),
                    'Profit Factor': metrics.get('profit_factor', 0)
                })
            
            df = pd.DataFrame(comparison_data)
            df = df.sort_values('Total Return (%)', ascending=False)
            
            logger.info(f"Strategy comparison completed for {len(strategy_results)} strategies")
            return df
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {e}")
            return pd.DataFrame()
    
    @log_function_call
    def generate_performance_report(self, metrics: Dict, 
                                  strategy_name: str = "Trading Strategy") -> str:
        """
        Generate a formatted performance report.
        
        Args:
            metrics: Performance metrics dictionary
            strategy_name: Name of the strategy
        
        Returns:
            Formatted report string
        """
        try:
            report = f"""
{'='*60}
{strategy_name.upper()} - PERFORMANCE REPORT
{'='*60}

RETURNS:
  Total Return: {metrics.get('total_return', 0):.2%}
  Buy & Hold Return: {metrics.get('buy_hold_return', 0):.2%}
  Excess Return: {metrics.get('excess_return', 0):.2%}
  Final Capital: ${metrics.get('final_capital', 0):,.2f}

RISK METRICS:
  Volatility: {metrics.get('volatility', 0):.2%}
  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
  Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}

TRADING METRICS:
  Number of Trades: {metrics.get('num_trades', 0)}
  Win Rate: {metrics.get('win_rate', 0):.2%}
  Profit Factor: {metrics.get('profit_factor', 0):.2f}

{'='*60}
"""
            logger.info(f"Performance report generated for {strategy_name}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return f"Error generating report: {e}"


# Utility functions
def calculate_annualized_return(total_return: float, days: int) -> float:
    """Calculate annualized return."""
    if days <= 0:
        return 0.0
    return (1 + total_return) ** (365 / days) - 1


def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino ratio (downside deviation)."""
    if not returns:
        return 0.0
    
    excess_returns = np.array(returns) - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)


def calculate_calmar_ratio(total_return: float, max_drawdown: float) -> float:
    """Calculate Calmar ratio (return / max drawdown)."""
    if max_drawdown == 0:
        return 0.0
    return total_return / max_drawdown




