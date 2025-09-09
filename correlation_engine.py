import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings

class CorrelationEngine:
    def __init__(self, data_path: str, pairs: List[str] = None, timeframe: str = "60", 
                 correlation_window: int = 50, spike_threshold: float = 0.8):
        self.data_path = Path(data_path)
        self.pairs = pairs or ['GBPUSD', 'XAUUSD']
        self.timeframe = timeframe
        self.correlation_window = correlation_window
        self.spike_threshold = spike_threshold
        
        self.pair_data = {}
        self.returns_data = pd.DataFrame()
        self.correlation_matrix = pd.DataFrame()
        self.correlation_history = []
        
        self._load_pair_data()
        self._initialize_returns()
    
    def _load_pair_data(self):
        for pair in self.pairs:
            csv_file = self.data_path / f"{pair}{self.timeframe}.csv"
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    if 'timestamp' not in df.columns:
                        df = pd.read_csv(csv_file, sep='\t', header=None, 
                                       names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    self.pair_data[pair] = df
                except Exception as e:
                    warnings.warn(f"Error loading {pair} data: {e}")
            else:
                warnings.warn(f"Data file not found: {csv_file}")
    
    def _initialize_returns(self):
        if not self.pair_data:
            return
        
        returns_dict = {}
        for pair, data in self.pair_data.items():
            returns_dict[pair] = data['close'].pct_change().dropna()
        
        self.returns_data = pd.DataFrame(returns_dict)
        self.returns_data = self.returns_data.dropna()
    
    def calculate_correlation_matrix(self, window_end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        if self.returns_data.empty:
            return pd.DataFrame()
        
        if window_end is None:
            data_slice = self.returns_data.tail(self.correlation_window)
        else:
            end_idx = self.returns_data.index.get_indexer([window_end], method='nearest')[0]
            start_idx = max(0, end_idx - self.correlation_window + 1)
            data_slice = self.returns_data.iloc[start_idx:end_idx + 1]
        
        if len(data_slice) < 10:
            return pd.DataFrame()
        
        correlation_matrix = data_slice.corr()
        self.correlation_matrix = correlation_matrix
        return correlation_matrix
    
    def update_correlation(self, current_timestamp: pd.Timestamp) -> pd.DataFrame:
        correlation_matrix = self.calculate_correlation_matrix(current_timestamp)
        if not correlation_matrix.empty:
            self.correlation_matrix = correlation_matrix
            self.correlation_history.append({
                'timestamp': current_timestamp,
                'correlation_matrix': correlation_matrix.copy()
            })
        return correlation_matrix
    
    def get_correlation_risk_score(self) -> float:
        if self.correlation_matrix.empty:
            return 0.0
        
        correlations = []
        n_pairs = len(self.correlation_matrix)
        
        for i in range(n_pairs):
            for j in range(i + 1, n_pairs):
                corr_value = abs(self.correlation_matrix.iloc[i, j])
                if not np.isnan(corr_value):
                    correlations.append(corr_value)
        
        if not correlations:
            return 0.0
        
        return np.mean(correlations)
    
    def detect_correlation_spikes(self) -> List[Tuple[str, str, float]]:
        if self.correlation_matrix.empty:
            return []
        
        spikes = []
        pairs = self.correlation_matrix.columns
        
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                corr_value = abs(self.correlation_matrix.iloc[i, j])
                if corr_value > self.spike_threshold:
                    spikes.append((pairs[i], pairs[j], corr_value))
        
        return spikes
    
    def check_conflicting_trades(self, positions: Dict[str, int]) -> List[Tuple[str, str, float]]:
        if self.correlation_matrix.empty:
            return []
        
        conflicts = []
        position_pairs = list(positions.keys())
        
        for i in range(len(position_pairs)):
            for j in range(i + 1, len(position_pairs)):
                pair1, pair2 = position_pairs[i], position_pairs[j]
                
                if pair1 in self.correlation_matrix.columns and pair2 in self.correlation_matrix.columns:
                    corr = self.correlation_matrix.loc[pair1, pair2]
                    
                    if abs(corr) > 0.7:
                        pos1, pos2 = positions[pair1], positions[pair2]
                        
                        if (corr > 0.7 and pos1 * pos2 < 0) or (corr < -0.7 and pos1 * pos2 > 0):
                            conflicts.append((pair1, pair2, corr))
        
        return conflicts
    
    def should_trigger_circuit_breaker(self, threshold: float = 0.85) -> bool:
        risk_score = self.get_correlation_risk_score()
        return risk_score > threshold
    
    def get_correlation_features(self, pair: str) -> Dict[str, float]:
        if self.correlation_matrix.empty or pair not in self.correlation_matrix.columns:
            return {}
        
        pair_correlations = self.correlation_matrix[pair].drop(pair)
        
        return {
            'max_correlation': pair_correlations.abs().max() if not pair_correlations.empty else 0.0,
            'avg_correlation': pair_correlations.abs().mean() if not pair_correlations.empty else 0.0,
            'correlation_exposure': (pair_correlations.abs() > 0.6).sum() / len(pair_correlations) if not pair_correlations.empty else 0.0
        }
    
    def get_portfolio_correlation_features(self) -> Dict[str, float]:
        if self.correlation_matrix.empty:
            return {
                'portfolio_correlation_risk': 0.0,
                'max_pair_correlation': 0.0,
                'correlation_diversity': 1.0
            }
        
        risk_score = self.get_correlation_risk_score()
        
        upper_triangle = np.triu(self.correlation_matrix.values, k=1)
        upper_triangle = upper_triangle[upper_triangle != 0]
        
        max_corr = np.abs(upper_triangle).max() if len(upper_triangle) > 0 else 0.0
        diversity = 1.0 - risk_score
        
        return {
            'portfolio_correlation_risk': risk_score,
            'max_pair_correlation': max_corr,
            'correlation_diversity': max(0.0, diversity)
        }