
import json
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Any
from pathlib import Path

class LazyCallable:
    def __init__(self, name):
        self.n, self.f = name, None
    
    def __call__(self, *a, **k):
        if self.f is None:
            modn, funcn = self.n.rsplit('.', 1)
            if modn not in sys.modules:
                __import__(modn)
            self.f = getattr(sys.modules[modn], funcn)
        return self.f(*a, **k)

class TechnicalIndicators:
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame, indicators: Dict[str, Dict]) -> pd.DataFrame:
        df_result = df.copy()
        
        for indicator, params in indicators.items():
            time_periods = params.get('time_periods', [])
            input_columns = params.get('input_columns', [])
            output_columns = params.get('output_columns', [])
            
            if isinstance(input_columns, str):
                input_columns = [input_columns]
            if isinstance(output_columns, str):
                output_columns = [output_columns]
            if not isinstance(time_periods, list) or time_periods == "":
                time_periods = [""]
            
            for time_period in time_periods:
                indicator_func = LazyCallable(f'talib.{indicator}')
                
                if len(output_columns) > 1:
                    column_names = [f'{indicator}{col}{time_period}' for col in output_columns]
                    
                    if time_period:
                        outputs = indicator_func(*[df_result[col] for col in input_columns], timeperiod=time_period)
                    else:
                        outputs = indicator_func(*[df_result[col] for col in input_columns])
                    
                    for col, output in zip(column_names, outputs):
                        df_result[col] = output
                else:
                    column_name = f'{indicator}{time_period}'
                    
                    if time_period:
                        df_result[column_name] = indicator_func(*[df_result[col] for col in input_columns], timeperiod=time_period)
                    else:
                        df_result[column_name] = indicator_func(*[df_result[col] for col in input_columns])
        
        return df_result

class RollingFeatures:
    SUPPORTED_FUNCTIONS = ['mean', 'sum', 'max', 'min', 'var', 'std', 'skew', 'kurt', 'shift', 'diff']
    
    @staticmethod
    def add_rolling_functions(df: pd.DataFrame, column_names: List[str], 
                            window_sizes: List[Union[int, str]], 
                            functions: List[str]) -> pd.DataFrame:
        df_result = df.copy()
        
        for column_name in column_names:
            if column_name not in df_result.columns:
                continue
                
            for window_size in window_sizes:
                for func in functions:
                    if func not in RollingFeatures.SUPPORTED_FUNCTIONS:
                        raise ValueError(f"Unsupported function: {func}")
                    
                    column_suffix = f'{column_name}{func.title()}{window_size}'
                    rolling_obj = df_result[column_name].rolling(window=window_size)
                    
                    if func == 'mean':
                        df_result[column_suffix] = rolling_obj.mean()
                    elif func == 'sum':
                        df_result[column_suffix] = rolling_obj.sum()
                    elif func == 'max':
                        df_result[column_suffix] = rolling_obj.max()
                    elif func == 'min':
                        df_result[column_suffix] = rolling_obj.min()
                    elif func == 'var':
                        df_result[column_suffix] = rolling_obj.var()
                    elif func == 'std':
                        df_result[column_suffix] = rolling_obj.std()
                    elif func == 'skew':
                        df_result[column_suffix] = rolling_obj.skew()
                    elif func == 'kurt':
                        df_result[column_suffix] = rolling_obj.kurt()
                    elif func == 'shift':
                        df_result[column_suffix] = df_result[column_name].shift(window_size)
                    elif func == 'diff':
                        df_result[column_suffix] = df_result[column_name].diff(window_size)
        
        return df_result

class PercentageChanges:
    PERIOD_MAP = {'W': 5, 'M': 21, 'Q': 63, 'Y': 252, '3Y': 756}
    
    @staticmethod
    def add_percentage_change(df: pd.DataFrame, column_name: str, periods: List[Union[str, int]]) -> pd.DataFrame:
        df_result = df.copy()
        
        for period in periods:
            if period == 'YTD':
                first_value = df_result[column_name].iloc[0]
                if first_value != 0:
                    df_result['YTD'] = (df_result[column_name] / first_value - 1) * 100
                else:
                    df_result['YTD'] = 0
            else:
                period_value = PercentageChanges.PERIOD_MAP.get(period, period)
                new_column_name = f'Chg{period}'
                df_result[new_column_name] = df_result[column_name].pct_change(periods=period_value) * 100
        
        return df_result

class PivotPoints:
    STANDARD_FORMULAS = {
        'PP': '(H + L + C) / 3',
        'S1': '(2 * PP) - H',
        'S2': 'PP - (H - L)',
        'S3': 'L - 2 * (H - PP)',
        'R1': '(2 * PP) - L',
        'R2': 'PP + (H - L)',
        'R3': 'H + 2 * (PP - L)'
    }
    
    WOODIE_FORMULAS = {
        'PP': '(H + L + 2 * C) / 4',
        'S1': '(2 * PP) - H',
        'S2': 'PP - (H - L)',
        'R1': '(2 * PP) - L',
        'R2': 'PP + (H - L)'
    }
    
    CAMARILLA_FORMULAS = {
        'PP': '(H + L + C) / 3',
        'S1': 'C - (H - L) * 1.1 / 12',
        'S2': 'C - (H - L) * 1.1 / 6',
        'S3': 'C - (H - L) * 1.1 / 4',
        'S4': 'C - (H - L) * 1.1 / 2',
        'R1': 'C + (H - L) * 1.1 / 12',
        'R2': 'C + (H - L) * 1.1 / 6',
        'R3': 'C + (H - L) * 1.1 / 4',
        'R4': 'C + (H - L) * 1.1 / 2'
    }
    
    @staticmethod
    def calculate_pivot_points(df: pd.DataFrame, suffix: str = '', 
                             pivot_type: str = 'standard') -> pd.DataFrame:
        df_result = df.copy()
        
        if pivot_type == 'standard':
            formulas = PivotPoints.STANDARD_FORMULAS
        elif pivot_type == 'woodie':
            formulas = PivotPoints.WOODIE_FORMULAS
        elif pivot_type == 'camarilla':
            formulas = PivotPoints.CAMARILLA_FORMULAS
        else:
            raise ValueError(f"Unsupported pivot type: {pivot_type}")
        
        high_col = f'High{suffix}' if f'High{suffix}' in df_result.columns else 'high'
        low_col = f'Low{suffix}' if f'Low{suffix}' in df_result.columns else 'low' 
        close_col = f'Close{suffix}' if f'Close{suffix}' in df_result.columns else 'close'
        
        for col, formula in formulas.items():
            formula_expr = formula.replace('H', f'df_result["{high_col}"]')
            formula_expr = formula_expr.replace('L', f'df_result["{low_col}"]')
            formula_expr = formula_expr.replace('C', f'df_result["{close_col}"]')
            formula_expr = formula_expr.replace('PP', 'df_result["PP"]')
            
            df_result[col] = eval(formula_expr)
        
        return df_result
    
    @staticmethod
    def calculate_pivot_location(df: pd.DataFrame, column: str, suffix: str = '',
                               pivot_points: List[str] = ['S3', 'S2', 'S1', 'PP', 'R1', 'R2', 'R3'],
                               choices: List[Any] = None) -> np.ndarray:
        if choices is None:
            choices = list(range(len(pivot_points) + 1))
        
        price_col = column + suffix
        conditions = []
        
        for i in range(len(pivot_points) - 1):
            condition = (df[price_col] > df[pivot_points[i]]) & (df[price_col] < df[pivot_points[i + 1]])
            conditions.append(condition)
        
        conditions.append(df[price_col] > df[pivot_points[-1]])
        conditions.append(df[price_col] < df[pivot_points[0]])
        
        choices_adjusted = choices[:len(conditions)]
        return np.select(conditions, choices_adjusted, default=np.nan)

class FibonacciLevels:
    STANDARD_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]
    EXTENDED_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.707, 0.786, 0.886, 1.382, 1.5, 1.618, 1.786, 1.886, 2.0, 2.618, 2.786, 2.886]
    IMPORTANT_LEVELS = [1.786, 1.886, 2.786, 2.886]
    
    @staticmethod
    def calculate_fib_levels(start: float, end: float, levels: List[float]) -> List[float]:
        ratios = sorted([0.0] + levels + [1.0])
        level_prices = [round(start + ratio * (end - start), 6) for ratio in ratios]
        return level_prices[1:-1]
    
    @staticmethod
    def add_fibonacci_levels(df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low',
                           levels: List[float] = None, level_type: str = 'standard') -> pd.DataFrame:
        df_result = df.copy()
        
        if levels is None:
            levels = FibonacciLevels.EXTENDED_LEVELS if level_type == 'extended' else FibonacciLevels.STANDARD_LEVELS
        
        fib_data = df_result.apply(
            lambda row: FibonacciLevels.calculate_fib_levels(row[low_col], row[high_col], levels), 
            axis=1
        )
        
        fib_df = pd.DataFrame(fib_data.tolist(), index=df_result.index)
        level_names = [f'fib_{level}' for level in levels]
        fib_df.columns = level_names
        
        return pd.concat([df_result, fib_df], axis=1)

class PriceTransformations:
    @staticmethod
    def add_basic_transformations(df: pd.DataFrame, 
                                open_col: str = 'open',
                                high_col: str = 'high', 
                                low_col: str = 'low',
                                close_col: str = 'close',
                                volume_col: str = 'volume') -> pd.DataFrame:
        df_result = df.copy()
        
        df_result['ohlc_average'] = (df_result[open_col] + df_result[high_col] + 
                                   df_result[low_col] + df_result[close_col]) / 4
        df_result['hl_average'] = (df_result[high_col] + df_result[low_col]) / 2
        df_result['oc_average'] = (df_result[open_col] + df_result[close_col]) / 2
        
        df_result['hl_range'] = df_result[high_col] - df_result[low_col]
        df_result['oc_range'] = abs(df_result[open_col] - df_result[close_col])
        
        df_result['upper_shadow'] = df_result[high_col] - df_result[[open_col, close_col]].max(axis=1)
        df_result['lower_shadow'] = df_result[[open_col, close_col]].min(axis=1) - df_result[low_col]
        df_result['real_body'] = abs(df_result[close_col] - df_result[open_col])
        
        df_result['typical_price'] = (df_result[high_col] + df_result[low_col] + df_result[close_col]) / 3
        df_result['weighted_close'] = (df_result[high_col] + df_result[low_col] + 2 * df_result[close_col]) / 4
        
        if volume_col in df_result.columns:
            df_result['price_volume'] = df_result[close_col] * df_result[volume_col]
            df_result['vwap_approx'] = (df_result['price_volume'].rolling(20).sum() / 
                                      df_result[volume_col].rolling(20).sum())
        
        for col in [open_col, high_col, low_col, close_col]:
            df_result[f'{col}_change'] = df_result[col].pct_change() * 100
            df_result[f'{col}_change_abs'] = df_result[f'{col}_change'].abs()
        
        return df_result
    
    @staticmethod
    def add_price_patterns(df: pd.DataFrame,
                         open_col: str = 'open', 
                         high_col: str = 'high',
                         low_col: str = 'low', 
                         close_col: str = 'close') -> pd.DataFrame:
        df_result = df.copy()
        
        body_size = abs(df_result[close_col] - df_result[open_col])
        range_size = df_result[high_col] - df_result[low_col]
        upper_shadow = df_result[high_col] - df_result[[open_col, close_col]].max(axis=1)
        lower_shadow = df_result[[open_col, close_col]].min(axis=1) - df_result[low_col]
        
        df_result['doji'] = (body_size / (range_size + 1e-8) < 0.1).astype(int)
        df_result['hammer'] = ((lower_shadow > 2 * body_size) & 
                             (upper_shadow < 0.1 * range_size)).astype(int)
        df_result['shooting_star'] = ((upper_shadow > 2 * body_size) & 
                                    (lower_shadow < 0.1 * range_size)).astype(int)
        df_result['spinning_top'] = ((body_size < 0.3 * range_size) & 
                                   (upper_shadow > 0.1 * range_size) & 
                                   (lower_shadow > 0.1 * range_size)).astype(int)
        
        df_result['bullish_candle'] = (df_result[close_col] > df_result[open_col]).astype(int)
        df_result['bearish_candle'] = (df_result[close_col] < df_result[open_col]).astype(int)
        
        return df_result

class AdvancedFeatures:
    @staticmethod
    def calculate_close_to_close_volatility(df, close_col='close', windows=[30], trading_periods=[252], clean=False):
        df_result = df.copy()
        for trading_period in trading_periods:
            for window in windows:
                df_result[f'log_return_{trading_period}_{window}'] = np.log(df_result[close_col] / df_result[close_col].shift(1))
                df_result[f'c_vol_{trading_period}_{window}'] = df_result[f'log_return_{trading_period}_{window}'].rolling(window=window).std() * np.sqrt(trading_period) * 100
                df_result.drop(columns=[f'log_return_{trading_period}_{window}'], inplace=True)
        if clean:
            df_result = df_result.dropna()
        return df_result

    @staticmethod
    def calculate_parkinson_volatility(df, high_col='high', low_col='low', windows=[30], trading_periods=[252], clean=False):
        df_result = df.copy()
        for trading_period in trading_periods:
            for window in windows:
                rs = (1.0 / (4.0 * np.log(2.0))) * ((df_result[high_col] / df_result[low_col]).apply(np.log)) ** 2.0
                def f(v):
                    return (trading_period * v.mean()) ** 0.5
                result_name = f'p_vol_{trading_period}_{window}'
                df_result[result_name] = rs.rolling(window=window, center=False).apply(func=f) * 100
        if clean:
            df_result = df_result.dropna()
        return df_result

    @staticmethod
    def calculate_garman_klass_volatility(df, high_col='high', low_col='low', close_col='close', open_col='open', windows=[30], trading_periods=[252], clean=False):
        df_result = df.copy()
        for trading_period in trading_periods:
            for window in windows:
                log_hl = np.log(df_result[high_col] / df_result[low_col])
                log_co = np.log(df_result[close_col] / df_result[open_col])
                rs = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
                def f(v):
                    return (trading_period * v.mean()) ** 0.5
                result_col_name = f'gk_vol_{trading_period}_{window}'
                df_result[result_col_name] = rs.rolling(window=window, center=False).apply(func=f) * 100
        if clean:
            df_result = df_result.dropna()
        return df_result

    @staticmethod
    def calculate_hodges_tompkins_volatility(df, close_col='close', windows=[30], trading_periods=[252], clean=False):
        df_result = df.copy()
        for trading_period in trading_periods:
            for window in windows:
                log_return = np.log(df_result[close_col] / df_result[close_col].shift(1))
                vol = log_return.rolling(window=window, center=False).std() * np.sqrt(trading_period)
                h = window
                n = (log_return.count() - h) + 1
                adj_factor = 1.0 / (1.0 - (h / n) + ((h ** 2 - 1) / (3 * n ** 2)))
                df_result[f'ht_vol_{trading_period}_{window}'] = vol * adj_factor * 100
        if clean:
            df_result = df_result.dropna()
        return df_result

    @staticmethod
    def calculate_rogers_satchell_volatility(df, high_col='high', low_col='low', close_col='close', open_col='open', windows=[30], trading_periods=[252], clean=False):
        df_result = df.copy()
        for trading_period in trading_periods:
            for window in windows:
                log_ho = np.log(df_result[high_col] / df_result[open_col])
                log_lo = np.log(df_result[low_col] / df_result[open_col])
                log_co = np.log(df_result[close_col] / df_result[open_col])
                rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
                def f(v):
                    return (trading_period * v.mean()) ** 0.5
                df_result[f'rs_vol_{trading_period}_{window}'] = rs.rolling(window=window, center=False).apply(func=f) * 100
        if clean:
            df_result = df_result.dropna()
        return df_result

    @staticmethod
    def calculate_yang_zhang_volatility(df, high_col='high', low_col='low', close_col='close', open_col='open', windows=[30], trading_periods=[252], clean=False):
        df_result = df.copy()
        for trading_period in trading_periods:
            for window in windows:
                log_ho = np.log(df_result[high_col] / df_result[open_col])
                log_lo = np.log(df_result[low_col] / df_result[open_col])
                log_co = np.log(df_result[close_col] / df_result[open_col])
                
                log_oc = np.log(df_result[open_col] / df_result[close_col].shift(1))
                log_oc_sq = log_oc ** 2
                
                log_cc = np.log(df_result[close_col] / df_result[close_col].shift(1))
                log_cc_sq = log_cc ** 2
                
                rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
                
                close_vol = log_cc_sq.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
                open_vol = log_oc_sq.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
                window_rs = rs.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
                
                k = 0.34 / (1.34 + (window + 1) / (window - 1))
                
                df_result[f'yz_vol_{trading_period}_{window}'] = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_period) * 100
        if clean:
            df_result = df_result.dropna()
        return df_result

    @staticmethod
    def add_volatility_features(df: pd.DataFrame, close_col: str = 'close',
                              high_col: str = 'high', low_col: str = 'low', open_col: str = 'open',
                              windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        df_result = df.copy()
        
        returns = df_result[close_col].pct_change()
        
        for window in windows:
            df_result[f'volatility_{window}'] = returns.rolling(window).std() * np.sqrt(252)
        
        df_result = AdvancedFeatures.calculate_parkinson_volatility(df_result, high_col, low_col, windows)
        df_result = AdvancedFeatures.calculate_garman_klass_volatility(df_result, high_col, low_col, close_col, open_col, windows)
        df_result = AdvancedFeatures.calculate_close_to_close_volatility(df_result, close_col, windows)
        df_result = AdvancedFeatures.calculate_hodges_tompkins_volatility(df_result, close_col, windows)
        df_result = AdvancedFeatures.calculate_rogers_satchell_volatility(df_result, high_col, low_col, close_col, open_col, windows)
        df_result = AdvancedFeatures.calculate_yang_zhang_volatility(df_result, high_col, low_col, close_col, open_col, windows)
        
        df_result['vol_regime'] = (df_result['volatility_20'] > df_result['volatility_20'].rolling(50).mean()).astype(int)
        
        return df_result

class TimeBasedFeatures:
    @staticmethod
    def add_time_features(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        df_result = df.copy()
        
        df_result['hour'] = df_result[timestamp_col].dt.hour
        df_result['day_of_week'] = df_result[timestamp_col].dt.dayofweek
        df_result['month'] = df_result[timestamp_col].dt.month
        df_result['is_weekend'] = (df_result['day_of_week'] >= 5).astype(int)
        
        df_result['trading_session'] = pd.cut(df_result['hour'], bins=[0, 8, 16, 24], 
                                            labels=['Asian', 'European', 'American'], include_lowest=True)
        
        return df_result

class CategoricalFeatures:
    @staticmethod
    def add_categorical_features(df: pd.DataFrame, close_col: str = 'close',
                               rsi_col: str = 'RSI14', volatility_col: str = 'volatility_20') -> pd.DataFrame:
        df_result = df.copy()
        
        if f'{close_col}_change' in df_result.columns:
            df_result['price_trend'] = pd.cut(df_result[f'{close_col}_change'], 
                                            bins=[-np.inf, -0.5, 0.5, np.inf], 
                                            labels=['Down', 'Flat', 'Up'])
        
        if volatility_col in df_result.columns:
            df_result['volatility_level'] = pd.cut(df_result[volatility_col], bins=3, 
                                                 labels=['Low', 'Medium', 'High'])
        
        if 'volume_ratio' in df_result.columns:
            df_result['volume_level'] = pd.cut(df_result['volume_ratio'], 
                                             bins=[0, 0.8, 1.2, np.inf], 
                                             labels=['Low', 'Normal', 'High'])
        
        if rsi_col in df_result.columns:
            df_result['rsi_zone'] = pd.cut(df_result[rsi_col], bins=[0, 30, 70, 100], 
                                         labels=['Oversold', 'Neutral', 'Overbought'])
        
        return df_result

class DataProcessor:
    def __init__(self, config_path=None):
        self.tech_indicators = TechnicalIndicators()
        self.rolling_features = RollingFeatures()
        self.percentage_changes = PercentageChanges()
        self.pivot_points = PivotPoints()
        self.fibonacci = FibonacciLevels()
        self.price_transforms = PriceTransformations()
        self.advanced_features = AdvancedFeatures()
        self.time_features = TimeBasedFeatures()
        self.categorical_features = CategoricalFeatures()
        
        self.config = self.load_config(config_path) if config_path else {}
    
    def load_config(self, config_path):
        if isinstance(config_path, (str, Path)):
            with open(config_path, 'r') as f:
                return json.load(f)
        elif isinstance(config_path, dict):
            return config_path
        return {}
    
    def process_dataframe(self, df, 
                        add_patterns=True,
                        add_volatility=True,
                        add_momentum=True,
                        add_fibonacci=False,
                        add_pivots=False,
                        add_time_features=True,
                        add_categorical=True):
        df_result = df.copy()
        
        if 'technical_indicators' in self.config:
            df_result = self.tech_indicators.add_technical_indicators(
                df_result, self.config['technical_indicators']
            )
        
        if 'rolling_features' in self.config:
            config = self.config['rolling_features']
            df_result = self.rolling_features.add_rolling_functions(
                df_result, 
                config.get('columns', ['close']),
                config.get('windows', [20]),
                config.get('functions', ['mean'])
            )
        
        if 'percentage_changes' in self.config:
            config = self.config['percentage_changes']
            for column, periods in config.items():
                df_result = self.percentage_changes.add_percentage_change(
                    df_result, column, periods
                )
        
        if 'pivot_points' in self.config or add_pivots:
            config = self.config.get('pivot_points', {})
            df_result = self.pivot_points.calculate_pivot_points(
                df_result,
                suffix=config.get('suffix', ''),
                pivot_type=config.get('type', 'standard')
            )
        
        if 'fibonacci' in self.config or add_fibonacci:
            config = self.config.get('fibonacci', {})
            df_result = self.fibonacci.add_fibonacci_levels(
                df_result,
                high_col=config.get('high_col', 'high'),
                low_col=config.get('low_col', 'low'),
                levels=config.get('levels'),
                level_type=config.get('level_type', 'standard')
            )
        
        if 'price_transforms' in self.config:
            config = self.config['price_transforms']
            if config.get('basic', True):
                df_result = self.price_transforms.add_basic_transformations(
                    df_result,
                    open_col=config.get('open_col', 'open'),
                    high_col=config.get('high_col', 'high'),
                    low_col=config.get('low_col', 'low'),
                    close_col=config.get('close_col', 'close'),
                    volume_col=config.get('volume_col', 'volume')
                )
            if config.get('patterns', False) or add_patterns:
                df_result = self.price_transforms.add_price_patterns(
                    df_result,
                    open_col=config.get('open_col', 'open'),
                    high_col=config.get('high_col', 'high'),
                    low_col=config.get('low_col', 'low'),
                    close_col=config.get('close_col', 'close')
                )
        else:
            df_result = self.price_transforms.add_basic_transformations(df_result)
            if add_patterns:
                df_result = self.price_transforms.add_price_patterns(df_result)
        
        if 'advanced_features' in self.config:
            config = self.config['advanced_features']
            if config.get('volatility', False) or add_volatility:
                df_result = self.advanced_features.add_volatility_features(
                    df_result,
                    close_col=config.get('close_col', 'close'),
                    high_col=config.get('high_col', 'high'),
                    low_col=config.get('low_col', 'low'),
                    windows=config.get('windows', [5, 10, 20, 50])
                )
            if config.get('momentum', False) or add_momentum:
                df_result = self.advanced_features.add_momentum_features(
                    df_result,
                    close_col=config.get('close_col', 'close'),
                    volume_col=config.get('volume_col', 'volume'),
                    periods=config.get('periods', [1, 3, 5, 10, 21])
                )
        else:
            if add_volatility:
                df_result = self.advanced_features.add_volatility_features(df_result)
            if add_momentum:
                df_result = self.advanced_features.add_momentum_features(df_result)

        if add_time_features and 'timestamp' in df_result.columns:
            df_result = self.time_features.add_time_features(df_result)
        
        if add_categorical:
            df_result = self.categorical_features.add_categorical_features(df_result)        
        
        return df_result.fillna(method='ffill').fillna(0)