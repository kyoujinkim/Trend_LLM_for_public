"""
Time Series Analyzer Module
Analyzes temporal patterns and cycles in keyword/topic data
"""
import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import linregress

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesAnalyzer:
    """Analyze time series patterns in keyword data"""

    def __init__(self, window_size: int = 7):
        """
        Initialize time series analyzer

        Args:
            window_size: Window size for moving average (in days)
        """
        self.window_size = window_size

    def calculate_moving_average(self, series: pd.Series) -> pd.Series:
        """
        Calculate moving average

        Args:
            series: Time series data

        Returns:
            Moving average series
        """
        return series.rolling(window=self.window_size, min_periods=1).mean()

    def detect_trend(self, series: pd.Series) -> Dict:
        """
        Detect trend in time series using linear regression

        Args:
            series: Time series data

        Returns:
            Dictionary with trend information
        """
        if len(series) < 2:
            return {
                'trend': 'insufficient_data',
                'slope': 0,
                'r_squared': 0,
                'p_value': 1.0
            }

        # Remove NaN values
        clean_series = series.dropna()
        if len(clean_series) < 2:
            return {
                'trend': 'insufficient_data',
                'slope': 0,
                'r_squared': 0,
                'p_value': 1.0
            }

        # Perform linear regression
        x = np.arange(len(clean_series))
        y = clean_series.values

        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        # Determine trend direction
        if p_value < 0.05:  # Statistically significant
            if slope > 0:
                trend = 'increasing'
            elif slope < 0:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'no_significant_trend'

        return {
            'trend': trend,
            'slope': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'intercept': intercept
        }

    def detect_cycles(self, series: pd.Series, min_period: int = 3) -> Dict:
        """
        Detect cyclical patterns using FFT

        Args:
            series: Time series data
            min_period: Minimum period length to detect

        Returns:
            Dictionary with cycle information
        """
        clean_series = series.dropna()

        if len(clean_series) < min_period * 2:
            return {
                'has_cycle': False,
                'period': None,
                'strength': 0
            }

        # Remove trend
        detrended = signal.detrend(clean_series.values)

        # Perform FFT
        fft = np.fft.fft(detrended)
        frequencies = np.fft.fftfreq(len(detrended))

        # Get power spectrum
        power = np.abs(fft) ** 2

        # Find dominant frequency (excluding DC component)
        positive_freqs = frequencies[1:len(frequencies)//2]
        positive_power = power[1:len(power)//2]

        if len(positive_power) == 0:
            return {
                'has_cycle': False,
                'period': None,
                'strength': 0
            }

        max_power_idx = np.argmax(positive_power)
        dominant_freq = positive_freqs[max_power_idx]

        if dominant_freq > 0:
            period = 1 / dominant_freq
            strength = positive_power[max_power_idx] / positive_power.sum()

            # Only consider significant cycles
            has_cycle = strength > 0.1 and period >= min_period

            return {
                'has_cycle': has_cycle,
                'period': period if has_cycle else None,
                'strength': strength
            }
        else:
            return {
                'has_cycle': False,
                'period': None,
                'strength': 0
            }

    def calculate_volatility(self, series: pd.Series) -> float:
        """
        Calculate volatility (standard deviation of changes)

        Args:
            series: Time series data

        Returns:
            Volatility measure
        """
        changes = series.diff().dropna()
        return changes.std()

    def identify_peaks_and_troughs(self, series: pd.Series,
                                   prominence: float = None) -> Dict:
        """
        Identify peaks and troughs in time series

        Args:
            series: Time series data
            prominence: Minimum prominence for peak detection

        Returns:
            Dictionary with peaks and troughs
        """
        clean_series = series.dropna()

        if len(clean_series) < 3:
            return {
                'peaks': [],
                'troughs': [],
                'num_peaks': 0,
                'num_troughs': 0
            }

        values = clean_series.values

        # Find peaks
        peaks, _ = signal.find_peaks(values, prominence=prominence)

        # Find troughs (peaks of inverted series)
        troughs, _ = signal.find_peaks(-values, prominence=prominence)

        return {
            'peaks': clean_series.index[peaks].tolist(),
            'troughs': clean_series.index[troughs].tolist(),
            'num_peaks': len(peaks),
            'num_troughs': len(troughs)
        }

    def analyze_keyword_series(self, series: pd.Series, keyword: str) -> Dict:
        """
        Comprehensive analysis of a keyword time series

        Args:
            series: Time series data for keyword
            keyword: Keyword name

        Returns:
            Dictionary with complete analysis
        """
        analysis = {
            'keyword': keyword,
            'data_points': len(series),
            'total_occurrences': series.sum(),
            'mean_frequency': series.mean(),
            'max_frequency': series.max(),
            'min_frequency': series.min()
        }

        # Moving average
        ma = self.calculate_moving_average(series)
        analysis['moving_average'] = ma.tolist()

        # Trend detection
        trend_info = self.detect_trend(series)
        analysis.update({
            'trend': trend_info['trend'],
            'trend_slope': trend_info['slope'],
            'trend_r_squared': trend_info['r_squared'],
            'trend_p_value': trend_info['p_value']
        })

        # Cycle detection
        cycle_info = self.detect_cycles(series)
        analysis.update({
            'has_cycle': cycle_info['has_cycle'],
            'cycle_period': cycle_info['period'],
            'cycle_strength': cycle_info['strength']
        })

        # Volatility
        analysis['volatility'] = self.calculate_volatility(series)

        # Peaks and troughs
        peaks_troughs = self.identify_peaks_and_troughs(series)
        analysis.update(peaks_troughs)

        return analysis

    def analyze_all_keywords(self, freq_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze all keywords in frequency dataframe

        Args:
            freq_df: DataFrame with keyword frequencies over time

        Returns:
            DataFrame with analysis results for all keywords
        """
        results = []

        logger.info(f"Analyzing {len(freq_df.columns)} keywords...")

        for keyword in freq_df.columns:
            series = freq_df[keyword]
            analysis = self.analyze_keyword_series(series, keyword)
            results.append(analysis)

        results_df = pd.DataFrame(results)

        # Sort by total occurrences
        results_df = results_df.sort_values('total_occurrences', ascending=False)

        logger.info("Time series analysis completed")
        return results_df

    def classify_keyword_status(self, analysis: Dict) -> str:
        """
        Classify keyword status based on analysis

        Args:
            analysis: Analysis dictionary for a keyword

        Returns:
            Status classification
        """
        trend = analysis['trend']
        volatility = analysis['volatility']
        mean_freq = analysis['mean_frequency']

        if trend == 'increasing':
            if mean_freq > 10:
                return 'dominant_rising'
            else:
                return 'emerging'
        elif trend == 'decreasing':
            if mean_freq < 2:
                return 'declining_fast'
            else:
                return 'declining'
        elif trend == 'stable':
            if mean_freq > 10:
                return 'dominant_stable'
            else:
                return 'stable'
        else:
            if volatility > mean_freq * 0.5:
                return 'volatile'
            else:
                return 'uncertain'

    def get_keyword_classifications(self, analysis_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Classify all keywords by status

        Args:
            analysis_df: DataFrame with keyword analyses

        Returns:
            Dictionary mapping status to keyword lists
        """
        classifications = {}

        for _, row in analysis_df.iterrows():
            status = self.classify_keyword_status(row.to_dict())

            if status not in classifications:
                classifications[status] = []

            classifications[status].append(row['keyword'])

        return classifications
