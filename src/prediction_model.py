"""
Prediction Model Module
Predicts keyword lifecycle and extinction probabilities
"""
import logging
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from scipy.stats import norm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KeywordLifecyclePredictor:
    """Predict keyword lifecycle and extinction probability"""

    def __init__(self, prediction_horizon_days: int = 365):
        """
        Initialize predictor

        Args:
            prediction_horizon_days: Number of days to predict ahead
        """
        self.prediction_horizon_days = prediction_horizon_days

    def calculate_extinction_probability(self, series: pd.Series,
                                        trend_slope: float,
                                        volatility: float) -> Dict:
        """
        Calculate probability of keyword extinction

        Args:
            series: Time series data
            trend_slope: Trend slope from time series analysis
            volatility: Volatility measure

        Returns:
            Dictionary with extinction probabilities
        """
        current_freq = series.iloc[-1] if len(series) > 0 else 0
        mean_freq = series.mean()

        if current_freq == 0 or mean_freq == 0:
            return {
                '30_days': 1.0,
                '90_days': 1.0,
                '180_days': 1.0,
                '365_days': 1.0,
                'confidence': 'high'
            }

        # Calculate decay rate based on trend
        if trend_slope < 0:
            # Declining trend - higher extinction probability
            decay_rate = abs(trend_slope) / mean_freq
        else:
            # Stable or rising - lower extinction probability
            decay_rate = 0.001  # minimal decay

        # Adjust for volatility
        volatility_factor = 1 + (volatility / mean_freq)

        # Calculate extinction probabilities for different horizons
        probabilities = {}
        horizons = [30, 90, 180, 365]

        for days in horizons:
            # Exponential decay model
            expected_freq = current_freq * np.exp(-decay_rate * days * volatility_factor)

            # Probability that frequency drops below threshold (1)
            if volatility > 0:
                z_score = (1 - expected_freq) / (volatility * np.sqrt(days / 30))
                prob = norm.cdf(z_score)
            else:
                prob = 1.0 if expected_freq < 1 else 0.0

            # Clip to [0, 1]
            prob = max(0.0, min(1.0, prob))
            probabilities[f'{days}_days'] = prob

        # Determine confidence based on data quality
        if len(series) >= 7 and volatility < mean_freq:
            confidence = 'high'
        elif len(series) >= 3:
            confidence = 'medium'
        else:
            confidence = 'low'

        probabilities['confidence'] = confidence

        return probabilities

    def predict_future_frequency(self, series: pd.Series,
                                 trend_slope: float,
                                 days_ahead: int = 30) -> Tuple[float, float, float]:
        """
        Predict future frequency with confidence intervals

        Args:
            series: Time series data
            trend_slope: Trend slope
            days_ahead: Number of days to predict ahead

        Returns:
            Tuple of (predicted_value, lower_bound, upper_bound)
        """
        if len(series) == 0:
            return (0, 0, 0)

        current_freq = series.iloc[-1]
        std_dev = series.std()

        # Linear projection
        predicted = current_freq + (trend_slope * days_ahead)
        predicted = max(0, predicted)  # Cannot be negative

        # Confidence interval (95%)
        margin = 1.96 * std_dev
        lower_bound = max(0, predicted - margin)
        upper_bound = predicted + margin

        return (predicted, lower_bound, upper_bound)

    def estimate_lifecycle_stage(self, analysis: Dict) -> Dict:
        """
        Estimate current lifecycle stage of keyword

        Args:
            analysis: Analysis dictionary from TimeSeriesAnalyzer

        Returns:
            Dictionary with lifecycle information
        """
        trend = analysis['trend']
        mean_freq = analysis['mean_frequency']
        volatility = analysis['volatility']
        total_occurrences = analysis['total_occurrences']

        # Determine lifecycle stage
        if trend == 'increasing' and mean_freq < 5:
            stage = 'introduction'
            description = 'Emerging topic, gaining attention'
        elif trend == 'increasing' and mean_freq >= 5:
            stage = 'growth'
            description = 'Rapidly growing topic, high momentum'
        elif trend == 'stable' and mean_freq >= 10:
            stage = 'maturity'
            description = 'Established topic, stable presence'
        elif trend == 'decreasing' and mean_freq >= 5:
            stage = 'decline'
            description = 'Declining topic, losing attention'
        elif trend == 'decreasing' and mean_freq < 5:
            stage = 'obsolescence'
            description = 'Fading topic, nearing extinction'
        else:
            stage = 'uncertain'
            description = 'Insufficient data or unclear pattern'

        # Estimate remaining lifetime
        if trend == 'decreasing' and analysis['trend_slope'] < 0:
            # Calculate days until frequency reaches near-zero
            current_freq = analysis['max_frequency']  # Use recent peak
            slope = analysis['trend_slope']
            days_remaining = int(-current_freq / slope) if slope < 0 else 999
            days_remaining = max(0, min(days_remaining, 999))
        else:
            days_remaining = None

        return {
            'stage': stage,
            'description': description,
            'days_remaining_estimate': days_remaining,
            'confidence': 'medium'
        }

    def predict_keyword_lifecycle(self, series: pd.Series,
                                  analysis: Dict) -> Dict:
        """
        Complete lifecycle prediction for a keyword

        Args:
            series: Time series data
            analysis: Analysis dictionary from TimeSeriesAnalyzer

        Returns:
            Complete prediction dictionary
        """
        keyword = analysis['keyword']
        trend_slope = analysis['trend_slope']
        volatility = analysis['volatility']

        # Extinction probabilities
        extinction_prob = self.calculate_extinction_probability(
            series, trend_slope, volatility
        )

        # Future frequency predictions
        predictions = {}
        for days in [30, 90, 180, 365]:
            pred, lower, upper = self.predict_future_frequency(
                series, trend_slope, days
            )
            predictions[f'{days}_days'] = {
                'predicted': pred,
                'lower_bound': lower,
                'upper_bound': upper
            }

        # Lifecycle stage
        lifecycle = self.estimate_lifecycle_stage(analysis)

        return {
            'keyword': keyword,
            'current_frequency': series.iloc[-1] if len(series) > 0 else 0,
            'trend': analysis['trend'],
            'lifecycle_stage': lifecycle['stage'],
            'lifecycle_description': lifecycle['description'],
            'extinction_probabilities': extinction_prob,
            'frequency_predictions': predictions,
            'days_remaining_estimate': lifecycle['days_remaining_estimate']
        }

    def predict_all_keywords(self, freq_df: pd.DataFrame,
                            analysis_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for all keywords

        Args:
            freq_df: Frequency dataframe
            analysis_df: Analysis results dataframe

        Returns:
            DataFrame with predictions for all keywords
        """
        predictions = []

        logger.info(f"Generating predictions for {len(freq_df.columns)} keywords...")

        for keyword in freq_df.columns:
            if keyword not in analysis_df['keyword'].values:
                continue

            series = freq_df[keyword]
            analysis = analysis_df[analysis_df['keyword'] == keyword].iloc[0].to_dict()

            prediction = self.predict_keyword_lifecycle(series, analysis)
            predictions.append(prediction)

        # Convert to DataFrame
        pred_df = pd.DataFrame(predictions)

        # Flatten nested dictionaries
        if len(pred_df) > 0:
            pred_df['extinction_prob_365_days'] = pred_df['extinction_probabilities'].apply(
                lambda x: x['365_days']
            )
            pred_df['extinction_confidence'] = pred_df['extinction_probabilities'].apply(
                lambda x: x['confidence']
            )

        logger.info("Predictions completed")
        return pred_df

    def get_high_risk_keywords(self, pred_df: pd.DataFrame,
                              threshold: float = 0.6) -> pd.DataFrame:
        """
        Get keywords with high extinction risk

        Args:
            pred_df: Predictions dataframe
            threshold: Probability threshold for high risk

        Returns:
            DataFrame with high-risk keywords
        """
        high_risk = pred_df[
            (pred_df['extinction_prob_365_days'] >= threshold) &
            (pred_df['lifecycle_stage'].isin(['decline', 'obsolescence'])) &
            (pred_df['trend'].isin(['decreasing']))
        ].copy()
        high_risk = high_risk.sort_values('extinction_prob_365_days', ascending=False)

        logger.info(f"Found {len(high_risk)} high-risk keywords")
        return high_risk

    def get_emerging_keywords(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get emerging keywords with growth potential

        Args:
            pred_df: Predictions dataframe

        Returns:
            DataFrame with emerging keywords
        """
        emerging = pred_df[
            (pred_df['lifecycle_stage'].isin(['introduction', 'growth'])) &
            (pred_df['trend'] == 'increasing') &
            (pred_df['extinction_prob_365_days'] < 0.9)
        ].copy()

        emerging = emerging.sort_values('current_frequency', ascending=False)

        logger.info(f"Found {len(emerging)} emerging keywords")
        return emerging
