"""
Keyword Extractor Module
Extracts and ranks keywords/topics from news articles
"""
import logging
from typing import List, Dict, Tuple
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KeywordExtractor:
    """Extract and rank keywords from text data"""

    def __init__(self, top_n: int = 50, min_df: int = 2, max_df: float = 0.8, huggingface_model: str = 'LiquidAI/LFM2-350M-Extract'):
        """
        Initialize keyword extractor

        Args:
            top_n: Number of top keywords to extract
            min_df: Minimum document frequency for TF-IDF
            max_df: Maximum document frequency for TF-IDF
        """
        self.top_n = top_n
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None
        self.tfidf_matrix = None

    def extract_by_frequency(self, texts: List[str],
                            top_n: int = None) -> List[Tuple[str, int]]:
        """
        Extract keywords by frequency

        Args:
            texts: List of text documents
            top_n: Number of top keywords (uses self.top_n if None)

        Returns:
            List of (keyword, frequency) tuples
        """
        if top_n is None:
            top_n = self.top_n

        # Combine all texts and split into words
        all_words = []
        for text in texts:
            if isinstance(text, str):
                all_words.extend(text.split())

        # Count frequencies
        word_freq = Counter(all_words)

        # Get top N
        top_keywords = word_freq.most_common(top_n)

        logger.info(f"Extracted {len(top_keywords)} keywords by frequency")
        return top_keywords

    def extract_by_tfidf(self, texts: List[str], top_n: int = None, ngram: tuple = (1,2)) -> List[Tuple[str, float]]:
        """
        Extract keywords using TF-IDF

        Args:
            texts: List of text documents
            top_n: Number of top keywords (uses self.top_n if None)
            ngram: N-gram range for TF-IDF vectorizer

        Returns:
            List of (keyword, tfidf_score) tuples
        """
        if top_n is None:
            top_n = self.top_n

        # Filter out empty texts
        texts = [text for text in texts if isinstance(text, str) and text.strip()]

        if not texts:
            logger.warning("No valid texts provided")
            return []

        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=ngram  # unigrams and bigrams
        )

        try:
            # Fit and transform
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)

            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()

            # Calculate average TF-IDF scores across all documents
            avg_tfidf = np.asarray(self.tfidf_matrix.mean(axis=0)).ravel()

            # Sort by score
            top_indices = avg_tfidf.argsort()[-top_n:][::-1]

            # Get top keywords with scores
            top_keywords = [(feature_names[i], avg_tfidf[i])
                           for i in top_indices]

            logger.info(f"Extracted {len(top_keywords)} keywords by TF-IDF")
            return top_keywords

        except Exception as e:
            logger.error(f"TF-IDF extraction failed: {e}")
            return []

    def extract_keywords_by_date(self, df: pd.DataFrame,
                                 text_column: str,
                                 date_column: str = 'date',
                                 method: str = 'tfidf') -> Dict[str, List[Tuple[str, float]]]:
        """
        Extract keywords for each date

        Args:
            df: DataFrame with articles
            text_column: Column containing text
            date_column: Column containing dates
            method: Extraction method ('frequency' or 'tfidf')

        Returns:
            Dictionary mapping dates to keyword lists
        """
        keywords_by_date = {}

        for date in df[date_column].unique():
            date_df = df[df[date_column] == date]
            texts = date_df[text_column].tolist()

            if method == 'frequency':
                keywords = self.extract_by_frequency(texts)
            elif method == 'tfidf':
                keywords = self.extract_by_tfidf(texts)
            else:
                raise ValueError(f"Unknown method: {method}")

            keywords_by_date[str(date)] = keywords

            logger.info(f"Extracted keywords for {date}: {len(keywords)} keywords")

        return keywords_by_date

    def get_keyword_frequency_over_time(self, df: pd.DataFrame,
                                       text_column: str,
                                       date_column: str = 'date',
                                       keywords: List[str] = None) -> pd.DataFrame:
        """
        Get frequency of specific keywords over time

        Args:
            df: DataFrame with articles
            text_column: Column containing text
            date_column: Column containing dates
            keywords: List of keywords to track (if None, extract automatically)

        Returns:
            DataFrame with keyword frequencies over time
        """
        if keywords is None:
            # Extract keywords from entire corpus
            all_texts = df[text_column].tolist()
            keyword_tuples = self.extract_by_tfidf(all_texts, top_n=self.top_n)
            keywords = [kw for kw, _ in keyword_tuples]

        # Create frequency matrix
        dates = sorted(df[date_column].unique())
        freq_data = []

        for date in dates:
            date_df = df[df[date_column] == date]
            date_texts = ' '.join(date_df[text_column].dropna().tolist())

            row = {'date': date}
            for keyword in keywords:
                # Count occurrences
                row[keyword] = date_texts.count(keyword)

            freq_data.append(row)

        freq_df = pd.DataFrame(freq_data)
        freq_df = freq_df.set_index('date')

        logger.info(f"Created frequency matrix: {freq_df.shape}")
        return freq_df

    def identify_emerging_keywords(self, freq_df: pd.DataFrame,
                                   threshold: float = 2.0) -> List[str]:
        """
        Identify keywords that are emerging (increasing frequency)

        Args:
            freq_df: DataFrame with keyword frequencies over time
            threshold: Multiplier threshold for emergence

        Returns:
            List of emerging keywords
        """
        emerging = []

        for keyword in freq_df.columns:
            values = freq_df[keyword].values
            if len(values) < 2:
                continue

            # Compare first half vs second half
            mid = len(values) // 2
            first_half_mean = values[:mid].mean()
            second_half_mean = values[mid:].mean()

            # Check if second half is significantly higher
            if first_half_mean > 0 and second_half_mean / first_half_mean >= threshold:
                emerging.append(keyword)

        logger.info(f"Identified {len(emerging)} emerging keywords")
        return emerging

    def identify_declining_keywords(self, freq_df: pd.DataFrame,
                                   threshold: float = 0.5) -> List[str]:
        """
        Identify keywords that are declining (decreasing frequency)

        Args:
            freq_df: DataFrame with keyword frequencies over time
            threshold: Multiplier threshold for decline

        Returns:
            List of declining keywords
        """
        declining = []

        for keyword in freq_df.columns:
            values = freq_df[keyword].values
            if len(values) < 2:
                continue

            # Compare first half vs second half
            mid = len(values) // 2
            first_half_mean = values[:mid].mean()
            second_half_mean = values[mid:].mean()

            # Check if second half is significantly lower
            if first_half_mean > 0 and second_half_mean / first_half_mean <= threshold:
                declining.append(keyword)

        logger.info(f"Identified {len(declining)} declining keywords")
        return declining

    def get_keyword_statistics(self, freq_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistics for each keyword

        Args:
            freq_df: DataFrame with keyword frequencies over time

        Returns:
            DataFrame with keyword statistics
        """
        stats = []

        for keyword in freq_df.columns:
            values = freq_df[keyword].values

            stat = {
                'keyword': keyword,
                'total_occurrences': values.sum(),
                'mean_frequency': values.mean(),
                'std_frequency': values.std(),
                'max_frequency': values.max(),
                'min_frequency': values.min(),
                'trend': 'increasing' if values[-1] > values[0] else 'decreasing'
            }
            stats.append(stat)

        stats_df = pd.DataFrame(stats)
        stats_df = stats_df.sort_values('total_occurrences', ascending=False)

        return stats_df
