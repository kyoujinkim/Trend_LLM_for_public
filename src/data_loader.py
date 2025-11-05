"""
Data Loader Module
Handles loading and parsing of CSV news data
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsDataLoader:
    """Load and parse news data from CSV files"""

    def __init__(self, csv_files: List[Path]):
        """
        Initialize the data loader

        Args:
            csv_files: List of paths to CSV files
        """
        self.csv_files = csv_files
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """
        Load all CSV files and combine into single DataFrame

        Returns:
            Combined DataFrame with all news articles
        """
        logger.info(f"Loading {len(self.csv_files)} CSV files...")

        dataframes = []
        for csv_file in tqdm(self.csv_files, desc="Loading CSV files"):
            if not csv_file.exists():
                logger.warning(f"File not found: {csv_file}")
                continue

            try:
                df = pd.read_csv(csv_file, index_col=0)
                logger.info(f"Loaded {len(df)} records from {csv_file.name}")
                dataframes.append(df)
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
                continue

        if not dataframes:
            raise ValueError("No data files could be loaded")

        # Combine all dataframes
        self.data = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Total records loaded: {len(self.data)}")

        # Convert date column to datetime
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'], format='%Y%m%d')

        return self.data

    def get_date_range(self) -> Dict[str, pd.Timestamp]:
        """
        Get the date range of loaded data

        Returns:
            Dictionary with 'start' and 'end' dates
        """
        if self.data is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")

        return {
            'start': self.data['date'].min(),
            'end': self.data['date'].max()
        }

    def get_articles_by_date(self, date: str) -> pd.DataFrame:
        """
        Get all articles for a specific date

        Args:
            date: Date string in format 'YYYY-MM-DD'

        Returns:
            DataFrame with articles from specified date
        """
        if self.data is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")

        target_date = pd.to_datetime(date)
        return self.data[self.data['date'] == target_date]

    def get_statistics(self) -> Dict:
        """
        Get basic statistics about the loaded data

        Returns:
            Dictionary with statistics
        """
        if self.data is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")

        stats = {
            'total_articles': len(self.data),
            'date_range': self.get_date_range(),
            'unique_dates': self.data['date'].nunique(),
            'unique_sources': self.data['source'].nunique() if 'source' in self.data.columns else 0,
            'articles_per_day': self.data.groupby('date').size().to_dict(),
            'columns': list(self.data.columns)
        }

        return stats

    def filter_by_ai_flag(self, ai_flag: str = 'Y') -> pd.DataFrame:
        """
        Filter articles by AI flag

        Args:
            ai_flag: AI flag value ('Y' or 'N')

        Returns:
            Filtered DataFrame
        """
        if self.data is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")

        if 'ai_flag' in self.data.columns:
            return self.data[self.data['ai_flag'].str.strip() == ai_flag]
        else:
            logger.warning("ai_flag column not found in data")
            return self.data
