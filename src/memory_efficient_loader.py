"""
Memory Efficient Data Loader Module
Handles loading and processing large CSV files with chunking and streaming
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Iterator, Optional
from tqdm import tqdm
import logging
import gc
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryEfficientLoader:
    """Memory-efficient data loader using chunking and streaming"""

    def __init__(self, csv_files: List[str], data_dir:str):
        """
        Initialize the memory-efficient loader

        Args:
            date_list: List of dates for CSV files
        """
        self.csv_files = csv_files
        self.data_dir = data_dir
        self.total_rows = 0

    def iter_chunks(self, columns:list=None) -> Iterator[pd.DataFrame]:
        """
        Iterate through CSV files in chunks

        Args:
            columns: Specific columns to load (None for all)

        Yields:
            DataFrame chunks
        """
        for f in self.csv_files:
            logger.info(f"Processing date: {f}")

            try:
                csv_file = Path(f"{self.data_dir}/{f}.csv")
                if not csv_file.exists():
                    logger.warning(f"File not found: {csv_file}")
                    continue

                if columns:
                    yield pd.read_csv(csv_file, usecols=columns)
                else:
                    yield pd.read_csv(csv_file)

                gc.collect()

            except Exception as e:
                logger.error(f"Error reading {f}: {e}")
                continue

    def load_by_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load only data within a specific date range

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Filtered DataFrame
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        logger.info(f"Loading data from {start_date} to {end_date}")

        filtered_chunks = []

        for chunk in self.iter_chunks():
            # Filter by date
            mask = (chunk['date'] >= start) & (chunk['date'] <= end)
            filtered = chunk[mask]

            if len(filtered) > 0:
                filtered_chunks.append(filtered)

            del chunk, filtered
            gc.collect()

        if filtered_chunks:
            result = pd.concat(filtered_chunks, ignore_index=True)
            logger.info(f"Loaded {len(result)} rows in date range")
            return result
        else:
            logger.warning("No data found in date range")
            return pd.DataFrame()

    def get_statistics_incremental(self) -> Dict:
        """
        Calculate statistics incrementally without loading all data

        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_articles': 0,
            'unique_dates': set(),
            'unique_sources': set(),
            'articles_per_day': {},
            'date_min': None,
            'date_max': None
        }

        for chunk in self.iter_chunks():
            stats['total_articles'] += len(chunk)

            if 'date' in chunk.columns:
                dates = chunk['date'].unique()
                stats['unique_dates'].update(dates)

                # Update min/max dates
                chunk_min = chunk['date'].min()
                chunk_max = chunk['date'].max()

                if stats['date_min'] is None or chunk_min < stats['date_min']:
                    stats['date_min'] = chunk_min
                if stats['date_max'] is None or chunk_max > stats['date_max']:
                    stats['date_max'] = chunk_max

                # Count articles per day
                for date, count in chunk['date'].value_counts().items():
                    stats['articles_per_day'][date] = stats['articles_per_day'].get(date, 0) + count

            if 'source' in chunk.columns:
                stats['unique_sources'].update(chunk['source'].unique())

            del chunk
            gc.collect()

        # Convert sets to counts
        stats['unique_dates'] = len(stats['unique_dates'])
        stats['unique_sources'] = len(stats['unique_sources'])
        stats['date_range'] = {
            'start': stats.pop('date_min'),
            'end': stats.pop('date_max')
        }

        return stats

    def create_date_index(self) -> Dict[str, List[int]]:
        """
        Create an index of row positions by date for efficient lookup

        Returns:
            Dictionary mapping dates to row indices
        """
        date_index = {}
        current_position = 0

        for chunk in self.iter_chunks(columns=['date']):
            for idx, date in enumerate(chunk['date']):
                date_str = str(date.date())
                if date_str not in date_index:
                    date_index[date_str] = []
                date_index[date_str].append(current_position + idx)

            current_position += len(chunk)
            del chunk
            gc.collect()

        logger.info(f"Created date index with {len(date_index)} unique dates")
        return date_index


class ChunkedDataProcessor:
    """Process data in chunks with various strategies"""

    def __init__(self, loader: MemoryEfficientLoader):
        """
        Initialize processor

        Args:
            loader: MemoryEfficientLoader instance
        """
        self.loader = loader

    def aggregate_by_date(self, text_column: str = 'content') -> pd.DataFrame:
        """
        Aggregate text by date to reduce memory usage

        Args:
            text_column: Column to aggregate

        Returns:
            DataFrame with aggregated text per date
        """
        logger.info("Aggregating text by date...")

        date_texts = {}

        for chunk in self.loader.iter_chunks(columns=['date', text_column]):
            for date, group in chunk.groupby('date'):
                date_str = str(date.date())

                if date_str not in date_texts:
                    date_texts[date_str] = []

                # Aggregate texts
                texts = group[text_column].dropna().tolist()
                date_texts[date_str].extend(texts)

            del chunk
            gc.collect()

        # Convert to DataFrame
        result = pd.DataFrame([
            {'date': pd.to_datetime(date), 'aggregated_text': ' '.join(texts)}
            for date, texts in date_texts.items()
        ])

        result = result.sort_values('date')
        logger.info(f"Aggregated to {len(result)} date groups")

        return result

    def extract_columns(self, columns: List[str], output_file: Path):
        """
        Extract specific columns to a new CSV file

        Args:
            columns: Columns to extract
            output_file: Output file path
        """
        logger.info(f"Extracting columns {columns} to {output_file}")

        first_chunk = True

        for chunk in self.loader.iter_chunks(columns=columns):
            chunk.to_csv(
                output_file,
                mode='w' if first_chunk else 'a',
                header=first_chunk,
                index=False
            )
            first_chunk = False

            del chunk
            gc.collect()

        logger.info(f"Extraction complete: {output_file}")

    def compute_rolling_statistics(self, column: str,
                                   window: int = 7) -> pd.DataFrame:
        """
        Compute rolling statistics over chunks

        Args:
            column: Column to analyze
            window: Rolling window size

        Returns:
            DataFrame with rolling statistics
        """
        # This requires more sophisticated state management
        # For now, load necessary columns only
        data = []

        for chunk in self.loader.iter_chunks(columns=['date', column]):
            data.append(chunk)

        if not data:
            return pd.DataFrame()

        df = pd.concat(data, ignore_index=True)
        df = df.sort_values('date')

        # Compute rolling statistics
        df[f'{column}_rolling_mean'] = df.groupby('date')[column].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

        # Clean up
        del data
        gc.collect()

        return df
