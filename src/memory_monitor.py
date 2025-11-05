"""
Memory Monitor Module
Track and optimize memory usage during processing
"""
import psutil
import gc
import logging
from typing import Dict, Optional
from functools import wraps
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor and manage memory usage"""

    def __init__(self, threshold_mb: int = 1000):
        """
        Initialize memory monitor

        Args:
            threshold_mb: Memory threshold in MB to trigger warnings
        """
        self.threshold_mb = threshold_mb
        self.process = psutil.Process()
        self.measurements = []

    def get_current_memory_mb(self) -> float:
        """
        Get current memory usage in MB

        Returns:
            Memory usage in MB
        """
        return self.process.memory_info().rss / 1024 / 1024

    def get_memory_stats(self) -> Dict:
        """
        Get comprehensive memory statistics

        Returns:
            Dictionary with memory stats
        """
        mem_info = self.process.memory_info()
        virtual_mem = psutil.virtual_memory()

        stats = {
            'current_mb': mem_info.rss / 1024 / 1024,
            'peak_mb': mem_info.vms / 1024 / 1024,
            'available_mb': virtual_mem.available / 1024 / 1024,
            'total_mb': virtual_mem.total / 1024 / 1024,
            'percent_used': virtual_mem.percent
        }

        return stats

    def check_memory(self, operation: str = ""):
        """
        Check current memory usage and log if above threshold

        Args:
            operation: Description of current operation
        """
        current_mb = self.get_current_memory_mb()
        self.measurements.append(current_mb)

        if current_mb > self.threshold_mb:
            logger.warning(
                f"Memory usage high: {current_mb:.1f} MB "
                f"(threshold: {self.threshold_mb} MB) - {operation}"
            )

        return current_mb

    def log_memory_stats(self, prefix: str = ""):
        """
        Log comprehensive memory statistics

        Args:
            prefix: Prefix for log message
        """
        stats = self.get_memory_stats()

        logger.info(
            f"{prefix}Memory: {stats['current_mb']:.1f} MB "
            f"(Peak: {stats['peak_mb']:.1f} MB, "
            f"Available: {stats['available_mb']:.1f} MB, "
            f"System: {stats['percent_used']:.1f}%)"
        )

    def force_garbage_collection(self):
        """Force garbage collection and log memory freed"""
        before_mb = self.get_current_memory_mb()

        gc.collect()

        after_mb = self.get_current_memory_mb()
        freed_mb = before_mb - after_mb

        if freed_mb > 10:  # Only log if significant memory freed
            logger.info(f"Garbage collection freed {freed_mb:.1f} MB")

        return freed_mb

    def get_summary(self) -> Dict:
        """
        Get summary of memory usage during session

        Returns:
            Summary statistics
        """
        if not self.measurements:
            return {}

        return {
            'min_mb': min(self.measurements),
            'max_mb': max(self.measurements),
            'avg_mb': sum(self.measurements) / len(self.measurements),
            'measurements': len(self.measurements)
        }


def memory_tracked(monitor: Optional[MemoryMonitor] = None):
    """
    Decorator to track memory usage of a function

    Args:
        monitor: MemoryMonitor instance (creates new one if None)
    """
    if monitor is None:
        monitor = MemoryMonitor()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__

            # Log before
            monitor.log_memory_stats(f"[Before {func_name}] ")
            start_time = time.time()

            # Execute function
            result = func(*args, **kwargs)

            # Log after
            elapsed = time.time() - start_time
            monitor.log_memory_stats(f"[After {func_name}] ")
            logger.info(f"{func_name} completed in {elapsed:.2f}s")

            # Garbage collect
            monitor.force_garbage_collection()

            return result

        return wrapper

    return decorator


class MemoryOptimizer:
    """Optimize memory usage with various strategies"""

    @staticmethod
    def optimize_dataframe(df):
        """
        Optimize DataFrame memory usage

        Args:
            df: DataFrame to optimize

        Returns:
            Optimized DataFrame
        """
        initial_mem = df.memory_usage(deep=True).sum() / 1024 / 1024

        # Optimize integer columns
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype('int32')

        # Optimize float columns
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')

        # Optimize object columns to category if beneficial
        for col in df.select_dtypes(include=['object']).columns:
            num_unique = df[col].nunique()
            num_total = len(df[col])

            # If less than 50% unique values, convert to category
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')

        final_mem = df.memory_usage(deep=True).sum() / 1024 / 1024
        reduction = (initial_mem - final_mem) / initial_mem * 100

        logger.info(
            f"DataFrame memory optimized: {initial_mem:.1f} MB -> {final_mem:.1f} MB "
            f"({reduction:.1f}% reduction)"
        )

        return df

    @staticmethod
    def suggest_chunk_size(available_mb: float, row_size_kb: float) -> int:
        """
        Suggest optimal chunk size based on available memory

        Args:
            available_mb: Available memory in MB
            row_size_kb: Estimated row size in KB

        Returns:
            Suggested chunk size
        """
        # Use 20% of available memory for safety
        usable_mb = available_mb * 0.2

        # Convert to KB
        usable_kb = usable_mb * 1024

        # Calculate number of rows
        chunk_size = int(usable_kb / row_size_kb)

        # Set reasonable bounds
        chunk_size = max(100, min(chunk_size, 50000))

        logger.info(f"Suggested chunk size: {chunk_size} rows")

        return chunk_size

    @staticmethod
    def estimate_memory_needed(file_size_mb: float, multiplier: float = 2.5) -> float:
        """
        Estimate memory needed to process a file

        Args:
            file_size_mb: File size in MB
            multiplier: Memory multiplication factor

        Returns:
            Estimated memory in MB
        """
        return file_size_mb * multiplier


def check_memory_sufficient(required_mb: float) -> bool:
    """
    Check if sufficient memory is available

    Args:
        required_mb: Required memory in MB

    Returns:
        True if sufficient memory available
    """
    virtual_mem = psutil.virtual_memory()
    available_mb = virtual_mem.available / 1024 / 1024

    sufficient = available_mb > required_mb

    if not sufficient:
        logger.warning(
            f"Insufficient memory: Need {required_mb:.1f} MB, "
            f"Available: {available_mb:.1f} MB"
        )

    return sufficient


def auto_configure_chunking(file_paths, default_chunk_size: int = 1000) -> int:
    """
    Automatically configure chunk size based on file sizes and available memory

    Args:
        file_paths: List of file paths
        default_chunk_size: Default chunk size

    Returns:
        Recommended chunk size
    """
    # Calculate total file size
    total_size_mb = sum(
        path.stat().st_size / 1024 / 1024
        for path in file_paths
        if path.exists()
    )

    # Get available memory
    virtual_mem = psutil.virtual_memory()
    available_mb = virtual_mem.available / 1024 / 1024

    # Estimate memory needed
    estimated_needed = total_size_mb * 2.5

    if estimated_needed > available_mb * 0.5:
        # Need chunking
        logger.warning(
            f"Large dataset detected ({total_size_mb:.1f} MB). "
            f"Using chunked processing for memory efficiency."
        )

        # Calculate optimal chunk size
        # Assume average row size of 50 KB (adjust based on your data)
        avg_row_size_kb = 50
        chunk_size = MemoryOptimizer.suggest_chunk_size(available_mb, avg_row_size_kb)

        return chunk_size
    else:
        # Can load in memory
        logger.info(
            f"Dataset size ({total_size_mb:.1f} MB) is manageable. "
            f"Available memory: {available_mb:.1f} MB"
        )
        return None  # No chunking needed
