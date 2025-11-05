"""
Memory Usage Comparison Script
Demonstrates the difference between full load and chunked processing
"""
import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.data_loader import NewsDataLoader
from src.memory_efficient_loader import MemoryEfficientLoader
from src.memory_monitor import MemoryMonitor


def test_full_load():
    """Test memory usage with full data load"""
    print("\n" + "=" * 80)
    print("TEST 1: Full Data Load (Original Method)")
    print("=" * 80)

    monitor = MemoryMonitor()
    monitor.log_memory_stats("[Start] ")

    start_time = time.time()

    # Load all data at once
    loader = NewsDataLoader(config.CSV_FILES)
    df = loader.load_data()

    end_time = time.time()

    monitor.log_memory_stats("[After Load] ")

    print(f"\n✓ Loaded {len(df):,} rows in {end_time - start_time:.2f}s")
    print(f"✓ DataFrame memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

    stats = monitor.get_memory_stats()
    print(f"\n📊 Memory Stats:")
    print(f"   Current: {stats['current_mb']:.1f} MB")
    print(f"   Peak: {stats['peak_mb']:.1f} MB")

    # Cleanup
    del df
    monitor.force_garbage_collection()

    return stats


def test_chunked_load():
    """Test memory usage with chunked processing"""
    print("\n" + "=" * 80)
    print("TEST 2: Chunked Processing (Memory-Efficient Method)")
    print("=" * 80)

    monitor = MemoryMonitor()
    monitor.log_memory_stats("[Start] ")

    start_time = time.time()

    # Process in chunks
    loader = MemoryEfficientLoader(config.CSV_FILES, chunk_size=1000)

    total_rows = 0
    for chunk in loader.iter_chunks():
        total_rows += len(chunk)
        # Simulate processing without accumulating
        _ = chunk['title'].str.len().sum()  # Dummy operation

    end_time = time.time()

    monitor.log_memory_stats("[After Processing] ")

    print(f"\n✓ Processed {total_rows:,} rows in {end_time - start_time:.2f}s")

    stats = monitor.get_memory_stats()
    print(f"\n📊 Memory Stats:")
    print(f"   Current: {stats['current_mb']:.1f} MB")
    print(f"   Peak: {stats['peak_mb']:.1f} MB")

    monitor.force_garbage_collection()

    return stats


def test_incremental_stats():
    """Test incremental statistics calculation"""
    print("\n" + "=" * 80)
    print("TEST 3: Incremental Statistics (No Full Load)")
    print("=" * 80)

    monitor = MemoryMonitor()
    monitor.log_memory_stats("[Start] ")

    start_time = time.time()

    # Get statistics without loading all data
    loader = MemoryEfficientLoader(config.CSV_FILES, chunk_size=1000)
    stats = loader.get_statistics_incremental()

    end_time = time.time()

    monitor.log_memory_stats("[After Stats] ")

    print(f"\n✓ Calculated statistics in {end_time - start_time:.2f}s")
    print(f"\n📋 Dataset Stats:")
    print(f"   Total articles: {stats['total_articles']:,}")
    print(f"   Unique dates: {stats['unique_dates']}")
    print(f"   Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")

    mem_stats = monitor.get_memory_stats()
    print(f"\n📊 Memory Stats:")
    print(f"   Current: {mem_stats['current_mb']:.1f} MB")
    print(f"   Peak: {mem_stats['peak_mb']:.1f} MB")

    return mem_stats


def main():
    """Run all tests and compare"""
    print("=" * 80)
    print("FACTOR-LLM Memory Usage Comparison")
    print("=" * 80)
    print("\nThis script compares memory usage between:")
    print("1. Full data load (original method)")
    print("2. Chunked processing (memory-efficient method)")
    print("3. Incremental statistics (no full load)")

    try:
        # Test 1: Full load
        full_stats = test_full_load()

        # Test 2: Chunked
        chunked_stats = test_chunked_load()

        # Test 3: Incremental
        incremental_stats = test_incremental_stats()

        # Comparison
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)

        print(f"\n{'Method':<20} {'Peak Memory (MB)':<20} {'Savings':<20}")
        print("-" * 60)

        full_peak = full_stats['peak_mb']
        chunked_peak = chunked_stats['peak_mb']
        incremental_peak = incremental_stats['peak_mb']

        print(f"{'Full Load':<20} {full_peak:<20.1f} {'Baseline':<20}")
        chunked_savings = (full_peak - chunked_peak) / full_peak * 100
        print(f"{'Chunked':<20} {chunked_peak:<20.1f} {f'{chunked_savings:.1f}% less':<20}")
        incremental_savings = (full_peak - incremental_peak) / full_peak * 100
        print(f"{'Incremental':<20} {incremental_peak:<20.1f} {f'{incremental_savings:.1f}% less':<20}")

        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        print("\n✅ Use Full Load when:")
        print("   • Dataset < 1 GB")
        print("   • Available RAM > 16 GB")
        print("   • Speed is critical")

        print("\n✅ Use Chunked Processing when:")
        print("   • Dataset > 1 GB")
        print("   • Limited RAM (< 8 GB)")
        print("   • Processing large datasets (3+ years)")
        print("   • Memory efficiency is priority")

        print("\n✅ Use Incremental Statistics when:")
        print("   • Only need summary statistics")
        print("   • Don't need full data processing")
        print("   • Very limited memory")

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\n✗ Error during comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
