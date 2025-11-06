"""
Memory-Efficient FACTOR-LLM Main Application
Uses chunking and streaming for large datasets
"""
import logging
from pathlib import Path
import argparse
from tqdm import tqdm
import gc
import pandas as pd
from collections import defaultdict

# Import configuration
import config

# Import modules
from src.memory_efficient_loader import MemoryEfficientLoader, ChunkedDataProcessor
from src.memory_monitor import MemoryMonitor, memory_tracked, auto_configure_chunking, MemoryOptimizer
from src.text_preprocessor import TextPreprocessor
from src.keyword_extractor import KeywordExtractor
from src.time_series_analyzer import TimeSeriesAnalyzer
from src.prediction_model import KeywordLifecyclePredictor
from src.llm_analyzer import LLMAnalyzer
from src.report_generator import ReportGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MemoryEfficientFactorLLM:
    """Memory-efficient version of FACTOR-LLM for large datasets"""

    def __init__(self, use_llm: bool = False, chunk_size: int = None, huggingface_model: str = 'LiquidAI/LFM2-350M-Extract', mecab_path: str = 'C:/mecab/mecab-ko-dic'):
        """
        Initialize Memory-Efficient FACTOR-LLM

        Args:
            use_llm: Whether to use LLM for interpretation
            chunk_size: Chunk size for processing (auto-configured if None)
        """
        self.use_llm = use_llm

        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(threshold_mb=config.MEMORY_THRESHOLD_MB)
        self.memory_monitor.log_memory_stats("[Init] ")

        # Auto-configure chunk size if needed
        #if chunk_size is None and config.AUTO_CONFIGURE_MEMORY:
        #    chunk_size = auto_configure_chunking(config.CSV_FILES, config.CHUNK_SIZE)

        self.chunk_size = chunk_size or config.CHUNK_SIZE

        logger.info(f"Using chunk size: {self.chunk_size} rows")

        # Initialize components
        logger.info("Initializing FACTOR-LLM components...")

        self.loader = MemoryEfficientLoader(config.CSVS, config.DATA_DIR)
        self.processor = ChunkedDataProcessor(self.loader)
        self.preprocessor = TextPreprocessor(huggingface_model=huggingface_model, mecab_path=mecab_path)

        self.keyword_extractor = KeywordExtractor(
            top_n=config.TOP_N_KEYWORDS,
            min_df=2,
            max_df=0.8
        )

        self.time_series_analyzer = TimeSeriesAnalyzer(
            window_size=config.WINDOW_SIZE
        )

        self.predictor = KeywordLifecyclePredictor(
            prediction_horizon_days=config.PREDICTION_HORIZON_DAYS
        )

        if self.use_llm:
            self.llm_analyzer = LLMAnalyzer(
                provider=config.LLM_PROVIDER,
                model=config.LLM_MODEL,
                api_key=config.ANTHROPIC_API_KEY or config.OPENAI_API_KEY
            )
        else:
            self.llm_analyzer = None
            logger.info("LLM analysis disabled")

        self.report_generator = ReportGenerator(config.REPORT_DIR)

        self.memory_monitor.log_memory_stats("[Init Complete] ")
        logger.info("Initialization complete")

    @memory_tracked()
    def run_only_keywords(self):
        """Run only keyword extraction in a memory-efficient way"""
        logger.info("=" * 80)
        logger.info("Memory-Efficient FACTOR-LLM Keyword Extraction Started")
        logger.info(f"Processing Mode: {config.PROCESSING_MODE}")
        logger.info("=" * 80)

        # Step 1: Preprocess text in chunks
        logger.info("\n[1/3] Preprocessing text (chunked)...")
        preprocessed_texts_by_date = self._preprocess_in_chunks()
        self.memory_monitor.check_memory("After preprocessing")

        # Step 2: Extract keywords incrementally
        logger.info("\n[2/3] Extracting keywords (incremental)...")
        keyword_freq_by_date = self._extract_keywords_incremental(preprocessed_texts_by_date, save_separately=True)
        self.memory_monitor.check_memory("After keyword extraction")

        logger.info("\n" + "=" * 80)
        logger.info("Memory-Efficient FACTOR-LLM Keyword Extraction Complete!")
        logger.info("=" * 80)
        logger.info(f"\nOutputs saved to: {config.OUTPUT_DIR}")

        return True

    @memory_tracked()
    def run_only_anlaysis(self, target_date:str):
        """Run only analysis on pre-extracted keyword frequencies"""
        logger.info("=" * 80)
        logger.info("Memory-Efficient FACTOR-LLM Analysis from Pre-Extracted Data Started")
        logger.info(f"Processing Mode: {config.PROCESSING_MODE}")
        logger.info("=" * 80)

        # Step 1: Load keyword frequency data
        logger.info("\n[1/5] Loading keyword frequency data...")
        keyword_freq_by_date = self._load_keyword_freq_by_date(target_date=target_date)
        self.memory_monitor.check_memory("After loading keyword frequencies")

        # Step 2: Build frequency DataFrame
        logger.info("\n[2/5] Building frequency matrix...")
        freq_df = self._build_frequency_dataframe(keyword_freq_by_date)

        if config.OPTIMIZE_DATAFRAMES:
            freq_df = MemoryOptimizer.optimize_dataframe(freq_df)

        self.memory_monitor.check_memory("After frequency matrix")

        # Step 3: Time series analysis
        logger.info("\n[3/5] Analyzing time series...")
        analysis_df = self.time_series_analyzer.analyze_all_keywords(freq_df)

        #if config.OPTIMIZE_DATAFRAMES:
        #    analysis_df = MemoryOptimizer.optimize_dataframe(analysis_df)

        self.memory_monitor.check_memory("After time series analysis")

        # Step 4: Generate predictions
        logger.info("\n[4/5] Generating predictions...")
        predictions_df = self.predictor.predict_all_keywords(freq_df, analysis_df)

        keywords = self.predictor.get_keywords(predictions_df)
        logger.info(f"Keywords: {len(keywords)}")\

        self.memory_monitor.check_memory("After predictions")

        # Step 7: Generate interpretations (process top keywords only to save memory)
        logger.info("\n[7/7] Generating interpretations...")

        interpretations = self._generate_interpretations_selective(
            keywords.nlargest(30, 'current_frequency'),
            analysis_df,
            target_date
        )

        self.memory_monitor.check_memory("After interpretations")

        # Step 8: Generate comprehensive report
        logger.info("\n[8/8] Generating comprehensive report...")
        report_data = self._prepare_report_data(
            freq_df=freq_df,
            analysis_df=analysis_df,
            predictions_df=predictions_df,
            interpretations=interpretations
        )

        report_path = self.report_generator.generate_markdown_report(report_data, target_date)
        logger.info(f"Report generated: {report_path}")

        # Memory summary
        logger.info("\n" + "=" * 80)
        logger.info("Memory Usage Summary")
        logger.info("=" * 80)
        memory_summary = self.memory_monitor.get_summary()
        logger.info(f"Peak memory: {memory_summary['max_mb']:.1f} MB")
        logger.info(f"Average memory: {memory_summary['avg_mb']:.1f} MB")

        logger.info("\n" + "=" * 80)
        logger.info("Memory-Efficient FACTOR-LLM Analysis Complete!")
        logger.info("=" * 80)
        logger.info(f"\nOutputs saved to: {config.OUTPUT_DIR}")

        return report_path

    def _preprocess_in_chunks(self):
        """Preprocess text data in chunks"""
        preprocessed_by_date = defaultdict(list)

        pbar = tqdm(self.loader.iter_chunks(), desc="Total rows processed")
        for chunk in pbar: # Preprocess chunk
            total_row = len(chunk)
            for _, row in chunk.iterrows():
                pbar.set_postfix(current = _, total = total_row)
                date = row['date']

                #combined = f"{row.get('title', '')} {row.get('content', '')}"
                #combined_keyword = self.preprocessor.keywordify_text(combined)
                combined_keyword = combined = row.get('title', '')

                # Clean title
                text_cleaned = self.preprocessor.preprocess_text(
                    combined_keyword,
                    extract_nouns_only=True
                )

                if combined.strip():
                    preprocessed_by_date[str(date)].append(text_cleaned)

            # Garbage collection
            del chunk
            if config.AGGRESSIVE_GC:
                gc.collect()

        logger.info(f"Preprocessed data for {len(preprocessed_by_date)} dates")
        return preprocessed_by_date

    def _extract_keywords_incremental(self, texts_by_date, save_separately=False):
        """Extract keywords incrementally by date"""
        keyword_freq_by_date = {}

        # First pass: collect all keywords across all dates
        all_keywords = set()

        logger.info("Collecting vocabulary...")
        for date, texts in tqdm(texts_by_date.items(), desc="Vocabulary"):
            keywords = self.keyword_extractor.extract_by_tfidf(texts, top_n=config.TOP_N_KEYWORDS, ngram=(2,3))
            date_keywords = set(kw for kw, _ in keywords)
            all_keywords.update(date_keywords)

        all_keywords = sorted(all_keywords)  # Limit total keywords
        logger.info(f"Total unique keywords: {len(all_keywords)}")

        # Second pass: count frequency of selected keywords per date
        logger.info("Counting keyword frequencies...")
        for date, texts in tqdm(texts_by_date.items(), desc="Frequency"):
            combined_text = ' '.join(texts)

            freq_dict = {}
            for keyword in all_keywords:
                freq_dict[keyword] = combined_text.count(keyword)

            if save_separately:
                pd.Series(freq_dict, name='Keyword').to_csv(f'{config.OUTPUT_DIR}/keyword_freq_{date}.csv', encoding='utf-8-sig')
            else:
                keyword_freq_by_date[date] = freq_dict

            if config.AGGRESSIVE_GC:
                gc.collect()

        return keyword_freq_by_date

    def _load_keyword_freq_by_date(self, target_date:str, date_range:int=52):
        """Load keyword frequency data from saved CSVs"""
        keyword_freq_by_date = {}
        target_date = pd.to_datetime(target_date)
        start_date = target_date - pd.Timedelta(date_range, unit='W')
        date_arr = pd.date_range(start=start_date, end=target_date)

        for curr_date in tqdm(date_arr, desc="Loading keyword frequencies"):
            date_str = curr_date.strftime('%Y%m%d')
            try:
                freq_series = pd.read_csv(f'{config.OUTPUT_DIR}/keyword_freq_{date_str}.csv', encoding='UTF-8-sig', index_col=0)['Keyword']
                freq_dict = freq_series.to_dict()
                keyword_freq_by_date[date_str] = freq_dict
            except:
                pass

            if config.AGGRESSIVE_GC:
                gc.collect()

        return keyword_freq_by_date


    def _build_frequency_dataframe(self, keyword_freq_by_date):
        """Build frequency DataFrame from dictionary"""
        rows = []

        for date, freq_dict in keyword_freq_by_date.items():
            row = {'date': pd.to_datetime(date)}
            row.update(freq_dict)
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.set_index('date')
        df = df.sort_index()

        # drop columns with all 0 values
        df = df.loc[:, (df != 0).any(axis=0)]

        logger.info(f"Frequency matrix shape: {df.shape}")
        return df

    def _generate_interpretations_selective(self, keywords_df, analysis_df, target_date):
        """Generate interpretations for selected keywords only"""
        interpretations = {}

        logger.info("Generating LLM interpretations for top keywords...")

        # LLM-based interpretations
        # To minimize API calls, bind all data in one go
        # combine emerging and declining
        input_dict = {}
        for _, pred_row in tqdm(keywords_df.iterrows(),
                               total=len(keywords_df),
                               desc="LLM interpretations"):
            keyword = pred_row['keyword']

            # Get analysis
            analysis = analysis_df[analysis_df['keyword'] == keyword]
            if len(analysis) == 0:
                continue

            analysis_dict = analysis.iloc[0].to_dict()

            input_dict[keyword] = {
                'analysis': analysis_dict,
                'prediction': pred_row.to_dict()
            }

        # Generate interpretation
        interpretation = self.llm_analyzer.generate_keyword_interpretation(input_dict)
        news = self.llm_analyzer.fetch_relevant_news(list(input_dict.keys()), target_date)

        if config.AGGRESSIVE_GC:
            gc.collect()

        overall = self.llm_analyzer.generate_overall_analysis(
                interpretation, news
        )

        interpretations = {
        'overall': overall,
        'interpretations': interpretation,
        'news': news
        }

        return interpretations

    def _prepare_report_data(self, freq_df, analysis_df, predictions_df, interpretations):
        """
        Prepare comprehensive data for report generation

        Args:
            freq_df: Frequency DataFrame with date index and keyword columns
            analysis_df: Analysis results DataFrame
            predictions_df: Predictions DataFrame
            interpretations: Dictionary with 'overall', 'interpretations', 'news'

        Returns:
            Dictionary with comprehensive data for markdown report generation
        """
        logger.info("Preparing comprehensive report data...")

        # 1. STATISTICS - General information about the dataset
        statistics = {
            'total_articles': int(freq_df.sum().sum()) if len(freq_df) > 0 else 0,
            'date_range': {
                'start': freq_df.index.min().strftime('%Y-%m-%d') if len(freq_df) > 0 else 'N/A',
                'end': freq_df.index.max().strftime('%Y-%m-%d') if len(freq_df) > 0 else 'N/A'
            },
            'unique_dates': len(freq_df.index.unique()) if len(freq_df) > 0 else 0,
            'unique_sources': 1  # Update if you track sources
        }

        # 2. TOP KEYWORDS - Merge analysis and predictions
        # get dominant_keywords from interpretations
        try:
            dominant_keywords = interpretations.get('interpretations', {}).dominant_keyword_list
        except:
            dominant_keywords = []

        # 3. EMERGING KEYWORDS
        try:
            rising_keywords = interpretations.get('interpretations', {}).rising_keyword_list
        except:
            rising_keywords = []

        # 4. DECLINING KEYWORDS - Extract from predictions_df
        try:
            declining_keywords = interpretations.get('interpretations', {}).declining_keyword_list
        except:
            declining_keywords = []

        # 5. Combine all keywords(dominant_keywords, rising_keywords, declining_keywords) and data(analysis_df, predictions_df)
        key_list = []
        for kw in declining_keywords:
            kw_word = kw.keyword
            kw_interpret = kw.interpretation
            key_list.append(['declining', kw_word, kw_interpret])
        for kw in rising_keywords:
            kw_word = kw.keyword
            kw_interpret = kw.interpretation
            key_list.append(['rising', kw_word, kw_interpret])
        for kw in dominant_keywords:
            kw_word = kw.keyword
            kw_interpret = kw.interpretation
            key_list.append(['dominant', kw_word, kw_interpret])
        key_df = pd.DataFrame(key_list, columns=['status', 'keyword', 'interpretation'])
        merged_data = pd.merge(key_df, analysis_df, on='keyword', how='left')
        merged_data = pd.merge(merged_data, predictions_df, on='keyword', how='left')

        # 6. OVERALL ANALYSIS - Industry trends
        overall_analysis = interpretations.get('overall', '')

        # Assemble final report data
        report_data = {
            'statistics': statistics,
            'keywords_data': merged_data,
            'overall_analysis': overall_analysis
        }

        logger.info(f"Report data prepared: {len(dominant_keywords)} top keywords, "
                   f"{len(rising_keywords)} emerging, {len(declining_keywords)} declining")

        return report_data


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Memory-Efficient FACTOR-LLM for Large Datasets'
    )
    parser.add_argument(
        '--use-llm',
        action='store_true',
        default=True,
        help='Enable LLM-based interpretation (requires API key)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='API key for LLM (Anthropic or OpenAI)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=None,
        help='Chunk size for processing (auto-configured if not specified)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'chunked', 'sample'],
        default='chunked',
        help='Processing mode'
    )

    args = parser.parse_args()

    # Set API key if provided
    if args.api_key:
        if config.LLM_PROVIDER == 'anthropic':
            config.ANTHROPIC_API_KEY = args.api_key
        else:
            config.OPENAI_API_KEY = args.api_key

    # Set processing mode
    config.PROCESSING_MODE = args.mode

    # Initialize and run
    app = MemoryEfficientFactorLLM(use_llm=args.use_llm, chunk_size=args.chunk_size, huggingface_model=None)

    for target_date in ['20241130', '20250131', '20250331', '20250531', '20250731', '20250930', '20251104']:
        logger.info(f"\nRunning analysis for target date: {target_date}")

        try:
            report_path = app.run_only_anlaysis(target_date)
            print(f"\n✓ Analysis complete! Report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Error during analysis: {e}", exc_info=True)
            print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    exit(main())
