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

    def __init__(self, use_llm: bool = False, chunk_size: int = None, mecab_path: str = None):
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

        self.loader = MemoryEfficientLoader(config.CSVS)
        self.processor = ChunkedDataProcessor(self.loader)
        self.preprocessor = TextPreprocessor(mecab_path)

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

        self.report_generator = ReportGenerator(config.OUTPUT_DIR)

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
    def run(self):
        """Execute the memory-efficient FACTOR-LLM pipeline"""
        logger.info("=" * 80)
        logger.info("Memory-Efficient FACTOR-LLM Analysis Pipeline Started")
        logger.info(f"Processing Mode: {config.PROCESSING_MODE}")
        logger.info("=" * 80)

        # Step 1: Get statistics without loading all data
        logger.info("\n[1/7] Analyzing data statistics...")
        stats = self.loader.get_statistics_incremental()
        logger.info(f"Total articles: {stats['total_articles']:,}")
        self.memory_monitor.check_memory("After statistics")

        # Step 2: Preprocess text in chunks
        logger.info("\n[2/7] Preprocessing text (chunked)...")
        preprocessed_texts_by_date = self._preprocess_in_chunks()
        self.memory_monitor.check_memory("After preprocessing")

        # Step 3: Extract keywords incrementally
        logger.info("\n[3/7] Extracting keywords (incremental)...")
        keyword_freq_by_date = self._extract_keywords_incremental(preprocessed_texts_by_date)
        self.memory_monitor.check_memory("After keyword extraction")

        # Step 4: Build frequency DataFrame
        logger.info("\n[4/7] Building frequency matrix...")
        freq_df = self._build_frequency_dataframe(keyword_freq_by_date)

        if config.OPTIMIZE_DATAFRAMES:
            freq_df = MemoryOptimizer.optimize_dataframe(freq_df)

        self.memory_monitor.check_memory("After frequency matrix")

        # Step 5: Time series analysis
        logger.info("\n[5/7] Analyzing time series...")
        analysis_df = self.time_series_analyzer.analyze_all_keywords(freq_df)

        if config.OPTIMIZE_DATAFRAMES:
            analysis_df = MemoryOptimizer.optimize_dataframe(analysis_df)

        self.memory_monitor.check_memory("After time series analysis")

        # Step 6: Generate predictions
        logger.info("\n[6/7] Generating predictions...")
        predictions_df = self.predictor.predict_all_keywords(freq_df, analysis_df)

        high_risk_keywords = self.predictor.get_high_risk_keywords(predictions_df, threshold=0.6)
        emerging_keywords = self.predictor.get_emerging_keywords(predictions_df)

        logger.info(f"High risk keywords: {len(high_risk_keywords)}")
        logger.info(f"Emerging keywords: {len(emerging_keywords)}")

        self.memory_monitor.check_memory("After predictions")

        # Step 7: Generate interpretations (process top keywords only to save memory)
        logger.info("\n[7/7] Generating interpretations...")
        top_n_for_interpretation = 20  # Limit to reduce memory and API calls

        interpretations = self._generate_interpretations_selective(
            predictions_df.nlargest(top_n_for_interpretation, 'current_frequency'),
            analysis_df
        )

        self.memory_monitor.check_memory("After interpretations")

        # Step 8: Generate report
        logger.info("\n[8/8] Generating report...")
        report_data = self._prepare_report_data(
            stats=stats,
            analysis_df=analysis_df.head(50),  # Limit for memory
            predictions_df=predictions_df.head(50),
            high_risk_keywords=high_risk_keywords.head(20),
            emerging_keywords=emerging_keywords.head(20),
            interpretations=interpretations
        )

        report_path = self.report_generator.generate_markdown_report(report_data)
        logger.info(f"Report generated: {report_path}")

        # Save additional data
        csv_path = self.report_generator.save_data_to_csv(
            predictions_df.head(100),  # Save top 100 only
            "predictions.csv"
        )
        logger.info(f"Predictions saved: {csv_path}")

        json_path = self.report_generator.generate_summary_json(report_data)
        logger.info(f"Summary saved: {json_path}")

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

                combined = f"{row.get('title', '')} {row.get('content', '')}"
                combined_keyword = self.preprocessor.keywordify_text(combined)

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

        all_keywords = sorted(all_keywords)[:config.TOP_N_KEYWORDS]  # Limit total keywords
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

        logger.info(f"Frequency matrix shape: {df.shape}")
        return df

    def _generate_interpretations_selective(self, predictions_df, analysis_df):
        """Generate interpretations for selected keywords only"""
        interpretations = {}

        if self.llm_analyzer and self.use_llm:
            logger.info("Generating LLM interpretations for top keywords...")

            for _, pred_row in tqdm(predictions_df.iterrows(),
                                   total=len(predictions_df),
                                   desc="LLM interpretations"):
                keyword = pred_row['keyword']

                # Get analysis
                analysis = analysis_df[analysis_df['keyword'] == keyword]
                if len(analysis) == 0:
                    continue

                analysis_dict = analysis.iloc[0].to_dict()

                # Generate interpretation
                interpretation = self.llm_analyzer.generate_keyword_interpretation(
                    keyword, analysis_dict, pred_row.to_dict()
                )

                rationale = self.llm_analyzer.generate_prediction_rationale(
                    keyword, pred_row.to_dict(), analysis_dict
                )

                interpretations[keyword] = {
                    'interpretation': interpretation,
                    'rationale': rationale
                }

                if config.AGGRESSIVE_GC:
                    gc.collect()

        else:
            # Rule-based interpretations
            logger.info("Generating rule-based interpretations...")
            llm_analyzer = LLMAnalyzer()  # Without API key

            for _, pred_row in predictions_df.iterrows():
                keyword = pred_row['keyword']

                analysis = analysis_df[analysis_df['keyword'] == keyword]
                if len(analysis) == 0:
                    continue

                analysis_dict = analysis.iloc[0].to_dict()

                interpretation = llm_analyzer._generate_rule_based_interpretation(
                    keyword, analysis_dict, pred_row.to_dict()
                )

                rationale = llm_analyzer.generate_prediction_rationale(
                    keyword, pred_row.to_dict(), analysis_dict
                )

                interpretations[keyword] = {
                    'interpretation': interpretation,
                    'rationale': rationale
                }

        # Generate overall analysis
        top_kw_list = predictions_df.to_dict('records')
        emerging_kw = predictions_df[
            predictions_df['lifecycle_stage'].isin(['introduction', 'growth'])
        ]['keyword'].tolist()
        declining_kw = predictions_df[
            predictions_df['lifecycle_stage'].isin(['decline', 'obsolescence'])
        ]['keyword'].tolist()

        if self.llm_analyzer and self.use_llm:
            overall = self.llm_analyzer.generate_overall_analysis(
                top_kw_list, emerging_kw, declining_kw
            )
        else:
            overall = LLMAnalyzer()._generate_rule_based_overall_analysis(
                top_kw_list, emerging_kw, declining_kw
            )

        interpretations['__overall__'] = overall

        return interpretations

    def _prepare_report_data(self, stats, analysis_df, predictions_df,
                            high_risk_keywords, emerging_keywords, interpretations):
        """Prepare data for report generation (memory-efficient)"""

        # Merge analysis and predictions for top keywords only
        top_keywords = []

        for _, pred_row in predictions_df.iterrows():
            keyword = pred_row['keyword']

            analysis = analysis_df[analysis_df['keyword'] == keyword]
            if len(analysis) == 0:
                continue

            analysis_dict = analysis.iloc[0].to_dict()

            kw_data = {
                **analysis_dict,
                **pred_row.to_dict(),
                'interpretation': interpretations.get(keyword, {}).get('interpretation', ''),
                'rationale': interpretations.get(keyword, {}).get('rationale', '')
            }
            top_keywords.append(kw_data)

        # Other keyword groups
        emerging_detail = emerging_keywords.to_dict('records')
        declining_kw = predictions_df[
            predictions_df['lifecycle_stage'].isin(['decline', 'obsolescence'])
        ]
        declining_detail = declining_kw.to_dict('records')
        high_risk_detail = high_risk_keywords.to_dict('records')

        return {
            'statistics': stats,
            'top_keywords': top_keywords,
            'emerging_keywords': emerging_keywords['keyword'].tolist() if len(emerging_keywords) > 0 else [],
            'emerging_keywords_detail': emerging_detail,
            'declining_keywords': declining_kw['keyword'].tolist() if len(declining_kw) > 0 else [],
            'declining_keywords_detail': declining_detail,
            'high_risk_keywords': high_risk_detail,
            'overall_analysis': interpretations.get('__overall__', '')
        }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Memory-Efficient FACTOR-LLM for Large Datasets'
    )
    parser.add_argument(
        '--use-llm',
        action='store_true',
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
    app = MemoryEfficientFactorLLM(use_llm=args.use_llm, chunk_size=args.chunk_size)

    try:
        report_path = app.run()
        print(f"\n✓ Analysis complete! Report saved to: {report_path}")
        return 0
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
