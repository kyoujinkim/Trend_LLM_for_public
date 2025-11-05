"""
FACTOR-LLM Main Application
Factor Analysis and Cyclical Trend Observation using LLMs
"""
import logging
from pathlib import Path
import argparse
from tqdm import tqdm

# Import configuration
import config

# Import modules
from src.data_loader import NewsDataLoader
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


class FactorLLM:
    """Main FACTOR-LLM application"""

    def __init__(self, use_llm: bool = False):
        """
        Initialize FACTOR-LLM

        Args:
            use_llm: Whether to use LLM for interpretation
        """
        self.use_llm = use_llm

        # Initialize components
        logger.info("Initializing FACTOR-LLM components...")

        self.data_loader = NewsDataLoader(config.CSV_FILES)
        self.preprocessor = TextPreprocessor()
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

        logger.info("Initialization complete")

    def run(self):
        """Execute the complete FACTOR-LLM pipeline"""
        logger.info("=" * 80)
        logger.info("FACTOR-LLM Analysis Pipeline Started")
        logger.info("=" * 80)

        # Step 1: Load data
        logger.info("\n[1/7] Loading data...")
        df = self.data_loader.load_data()
        stats = self.data_loader.get_statistics()
        logger.info(f"Loaded {stats['total_articles']:,} articles")

        # Step 2: Preprocess text
        logger.info("\n[2/7] Preprocessing text...")
        df = self.preprocessor.preprocess_dataframe(
            df,
            text_columns=['title', 'content'],
            extract_nouns_only=True
        )
        logger.info("Text preprocessing complete")

        # Combine cleaned text
        df['combined_text'] = df['title_cleaned'] + ' ' + df['content_cleaned']

        # Step 3: Extract keywords
        logger.info("\n[3/7] Extracting keywords...")
        freq_df = self.keyword_extractor.get_keyword_frequency_over_time(
            df,
            text_column='combined_text',
            date_column='date'
        )
        logger.info(f"Extracted {len(freq_df.columns)} keywords")

        keyword_stats = self.keyword_extractor.get_keyword_statistics(freq_df)
        logger.info(f"Top keyword: {keyword_stats.iloc[0]['keyword']} "
                   f"({keyword_stats.iloc[0]['total_occurrences']} occurrences)")

        # Step 4: Time series analysis
        logger.info("\n[4/7] Analyzing time series...")
        analysis_df = self.time_series_analyzer.analyze_all_keywords(freq_df)
        logger.info("Time series analysis complete")

        # Classify keywords
        classifications = self.time_series_analyzer.get_keyword_classifications(analysis_df)
        logger.info(f"Keyword classifications: {list(classifications.keys())}")

        # Step 5: Generate predictions
        logger.info("\n[5/7] Generating predictions...")
        predictions_df = self.predictor.predict_all_keywords(freq_df, analysis_df)
        logger.info("Predictions complete")

        # Get specific keyword groups
        high_risk_keywords = self.predictor.get_high_risk_keywords(predictions_df, threshold=0.6)
        emerging_keywords = self.predictor.get_emerging_keywords(predictions_df)
        logger.info(f"High risk keywords: {len(high_risk_keywords)}")
        logger.info(f"Emerging keywords: {len(emerging_keywords)}")

        # Step 6: LLM interpretation (optional)
        logger.info("\n[6/7] Generating interpretations...")
        if self.llm_analyzer and self.use_llm:
            interpretations = self._generate_llm_interpretations(
                predictions_df, analysis_df
            )
        else:
            interpretations = self._generate_rule_based_interpretations(
                predictions_df, analysis_df
            )
        logger.info("Interpretations complete")

        # Step 7: Generate report
        logger.info("\n[7/7] Generating report...")
        report_data = self._prepare_report_data(
            stats=stats,
            analysis_df=analysis_df,
            predictions_df=predictions_df,
            high_risk_keywords=high_risk_keywords,
            emerging_keywords=emerging_keywords,
            interpretations=interpretations,
            classifications=classifications
        )

        report_path = self.report_generator.generate_markdown_report(report_data)
        logger.info(f"Report generated: {report_path}")

        # Save additional data
        csv_path = self.report_generator.save_data_to_csv(
            predictions_df,
            "predictions.csv"
        )
        logger.info(f"Predictions saved: {csv_path}")

        json_path = self.report_generator.generate_summary_json(report_data)
        logger.info(f"Summary saved: {json_path}")

        logger.info("\n" + "=" * 80)
        logger.info("FACTOR-LLM Analysis Complete!")
        logger.info("=" * 80)
        logger.info(f"\nOutputs saved to: {config.OUTPUT_DIR}")

        return report_path

    def _generate_llm_interpretations(self, predictions_df, analysis_df):
        """Generate LLM-based interpretations"""
        interpretations = {}

        logger.info("Generating LLM interpretations for top keywords...")

        # Get top 20 keywords for detailed interpretation
        top_keywords = predictions_df.nlargest(20, 'current_frequency')

        for _, pred_row in tqdm(top_keywords.iterrows(), total=len(top_keywords),
                               desc="LLM interpretations"):
            keyword = pred_row['keyword']

            # Get analysis for this keyword
            analysis = analysis_df[analysis_df['keyword'] == keyword].iloc[0].to_dict()

            # Generate interpretation
            interpretation = self.llm_analyzer.generate_keyword_interpretation(
                keyword, analysis, pred_row.to_dict()
            )

            # Generate rationale
            rationale = self.llm_analyzer.generate_prediction_rationale(
                keyword, pred_row.to_dict(), analysis
            )

            interpretations[keyword] = {
                'interpretation': interpretation,
                'rationale': rationale
            }

        # Generate overall analysis
        top_kw_list = predictions_df.nlargest(10, 'current_frequency').to_dict('records')
        emerging_kw = predictions_df[
            predictions_df['lifecycle_stage'].isin(['introduction', 'growth'])
        ]['keyword'].tolist()
        declining_kw = predictions_df[
            predictions_df['lifecycle_stage'].isin(['decline', 'obsolescence'])
        ]['keyword'].tolist()

        overall_analysis = self.llm_analyzer.generate_overall_analysis(
            top_kw_list, emerging_kw, declining_kw
        )
        interpretations['__overall__'] = overall_analysis

        return interpretations

    def _generate_rule_based_interpretations(self, predictions_df, analysis_df):
        """Generate rule-based interpretations"""
        interpretations = {}

        logger.info("Generating rule-based interpretations...")

        llm_analyzer = LLMAnalyzer()  # Without API key for rule-based

        for _, pred_row in predictions_df.nlargest(20, 'current_frequency').iterrows():
            keyword = pred_row['keyword']
            analysis = analysis_df[analysis_df['keyword'] == keyword].iloc[0].to_dict()

            interpretation = llm_analyzer._generate_rule_based_interpretation(
                keyword, analysis, pred_row.to_dict()
            )

            rationale = llm_analyzer.generate_prediction_rationale(
                keyword, pred_row.to_dict(), analysis
            )

            interpretations[keyword] = {
                'interpretation': interpretation,
                'rationale': rationale
            }

        # Overall analysis
        top_kw_list = predictions_df.nlargest(10, 'current_frequency').to_dict('records')
        emerging_kw = predictions_df[
            predictions_df['lifecycle_stage'].isin(['introduction', 'growth'])
        ]['keyword'].tolist()
        declining_kw = predictions_df[
            predictions_df['lifecycle_stage'].isin(['decline', 'obsolescence'])
        ]['keyword'].tolist()

        overall_analysis = llm_analyzer._generate_rule_based_overall_analysis(
            top_kw_list, emerging_kw, declining_kw
        )
        interpretations['__overall__'] = overall_analysis

        return interpretations

    def _prepare_report_data(self, stats, analysis_df, predictions_df,
                            high_risk_keywords, emerging_keywords,
                            interpretations, classifications):
        """Prepare data for report generation"""

        # Merge analysis and predictions
        top_keywords = []
        for _, pred_row in predictions_df.nlargest(50, 'current_frequency').iterrows():
            keyword = pred_row['keyword']
            analysis = analysis_df[analysis_df['keyword'] == keyword].iloc[0].to_dict()

            kw_data = {
                **analysis,
                **pred_row.to_dict(),
                'interpretation': interpretations.get(keyword, {}).get('interpretation', ''),
                'rationale': interpretations.get(keyword, {}).get('rationale', '')
            }
            top_keywords.append(kw_data)

        # Emerging keywords detail
        emerging_detail = []
        for _, row in emerging_keywords.iterrows():
            kw_data = row.to_dict()
            kw_data['interpretation'] = interpretations.get(
                row['keyword'], {}
            ).get('interpretation', '')
            emerging_detail.append(kw_data)

        # Declining keywords detail
        declining_kw = predictions_df[
            predictions_df['lifecycle_stage'].isin(['decline', 'obsolescence'])
        ]
        declining_detail = []
        for _, row in declining_kw.iterrows():
            kw_data = row.to_dict()
            declining_detail.append(kw_data)

        # High risk keywords detail
        high_risk_detail = []
        for _, row in high_risk_keywords.iterrows():
            kw_data = row.to_dict()
            kw_data['rationale'] = interpretations.get(
                row['keyword'], {}
            ).get('rationale', '')
            high_risk_detail.append(kw_data)

        return {
            'statistics': stats,
            'top_keywords': top_keywords,
            'emerging_keywords': emerging_keywords['keyword'].tolist() if len(emerging_keywords) > 0 else [],
            'emerging_keywords_detail': emerging_detail,
            'declining_keywords': declining_kw['keyword'].tolist() if len(declining_kw) > 0 else [],
            'declining_keywords_detail': declining_detail,
            'high_risk_keywords': high_risk_detail,
            'classifications': classifications,
            'overall_analysis': interpretations.get('__overall__', '')
        }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='FACTOR-LLM: Factor Analysis and Cyclical Trend Observation using LLMs'
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

    args = parser.parse_args()

    # Set API key if provided
    if args.api_key:
        if config.LLM_PROVIDER == 'anthropic':
            config.ANTHROPIC_API_KEY = args.api_key
        else:
            config.OPENAI_API_KEY = args.api_key

    # Initialize and run
    app = FactorLLM(use_llm=args.use_llm)

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
