"""
LLM Analyzer Module
Integrates LLM for interpretable analysis and reasoning
"""
import logging
from typing import Dict, List
from pydantic import BaseModel
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """Use LLM to generate interpretable analysis and insights"""

    def __init__(self, provider: str = "anthropic",
                 model: str = "claude-3-5-sonnet-20241022",
                 api_key: str = None):
        """
        Initialize LLM analyzer

        Args:
            provider: LLM provider ("anthropic" or "openai")
            model: Model name
            api_key: API key
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.client = None

        self._initialize_client()

    def _initialize_client(self):
        """Initialize LLM client"""
        if not self.api_key:
            logger.warning("No API key provided. LLM analysis will be skipped.")
            return

        try:
            if self.provider == "anthropic":
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
                logger.info("Anthropic client initialized")
            elif self.provider == "openai":
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized")
            else:
                logger.error(f"Unknown provider: {self.provider}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")

    def generate_keyword_interpretation(self, input_dict: Dict) -> str:
        """
        Generate interpretation for a single keyword

        Args:
            input_dict: Dictionary with 'keyword' keys and 'analysis', and 'prediction' items

        Returns:
            Interpretation text
        """
        class Keyword(BaseModel):
            keyword: str
            interpretation: str
        class OutputFormat(BaseModel):
            declining_keyword_list: List[Keyword]
            dominant_keyword_list: List[Keyword]
            rising_keyword_list: List[Keyword]

        prompt = f"""주어진 데이터는 각 키워드에 대한 정보를 담은 Dict 데이터입니다.
        모든 데이터를 상승 추세, 안정 추세, 하락 추세로 나누고, 키워드에 대한 분석 결과를 바탕으로 해석과 통찰을 제공해주세요.

        데이터 : {json.dumps(input_dict, ensure_ascii=False)}

        데이터 구성:
        - 트렌드: trend
        - 평균 빈도: mean_frequency
        - 변동성: volatility
        - 총 출현 횟수: total_occurrences
        - 라이프사이클 단계: lifecycle_stage
        - 1년 내 소멸 확률: extinction_prob_365_days
 
        분석 결과 양식:
        다음 내용을 포함하여 2-3문장으로 해석을 작성해주세요:
        1. 현재 키워드의 상태와 의미
        2. 관찰된 패턴의 원인 또는 배경
        3. 향후 전망과 시사점
        
        한국어로 작성해주세요."""

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            elif self.provider == "openai":
                response = self.client.responses.parse(
                    model=self.model,
                    input=[{"role": "user", "content": prompt}],
                    text_format=OutputFormat
                )
                return response.output_parsed
        except Exception as e:
            return f"LLM interpretation failed: {e}"

    def _generate_rule_based_interpretation(self, keyword: str,
                                           analysis: Dict,
                                           prediction: Dict) -> str:
        """
        Generate rule-based interpretation when LLM is not available

        Args:
            keyword: Keyword name
            analysis: Analysis results
            prediction: Prediction results

        Returns:
            Interpretation text
        """
        trend = analysis.get('trend', 'unknown')
        lifecycle = prediction.get('lifecycle_stage', 'unknown')
        extinction_prob = prediction.get('extinction_prob_365_days', 0) * 100

        interpretations = []

        # Trend interpretation
        if trend == 'increasing':
            interpretations.append(f"'{keyword}'는 상승 트렌드를 보이며 주목받고 있습니다.")
        elif trend == 'decreasing':
            interpretations.append(f"'{keyword}'는 하락 트렌드로 관심이 감소하고 있습니다.")
        else:
            interpretations.append(f"'{keyword}'는 안정적인 패턴을 유지하고 있습니다.")

        # Lifecycle interpretation
        lifecycle_desc = {
            'introduction': '초기 도입 단계',
            'growth': '성장 단계',
            'maturity': '성숙 단계',
            'decline': '쇠퇴 단계',
            'obsolescence': '소멸 단계'
        }
        if lifecycle in lifecycle_desc:
            interpretations.append(f"현재 {lifecycle_desc[lifecycle]}에 있습니다.")

        # Prediction interpretation
        if extinction_prob > 60:
            interpretations.append(f"1년 내 소멸 확률이 {extinction_prob:.1f}%로 높아 주의가 필요합니다.")
        elif extinction_prob < 20:
            interpretations.append(f"1년 내 소멸 확률이 {extinction_prob:.1f}%로 낮아 지속적인 관심이 예상됩니다.")
        else:
            interpretations.append(f"1년 내 소멸 확률은 {extinction_prob:.1f}%입니다.")

        return ' '.join(interpretations)

    def generate_overall_analysis(self, interpretation, news):
        """
        Generate overall market/trend analysis

        Args:
            interpretation: List of top keyword interpretations
            news: Comprehensive views on which industries are related with keywords based on news articles

        Returns:
            Overall analysis text
        """
        class KeywordInterpret(BaseModel):
            keyword: str
            interpretation: str
        class IndustryTrend(BaseModel):
            industry: str
            keywords: List[KeywordInterpret]
        class OutputFormat(BaseModel):
            trend_industry: List[IndustryTrend]
            rising_industry: List[IndustryTrend]

        try:
            interpretation_str = 'Dominant Keywords Interpretations:\n\n'
            for k in interpretation.dominant_keyword_list:
                interpretation_str += f"Keyword: {k.keyword}, Interpretation: {k.interpretation} \n\n"
        except:
            interpretation_str = ''
        try:
            interpretation_str += '\nRising Keywords Interpretations:\n\n'
            for k in interpretation.rising_keyword_list:
                interpretation_str += f"Keyword: {k.keyword}, Interpretation: {k.interpretation} \n\n"
        except:
            pass
        try:
            interpretation_str += '\nDeclining Keywords Interpretations:\n\n'
            for k in interpretation.declining_keyword_list:
                interpretation_str += f"Keyword: {k.keyword}, Interpretation: {k.interpretation} \n\n"
        except:
            pass

        prompt = f"""다음은 키워드 분석과 관련 뉴스 데이터를 바탕으로 전체 시장 및 트렌드에 대한 종합적인 분석입니다.
        키워드 해석: {interpretation_str}
        관련 뉴스: {news}
        
        위 정보를 바탕으로 어떤 산업(반도체, 화학, 2차전지 등 주식 연관)에 유효한 트렌드가 형성되고 있는지, 주요 트렌드, 신규 부상 트렌드, 향후 전망 등을 포함하여 종합적인 분석을 작성해주세요.
        각 분류별로 산업은 GICS 산업으로 3개씩 제시해주세요.
        한국어로 작성해주세요."""
        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            elif self.provider == "openai":
                response = self.client.responses.parse(
                    model=self.model,
                    input=[{"role": "user", "content": prompt}],
                    text_format=OutputFormat
                )
                return response.output_parsed
        except Exception as e:
            return f"LLM overall analysis failed: {e}"


    def _generate_rule_based_overall_analysis(self, top_keywords: List[Dict],
                                             emerging_keywords: List[str],
                                             declining_keywords: List[str]) -> str:
        """
        Generate rule-based overall analysis

        Args:
            top_keywords: List of top keyword analyses
            emerging_keywords: List of emerging keywords
            declining_keywords: List of declining keywords

        Returns:
            Overall analysis text
        """
        sections = []

        # Main trends
        top_kw_names = [kw.get('keyword', '') for kw in top_keywords[:5]]
        sections.append(f"**주요 트렌드**: 현재 '{', '.join(top_kw_names)}' 등이 주요 키워드로 나타나고 있습니다.")

        # Emerging trends
        if emerging_keywords:
            sections.append(f"**신규 부상 트렌드**: '{', '.join(emerging_keywords[:5])}' 등이 새롭게 주목받고 있습니다.")
        else:
            sections.append("**신규 부상 트렌드**: 현재 특별히 부상하는 새로운 키워드는 관찰되지 않았습니다.")

        # Declining trends
        if declining_keywords:
            sections.append(f"**쇠퇴 트렌드**: '{', '.join(declining_keywords[:5])}' 등에 대한 관심이 감소하고 있습니다.")
        else:
            sections.append("**쇠퇴 트렌드**: 현재 특별히 쇠퇴하는 키워드는 관찰되지 않았습니다.")

        # Outlook
        sections.append("**향후 전망**: 지속적인 모니터링을 통해 트렌드 변화를 추적할 필요가 있습니다.")

        return '\n\n'.join(sections)

    def generate_prediction_rationale(self, keyword: str,
                                     prediction: Dict,
                                     analysis: Dict) -> str:
        """
        Generate rationale for prediction

        Args:
            keyword: Keyword name
            prediction: Prediction results
            analysis: Analysis results

        Returns:
            Rationale text
        """
        extinction_prob = prediction.get('extinction_prob_365_days', 0) * 100
        trend = analysis.get('trend', 'unknown')
        trend_slope = analysis.get('trend_slope', 0)

        rationale_parts = []

        rationale_parts.append(f"'{keyword}'의 1년 내 소멸 확률 {extinction_prob:.1f}% 예측 근거:")

        # Trend-based rationale
        if trend == 'decreasing':
            rationale_parts.append(
                f"- 하락 트렌드 확인 (기울기: {trend_slope:.4f})"
            )
        elif trend == 'increasing':
            rationale_parts.append(
                f"- 상승 트렌드 확인 (기울기: {trend_slope:.4f})"
            )

        # Frequency-based rationale
        mean_freq = analysis.get('mean_frequency', 0)
        current_freq = prediction.get('current_frequency', 0)
        if current_freq < mean_freq * 0.5:
            rationale_parts.append(
                f"- 현재 빈도가 평균 대비 낮음 (현재: {current_freq:.1f}, 평균: {mean_freq:.1f})"
            )

        # Volatility-based rationale
        volatility = analysis.get('volatility', 0)
        if volatility > mean_freq * 0.5:
            rationale_parts.append(
                f"- 높은 변동성 관찰 (변동성: {volatility:.2f})"
            )

        # Confidence
        confidence = prediction.get('extinction_confidence', 'medium')
        rationale_parts.append(f"- 예측 신뢰도: {confidence}")

        return '\n'.join(rationale_parts)

    def fetch_relevant_news(self, keywords: list[str], target_date: str) -> List[Dict]:
        """
        Fetch relevant news articles for a keyword

        Args:
            keywords: Keywords list to search
            target_date: Date string in 'YYYYMMDD' format
        Returns:
            Comprehensive view on which industries are related with keywords based on news articles
        """
        if not self.client:
            logger.warning("LLM client not initialized. Cannot fetch relevant news.")
            return []

        prompt = f"""다음 키워드와 날짜에 대한 관련 뉴스 기사를 통해 각각의 키워드와 어떤 산업(반도체, 2차전지 등)이 연관이 있는지 연관 산업을 제시해주세요.
                    키워드: {', '.join(keywords)}
                    날짜: {target_date}"""
        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
            elif self.provider == "openai":
                response = self.client.responses.create(
                    model=self.model,
                    tools=[{"type": "web_search"}],
                    input=prompt
                )

            news_articles = response.output_text
        except:
            news_articles = ''

        return news_articles

