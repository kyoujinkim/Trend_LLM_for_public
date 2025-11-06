"""
Report Generator Module
Generates comprehensive reports with predictions and visualizations
"""
import logging
from typing import Dict, List
from datetime import datetime
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive analysis reports"""

    def __init__(self, output_dir: Path):
        """
        Initialize report generator

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_markdown_report(self, data: Dict, target_date: str = None) -> str:
        """
        Generate markdown report

        Args:
            data: Dictionary containing all analysis results
            filename: Output filename (auto-generated if None)

        Returns:
            Path to generated report
        """
        if target_date is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"FACTOR-LLM_Report_{timestamp}.md"
        else:
            filename = f"FACTOR-LLM_Report_{target_date}.md"

        report_path = self.output_dir / filename

        # Generate report content
        content = self._build_markdown_content(data)

        # Write to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Report saved to: {report_path}")
        return str(report_path)

    def _build_markdown_content(self, data: Dict) -> str:
        """
        Build markdown content

        Args:
            data: Analysis data

        Returns:
            Markdown formatted string
        """
        sections = []

        # Header
        sections.append(self._generate_header())

        # Executive Summary
        sections.append(self._generate_executive_summary(data))

        # Industry Trends - NEW SECTION
        if 'overall_analysis' in data:
            industry_section = self._generate_industry_trends_section(data)
            if industry_section:
                sections.append(industry_section)

        # Data Overview
        sections.append(self._generate_data_overview(data))

        # Top Keywords
        sections.append(self._generate_top_keywords_section(data))

        # Emerging Keywords
        sections.append(self._generate_emerging_keywords_section(data))

        # Declining Keywords
        sections.append(self._generate_declining_keywords_section(data))

        # Detailed Analysis
        detailed = self._generate_detailed_analysis_section(data)
        if detailed:
            sections.append(detailed)

        # Methodology
        sections.append(self._generate_methodology_section())

        # Footer
        sections.append(self._generate_footer())

        return '\n\n'.join(sections)

    def _generate_header(self) -> str:
        """Generate report header"""
        return f"""# FACTOR-LLM Analysis Report
        ## Factor Analysis and Cyclical Trend Observation using LLMs
        
        **Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        ---"""

    def _generate_executive_summary(self, data: Dict) -> str:
        """Generate executive summary"""
        stats = data.get('statistics', {})
        keyword = data.get('keywords_data', [])
        top = keyword[keyword['status']=='dominant']
        rising = keyword[keyword['status']=='rising']
        declining = keyword[keyword['status']=='declining']

        summary = f"""## Executive Summary

        이 보고서는 {stats.get('total_articles', 0):,}개의 뉴스 기사를 분석하여 산업 트렌드와 키워드 라이프사이클을 예측합니다.
        
        ### 주요 발견사항
        
        - **분석 기간**: {stats.get('date_range', {}).get('start', 'N/A')} ~ {stats.get('date_range', {}).get('end', 'N/A')}
        - **총 분석 기사 수**: {stats.get('total_articles', 0):,}개
        - **주요 키워드**: {', '.join(top.keyword[:5]) if ~top.empty else 'N/A'}
        - **신규 부상 키워드**: {', '.join(rising.keyword[:5]) if ~rising.empty else 'N/A'}
        - **쇠퇴 중인 키워드**: {', '.join(declining.keyword[:5]) if ~declining.empty else 'N/A'}
        """
        return summary

    def _generate_industry_trends_section(self, data: Dict) -> str:
        """Generate industry trends section in hierarchical format"""
        overall_analysis = data.get('overall_analysis', None)

        if not overall_analysis:
            return ""

        # Handle string response (fallback for old format)
        if isinstance(overall_analysis, str):
            return ""

        content = ["## 산업 트렌드 종합\n"]

        # Trend Industries
        trend_industries = None
        if hasattr(overall_analysis, 'trend_industry'):
            trend_industries = overall_analysis.trend_industry
        elif isinstance(overall_analysis, dict) and 'trend_industry' in overall_analysis:
            trend_industries = overall_analysis['trend_industry']

        if trend_industries:
            content.append("### 주요 트렌드 산업 (Trend Industries)\n")
            for industry_data in trend_industries:
                industry_name = industry_data.industry if hasattr(industry_data, 'industry') else industry_data.get('industry', '')
                content.append(f"- **{industry_name}**")

                keywords = industry_data.keywords if hasattr(industry_data, 'keywords') else industry_data.get('keywords', [])
                if keywords:
                    for kw in keywords:
                        kw_name = kw.keyword if hasattr(kw, 'keyword') else kw.get('keyword', '')
                        kw_interp = kw.interpretation if hasattr(kw, 'interpretation') else kw.get('interpretation', '')
                        content.append(f"  - **{kw_name}**: {kw_interp}")
                content.append("")

        # Rising Industries
        rising_industries = None
        if hasattr(overall_analysis, 'rising_industry'):
            rising_industries = overall_analysis.rising_industry
        elif isinstance(overall_analysis, dict) and 'rising_industry' in overall_analysis:
            rising_industries = overall_analysis['rising_industry']

        if rising_industries:
            content.append("### 신규 부상 산업 (Rising Industries)\n")
            for industry_data in rising_industries:
                industry_name = industry_data.industry if hasattr(industry_data, 'industry') else industry_data.get('industry', '')
                content.append(f"- **{industry_name}**")

                keywords = industry_data.keywords if hasattr(industry_data, 'keywords') else industry_data.get('keywords', [])
                if keywords:
                    for kw in keywords:
                        kw_name = kw.keyword if hasattr(kw, 'keyword') else kw.get('keyword', '')
                        kw_interp = kw.interpretation if hasattr(kw, 'interpretation') else kw.get('interpretation', '')
                        content.append(f"  - **{kw_name}**: {kw_interp}")
                content.append("")

        # Return empty if no content was added
        if len(content) == 1:
            return ""

        return '\n'.join(content)

    def _generate_data_overview(self, data: Dict) -> str:
        """Generate data overview section"""
        stats = data.get('statistics', {})

        return f"""## 데이터 개요

        - **총 기사 수**: {stats.get('total_articles', 0):,}
        - **분석 일자 수**: {stats.get('unique_dates', 0)}일
        - **데이터 소스 수**: {stats.get('unique_sources', 0)}
        - **추출된 키워드 수**: {len(data.get('top_keywords', []))}
        """

    def _generate_top_keywords_section(self, data: Dict) -> str:
        """Generate top keywords section"""
        keywords = data.get('keywords_data', [])
        top_keywords = keywords[keywords['status']=='dominant'][:10]

        if top_keywords.empty:
            return "## 주요 키워드\n\n데이터 없음"

        content = ["## 주요 키워드\n"]
        content.append("현재 가장 주도적인 키워드들입니다.\n")
        content.append("| 순위 | 키워드 | 출현 빈도 | 트렌드 | 라이프사이클 | 소멸 확률(1개월) | (3개월) | (6개월) | (1년) |")
        content.append("|------|--------|-----------|--------|--------------|---------------|--------------|--------------|---------|")

        for idx, kw in top_keywords.iterrows():
            keyword = kw['keyword']
            freq = kw['current_frequency']
            trend = kw['trend_x']
            lifecycle = kw['lifecycle_description']
            extinction_prob = kw['extinction_probabilities']
            extinction_prob_1m = extinction_prob.get('30_days', 0) * 100
            extinction_prob_3m = extinction_prob.get('90_days', 0) * 100
            extinction_prob_6m = extinction_prob.get('180_days', 0) * 100
            extinction_prob_1y = extinction_prob.get('365_days', 0) * 100

            # Translate trend to Korean
            trend_kr = {
                'increasing': '상승',
                'decreasing': '하락',
                'stable': '안정',
                'no_significant_trend': '불명확'
            }.get(trend, trend)

            # Translate lifecycle to Korean
            lifecycle_kr = {
                'introduction': '도입기',
                'growth': '성장기',
                'maturity': '성숙기',
                'decline': '쇠퇴기',
                'obsolescence': '소멸기',
                'uncertain': '불명확'
            }.get(lifecycle, lifecycle)

            content.append(
                f"| {idx} | {keyword} | {freq} | {trend_kr} | {lifecycle_kr} | {extinction_prob_1m:.1f}% | {extinction_prob_3m:.1f}% | {extinction_prob_6m:.1f}% | {extinction_prob_1y:.1f}% |"
            )

        return '\n'.join(content)

    def _generate_emerging_keywords_section(self, data: Dict) -> str:
        """Generate emerging keywords section"""
        keywords = data.get('keywords_data', [])
        rising_keywords = keywords[keywords['status']=='rising'][:10]

        if rising_keywords.empty:
            return "## 신규 부상 키워드\n\n현재 특별히 부상하는 키워드가 관찰되지 않았습니다."

        content = ["## 신규 부상 키워드\n"]
        content.append("새롭게 주목받고 있는 키워드들입니다.\n")
        content.append("| 순위 | 키워드 | 출현 빈도 | 트렌드 | 라이프사이클 | 소멸 확률(1개월) | (3개월) | (6개월) | (1년) |")
        content.append("|------|--------|-----------|--------|--------------|---------------|--------------|--------------|---------|")

        for idx, kw in rising_keywords.iterrows():
            keyword = kw['keyword']
            freq = kw['current_frequency']
            trend = kw['trend_x']
            lifecycle = kw['lifecycle_description']
            extinction_prob = kw['extinction_probabilities']
            extinction_prob_1m = extinction_prob.get('30_days', 0) * 100
            extinction_prob_3m = extinction_prob.get('90_days', 0) * 100
            extinction_prob_6m = extinction_prob.get('180_days', 0) * 100
            extinction_prob_1y = extinction_prob.get('365_days', 0) * 100

            # Translate trend to Korean
            trend_kr = {
                'increasing': '상승',
                'decreasing': '하락',
                'stable': '안정',
                'no_significant_trend': '불명확'
            }.get(trend, trend)

            # Translate lifecycle to Korean
            lifecycle_kr = {
                'introduction': '도입기',
                'growth': '성장기',
                'maturity': '성숙기',
                'decline': '쇠퇴기',
                'obsolescence': '소멸기',
                'uncertain': '불명확'
            }.get(lifecycle, lifecycle)

            content.append(
                f"| {idx} | {keyword} | {freq} | {trend_kr} | {lifecycle_kr} | {extinction_prob_1m:.1f}% | {extinction_prob_3m:.1f}% | {extinction_prob_6m:.1f}% | {extinction_prob_1y:.1f}% |"
            )

        return '\n'.join(content)

    def _generate_declining_keywords_section(self, data: Dict) -> str:
        """Generate declining keywords section"""
        keywords = data.get('keywords_data', [])
        declining_keywords = keywords[keywords['status']=='declining'][:10]

        if declining_keywords.empty:
            return "## 쇠퇴 중인 키워드\n\n현재 특별히 쇠퇴하는 키워드가 관찰되지 않았습니다."

        content = ["## 쇠퇴 중인 키워드\n"]
        content.append("관심이 감소하고 있는 키워드들입니다.\n")
        content.append("| 순위 | 키워드 | 출현 빈도 | 트렌드 | 라이프사이클 | 소멸 확률(1개월) | (3개월) | (6개월) | (1년) |")
        content.append("|------|--------|-----------|--------|--------------|---------------|--------------|--------------|---------|")

        for idx, kw in declining_keywords.iterrows():
            keyword = kw['keyword']
            freq = kw['current_frequency']
            trend = kw['trend_x']
            lifecycle = kw['lifecycle_description']
            extinction_prob = kw['extinction_probabilities']
            extinction_prob_1m = extinction_prob.get('30_days', 0) * 100
            extinction_prob_3m = extinction_prob.get('90_days', 0) * 100
            extinction_prob_6m = extinction_prob.get('180_days', 0) * 100
            extinction_prob_1y = extinction_prob.get('365_days', 0) * 100

            # Translate trend to Korean
            trend_kr = {
                'increasing': '상승',
                'decreasing': '하락',
                'stable': '안정',
                'no_significant_trend': '불명확'
            }.get(trend, trend)

            # Translate lifecycle to Korean
            lifecycle_kr = {
                'introduction': '도입기',
                'growth': '성장기',
                'maturity': '성숙기',
                'decline': '쇠퇴기',
                'obsolescence': '소멸기',
                'uncertain': '불명확'
            }.get(lifecycle, lifecycle)

            content.append(
                f"| {idx} | {keyword} | {freq} | {trend_kr} | {lifecycle_kr} | {extinction_prob_1m:.1f}% | {extinction_prob_3m:.1f}% | {extinction_prob_6m:.1f}% | {extinction_prob_1y:.1f}% |"
            )

        return '\n'.join(content)

    def _generate_high_risk_section(self, data: Dict) -> str:
        """Generate high risk keywords section"""
        high_risk = data.get('high_risk_keywords', [])

        if not high_risk:
            return "## 고위험 키워드\n\n1년 내 소멸 확률이 높은 키워드가 관찰되지 않았습니다."

        content = ["## 고위험 키워드 (높은 소멸 확률)\n"]
        content.append("1년 내 소멸 확률이 60% 이상인 키워드들입니다.\n")
        content.append("| 키워드 | 1년 소멸 확률 | 현재 빈도 | 예측 근거 |")
        content.append("|--------|---------------|-----------|-----------|")

        for kw_data in high_risk[:10]:
            keyword = kw_data.get('keyword', '')
            prob = kw_data.get('extinction_prob_365_days', 0) * 100
            freq = kw_data.get('current_frequency', 0)
            rationale = kw_data.get('rationale', '').replace('\n', ' ')[:80]

            content.append(
                f"| {keyword} | {prob:.1f}% | {freq:.1f} | {rationale}... |"
            )

        return '\n'.join(content)

    def _generate_detailed_analysis_section(self, data: Dict) -> str:
        """Generate detailed analysis for selected keywords"""
        keywords = data.get('keywords_data', [])
        top_keywords = keywords[keywords['status']=='rising'].iloc[:5]

        if top_keywords.empty:
            return ""

        content = ["## 부상 키워드 상세 분석\n"]

        for idx, kw in top_keywords.iterrows():
            keyword = kw['keyword']
            interp = kw['interpretation']

            if interp and interp != '해석 없음':
                content.append(f"### {keyword}\n")
                content.append(f"**해석**: {interp}\n")

        # Return empty if no detailed content
        if len(content) == 1:
            return ""

        return '\n'.join(content)

    def _generate_overall_analysis_section(self, data: Dict) -> str:
        """Generate overall analysis section"""
        overall = data.get('overall_analysis', '')

        if not overall:
            return ""

        # Don't duplicate if it's a Pydantic model (already shown in industry trends)
        if not isinstance(overall, str):
            return ""

        return f"""## 종합 분석

        {overall}
        """

    def _generate_methodology_section(self) -> str:
        """Generate methodology section"""
        return """## 방법론

                ### 데이터 처리
                1. CSV 형식의 뉴스 데이터 로드
                2. HTML 태그 제거 및 텍스트 정제
                3. 한국어 형태소 분석 (Mecab)을 통한 명사 추출
                
                ### 키워드 추출
                - TF-IDF 알고리즘을 사용한 중요 키워드 추출
                - 날짜별 키워드 빈도 추적
                
                ### 시계열 분석
                - 이동평균을 통한 트렌드 평활화
                - 선형 회귀를 통한 트렌드 방향 감지
                - FFT를 이용한 주기적 패턴 감지
                
                ### 예측 모델
                - 지수 감쇠 모델을 기반으로 한 미래 빈도 예측
                - 정규분포 가정 하의 소멸 확률 계산
                - 트렌드와 변동성을 고려한 확률 조정
                
                ### LLM 통합
                - Claude/GPT를 활용한 해석 가능한 분석 생성
                - 데이터 패턴에 대한 맥락적 이해 제공
                
                ### 신뢰도
                - 데이터 양과 품질에 따라 high/medium/low로 분류
                - 통계적 유의성 검증 (p-value < 0.05)
                """

    def _generate_footer(self) -> str:
        """Generate report footer"""
        return """---

                **FACTOR-LLM**
                *Factor Analysis and Cyclical Trend Observation using LLMs*
                
                본 보고서는 자동 생성되었으며, 참고 목적으로만 사용되어야 합니다.
                """