# FACTOR-LLM Algorithm Report
## Factor Analysis and Cyclical Trend Observation using Large Language Models

**Document Version:** 1.0
**Generated:** 2025-11-06
**System:** Memory-Efficient FACTOR-LLM for Stock Market Trend Keyword Extraction

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Introduction](#introduction)
3. [System Architecture](#system-architecture)
4. [Algorithm Overview](#algorithm-overview)
5. [Phase 1: Data Loading and Memory Management](#phase-1-data-loading-and-memory-management)
6. [Phase 2: Text Preprocessing](#phase-2-text-preprocessing)
7. [Phase 3: Keyword Extraction](#phase-3-keyword-extraction)
8. [Phase 4: Time Series Analysis](#phase-4-time-series-analysis)
9. [Phase 5: Lifecycle Prediction](#phase-5-lifecycle-prediction)
10. [Phase 6: LLM-based Interpretation](#phase-6-llm-based-interpretation)
11. [Phase 7: Report Generation](#phase-7-report-generation)
12. [Mathematical Models and Algorithms](#mathematical-models-and-algorithms)
13. [Memory Efficiency Techniques](#memory-efficiency-techniques)
14. [Configuration Parameters](#configuration-parameters)
15. [Use Cases and Applications](#use-cases-and-applications)
16. [Conclusion](#conclusion)

---

## Executive Summary

FACTOR-LLM is an advanced system designed to extract and analyze trending keywords from stock market news data. The system processes large volumes of news articles to identify emerging trends, dominant themes, and declining topics in the financial markets. By combining traditional natural language processing techniques with modern Large Language Models (LLMs), the system provides both quantitative analysis and qualitative interpretations of market trends.

**Key Capabilities:**
- **Memory-Efficient Processing**: Handles large datasets (millions of articles) through chunking and streaming
- **Multi-Stage Analysis**: Combines TF-IDF, time series analysis, and predictive modeling
- **LLM Integration**: Uses Claude/GPT for interpretable insights and industry trend analysis
- **Lifecycle Prediction**: Predicts keyword extinction probabilities and lifecycle stages
- **Automated Reporting**: Generates comprehensive markdown reports with visualizations

The system operates in two main modes:
1. **Keyword Extraction Mode** (`run_only_keywords`): Processes news articles and extracts keywords
2. **Analysis Mode** (`run_only_analysis`): Analyzes pre-extracted keywords and generates insights

---

## Introduction

### Background

In financial markets, understanding emerging trends and topics is crucial for making informed investment decisions. News articles contain valuable information about market sentiment, emerging technologies, industry shifts, and policy changes. However, manually analyzing thousands of news articles daily is impractical.

FACTOR-LLM addresses this challenge by automating the extraction and analysis of keywords from news data. The system identifies:
- **Dominant Keywords**: Topics currently receiving the most attention
- **Rising Keywords**: Emerging trends gaining momentum
- **Declining Keywords**: Topics losing relevance

### Problem Statement

Traditional keyword extraction methods face several challenges:
1. **Scale**: Processing millions of articles requires efficient memory management
2. **Context**: Simple frequency counts miss temporal patterns and trends
3. **Interpretation**: Raw statistics lack business context and actionable insights
4. **Language**: Korean text requires specialized tokenization and processing

### Solution Approach

FACTOR-LLM solves these challenges through:
1. **Chunked Processing**: Streaming data processing to handle large datasets
2. **Multi-Stage Pipeline**: Sequential processing stages with garbage collection
3. **Time Series Analysis**: Detecting trends, cycles, and patterns over time
4. **LLM Integration**: Generating human-readable interpretations of statistical patterns
5. **Korean NLP**: Using Mecab tokenizer and Korean-optimized LLMs

---

## System Architecture

The FACTOR-LLM system follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Application                          │
│              (MemoryEfficientFactorLLM.py)                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Data       │    │     Text     │    │   Keyword    │
│   Loader     │───▶│ Preprocessor │───▶│  Extractor   │
└──────────────┘    └──────────────┘    └──────────────┘
                                                 │
                                                 ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Report     │◀───│     LLM      │◀───│ Time Series  │
│  Generator   │    │   Analyzer   │    │   Analyzer   │
└──────────────┘    └──────────────┘    └──────────────┘
                            ▲                   │
                            │                   ▼
                            │            ┌──────────────┐
                            └────────────│  Prediction  │
                                        │    Model     │
                                        └──────────────┘
```

### Core Components

1. **MemoryEfficientFactorLLM** (Main orchestrator)
   - Coordinates all processing stages
   - Manages memory and garbage collection
   - Provides two execution modes: keyword extraction and analysis

2. **MemoryEfficientLoader** (Data loading)
   - Loads CSV files in chunks
   - Streams data to avoid memory overflow
   - Supports date range filtering

3. **TextPreprocessor** (Text cleaning)
   - Removes HTML, URLs, and special characters
   - Tokenizes Korean text using Mecab
   - Extracts keywords using Hugging Face LLM

4. **KeywordExtractor** (Keyword identification)
   - Applies TF-IDF to identify important keywords
   - Supports n-gram extraction (unigrams, bigrams, trigrams)
   - Tracks keyword frequencies over time

5. **TimeSeriesAnalyzer** (Temporal pattern detection)
   - Calculates moving averages
   - Detects trends using linear regression
   - Identifies cycles using FFT
   - Measures volatility

6. **KeywordLifecyclePredictor** (Future prediction)
   - Predicts future keyword frequencies
   - Calculates extinction probabilities
   - Estimates lifecycle stages

7. **LLMAnalyzer** (Interpretation generation)
   - Generates human-readable interpretations
   - Identifies industry trends
   - Fetches relevant news context

8. **ReportGenerator** (Output generation)
   - Creates markdown reports
   - Formats tables and visualizations
   - Organizes findings by category

---

## Algorithm Overview

The FACTOR-LLM algorithm operates in seven distinct phases:

### Phase Breakdown

**Phase 1: Data Loading** (MemoryEfficientFactorLLM.py:198-228)
- Load news articles from CSV files
- Stream data in chunks to manage memory
- Extract date and text fields

**Phase 2: Text Preprocessing** (MemoryEfficientFactorLLM.py:198-228)
- Clean HTML and special characters
- Use LLM to extract main topics from article content
- Tokenize text using Mecab
- Extract nouns and relevant terms

**Phase 3: Keyword Extraction** (MemoryEfficientFactorLLM.py:230-263)
- Apply TF-IDF to identify important keywords
- Build vocabulary across all dates
- Count keyword frequencies per date
- Save keyword frequencies to disk

**Phase 4: Time Series Analysis** (MemoryEfficientFactorLLM.py:143-149)
- Load keyword frequency data
- Build frequency matrix (dates × keywords)
- Detect trends, cycles, and patterns
- Calculate statistical measures

**Phase 5: Lifecycle Prediction** (MemoryEfficientFactorLLM.py:151-158)
- Predict future keyword frequencies
- Calculate extinction probabilities
- Estimate lifecycle stages
- Identify emerging/declining keywords

**Phase 6: LLM-based Interpretation** (MemoryEfficientFactorLLM.py:161-169)
- Generate keyword interpretations
- Fetch relevant news articles
- Identify industry trends
- Produce comprehensive analysis

**Phase 7: Report Generation** (MemoryEfficientFactorLLM.py:172-181)
- Prepare structured report data
- Format tables and statistics
- Generate markdown document
- Save to output directory

---

## Phase 1: Data Loading and Memory Management

### Overview

The data loading phase handles large CSV files containing news articles. Given that datasets can contain millions of rows, the system uses streaming and chunking techniques to avoid loading all data into memory at once.

### Implementation Details

**File:** `src/memory_efficient_loader.py`

#### MemoryEfficientLoader Class

```python
class MemoryEfficientLoader:
    def __init__(self, csv_files: List[str], data_dir: str)
```

This class manages the loading of multiple CSV files from a directory structure organized by date.

**Key Methods:**

1. **iter_chunks()** (Lines 30-58)
   - **Purpose**: Iterate through CSV files one at a time
   - **Process**:
     - Loop through each CSV file in the file list
     - Load one file at a time using pandas
     - Yield the DataFrame chunk
     - Call garbage collection after each file
   - **Memory Benefit**: Only one file is in memory at any time

2. **load_by_date_range()** (Lines 60-95)
   - **Purpose**: Load data for specific date range
   - **Process**:
     - Filter chunks by date range
     - Accumulate only matching rows
     - Concatenate filtered results
   - **Use Case**: When analyzing specific time periods

3. **get_statistics_incremental()** (Lines 97-147)
   - **Purpose**: Calculate dataset statistics without loading all data
   - **Process**:
     - Accumulate counts incrementally
     - Track unique dates and sources
     - Calculate per-day article counts
   - **Output**: Total articles, date range, unique dates/sources

#### ChunkedDataProcessor Class

**Purpose**: Process data in chunks with various aggregation strategies

**Key Methods:**

1. **aggregate_by_date()** (Lines 186-223)
   - Combines all articles for each date into single text
   - Reduces data size by grouping temporal information
   - Useful for daily trend analysis

### Memory Management Strategy

The system employs several memory management techniques:

1. **Streaming**: Process one chunk at a time
2. **Garbage Collection**: Explicit `gc.collect()` calls after processing
3. **Column Selection**: Load only required columns
4. **Early Deletion**: Delete DataFrames immediately after use

**Example from MemoryEfficientFactorLLM.py:222-225:**
```python
del chunk
if config.AGGRESSIVE_GC:
    gc.collect()
```

### Input Data Format

The system expects CSV files with the following columns:
- **date**: Publication date (YYYYMMDD format)
- **title**: Article headline
- **content**: Article text content
- **source** (optional): News source identifier

### Configuration

**From config.py:**
- `DATA_DIR`: Base directory containing CSV files
- `CSVS`: List of CSV file names to process
- `CHUNK_SIZE`: Number of rows per chunk (default: 1000)
- `PROCESSING_MODE`: 'chunked', 'full', or 'sample'
- `AGGRESSIVE_GC`: Enable/disable aggressive garbage collection

---

## Phase 2: Text Preprocessing

### Overview

Text preprocessing is critical for extracting meaningful keywords from news articles. Raw text contains HTML tags, URLs, special characters, and non-informative content that must be removed. Additionally, Korean text requires specialized tokenization.

### Implementation Details

**File:** `src/text_preprocessor.py`

#### TextPreprocessor Class

```python
class TextPreprocessor:
    def __init__(self, huggingface_model: str, mecab_path: str)
```

This class handles all text cleaning and tokenization operations.

### Sub-Phase 2.1: Hugging Face LLM Keyword Extraction

**Method:** `keywordify_text()` (Lines 66-108)

**Purpose**: Use a Large Language Model to extract the main topics/keywords from article content

**Process:**
1. **Model Loading**:
   - Load Hugging Face model (default: 'LiquidAI/LFM2-350M-Extract')
   - Initialize tokenizer with proper padding tokens
   - Configure for decoder-only architecture
   - Use bfloat16 for memory efficiency

2. **Prompt Engineering**:
   ```python
   prompt = "이 글에서 핵심 주제를 추출해줘"  # Extract core topics from this text
   ```

3. **Token Generation**:
   - Apply chat template to format input
   - Generate tokens with:
     - `do_sample=True`: Enable sampling
     - `temperature=0.3`: Low temperature for focused output
     - `min_p=0.15`: Minimum probability threshold
     - `repetition_penalty=1.05`: Reduce repetition
     - `max_new_tokens=256`: Limit output length

4. **Output Cleaning**:
   - Decode generated tokens
   - Extract assistant response
   - Remove special characters and formatting
   - Return cleaned keyword text

**Batch Processing:** `keywordify_text_batch()` (Lines 110-190)
- Processes multiple texts in parallel for better GPU utilization
- Uses padding to create uniform batch sizes
- Recommended batch size: 8 (configurable)

**Example from MemoryEfficientFactorLLM.py:209-210:**
```python
combined = f"{row.get('title', '')} {row.get('content', '')}"
combined_keyword = self.preprocessor.keywordify_text(combined)
```

### Sub-Phase 2.2: HTML and Special Character Cleaning

**Method:** `clean_html()` (Lines 192-220)

**Process:**
1. Parse HTML using BeautifulSoup
2. Remove `<script>` and `<style>` tags
3. Extract plain text
4. Clean up excessive whitespace
5. Join text chunks with single spaces

**Method:** `remove_urls()` (Lines 222-233)
- Uses regex pattern to match URLs: `https?://\S+|www\.\S+`
- Removes all matching URLs from text

**Method:** `remove_email()` (Lines 235-246)
- Uses regex pattern to match emails: `\S+@\S+`
- Removes all email addresses

**Method:** `remove_special_chars()` (Lines 248-260)
- Keeps only: Korean (Hangul), English, numbers, basic punctuation
- Pattern: `[^가-힣a-zA-Z0-9\s.,!?()]+`
- Replaces all other characters with spaces

**Method:** `remove_non_meaningful_chars()` (Lines 262-281)
- Removes common boilerplate text:
  - Copyright notices: '무단', '전재', '금지'
  - Generic words: '학습', '활용', '저작'
  - Date components: '년', '월', '일'
  - News agency names: '로이터'

### Sub-Phase 2.3: Korean Tokenization

**Method:** `_initialize_tokenizer()` (Lines 28-36)

**Process:**
1. Import Mecab from KoNLPy library
2. Load Mecab dictionary from specified path
3. Fallback to simple whitespace tokenization if Mecab unavailable

**Method:** `extract_nouns()` (Lines 306-327)

**Purpose**: Extract only noun words from text (nouns carry the most semantic meaning)

**Process:**
1. Use Mecab's `nouns()` method
2. Return list of noun tokens
3. Handle exceptions gracefully

**Method:** `extract_pos()` (Lines 329-348)

**Purpose**: Extract specific part-of-speech tags

**Parameters:**
- `pos_tags`: List of POS tags (default: ['NNG', 'NNP'])
  - NNG: Common nouns
  - NNP: Proper nouns

**Process:**
1. Use Mecab's `pos()` method to tag words
2. Filter words matching specified POS tags
3. Return filtered word list

### Sub-Phase 2.4: Complete Preprocessing Pipeline

**Method:** `preprocess_text()` (Lines 350-379)

This method orchestrates the entire preprocessing pipeline:

```python
def preprocess_text(self, text: str, extract_nouns_only: bool = True) -> str:
    # Step 1: Clean HTML
    text = self.clean_html(text)

    # Step 2: Remove URLs and emails
    text = self.remove_urls(text)
    text = self.remove_email(text)

    # Step 3: Remove special characters
    text = self.remove_special_chars(text)

    # Step 4: Remove non-meaningful characters
    text = self.remove_non_meaningful_chars(text)

    # Step 5: Extract nouns if requested
    if extract_nouns_only:
        nouns = self.extract_nouns(text)
        return ' '.join(nouns)
    else:
        return text
```

### Integration in Main Pipeline

**From MemoryEfficientFactorLLM.py:198-228:**

The preprocessing happens in the `_preprocess_in_chunks()` method:

1. **Iterate through data chunks**:
   - Process articles in manageable batches
   - Track progress with tqdm progress bar

2. **For each article**:
   - Combine title and content
   - Extract main keywords using LLM (`keywordify_text`)
   - Clean and tokenize text (`preprocess_text`)
   - Extract nouns only

3. **Group by date**:
   - Store preprocessed text by publication date
   - Build dictionary: `{date: [text1, text2, ...]}`

4. **Memory management**:
   - Delete processed chunks
   - Call garbage collection if configured

**Output**: Dictionary mapping dates to lists of preprocessed texts
```python
preprocessed_by_date = {
    '20251030': ['삼성전자 반도체', '현대차 전기차', ...],
    '20251031': ['LG화학 배터리', '네이버 AI', ...],
    ...
}
```

---

## Phase 3: Keyword Extraction

### Overview

After preprocessing, the system extracts the most important keywords from the text data. This phase uses TF-IDF (Term Frequency-Inverse Document Frequency) to identify keywords that are both frequent and distinctive.

### Implementation Details

**File:** `src/keyword_extractor.py`

#### KeywordExtractor Class

```python
class KeywordExtractor:
    def __init__(self, top_n: int = 50, min_df: int = 2, max_df: float = 0.8)
```

**Parameters:**
- `top_n`: Number of top keywords to extract (default: 50)
- `min_df`: Minimum document frequency (ignore terms appearing in fewer than N documents)
- `max_df`: Maximum document frequency (ignore terms appearing in more than X% of documents)

### Sub-Phase 3.1: TF-IDF Keyword Extraction

**Method:** `extract_by_tfidf()` (Lines 65-117)

**Purpose**: Extract keywords using TF-IDF scoring

**Algorithm:**

1. **Filter Valid Texts**:
   ```python
   texts = [text for text in texts if isinstance(text, str) and text.strip()]
   ```

2. **Create TF-IDF Vectorizer**:
   ```python
   vectorizer = TfidfVectorizer(
       max_features=1000,      # Limit vocabulary size
       min_df=self.min_df,     # Minimum document frequency
       max_df=self.max_df,     # Maximum document frequency
       ngram_range=(2, 3)      # Extract bigrams and trigrams
   )
   ```

3. **Fit and Transform**:
   - Convert texts to TF-IDF matrix
   - Each row represents a document
   - Each column represents a term
   - Values are TF-IDF scores

4. **Calculate Average Scores**:
   ```python
   avg_tfidf = np.asarray(self.tfidf_matrix.mean(axis=0)).ravel()
   ```
   - Average TF-IDF score across all documents
   - Higher scores indicate more important keywords

5. **Select Top Keywords**:
   ```python
   top_indices = avg_tfidf.argsort()[-top_n:][::-1]
   top_keywords = [(feature_names[i], avg_tfidf[i]) for i in top_indices]
   ```

**Output**: List of (keyword, score) tuples
```python
[
    ('반도체 공급망', 0.342),
    ('2차전지 시장', 0.298),
    ('전기차 배터리', 0.276),
    ...
]
```

### Sub-Phase 3.2: N-gram Configuration

**N-gram Types:**
- **Unigram (1-gram)**: Single words - '삼성', '반도체', '시장'
- **Bigram (2-gram)**: Two-word phrases - '삼성 반도체', '반도체 시장'
- **Trigram (3-gram)**: Three-word phrases - '삼성 반도체 공장'

**Configuration in Main Pipeline:**
```python
keywords = self.keyword_extractor.extract_by_tfidf(
    texts,
    top_n=config.TOP_N_KEYWORDS,
    ngram=(2, 3)  # Extract bigrams and trigrams
)
```

**Rationale**: Bigrams and trigrams capture more specific concepts than single words:
- 'AI' vs 'AI 반도체' vs 'AI 반도체 개발'

### Sub-Phase 3.3: Incremental Keyword Extraction

**Method:** `_extract_keywords_incremental()` in MemoryEfficientFactorLLM.py (Lines 230-263)

This method performs a two-pass extraction to ensure consistency:

**Pass 1: Vocabulary Collection**
```python
all_keywords = set()
for date, texts in texts_by_date.items():
    keywords = self.keyword_extractor.extract_by_tfidf(
        texts,
        top_n=config.TOP_N_KEYWORDS,
        ngram=(2, 3)
    )
    date_keywords = set(kw for kw, _ in keywords)
    all_keywords.update(date_keywords)
```

**Purpose**: Collect all unique keywords across all dates

**Process:**
1. For each date, extract top N keywords using TF-IDF
2. Add keywords to global set
3. Sort keywords alphabetically
4. Log total unique keyword count

**Pass 2: Frequency Counting**
```python
for date, texts in texts_by_date.items():
    combined_text = ' '.join(texts)
    freq_dict = {}
    for keyword in all_keywords:
        freq_dict[keyword] = combined_text.count(keyword)
```

**Purpose**: Count how many times each keyword appears on each date

**Process:**
1. Combine all texts for a date
2. For each keyword in vocabulary, count occurrences
3. Store frequency dictionary
4. Save to CSV if `save_separately=True`

**Why Two Passes?**
- Ensures all keywords are tracked across all dates
- Prevents missing keywords that appear on some dates but not others
- Creates consistent time series data structure

### Sub-Phase 3.4: Frequency Matrix Construction

**Output Storage**:
- Individual CSV files per date: `keyword_freq_YYYYMMDD.csv`
- Each file contains keyword-frequency pairs for that date

**Example CSV Structure:**
```
Keyword,Frequency
반도체 공급망,45
2차전지 시장,32
전기차 배터리,28
```

### Mathematical Foundation

**TF-IDF Formula:**

TF-IDF(t, d) = TF(t, d) × IDF(t)

Where:
- **TF(t, d)** = Term Frequency: How often term t appears in document d
  ```
  TF(t, d) = (Number of times term t appears in document d) / (Total terms in document d)
  ```

- **IDF(t)** = Inverse Document Frequency: How rare the term is across all documents
  ```
  IDF(t) = log(Total documents / Documents containing term t)
  ```

**Intuition:**
- High TF: Term appears frequently in a document (indicates relevance)
- High IDF: Term is rare across documents (indicates distinctiveness)
- High TF-IDF: Term is both frequent and distinctive (indicates importance)

**Example:**

Consider three documents:
- Doc 1: "삼성 반도체 공장 증설"
- Doc 2: "SK 반도체 신규 투자"
- Doc 3: "LG 디스플레이 공장 가동"

For term "반도체":
- TF(반도체, Doc1) = 1/4 = 0.25
- TF(반도체, Doc2) = 1/4 = 0.25
- IDF(반도체) = log(3/2) = 0.176
- TF-IDF(반도체, Doc1) = 0.25 × 0.176 = 0.044

For term "삼성":
- TF(삼성, Doc1) = 1/4 = 0.25
- TF(삼성, Doc2) = 0
- IDF(삼성) = log(3/1) = 0.477
- TF-IDF(삼성, Doc1) = 0.25 × 0.477 = 0.119

"삼성" gets higher TF-IDF score because it's more distinctive (appears in fewer documents).

---

## Phase 4: Time Series Analysis

### Overview

Once keyword frequencies are extracted, the system analyzes temporal patterns to understand how keyword usage changes over time. This phase detects trends, cycles, volatility, and statistical patterns.

### Implementation Details

**File:** `src/time_series_analyzer.py`

#### TimeSeriesAnalyzer Class

```python
class TimeSeriesAnalyzer:
    def __init__(self, window_size: int = 7)
```

**Parameters:**
- `window_size`: Window size for moving average calculation (default: 7 days)

### Sub-Phase 4.1: Building Frequency DataFrame

**Method:** `_build_frequency_dataframe()` in MemoryEfficientFactorLLM.py (Lines 287-304)

**Purpose**: Convert keyword frequency dictionaries into a structured DataFrame

**Process:**

1. **Load Frequency Data**:
   ```python
   keyword_freq_by_date = self._load_keyword_freq_by_date(
       target_date=target_date,
       date_range=52  # 52 weeks = 1 year
   )
   ```

2. **Build DataFrame Rows**:
   ```python
   rows = []
   for date, freq_dict in keyword_freq_by_date.items():
       row = {'date': pd.to_datetime(date)}
       row.update(freq_dict)
       rows.append(row)
   ```

3. **Create and Index DataFrame**:
   ```python
   df = pd.DataFrame(rows)
   df = df.set_index('date')
   df = df.sort_index()
   ```

4. **Remove Zero Columns**:
   ```python
   df = df.loc[:, (df != 0).any(axis=0)]
   ```

**Output Structure:**
```
                반도체 공급망  2차전지 시장  전기차 배터리
date
2024-10-01         45         32         28
2024-10-02         52         35         24
2024-10-03         48         38         31
...
```

### Sub-Phase 4.2: Moving Average Calculation

**Method:** `calculate_moving_average()` (Lines 29-39)

**Purpose**: Smooth out short-term fluctuations and highlight longer-term trends

**Algorithm:**
```python
def calculate_moving_average(self, series: pd.Series) -> pd.Series:
    return series.rolling(window=self.window_size, min_periods=1).mean()
```

**Example:**

Original series: [10, 12, 15, 11, 13, 16, 14]
7-day MA: [10, 11, 12.33, 12, 12.2, 12.83, 13]

**Interpretation:**
- Reduces noise from daily fluctuations
- Reveals underlying trend direction
- Helps identify genuine patterns vs random variation

### Sub-Phase 4.3: Trend Detection

**Method:** `detect_trend()` (Lines 41-92)

**Purpose**: Determine if keyword usage is increasing, decreasing, or stable

**Algorithm:**

1. **Data Preparation**:
   ```python
   clean_series = series.dropna()
   x = np.arange(len(clean_series))  # Time points
   y = clean_series.values            # Frequencies
   ```

2. **Linear Regression**:
   ```python
   slope, intercept, r_value, p_value, std_err = linregress(x, y)
   ```

   Fits a line: y = slope × x + intercept

3. **Statistical Significance Check**:
   ```python
   if p_value < 0.05:  # 95% confidence level
       if slope > 0.001:
           trend = 'increasing'
       elif slope < 0:
           trend = 'decreasing'
       else:
           trend = 'stable'
   else:
       trend = 'no_significant_trend'
   ```

4. **Return Trend Information**:
   ```python
   return {
       'trend': trend,                # Direction
       'slope': slope,                # Rate of change
       'r_squared': r_value ** 2,     # Goodness of fit
       'p_value': p_value             # Statistical significance
   }
   ```

**Interpretation:**
- **slope > 0**: Keyword usage is increasing over time
- **slope < 0**: Keyword usage is decreasing over time
- **r_squared**: How well the trend line fits the data (0-1)
- **p_value < 0.05**: Trend is statistically significant

**Example:**

For keyword "AI 반도체":
```python
{
    'trend': 'increasing',
    'slope': 0.8,           # Gaining ~0.8 mentions per day
    'r_squared': 0.72,      # 72% of variance explained by trend
    'p_value': 0.001        # Highly significant
}
```

### Sub-Phase 4.4: Cycle Detection

**Method:** `detect_cycles()` (Lines 94-155)

**Purpose**: Identify repeating patterns (e.g., weekly cycles, monthly cycles)

**Algorithm:**

1. **Detrend Data**:
   ```python
   detrended = signal.detrend(clean_series.values)
   ```
   Remove linear trend to focus on cyclical patterns

2. **Fast Fourier Transform (FFT)**:
   ```python
   fft = np.fft.fft(detrended)
   frequencies = np.fft.fftfreq(len(detrended))
   ```

   Decomposes signal into frequency components

3. **Power Spectrum**:
   ```python
   power = np.abs(fft) ** 2
   ```

   Measures strength of each frequency

4. **Find Dominant Frequency**:
   ```python
   positive_freqs = frequencies[1:len(frequencies)//2]
   positive_power = power[1:len(power)//2]
   max_power_idx = np.argmax(positive_power)
   dominant_freq = positive_freqs[max_power_idx]
   ```

5. **Calculate Period**:
   ```python
   period = 1 / dominant_freq
   strength = positive_power[max_power_idx] / positive_power.sum()
   has_cycle = strength > 0.1 and period >= min_period
   ```

**Output:**
```python
{
    'has_cycle': True,
    'period': 7.2,      # Repeats every ~7 days (weekly pattern)
    'strength': 0.25    # 25% of signal variance from this cycle
}
```

**Interpretation:**
- **period = 7**: Weekly cycle (e.g., news published on specific weekdays)
- **period = 30**: Monthly cycle (e.g., end-of-month reports)
- **strength > 0.1**: Cycle is significant and not just noise

### Sub-Phase 4.5: Volatility Calculation

**Method:** `calculate_volatility()` (Lines 157-168)

**Purpose**: Measure how unstable keyword frequency is

**Algorithm:**
```python
def calculate_volatility(self, series: pd.Series) -> float:
    changes = series.diff().dropna()  # Day-to-day changes
    return changes.std()               # Standard deviation
```

**Interpretation:**
- **Low volatility**: Keyword appears consistently (stable topic)
- **High volatility**: Keyword appears sporadically (event-driven topic)

**Example:**
- "삼성전자" (stable company): Low volatility
- "태풍" (weather event): High volatility

### Sub-Phase 4.6: Comprehensive Keyword Analysis

**Method:** `analyze_keyword_series()` (Lines 207-255)

**Purpose**: Run all analyses for a single keyword

**Process:**
```python
def analyze_keyword_series(self, series: pd.Series, keyword: str) -> Dict:
    analysis = {
        'keyword': keyword,
        'data_points': len(series),
        'total_occurrences': series.sum(),
        'mean_frequency': series.mean(),
        'max_frequency': series.max(),
        'min_frequency': series.min()
    }

    # Moving average
    ma = self.calculate_moving_average(series)
    analysis['moving_average'] = ma.tolist()

    # Trend detection
    trend_info = self.detect_trend(series)
    analysis.update({
        'trend': trend_info['trend'],
        'trend_slope': trend_info['slope'],
        'trend_r_squared': trend_info['r_squared'],
        'trend_p_value': trend_info['p_value']
    })

    # Cycle detection
    cycle_info = self.detect_cycles(series)
    analysis.update({
        'has_cycle': cycle_info['has_cycle'],
        'cycle_period': cycle_info['period'],
        'cycle_strength': cycle_info['strength']
    })

    # Volatility
    analysis['volatility'] = self.calculate_volatility(series)

    return analysis
```

**Output Example:**
```python
{
    'keyword': 'AI 반도체',
    'data_points': 365,
    'total_occurrences': 1250,
    'mean_frequency': 3.42,
    'max_frequency': 12,
    'min_frequency': 0,
    'trend': 'increasing',
    'trend_slope': 0.015,
    'trend_r_squared': 0.68,
    'trend_p_value': 0.002,
    'has_cycle': True,
    'cycle_period': 7.1,
    'cycle_strength': 0.18,
    'volatility': 2.34
}
```

### Sub-Phase 4.7: Batch Analysis

**Method:** `analyze_all_keywords()` (Lines 257-284)

**Purpose**: Analyze all keywords in the frequency DataFrame

**Process:**
1. Loop through all keyword columns
2. Call `analyze_keyword_series()` for each
3. Filter out keywords with no significant trend
4. Sort by total occurrences
5. Return comprehensive DataFrame

**From MemoryEfficientFactorLLM.py:143-149:**
```python
analysis_df = self.time_series_analyzer.analyze_all_keywords(freq_df)
```

### Keyword Classification

**Method:** `classify_keyword_status()` (Lines 286-319)

**Purpose**: Classify keywords by lifecycle status

**Classification Logic:**
```python
if trend == 'increasing':
    if mean_freq > 1:
        return 'dominant_rising'     # Already popular and growing
    else:
        return 'emerging'            # New topic gaining attention
elif trend == 'decreasing':
    if mean_freq < 1:
        return 'declining_fast'      # Rapidly fading
    else:
        return 'declining'           # Losing attention
elif trend == 'stable':
    if mean_freq > 1:
        return 'dominant_stable'     # Established topic
    else:
        return 'stable'              # Consistent but niche
else:
    if volatility > mean_freq * 0.5:
        return 'volatile'            # Unpredictable
    else:
        return 'uncertain'           # Unclear pattern
```

---

## Phase 5: Lifecycle Prediction

### Overview

Based on the time series analysis, the system predicts the future trajectory of each keyword. This includes forecasting future frequencies and calculating extinction probabilities.

### Implementation Details

**File:** `src/prediction_model.py`

#### KeywordLifecyclePredictor Class

```python
class KeywordLifecyclePredictor:
    def __init__(self, prediction_horizon_days: int = 365)
```

**Parameters:**
- `prediction_horizon_days`: How far into the future to predict (default: 365 days)

### Sub-Phase 5.1: Future Frequency Prediction

**Method:** `predict_future_frequency()` (Lines 95-124)

**Purpose**: Predict keyword frequency at a future time point

**Algorithm:**

1. **Linear Projection**:
   ```python
   current_freq = series.iloc[-1]
   predicted = current_freq + (trend_slope * days_ahead)
   predicted = max(0, predicted)  # Cannot be negative
   ```

2. **Confidence Interval (95%)**:
   ```python
   std_dev = series.std()
   margin = 1.96 * std_dev  # 95% confidence = 1.96 standard deviations
   lower_bound = max(0, predicted - margin)
   upper_bound = predicted + margin
   ```

3. **Return Prediction**:
   ```python
   return (predicted, lower_bound, upper_bound)
   ```

**Example:**

For "AI 반도체" with:
- Current frequency: 5.0
- Trend slope: 0.02
- Standard deviation: 1.5

Prediction for 30 days ahead:
- Predicted: 5.0 + (0.02 × 30) = 5.6
- Lower bound: 5.6 - (1.96 × 1.5) = 2.66
- Upper bound: 5.6 + (1.96 × 1.5) = 8.54

**Interpretation:**
- Expected frequency: 5.6 mentions per day
- 95% confidence: between 2.66 and 8.54 mentions per day

### Sub-Phase 5.2: Extinction Probability Calculation

**Method:** `calculate_extinction_probability()` (Lines 27-93)

**Purpose**: Calculate probability that keyword usage will drop to near-zero

**Algorithm:**

1. **Check for Already Extinct**:
   ```python
   if current_freq == 0 or mean_freq == 0:
       return {'30_days': 1.0, '90_days': 1.0, '180_days': 1.0, '365_days': 1.0}
   ```

2. **Calculate Decay Rate**:
   ```python
   if trend_slope < 0:
       # Declining trend - higher extinction probability
       decay_rate = abs(trend_slope) / mean_freq
   else:
       # Stable or rising - lower extinction probability
       decay_rate = 0.001  # minimal decay
   ```

3. **Adjust for Volatility**:
   ```python
   volatility_factor = 1 + (volatility / mean_freq)
   ```

   Higher volatility increases uncertainty and extinction risk

4. **Exponential Decay Model**:
   ```python
   for days in [30, 90, 180, 365]:
       expected_freq = current_freq * np.exp(-decay_rate * days * volatility_factor)
   ```

5. **Normal Distribution Probability**:
   ```python
   if volatility > 0:
       z_score = (1 - expected_freq) / (volatility * np.sqrt(days / 30))
       prob = norm.cdf(z_score)
   else:
       prob = 1.0 if expected_freq < 1 else 0.0
   ```

6. **Confidence Assessment**:
   ```python
   if len(series) >= 7 and volatility < mean_freq:
       confidence = 'high'
   elif len(series) >= 3:
       confidence = 'medium'
   else:
       confidence = 'low'
   ```

**Mathematical Model:**

The extinction probability uses an exponential decay model:

```
f(t) = f₀ × e^(-λt)
```

Where:
- f(t) = frequency at time t
- f₀ = current frequency
- λ = decay rate × volatility factor
- t = time in days

The probability that frequency drops below threshold (1.0) is calculated using the cumulative distribution function (CDF) of the normal distribution:

```
P(extinction) = Φ((threshold - E[f(t)]) / σ(t))
```

Where:
- Φ = CDF of standard normal distribution
- E[f(t)] = expected frequency at time t
- σ(t) = standard deviation at time t
- threshold = 1.0 (minimum viable frequency)

**Example Output:**
```python
{
    '30_days': 0.05,    # 5% chance of extinction in 1 month
    '90_days': 0.18,    # 18% chance in 3 months
    '180_days': 0.42,   # 42% chance in 6 months
    '365_days': 0.71,   # 71% chance in 1 year
    'confidence': 'high'
}
```

### Sub-Phase 5.3: Lifecycle Stage Estimation

**Method:** `estimate_lifecycle_stage()` (Lines 126-176)

**Purpose**: Classify keyword into product lifecycle stages

**Lifecycle Stages:**

1. **Introduction**:
   - Trend: Increasing
   - Mean frequency: < 1
   - Description: "Emerging topic, gaining attention"
   - Example: New technology just announced

2. **Growth**:
   - Trend: Increasing
   - Mean frequency: ≥ 1
   - Description: "Rapidly growing topic, high momentum"
   - Example: Viral trend, major industry shift

3. **Maturity**:
   - Trend: Stable
   - Mean frequency: ≥ 1
   - Description: "Established topic, stable presence"
   - Example: Established company or product

4. **Decline**:
   - Trend: Decreasing
   - Mean frequency: ≥ 1
   - Description: "Declining topic, losing attention"
   - Example: Fading technology

5. **Obsolescence**:
   - Trend: Decreasing
   - Mean frequency: < 1
   - Description: "Fading topic, nearing extinction"
   - Example: Obsolete technology

6. **Uncertain**:
   - Trend: No significant trend
   - Description: "Insufficient data or unclear pattern"

**Algorithm:**
```python
if trend == 'increasing' and mean_freq < 1:
    stage = 'introduction'
elif trend == 'increasing' and mean_freq >= 1:
    stage = 'growth'
elif trend == 'stable' and mean_freq >= 1:
    stage = 'maturity'
elif trend == 'decreasing' and mean_freq >= 1:
    stage = 'decline'
elif trend == 'decreasing' and mean_freq < 1:
    stage = 'obsolescence'
else:
    stage = 'uncertain'
```

**Days Remaining Estimate:**
```python
if trend == 'decreasing' and trend_slope < 0:
    current_freq = max_frequency
    slope = trend_slope
    days_remaining = int(-current_freq / slope)  # When frequency reaches 0
    days_remaining = max(0, min(days_remaining, 999))
else:
    days_remaining = None
```

**Example:**
```python
{
    'stage': 'decline',
    'description': 'Declining topic, losing attention',
    'days_remaining_estimate': 145,  # ~5 months until extinction
    'confidence': 'medium'
}
```

### Sub-Phase 5.4: Complete Lifecycle Prediction

**Method:** `predict_keyword_lifecycle()` (Lines 178-223)

**Purpose**: Generate comprehensive prediction for a keyword

**Process:**
```python
def predict_keyword_lifecycle(self, series: pd.Series, analysis: Dict) -> Dict:
    # Extract parameters
    keyword = analysis['keyword']
    trend_slope = analysis['trend_slope']
    volatility = analysis['volatility']

    # Calculate extinction probabilities
    extinction_prob = self.calculate_extinction_probability(
        series, trend_slope, volatility
    )

    # Predict future frequencies
    predictions = {}
    for days in [30, 90, 180, 365]:
        pred, lower, upper = self.predict_future_frequency(
            series, trend_slope, days
        )
        predictions[f'{days}_days'] = {
            'predicted': pred,
            'lower_bound': lower,
            'upper_bound': upper
        }

    # Estimate lifecycle stage
    lifecycle = self.estimate_lifecycle_stage(analysis)

    # Combine all predictions
    return {
        'keyword': keyword,
        'current_frequency': series.iloc[-1],
        'trend': analysis['trend'],
        'lifecycle_stage': lifecycle['stage'],
        'lifecycle_description': lifecycle['description'],
        'extinction_probabilities': extinction_prob,
        'frequency_predictions': predictions,
        'days_remaining_estimate': lifecycle['days_remaining_estimate']
    }
```

**Example Complete Prediction:**
```python
{
    'keyword': 'AI 반도체',
    'current_frequency': 5.2,
    'trend': 'increasing',
    'lifecycle_stage': 'growth',
    'lifecycle_description': 'Rapidly growing topic, high momentum',
    'extinction_probabilities': {
        '30_days': 0.01,
        '90_days': 0.03,
        '180_days': 0.08,
        '365_days': 0.15,
        'confidence': 'high'
    },
    'frequency_predictions': {
        '30_days': {'predicted': 6.4, 'lower_bound': 3.5, 'upper_bound': 9.3},
        '90_days': {'predicted': 8.8, 'lower_bound': 4.2, 'upper_bound': 13.4},
        '180_days': {'predicted': 12.6, 'lower_bound': 5.1, 'upper_bound': 20.1},
        '365_days': {'predicted': 19.2, 'lower_bound': 6.8, 'upper_bound': 31.6}
    },
    'days_remaining_estimate': None  # Not declining
}
```

### Sub-Phase 5.5: Batch Prediction

**Method:** `predict_all_keywords()` (Lines 225-267)

**Purpose**: Generate predictions for all keywords

**Process:**
1. Loop through all keywords in frequency DataFrame
2. Match with corresponding analysis data
3. Call `predict_keyword_lifecycle()` for each
4. Filter out non-significant trends
5. Flatten nested dictionaries for easier access
6. Return comprehensive predictions DataFrame

**From MemoryEfficientFactorLLM.py:151-158:**
```python
predictions_df = self.predictor.predict_all_keywords(freq_df, analysis_df)
keywords = self.predictor.get_keywords(predictions_df)
```

### Sub-Phase 5.6: Keyword Filtering

**Method:** `get_keywords()` (Lines 269-287)

**Purpose**: Filter for promising keywords (excluding nearly extinct ones)

**Algorithm:**
```python
keywords = pred_df[
    (pred_df['extinction_prob_365_days'] < 0.99)  # Less than 99% extinction probability
].copy()

keywords = keywords.sort_values('current_frequency', ascending=False)
```

**Rationale**: Focus on keywords that have future potential rather than those almost certain to disappear

---

## Phase 6: LLM-based Interpretation

### Overview

While statistical analysis provides quantitative insights, Large Language Models (LLMs) add qualitative interpretation and business context. This phase uses Claude or GPT to generate human-readable explanations of the data patterns.

### Implementation Details

**File:** `src/llm_analyzer.py`

#### LLMAnalyzer Class

```python
class LLMAnalyzer:
    def __init__(self, provider: str = "anthropic",
                 model: str = "claude-3-5-sonnet-20241022",
                 api_key: str = None)
```

**Supported Providers:**
- **Anthropic**: Claude models (claude-3-5-sonnet, claude-3-opus)
- **OpenAI**: GPT models (gpt-4, gpt-5-mini)

### Sub-Phase 6.1: Client Initialization

**Method:** `_initialize_client()` (Lines 35-53)

**Process:**
```python
if self.provider == "anthropic":
    from anthropic import Anthropic
    self.client = Anthropic(api_key=self.api_key)
elif self.provider == "openai":
    from openai import OpenAI
    self.client = OpenAI(api_key=self.api_key)
```

### Sub-Phase 6.2: Keyword Interpretation Generation

**Method:** `generate_keyword_interpretation()` (Lines 55-109)

**Purpose**: Generate detailed interpretations for multiple keywords at once

**Input Structure:**
```python
input_dict = {
    'AI 반도체': {
        'analysis': {
            'trend': 'increasing',
            'mean_frequency': 5.2,
            'volatility': 1.8,
            'total_occurrences': 1890,
            'trend_slope': 0.023
        },
        'prediction': {
            'lifecycle_stage': 'growth',
            'extinction_prob_365_days': 0.15,
            'current_frequency': 5.8
        }
    },
    '2차전지 시장': { ... },
    ...
}
```

**Pydantic Output Schema:**
```python
class Keyword(BaseModel):
    keyword: str
    interpretation: str

class OutputFormat(BaseModel):
    declining_keyword_list: List[Keyword]
    dominant_keyword_list: List[Keyword]
    rising_keyword_list: List[Keyword]
```

**LLM Prompt (Korean):**
```python
prompt = f"""주어진 데이터는 각 키워드에 대한 정보를 담은 Dict 데이터입니다.
모든 데이터를 상승 추세, 안정 추세, 하락 추세로 나누고,
키워드에 대한 분석 결과를 바탕으로 해석과 통찰을 제공해주세요.

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
```

**API Call:**

For Anthropic:
```python
response = self.client.messages.create(
    model=self.model,
    messages=[{"role": "user", "content": prompt}]
)
return response.content[0].text
```

For OpenAI:
```python
response = self.client.responses.parse(
    model=self.model,
    input=[{"role": "user", "content": prompt}],
    text_format=OutputFormat
)
return response.output_parsed
```

**Example Output:**
```python
OutputFormat(
    rising_keyword_list=[
        Keyword(
            keyword='AI 반도체',
            interpretation='AI 반도체는 빠르게 성장하고 있으며 평균 5.2회의 언급으로 높은 관심을 받고 있습니다.
                          글로벌 AI 기술 경쟁이 심화되면서 반도체 수요가 급증하고 있으며,
                          향후 1년간 지속적인 성장이 예상됩니다.'
        ),
        ...
    ],
    dominant_keyword_list=[
        Keyword(
            keyword='삼성전자',
            interpretation='삼성전자는 안정적으로 높은 빈도를 유지하며 시장의 주요 화제입니다.
                          국내 대표 기업으로서 지속적인 관심을 받고 있으며,
                          성숙 단계에 있어 안정적인 패턴을 보이고 있습니다.'
        ),
        ...
    ],
    declining_keyword_list=[
        Keyword(
            keyword='코로나19',
            interpretation='코로나19 관련 키워드는 하락 추세를 보이며 관심이 감소하고 있습니다.
                          팬데믹 상황이 안정화되면서 뉴스 빈도가 줄어들고 있으며,
                          1년 내 소멸 확률이 71%로 높게 나타났습니다.'
        ),
        ...
    ]
)
```

### Sub-Phase 6.3: News Article Fetching

**Method:** `fetch_relevant_news()` (Lines 315-349)

**Purpose**: Fetch recent news articles related to keywords for additional context

**Process:**

1. **Create Search Prompt**:
   ```python
   prompt = f"""다음 키워드와 날짜에 대한 관련 뉴스 기사를 통해
                각각의 키워드와 어떤 산업(반도체, 2차전지 등)이 연관이 있는지
                연관 산업을 제시해주세요.
                키워드: {', '.join(keywords)}
                날짜: {target_date}"""
   ```

2. **Call LLM with Web Search**:

   For Anthropic:
   ```python
   response = self.client.messages.create(
       model=self.model,
       messages=[{"role": "user", "content": prompt}]
   )
   ```

   For OpenAI (with web search tool):
   ```python
   response = self.client.responses.create(
       model=self.model,
       tools=[{"type": "web_search"}],
       input=prompt
   )
   ```

3. **Extract News Summary**:
   ```python
   news_articles = response.output_text
   ```

**Example Output:**
```
AI 반도체: 반도체 산업, 특히 NVIDIA, AMD, 삼성전자의 GPU 및 AI 전용 칩 개발과 관련
2차전지: 전기차 및 에너지 저장 산업, LG화학, 삼성SDI, SK온의 배터리 기술 발전
자율주행: 자동차 산업, 현대차, 테슬라의 자율주행 기술 개발
```

### Sub-Phase 6.4: Overall Analysis Generation

**Method:** `generate_overall_analysis()` (Lines 160-222)

**Purpose**: Generate comprehensive industry trend analysis

**Input Structure:**
```python
interpretation: OutputFormat  # From generate_keyword_interpretation()
news: str                     # From fetch_relevant_news()
```

**Pydantic Output Schema:**
```python
class KeywordInterpret(BaseModel):
    keyword: str
    interpretation: str

class IndustryTrend(BaseModel):
    industry: str
    keywords: List[KeywordInterpret]

class OutputFormat(BaseModel):
    trend_industry: List[IndustryTrend]      # Main trending industries
    rising_industry: List[IndustryTrend]     # Emerging industries
```

**Prompt Construction:**
```python
# Build interpretation string
interpretation_str = 'Dominant Keywords Interpretations:\n\n'
for k in interpretation.dominant_keyword_list:
    interpretation_str += f"Keyword: {k.keyword}, Interpretation: {k.interpretation} \n\n"

interpretation_str += '\nRising Keywords Interpretations:\n\n'
for k in interpretation.rising_keyword_list:
    interpretation_str += f"Keyword: {k.keyword}, Interpretation: {k.interpretation} \n\n"

interpretation_str += '\nDeclining Keywords Interpretations:\n\n'
for k in interpretation.declining_keyword_list:
    interpretation_str += f"Keyword: {k.keyword}, Interpretation: {k.interpretation} \n\n"

# Create analysis prompt
prompt = f"""다음은 키워드 분석과 관련 뉴스 데이터를 바탕으로
             전체 시장 및 트렌드에 대한 종합적인 분석입니다.
             키워드 해석: {interpretation_str}
             관련 뉴스: {news}

             위 정보를 바탕으로 어떤 산업(반도체, 화학, 2차전지 등 주식 연관)에
             유효한 트렌드가 형성되고 있는지, 주요 트렌드, 신규 부상 트렌드,
             향후 전망 등을 포함하여 종합적인 분석을 작성해주세요.
             각 분류별로 산업은 GICS 산업으로 3개씩 제시해주세요.
             한국어로 작성해주세요."""
```

**Example Output:**
```python
OutputFormat(
    trend_industry=[
        IndustryTrend(
            industry='반도체 (Semiconductors)',
            keywords=[
                KeywordInterpret(
                    keyword='AI 반도체',
                    interpretation='생성형 AI 수요 증가로 고성능 GPU 및 AI 칩 시장이 급성장 중'
                ),
                KeywordInterpret(
                    keyword='반도체 공급망',
                    interpretation='지정학적 리스크로 인한 공급망 재편이 진행 중'
                ),
                KeywordInterpret(
                    keyword='파운드리',
                    interpretation='첨단 공정 경쟁 심화, TSMC와 삼성전자의 3nm 경쟁'
                )
            ]
        ),
        IndustryTrend(
            industry='2차전지 (Battery)',
            keywords=[
                KeywordInterpret(
                    keyword='전기차 배터리',
                    interpretation='전기차 시장 성장으로 배터리 수요 지속 증가'
                ),
                ...
            ]
        ),
        IndustryTrend(
            industry='자동차 (Automobile)',
            keywords=[...]
        )
    ],
    rising_industry=[
        IndustryTrend(
            industry='양자컴퓨팅 (Quantum Computing)',
            keywords=[...]
        ),
        IndustryTrend(
            industry='바이오제약 (Biopharmaceuticals)',
            keywords=[...]
        ),
        IndustryTrend(
            industry='우주항공 (Aerospace)',
            keywords=[...]
        )
    ]
)
```

### Sub-Phase 6.5: Integration in Main Pipeline

**From MemoryEfficientFactorLLM.py:161-169:**

```python
def _generate_interpretations_selective(self, keywords_df, analysis_df, target_date):
    """Generate interpretations for selected keywords only"""

    # Prepare input dictionary
    input_dict = {}
    for _, pred_row in keywords_df.iterrows():
        keyword = pred_row['keyword']
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

    # Generate overall analysis
    overall = self.llm_analyzer.generate_overall_analysis(interpretation, news)

    # Combine results
    interpretations = {
        'overall': overall,
        'interpretations': interpretation,
        'news': news
    }

    return interpretations
```

**Why LLM Integration?**

1. **Contextual Understanding**: LLMs understand industry context and market dynamics
2. **Natural Language Output**: Converts statistics into readable explanations
3. **Industry Mapping**: Links keywords to specific GICS industries
4. **Causal Reasoning**: Explains WHY trends are happening
5. **Forward-Looking**: Provides actionable insights and forecasts

**Cost Optimization:**

- Process only top 30 keywords (line 164): `keywords.nlargest(30, 'current_frequency')`
- Batch keywords together in single API call rather than individual calls
- Use structured output (Pydantic) to ensure consistent formatting

---

## Phase 7: Report Generation

### Overview

The final phase compiles all analysis results into a comprehensive, readable markdown report. The report includes statistics, tables, classifications, and interpretations organized for easy consumption.

### Implementation Details

**File:** `src/report_generator.py`

#### ReportGenerator Class

```python
class ReportGenerator:
    def __init__(self, output_dir: Path)
```

**Purpose**: Generate structured markdown reports from analysis data

### Sub-Phase 7.1: Report Data Preparation

**Method:** `_prepare_report_data()` in MemoryEfficientFactorLLM.py (Lines 352-428)

**Purpose**: Structure all analysis results into report-ready format

**Process:**

1. **Calculate Statistics**:
   ```python
   statistics = {
       'total_articles': int(freq_df.sum().sum()),
       'date_range': {
           'start': freq_df.index.min().strftime('%Y-%m-%d'),
           'end': freq_df.index.max().strftime('%Y-%m-%d')
       },
       'unique_dates': len(freq_df.index.unique()),
       'unique_sources': 1  # Update if tracking sources
   }
   ```

2. **Extract Keyword Lists**:
   ```python
   # Dominant keywords
   dominant_keywords = interpretations.get('interpretations', {}).dominant_keyword_list

   # Rising keywords
   rising_keywords = interpretations.get('interpretations', {}).rising_keyword_list

   # Declining keywords
   declining_keywords = interpretations.get('interpretations', {}).declining_keyword_list
   ```

3. **Create Classification DataFrame**:
   ```python
   key_list = []

   for kw in declining_keywords:
       key_list.append(['declining', kw.keyword, kw.interpretation])

   for kw in rising_keywords:
       key_list.append(['rising', kw.keyword, kw.interpretation])

   for kw in dominant_keywords:
       key_list.append(['dominant', kw.keyword, kw.interpretation])

   key_df = pd.DataFrame(key_list, columns=['status', 'keyword', 'interpretation'])
   ```

4. **Merge All Data**:
   ```python
   merged_data = pd.merge(key_df, analysis_df, on='keyword', how='left')
   merged_data = pd.merge(merged_data, predictions_df, on='keyword', how='left')
   ```

5. **Extract Overall Analysis**:
   ```python
   overall_analysis = interpretations.get('overall', '')
   ```

6. **Assemble Report Data**:
   ```python
   report_data = {
       'statistics': statistics,
       'keywords_data': merged_data,
       'overall_analysis': overall_analysis
   }
   ```

### Sub-Phase 7.2: Markdown Generation

**Method:** `generate_markdown_report()` (Lines 28-53)

**Process:**

1. **Generate Filename**:
   ```python
   if filename is None:
       timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
       filename = f"FACTOR-LLM_Report_{timestamp}.md"
   ```

2. **Build Content**:
   ```python
   content = self._build_markdown_content(data)
   ```

3. **Write to File**:
   ```python
   with open(report_path, 'w', encoding='utf-8') as f:
       f.write(content)
   ```

**Method:** `_build_markdown_content()` (Lines 55-102)

**Purpose**: Orchestrate all report sections

**Structure:**
```python
sections = []

# 1. Header
sections.append(self._generate_header())

# 2. Executive Summary
sections.append(self._generate_executive_summary(data))

# 3. Industry Trends (from LLM analysis)
if 'overall_analysis' in data:
    sections.append(self._generate_industry_trends_section(data))

# 4. Data Overview
sections.append(self._generate_data_overview(data))

# 5. Top Keywords
sections.append(self._generate_top_keywords_section(data))

# 6. Emerging Keywords
sections.append(self._generate_emerging_keywords_section(data))

# 7. Declining Keywords
sections.append(self._generate_declining_keywords_section(data))

# 8. Detailed Analysis
sections.append(self._generate_detailed_analysis_section(data))

# 9. Methodology
sections.append(self._generate_methodology_section())

# 10. Footer
sections.append(self._generate_footer())

return '\n\n'.join(sections)
```

### Sub-Phase 7.3: Report Sections

#### 1. Header Section

**Method:** `_generate_header()` (Lines 104-111)

```markdown
# FACTOR-LLM Analysis Report
## Factor Analysis and Cyclical Trend Observation using LLMs

**Generated:** 2024-11-06 14:23:45

---
```

#### 2. Executive Summary Section

**Method:** `_generate_executive_summary()` (Lines 113-133)

```markdown
## Executive Summary

이 보고서는 1,234,567개의 뉴스 기사를 분석하여 산업 트렌드와 키워드 라이프사이클을 예측합니다.

### 주요 발견사항

- **분석 기간**: 2024-01-01 ~ 2024-11-06
- **총 분석 기사 수**: 1,234,567개
- **주요 키워드**: AI 반도체, 2차전지 시장, 전기차 배터리, 자율주행, 메타버스
- **신규 부상 키워드**: 양자컴퓨팅, 그린수소, 차세대 배터리, UAM, 디지털 트윈
- **쇠퇴 중인 키워드**: 코로나19, 비대면, 재택근무, 메타버스 플랫폼, NFT
```

#### 3. Industry Trends Section

**Method:** `_generate_industry_trends_section()` (Lines 135-194)

```markdown
## 산업 트렌드 종합

### 주요 트렌드 산업 (Trend Industries)

- **반도체 (Semiconductors)**
  - **AI 반도체**: 생성형 AI 수요 증가로 고성능 GPU 및 AI 칩 시장이 급성장 중
  - **반도체 공급망**: 지정학적 리스크로 인한 공급망 재편이 진행 중
  - **파운드리**: 첨단 공정 경쟁 심화, TSMC와 삼성전자의 3nm 경쟁

- **2차전지 (Battery)**
  - **전기차 배터리**: 전기차 시장 성장으로 배터리 수요 지속 증가
  - **배터리 소재**: 니켈, 리튬 등 핵심 소재 확보 경쟁
  - **전고체 배터리**: 차세대 배터리 기술 개발 경쟁

### 신규 부상 산업 (Rising Industries)

- **양자컴퓨팅 (Quantum Computing)**
  - **양자 프로세서**: IBM, Google의 양자컴퓨터 상용화 진전
  - **양자 암호**: 보안 강화를 위한 양자 암호 기술 개발
```

#### 4. Data Overview Section

**Method:** `_generate_data_overview()` (Lines 196-206)

```markdown
## 데이터 개요

- **총 기사 수**: 1,234,567
- **분석 일자 수**: 310일
- **데이터 소스 수**: 3
- **추출된 키워드 수**: 847
```

#### 5. Top Keywords Table

**Method:** `_generate_top_keywords_section()` (Lines 208-254)

```markdown
## 주요 키워드

현재 가장 주도적인 키워드들입니다.

| 순위 | 키워드 | 출현 빈도 | 트렌드 | 라이프사이클 | 소멸 확률(1개월) | (3개월) | (6개월) | (1년) |
|------|--------|-----------|--------|--------------|---------------|--------------|--------------|---------|
| 1 | AI 반도체 | 5.8 | 상승 | 성장기 | 1.2% | 3.5% | 8.1% | 15.3% |
| 2 | 2차전지 시장 | 4.9 | 상승 | 성장기 | 2.1% | 5.8% | 12.4% | 23.7% |
| 3 | 전기차 배터리 | 4.2 | 안정 | 성숙기 | 3.5% | 8.9% | 18.2% | 32.1% |
...
```

**Process:**
1. Filter keywords with status='dominant'
2. Take top 10 by current frequency
3. Format as markdown table
4. Translate trend and lifecycle to Korean
5. Format extinction probabilities as percentages

#### 6. Emerging Keywords Table

**Method:** `_generate_emerging_keywords_section()` (Lines 256-302)

Similar to top keywords, but filters for status='rising'

```markdown
## 신규 부상 키워드

새롭게 주목받고 있는 키워드들입니다.

| 순위 | 키워드 | 출현 빈도 | 트렌드 | 라이프사이클 | 소멸 확률(1개월) | (3개월) | (6개월) | (1년) |
|------|--------|-----------|--------|--------------|---------------|--------------|--------------|---------|
| 1 | 양자컴퓨팅 | 1.2 | 상승 | 도입기 | 5.3% | 12.1% | 24.5% | 41.2% |
...
```

#### 7. Declining Keywords Table

**Method:** `_generate_declining_keywords_section()` (Lines 304-350)

Similar structure, but filters for status='declining'

```markdown
## 쇠퇴 중인 키워드

관심이 감소하고 있는 키워드들입니다.

| 순위 | 키워드 | 출현 빈도 | 트렌드 | 라이프사이클 | 소멸 확률(1개월) | (3개월) | (6개월) | (1년) |
|------|--------|-----------|--------|--------------|---------------|--------------|--------------|---------|
| 1 | 코로나19 | 1.8 | 하락 | 쇠퇴기 | 15.2% | 38.7% | 64.3% | 85.1% |
...
```

#### 8. Detailed Analysis Section

**Method:** `_generate_detailed_analysis_section()` (Lines 376-398)

**Purpose**: Show LLM-generated interpretations for rising keywords

```markdown
## 부상 키워드 상세 분석

### 양자컴퓨팅

**해석**: 양자컴퓨팅은 초기 도입 단계에 있으며 빠르게 성장하고 있습니다.
IBM과 Google의 기술 발전으로 상용화 가능성이 높아지고 있으며,
금융, 제약, 보안 등 다양한 분야에서 활용이 기대됩니다.
향후 3-5년 내 본격적인 시장 형성이 예상됩니다.

### 그린수소

**해석**: 그린수소는 탄소중립 목표 달성을 위한 핵심 에너지원으로 주목받고 있습니다.
정부의 정책 지원과 대기업들의 투자 확대로 인프라 구축이 가속화되고 있으며,
2030년까지 수소경제 생태계가 본격 형성될 것으로 전망됩니다.
```

#### 9. Methodology Section

**Method:** `_generate_methodology_section()` (Lines 416-446)

**Purpose**: Explain the analysis methodology for transparency

```markdown
## 방법론

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
```

#### 10. Footer Section

**Method:** `_generate_footer()` (Lines 448-456)

```markdown
---

**FACTOR-LLM**
*Factor Analysis and Cyclical Trend Observation using LLMs*

본 보고서는 자동 생성되었으며, 참고 목적으로만 사용되어야 합니다.
```

### Sub-Phase 7.4: Report Output

**From MemoryEfficientFactorLLM.py:180-181:**

```python
report_path = self.report_generator.generate_markdown_report(report_data)
logger.info(f"Report generated: {report_path}")
```

**Output Location:**
- Directory: `output/`
- Filename: `FACTOR-LLM_Report_YYYYMMDD_HHMMSS.md`

**Report Features:**
- **Structured Format**: Clear sections with headers
- **Tables**: Easy-to-read keyword tables with metrics
- **Bilingual**: Korean for business users, English for technical terms
- **Comprehensive**: Combines quantitative metrics with qualitative insights
- **Actionable**: Provides industry trends and forward-looking analysis

---

## Mathematical Models and Algorithms

### TF-IDF (Term Frequency-Inverse Document Frequency)

**Purpose**: Identify important keywords that are both frequent and distinctive

**Formula:**

```
TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
```

**Term Frequency (TF):**
```
TF(t, d) = f(t, d) / Σ f(w, d)
```
Where:
- f(t, d) = frequency of term t in document d
- Σ f(w, d) = total terms in document d

**Inverse Document Frequency (IDF):**
```
IDF(t, D) = log(|D| / |{d ∈ D : t ∈ d}|)
```
Where:
- |D| = total number of documents
- |{d ∈ D : t ∈ d}| = number of documents containing term t

**Implementation:** sklearn.feature_extraction.text.TfidfVectorizer

**Parameters:**
- max_features=1000: Limit vocabulary size
- min_df=2: Ignore terms in fewer than 2 documents
- max_df=0.8: Ignore terms in more than 80% of documents
- ngram_range=(2,3): Extract bigrams and trigrams

### Linear Regression for Trend Detection

**Purpose**: Determine if keyword frequency is increasing, decreasing, or stable

**Model:**
```
y = β₀ + β₁x + ε
```
Where:
- y = keyword frequency
- x = time index
- β₀ = intercept
- β₁ = slope (trend)
- ε = error term

**Implementation:** scipy.stats.linregress

**Interpretation:**
- β₁ > 0: Increasing trend
- β₁ < 0: Decreasing trend
- β₁ ≈ 0: Stable trend

**Statistical Significance:**
- p-value < 0.05: Trend is statistically significant
- R² > 0.5: Trend explains >50% of variance

**Goodness of Fit:**
```
R² = 1 - (SS_res / SS_tot)
```
Where:
- SS_res = Σ(yᵢ - ŷᵢ)² (residual sum of squares)
- SS_tot = Σ(yᵢ - ȳ)² (total sum of squares)

### Moving Average

**Purpose**: Smooth short-term fluctuations to reveal trends

**Simple Moving Average (SMA):**
```
SMA(t) = (1/n) Σ yᵢ   for i = t-n+1 to t
```
Where:
- n = window size (default: 7 days)
- yᵢ = frequency at time i

**Implementation:** pandas.Series.rolling()

**Effect:**
- Reduces noise from daily fluctuations
- Preserves overall trend direction
- Lag effect: Slower to respond to sudden changes

### Fast Fourier Transform (FFT) for Cycle Detection

**Purpose**: Identify periodic patterns in keyword frequency

**Algorithm:**
1. **Detrend**: Remove linear trend
   ```
   y_detrended = y - (β₀ + β₁x)
   ```

2. **FFT**: Transform to frequency domain
   ```
   Y(f) = Σ y(t) e^(-i2πft)
   ```

3. **Power Spectrum**: Measure strength of each frequency
   ```
   P(f) = |Y(f)|²
   ```

4. **Find Dominant Frequency**:
   ```
   f_dominant = argmax(P(f))   for f > 0
   ```

5. **Calculate Period**:
   ```
   T = 1 / f_dominant
   ```

**Implementation:** scipy.fft.fft

**Interpretation:**
- T ≈ 7: Weekly cycle (news published on specific weekdays)
- T ≈ 30: Monthly cycle (end-of-month reports)
- T ≈ 365: Annual cycle (seasonal events)

**Cycle Strength:**
```
strength = P(f_dominant) / Σ P(f)
```
- strength > 0.1: Significant cycle
- strength < 0.1: Weak or no cycle

### Volatility Calculation

**Purpose**: Measure stability of keyword frequency

**Standard Deviation of Changes:**
```
σ_changes = sqrt((1/(n-1)) Σ (Δyᵢ - μ_Δy)²)
```
Where:
- Δyᵢ = yᵢ - yᵢ₋₁ (day-to-day change)
- μ_Δy = mean of changes

**Implementation:** pandas.Series.diff().std()

**Interpretation:**
- Low volatility (σ < μ): Stable, consistent topic
- High volatility (σ > μ): Sporadic, event-driven topic

### Exponential Decay Model

**Purpose**: Model keyword frequency decay over time

**Formula:**
```
f(t) = f₀ × e^(-λt)
```
Where:
- f(t) = frequency at time t
- f₀ = current frequency
- λ = decay rate
- t = time in days

**Decay Rate Calculation:**
```
λ = |slope| / mean_frequency   if slope < 0
λ = 0.001                       if slope ≥ 0
```

**Volatility Adjustment:**
```
λ_adj = λ × (1 + volatility / mean_frequency)
```

**Implementation:** MemoryEfficientFactorLLM.py (Lines 54-70)

**Interpretation:**
- High λ: Rapid decay (topic fading quickly)
- Low λ: Slow decay (topic stable)

### Extinction Probability Model

**Purpose**: Predict probability that keyword frequency drops below threshold

**Model:** Assumes normal distribution of future frequency

**Expected Frequency:**
```
E[f(t)] = f₀ × e^(-λt)
```

**Standard Deviation:**
```
σ(t) = volatility × sqrt(t / 30)
```

**Z-Score:**
```
z = (threshold - E[f(t)]) / σ(t)
```
Where threshold = 1.0 (minimum viable frequency)

**Extinction Probability:**
```
P(extinction) = Φ(z)
```
Where Φ is the CDF of standard normal distribution

**Implementation:** scipy.stats.norm.cdf()

**Interpretation:**
- P > 0.7: High extinction risk
- 0.3 < P < 0.7: Moderate risk
- P < 0.3: Low extinction risk

### Confidence Intervals for Predictions

**Purpose**: Quantify uncertainty in future frequency predictions

**Point Prediction:**
```
ŷ(t) = y₀ + slope × t
```

**95% Confidence Interval:**
```
CI = ŷ(t) ± 1.96 × σ
```
Where:
- σ = standard deviation of historical frequencies
- 1.96 = z-score for 95% confidence

**Implementation:** MemoryEfficientFactorLLM.py (Lines 117-124)

**Interpretation:**
- Narrow CI: High confidence in prediction
- Wide CI: High uncertainty

---

## Memory Efficiency Techniques

### Challenge

Processing millions of news articles requires efficient memory management to avoid:
- Out-of-memory errors
- System slowdowns
- Long processing times

### Solution Strategies

#### 1. Chunked Data Loading

**Technique:** Load and process data in small chunks rather than loading entire dataset

**Implementation:** MemoryEfficientLoader.iter_chunks()

```python
for chunk in self.loader.iter_chunks():
    # Process chunk
    process(chunk)

    # Delete chunk
    del chunk

    # Force garbage collection
    if config.AGGRESSIVE_GC:
        gc.collect()
```

**Benefits:**
- Memory usage remains constant regardless of dataset size
- Can process datasets larger than available RAM
- Prevents memory leaks from accumulating

**Configuration:**
- CHUNK_SIZE: Number of rows per chunk (default: 1000)
- PROCESSING_MODE: 'chunked', 'full', or 'sample'

#### 2. Streaming Processing

**Technique:** Process data as it's loaded without storing intermediate results

**Implementation:** MemoryEfficientFactorLLM._preprocess_in_chunks()

```python
preprocessed_by_date = defaultdict(list)

for chunk in self.loader.iter_chunks():
    for _, row in chunk.iterrows():
        # Process row immediately
        text = preprocess(row)

        # Store only processed result
        preprocessed_by_date[date].append(text)

    # Discard raw chunk
    del chunk
```

**Benefits:**
- Reduces peak memory usage
- Eliminates need to store raw data
- Enables pipeline parallelization

#### 3. Incremental Aggregation

**Technique:** Build results incrementally rather than in memory

**Implementation:** MemoryEfficientFactorLLM._extract_keywords_incremental()

**Two-Pass Approach:**

**Pass 1: Vocabulary Collection**
```python
all_keywords = set()
for date, texts in texts_by_date.items():
    keywords = extract_keywords(texts)
    all_keywords.update(keywords)
    # Don't store frequencies yet
```

**Pass 2: Frequency Counting**
```python
for date, texts in texts_by_date.items():
    freq_dict = count_frequencies(texts, all_keywords)
    # Save immediately to disk
    save_to_csv(freq_dict, date)
    # Don't keep in memory
```

**Benefits:**
- Separates vocabulary construction from frequency counting
- Allows disk-based storage of intermediate results
- Enables date-range filtering without reprocessing

#### 4. Explicit Garbage Collection

**Technique:** Force Python garbage collector to run after heavy operations

**Implementation:**
```python
import gc

# After processing chunk
del large_dataframe
gc.collect()  # Force garbage collection
```

**Configuration:**
- AGGRESSIVE_GC: Enable/disable forced collection
- Trade-off: Collection takes time but frees memory

**When to Use:**
- After processing large DataFrames
- After LLM inference
- Between major processing stages

#### 5. DataFrame Optimization

**Technique:** Reduce DataFrame memory footprint using optimal data types

**Implementation:** MemoryOptimizer.optimize_dataframe()

```python
def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        col_type = df[col].dtype

        if col_type == 'float64':
            df[col] = df[col].astype('float32')  # Half the memory
        elif col_type == 'int64':
            df[col] = df[col].astype('int32')    # Half the memory
        elif col_type == 'object':
            # Try converting to category if few unique values
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')

    return df
```

**Benefits:**
- Reduces memory by 50-75% for numeric columns
- Categorical encoding reduces string overhead
- No loss of precision for most use cases

**Configuration:**
- OPTIMIZE_DATAFRAMES: Enable/disable optimization

#### 6. Disk-Based Intermediate Storage

**Technique:** Save intermediate results to disk instead of keeping in memory

**Implementation:**
```python
# Save keyword frequencies per date
for date in dates:
    freq_dict = compute_frequencies(date)
    pd.Series(freq_dict).to_csv(f'keyword_freq_{date}.csv')

# Later: Load only required date range
def load_date_range(start, end):
    for date in pd.date_range(start, end):
        yield pd.read_csv(f'keyword_freq_{date}.csv')
```

**Benefits:**
- Supports analysis of arbitrary date ranges
- Enables checkpoint/resume functionality
- Reduces reprocessing if analysis fails

**Trade-off:**
- Disk I/O is slower than memory access
- Requires sufficient disk space

#### 7. Selective Processing

**Technique:** Process only necessary data for specific analyses

**Implementation:**

**Column Selection:**
```python
# Load only required columns
chunk = pd.read_csv(file, usecols=['date', 'title', 'content'])
```

**Date Range Filtering:**
```python
# Process only recent data
loader.load_by_date_range('2024-10-01', '2024-11-06')
```

**Top-N Filtering:**
```python
# Analyze only top 30 keywords
keywords = keywords.nlargest(30, 'current_frequency')
```

**Benefits:**
- Reduces processing time
- Lowers memory requirements
- Focuses computation on relevant data

#### 8. Memory Monitoring

**Technique:** Track memory usage to detect leaks and optimize allocation

**Implementation:** MemoryMonitor class

```python
class MemoryMonitor:
    def __init__(self, threshold_mb=1000):
        self.threshold_mb = threshold_mb
        self.measurements = []

    def check_memory(self, label=""):
        current_mb = psutil.Process().memory_info().rss / 1024 / 1024
        self.measurements.append((label, current_mb))

        if current_mb > self.threshold_mb:
            logger.warning(f"Memory usage high: {current_mb:.1f} MB at {label}")

    def get_summary(self):
        return {
            'max_mb': max(m[1] for m in self.measurements),
            'avg_mb': sum(m[1] for m in self.measurements) / len(self.measurements)
        }
```

**Usage:**
```python
self.memory_monitor.check_memory("After preprocessing")
self.memory_monitor.check_memory("After keyword extraction")
```

**Benefits:**
- Identifies memory-intensive operations
- Detects memory leaks early
- Guides optimization efforts

### Memory Usage Summary

**From MemoryEfficientFactorLLM.py:183-189:**
```python
memory_summary = self.memory_monitor.get_summary()
logger.info(f"Peak memory: {memory_summary['max_mb']:.1f} MB")
logger.info(f"Average memory: {memory_summary['avg_mb']:.1f} MB")
```

**Typical Memory Profile:**
- Initialization: 200-300 MB
- Text preprocessing: 500-800 MB (per chunk)
- Keyword extraction: 400-600 MB
- Time series analysis: 300-500 MB
- LLM inference: 2000-4000 MB (GPU memory)
- Report generation: 100-200 MB

**Total Peak Memory:** ~4-5 GB (including GPU)

**Scalability:**
- Can process datasets with 10M+ articles
- Memory usage independent of dataset size (due to chunking)
- Limited only by disk space for intermediate files

---

## Configuration Parameters

### File Locations

**config.py** (Lines 7-11)

```python
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "news_data_by_date"
OUTPUT_DIR = BASE_DIR / "output"
```

- **BASE_DIR**: Project root directory
- **DATA_DIR**: Directory containing CSV files
- **OUTPUT_DIR**: Directory for reports and intermediate files

### Data Settings

**config.py** (Lines 13-18)

```python
CSVS = ["data_1", "data_2", "data_4"]
BATCH_SIZE = 8
```

- **CSVS**: List of CSV file names to process (without .csv extension)
- **BATCH_SIZE**: Batch size for LLM keyword extraction

### Text Processing Settings

**config.py** (Lines 20-26)

```python
MIN_KEYWORD_LENGTH = 2
MAX_KEYWORD_LENGTH = 20
TOP_N_KEYWORDS = 50
```

- **MIN_KEYWORD_LENGTH**: Minimum characters for valid keyword
- **MAX_KEYWORD_LENGTH**: Maximum characters for valid keyword
- **TOP_N_KEYWORDS**: Number of top keywords to extract per date

**Rationale:**
- MIN_KEYWORD_LENGTH=2: Filter single-character noise
- TOP_N_KEYWORDS=50: Balance between coverage and processing time

### Time Series Settings

**config.py** (Lines 28-31)

```python
WINDOW_SIZE = 7
THRESHOLD_EMERGENCE = 0.7
THRESHOLD_DECLINE = 0.3
```

- **WINDOW_SIZE**: Moving average window in days
- **THRESHOLD_EMERGENCE**: Threshold for emerging trend classification
- **THRESHOLD_DECLINE**: Threshold for declining trend classification

**Interpretation:**
- WINDOW_SIZE=7: One-week smoothing (balances noise reduction and responsiveness)
- THRESHOLD_EMERGENCE=0.7: Keyword must show 70%+ growth to be "emerging"
- THRESHOLD_DECLINE=0.3: Keyword must show 70%+ decline to be "declining"

### Prediction Settings

**config.py** (Lines 33-35)

```python
PREDICTION_HORIZON_DAYS = 365
CONFIDENCE_LEVEL = 0.95
```

- **PREDICTION_HORIZON_DAYS**: How far ahead to predict (1 year)
- **CONFIDENCE_LEVEL**: Confidence level for prediction intervals (95%)

**Horizons:**
- 30 days: Short-term (1 month)
- 90 days: Medium-term (3 months)
- 180 days: Long-term (6 months)
- 365 days: Very long-term (1 year)

### LLM Settings

**config.py** (Lines 37-41)

```python
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-5-mini-2025-08-07"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 4000
```

- **LLM_PROVIDER**: "anthropic" or "openai"
- **LLM_MODEL**: Specific model name
- **LLM_TEMPERATURE**: Sampling temperature (0-1)
- **LLM_MAX_TOKENS**: Maximum output length

**Model Options:**
- Anthropic: claude-3-5-sonnet-20241022, claude-3-opus-20240229
- OpenAI: gpt-4, gpt-5-mini-2025-08-07

**Temperature:**
- 0.0: Deterministic (always same output)
- 0.7: Balanced creativity and consistency
- 1.0: Maximum creativity

### API Keys

**config.py** (Lines 43-49)

```python
import configparser
config = configparser.ConfigParser()
config.read('D:/config.ini')

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = config.get('openai', 'API_KEY')
```

**Security Note:** API keys stored externally, not in code

### Memory Efficiency Settings

**config.py** (Lines 55-61)

```python
USE_MEMORY_EFFICIENT_MODE = True
CHUNK_SIZE = 1000
AUTO_CONFIGURE_MEMORY = True
MEMORY_THRESHOLD_MB = 1000
OPTIMIZE_DATAFRAMES = True
```

- **USE_MEMORY_EFFICIENT_MODE**: Enable chunked processing
- **CHUNK_SIZE**: Rows per chunk
- **AUTO_CONFIGURE_MEMORY**: Automatically adjust chunk size
- **MEMORY_THRESHOLD_MB**: Warning threshold for memory usage
- **OPTIMIZE_DATAFRAMES**: Enable DataFrame optimization

**Tuning Chunk Size:**
- Smaller chunks: Lower memory, slower processing
- Larger chunks: Higher memory, faster processing
- Recommended: 500-2000 based on available RAM

### Processing Mode

**config.py** (Lines 63-64)

```python
PROCESSING_MODE = "chunked"
SAMPLE_SIZE = 10000
```

- **PROCESSING_MODE**:
  - "chunked": Process data in chunks (recommended for large datasets)
  - "full": Load all data at once (only for small datasets)
  - "sample": Random sample for testing
- **SAMPLE_SIZE**: Number of rows if using sample mode

### Garbage Collection

**config.py** (Lines 66-67)

```python
AGGRESSIVE_GC = True
```

- **AGGRESSIVE_GC**: Force garbage collection after each operation
- **Trade-off**: Frees memory but adds 10-20% processing time

**Recommendation:**
- Enable for production (prevents memory leaks)
- Disable for development (faster iteration)

---

## Use Cases and Applications

### 1. Investment Strategy Development

**Scenario:** Portfolio manager wants to identify emerging technology trends for investment

**Process:**
1. **Run Keyword Extraction** on financial news data
2. **Identify Rising Keywords** with low extinction probability
3. **Map Keywords to Industries** using LLM analysis
4. **Prioritize Industries** with multiple rising keywords
5. **Research Specific Companies** in prioritized industries

**Example Output:**
```
Rising Industries:
- AI Semiconductors: nvidia, tsmc, samsung_electronics
- Quantum Computing: ibm, google, ionq
- Green Hydrogen: plug_power, nel_asa, ballard_power
```

**Action:** Increase allocation to semiconductor and quantum computing stocks

### 2. Market Trend Monitoring

**Scenario:** Analyst needs to track market sentiment changes over time

**Process:**
1. **Run Daily Analysis** on latest news
2. **Compare with Historical Trends** from past week/month
3. **Identify Emerging Topics** crossing threshold
4. **Alert on Declining Topics** with high extinction probability
5. **Generate Weekly Report** summarizing changes

**Example Alert:**
```
ALERT: "Supply Chain Disruption" keyword declining rapidly
- Current frequency: 3.2 → 1.8 (44% drop in 7 days)
- Extinction probability (90 days): 72%
- Recommendation: Monitor for supply chain recovery
```

### 3. Competitive Intelligence

**Scenario:** Company wants to monitor competitor mentions and market positioning

**Process:**
1. **Define Competitor Keywords** (company names, products)
2. **Track Frequency Over Time** for each competitor
3. **Analyze Trend Direction** (increasing/stable/decreasing)
4. **Compare Against Own Company** to gauge relative attention
5. **Identify Emerging Competitors** with rising mentions

**Example Dashboard:**
```
Competitor Mentions (30-day trend):
- Samsung Electronics: 125 → 145 (+16%) ↑
- TSMC: 98 → 112 (+14%) ↑
- Intel: 87 → 76 (-13%) ↓
- Our Company: 45 → 52 (+16%) ↑

Insight: TSMC and Samsung gaining market attention, Intel declining
```

### 4. Industry Research

**Scenario:** Research team needs to understand technology adoption cycles

**Process:**
1. **Extract Keywords** from industry publications
2. **Classify Lifecycle Stages** (introduction, growth, maturity, decline)
3. **Identify Technologies** in different lifecycle stages
4. **Predict Future Trajectories** using extinction probabilities
5. **Generate Industry Report** with LLM insights

**Example Report Section:**
```
Technology Lifecycle Analysis:

Introduction Stage:
- Quantum Computing: High growth potential, 5-10 year horizon
- UAM (Urban Air Mobility): Early adoption, regulatory pending

Growth Stage:
- Electric Vehicles: Rapid expansion, mainstream adoption beginning
- AI Semiconductors: High demand, supply constraints

Maturity Stage:
- Cloud Computing: Established market, stable growth
- Smartphones: Saturated market, incremental improvements

Decline Stage:
- 4G Networks: Being replaced by 5G
- Traditional Combustion Engines: Shift to electric
```

### 5. Risk Management

**Scenario:** Risk manager wants to identify emerging threats and risks

**Process:**
1. **Monitor Negative Keywords** (crisis, disruption, shortage, etc.)
2. **Track Extinction Probabilities** for positive keywords
3. **Detect Cyclical Patterns** indicating recurring risks
4. **Generate Risk Alerts** when thresholds crossed
5. **Correlate with Market Events** for validation

**Example Risk Dashboard:**
```
Emerging Risks:
- "Semiconductor Shortage": +45% in 7 days, high volatility
- "Supply Chain Disruption": Recurring 30-day cycle detected
- "Regulatory Changes": Increasing trend in fintech sector

Declining Positive Indicators:
- "Economic Growth": -28% in 30 days
- "Consumer Confidence": 65% extinction probability (6 months)

Recommendation: Review exposure to semiconductor-dependent sectors
```

### 6. Content Strategy Planning

**Scenario:** Marketing team wants to create content aligned with trending topics

**Process:**
1. **Identify Rising Keywords** in target industry
2. **Analyze Lifecycle Stage** to determine timing
3. **Generate Content Calendar** based on keyword trends
4. **Monitor Keyword Performance** after content publication
5. **Optimize Based on Results**

**Example Content Plan:**
```
Q1 2025 Content Strategy:

High Priority (Rising keywords):
- "AI Semiconductors": Create deep-dive technical article
- "Quantum Computing Applications": Beginner-friendly guide
- "Green Hydrogen Economics": ROI analysis whitepaper

Medium Priority (Stable keywords):
- "Cloud Security": Update existing content
- "5G Networks": Quarterly trend report

Low Priority (Declining keywords):
- "COVID-19 Impact": Retire old content
- "Remote Work Tools": Archive or consolidate
```

### 7. Academic Research

**Scenario:** Researcher studying technology diffusion patterns

**Process:**
1. **Collect Historical News Data** (5-10 years)
2. **Run FACTOR-LLM Analysis** for each year
3. **Compare Lifecycle Patterns** across technologies
4. **Identify Common Adoption Curves** and patterns
5. **Publish Findings** on technology lifecycle prediction

**Research Questions:**
- Do all technologies follow similar adoption curves?
- What predicts successful vs failed technology adoption?
- How do media cycles relate to actual market adoption?

### 8. Product Launch Timing

**Scenario:** Company planning product launch wants optimal timing

**Process:**
1. **Monitor Related Keywords** in target market
2. **Identify Market Readiness** through keyword lifecycle stage
3. **Detect Competitor Activity** through rising mentions
4. **Predict Market Saturation** using extinction models
5. **Recommend Launch Window** based on analysis

**Example Analysis:**
```
Product: AI-Powered Analytics Platform

Market Analysis:
- "AI Analytics": Growth stage, high momentum (+35% monthly)
- "Business Intelligence AI": Introduction stage, emerging
- "Traditional BI Tools": Decline stage, being replaced

Competitor Activity:
- Major players launching similar products: 3 in next 6 months
- Market attention: Peak expected in Q2 2025

Recommendation: Launch in Q1 2025 to capture early adopter wave
before market saturation in Q3 2025
```

---

## Conclusion

### Summary

FACTOR-LLM is a comprehensive system for extracting and analyzing keyword trends from large-scale news data. By combining classical NLP techniques (TF-IDF, time series analysis) with modern LLMs (Claude, GPT), the system provides both quantitative metrics and qualitative insights.

### Key Strengths

1. **Scalability**: Memory-efficient design handles millions of articles
2. **Comprehensiveness**: Multi-stage pipeline covers extraction through interpretation
3. **Accuracy**: Statistical rigor with significance testing and confidence intervals
4. **Interpretability**: LLM integration provides business context and actionable insights
5. **Flexibility**: Modular architecture allows customization and extension

### Technical Innovations

1. **Memory Efficiency**: Chunking, streaming, and incremental processing
2. **Two-Pass Extraction**: Ensures consistent keyword tracking across dates
3. **Hybrid Analysis**: Combines statistical models with LLM reasoning
4. **Lifecycle Prediction**: Novel application of decay models to keyword trends
5. **Korean NLP**: Optimized for Korean text with Mecab and Korean LLMs

### Limitations and Future Work

**Current Limitations:**

1. **Language Support**: Optimized for Korean; other languages need adaptation
2. **Linear Trends**: Assumes linear trends; may miss non-linear patterns
3. **Static Configuration**: Parameters require manual tuning
4. **Single Market**: Designed for Korean stock market news
5. **Keyword Granularity**: Focuses on 2-3 word phrases; may miss longer concepts

**Future Enhancements:**

1. **Multi-Language Support**: Extend to English, Chinese, Japanese
2. **Non-Linear Models**: Integrate LSTM/Transformer for better predictions
3. **Auto-Parameter Tuning**: Use optimization algorithms for parameter selection
4. **Real-Time Processing**: Add streaming data ingestion and analysis
5. **Causal Analysis**: Identify cause-effect relationships between keywords
6. **Entity Recognition**: Link keywords to specific companies and products
7. **Sentiment Analysis**: Track positive/negative sentiment alongside frequency
8. **Visual Dashboard**: Interactive web interface for exploring trends

### Practical Applications

FACTOR-LLM is valuable for:

- **Investment Firms**: Identifying emerging investment opportunities
- **Market Researchers**: Understanding industry trends and dynamics
- **Risk Managers**: Detecting emerging threats and vulnerabilities
- **Content Creators**: Aligning content with trending topics
- **Product Managers**: Timing product launches based on market readiness
- **Academics**: Studying technology diffusion and media cycles

### Getting Started

**Basic Usage:**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure settings
edit config.py  # Set data paths, API keys

# 3. Run keyword extraction
python MemoryEfficientFactorLLM.py --mode=keywords

# 4. Run analysis
python MemoryEfficientFactorLLM.py --mode=analysis --target-date=20241106

# 5. View report
cat output/FACTOR-LLM_Report_*.md
```

**Configuration Tips:**

- Start with CHUNK_SIZE=1000 and adjust based on memory
- Enable AGGRESSIVE_GC for production runs
- Use TOP_N_KEYWORDS=30-50 to balance coverage and performance
- Set WINDOW_SIZE=7 for weekly trends, 30 for monthly

### Performance Metrics

**Typical Performance (1M articles):**

- Keyword Extraction: 2-4 hours
- Time Series Analysis: 10-20 minutes
- Lifecycle Prediction: 5-10 minutes
- LLM Interpretation: 2-5 minutes (depends on API)
- Report Generation: <1 minute

**Total Runtime:** 3-5 hours for complete pipeline

**Memory Usage:**
- Peak: 4-5 GB (including GPU)
- Average: 2-3 GB
- Scalable to datasets >10M articles

### Conclusion

FACTOR-LLM represents a powerful approach to automated trend analysis, combining the precision of statistical methods with the interpretability of Large Language Models. Its memory-efficient design enables practical application to real-world datasets, while its modular architecture supports customization and extension.

The system demonstrates that carefully engineered pipelines can extract meaningful insights from massive unstructured text data, providing actionable intelligence for decision-making in finance, marketing, product development, and research.

For questions, contributions, or support, please refer to the project repository and documentation.

---

**End of Report**
