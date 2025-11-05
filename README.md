# FACTOR-LLM

**Factor Analysis and Cyclical Trend Observation using LLMs**

FACTOR-LLM은 뉴스 데이터의 시계열 분석을 통해 산업 트렌드의 부상 및 쇠퇴, 잠재적 경기 순환 패턴을 식별하고, LLM을 활용하여 해석 가능한 리포트를 생성하는 시스템입니다.

## 주요 기능

- 📰 **뉴스 데이터 분석**: CSV 형식의 대량 뉴스 데이터 처리
- 🔍 **키워드 추출**: TF-IDF 기반 중요 키워드 자동 추출
- 📊 **시계열 분석**: 트렌드 감지, 주기 패턴 분석
- 🔮 **예측 모델**: 키워드 라이프사이클 및 소멸 확률 예측
- 🤖 **LLM 통합**: Claude/GPT를 활용한 해석 가능한 분석
- 📈 **종합 리포트**: Markdown 형식의 상세 분석 리포트 생성

## 설치

### 필수 요구사항

- Python 3.8 이상
- Mecab (한국어 형태소 분석기)

### 설치 방법

1. 저장소 클론 또는 다운로드

2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

3. Mecab 설치 (Windows):
```bash
# pip를 통한 설치
pip install mecab-python3
pip install mecab-ko
```

## 사용 방법

### 기본 실행

```bash
python main.py
```

### LLM 분석 사용 (API 키 필요)

```bash
python main.py --use-llm --api-key YOUR_API_KEY
```

### 대용량 데이터 처리 (메모리 효율 모드)

데이터가 1GB 이상이거나 RAM이 제한적인 경우:

```bash
# 메모리 효율 모드 (청킹 처리)
python MemoryEfficientFactorLLM.py

# LLM 사용
python MemoryEfficientFactorLLM.py --use-llm --api-key YOUR_API_KEY

# 커스텀 청크 크기
python MemoryEfficientFactorLLM.py --chunk-size 500

# 샘플링 모드 (빠른 테스트)
python MemoryEfficientFactorLLM.py --mode sample
```

**메모리 효율 모드 장점:**
- ✅ 데이터 크기에 관계없이 일정한 메모리 사용
- ✅ RAM보다 큰 데이터셋 처리 가능
- ✅ 자동 메모리 모니터링 및 최적화
- ✅ 3년치 데이터(~2.1M 기사)를 4GB RAM에서 처리 가능

자세한 내용은 [MEMORY_EFFICIENT_GUIDE.md](MEMORY_EFFICIENT_GUIDE.md) 참조

### 환경 변수 설정

`.env` 파일 생성:
```bash
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
```

## 프로젝트 구조

```
AI_Reporter/
├── main.py                 # 메인 애플리케이션
├── config.py              # 설정 파일
├── requirements.txt       # 패키지 의존성
├── README.md             # 프로젝트 문서
├── TODO.md               # 작업 목록
├── Purpose.txt           # 프로젝트 목적
├── src/                  # 소스 코드 모듈
│   ├── __init__.py
│   ├── data_loader.py           # 데이터 로더
│   ├── text_preprocessor.py     # 텍스트 전처리
│   ├── keyword_extractor.py     # 키워드 추출
│   ├── time_series_analyzer.py  # 시계열 분석
│   ├── prediction_model.py      # 예측 모델
│   ├── llm_analyzer.py          # LLM 분석
│   └── report_generator.py      # 리포트 생성
├── news_data_by_date/    # 뉴스 데이터 (CSV)
│   ├── 20251030.csv
│   └── 20251101.csv
└── output/               # 출력 파일 (자동 생성)
    ├── FACTOR-LLM_Report_YYYYMMDD_HHMMSS.md
    ├── predictions.csv
    └── summary_YYYYMMDD_HHMMSS.json
```

## 출력 결과

실행 후 `output/` 디렉토리에 다음 파일들이 생성됩니다:

1. **FACTOR-LLM_Report_[timestamp].md**: 종합 분석 리포트
   - 주요 키워드 분석
   - 신규 부상 키워드
   - 쇠퇴 중인 키워드
   - 고위험 키워드 (높은 소멸 확률)
   - 상세 분석 및 예측 근거

2. **predictions.csv**: 모든 키워드의 예측 결과 데이터

3. **summary_[timestamp].json**: JSON 형식의 요약 데이터

## 분석 방법론

### 1. 데이터 처리
- CSV 뉴스 데이터 로드
- HTML 태그 제거 및 텍스트 정제
- 한국어 형태소 분석 (Mecab)

### 2. 키워드 추출
- TF-IDF 알고리즘 적용
- 날짜별 키워드 빈도 추적

### 3. 시계열 분석
- 이동평균을 통한 트렌드 평활화
- 선형 회귀 기반 트렌드 감지
- FFT를 이용한 주기 패턴 감지

### 4. 예측 모델
- 지수 감쇠 모델 기반 미래 빈도 예측
- 정규분포 가정의 소멸 확률 계산
- 트렌드와 변동성 고려

### 5. LLM 통합
- Claude/GPT를 활용한 해석 생성
- 데이터 패턴에 대한 맥락적 이해 제공

## 설정 커스터마이징

`config.py` 파일에서 다양한 설정을 변경할 수 있습니다:

```python
# 키워드 추출 설정
TOP_N_KEYWORDS = 50        # 추출할 키워드 수

# 시계열 설정
WINDOW_SIZE = 7            # 이동평균 윈도우 (일)

# 예측 설정
PREDICTION_HORIZON_DAYS = 365  # 예측 기간 (일)

# LLM 설정
LLM_PROVIDER = "anthropic"     # "anthropic" 또는 "openai"
LLM_MODEL = "claude-3-5-sonnet-20241022"
```

## 데이터 형식

입력 CSV 파일은 다음 컬럼을 포함해야 합니다:

- `date`: 날짜 (YYYYMMDD 형식)
- `title`: 뉴스 제목
- `content`: 뉴스 본문
- 기타 메타데이터 컬럼 (선택사항)

## 향후 확장

현재 2일치 데이터로 테스트 중이며, 향후 약 3년간의 일간 데이터(~1,095일)로 확장 예정입니다.

## 라이선스

이 프로젝트는 연구 및 교육 목적으로 사용됩니다.

## 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.
