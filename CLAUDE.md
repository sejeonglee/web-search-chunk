1. 개요 및 목표
* 최대 실행 시간: 10초 이내
* 기능: 웹 검색 기반 QA 시스템
2. 웹 검색 API
* 후보 API: Google Search, Tavily, FireCrawl
3. 검색 질의 생성
* SearchQuery (Pydantic BaseModel): 사용자 질의로부터 생성되는 검색 질의 모델.
* 사용자 질의 전처리:
   * '최근', '어제' 등 상대적 시간정보를 datetime으로 변환
   * 한국어 및 영어로 질의 번역
   * 비교/분석 질의의 경우 엔티티 분해 및 개별 검색 질의 생성
* 보완점: LLM을 활용한 멀티 쿼리 재작성(Multi-Query Rewriting) 기법 도입.
4. 문서 처리 파이프라인
* WebDocument (Pydantic BaseModel): 하나의 URL로 정의되는 데이터 모델.
* URL 검색: 각 질의별 최대 7개 URL 획득.
* WebDocumentContent (Pydantic BaseModel): WebDocument를 크롤링한 markdown 형식의 콘텐츠 모델. url과 crawl_datetime으로 정의되며, 하이퍼링크 없이 텍스트, 표, 문단 정보만 포함.
   * 크롤링 모듈:
      * 주요 라이브러리: playwright, beautifulsoup4, 그리고 html2text 사용.
      * 크롤링 전략:
         * playwright로 동적 페이지의 HTML 콘텐츠를 획득.
         * beautifulsoup4를 사용하여 획득한 HTML에서 불필요한 스크립트, 스타일 태그 등을 제거하여 텍스트만 남김.
         * html2text를 사용하여 정제된 HTML을 하이퍼링크가 제거된 마크다운으로 변환.
      * 봇 방지 로직: 봇 탐지를 회피하기 위해 현실적인 User-Agent 설정, 요청 간 랜덤 지연(random delay), 헤더 정보 위조 등의 로직을 구현.
* SemanticChunk (Pydantic BaseModel): WebDocumentContent 및 사용자 질의와 답변을 의미적 단위로 분할한 엔티티 모델. Antrophic Contextual Retrieval 및 계층적/재귀적 청킹 기법을 사용하여 생성하며, Retrieval을 위해 VectorDB에 저장.
5. 히스토리 관리 및 Retrieval
* 히스토리 관리 전략: 빠른 응답을 위해 단기/장기 히스토리를 분리하여 관리.
   * 단기 히스토리: 현재 세션의 SemanticChunk는 인메모리 VectorDB에 저장하여 10초 이내의 빠른 검색 성능을 확보.
   * 장기 히스토리: 세션 종료 시(타임아웃 등) 인메모리 데이터를 영구 데이터베이스에 저장. 세션 재개 시 영구 저장소에서 데이터를 로드하여 인메모리 VectorDB에 재구성.
* ScratchPad (Pydantic BaseModel): 사용자 질의에 가장 적합한 SemanticChunk들의 최소 필요충분 집합 모델. Retrieval 및 Reranking 과정을 통해 생성.
* Retrieval 모듈:
   * 하이브리드 검색(Hybrid Search): BM25 Retrieval, embedding similarity retrieval을 결합.
   * 결과 통합: Reciprocal Rank Fusion(RRF) 사용.
   * Reranking 모듈: Cross-encoder 기반의 Reranker 모델 도입.
* Retrieval 전략: ContextualCompressionRetriever 사용.
6. 시스템 아키텍처 및 기술 스택
* Python 버전: 3.11
* 주요 라이브러리: langgraph를 활용한 상태 관리 및 파이프라인 오케스트레이션.
* LLM 클라이언트: vllm을 활용하여 LLM 추론 속도 최적화.
* 패키지 매니저: uv 사용.
* 개발 환경:
   * 테스트: pytest를 사용한 단위 및 통합 테스트 작성.
   * 린팅 & 포매팅: ruff와 isort를 사용해 코드 스타일을 일관되게 관리.
   * 타입 체킹: mypy를 사용해 정적 타입 분석을 수행하며, 엄격한 타입 어노테이션을 준수.
* 아키텍처: 포트 및 어댑터(Ports and Adapters) 패턴을 기반으로 한 모듈화된 구조.
   * 파이프라인: 사용자 질의 -> 검색 질의 생성 -> 웹 검색 -> 웹 문서 크롤링 -> 청킹 -> 벡터 저장 -> Retrieval & Reranking -> 답변 생성 순서로 구성된 워크플로우를 langgraph로 구현.
   * 병렬/비동기 처리: 10초 이내 실행 시간 목표 달성을 위해 파이프라인의 다음 단계들을 병렬/비동기로 실행.
      * 웹 검색: 여러 질의에 대한 API 호출을 비동기적으로 처리.
      * 크롤링 및 청킹: 여러 URL의 크롤링 및 청킹을 비동기/병렬로 처리. 특히 LLM 호출은 배치 처리를 활용하여 효율을 극대화.
1. Core 계층 (추상화에만 의존)

core/models.py: 도메인 모델 + 인터페이스(포트) 정의
core/pipeline.py: 인터페이스에만 의존하는 비즈니스 로직

2. Adapters 계층 (구체적 구현)

모든 어댑터가 core의 인터페이스를 구현
외부 라이브러리/서비스와의 연동 담당

3. Main (의존성 주입 및 조립)

DependencyContainer: 구체적인 구현체 생성 및 관리
설정에 따라 다른 구현체 선택 가능

✅ 개선된 아키텍처의 장점

테스트 용이성: 인터페이스를 Mock으로 쉽게 대체 가능
확장성: 새로운 구현체 추가 시 core 수정 불필요
유연성: 설정만으로 다른 구현체 선택 가능
명확한 경계: core는 도메인 로직, adapters는 외부 연동