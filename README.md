# DPT_Chatbot_Final_Test

## 프로젝트 개요
**DPT Chatbot** – 디지털경영전공 학부생을 위한 맞춤형 정보 제공 챗봇 테스트

## 폴더 위치
Chatbot -> chatbot_test

### 주요 기능
- **RAG(Retrieval-Augmented Generation) 기반 검색**
  - 전공 관련 PDF 문서에서 학사일정, 졸업요건, 교환학생 정보 등을 검색
  - LangChain 기반 문서 Chunking + FAISS 벡터 검색
- **실험 기능**
  - 각각 LLM의 Test : llm_evaluate.ipynb
  - 다양한 chunk size, top-k 설정에 따른 RAG 성능 실험 : chunk_size_experiment
  - LLM 응답의 표현력, 정확성, 적합성 평가 : evaluate_적합성,정확성,표현력

---

## 🧪 실험 및 테스트

### 1. 모델 응답 평가 (`evaluate_표현력`, `evaluate_정확성`, `evaluate_적합성`)
- **목적**: 기존 챗봇의 응답 품질을 **표현력**, **정확성**, **적합성** 기준으로 측정
- **내용**:
  - 사전에 준비된 테스트 질문 + 정답 데이터를 바탕으로 LLM 응답을 평가 (LLM-as-a-judge : 모델은 GPt-4.1)
  - 기준별 점수(1~5점)를 부여해 결과를 수집

---

### 2. Chunk Size / k 값 실험 (`chunk_size_experiment.py`)
- **목적**: RAG 검색 성능에 영향을 주는 **chunk size**와 **Top-k** 개수의 최적값 찾기
- **내용**:
  - chunk size: 100, 300, 400, 500, 700 등 다양한 크기로 실험
  - k 값: 1, 3, 5, 7 설정 후 검색 정확도 비교
  - 성능 비교 그래프 및 결과 데이터 기록
  - 
---

### 3. 다양한 LLM 테스트
  - 목적 : 챗봇을 다양한 LLM으로 적용했을때 어떤 성능이 나오는지 Test
  - 내용 : 사전 준비된 테스트 질문 + 정답 데이터를 바탕으로 llm-as-a-judge로 테스트

---
