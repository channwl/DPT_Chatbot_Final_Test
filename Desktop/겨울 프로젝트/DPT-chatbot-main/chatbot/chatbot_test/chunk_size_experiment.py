import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import time
import sys
import os
from langchain_openai import OpenAIEmbeddings

# Django 설정
sys.path.append('/Users/gimchan-ul/Desktop/겨울 프로젝트/DPT-chatbot-main')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dpt_project.settings')
import django
django.setup()

from chatbot.utils import RAGSystem
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dpt_project import settings

class RAGExperiment:
    def __init__(self):
        self.test_questions = self.load_test_questions()
        self.results = []
        
    def load_test_questions(self) -> List[Dict]:
        """테스트 질문 로드"""
        df = pd.read_excel('chatbot/최종_Testset.xlsx')
        # 질문 ID 1-20만 사용 (실험용, API 비용 고려)
        test_df = df[(df['질문 ID'] >= 1) & (df['질문 ID'] <= 20)]
        
        questions = []
        for _, row in test_df.iterrows():
            questions.append({
                'id': row['질문 ID'],
                'question': row['질문'],
                'reference': row['정답']
            })
        return questions
    
    def generate_faiss_index_with_chunk_size(self, chunk_size: int):
        """특정 chunk size로 FAISS 인덱스 생성"""
        pdf_dir = "chatbot/data/"
        all_documents = []
        
        pdf_files = [file for file in os.listdir(pdf_dir) if file.endswith(".pdf")]
        
        for file_name in pdf_files:
            from langchain_community.document_loaders import PyMuPDFLoader
            loader = PyMuPDFLoader(os.path.join(pdf_dir, file_name))
            documents = loader.load()
            for d in documents:
                d.metadata['file_path'] = os.path.join(pdf_dir, file_name)
            all_documents.extend(documents)
        
        # 새로운 chunk size로 문서 분할
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_size // 8  # chunk_size의 1/8로 overlap 설정
        )
        chunks = splitter.split_documents(all_documents)
        
        # FAISS 인덱스 생성
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key='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'
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(f"faiss_index_chunk_{chunk_size}")
        print(f"✅ Chunk size {chunk_size}로 인덱스 생성 완료!")
        
    def evaluate_retrieval_accuracy(self, chunk_size: int, k: int) -> float:
        """특정 chunk_size와 k 값으로 검색 정확도 평가 (LLM-as-a-Judge 방식, Memory 포함)"""
        
        # RAGSystem 인스턴스 생성 (Memory 포함)
        rag_system = RAGSystem(llm_type="openai")
        
        # 해당 chunk_size의 인덱스로 교체
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=''
        )
        vector_store = FAISS.load_local(
            f"faiss_index_chunk_{chunk_size}", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # RAGSystem의 vector_db를 교체
        rag_system._vector_db = vector_store
        rag_system._retriever = vector_store.as_retriever(search_kwargs={"k": k})
        
        # LLM-as-a-Judge를 위한 OpenAI 클라이언트
        from openai import OpenAI
        client = OpenAI(api_key='')
        
        total_score = 0
        valid_questions = 0
        
        for question_data in self.test_questions:
            try:
                # RAGSystem으로 답변 생성 (Memory 포함)
                answer = rag_system.process_question(question_data['question'])
                
                # LLM-as-a-Judge 평가 프롬프트 (Memory 포함된 답변 평가)
                judge_prompt = f"""
당신은 AI 모델의 응답을 검토하고 정확성을 평가하는 전문가입니다.
당신의 목표는 기준 정답과 비교하여 AI 응답이 사실을 얼마나 정확히 반영했는지를 판단하는 것입니다.

다음은 사용자 질문, 기준 정답, 그리고 AI 모델의 응답입니다.

아래 응답의 정확성을 기준 정답과 비교하여 단계적으로 생각하고, 그 과정은 내부적으로만 수행하세요.
출력은 오직 1~5 사이의 숫자 중 하나만 포함해야 하며, 설명은 출력하지 마세요.

### 사용자 질문:
{question_data['question']}

### 기준 정답:
{question_data['reference']}

### AI 응답 (Memory 포함):
{answer}

[정확성 평가 기준]
- 5점: 기준 정답의 사실을 완벽히 포함하고 있고, 오류 없이 정확함.
- 4점: 거의 모든 사실을 포함하나, 사소한 누락이나 표현상의 부정확성이 있음.
- 3점: 주요 정보는 포함하나 일부 오류나 누락이 있음.
- 2점: 중요한 정보가 부족하거나 명백한 오류가 포함됨.
- 1점: 잘못된 정보 또는 전혀 맞지 않는 사실 포함.

이제 질문을 보고 생각 과정을 거친 뒤, 마지막 줄에 1~5 중 하나의 숫자만 작성하세요.
"""
                
                # LLM 평가 요청
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "당신은 AI 평가자입니다."},
                        {"role": "user", "content": judge_prompt}
                    ],
                    temperature=0
                )
                
                score_text = response.choices[0].message.content.strip()
                
                # 숫자 추출 (1-5 범위)
                try:
                    score = float(score_text)
                    if 1 <= score <= 5:
                        total_score += score
                        valid_questions += 1
                        print(f"✅ {question_data['id']}번 완료 - 점수: {score}")
                    else:
                        print(f"오류 발생 (질문 ID {question_data['id']}): 점수 범위 오류 ({score})")
                except ValueError:
                    print(f"오류 발생 (질문 ID {question_data['id']}): 점수 파싱 오류 ({score_text})")
                
                time.sleep(1.5)  # API 속도 제한 방지
                
            except Exception as e:
                print(f"질문 {question_data['id']} 평가 중 오류: {e}")
                continue
        
        return total_score / valid_questions if valid_questions > 0 else 0
    
    def run_experiment(self):
        """전체 실험 실행"""
        chunk_sizes = [100,300,500,700]
        k_values = [1, 3, 5, 7]
        
        print("🚀 RAG 파라미터 실험 시작!")
        
        for chunk_size in chunk_sizes:
            print(f"\n📊 Chunk size {chunk_size} 실험 중...")
            
            # 해당 chunk_size로 인덱스 생성
            self.generate_faiss_index_with_chunk_size(chunk_size)
            
            for k in k_values:
                print(f"  - k={k} 평가 중...")
                accuracy = self.evaluate_retrieval_accuracy(chunk_size, k)
                
                self.results.append({
                    'chunk_size': chunk_size,
                    'k': k,
                    'accuracy': accuracy
                })
                
                print(f"    결과: {accuracy:.4f}")
        
        # 결과 저장
        results_df = pd.DataFrame(self.results)
        results_df.to_excel('rag_experiment_results.xlsx', index=False)
        print(f"\n✅ 실험 완료! 결과가 'rag_experiment_results.xlsx'에 저장되었습니다.")
        
        return results_df
    
    def plot_results(self, results_df: pd.DataFrame):
        """결과 시각화"""
        plt.figure(figsize=(12, 8))
        
        # k 값별로 다른 선 스타일
        line_styles = ['-', '--', '-.', ':']
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, k in enumerate([1, 3, 5, 7]):
            k_data = results_df[results_df['k'] == k]
            plt.plot(
                k_data['chunk_size'], 
                k_data['accuracy'], 
                marker='o', 
                linestyle=line_styles[i],
                color=colors[i],
                linewidth=2,
                markersize=8,
                label=f'k={k}'
            )
        
        plt.xlabel('Chunk Size', fontsize=14)
        plt.ylabel('검색 정확도', fontsize=14)
        plt.title('RAG 시스템: Chunk Size와 k 값에 따른 검색 정확도', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 그래프 저장
        plt.savefig('rag_experiment_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📈 그래프가 'rag_experiment_results.png'로 저장되었습니다.")

def main():
    """메인 실행 함수"""
    experiment = RAGExperiment()
    
    # 실험 실행
    results = experiment.run_experiment()
    
    # 결과 출력
    print("\n📋 실험 결과 요약:")
    print(results.pivot_table(index='chunk_size', columns='k', values='accuracy'))
    
    # 그래프 생성
    experiment.plot_results(results)

if __name__ == "__main__":
    main() 
