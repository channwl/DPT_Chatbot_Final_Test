import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import time
import sys
import os

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

class MemoryExperiment:
    def __init__(self):
        self.test_questions = self.load_test_questions()
        self.results = []
        
    def load_test_questions(self) -> List[Dict]:
        """테스트 질문 로드"""
        df = pd.read_excel('chatbot/최종_Testset.xlsx')
        # 질문 ID 1-30만 사용 (실험용)
        test_df = df[(df['질문 ID'] >= 1) & (df['질문 ID'] <= 30)]
        
        questions = []
        for _, row in test_df.iterrows():
            questions.append({
                'id': row['질문 ID'],
                'question': row['질문'],
                'reference': row['정답']
            })
        return questions
    
    def evaluate_with_memory(self, question_data: Dict) -> str:
        """Memory를 포함한 답변 생성 (원래 RAGSystem 그대로 사용)"""
        # RAGSystem 인스턴스 생성 (Memory 포함)
        rag_system = RAGSystem(llm_type="openai")
        
        # 원래 process_question 메서드 사용 (Memory 포함)
        answer = rag_system.process_question(question_data['question'])
        return answer
    
    def evaluate_without_memory(self, question_data: Dict) -> str:
        """Memory 없이 답변 생성 (원래 RAGSystem 구조에서 Memory만 제거)"""
        # RAGSystem 인스턴스 생성
        rag_system = RAGSystem(llm_type="openai")
        
        # 원래 process_question과 동일한 구조이지만 Memory 제거
        try:
            vector_db = rag_system.get_vector_db()
            retriever = vector_db.as_retriever(search_kwargs={"k": 7})
            docs = retriever.invoke(question_data['question'])
            
            # Memory 대신 빈 문자열 사용 (맥락 제거)
            history_summary = ""  # ← 여기가 핵심! Memory 대신 빈 문자열

            answer = rag_system.rag_chain.invoke({
                "question": question_data['question'],
                "context": docs,
                "history": history_summary  # ← 빈 맥락 전달
            })

            # Memory 저장하지 않음 (맥락 누적 안함)
            # self.memory.save_context({"input": question}, {"output": answer}) ← 이 부분 제거
            
            return answer

        except Exception as e:
            print("오류 발생!")
            print(f"오류 종류: {type(e).__name__}")
            print(f"오류 메시지: {e}")
            return "질문 처리 중 오류가 발생했습니다."
    
    def evaluate_answer_accuracy(self, question_data: Dict, answer: str) -> float:
        """LLM-as-a-Judge로 답변 정확도 평가"""
        from openai import OpenAI
        client = OpenAI(api_key='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')
        
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

### AI 응답:
{answer}

[정확성 평가 기준]
- 5점: 기준 정답의 사실을 완벽히 포함하고 있고, 오류 없이 정확함.
- 4점: 거의 모든 사실을 포함하나, 사소한 누락이나 표현상의 부정확성이 있음.
- 3점: 주요 정보는 포함하나 일부 오류나 누락이 있음.
- 2점: 중요한 정보가 부족하거나 명백한 오류가 포함됨.
- 1점: 잘못된 정보 또는 전혀 맞지 않는 사실 포함.

이제 질문을 보고 생각 과정을 거친 뒤, 마지막 줄에 1~5 중 하나의 숫자만 작성하세요.
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 AI 평가자입니다."},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0
        )
        
        score_text = response.choices[0].message.content.strip()
        
        try:
            score = float(score_text)
            if 1 <= score <= 5:
                return score
            else:
                print(f"점수 범위 오류: {score}")
                return 0
        except ValueError:
            print(f"점수 파싱 오류: {score_text}")
            return 0
    
    def run_experiment(self):
        """Memory 유무 실험 실행"""
        print("🚀 Memory 유무 실험 시작!")
        
        memory_scores = []
        no_memory_scores = []
        
        for i, question_data in enumerate(self.test_questions):
            print(f"\n📝 질문 {question_data['id']} 평가 중...")
            
            try:
                # Memory 포함 답변 생성 및 평가
                print("  - Memory 포함 답변 생성 중...")
                memory_answer = self.evaluate_with_memory(question_data)
                memory_score = self.evaluate_answer_accuracy(question_data, memory_answer)
                memory_scores.append(memory_score)
                print(f"    Memory 포함 점수: {memory_score}")
                
                time.sleep(1.5)  # API 속도 제한 방지
                
                # Memory 없이 답변 생성 및 평가
                print("  - Memory 없이 답변 생성 중...")
                no_memory_answer = self.evaluate_without_memory(question_data)
                no_memory_score = self.evaluate_answer_accuracy(question_data, no_memory_answer)
                no_memory_scores.append(no_memory_score)
                print(f"    Memory 없음 점수: {no_memory_score}")
                
                time.sleep(1.5)  # API 속도 제한 방지
                
                # 결과 저장
                self.results.append({
                    'question_id': question_data['id'],
                    'question': question_data['question'],
                    'memory_answer': memory_answer,
                    'no_memory_answer': no_memory_answer,
                    'memory_score': memory_score,
                    'no_memory_score': no_memory_score,
                    'score_difference': memory_score - no_memory_score
                })
                
                print(f"✅ 질문 {question_data['id']} 완료 - 차이: {memory_score - no_memory_score}")
                
            except Exception as e:
                print(f"❌ 질문 {question_data['id']} 평가 중 오류: {e}")
                continue
        
        # 결과 저장
        results_df = pd.DataFrame(self.results)
        results_df.to_excel('memory_experiment_results.xlsx', index=False)
        print(f"\n✅ 실험 완료! 결과가 'memory_experiment_results.xlsx'에 저장되었습니다.")
        
        return results_df
    
    def plot_results(self, results_df: pd.DataFrame):
        """결과 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. 점수 비교 그래프
        x = range(len(results_df))
        ax1.plot(x, results_df['memory_score'], 'o-', label='Memory 포함', color='blue', linewidth=2, markersize=6)
        ax1.plot(x, results_df['no_memory_score'], 's-', label='Memory 없음', color='red', linewidth=2, markersize=6)
        ax1.set_xlabel('질문 번호', fontsize=12)
        ax1.set_ylabel('정확도 점수', fontsize=12)
        ax1.set_title('Memory 유무에 따른 정확도 비교', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 5.5)
        
        # 2. 점수 차이 히스토그램
        score_diff = results_df['score_difference']
        ax2.hist(score_diff, bins=10, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(score_diff.mean(), color='red', linestyle='--', linewidth=2, label=f'평균: {score_diff.mean():.2f}')
        ax2.set_xlabel('점수 차이 (Memory 포함 - Memory 없음)', fontsize=12)
        ax2.set_ylabel('빈도', fontsize=12)
        ax2.set_title('Memory 효과 분포', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('memory_experiment_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📈 그래프가 'memory_experiment_results.png'로 저장되었습니다.")
    
    def print_summary(self, results_df: pd.DataFrame):
        """실험 결과 요약 출력"""
        print("\n" + "="*60)
        print("📊 MEMORY 실험 결과 요약")
        print("="*60)
        
        memory_avg = results_df['memory_score'].mean()
        no_memory_avg = results_df['no_memory_score'].mean()
        diff_avg = results_df['score_difference'].mean()
        
        print(f"Memory 포함 평균 점수: {memory_avg:.3f}")
        print(f"Memory 없음 평균 점수: {no_memory_avg:.3f}")
        print(f"평균 점수 차이: {diff_avg:.3f}")
        print(f"Memory 효과: {'긍정적' if diff_avg > 0 else '부정적' if diff_avg < 0 else '중립'}")
        
        # 통계적 유의성 (간단한 t-test)
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(results_df['memory_score'], results_df['no_memory_score'])
        print(f"t-통계량: {t_stat:.3f}")
        print(f"p-값: {p_value:.3f}")
        print(f"통계적 유의성: {'유의함' if p_value < 0.05 else '유의하지 않음'} (α=0.05)")
        
        print("="*60)

def main():
    """메인 실행 함수"""
    experiment = MemoryExperiment()
    
    # 실험 실행
    results = experiment.run_experiment()
    
    # 결과 요약 출력
    experiment.print_summary(results)
    
    # 그래프 생성
    experiment.plot_results(results)

if __name__ == "__main__":
    main() 