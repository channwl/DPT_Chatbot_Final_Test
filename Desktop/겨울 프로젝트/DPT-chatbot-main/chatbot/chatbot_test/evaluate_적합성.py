import openai
import pandas as pd
import time

import pandas as pd

df = pd.read_excel('/Users/gimchan-ul/Desktop/겨울 프로젝트/DPT-chatbot-main/llm_evaluate_scored_1_249.xlsx')

questions = df['질문']
references = df['정답']
answers = df['답변']

openai.api_key = ''



# 평가 프롬프트 함수 (CoT 방식, 출력은 점수만)
def build_prompt(question, reference, answer):
    return f"""
당신은 AI 모델의 응답을 검토하고 질문에 대한 적합성을 평가하는 전문가입니다.
당신의 목표는 AI 응답이 질문의 의도에 얼마나 부합하는지를 판단하는 것입니다.

다음은 사용자 질문과 AI 모델의 응답입니다.

응답이 질문의 목적, 요구 정보, 맥락에 적절히 맞는지 단계적으로 생각해보세요.  
그러나 출력은 오직 1~5 사이의 숫자만 포함해야 하며, 설명은 출력하지 마세요.

### 사용자 질문:
{question}

### 기준 정답:
{reference}

### AI 응답:
{answer}

[적합성 평가 기준]
- 5점: 질문 의도에 정확히 부합하고, 핵심 내용을 충실히 다룸
- 4점: 거의 부합하나, 사소한 방향성 누락 또는 부가 정보 중심
- 3점: 부분적으로 부합하지만 핵심에서 벗어난 부분이 존재
- 2점: 질문의 의도와 다소 어긋나거나 관련성이 약함
- 1점: 질문과 거의 무관한 응답임

이제 질문을 보고 생각 과정을 거친 뒤, 마지막 줄에 1~5 중 하나의 숫자만 작성하세요.
"""

# "질문 ID" 1~249인 행만 선택
target_df = df[(df["질문 ID"] >= 1) & (df["질문 ID"] <= 249)].copy()

# "정확성_점수" 열 초기화 (선택)
target_df['적합성_점수'] = ''

# 평가 루프 실행 (target_df만)
for idx, row in target_df.iterrows():
    prompt = build_prompt(row['질문'], row['정답'], row['답변'])
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "당신은 AI 평가자입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        score = response.choices[0].message['content'].strip()
        target_df.at[idx, '정확성_점수'] = score
        print(f"✅ {row['질문 ID']}번 완료 - 점수: {score}")
        time.sleep(1.5)
    except Exception as e:
        target_df.at[idx, '정확성_점수'] = 'error'
        print(f"오류 발생 (질문 ID {row['질문 ID']}): {e}")

# 결과 저장 (원본과 구분되는 새 파일로 저장 권장)
target_df.to_excel('llm_evaluate_scored_final.xlsx', index=False)
print("완료! 'llm_evaluate_scored_1_249.xlsx' 파일로 저장되었습니다.")
