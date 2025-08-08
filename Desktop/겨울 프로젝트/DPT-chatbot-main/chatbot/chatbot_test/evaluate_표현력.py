import openai
import pandas as pd
import time

import pandas as pd

df = pd.read_excel('/Users/gimchan-ul/Desktop/겨울 프로젝트/DPT-chatbot-main/llm_evaluate_scored_1_249.xlsx')

questions = df['질문']
references = df['정답']
answers = df['답변']

openai.api_key = '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'



# 평가 프롬프트 함수 (CoT 방식, 출력은 점수만)
def build_prompt(question, reference, answer):
    return f"""
당신은 AI 모델의 응답을 검토하고 **표현의 자연스러움과 명확성**을 평가하는 전문가입니다.
당신의 목표는 AI 응답이 문법적으로 자연스럽고, 문장 구성이 매끄럽고 명확한지 판단하는 것입니다.

아래는 사용자 질문과 AI의 응답입니다.

### 사용자 질문:
{question}

### 기준 정답:
{reference}

### AI 응답:
{answer}

문장의 자연스러움, 명확성, 문법적 완성도를 단계적으로 판단하세요.
하지만 출력은 **오직 1~5 중 하나의 숫자**만 포함하고, 설명은 출력하지 마세요.



[표현력 평가 기준]
- 5점: 문법적 오류 없고 매우 자연스러우며 이해가 쉬움
- 4점: 거의 매끄럽고 이해 가능하나, 일부 어색한 표현 존재
- 3점: 이해는 되나 다소 부자연스러운 문장이나 어색한 표현 있음
- 2점: 표현이 자주 어색하거나, 문장 구성에 혼란을 줌
- 1점: 문장이 매우 부자연스럽고 이해가 어려움

질문을 참고하여 응답의 표현력을 평가한 후, 마지막 줄에 1~5 중 하나의 숫자만 작성하세요.

"""

# "질문 ID" 1~249인 행만 선택
target_df = df[(df["질문 ID"] >= 1) & (df["질문 ID"] <= 249)].copy()

# "정확성_점수" 열 초기화 (선택)
target_df['표현성_점수'] = ''

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
target_df.to_excel('llm_evaluate_scored_final_2.xlsx', index=False)
print("완료! 'llm_evaluate_scored_1_249.xlsx' 파일로 저장되었습니다.")