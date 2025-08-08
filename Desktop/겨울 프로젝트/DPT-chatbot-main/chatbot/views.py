from django.shortcuts import render
from django.http import JsonResponse
from .utils import RAGSystem, generate_faiss_index

def index(request):
    return render(request, 'chatbot/chatbot.html')

def ask_question(request):
    print("🚀 [ask_question] POST 요청 받음")  

    if request.method == 'POST':
        question = request.POST.get('question')
        print(f"📝 [사용자 질문] {question}")  
        if question:
            try:
                rag = RAGSystem()
                answer = rag.process_question(question)
                print(f"💬 [생성된 답변] {answer}")  
                return JsonResponse({'response': answer})
            except Exception as e:
                print(f"❌ [Exception 발생] {e}")
                return JsonResponse({'error': str(e)}, status=500)

    print("⚠️ [POST 아님] 또는 질문 없음") 
    return JsonResponse({'error': '잘못된 요청입니다.'}, status=400)

def create_index(request):
    generate_faiss_index()
    return JsonResponse({'response': '✅ PDF 인덱스가 새로 생성되었습니다!'})