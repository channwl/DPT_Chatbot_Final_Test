from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='chatbot_home'),           # 기본 화면
    path('ask/', views.ask_question, name='chatbot_ask'),  # 질문 보내기
    path('create_index/', views.create_index, name='create_index'),  # 인덱스 생성
]
