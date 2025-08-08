from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect  # ✅ 추가

urlpatterns = [
    path('admin/', admin.site.urls),
    path('chatbot/', include(('chatbot.urls', 'chatbot'), namespace='chatbot')),
    path('', lambda request: redirect('chatbot:chatbot_home')),  # ✅ 추가
]
