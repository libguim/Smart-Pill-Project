# pills/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import PillAnalysisView, PillMasterViewSet, PillSearchView

router = DefaultRouter()
router.register(r'', PillMasterViewSet, basename='pillmaster')

urlpatterns = [
    path('analyze/', PillAnalysisView.as_view(), name='pill_analyze'),  # AI 분석용
    path('search/', PillSearchView.as_view(), name='pill_search'),       # 이름·각인 검색
    path('', include(router.urls)),                                       # 목록·단건 조회 (읽기 전용)
]