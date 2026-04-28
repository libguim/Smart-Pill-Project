"""
URL configuration for config project.
"""

import os
import json
from pathlib import Path
from django.conf import settings
from django.contrib import admin
from django.urls import path, include
from django.http import HttpResponse, JsonResponse


def home(request):
    json_path = settings.BASE_DIR / 'pills' / 'ai_engine' / 'response.json'

    if not json_path.exists():
        return JsonResponse({"error": "JSON 파일을 찾을 수 없습니다."}, status=404, json_dumps_params={'ensure_ascii': False})

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return JsonResponse(data, json_dumps_params={'ensure_ascii': False})
    except json.JSONDecodeError:
        return JsonResponse({"error": "JSON 형식이 올바르지 않습니다."}, status=500, json_dumps_params={'ensure_ascii': False})
    except Exception as e:
        return JsonResponse({"error": f"서버 내부 오류: {str(e)}"}, status=500, json_dumps_params={'ensure_ascii': False})


urlpatterns = [
    path("", home, name='root_home'),
    path("admin/", admin.site.urls),
    path('api/pills/', include('pills.urls')),
]