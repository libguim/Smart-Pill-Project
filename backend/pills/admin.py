from django.contrib import admin
from .models import PillMaster, PillDetail

@admin.register(PillMaster)
class PillMasterAdmin(admin.ModelAdmin):
    list_display = ('id', 'item_seq', 'dl_name', 'dl_company', 'created_at') 
    search_fields = ('dl_name', 'item_seq', 'dl_company') 
    list_filter = ('dl_company',) 

@admin.register(PillDetail)
class PillDetailAdmin(admin.ModelAdmin):
    list_display = ('id', 'item_seq', 'crawl_status', 'crawled_at')
    search_fields = ('item_seq',)
    list_filter = ('crawl_status',)

