from django.db import models

class PillMaster(models.Model):
    id = models.BigAutoField(primary_key=True)
    k_code = models.CharField(db_column='k_code', unique=True, max_length=20)
    item_seq = models.CharField(db_column='item_seq', unique=True, max_length=50, blank=True, null=True)
    dl_name = models.CharField(db_column='dl_name', max_length=255, blank=True, null=True)
    dl_name_en = models.CharField(db_column='dl_name_en', max_length=255, blank=True, null=True)
    dl_company = models.CharField(db_column='dl_company', max_length=255, blank=True, null=True)
    dl_company_en = models.CharField(db_column='dl_company_en', max_length=255, blank=True, null=True)
    
    # --- 시각 정보 컬럼 추가 ---
    drug_shape = models.CharField(db_column='drug_shape', max_length=100, blank=True, null=True)
    color_class1 = models.CharField(db_column='color_class1', max_length=100, blank=True, null=True)
    color_class2 = models.CharField(db_column='color_class2', max_length=100, blank=True, null=True)
    print_front = models.CharField(db_column='print_front', max_length=100, blank=True, null=True)
    print_back = models.CharField(db_column='print_back', max_length=100, blank=True, null=True)
    
    # --- 수치 및 제형 정보 컬럼 추가 ---
    leng_long = models.CharField(db_column='leng_long', max_length=50, blank=True, null=True)
    leng_short = models.CharField(db_column='leng_short', max_length=50, blank=True, null=True)
    thick = models.CharField(db_column='thick', max_length=50, blank=True, null=True)
    chart = models.TextField(db_column='chart', blank=True, null=True)
    dl_material = models.TextField(db_column='dl_material', blank=True, null=True)
    di_class_no = models.CharField(db_column='di_class_no', max_length=255, blank=True, null=True)
    di_etc_otc_code = models.CharField(db_column='di_etc_otc_code', max_length=100, blank=True, null=True)
    di_edi_code = models.CharField(db_column='di_edi_code', max_length=50, blank=True, null=True)
    form_code_name = models.CharField(db_column='form_code_name', max_length=100, blank=True, null=True)
    
    created_at = models.DateTimeField(db_column='created_at', auto_now_add=True)
    updated_at = models.DateTimeField(db_column='updated_at', auto_now=True)

    class Meta:
        managed = False  # 이미 존재하는 DB 테이블을 사용하므로 False
        db_table = 'pill_master'

class PillDetail(models.Model):
    id = models.BigAutoField(primary_key=True)
    item_seq = models.CharField(db_column='item_seq', unique=True, max_length=20)
    effect_text = models.TextField(db_column='effect_text', blank=True, null=True)
    usage_text = models.TextField(db_column='usage_text', blank=True, null=True)
    warning_text = models.TextField(db_column='warning_text', blank=True, null=True)
    storage_text = models.TextField(db_column='storage_text', blank=True, null=True)
    source_url = models.CharField(db_column='source_url', max_length=500, blank=True, null=True)
    crawl_status = models.CharField(db_column='crawl_status', max_length=255, blank=True, null=True)
    crawled_at = models.DateTimeField(db_column='crawled_at', blank=True, null=True)
    created_at = models.DateTimeField(db_column='created_at', auto_now_add=True)
    updated_at = models.DateTimeField(db_column='updated_at', auto_now=True)

    class Meta:
        managed = False
        db_table = 'pill_detail'