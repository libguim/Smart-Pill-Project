from rest_framework import serializers
from .models import PillMaster, PillDetail

class PillDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = PillDetail
        fields = '__all__'

class PillMasterSerializer(serializers.ModelSerializer):
    # PillDetail 정보를 포함하기 위해 SerializerMethodField를 정의합니다.
    detail_info = serializers.SerializerMethodField()

    class Meta:
        model = PillMaster
        fields = '__all__'

    def get_detail_info(self, obj):
        """
        PillMaster의 item_seq와 일치하는 PillDetail 데이터를 찾아 반환합니다.
        """
        # item_seq를 기준으로 상세 정보를 조회합니다.
        detail = PillDetail.objects.filter(item_seq=obj.item_seq).first()
        
        if detail:
            # 상세 정보가 존재하면 PillDetailSerializer를 통해 시리얼라이즈된 데이터를 반환합니다.
            return PillDetailSerializer(detail).data
        
        # 상세 정보가 없는 경우 null(None)을 반환합니다.
        return None
    
class PillImageSerializer(serializers.Serializer):
    image = serializers.ImageField() # 이 필드가 있어야 화면에 파일 선택 버튼이 생깁니다.