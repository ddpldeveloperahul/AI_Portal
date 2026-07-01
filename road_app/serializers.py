from rest_framework import serializers
from .models import RoadDetection

class RoadDetectionSerializer(serializers.ModelSerializer):

    class Meta:
        model = RoadDetection
        fields = '__all__'