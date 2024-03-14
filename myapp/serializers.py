from rest_framework import serializers

class PredictionSerializer(serializers.Serializer):
    text = serializers.CharField(max_length=1000)
