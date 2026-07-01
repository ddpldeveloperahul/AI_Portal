from django import forms

from .models import RoadDetection

class UploadForm(forms.ModelForm):
    class Meta:

        model = RoadDetection

        fields = ['input_file']

        widgets = {

            'input_file': forms.FileInput(attrs={'class': 'form-control','accept': '.png,.jpg,.jpeg,.tif,.tiff,image/png,image/jpeg,image/tiff'})
        }


