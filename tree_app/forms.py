from django import forms
from .models import TreeDetection


class UploadForm(forms.ModelForm):
    class Meta:
        model = TreeDetection
        fields = ["image"]
        widgets = {
            "image": forms.FileInput(
                attrs={
                    "class": "form-control",
                    "accept": ".png,.jpg,.jpeg,.tif,.tiff,image/png,image/jpeg,image/tiff",
                }
            )
        }
