from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate
from django.core.exceptions import ValidationError
from myapp.models import change_detection_with_building_footprint, SpatialJoinResult

from .models import *


class SignupForm(UserCreationForm):
    email = forms.EmailField(
        widget=forms.EmailInput(
            attrs={
                'class':'form-control',
                'placeholder':'Enter email'
            }
        )
    )

    username = forms.CharField(
        widget=forms.TextInput(
            attrs={
                'class':'form-control',
                'placeholder':'Enter username'
            }
        )
    )

    password1 = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                'class':'form-control',
                'placeholder':'Enter password'
            }
        )
    )

    password2 = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                'class':'form-control',
                'placeholder':'Confirm password'
            }
        )
    )


    class Meta:
        model = CustomUser

        fields = [
            'username',
            'email',
            'password1',
            'password2'
        ]


    def clean_username(self):

        username = self.cleaned_data.get(
            'username'
        )

        if len(username) < 4:

            raise ValidationError(
                "Username must contain at least 4 characters"
            )


        if CustomUser.objects.filter(
            username=username
        ).exists():

            raise ValidationError(
                "Username already exists"
            )

        return username


    def clean_email(self):

        email = self.cleaned_data.get(
            'email'
        )

        if CustomUser.objects.filter(
            email=email
        ).exists():

            raise ValidationError(
                "Email already registered"
            )

        return email



class LoginForm(forms.Form):

    email = forms.EmailField(
        widget=forms.EmailInput(
            attrs={
                'class':'form-control',
                'placeholder':'Enter email'
            }
        )
    )


    password = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                'class':'form-control',
                'placeholder':'Enter password'
            }
        )
    )


    def clean(self):

        cleaned_data=super().clean()

        email=cleaned_data.get(
            'email'
        )

        password=cleaned_data.get(
            'password'
        )

        user=authenticate(
            username=email,
            password=password
        )

        if not user:

            raise ValidationError(
                "Invalid email or password"
            )

        if not user.is_active:

            raise ValidationError(
                "Account disabled"
            )

        self.user=user

        return cleaned_data


    def get_user(self):

        return self.user



class ProfileUpdateForm(forms.ModelForm):

    class Meta:

        model=CustomUser

        fields=[
            'username',
            'email',
            'profile_image'
        ]

        widgets={

            'username':forms.TextInput(
                attrs={
                    'class':'form-control'
                }
            ),

            'email':forms.EmailInput(
                attrs={
                    'class':'form-control'
                }
            ),

        }


    def clean_email(self):

        email=self.cleaned_data.get(
            'email'
        )

        qs=CustomUser.objects.filter(
            email=email
        ).exclude(
            id=self.instance.id
        )

        if qs.exists():

            raise ValidationError(
                "Email already exists"
            )

        return email
    
    
class UploadForm(forms.ModelForm):

    class Meta:

        model = change_detection_with_building_footprint

        fields = ['uploaded_2023', 'uploaded_2025']

        widgets = {

            'uploaded_2023': forms.ClearableFileInput(
                attrs={
                    'class': 'form-control',
                    'accept': '.tif,.tiff,.png,.jpg,.jpeg'
                }
            ),

            'uploaded_2025': forms.ClearableFileInput(
                attrs={
                    'class': 'form-control',
                    'accept': '.tif,.tiff,.png,.jpg,.jpeg'
                }
            ),
        }
        
        
        


class ChangeDetectionForm(forms.ModelForm):
    class Meta:
        model = change_detection_with_building_footprint
        fields = ['uploaded_2023', 'uploaded_2025']

        widgets = {
            'uploaded_2023': forms.ClearableFileInput(attrs={
                'class': 'form-control',
                'accept': '.tif,.tiff'
            }),
            'uploaded_2025': forms.ClearableFileInput(attrs={
                'class': 'form-control',
                'accept': '.tif,.tiff'
            }),
        }


class SpatialJoinForm(forms.Form):
    main_shapefile = forms.FileField()
    change_shapefile = forms.FileField()


class ContactForm(forms.ModelForm):
    class Meta:
        model = ContactMessage
        fields = ['name', 'email', 'phone', 'subject', 'message']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Your full name',
                'required': 'required'
            }),
            'email': forms.EmailInput(attrs={
                'class': 'form-control',
                'placeholder': 'your@email.com',
                'required': 'required'
            }),
            'phone': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '+1 (555) 123-4567'
            }),
            'subject': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'What is this about?',
                'required': 'required'
            }),
            'message': forms.Textarea(attrs={
                'class': 'form-control',
                'placeholder': 'Tell us more about your inquiry...',
                'rows': 5,
                'required': 'required'
            }),
        }

