from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

# Form for text processing
from django import forms


class TextProcessorForm(forms.Form):
    user_text = forms.CharField(
        widget=forms.Textarea(
            attrs={
                'rows': 4,
                'class': 'form-control',
                'placeholder': 'Enter your text here...'
            }
        ),
        label='Enter your text'
    )
    
    LANGUAGE_CHOICES = [
        ('english', 'English'),
        ('hindi', 'Hindi'),
        ('spanish', 'Spanish'),
        ('french', 'French'),
    ]
    
    output_language = forms.ChoiceField(
        choices=LANGUAGE_CHOICES,
        label='Select the language in which you want the response to be displayed in',
        widget=forms.Select(attrs={'class': 'form-control mt-3'})
    )

# Form for user registration
class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    
    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add Bootstrap classes to all fields
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'