from django.db import models
from django.contrib.auth.models import User

class ProcessedText(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    input_text = models.TextField()
    output_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)