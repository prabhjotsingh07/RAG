# processor/urls.py
from django.urls import path, include
from . import views
from django.urls import path


urlpatterns = [
    path('', views.text_processor, name='text_processor'),
    path('accounts/signup/', views.signup, name='signup'),
    path('accounts/logout/', views.logout_view, name='logout'),  # Add this line
    path('accounts/', include('django.contrib.auth.urls')), 
    path('history/', views.history, name='history'),
]