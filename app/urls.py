from django.urls import path
from . import views

urlpatterns = [
    path('', views.verify_image_interface, name='verify_image_interface'),
    path('verify-image/', views.verify_image, name='verify_image'),
]
