from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("update/", views.update_matrix, name="update_matrix"),
    path("reset/", views.reset_matrix, name="reset_matrix"),
]
