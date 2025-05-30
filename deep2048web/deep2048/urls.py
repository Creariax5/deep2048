from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("update/", views.update_matrix, name="update_matrix"),
    path("get/", views.get_matrix, name="get_matrix"),
    path("reset/", views.reset_matrix, name="reset_matrix"),
    path("play/", views.play, name="play"),
    path("play_ia/", views.play_ia, name="play_ia"),
    path("pause/", views.pause, name="pause"),
    path("update_rules/", views.update_rules, name="update_rules"),
    path("update_size/", views.update_rules, name="update_size"),
    path("update_random/", views.update_random, name="update_random"),
    path("basic/", views.basic, name="basic"),
    path("move/", views.move, name="move"),
    path("get_score/", views.get_score, name="get_score"),
    path("load_score/", views.load_score, name="load_score"),
    path("ia_game/", views.update_model, name="update_ia")
]
