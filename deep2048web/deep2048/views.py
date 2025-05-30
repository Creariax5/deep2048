# views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .matrix import Matrix
from .models import Score
from .player import Player
from random import randint
from copy import deepcopy
import re
from . import newMinmax
matrixBefore = Matrix("Jeremy", 4)
matrix = Matrix("Jeremy", 4)

def extraire_nombre_fin(chaine):
    match = re.search(r'(\d+)$', chaine) 
    return int(match.group(1)) if match else None 

def index(request):
    matrix.reset()
    context = {
        "matrix": matrix.matrix,
    }
    return render(request, "index.html", context)

def get_matrix(request):
    if request.method == "POST":
        matrix.playing = matrix.test_loose()
        return JsonResponse({'matrix': matrix.matrix, 'score': matrix.player.score, 'win': matrix.win})

def update_matrix(request):
    if request.method == "POST":
        matrix.test_loose()
        direction = request.POST.get('direction')
        matrix.move_inp(direction, True)
        return JsonResponse({'matrix': matrix.matrix, 'score': matrix.player.score, 'win': matrix.win, 'move_history': matrix.get_move_history()})

def reset_matrix(request):
    if request.method == "POST":
        matrix.reset()
        return JsonResponse({'matrix': matrix.matrix, 'score': matrix.player.score, 'win': matrix.win})

def update_rules(request):
    if request.method == "POST":
        matrix.size = int(request.POST.get('size'))
        #matrix.set_model(request.POST.get('model'))
        matrix.reset()
        return JsonResponse({'matrix': matrix.matrix, 'score': matrix.player.score, 'win': matrix.win})

def update_size(request):
    if request.method == "POST":
        matrix.size = int(request.POST.get('size'))
        matrix.reset()
        return JsonResponse({'matrix': matrix.matrix, 'score': matrix.player.score, 'win': matrix.win})
    
def test_loose(request):
    if request.method == "POST":
        playing = matrix.test_loose()
        return JsonResponse({'matrix': matrix.matrix, 'score': matrix.player.score, 'win': matrix.win})

def update_random(request):
    if request.method == "POST":
        matrix.random_move()
        return JsonResponse({'matrix': matrix.matrix, 'score': matrix.player.score, 'win': matrix.win})
    
def update_model(request):
    if request.method == "POST":
        matrix.model_move(newMinmax.best_move)
        return JsonResponse({'matrix': matrix.matrix, 'score': matrix.player.score, 'win': matrix.win})

def play(request):
    if request.method == "POST":
        matrix.playing = True
        matrix.reset()
        while matrix.playing:
            matrix.random_move()
        return JsonResponse({'matrix': matrix.matrix, 'score': matrix.player.score, 'win': matrix.win})
    
def play_ia(request):
    if request.method == "POST":
        matrix.playing = True
        matrix.reset()
        while matrix.playing:
            matrix.model_move(newMinmax.best_move)
        return JsonResponse({'matrix': matrix.matrix, 'score': matrix.player.score, 'win': matrix.win})

def pause(request):
    if request.method == "POST":
        matrix.playing = False
        return JsonResponse({'matrix': matrix.matrix, 'score': matrix.player.score, 'win': matrix.win})

def basic(request):
    return render(request, "basic.html")

def move(request):
    return render(request, "move.html")

def get_score(request):
    if request.method == "POST":
        name = request.POST.get('name')
        size = int(request.POST.get('size'))
        score = extraire_nombre_fin(request.POST.get('score'))
        Score.objects.create(name=name, size=size, score=score)

def load_score(request):
    if request.method == "POST":
        liste_score = list(Score.objects.values())
        #print(liste_score)
        return JsonResponse({'liste_score' : liste_score})
