# views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .matrix import Matrix
from .player import Player
from random import randint
from copy import deepcopy

matrixBefore = Matrix("Jeremy", 4)
matrix = Matrix("Jeremy", 4)

def index(request):
    matrix.reset()
    context = {
        "matrix": matrix.matrix,
    }
    return render(request, "index.html", context)

def get_matrix(request):
    if request.method == "POST":
        return JsonResponse({'matrix': matrix.matrix, 'score': matrix.player.score})

def update_matrix(request):
    if request.method == "POST":
        matrix.test_loose()
        direction = request.POST.get('direction')
        matrix.move_inp(direction)
        return JsonResponse({'matrix': matrix.matrix, 'score': matrix.player.score, 'win': matrix.win})

def reset_matrix(request):
    if request.method == "POST":
        matrix.reset()
        return JsonResponse({'matrix': matrix.matrix, 'score': matrix.player.score})
    
def update_rules(request):
    if request.method == "POST":
        matrix.size = int(request.POST.get('size'))
        print(request.POST.get('model'))
        #matrix.set_model(request.POST.get('model'))
        matrix.reset()
        return JsonResponse({'matrix': matrix.matrix, 'score': matrix.player.score})
    
def update_size(request):
    if request.method == "POST":
        matrix.size = int(request.POST.get('size'))
        matrix.reset()
        return JsonResponse({'matrix': matrix.matrix, 'score': matrix.player.score})
    
def test_loose(request):
    if request.method == "POST":
        matrix.test_loose()
        return JsonResponse({'matrix': matrix.matrix, 'score': matrix.player.score})
    
def update_random(request):
    if request.method == "POST":
        matrix.random_move()
        return JsonResponse({'matrix': matrix.matrix, 'score': matrix.player.score, 'win': matrix.win})

def play(request):
    if request.method == "POST":
        matrix.reset()
        while matrix.test_loose():
            matrix.random_move()
        return JsonResponse({'matrix': matrix.matrix, 'score': matrix.player.score, 'win': matrix.win})
