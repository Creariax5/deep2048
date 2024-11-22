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

def update_matrix(request):
    if request.method == "POST":
        matrixBefore = deepcopy(matrix)
        direction = request.POST.get('direction')
        matrix.move_inp(direction)
        if matrix.matrix != matrixBefore.matrix:
            matrix.player.moves += 1
            nb = randint(0, 9)
            if (nb == 0):
                matrix.set_rnd_empty_case(4)
            else:
                matrix.set_rnd_empty_case(2)
        return JsonResponse({'matrix': matrix.matrix, 'score': matrix.player.score})

def reset_matrix(request):
    if request.method == "POST":
        matrix.reset()
        return JsonResponse({'matrix': matrix.matrix, 'score': matrix.player.score})
