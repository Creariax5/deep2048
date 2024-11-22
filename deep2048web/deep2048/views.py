# views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .matrix import Matrix
from .player import Player

matrix = Matrix("Jeremy", 4)

def index(request):
    matrix.reset()
    context = {
        "matrix": matrix.matrix,
    }
    return render(request, "index.html", context)

def update_matrix(request):
    if request.method == "POST":
        direction = request.POST.get('direction')
        matrix.move_inp(direction)  # Update the matrix
        matrix.set_rnd_empty_case(2)
        return JsonResponse({'matrix': matrix.matrix, 'score': matrix.player.score})

def reset_matrix(request):
    if request.method == "POST":
        matrix.reset()
        return JsonResponse({'matrix': matrix.matrix, 'score': matrix.player.score})
