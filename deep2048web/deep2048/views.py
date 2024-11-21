# views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .matrix import Matrix

matrix = Matrix()

def index(request):
    reset_matrix(request)
    context = {
        "matrix": matrix.matrix,
    }
    return render(request, "index.html", context)

def update_matrix(request):
    if request.method == "POST":
        direction = request.POST.get('direction')
        matrix.move_up()  # Update the matrix
        matrix.set_rnd_empty_case(2)
        return JsonResponse({'matrix': matrix.matrix})

def reset_matrix(request):
    if request.method == "POST":
        matrix.reset()
        return JsonResponse({'matrix': matrix.matrix})
