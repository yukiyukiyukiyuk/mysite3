from django.shortcuts import render

def home(request):
    # ホームページのビューの実装
    return render(request, 'home.html')