from django.views.generic import TemplateView
from django.shortcuts import render

class home(TemplateView): 
    
    def get(self, request): return render(request, 'home/index.html')