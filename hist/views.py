from django.views.generic import TemplateView
from django.shortcuts import render

from django.db import models
class histModel(models.Model): model_pic = models.ImageField(upload_to = '')

from django import forms
class histForm(forms.Form):
    
    file = forms.ImageField(widget = forms.ClearableFileInput(attrs={'class': 'inputfile inputfile-5', 'style': 'display:none'}))
    
    visualize = forms.NullBooleanField(required=False, widget = forms.CheckboxInput(attrs={'style': 'height:40px;width:40px;'}))

from hist.ml import Predictor
import keras, pandas, numpy, os, scipy, glob, shutil

class hist(TemplateView):
    
    def get(self, request): return render(request, 'hist/index.html', {'form': histForm()})
    
    def post(self, request):
        
        if request.method == 'POST':
            
            form = histForm(request.POST, request.FILES)
            
            if form.is_valid():
                
                row = histModel(model_pic=form.cleaned_data['file']); row.save()
                
                if form.cleaned_data['visualize'] == True:
                    y_pred, filters = Predictor.pred(str(form.cleaned_data['file'].name), keras, 
                                                   pandas, numpy, os, scipy, glob, shutil, form.cleaned_data['visualize']) 
                    return render(request, 'hist/submission.html', 
                                  {'form': form, 'y_pred': y_pred, 'filters': filters})
                else:
                    y_pred = Predictor.pred(str(form.cleaned_data['file'].name), keras, 
                                          pandas, numpy, os, scipy, glob, shutil, form.cleaned_data['visualize']) 
                    return render(request, 'hist/index.html', 
                                  {'form': form, 'y_pred': y_pred})