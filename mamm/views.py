from django.views.generic import TemplateView
from django.shortcuts import render

from mamm.ml import Predictor
import keras, pandas, numpy, os, scipy, glob, joblib, shutil

from django.db import models
class mammModel(models.Model): model_pic = models.ImageField(upload_to = '')
    
from django import forms
class mammForm(forms.Form):
    
    file = forms.ImageField(widget = forms.ClearableFileInput(attrs={'class': 'inputfile inputfile-5', 'style': 'display:none'}))
    
    visualize = forms.NullBooleanField(required=False, widget = forms.CheckboxInput(attrs={'id': 'chk', 'style': 'height:40px;width:40px;', 'onclick': 'hide_show()'}))
    
    density = forms.ChoiceField(choices=[('pbd', 'Percent Based Density (PBD)'), 
                                         ('birads5', 'Breast Imaging Reporting and Data System (BIRADS5)')], 
                                widget=forms.RadioSelect, initial='pbd')
    
class mamm(TemplateView):
    
    def get(self, request): return render(request, 'mamm/index.html', {'form': mammForm()})
    
    def post(self, request):
        
        if request.method == 'POST':
            
            form = mammForm(request.POST, request.FILES)
            
            if form.is_valid():
                
                row = mammModel(model_pic = form.cleaned_data['file']); row.save()
                
                if form.cleaned_data['visualize'] == True:
                    y_pred, pbd, categorized_pbd, birads5, filters = Predictor.pred(str(form.cleaned_data['file'].name), keras, 
                                      pandas, numpy, os, scipy, glob, shutil, joblib, 
                                      form.cleaned_data['visualize'], form.cleaned_data['density']) 
                    return render(request, 'mamm/submission.html', 
                                  {'form': form, 'y_pred': y_pred, 'pbd': pbd, 'categorized_pbd': categorized_pbd, 'birads5': birads5, 
                                   'filters': filters, 'density': form.cleaned_data['density'].upper()})
                else:
                    y_pred, pbd, categorized_pbd, birads5 = Predictor.pred(str(form.cleaned_data['file'].name), keras, 
                                           pandas, numpy, os, scipy, glob, shutil, joblib, 
                                           form.cleaned_data['visualize'], form.cleaned_data['density']) 
                    return render(request, 'mamm/index.html', 
                                  {'form': form, 'y_pred': y_pred, 'pbd': pbd, 'categorized_pbd': categorized_pbd, 'birads5': birads5})