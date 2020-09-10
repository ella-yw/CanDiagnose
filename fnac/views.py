from django.views.generic import TemplateView
from django.shortcuts import render

from fnac.forms import fnacForm
from fnac.ml import Predictor

import keras, pandas#, subprocess
    
class fnac(TemplateView):
    
    def get(self, request): return render(request, 'fnac/index.html', {'form': fnacForm()})
    
    def post(self, request):
        
        form = fnacForm(request.POST)
        
        if form.is_valid():
            
            ClumpThickness = form.cleaned_data['ClumpThickness']
            UniformityofCellSize = form.cleaned_data['UniformityofCellSize']
            UniformityofCellShape = form.cleaned_data['UniformityofCellShape']
            MarginalAdhesion = form.cleaned_data['MarginalAdhesion']
            SingleEpithelialCellSize = form.cleaned_data['SingleEpithelialCellSize']
            BareNuclei = form.cleaned_data['BareNuclei']
            BlandChromatin = form.cleaned_data['BlandChromatin']
            NormalNucleoli = form.cleaned_data['NormalNucleoli']
            Mitoses = form.cleaned_data['Mitoses']
            
            params = [int(ClumpThickness), int(UniformityofCellSize), int(UniformityofCellShape), int(MarginalAdhesion),
                      int(SingleEpithelialCellSize), int(BareNuclei), int(BlandChromatin), int(NormalNucleoli), int(Mitoses)]
            
            #cmd = subprocess.Popen('python -c "from fnac.ml import Predictor; print(Predictor.pred(%s))"' % params, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            #y_pred = cmd.stdout.readlines()[1].strip().decode('ascii')
            
            y_pred = Predictor.pred(params, keras, pandas) 
              
            return render(request, 'fnac/index.html', {'form': form, 'y_pred': y_pred})