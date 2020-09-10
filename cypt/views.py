from django.views.generic import TemplateView
from django.shortcuts import render

from cypt.forms import cyptForm
from cypt.ml import Predictor

import keras, pandas#, subprocess

class cypt(TemplateView):
    
    def get(self, request): return render(request, 'cypt/index.html', {'form': cyptForm()})
    
    def post(self, request):
        
        form = cyptForm(request.POST)
        
        if form.is_valid():
            
            RadiusMean = form.cleaned_data['RadiusMean']
            TextureMean = form.cleaned_data['TextureMean']
            PerimeterMean = form.cleaned_data['PerimeterMean']
            AreaMean = form.cleaned_data['AreaMean']
            SmoothnessMean = form.cleaned_data['SmoothnessMean']
            CompactnessMean = form.cleaned_data['CompactnessMean']
            ConcavityMean = form.cleaned_data['ConcavityMean']
            ConcavePointsMean = form.cleaned_data['ConcavePointsMean']
            SymmetryMean = form.cleaned_data['SymmetryMean']
            FractalDimensionMean = form.cleaned_data['FractalDimensionMean']
            RadiusSE = form.cleaned_data['RadiusSE']
            TextureSE = form.cleaned_data['TextureSE']
            PerimeterSE = form.cleaned_data['PerimeterSE']
            AreaSE = form.cleaned_data['AreaSE']
            SmoothnessSE = form.cleaned_data['SmoothnessSE']
            CompactnessSE = form.cleaned_data['CompactnessSE']
            ConcavitySE = form.cleaned_data['ConcavitySE']
            ConcavePointsSE = form.cleaned_data['ConcavePointsSE']
            SymmetrySE = form.cleaned_data['SymmetrySE']
            FractalDimensionSE = form.cleaned_data['FractalDimensionSE']
            RadiusWorst = form.cleaned_data['RadiusWorst']
            TextureWorst = form.cleaned_data['TextureWorst']
            PerimeterWorst = form.cleaned_data['PerimeterWorst']
            AreaWorst = form.cleaned_data['AreaWorst']
            SmoothnessWorst = form.cleaned_data['SmoothnessWorst']
            CompactnessWorst = form.cleaned_data['CompactnessWorst']
            ConcavityWorst = form.cleaned_data['ConcavityWorst']
            ConcavePointsWorst = form.cleaned_data['ConcavePointsWorst']
            SymmetryWorst = form.cleaned_data['SymmetryWorst']
            FractalDimensionWorst = form.cleaned_data['FractalDimensionWorst']
            
            params = [RadiusMean, TextureMean, PerimeterMean, AreaMean, SmoothnessMean, CompactnessMean,
                      ConcavityMean, ConcavePointsMean, SymmetryMean, FractalDimensionMean, RadiusSE, TextureSE,
                      PerimeterSE, AreaSE, SmoothnessSE, CompactnessSE, ConcavitySE, ConcavePointsSE, SymmetrySE,
                      FractalDimensionSE, RadiusWorst, TextureWorst, PerimeterWorst, AreaWorst, SmoothnessWorst,
                      CompactnessWorst, ConcavityWorst, ConcavePointsWorst, SymmetryWorst, FractalDimensionWorst]
            
            #cmd = subprocess.Popen('python -c "from cypt.ml import Predictor; print(Predictor.pred(%s))"' % params, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            #y_pred = cmd.stdout.readlines()[1].strip().decode('ascii')
            
            y_pred = Predictor.pred(params, keras, pandas) 
            
            return render(request, 'cypt/index.html', {'form': form, 'y_pred': y_pred})