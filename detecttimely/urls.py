from django.contrib import admin
from django.urls import path

from home.views import home
from cypt.views import cypt
from fnac.views import fnac
from mamm.views import mamm
from hist.views import hist

urlpatterns = [
    path('', home.as_view()),
    path('admin/', admin.site.urls),
    path('cypt/', cypt.as_view()),
    path('fnac/', fnac.as_view()),
    path('mamm/', mamm.as_view()),
    path('hist/', hist.as_view()),
]
