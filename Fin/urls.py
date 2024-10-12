from django.urls import path

from . import views
from Fin.financial_detection.V_2_download import main_GCN, main_GraphSAGE

urlpatterns = [
    path('1/', views.main_GCN_output, name="GCN_main"),
    path("2/", views.main_GraphSage_output, name="GraphSage_main"),
    path("3/", views.main_GEARSage_output, name="GEARSage_main")
]
