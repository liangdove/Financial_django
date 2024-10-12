from django.shortcuts import render
from tqdm import tqdm
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from Fin.financial_detection.V_2_download.models.GraphSAGE import GraphSAGE as Model
from Fin.financial_detection.V_2_download.utils.dgraphfin import load_data, AdjacentNodesDataset
from Fin.financial_detection.V_2_download.utils.evaluator import Evaluator
from django.http import HttpResponse
from Fin.financial_detection.V_2_download import main_GCN, main_GraphSAGE, main_GEARSage


def main_GCN_output(request):
    output = main_GCN.GCN_main(request)
    return HttpResponse(output)

def main_GraphSage_output(request):
    output = main_GraphSAGE.GraphSAGE_main(request)
    return HttpResponse(output)

def main_GEARSage_output(request):
    output = main_GEARSage.GEARSage_main(request)
    return HttpResponse(output)

