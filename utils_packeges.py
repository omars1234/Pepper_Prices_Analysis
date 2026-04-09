import os
import math
import warnings
import requests
from io import StringIO
from datetime import datetime, time
from itertools import combinations

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import plotly.express as px
from plotly.offline import iplot, plot, init_notebook_mode, download_plotlyjs

import missingno as msno
import umap

from scipy import stats
from scipy.stats import (
    pearsonr, spearmanr, kendalltau,
    shapiro, chi2_contingency,
    mannwhitneyu, wilcoxon, ranksums,
    fisher_exact, median_test, mode
)
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.correlation_tools import corr_nearest, corr_clipped
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.power import (
    FTestAnovaPower, TTestIndPower, TTestPower
)

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.metrics import silhouette_score, mutual_info_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

from xgboost import XGBRegressor

# -------------------------
# Settings
# -------------------------
warnings.filterwarnings('ignore')

sns.set(style="whitegrid")
matplotlib.rcParams["figure.figsize"] = (20, 10)
plt.rcParams['font.size'] = 12

init_notebook_mode(connected=True)