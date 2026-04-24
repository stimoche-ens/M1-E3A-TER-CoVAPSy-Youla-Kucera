#import plotly.express as px
import os
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sympy as sp
import my_plotly_utils as pli
import csv
import numpy as np

reader = csv.reader(open("rosbag2_2026_04_22-12_28_36.csv", "r"), delimiter=",")
next(reader) 
x = list(reader)
result = np.array(x).astype("float")
l0 = result[:,3:]
lidars_0 = np.array(l0, dtype=float)
lidars_0 = np.roll(lidars_0, shift=180, axis=1)
l0_means = np.nanmean(lidars_0, axis=0)
l0_shift = np.roll(l0_means, shift=180, axis=0)


reader = csv.reader(open("rosbag2_2026_04_22-12_30_39.csv", "r"), delimiter=",")
next(reader) 
x = list(reader)
result = np.array(x).astype("float")
l90 = result[:,3:]
lidars_90 = np.array(l90, dtype=float)
lidars_90 = np.roll(lidars_90, shift=180, axis=1)
l90_means = np.nanmean(lidars_90, axis=0)
l90_shift = np.roll(l90_means, shift=180, axis=0)


df = pd.DataFrame({
    'ang': pd.Series([i for i in range(-180, 181, 1)]),
    'l0_means': pd.Series(l0_means),
    'l90_means': pd.Series(l90_means),
    'l0_shift': pd.Series(l0_shift),
    'l90_shift': pd.Series(l90_shift),
})




###################################################################################################
subplot_titles = [
    "Fréquence générée par le VCO en fonction de la tension appliquée", "Répartition fréquentielle des modes propres de la cavité", 
    "Répartition fréquentielle des modes propres de la cavité", "Régression: pression en fonction du temps de réponse", 
    "Evolution du temps de réponse associé au mode propre (cavité excitée)","1/(nombre densite electronique) en fonction du temps de réponse"
]
xaxes_labels = [
                "Tension [V]",  "Fréquence [GHz]",
                "Fréquence [GHz]",  "temps de reponse [ms]",
                "Fréquence [GHz]",  "Temps de réponse [ms]",]
yaxes_labels = [
                "Fréquence [GHz]",  "Tension [mV]",
                "Tension [mV]", "Pression [bar]",
                "Temps de réponse [ms]",  "1/n_e [sans unité]",]
fig = make_subplots(
    rows=3, cols=2, 
    subplot_titles=subplot_titles
)
fig.layout.meta = {'grid_titles': subplot_titles}
pli.update_axes_labels(fig, xaxes_labels, yaxes_labels)
#fig.update_yaxes(type="log", row=1, col=2)
fig.update_layout(width=2500, height=2500)
######################################################################################################


pli.plt_trace(fig, df, x="ang", y="l0_means", label="l0", color="red", row=1, col=1)
pli.plt_trace(fig, df, x="ang", y="l90_means", label="l90", color="blue", row=1, col=1)
pli.plt_trace(fig, df, x="ang", y="l0_shift", label="l0", color="red", row=1, col=2)
pli.plt_trace(fig, df, x="ang", y="l90_shift", label="l90", color="blue", row=1, col=2)






####################################################################################################
pli.show(fig)
pli.subplots_to_img(fig,os.path.dirname(os.path.realpath(__file__)))