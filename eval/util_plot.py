"""
@author: qingyi 

Functions in this script:
    MidpointNormalize: helper to anchor colorbar with center 0

 - FIT
    plot_fit_res(test_y, test_out, test_res): scatterplot fit and residuals
    plot_std_res(test_y, test_out, test_res, test_std): scatter plot standardized residuals

 - PREDICTION INTERVAL
    plot_pi_grid
    plot_pi_graph(ts_test, y_test, pred_test, pred_std, z, time_size, models, plot_cell=None)
        helper function: plot_pi(y_test, pred_test, pred_std, z, plot_cell=None) 

 - TEMPORAL AGGREGATION
    plot_res_temp_ind(test_y, test_res, test_ts, spatial_units, time_size, include_zeros=True): bar plot of temporal residuals 
    plot_res_temp_agg(test_y, test_res, test_ts, time_size, include_zeros=True): scatter plot of temporally aggregated residuals

 - SPATIAL AGGREGATION
    plot_output_graph(stations, df, merge_cols_left, merge_cols_right, value_col, title, cmap, scale=(5,30)): spatially aggregated graph
    plot_output_grid(test_out_mean, test_out_std, test_y, include_zeros=True): spatially aggregated (observed, modelled mean, modelled stdev, residuals stdev)
    plot_res_grid(test_y, test_res, include_zeros=True): spatially aggregated (observed, residuals, residual stdev)

"""

import geopandas as gpd
# import geoplot as gplt
# import geoplot.crs as gcrs
# import mapclassify as mc
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def plot_calibration(ax, p, p_hat, label, clr):

    for a,b in zip(p, p_hat):
        ax.scatter(a, b, c=clr, s=15)
    ax.plot(p, p_hat, c=clr, label=label, linewidth=2)
    
    ax.plot(p, p, c='grey', linewidth=2)
    ax.set_xticks(np.arange(0,1.1,0.2))
    ax.set_yticks(np.arange(0,1.1,0.2))

    ax.set_xlabel("Expected Quantile (Predicted)")
    ax.set_ylabel("Observed Quantile (from Data)")

    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.grid();
    ax.legend();

    return ax
