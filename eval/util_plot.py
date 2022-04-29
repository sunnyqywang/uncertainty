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
import mapclassify as mc
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
    ax.set_ylabel("Observed Quantile (from Data")

    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.grid();
    ax.legend();

    return ax

def plot_fit_res(test_y, test_out, test_res=None):
    # plot observed vs. fit 
    # and residual distribution

    if test_res is None:
        test_res = test_y - test_out

    m = np.min((np.max(test_y), np.max(test_out)))
    fig, ax = plt.subplots(1,2, figsize=(10, 4))
    ax[0].scatter(test_y.flatten(), test_out.flatten(), color='gray', s = 15, alpha=0.7)
    ax[0].plot([0,m], [0,m], linewidth=4, color='k')
    ax[0].set_xlabel('observed')
    ax[0].set_ylabel('fitted')
    ax[1].scatter(test_out.flatten(), test_res.flatten(), color='gray', s = 15, alpha=0.7)
    ax[1].set_xlabel('fitted')
    ax[1].set_ylabel('residual')
    display_thresh = np.percentile(test_out.flatten(), 75)
    ax[1].set_xlim([0, display_thresh])
    keep = test_out.flatten() < display_thresh
    ax[1].set_ylim([np.min(test_res.flatten()[keep]), np.max(test_res.flatten()[keep])])
    plt.tight_layout()
    plt.show()

def plot_std_res(test_y, test_out, test_res, test_std):
    # plot standardized residuals

    z = np.divide(test_res, test_std)
    plt.scatter(test_y.flatten(), z.flatten())
    plt.xlim([0, np.percentile(test_y.flatten(), 99.5)])
    plt.show()

def plot_pi_by_bounds(ax, m, ts, lb, ub, time_size, duration=1, c='cyan'):
    # duration: plot duration in days
    
    ntime = 96 // time_size
    st = (ts.min()//ntime+1)*ntime-ts.min()-1
    dur = st+duration*(ntime-7)
    et = np.min((len(ts), dur))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6*duration,5))
    else:
        fig = None
    ax.fill_between(ts[st:et], lb[st:et], ub[st:et], color=c, alpha=0.3, label=m)
    ax.plot(ts[st:et], lb[st:et], linewidth=1, color=c, alpha=0.5)
    ax.plot(ts[st:et], ub[st:et], linewidth=1, color=c, alpha=0.5)
   
    return fig, ax

def plot_pi(ax, m, ts, pred, pred_std, time_size, z=1.96, c='cyan', linewidth=1):
    ax.fill_between(ts, np.maximum(pred-z*pred_std,0), (pred+z*pred_std), color=c, alpha=0.3)
    ax.plot(ts, np.maximum(pred-z*pred_std,0), linewidth=linewidth, color=c, alpha=0.5)
    ax.plot(ts, pred+(z*pred_std), linewidth=linewidth, color=c, alpha=0.5)
    ax.plot(ts, pred, linewidth=2*linewidth, color=c, label=m)
    
def plot_pi_graph(ts_test, y_test, pred_test, pred_std, z, time_size, models, duration=1, ax=None, plot_cell=None, colors=['cornflowerblue','coral','forestgreen'], linewidth=2):

    assert len(colors) >= len(pred_test) == len(pred_std)
    # cell with maximum demand to plot time series
    if plot_cell is None:
        plot_cell = np.unravel_index(np.argmax(np.mean(y_test, axis=0), axis=None), y_test.shape[1:])
        print("Plotting cell with max flow: ", plot_cell)
    elif plot_cell == -1:
        plot_cell = np.unravel_index(np.argmin(np.mean(y_test, axis=0), axis=None), y_test.shape[1:])
        print("Plotting cell with min flow: ", plot_cell)
    else:
        print("Plotting cell: ", plot_cell)

    
    ntime = 96 // time_size
    st = (ts_test.min()//ntime+1)*ntime-ts_test.min()-1
    dur = st+duration*(ntime-7)
    et = np.min((len(ts_test), dur))

    if ax is None:
        fig, ax = plt.subplots(figsize=(2.5*duration,2))
    else:
        fig = None
    for i in range(len(pred_test)):
        plot_pi(ax, m=models[i], ts=ts_test[st:et], pred=pred_test[i][st:et, plot_cell].flatten(), 
                pred_std=pred_std[i][st:et,plot_cell].flatten(), time_size=time_size, z=z, c=colors[i], linewidth=linewidth)

    ts = ts_test[st:et]
    ax.plot(ts, y_test[st:et,plot_cell].flatten(), linewidth=1, color='k', label='True Demand')
    ticks = np.concatenate([np.arange(0,len(ts),17),np.arange(5,len(ts),17), np.arange(11,len(ts), 17), np.arange(16,len(ts),17)])
    ticks = np.sort(ts[ticks])
    ax.set_xticks(ticks)
    ax.set_xticklabels(np.array((ticks%24)/4*time_size,dtype=int))
    ax.set_xlabel('Hour')
#     ax.set_ylabel('# Tap-in')
#     ax.legend()
    ax.grid(True)


    return fig,ax


def plot_pi_grid(ts_test, y_test, pred_test, pred_std, z, time_size, plot_cell=None):
    # need to be modified as plot_pi has changed

    # cell with maximum demand to plot time series
    if plot_cell is None:
        plot_row, plot_col = np.unravel_index(np.argmax(np.mean(y_test, axis=0), axis=None), y_test.shape[1:])
    else:
        plot_row, plot_col = plot_cell

    ntime = 96 // time_size
    st = (ts_test.min()//ntime+1)*ntime-ts_test.min()
    # plot the first week in the test period, or the whole test period if it is less than a week
    duration = st+7*ntime
    et = np.min((len(ts_test), duration))

    print("Plotting cell with max flow: ", plot_row, plot_col)

    plot_pi(ts=ts_test[st:et], true=y_test[:,plot_row, plot_col], pred=pred_test[:, plot_row, plot_col], pred_std=pred_std[:,plot_row,plot_col],time_size=time_size, z=z)
    

def plot_time_series(ts, timeseries, label1, label2, axes=None, time_size = 4, duration = 1, colors=['cornflowerblue','coral','forestgreen'], linewidth=1):
    
    # label1: number of plots
    # label2: number of time series in each plot
    # duration: plot duration in days
    
    assert len(colors) >= len(label2)

    ntime = 96 // time_size
    st = (ts.min()//ntime+1)*ntime-ts.min()-1
    dur = st+duration*(ntime-7)
    et = np.min((len(ts), dur))
    ts = ts[st:et]

    if axes is not None:
        assert len(axes) >= len(timeseries)
        create_axes = False
    else:
        create_axes = True
        assert len(timeseries) == len(label1)
        axes = []
    return_figs = []

    if label1 is None:
        label1 = ['']*len(timeseries)

    for s,l in zip(range(len(timeseries)),label1):
        if create_axes: 
            fig, ax = plt.subplots(figsize=(2.5*duration, 2))
            axes.append(ax)
            return_figs.append(fig)
        else:
            ax = axes[s]

        for i,c,l2 in zip(range(len(timeseries[s])),colors,label2):
            ax.plot(ts, timeseries[s][i][st:et], c=c, label=l2, linewidth=linewidth, alpha=0.8)

        ticks = np.concatenate([np.arange(0,len(ts),17),np.arange(5,len(ts),17), np.arange(11,len(ts), 17), np.arange(16,len(ts),17)])
        ticks = np.sort(ts[ticks])
        ax.set_xticks(ticks)
        ax.set_xticklabels(np.array((ticks%24)/4*time_size,dtype=int))
        ax.set_xlabel('Hour')
        if l != '':
            ax.set_ylabel(l)
#         ax.legend()
        ax.grid(True)


    return return_figs, axes

def plot_res_temp_ind(test_y, test_res, test_ts, spatial_units, time_size, include_zeros=True):
    # temporal distribution of residuals

    # two plots: with and without outliers
    # boxplot of all spatial units for each timestamp

    fig, ax = plt.subplots(1,2, figsize=(15, 5))
    test_ts_expanded = np.repeat(test_ts, spatial_units) % (96//time_size)
    test_ts_hour = (test_ts_expanded / time_size) * 4
    df_res = pd.DataFrame(np.array([test_ts_hour, test_res.flatten(), test_y.flatten()]).T,
                                              columns = ['Hour','Residual','Observed'])
    if ~include_zeros:
            df_res = df_res[df_res['Observed']!=0]

    df_res = df_res.groupby('Hour', as_index=False).agg({'Residual':list})
    ax[0].set_title('All Data')
    ax[1].set_title('Outliers in the left plot removed')
    ax[0].boxplot([l for l in df_res['Residual']])
    ax[1].boxplot([l for l in df_res['Residual']], sym='')
    ax[0].set_xlabel('Hour')
    ax[0].set_ylabel('Residual')
    ax[1].set_xlabel('Hour')
    ax[1].set_ylabel('Residual')
    ax[0].set_xticks(np.arange(1, len(df_res)+1))
    ax[1].set_xticks(np.arange(1, len(df_res)+1))
    ax[0].set_xticklabels(df_res['Hour'].astype(int))
    ax[1].set_xticklabels(df_res['Hour'].astype(int))
    plt.show()

def plot_res_temp_agg(test_y, test_res, test_ts, time_size, include_zeros=True):
    # aggregate temporal distribution of residuals
    # scatter plot
    # summed across the region

    if include_zeros:
            temp_res = test_res
    else:
            temp_res = test_res.copy()
            temp_res[test_y==0] = np.NaN

    test_ts_hour = test_ts % (96//time_size)
    test_ts_hour = test_ts_hour / time_size * 4

    spatial_axes = tuple(np.arange(1, len(test_y.shape)))
    df_res = pd.DataFrame(np.array([np.nansum(temp_res, axis=spatial_axes), test_ts_hour, np.sum(test_y, axis=spatial_axes)]).T,
                                              columns = ['Residual','Hour','Observed'])

    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    ax[0].scatter(df_res['Hour'], df_res['Residual'])
    ax[0].set_xlabel('Hour')
    ax[0].set_ylabel('Abs Residual')
    ax[1].scatter(df_res['Hour'], df_res['Residual'] / df_res['Observed']*100)
    ax[1].set_xlabel('Hour')
    ax[1].set_ylabel('Abs Residual (%)')
    plt.tight_layout()
    plt.show()

    return fig


# def plot_output_graph(stations, df, merge_cols_left, merge_cols_right, value_col, title, cmap, scale=(5,30)):
#     # graph plot only
#     # spatial distribution of the df[value_col] with spatial information in 'stations'

#     temp = pd.merge(stations, df, left_on=merge_cols_left, right_on=merge_cols_right)
#     scheme = mc.Quantiles(temp[value_col], k=5)
#     # get all upper bounds
#     upper_bounds = scheme.bins
#     # get and format bounds
#     bounds = []
#     for index, upper_bound in enumerate(upper_bounds):
#         if index == 0:
#             lower_bound = df[value_col].min()
#         else:
#             lower_bound = upper_bounds[index-1]

#         # format the numerical legend here
#         bound = f'{lower_bound:.0f} - {upper_bound:.0f}'
#         bounds.append(bound)

#     chicago = (stations.geometry.x.min(), stations.geometry.y.min(),
#                        stations.geometry.x.max(), stations.geometry.y.max())
#     downtown = (-87.645168, 41.872034, -87.623921, 41.889749)
#     contiguous_usa = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))
#     a = gplt.pointplot(temp, projection=gcrs.WebMercator(), cmap=cmap,
#                               hue=value_col, scale=value_col, limits=scale, scheme=scheme, extent=chicago, legend=False)
#     gplt.webmap(contiguous_usa, projection=gcrs.WebMercator(), ax=a, extent=chicago)
#     # this patch won't show up with webmap, need to investigate further
#     #gplt.gca().add_patch(Rectangle((-87.645168, 41.872034),0.021247,0.017715,linewidth=5,edgecolor='r',facecolor='r'))
#     # this does not work either, cannot overlay polygon on webmap, seems that only pointplot works.
#     #a = gplt.polyplot(poly, projection=gcrs.WebMercator(), extent=chicago)
#     plt.title(title)

#     #plt.savefig("../figures/20201201/"+period+"_"+value_col+".png",bbox_inches='tight')
#     a = gplt.pointplot(temp, projection=gcrs.WebMercator(), hue=value_col, scale=value_col, limits=scale,
#                                    scheme=scheme, legend=True, legend_var='hue',cmap=cmap,
#                               legend_kwargs={'bbox_to_anchor': (1, 0.5), 'frameon': True})
#     gplt.webmap(contiguous_usa, projection=gcrs.WebMercator(), ax=a, extent=downtown)
#     plt.title('Downtown Zoom')
#     # get all the legend labels
#     legend_labels = a.get_legend().get_texts()

#     # replace the legend labels
#     for bound, legend_label in zip(bounds, legend_labels):
#         legend_label.set_text(bound)

#     #plt.savefig("../figures/20201201/"+period+"_"+value_col+"_downtown.png", bbox_inches='tight')


# def plot_output_grid(test_out_mean, test_out_std, test_y, include_zeros=True):
#         fig, ax = plt.subplots(1,4, figsize = (20, 10))
#         ax[0].set_title('Demand')
#         ax[1].set_title('Modelled Mean')
#         ax[2].set_title('Modelled Std Dev')
#         ax[3].set_title('Residual Std Dev')

#         temp_res = test_y - test_out_mean

#         ax[0].set_xticks([])
#         ax[1].set_xticks([])
#         ax[2].set_xticks([])
#         ax[3].set_xticks([])

#         ax[0].set_yticks([])
#         ax[1].set_yticks([])
#         ax[2].set_yticks([])
#         ax[3].set_yticks([])

#         if include_zeros:
#                 temp_y = test_y
#                 temp_out_mean = test_out_mean
#                 temp_out_std = test_out_std
#         else:
#                 temp_y = test_y.copy()
#                 temp_out_mean = test_out_mean.copy()
#                 temp_out_std = test_out_std.copy()
#                 zero_mask = test_y==0
#                 temp_y[zero_mask] = np.NaN
#                 temp_out_mean[zero_mask] = np.NaN
#                 temp_out_std[zero_mask] = np.NaN
#                 temp_res[zero_mask] = np.NaN

#         l = np.min([np.nanmean(temp_y, axis=0), np.nanmean(temp_out_mean, axis=0)])
#         u = np.max([np.nanmean(temp_y, axis=0), np.nanmean(temp_out_mean, axis=0)])
#         im0 = ax[0].imshow(np.nanmean(temp_y, axis=0), cmap='coolwarm',
#                         clim=[l,u])
#         im1 = ax[1].imshow(np.nanmean(temp_out_mean, axis=0), cmap='coolwarm',
#                         clim=[l,u])
#         im2 = ax[2].imshow(np.nanmean(temp_out_std, axis=0), cmap='coolwarm')
#         #norm = MidpointNormalize(vmin=np.min(np.nanmean(temp_res, axis=0)),
#         #                        vmax=np.max(np.nanmean(temp_res, axis=0)), midpoint=0)
#         #im3 = ax[3].imshow(np.nanmean(temp_res, axis=0), cmap='coolwarm', norm=norm)
#         im3 = ax[3].imshow(np.nanstd(temp_res, axis=0), cmap='coolwarm', clim=[0,12])

#         #cax = fig.add_axes([0.225, 0.2, 0.02, 0.6])
#         cax = fig.add_axes([0.25, 0.25, 0.015, 0.5])
#         fig.colorbar(im0, cax=cax, orientation='vertical')
#         #cax = fig.add_axes([0.475, 0.2, 0.02, 0.6])
#         cax = fig.add_axes([0.485, 0.25, 0.015, 0.5])
#         fig.colorbar(im1, cax=cax, orientation='vertical')
#         #cax = fig.add_axes([0.7, 0.2, 0.02, 0.6])
#         cax = fig.add_axes([0.725, 0.25, 0.015, 0.5])
#         fig.colorbar(im2, cax=cax, orientation='vertical')
#         #cax = fig.add_axes([0.925, 0.2, 0.02, 0.6])
#         cax = fig.add_axes([0.96, 0.25, 0.015, 0.5])
#         fig.colorbar(im3, cax=cax, orientation='vertical')
#         plt.tight_layout(pad = 4)

#         return fig

# def plot_res_grid(test_y, test_res, include_zeros=True):
#         fig, ax = plt.subplots(1,3, figsize = (15, 10))
#         ax[0].set_title('Demand')
#         ax[1].set_title('Residuals')
#         ax[2].set_title('Res Stdev')

#         if include_zeros:
#                 temp_y = test_y
#                 temp_res = test_res
#         else:
#                 temp_y = test_y.copy()
#                 temp_res = test_res.copy()
#                 zero_mask = test_y==0
#                 temp_y[zero_mask] = np.NaN
#                 temp_res[zero_mask] = np.NaN

#         im0 = ax[0].imshow(np.nanmean(temp_y, axis=0), cmap='coolwarm')
#         norm = MidpointNormalize(vmin=np.min(np.nanmean(temp_res, axis=0)),
#                                 vmax=np.max(np.nanmean(temp_res, axis=0)), midpoint=0)
#         im1 = ax[1].imshow(np.nanmean(temp_res, axis=0), cmap='coolwarm', norm=norm)
#         im2 = ax[2].imshow(np.nanstd(temp_res, axis=0), cmap='coolwarm', clim=[0,12])

#         ax[0].set_xticks([])
#         ax[1].set_xticks([])
#         ax[2].set_xticks([])

#         ax[0].set_yticks([])
#         ax[1].set_yticks([])
#         ax[2].set_yticks([])

#         # left bot width height
#         #cax = fig.add_axes([0.35, 0.2, 0.02, 0.6])
#         cax = fig.add_axes([0.33, 0.25, 0.016, 0.5])
#         fig.colorbar(im0, cax=cax, orientation='vertical')
#         #cax = fig.add_axes([0.65, 0.2, 0.02, 0.6])
#         cax = fig.add_axes([0.63, 0.25, 0.016, 0.5])
#         fig.colorbar(im1, cax=cax, orientation='vertical')
#         #cax = fig.add_axes([0.95, 0.2, 0.02, 0.6])
#         cax = fig.add_axes([0.95, 0.25, 0.016, 0.5])
#         fig.colorbar(im2, cax=cax, orientation='vertical')
#         #plt.tight_layout(pad = 2)
#         plt.tight_layout(pad = 4)

#         return fig



