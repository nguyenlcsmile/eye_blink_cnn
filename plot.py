import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 

# Draw line plot
def draw2D(X, Y, order, xname, yname, params, xlim=None, ylim=None, rcparams=None, legend_loc=0):

    title = params['title']
    colors = params['colors']
    markers = params['markers']
    linewidth = params['linewidth']
    markersize = params['markersize']
    figsize = params['figsize']

    if rcparams is None:
        rcparams = {
            'figure.autolayout': True,
            'legend.fontsize': 15,
            'axes.labelsize': 25,
            'axes.titlesize': 25,
            'xtick.labelsize': 25,
            'ytick.labelsize': 25,
            }
        matplotlib.rcParams.update(rcparams)

    X = np.array(X)
    Y = np.array(Y)

    fig = plt.figure(facecolor='white',figsize=figsize)
    plt.title(title)
    plt.ylabel(yname)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.xlabel(xname)
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])

    for i, type_name in enumerate(order):
        plt.plot(X[i], Y[i], colors[i], label=type_name, linewidth=linewidth, markersize=markersize, marker=markers[i])

    plt.grid()
    plt.legend(loc=legend_loc)
    fig.canvas.draw()
    # grab the pixel buffer and dump it into a numpy array
    im = np.array(fig.canvas.renderer._renderer)[:, :, :-1]
    # plt.show()
    plt.close()
    return im[:, :, (2, 1, 0)]

def plot_video(frame_img, fps, i, total_eye1_prob, total_eye2_prob):
    max_X = frame_img / fps
    params = {}
    params['title'] = 'Eye-state-probability'
    params['colors'] = ['b-']
    params['markers'] = [None]
    params['linewidth'] = 3
    params['markersize'] = None
    params['figsize'] = None

    x_axis = np.arange(frame_img) / fps
    # Vis plots
    prob_plot_1 = draw2D([x_axis[:i + 1]],
                        [total_eye1_prob],
                        order=[''],
                        xname='time',
                        yname='eye state',
                        params=params,
                        xlim=[0, max_X],
                        ylim=[-1, 2])
    prob_plot_2 = draw2D([x_axis[:i + 1]],
                            [total_eye2_prob],
                            order=[''],
                            xname='time',
                            yname='eye state',
                            params=params,
                            xlim=[0, max_X],
                            ylim=[-1, 2])

    vis = np.concatenate([prob_plot_1, prob_plot_2], axis=1)
    scale = float(300) / vis.shape[0]
    # Resize plot size to same size with video
    vis = cv2.resize(vis, None, None, fx=scale, fy=scale)

    return vis

