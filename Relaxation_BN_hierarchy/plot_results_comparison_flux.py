# Copyright 2021 SAMURAI TEAM. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib import animation
from matplotlib import rc
import argparse

def read_mesh(filename, ite=None):
    return h5py.File(filename + '.h5', 'r')['mesh']

def scatter_plot(ax, points):
    return ax.scatter(points[:, 0], points[:, 1], marker='+')

def scatter_update(scatter, points):
    scatter.set_offsets(points[:, :2])

def line_plot(ax, x_1, y_1, \
                  x_2, y_2, \
                  x_3, y_3, \
                  x_4, y_4, \
                  x_5, y_5, \
                  x_6, y_6, \
                  x_7, y_7):
    #Plot results
    plot_1 = ax.plot(x_1, y_1, 'b-', linewidth=1, markersize=4)[0]
    plot_2 = ax.plot(x_2, y_2, 'ro', linewidth=1, markersize=4, alpha=1, markevery=256)[0]
    plot_3 = ax.plot(x_3, y_3, 'gx', linewidth=1, markersize=4, alpha=1, markevery=2)[0]
    plot_4 = ax.plot(x_4, y_4, 'ys', linewidth=1, markersize=4, alpha=1, markevery=256)[0]
    plot_5 = ax.plot(x_5, y_5, 'o', color='orange', linewidth=1, markersize=4, alpha=1, markevery=256)[0]
    plot_6 = ax.plot(x_6, y_6, 'bd', linewidth=1, markersize=4, alpha=1, markevery=2)[0]
    plot_7 = ax.plot(x_7, y_7, 'k-', linewidth=1, markersize=4, alpha=1)[0]

    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.yaxis.get_offset_text().set_fontsize(20)
    #np.savetxt('DOUBLE_RAREFACTION/numerical_results_alpha2.dat', \
    #           np.c_[x_1, y_1,\
    #                 x_4, y_4, \
    #                 x_7, y_7])

    #Read and plot the analytical results
    if args.analytical is not None:
        if args.column_analytical is None:
            sys.exit("Unknown column to be read for the analytical solution")
        data_analytical = np.genfromtxt(args.analytical)
        plot_analytical = ax.plot(data_analytical[:,0], data_analytical[:,args.column_analytical], 'k-', linewidth=1.5, markersize=3, alpha=1)[0]

    #Read and plot the reference results
    if args.reference is not None:
        if args.column_reference is None:
            sys.exit("Unknown column to be read for the reference solution")
        data_ref = np.genfromtxt(args.reference)
        plot_ref = ax.plot(data_ref[:,0], data_ref[:,args.column_reference], 'b-', linewidth=1.5, markersize=3, alpha=1)[0]

    #Add legend
    if args.analytical is not None:
        if args.reference is not None:
            ax.legend([plot_1, plot_2, plot_3, \
                       plot_4, plot_5, plot_6, plot_7, \
                       plot_analytical, plot_ref], \
                      ['HLL + BR (27)', 'HLL + BR (28)', 'HLL + Crouzet et al.', \
                       'HLLC + BR (27)', 'HLLC + BR (28)', 'HLLC + Crouzet et al.', 'HLLC (wave-propagation)', \
                       'Analytical (5 equations)', 'Reference results'], fontsize="20", loc="best")
        else:
            ax.legend([plot_1, plot_2, plot_3, \
                       plot_4, plot_5, plot_6, plot_7, \
                       plot_analytical], \
                      ['HLL + BR (27)', 'HLL + BR (28)', 'HLL + Crouzet et al.', \
                       'HLLC + BR (27)', 'HLLC + BR (28)', 'HLLC + Crouzet et al.', 'HLLC (wave-propagation)', \
                       'Analytical (5 equations)'], fontsize="20", loc="best")
    elif args.reference is not None:
        ax.legend([plot_1, plot_2, plot_3, \
                   plot_4, plot_5, plot_6, plot_7, \
                   plot_ref], \
                  ['HLL + BR (27)', 'HLL + BR (28)', 'HLL + Crouzet et al.', \
                   'HLLC + BR (27)', 'HLLC + BR (28)', 'HLLC + Crouzet et al.', 'HLLC (wave-propagation)', \
                   'Reference results'], fontsize="20", loc="best")
    else:
        ax.legend([plot_1, plot_2, plot_3, \
                   plot_4, plot_5, plot_6, plot_7], \
                  ['HLL + BR (27)', 'HLL + BR (28)', 'HLL + Crouzet et al.', \
                   'HLLC + BR (27)', 'HLLC + BR (28)', 'HLLC + Crouzet et al.', 'HLLC (wave-propagation)'], fontsize="20", loc="best")

    return plot_1, plot_2, plot_3, \
           plot_4, plot_5, plot_6, plot_7

def line_update(lines, x_1, y_1, \
                       x_2, y_2, \
                       x_3, y_3, \
                       x_4, y_4, \
                       x_5, y_5, \
                       x_6, y_6, \
                       x_7, y_7):
    lines[1].set_data(x_1, y_1)
    lines[2].set_data(x_2, y_2)
    lines[3].set_data(x_3, y_3)
    lines[4].set_data(x_4, y_4)
    lines[5].set_data(x_5, y_5)
    lines[6].set_data(x_6, y_6)
    lines[7].set_data(x_7, y_7)

class Plot:
    def __init__(self, filename_1, \
                       filename_2, \
                       filename_3, \
                       filename_4, \
                       filename_5, \
                       filename_6, \
                       filename_7):
        self.fig = plt.figure()
        self.artists = []
        self.ax = []

        mesh_1 = read_mesh(filename_1)
        if args.field is None:
            ax = plt.subplot(111)
            self.plot(ax, mesh_1)
            ax.set_title("Mesh")
            self.ax = [ax]
        else:
            mesh_2 = read_mesh(filename_2)
            mesh_3 = read_mesh(filename_3)
            mesh_4 = read_mesh(filename_4)
            mesh_5 = read_mesh(filename_5)
            mesh_6 = read_mesh(filename_6)
            mesh_7 = read_mesh(filename_7)
            for i, f in enumerate(args.field):
                ax = plt.subplot(1, len(args.field), i + 1)
                self.plot(ax, mesh_1, mesh_2, mesh_3, \
                              mesh_4, mesh_5, mesh_6, mesh_7, f)
                ax.set_title(r"$\rho$",fontsize=20)

    def plot(self, ax, mesh_1, \
                       mesh_2=None, \
                       mesh_3=None, \
                       mesh_4=None, \
                       mesh_5=None, \
                       mesh_6=None, \
                       mesh_7=None, field=None, init=True):
        points_1       = mesh_1['points']
        connectivity_1 = mesh_1['connectivity']

        segments_1 = np.zeros((connectivity_1.shape[0], 2, 2))
        segments_1[:, :, 0] = points_1[:][connectivity_1[:]][:, :, 0]

        if field is None:
            segments_1[:, :, 1] = 0
            if init:
                self.artists.append(scatter_plot(ax, points))
                self.lc = mc.LineCollection(segments_1, colors='b', linewidths=2)
                self.lines = ax.add_collection(self.lc)
            else:
                scatter_update(self.artists[self.index], points)
                self.index += 1
                # self.lc.set_array(segments)
        else:
            data_1    = mesh_1['fields'][field][:]
            centers_1 = 0.5*(segments_1[:, 0, 0] + segments_1[:, 1, 0])
            segments_1[:, :, 1] = data_1[:, np.newaxis]
            # ax.scatter(centers, data, marker='+')
            index_1 = np.argsort(centers_1)

            points_2       = mesh_2['points']
            connectivity_2 = mesh_2['connectivity']
            segments_2     = np.zeros((connectivity_2.shape[0], 2, 2))
            segments_2[:, :, 0] = points_2[:][connectivity_2[:]][:, :, 0]
            data_2    = mesh_2['fields'][field][:]
            centers_2 = 0.5*(segments_2[:, 0, 0] + segments_2[:, 1, 0])
            segments_2[:, :, 1] = data_2[:, np.newaxis]
            index_2 = np.argsort(centers_2)

            points_3       = mesh_3['points']
            connectivity_3 = mesh_3['connectivity']
            segments_3     = np.zeros((connectivity_3.shape[0], 2, 2))
            segments_3[:, :, 0] = points_3[:][connectivity_3[:]][:, :, 0]
            data_3    = mesh_3['fields'][field][:]
            centers_3 = 0.5*(segments_3[:, 0, 0] + segments_3[:, 1, 0])
            segments_3[:, :, 1] = data_3[:, np.newaxis]
            index_3 = np.argsort(centers_3)

            points_4       = mesh_4['points']
            connectivity_4 = mesh_4['connectivity']
            segments_4     = np.zeros((connectivity_4.shape[0], 2, 2))
            segments_4[:, :, 0] = points_4[:][connectivity_4[:]][:, :, 0]
            data_4    = mesh_4['fields'][field][:]
            centers_4 = 0.5*(segments_4[:, 0, 0] + segments_4[:, 1, 0])
            segments_4[:, :, 1] = data_4[:, np.newaxis]
            index_4 = np.argsort(centers_4)

            points_5       = mesh_5['points']
            connectivity_5 = mesh_5['connectivity']
            segments_5     = np.zeros((connectivity_5.shape[0], 2, 2))
            segments_5[:, :, 0] = points_5[:][connectivity_5[:]][:, :, 0]
            data_5    = mesh_5['fields'][field][:]
            centers_5 = 0.5*(segments_5[:, 0, 0] + segments_5[:, 1, 0])
            segments_5[:, :, 1] = data_5[:, np.newaxis]
            index_5 = np.argsort(centers_5)

            points_6       = mesh_6['points']
            connectivity_6 = mesh_6['connectivity']
            segments_6     = np.zeros((connectivity_6.shape[0], 2, 2))
            segments_6[:, :, 0] = points_6[:][connectivity_6[:]][:, :, 0]
            data_6    = mesh_6['fields'][field][:]
            centers_6 = 0.5*(segments_6[:, 0, 0] + segments_6[:, 1, 0])
            segments_6[:, :, 1] = data_6[:, np.newaxis]
            index_6 = np.argsort(centers_6)

            points_7       = mesh_7['points']
            connectivity_7 = mesh_7['connectivity']
            segments_7     = np.zeros((connectivity_7.shape[0], 2, 2))
            segments_7[:, :, 0] = points_7[:][connectivity_7[:]][:, :, 0]
            data_7     = mesh_7['fields'][field][:]
            centers_7  = .5*(segments_7[:, 0, 0] + segments_7[:, 1, 0])
            segments_7[:, :, 1] = data_7[:, np.newaxis]
            index_7 = np.argsort(centers_7)
            if init:
                self.artists.append(line_plot(ax, centers_1[index_1], data_1[index_1], \
                                                  centers_2[index_2], data_2[index_2], \
						                          centers_3[index_3], data_3[index_3], \
                                                  centers_4[index_4], data_4[index_4], \
                                                  centers_5[index_5], data_5[index_5], \
                                  		          centers_6[index_6], data_6[index_6], \
						                          centers_7[index_7], data_7[index_7]))
            else:
                line_update(self.artists[self.index], centers_1[index_1], data_1[index_1], \
                                                      centers_2[index_2], data_2[index_2], \
						                              centers_3[index_3], data_3[index_3], \
                                                      centers_4[index_4], data_4[index_4], \
                                                      centers_5[index_5], data_5[index_5], \
                                  		              centers_6[index_6], data_6[index_6], \
						                              centers_7[index_7], data_7[index_7])
                self.index += 1

        for aax in self.ax:
            aax.relim()
            aax.autoscale_view()

    def update(self, filename_1, filename_2, filename_3, \
                     filename_4, filename_5, filename_6, \
                     filename_7):
        mesh_1 = read_mesh(filename_1)
        self.index = 0
        if args.field is None:
            self.plot(None, mesh_1, init=False)
        else:
            mesh_2 = read_mesh(filename_2)
            mesh_3 = read_mesh(filename_3)
            mesh_4 = read_mesh(filename_4)
            mesh_5 = read_mesh(filename_5)
            mesh_6 = read_mesh(filename_6)
            mesh_7 = read_mesh(filename_7)

            for i, f in enumerate(args.field):
                self.plot(None, mesh_1, mesh_2, mesh_3, \
                                mesh_4, mesh_5, mesh_7, f, init=False)

    def get_artist(self):
        return self.artists

parser = argparse.ArgumentParser(description='Plot 1d mesh and field from samurai simulations.')
parser.add_argument('filename_1', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('filename_2', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('filename_3', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('filename_4', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('filename_5', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('filename_6', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('filename_7', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('--field', nargs="+", type=str, required=False, help='list of fields to plot')
parser.add_argument('--start', type=int, required=False, default=0, help='iteration start')
parser.add_argument('--end', type=int, required=False, default=None, help='iteration end')
parser.add_argument('--save', type=str, required=False, help='output file')
parser.add_argument('--wait', type=int, default=200, required=False, help='time between two plot in ms')
parser.add_argument('--analytical', type=str, required=False, help='analytical solution 5 equations model')
parser.add_argument('--column_analytical', type=int, required=False, help='variable of analytical solution 5 equations model')
parser.add_argument('--reference', type=str, required=False, help='reference results')
parser.add_argument('--column_reference', type=int, required=False, help='variable of reference results')
args = parser.parse_args()

if args.end is None:
    Plot(args.filename_1, \
         args.filename_2, \
         args.filename_3, \
         args.filename_4, \
         args.filename_5, args.filename_6, \
         args.filename_7)
else:
    p = Plot(f"{args.filename_1}{args.start}", \
             f"{args.filename_2}{args.start}", \
             f"{args.filename_3}{args.start}", \
             f"{args.filename_4}{args.start}", \
             f"{args.filename_5}{args.start}", \
             f"{args.filename_6}{args.start}", \
             f"{args.filename_7}{args.start}")
    def animate(i):
        p.fig.suptitle(f"iteration {i + args.start}")
        p.update(f"{args.filename_1}{i + args.start}", \
                 f"{args.filename_2}{i + args.start}", \
                 f"{args.filename_3}{i + args.start}", \
                 f"{args.filename_4}{i + args.start}", \
                 f"{args.filename_5}{i + args.start}", \
                 f"{args.filename_6}{i + args.start}", \
                 f"{args.filename_7}{i + args.start}")
        return p.get_artist()
    ani = animation.FuncAnimation(p.fig, animate, frames=args.end-args.start, interval=args.wait, repeat=True)

if args.save:
    if args.end is None:
        plt.savefig(args.save + '.png', dpi=300)
    else:
        writermp4 = animation.FFMpegWriter(fps=1)
        ani.save(args.save + '.mp4', dpi=300)
else:
    plt.show()
