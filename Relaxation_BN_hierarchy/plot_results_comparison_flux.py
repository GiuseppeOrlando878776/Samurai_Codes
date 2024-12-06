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

def line_plot(ax, x_Rusanov_BR_Orlando, y_Rusanov_BR_Orlando, \
                  x_Rusanov_BR_Tumolo, y_Rusanov_BR_Tumolo, \
                  x_Rusanov_centered, y_Rusanov_centered, \
                  x_HLLC_BR_Orlando, y_HLLC_BR_Orlando, \
                  x_HLLC_BR_Tumolo, y_HLLC_BR_Tumolo, \
                  x_HLLC_centered, y_HLLC_centered, \
                  x_HLLC, y_HLLC):
    #Plot results
    plot_Rusanov_BR_Orlando = ax.plot(x_Rusanov_BR_Orlando, y_Rusanov_BR_Orlando, 'b-', linewidth=1, markersize=4)[0]
    plot_Rusanov_BR_Tumolo  = ax.plot(x_Rusanov_BR_Tumolo, y_Rusanov_BR_Tumolo, 'ro', linewidth=1, markersize=4, alpha=1, markevery=3)[0]
    plot_Rusanov_centered   = ax.plot(x_Rusanov_centered, y_Rusanov_centered, 'gx', linewidth=1, markersize=4, alpha=1, markevery=2)[0]
    plot_HLLC_BR_Orlando    = ax.plot(x_HLLC_BR_Orlando, y_HLLC_BR_Orlando, 'ys', linewidth=1, markersize=4, alpha=1, markevery=6)[0]
    plot_HLLC_BR_Tumolo     = ax.plot(x_HLLC_BR_Tumolo, y_HLLC_BR_Tumolo, 'o', color='orange', linewidth=1, markersize=4, alpha=1, markevery=6)[0]
    plot_HLLC_centered      = ax.plot(x_HLLC_centered, y_HLLC_centered, 'bd', linewidth=1, markersize=4, alpha=1, markevery=8)[0]
    plot_HLLC               = ax.plot(x_HLLC, y_HLLC, 'k-', linewidth=1, markersize=4, alpha=1)[0]

    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

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
            ax.legend([plot_Rusanov_BR_Orlando, plot_Rusanov_BR_Tumolo, plot_Rusanov_centered, \
                       plot_HLLC_BR_Orlando, plot_HLLC_BR_Tumolo, plot_HLLC_centered, plot_HLLC, \
                       plot_analytical, plot_ref], \
                      ['Rusanov + BR (34)', 'Rusanov + BR (35)', 'Rusanov + Crouzet et al.', \
                       'HLLC + BR (34)', 'HLLC + BR (35)', 'HLLC + Crouzet et al.', 'HLLC (wave-propagation)', \
                       'Analytical (5 equations)', 'Reference results'], fontsize="20", loc="best")
        else:
            ax.legend([plot_Rusanov_BR_Orlando, plot_Rusanov_BR_Tumolo, plot_Rusanov_centered, \
                       plot_HLLC_BR_Orlando, plot_HLLC_BR_Tumolo, plot_HLLC_centered, plot_HLLC, \
                       plot_analytical], \
                      ['Rusanov + BR (34)', 'Rusanov + BR (35)', 'Rusanov + Crouzet et al.', \
                       'HLLC + BR (34)', 'HLLC + BR (35)', 'HLLC + Crouzet et al.', 'HLLC (wave-propagation)', \
                       'Analytical (5 equations)'], fontsize="20", loc="best")
    elif args.reference is not None:
        ax.legend([plot_Rusanov_BR_Orlando, plot_Rusanov_BR_Tumolo, plot_Rusanov_centered, \
                   plot_HLLC_BR_Orlando, plot_HLLC_BR_Tumolo, plot_HLLC_centered, plot_HLLC, \
                   plot_ref], \
                  ['Rusanov + BR (34)', 'Rusanov + BR (35)', 'Rusanov + Crouzet et al.', \
                   'HLLC + BR (34)', 'HLLC + BR (35)', 'HLLC + Crouzet et al.', 'HLLC (wave-propagation)', \
                   'Reference results'], fontsize="20", loc="best")
    else:
        ax.legend([plot_Rusanov_BR_Orlando, plot_Rusanov_BR_Tumolo, plot_Rusanov_centered, \
                   plot_HLLC_BR_Orlando, plot_HLLC_BR_Tumolo, plot_HLLC_centered, plot_HLLC], \
                  ['Rusanov + BR (34)', 'Rusanov + BR (35)', 'Rusanov + Crouzet et al.', \
                   'HLLC + BR (34)', 'HLLC + BR (35)', 'HLLC + Crouzet et al.', 'HLLC (wave-propagation)'], fontsize="20", loc="best")

    return plot_Rusanov_BR_Orlando, plot_Rusanov_BR_Tumolo, plot_Rusanov_centered, \
           plot_HLLC_BR_Orlando, plot_HLLC_BR_Tumolo, plot_HLLC_centered, plot_HLLC

def line_update(lines, x_Rusanov_BR_Orlando, y_Rusanov_BR_Orlando, \
                       x_Rusanov_BR_Tumolo, y_Rusanov_BR_Tumolo, \
                       x_Rusanov_centered, y_Rusanov_centered, \
                       x_HLLC_BR_Orlando, y_HLLC_BR_Orlando, \
                       x_HLLC_BR_Tumolo, y_HLLC_BR_Tumolo, \
                       x_HLLC_centered, y_HLLC_centered, \
                       x_HLLC, y_HLLC):
    lines[1].set_data(x_Rusanov_BR_Orlando, y_Rusanov_BR_Orlando)
    lines[2].set_data(x_Rusanov_BR_Tumolo, y_Rusanov_BR_Tumolo)
    lines[3].set_data(x_Rusanov_centered, y_Rusanov_centered)
    lines[4].set_data(x_HLLC_BR_Orlando, y_HLLC_BR_Orlando)
    lines[5].set_data(x_HLLC_BR_Tumolo, y_HLLC_BR_Tumolo)
    lines[6].set_data(x_HLLC_centered, y_HLLC_centered)
    lines[7].set_data(x_HLLC, y_HLLC)

class Plot:
    def __init__(self, filename_Rusanov_BR_Orlando, \
                       filename_Rusanov_BR_Tumolo, \
                       filename_Rusanov_centered, \
                       filename_HLLC_BR_Orlando, \
                       filename_HLLC_BR_Tumolo, \
                       filename_HLLC_centered, \
                       filename_HLLC):
        self.fig = plt.figure()
        self.artists = []
        self.ax = []

        mesh_Rusanov_BR_Orlando = read_mesh(filename_Rusanov_BR_Orlando)
        if args.field is None:
            ax = plt.subplot(111)
            self.plot(ax, mesh_Rusanov_BR_Orlando)
            ax.set_title("Mesh")
            self.ax = [ax]
        else:
            mesh_Rusanov_BR_Tumolo = read_mesh(filename_Rusanov_BR_Tumolo)
            mesh_Rusanov_centered = read_mesh(filename_Rusanov_centered)
            mesh_HLLC_BR_Orlando = read_mesh(filename_HLLC_BR_Orlando)
            mesh_HLLC_BR_Tumolo = read_mesh(filename_HLLC_BR_Tumolo)
            mesh_HLLC_centered = read_mesh(filename_HLLC_centered)
            mesh_HLLC = read_mesh(filename_HLLC)
            for i, f in enumerate(args.field):
                ax = plt.subplot(1, len(args.field), i + 1)
                self.plot(ax, mesh_Rusanov_BR_Orlando, mesh_Rusanov_BR_Tumolo, mesh_Rusanov_centered, \
                              mesh_HLLC_BR_Orlando, mesh_HLLC_BR_Tumolo, mesh_HLLC_centered, mesh_HLLC, f)
                ax.set_title(r"$\alpha_{1}$",fontsize=20)

    def plot(self, ax, mesh_Rusanov_BR_Orlando, \
                       mesh_Rusanov_BR_Tumolo=None, \
                       mesh_Rusanov_centered=None, \
                       mesh_HLLC_BR_Orlando=None, \
                       mesh_HLLC_BR_Tumolo=None, \
                       mesh_HLLC_centered=None, \
                       mesh_HLLC=None, field=None, init=True):
        points_Rusanov_BR_Orlando       = mesh_Rusanov_BR_Orlando['points']
        connectivity_Rusanov_BR_Orlando = mesh_Rusanov_BR_Orlando['connectivity']

        segments_Rusanov_BR_Orlando = np.zeros((connectivity_Rusanov_BR_Orlando.shape[0], 2, 2))
        segments_Rusanov_BR_Orlando[:, :, 0] = points_Rusanov_BR_Orlando[:][connectivity_Rusanov_BR_Orlando[:]][:, :, 0]

        if field is None:
            segments_Rusanov_BR_Orlando[:, :, 1] = 0
            if init:
                self.artists.append(scatter_plot(ax, points))
                self.lc = mc.LineCollection(segments_Rusanov_BR_Orlando, colors='b', linewidths=2)
                self.lines = ax.add_collection(self.lc)
            else:
                scatter_update(self.artists[self.index], points)
                self.index += 1
                # self.lc.set_array(segments)
        else:
            data_Rusanov_BR_Orlando    = mesh_Rusanov_BR_Orlando['fields'][field][:]
            centers_Rusanov_BR_Orlando = 0.5*(segments_Rusanov_BR_Orlando[:, 0, 0] + segments_Rusanov_BR_Orlando[:, 1, 0])
            segments_Rusanov_BR_Orlando[:, :, 1] = data_Rusanov_BR_Orlando[:, np.newaxis]
            # ax.scatter(centers, data, marker='+')
            index_Rusanov_BR_Orlando = np.argsort(centers_Rusanov_BR_Orlando)

            points_Rusanov_BR_Tumolo       = mesh_Rusanov_BR_Tumolo['points']
            connectivity_Rusanov_BR_Tumolo = mesh_Rusanov_BR_Tumolo['connectivity']
            segments_Rusanov_BR_Tumolo     = np.zeros((connectivity_Rusanov_BR_Tumolo.shape[0], 2, 2))
            segments_Rusanov_BR_Tumolo[:, :, 0] = points_Rusanov_BR_Tumolo[:][connectivity_Rusanov_BR_Tumolo[:]][:, :, 0]
            data_Rusanov_BR_Tumolo    = mesh_Rusanov_BR_Tumolo['fields'][field][:]
            centers_Rusanov_BR_Tumolo = 0.5*(segments_Rusanov_BR_Tumolo[:, 0, 0] + segments_Rusanov_BR_Tumolo[:, 1, 0])
            segments_Rusanov_BR_Tumolo[:, :, 1] = data_Rusanov_BR_Tumolo[:, np.newaxis]
            index_Rusanov_BR_Tumolo = np.argsort(centers_Rusanov_BR_Tumolo)

            points_Rusanov_centered       = mesh_Rusanov_centered['points']
            connectivity_Rusanov_centered = mesh_Rusanov_centered['connectivity']
            segments_Rusanov_centered     = np.zeros((connectivity_Rusanov_centered.shape[0], 2, 2))
            segments_Rusanov_centered[:, :, 0] = points_Rusanov_centered[:][connectivity_Rusanov_centered[:]][:, :, 0]
            data_Rusanov_centered    = mesh_Rusanov_centered['fields'][field][:]
            centers_Rusanov_centered = 0.5*(segments_Rusanov_centered[:, 0, 0] + segments_Rusanov_centered[:, 1, 0])
            segments_Rusanov_centered[:, :, 1] = data_Rusanov_centered[:, np.newaxis]
            index_Rusanov_centered = np.argsort(centers_Rusanov_centered)

            points_HLLC_BR_Orlando       = mesh_HLLC_BR_Orlando['points']
            connectivity_HLLC_BR_Orlando = mesh_HLLC_BR_Orlando['connectivity']
            segments_HLLC_BR_Orlando     = np.zeros((connectivity_HLLC_BR_Orlando.shape[0], 2, 2))
            segments_HLLC_BR_Orlando[:, :, 0] = points_HLLC_BR_Orlando[:][connectivity_HLLC_BR_Orlando[:]][:, :, 0]
            data_HLLC_BR_Orlando    = mesh_HLLC_BR_Orlando['fields'][field][:]
            centers_HLLC_BR_Orlando = 0.5*(segments_HLLC_BR_Orlando[:, 0, 0] + segments_HLLC_BR_Orlando[:, 1, 0])
            segments_HLLC_BR_Orlando[:, :, 1] = data_HLLC_BR_Orlando[:, np.newaxis]
            index_HLLC_BR_Orlando = np.argsort(centers_HLLC_BR_Orlando)

            points_HLLC_BR_Tumolo       = mesh_HLLC_BR_Tumolo['points']
            connectivity_HLLC_BR_Tumolo = mesh_HLLC_BR_Tumolo['connectivity']
            segments_HLLC_BR_Tumolo     = np.zeros((connectivity_HLLC_BR_Tumolo.shape[0], 2, 2))
            segments_HLLC_BR_Tumolo[:, :, 0] = points_HLLC_BR_Tumolo[:][connectivity_HLLC_BR_Tumolo[:]][:, :, 0]
            data_HLLC_BR_Tumolo    = mesh_HLLC_BR_Tumolo['fields'][field][:]
            centers_HLLC_BR_Tumolo = 0.5*(segments_HLLC_BR_Tumolo[:, 0, 0] + segments_HLLC_BR_Tumolo[:, 1, 0])
            segments_HLLC_BR_Tumolo[:, :, 1] = data_HLLC_BR_Tumolo[:, np.newaxis]
            index_HLLC_BR_Tumolo = np.argsort(centers_HLLC_BR_Tumolo)

            points_HLLC_centered       = mesh_HLLC_centered['points']
            connectivity_HLLC_centered = mesh_HLLC_centered['connectivity']
            segments_HLLC_centered     = np.zeros((connectivity_HLLC_centered.shape[0], 2, 2))
            segments_HLLC_centered[:, :, 0] = points_HLLC_centered[:][connectivity_HLLC_centered[:]][:, :, 0]
            data_HLLC_centered    = mesh_HLLC_centered['fields'][field][:]
            centers_HLLC_centered = 0.5*(segments_HLLC_centered[:, 0, 0] + segments_HLLC_centered[:, 1, 0])
            segments_HLLC_centered[:, :, 1] = data_HLLC_centered[:, np.newaxis]
            index_HLLC_centered = np.argsort(centers_HLLC_centered)

            points_HLLC       = mesh_HLLC['points']
            connectivity_HLLC = mesh_HLLC['connectivity']
            segments_HLLC     = np.zeros((connectivity_HLLC.shape[0], 2, 2))
            segments_HLLC[:, :, 0] = points_HLLC[:][connectivity_HLLC[:]][:, :, 0]
            data_HLLC     = mesh_HLLC['fields'][field][:]
            centers_HLLC  = .5*(segments_HLLC[:, 0, 0] + segments_HLLC[:, 1, 0])
            segments_HLLC[:, :, 1] = data_HLLC[:, np.newaxis]
            index_HLLC = np.argsort(centers_HLLC)
            if init:
                self.artists.append(line_plot(ax, centers_Rusanov_BR_Orlando[index_Rusanov_BR_Orlando], data_Rusanov_BR_Orlando[index_Rusanov_BR_Orlando], \
                                                  centers_Rusanov_BR_Tumolo[index_Rusanov_BR_Tumolo], data_Rusanov_BR_Tumolo[index_Rusanov_BR_Tumolo], \
						                          centers_Rusanov_centered[index_Rusanov_centered], data_Rusanov_centered[index_Rusanov_centered], \
                                                  centers_HLLC_BR_Orlando[index_HLLC_BR_Orlando], data_HLLC_BR_Orlando[index_HLLC_BR_Orlando], \
                                                  centers_HLLC_BR_Tumolo[index_HLLC_BR_Tumolo], data_HLLC_BR_Tumolo[index_HLLC_BR_Tumolo], \
                                  		          centers_HLLC_centered[index_HLLC_centered], data_HLLC_centered[index_HLLC_centered], \
						                          centers_HLLC[index_HLLC], data_HLLC[index_HLLC]))
            else:
                line_update(self.artists[self.index], centers_Rusanov_BR_Orlando[index_Rusanov_BR_Orlando], data_Rusanov_BR_Orlando[index_Rusanov_BR_Orlando], \
                                                      centers_Rusanov_BR_Tumolo[index_Rusanov_BR_Tumolo], data_Rusanov_BR_Tumolo[index_Rusanov_BR_Tumolo], \
						                              centers_Rusanov_centered[index_Rusanov_centered], data_Rusanov_centered[index_Rusanov_centered], \
                                                      centers_HLLC_BR_Orlando[index_HLLC_BR_Orlando], data_HLLC_BR_Orlando[index_HLLC_BR_Orlando], \
                                                      centers_HLLC_BR_Tumolo[index_HLLC_BR_Tumolo], data_HLLC_BR_Tumolo[index_HLLC_BR_Tumolo], \
                                  		              centers_HLLC_centered[index_HLLC_centered], data_HLLC_centered[index_HLLC_centered], \
						                              centers_HLLC[index_HLLC], data_HLLC[index_HLLC])
                self.index += 1

        for aax in self.ax:
            aax.relim()
            aax.autoscale_view()

    def update(self, filename_Rusanov_BR_Orlando, filename_Rusanov_BR_Tumolo, filename_Rusanov_centered, \
                     filename_HLLC_BR_Orlando, filename_HLLC_BR_Tumolo, filename_HLLC_centered, \
                     filename_HLLC):
        mesh_Rusanov_BR_Orlando = read_mesh(filename_Rusanov_BR_Orlando)
        self.index = 0
        if args.field is None:
            self.plot(None, mesh_Rusanov_BR_Orlando, init=False)
        else:
            mesh_Rusanov_BR_Tumolo = read_mesh(filename_Rusanov_BR_Tumolo)
            mesh_Rusanov_centered = read_mesh(filename_Rusanov_centered)
            mesh_HLLC_BR_Orlando = read_mesh(filename_HLLC_BR_Orlando)
            mesh_HLLC_BR_Tumolo = read_mesh(filename_HLLC_BR_Tumolo)
            mesh_HLLC_centered = read_mesh(filename_HLLC_centered)
            mesh_HLLC = read_mesh(filename_HLLC)

            for i, f in enumerate(args.field):
                self.plot(None, mesh_Rusanov_BR_Orlando, mesh_Rusanov_BR_Tumolo, mesh_Rusanov_centered, \
                                mesh_HLLC_BR_Orlando, mesh_HLLC_BR_Tumolo, mesh_HLLC, f, init=False)

    def get_artist(self):
        return self.artists

parser = argparse.ArgumentParser(description='Plot 1d mesh and field from samurai simulations.')
parser.add_argument('filename_Rusanov_BR_Orlando', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('filename_Rusanov_BR_Tumolo', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('filename_Rusanov_centered', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('filename_HLLC_BR_Orlando', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('filename_HLLC_BR_Tumolo', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('filename_HLLC_centered', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('filename_HLLC', type=str, help='hdf5 file to plot without .h5 extension')
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
    Plot(args.filename_Rusanov_BR_Orlando, \
         args.filename_Rusanov_BR_Tumolo, \
         args.filename_Rusanov_centered, \
         args.filename_HLLC_BR_Orlando, \
         args.filename_HLLC_BR_Tumolo, args.filename_HLLC_centered, \
         args.filename_HLLC)
else:
    p = Plot(f"{args.filename_Rusanov_BR_Orlando}{args.start}", \
             f"{args.filename_Rusanov_BR_Tumolo}{args.start}", \
             f"{args.filename_Rusanov_centered}{args.start}", \
             f"{args.filename_HLLC_BR_Orlando}{args.start}", \
             f"{args.filename_HLLC_BR_Tumolo}{args.start}", \
             f"{args.filename_HLLC_centered}{args.start}", \
             f"{args.filename_HLLC}{args.start}")
    def animate(i):
        p.fig.suptitle(f"iteration {i + args.start}")
        p.update(f"{args.filename_Rusanov_BR_Orlando}{i + args.start}", \
                 f"{args.filename_Rusanov_BR_Tumolo}{i + args.start}", \
                 f"{args.filename_Rusanov_centered}{args.start}", \
                 f"{args.filename_HLLC_BR_Orlando}{i + args.start}", \
                 f"{args.filename_HLLC_BR_Tumolo}{i + args.start}", \
                 f"{args.filename_HLLC_centered}{args.start}", \
                 f"{args.filename_HLLC}{args.start}")
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
