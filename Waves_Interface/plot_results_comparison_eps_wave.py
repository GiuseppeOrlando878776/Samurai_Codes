# Copyright 2021 SAMURAI TEAM. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib import animation
from matplotlib import rc
from matplotlib import ticker
import argparse

def read_mesh(filename, ite=None):
    return h5py.File(filename + '.h5', 'r')['mesh']

def scatter_plot(ax, points):
    return ax.scatter(points[:, 0], points[:, 1], marker='+')

def scatter_update(scatter, points):
    scatter.set_offsets(points[:, :2])

def line_plot(ax, x_sharp, y_sharp, \
                  x_eps_025_dx, y_eps_025_dx, \
                  x_eps_1_dx, y_eps_1_dx, \
                  x_eps_3_dx, y_eps_3_dx, \
                  x_eps_10_dx, y_eps_10_dx):
    #Plot results
    plot_sharp      = ax.plot(x_sharp, y_sharp, 'k-', linewidth=2, markersize=4, alpha=1)[0]
    plot_eps_025_dx = ax.plot(x_eps_025_dx, y_eps_025_dx, 'ro--', linewidth=2, markersize=4, alpha=1, markevery=4)[0]
    plot_eps_1_dx   = ax.plot(x_eps_1_dx, y_eps_1_dx, 'bx--', linewidth=2, markersize=6, alpha=1, markevery=6)[0]
    plot_eps_3_dx   = ax.plot(x_eps_3_dx, y_eps_3_dx, 'gd--', linewidth=2, markersize=4, alpha=1, markevery=7)[0]
    plot_eps_10_dx  = ax.plot(x_eps_10_dx, y_eps_10_dx, 'ms--', linewidth=2, markersize=4, alpha=1, markevery=9)[0]

    #plt.xlim(0.65,0.75)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    plot_int = ax.axvline(x=0.7, color='b', linewidth=2, linestyle='--')
    plt.text(0.7,2e5,'Interface',fontsize=20)

    ax.legend([plot_sharp, plot_eps_025_dx, plot_eps_1_dx, plot_eps_3_dx, plot_eps_10_dx], \
              ['sharp', 'eps = 0.25dx', 'eps = dx', 'eps = 3dx', 'eps = 10dx'], \
               fontsize="28", loc="best")

    return plot_sharp, plot_eps_025_dx, plot_eps_1_dx, plot_eps_3_dx, plot_eps_10_dx

def line_update(lines, x_sharp_dx, y_sharp_dx, \
                       x_eps_025_dx, y_eps_025_dx, \
                       x_eps_1_dx, y_eps_1_dx, \
                       x_eps_3_dx, y_eps_3_dx, \
                       x_eps_10_dx, y_eps_10_dx):
    lines[1].set_data(x_eps_sharp, y_eps_sharp)
    lines[2].set_data(x_eps_025_dx, y_eps_025_dx)
    lines[3].set_data(x_eps_1_dx, y_eps_1_dx)
    lines[4].set_data(x_eps_3_dx, y_eps_3_dx)
    lines[5].set_data(x_eps_10_dx, y_eps_10_dx)

class Plot:
    def __init__(self, filename_sharp, \
                       filename_eps_025_dx, \
                       filename_eps_1_dx, \
                       filename_eps_3_dx, \
                       filename_eps_10_dx):
        self.fig     = plt.figure()
        self.artists = []
        self.ax      = []

        mesh_sharp = read_mesh(filename_sharp)
        if args.field is None:
            ax = plt.subplot(111)
            self.plot(ax, mesh_eps_025_dx)
            ax.set_title("Mesh")
            self.ax = [ax]
        else:
            mesh_eps_025_dx = read_mesh(filename_eps_025_dx)
            mesh_eps_1_dx   = read_mesh(filename_eps_1_dx)
            mesh_eps_3_dx   = read_mesh(filename_eps_3_dx)
            mesh_eps_10_dx  = read_mesh(filename_eps_10_dx)
            for i, f in enumerate(args.field):
                ax = plt.subplot(1, len(args.field), i + 1)
                self.plot(ax, mesh_sharp, mesh_eps_025_dx, mesh_eps_1_dx, mesh_eps_3_dx, mesh_eps_10_dx, f)
                ax.set_title(r"$p - p_{0}$",fontsize=24)

    def plot(self, ax, mesh_sharp, mesh_eps_025_dx, mesh_eps_1_dx, mesh_eps_3_dx, mesh_eps_10_dx, field=None, init=True):
        points_sharp       = mesh_sharp['points']
        connectivity_sharp = mesh_sharp['connectivity']

        segments_sharp = np.zeros((connectivity_sharp.shape[0], 2, 2))
        segments_sharp[:, :, 0] = points_sharp[:][connectivity_sharp[:]][:, :, 0]

        if field is None:
            segments_sharp[:, :, 1] = 0
            if init:
                self.artists.append(scatter_plot(ax, points_sharp))
                self.lc    = mc.LineCollection(segments_sharp, colors='b', linewidths=2)
                self.lines = ax.add_collection(self.lc)
            else:
                scatter_update(self.artists[self.index], points)
                self.index += 1
                # self.lc.set_array(segments)
        else:
            data_sharp    = mesh_sharp['fields'][field][:]
            centers_sharp = 0.5*(segments_sharp[:, 0, 0] + segments_sharp[:, 1, 0])
            segments_sharp[:, :, 1] = data_sharp[:, np.newaxis]
            # ax.scatter(centers, data, marker='+')
            index_sharp = np.argsort(centers_sharp)

            points_eps_025_dx       = mesh_eps_025_dx['points']
            connectivity_eps_025_dx = mesh_eps_025_dx['connectivity']
            segments_eps_025_dx     = np.zeros((connectivity_eps_025_dx.shape[0], 2, 2))
            segments_eps_025_dx[:, :, 0] = points_eps_025_dx[:][connectivity_eps_025_dx[:]][:, :, 0]
            data_eps_025_dx    = mesh_eps_025_dx['fields'][field][:]
            centers_eps_025_dx = 0.5*(segments_eps_025_dx[:, 0, 0] + segments_eps_025_dx[:, 1, 0])
            segments_eps_025_dx[:, :, 1] = data_eps_025_dx[:, np.newaxis]
            index_eps_025_dx = np.argsort(centers_eps_025_dx)

            points_eps_1_dx       = mesh_eps_1_dx['points']
            connectivity_eps_1_dx = mesh_eps_1_dx['connectivity']
            segments_eps_1_dx     = np.zeros((connectivity_eps_1_dx.shape[0], 2, 2))
            segments_eps_1_dx[:, :, 0] = points_eps_1_dx[:][connectivity_eps_1_dx[:]][:, :, 0]
            data_eps_1_dx    = mesh_eps_1_dx['fields'][field][:]
            centers_eps_1_dx = 0.5*(segments_eps_1_dx[:, 0, 0] + segments_eps_1_dx[:, 1, 0])
            segments_eps_1_dx[:, :, 1] = data_eps_1_dx[:, np.newaxis]
            index_eps_1_dx = np.argsort(centers_eps_1_dx)

            points_eps_3_dx       = mesh_eps_3_dx['points']
            connectivity_eps_3_dx = mesh_eps_3_dx['connectivity']
            segments_eps_3_dx     = np.zeros((connectivity_eps_3_dx.shape[0], 2, 2))
            segments_eps_3_dx[:, :, 0] = points_eps_3_dx[:][connectivity_eps_3_dx[:]][:, :, 0]
            data_eps_3_dx    = mesh_eps_3_dx['fields'][field][:]
            centers_eps_3_dx = 0.5*(segments_eps_3_dx[:, 0, 0] + segments_eps_3_dx[:, 1, 0])
            segments_eps_3_dx[:, :, 1] = data_eps_3_dx[:, np.newaxis]
            index_eps_3_dx = np.argsort(centers_eps_3_dx)

            points_eps_10_dx       = mesh_eps_10_dx['points']
            connectivity_eps_10_dx = mesh_eps_10_dx['connectivity']
            segments_eps_10_dx     = np.zeros((connectivity_eps_10_dx.shape[0], 2, 2))
            segments_eps_10_dx[:, :, 0] = points_eps_10_dx[:][connectivity_eps_10_dx[:]][:, :, 0]
            data_eps_10_dx    = mesh_eps_10_dx['fields'][field][:]
            centers_eps_10_dx = 0.5*(segments_eps_10_dx[:, 0, 0] + segments_eps_10_dx[:, 1, 0])
            segments_eps_10_dx[:, :, 1] = data_eps_10_dx[:, np.newaxis]
            index_eps_10_dx = np.argsort(centers_eps_10_dx)

            if init:
                self.artists.append(line_plot(ax, centers_sharp[index_sharp], data_sharp[index_sharp], \
                                                  centers_eps_025_dx[index_eps_025_dx], data_eps_025_dx[index_eps_025_dx], \
						                          centers_eps_1_dx[index_eps_1_dx], data_eps_1_dx[index_eps_1_dx], \
                                                  centers_eps_3_dx[index_eps_3_dx], data_eps_3_dx[index_eps_3_dx], \
                                                  centers_eps_10_dx[index_eps_10_dx], data_eps_10_dx[index_eps_10_dx]))
            else:
                line_update(self.artists[self.index], centers_sharp[index_sharp], data_sharp[index_sharp], \
                                                      centers_eps_025_dx[index_eps_025_dx], data_eps_025_dx[index_eps_025_dx], \
						                              centers_eps_1_dx[index_eps_1_dx], data_eps_1_dx[index_eps_1_dx], \
                                                      centers_eps_3_dx[index_eps_3_dx], data_eps_3_dx[index_eps_3_dx], \
                                                      centers_eps_10_dx[index_eps_10_dx], data_eps_10_dx[index_eps_10_dx])
                self.index += 1

        for aax in self.ax:
            aax.relim()
            aax.autoscale_view()

    def update(self, filename_sharp, filename_eps_025_dx, filename_eps_1_dx, filename_eps_3dx, filename_eps_10dx):
        mesh_sharp = read_mesh(filenamesharp)
        self.index = 0
        if args.field is None:
            self.plot(None, mesh_eps_025_dx, init=False)
        else:
            mesh_eps_025_dx = read_mesh(filename_eps_025_dx)
            mesh_eps_1_dx   = read_mesh(filename_eps_1_dx)
            mesh_eps_3_dx   = read_mesh(filename_eps_3_dx)
            mesh_eps_10_dx  = read_mesh(filename_eps_10_dx)

            for i, f in enumerate(args.field):
                self.plot(None, mesh_sharp, mesh_eps_025_dx, mesh_eps_1_dx, mesh_eps_3_dx, mesh_eps_10_dx, f, init=False)

    def get_artist(self):
        return self.artists

parser = argparse.ArgumentParser(description='Plot 1d mesh and field from samurai simulations.')
parser.add_argument('filename_sharp', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('filename_eps_025_dx', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('filename_eps_1_dx', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('filename_eps_3_dx', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('filename_eps_10_dx', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('--field', nargs="+", type=str, required=False, help='list of fields to plot')
parser.add_argument('--start', type=int, required=False, default=0, help='iteration start')
parser.add_argument('--end', type=int, required=False, default=None, help='iteration end')
parser.add_argument('--save', type=str, required=False, help='output file')
parser.add_argument('--wait', type=int, default=200, required=False, help='time between two plot in ms')
args = parser.parse_args()

if args.end is None:
    Plot(args.filename_sharp, args.filename_eps_025_dx, args.filename_eps_1_dx, args.filename_eps_3_dx, args.filename_eps_10_dx)
else:
    p = Plot(f"{args.filename_sharp}{args.start}", \
             f"{args.filename_eps_025_dx}{args.start}", \
             f"{args.filename_eps_1_dx}{args.start}", \
             f"{args.filename_eps_3_dx}{args.start}", \
             f"{args.filename_eps_10_dx}{args.start}")
    def animate(i):
        p.fig.suptitle(f"iteration {i + args.start}")
        p.update(f"{args.filename_sharp}{args.start}", \
                 f"{args.filename_eps_025_dx}{args.start}", \
                 f"{args.filename_eps_1_dx}{args.start}", \
                 f"{args.filename_eps_3_dx}{args.start}", \
                 f"{args.filename_eps_10_dx}{args.start}")
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
