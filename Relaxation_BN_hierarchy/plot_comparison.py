import numpy as np
import matplotlib.pyplot as plt

################################### WATER-AIR SHOCK TUBE ####################################
data_analytical         = np.genfromtxt('WATER_AIR_SHOCK_TUBE/exact_Euler_SG.dat')
data_numerical          = np.genfromtxt('WATER_AIR_SHOCK_TUBE/HOMOGENEOUS_6EQS/numerical_results_u.dat')
ax                      = plt.subplot(1, 1, 1)
plot_HLLC_BR_Orlando    = ax.plot(data_numerical[:,0], data_numerical[:,1], 'r-', linewidth=1, markersize=4, alpha=1)[0]
plot_HLLC               = ax.plot(data_numerical[:,4], data_numerical[:,5], 'b-', linewidth=1, markersize=4, alpha=1)[0]
plot_analytical         = ax.plot(data_analytical[:,0], data_analytical[:,2], 'k-', linewidth=1, markersize=4, alpha=1)[0]

ax.tick_params(axis='x',labelsize=20)
ax.tick_params(axis='y',labelsize=20)
ax.yaxis.get_offset_text().set_fontsize(20)
ax.set_title(r"$u$",fontsize=20)

ax.legend([plot_HLLC_BR_Orlando, \
           plot_HLLC, \
           plot_analytical], \
          ['HLLC + BR-2023', \
           'HLLC (wave-propagation)', \
           'Exact solution Riemann problem Euler equations'], fontsize="20", loc="best")
#plt.xlim(0.75, 0.95)
#plt.ylim(-20.0, 30.0)
plt.show()

#data_analytical         = np.genfromtxt('WATER_AIR_SHOCK_TUBE/exact.txt')
#data_numerical          = np.genfromtxt('WATER_AIR_SHOCK_TUBE/RELAXATION_5EQS/numerical_results_rho.dat')
#ax                      = plt.subplot(1, 1, 1)
#plot_Rusanov_BR_Orlando = ax.plot(data_numerical[:,0], data_numerical[:,1], 'm-',linewidth=1, markersize=4, alpha=1)[0]
#plot_Rusanov_BR_Tumolo  = ax.plot(data_numerical[:,2], data_numerical[:,3], 'go', linewidth=1, markersize=4, alpha=1, markevery=32)[0]
#plot_HLLC_BR_Orlando    = ax.plot(data_numerical[:,4], data_numerical[:,5], 'ro', linewidth=1, markersize=4, alpha=1, markevery=32)[0]
#plot_HLLC_BR_Tumolo     = ax.plot(data_numerical[:,6], data_numerical[:,7], 'o', color='orange', linewidth=1, markersize=4, alpha=1, markevery=32)[0]
#plot_HLLC               = ax.plot(data_numerical[:,8], data_numerical[:,9], 'b-', linewidth=1, markersize=4, alpha=1)[0]
#plot_analytical         = ax.plot(data_analytical[:,-1], data_analytical[:,1], 'k-', linewidth=1, markersize=4, alpha=1)[0]

#ax.tick_params(axis='x',labelsize=20)
#ax.tick_params(axis='y',labelsize=20)
#ax.yaxis.get_offset_text().set_fontsize(20)
#ax.set_title(r"$\rho$",fontsize=20)

#ax.legend([plot_Rusanov_BR_Orlando, plot_Rusanov_BR_Tumolo, \
#           plot_HLLC_BR_Orlando, plot_HLLC_BR_Tumolo, \
#           plot_HLLC, \
#           plot_analytical], \
#          ['Rusanov + BR-2023', 'Rusanov + BR-2015', \
#           'HLLC + BR-2023', 'HLLC + BR-2015', \
#           'HLLC (wave-propagation)', \
#           'Approximated jump relations solution Kapila model'], fontsize="20", loc="best")
#plt.xlim(0.80, 0.90)
#plt.ylim(-10.0, 15.0)
##plt.ylim(480.0, 500.0)
#plt.show()

################################### DOUBLE RAREFACTION ####################################
#data_analytical         = np.genfromtxt('DOUBLE_RAREFACTION/analytical_results.dat')
#data_numerical          = np.genfromtxt('DOUBLE_RAREFACTION/numerical_results_rho.dat')
#ax                      = plt.subplot(1, 1, 1)
#plot_Rusanov_BR_Orlando = ax.plot(data_numerical[:,0], data_numerical[:,1], 'r-',linewidth=1, markersize=4, alpha=1)[0]
#plot_HLLC_BR_Orlando    = ax.plot(data_numerical[:,2], data_numerical[:,3], 'go', linewidth=1, markersize=4, alpha=1, markevery=256)[0]
#plot_HLLC               = ax.plot(data_numerical[:,4], data_numerical[:,5], 'b-', linewidth=1, markersize=4, alpha=1)[0]
#plot_analytical         = ax.plot(data_analytical[:,0], data_analytical[:,4], 'k-', linewidth=1, markersize=4, alpha=1)[0]

#ax.tick_params(axis='x',labelsize=20)
#ax.tick_params(axis='y',labelsize=20)
#ax.yaxis.get_offset_text().set_fontsize(20)
#ax.set_title(r"$\rho$",fontsize=20)

#ax.legend([plot_Rusanov_BR_Orlando, \
#           plot_HLLC_BR_Orlando, \
#           plot_HLLC, \
#           plot_analytical], \
#          ['Rusanov + BR-2023', \
#           'HLLC + BR-2023', 'HLLC (wave-propagation)', \
#           'Analytical solution'], fontsize="20", loc="best")
#plt.show()

################################### EPOXY-SPINEL SHOCK ####################################
#data_analytical         = np.genfromtxt('EPOXY_SPINEL_SHOCK/exact.txt')
#data_numerical          = np.genfromtxt('EPOXY_SPINEL_SHOCK/RELAXATION_5EQS/numerical_results_u.dat')
#ax                      = plt.subplot(1, 1, 1)
#plot_Rusanov_BR_Orlando = ax.plot(data_numerical[:,0], data_numerical[:,1], 'r-',linewidth=1, markersize=4, alpha=1)[0]
#plot_HLLC_BR_Orlando    = ax.plot(data_numerical[:,2], data_numerical[:,3], 'go', linewidth=1, markersize=4, alpha=1, markevery=256)[0]
#plot_HLLC               = ax.plot(data_numerical[:,4], data_numerical[:,5], 'b-', linewidth=1, markersize=4, alpha=1)[0]
#plot_analytical         = ax.plot(data_analytical[:,-1], data_analytical[:,2], 'k-', linewidth=1, markersize=4, alpha=1)[0]

#ax.tick_params(axis='x',labelsize=20)
#ax.tick_params(axis='y',labelsize=20)
#ax.yaxis.get_offset_text().set_fontsize(20)
#ax.set_title(r"$u$",fontsize=20)

#ax.legend([plot_Rusanov_BR_Orlando, \
#           plot_HLLC_BR_Orlando, \
#           plot_HLLC, \
#           plot_analytical], \
#          [r'Rusanov + BR-2023', \
#           r'HLLC + BR-2023', 'HLLC (wave-propagation)', \
#           'Approximated jump relations solution Kapila model'], fontsize="20", loc="best")
#plt.show()
