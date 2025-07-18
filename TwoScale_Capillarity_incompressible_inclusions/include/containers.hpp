// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#ifndef containers_hpp
#define containers_hpp

// Declare a struct with the simulation parameters
// (domain, levels, final time, and Courant number)
//
struct Simulation_Paramaters {
  /*--- Physical paramters ---*/
  double xL;
  double xR;
  double yL;
  double yR;

  double Tf;

  double sigma;

  bool   apply_relaxation;
  bool   mass_transfer;
  double Hmax;
  double kappa;
  double alpha1d_max;
  double alpha1_min;
  double alpha1_max;

  double x0;
  double y0;
  double U0;
  double U1;
  double V0;
  double R;
  double eps_over_R;

  /*--- Numerical parameters ---*/
  double Courant;

  double alpha_residual;
  double mod_grad_alpha1_min;

  double      lambda;
  double      atol_Newton;
  double      rtol_Newton;
  std::size_t max_Newton_iters;

  /*--- MR parameters ---*/
  std::size_t min_level;
  std::size_t max_level;
  double      MR_param;
  double      MR_regularity;

  /*--- Output parameters ---*/
  std::size_t nfiles;
};

// Declare a struct with EOS parameters
//
struct EOS_Parameters {
  double p0_phase1;
  double rho0_phase1;
  double c0_phase1;

  double p0_phase2;
  double rho0_phase2;
  double c0_phase2;
};

#endif
