// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#ifndef containers_hpp
#define containers_hpp

// Declare a struct with the simulation parameters
// (domain, levels, final time, and Courant number)
//
template<typename T = double>
struct Simulation_Paramaters {
  /*--- Physical paramters ---*/
  double xL;
  double xR;
  double yL;
  double yR;

  T t0;
  T Tf;

  T sigma;

  bool   apply_relaxation;

  T x0;
  T y0;
  T U0;
  T U1;
  T V0;
  T R;
  T eps_over_R;

  /*--- Numerical parameters ---*/
  T Courant;

  T alpha_residual;
  T mod_grad_alpha1_min;

  T           lambda;
  T           atol_Newton;
  T           rtol_Newton;
  std::size_t max_Newton_iters;

  T atol_Newton_p_star;
  T rtol_Newton_p_star;

  /*--- MR parameters ---*/
  std::size_t min_level;
  std::size_t max_level;
  double      MR_param;
  double      MR_regularity;

  /*--- Output parameters ---*/
  std::size_t nfiles;

  /*--- Restart file ---*/
  std::string restart_file;
};

// Declare a struct with EOS parameters
//
template<typename T = double>
struct EOS_Parameters {
  T p0_phase1;
  T rho0_phase1;
  T c0_phase1;

  T p0_phase2;
  T rho0_phase2;
  T c0_phase2;
};

#endif
