// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#pragma once

// Declare a struct with the simulation parameters
// (domain, levels, final time, and Courant number)
//
template<typename T = double>
struct Simulation_Parameters {
  /*--- Physical parameters ---*/
  double xL;
  double xR;
  double yL;
  double yR;

  T t0;
  T Tf;

  bool apply_relaxation;

  /*--- Numerical parameters ---*/
  T Courant;
  T dt;

  bool apply_finite_rate_relaxation;
  bool splitting_in_relaxation;
  bool relax_instantaneous_velocity;
  bool relax_velocity;
  bool relax_pressure;
  bool relax_temperature;

  T    tau_u;
  T    tau_p;
  T    tau_T;

  T           atol_Newton_Suliciu;
  T           rtol_Newton_Suliciu;
  std::size_t max_Newton_iters;

  T atol_Newton_relaxation;
  T rtol_Newton_relaxation;

  /*--- MR parameters ---*/
  std::size_t min_level;
  std::size_t max_level;
  double      MR_param;
  double      MR_regularity;

  /*--- Output parameters ---*/
  std::string save_dir;
  std::size_t nfiles;

  /*--- Restart file ---*/
  std::string restart_file;
};

// Declare a struct with EOS parameters
//
template<typename T = double>
struct EOS_Parameters {
  /*--- SG-EOS parameters phase 1 ---*/
  T gamma_1;
  T pi_infty_1;
  T q_infty_1;
  T c_v_1;

  /*--- SG-EOS parameters phase 2 ---*/
  T gamma_2;
  T pi_infty_2;
  T q_infty_2;
  T c_v_2;
};

// Declare a struct with Riemann problem parameters
//
template<typename T = double>
struct Riemann_Parameters {
  /*--- Initial discontinuity location ---*/
  T xd;

  /*--- Left state ---*/
  T alpha1L;
  T rho1L;
  T p1L;
  T T1L;
  T u1L;
  T v1L;
  T rho2L;
  T p2L;
  T T2L;
  T u2L;
  T v2L;

  /*--- Right state ---*/
  T alpha1R;
  T rho1R;
  T p1R;
  T T1R;
  T u1R;
  T v1R;
  T rho2R;
  T p2R;
  T T2R;
  T u2R;
  T v2R;
};
