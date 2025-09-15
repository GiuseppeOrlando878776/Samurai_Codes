// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#pragma once

// Declare a struct with the simulation parameters
//
template<typename T = double>
struct Simulation_Parameters {
  /*--- Physical parameters ---*/
  double xL;
  double xR;

  T t0;
  T Tf;

  bool apply_pressure_relax;

  /*--- Numerical parameters ---*/
  T Courant;

  bool apply_finite_rate_relax;
  T    tau_p;
  bool use_exact_relax;

  /*--- MR parameters ---*/
  std::size_t min_level;
  std::size_t max_level;

  /*--- Output parameters ---*/
  std::size_t nfiles;
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
  /*--- Initial discontinuity location ----*/
  T xd;

  /*--- Left state Riemann problem ---*/
  T alpha1L;
  T rho1L;
  T p1L;
  T uL;
  T rho2L;
  T p2L;

  /*--- Right state Riemann problem ---*/
  T alpha1R;
  T rho1R;
  T p1R;
  T uR;
  T rho2R;
  T p2R;
};
