// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#ifndef containers_hpp
#define containers_hpp

// Declare a struct with the simulation parameters
// (domain, levels, final time, and Courant number)
//
struct Simulation_Parameters {
  double xL;
  double xR;
  double yL;
  double yR;

  std::size_t min_level;
  std::size_t max_level;

  double Tf;
  double Courant;

  std::size_t nfiles;

  bool   apply_pressure_relax;
  bool   apply_finite_rate_relax;
  double tau_p;
  bool   use_exact_relax;
};

// Declare a struct with EOS parameters
//
struct EOS_Parameters {
  double gamma_1;
  double pi_infty_1;
  double q_infty_1;
  double c_v_1;

  double gamma_2;
  double pi_infty_2;
  double q_infty_2;
  double c_v_2;
};

// Declare a struct with Riemann problem parameters
//
struct Riemann_Parameters {
  double xd;

  double alpha1L;
  double rho1L;
  double p1L;
  double uL;
  double vL;
  double rho2L;
  double p2L;

  double alpha1R;
  double rho1R;
  double p1R;
  double uR;
  double vR;
  double rho2R;
  double p2R;
};

#endif
