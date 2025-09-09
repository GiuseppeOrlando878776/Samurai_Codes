// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#pragma once

// Declare a struct with the simulation parameters
// (domain, levels, final time, and Courant number)
template<typename T = double>
struct Simulation_Paramaters {
  /*--- Physical paramters ---*/
  double xL;
  double xR;
  double yL;
  double yR;
  double zL;
  double zR;

  T t0;
  T Tf;

  bool apply_relaxation;

  T L0;
  T H0;
  T W0;

  /*--- Numerical parameters ---*/
  T Courant;

  T           lambda;
  T           atol_Newton;
  T           rtol_Newton;
  std::size_t max_Newton_iters;

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
