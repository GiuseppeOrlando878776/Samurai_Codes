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
  /*--- Physical parameters ---*/
  double xL;
  double xR;
  double yL;
  double yR;

  T t0;
  T Tf;

  /*--- Numerical parameters ---*/
  T Courant;

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
  T gamma;
  T pi_infty;
  T q_infty;
};

// Declare a struct with Riemann problem parameters
//
template<typename T = double>
struct Riemann_Parameters {
  T xd;

  T rhoL;
  T pL;
  T uL;
  T vL;

  T rhoR;
  T pR;
  T uR;
  T vR;
};

#endif
