// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#ifndef containers_hpp
#define containers_hpp

// Declare a struct with the simulation parameters
// (domain, levels, final time, and Courant number)
struct Simulation_Paramaters {
  double xL;
  double xR;
  double yL;
  double yR;
  std::size_t min_level;
  std::size_t max_level;
  double MR_param;
  double MR_regularity;

  double Tf;
  double Courant;

  std::size_t nfiles;
};

// Declare a struct with EOS parameters
struct EOS_Parameters {
  double gamma;
  double pi_infty;
  double q_infty;
};

// Declare a struct with Riemann problem parameters
struct Riemann_Parameters {
  double xd;

  double rhoL;
  double pL;
  double uL;
  double vL;

  double rhoR;
  double pR;
  double uR;
  double vR;
};


#endif
