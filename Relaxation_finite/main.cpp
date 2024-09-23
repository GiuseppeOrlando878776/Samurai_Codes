// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#include <CLI/CLI.hpp>

#include "include/BN_solver.hpp"

// Main function to run the program
//
int main()
{
  CLI::App app{"Suliciu-type relaxation scheme for the 1D Baer-Nunziato model"};

  // Set and declare simulation parameters related to mesh, final time and Courant
  Simulation_Paramaters sim_param;
  sim_param.xL = 0.0;
  sim_param.xR = 1.0;
  sim_param.min_level = 7;
  sim_param.max_level = 7;
  sim_param.Tf = 0.007;
  sim_param.Courant = 0.45;
  sim_param.nfiles = 10;

  app.add_option("--cfl", sim_param.Courant, "The Courant number")->capture_default_str()->group("Simulation parameters");
  app.add_option("--Tf", sim_param.Tf, "Final time")->capture_default_str()->group("Simulation parameters");
  app.add_option("--xL", sim_param.xL, "Left-end of the domain")->capture_default_str()->group("Simulation parameters");
  app.add_option("--xR", sim_param.xR, "Right-end of the domain")->capture_default_str()->group("Simulation parameters");
  app.add_option("--min-level", sim_param.min_level, "Minimum level of the AMR")->capture_default_str()->group("AMR parameter");
  app.add_option("--max-level", sim_param.max_level, "Maximum level of the AMR")->capture_default_str()->group("AMR parameter");
  app.add_option("--nfiles", sim_param.nfiles, "Number of output files")->capture_default_str()->group("Ouput");

  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> min_corner = {sim_param.xL};
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> max_corner = {sim_param.xR};

  // Create the instance of the class to perform the simulation
  auto BN_Solver_Sim = BN_Solver(min_corner, max_corner, sim_param);

  BN_Solver_Sim.run();

  return 0;
}
