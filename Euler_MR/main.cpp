// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#include <CLI/CLI.hpp>

#include "include/Euler_MR.hpp"

// Main function to run the program
//
int main(int argc, char* argv[]) {
  CLI::App app{"Solver for the Euler equations with MR"};

  // Set and declare simulation parameters related to mesh, final time and Courant
  Simulation_Paramaters sim_param;

  sim_param.xL = 0.0;
  sim_param.xR = 1.0;

  sim_param.min_level = 7;
  sim_param.max_level = 10;
  sim_param.MR_param = 1e-2;
  sim_param.MR_regularity = 0;

  sim_param.Tf = 0.15;
  sim_param.Courant = 0.45;
  sim_param.nfiles = 10;

  app.add_option("--cfl", sim_param.Courant, "The Courant number")->capture_default_str()->group("Simulation parameters");
  app.add_option("--Tf", sim_param.Tf, "Final time")->capture_default_str()->group("Simulation parameters");
  app.add_option("--xL", sim_param.xL, "x Left-end of the domain")->capture_default_str()->group("Simulation parameters");
  app.add_option("--xR", sim_param.xR, "x Right-end of the domain")->capture_default_str()->group("Simulation parameters");
  app.add_option("--min-level", sim_param.min_level, "Minimum level of the AMR")->capture_default_str()->group("AMR parameter");
  app.add_option("--max-level", sim_param.max_level, "Maximum level of the AMR")->capture_default_str()->group("AMR parameter");
  app.add_option("--MR_param", sim_param.MR_param, "Multiresolution parameter")->capture_default_str()->group("AMR parameter");
  app.add_option("--MR_regularity", sim_param.MR_regularity, "Multiresolution regularity")->capture_default_str()->group("AMR parameter");
  app.add_option("--nfiles", sim_param.nfiles, "Number of output files")->capture_default_str()->group("Ouput");

  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> min_corner = {sim_param.xL};
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> max_corner = {sim_param.xR};

  // Set and declare simulation parameters related to EOS
  EOS_Parameters eos_param;

  eos_param.gamma = 1.4;
  eos_param.pi_infty = 0.0;
  eos_param.q_infty = 0.0;

  app.add_option("--gammma", eos_param.gamma, "gamma")->capture_default_str()->group("EOS parameters");
  app.add_option("--pi_infty", eos_param.pi_infty, "pi_infty")->capture_default_str()->group("EOS parameters");
  app.add_option("--q_infty", eos_param.q_infty, "q_infty")->capture_default_str()->group("EOS parameters");

  // Set and declare simulation parameters related to initial condition
  Riemann_Parameters Riemann_param;

  Riemann_param.xd = 0.5;

  Riemann_param.rhoL = 1.0;
  Riemann_param.pL   = 0.4;
  Riemann_param.uL   = -2.0;

  Riemann_param.rhoR = 1.0;
  Riemann_param.pR   = 0.4;
  Riemann_param.uR   = 2.0;

  app.add_option("--xd", Riemann_param.xd, "Initial discontinuity location")->capture_default_str()->group("Initial conditions");
  app.add_option("--rhoL", Riemann_param.rhoL, "Initial density at left")->capture_default_str()->group("Initial conditions");
  app.add_option("--pL", Riemann_param.pL, "Initial pressure at left")->capture_default_str()->group("Initial conditions");
  app.add_option("--uL", Riemann_param.uL, "Initial velocity at left")->capture_default_str()->group("Initial conditions");
  app.add_option("--rhoR", Riemann_param.rhoR, "Initial density at right")->capture_default_str()->group("Initial conditions");
  app.add_option("--pR", Riemann_param.pR, "Initial pressure at right")->capture_default_str()->group("Initial conditions");
  app.add_option("--uR", Riemann_param.uR, "Initial velocity at right")->capture_default_str()->group("Initial conditions");

  // Create the instance of the class to perform the simulation
  CLI11_PARSE(app, argc, argv);
  auto Euler_MR_sim = Euler_MR(min_corner, max_corner,
                               sim_param, eos_param,
                               Riemann_param);

  Euler_MR_sim.run();

  return 0;
}
