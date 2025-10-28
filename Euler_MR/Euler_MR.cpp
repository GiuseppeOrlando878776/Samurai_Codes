// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#include <CLI/CLI.hpp>

#include <nlohmann/json.hpp>

#include "include/Euler_MR.hpp"

// Main function to run the program
//
int main(int argc, char* argv[]) {
  using json = nlohmann::json;

  auto& app = samurai::initialize("Explicit finite volume solver for the Euler equations", argc, argv);

  std::ifstream ifs("input.json"); // Read a JSON file
  json input = json::parse(ifs);

  /*--- Set and declare some simulation parameters ---*/
  using Number = Euler_MR<EquationData::dim>::Number;
  Simulation_Paramaters<Number> sim_param;

  // Physical parameters
  sim_param.xL = input.value("xL", 0.0);
  sim_param.xR = input.value("xR", 1.0);

  sim_param.t0 = input.value("t0", 0.0);
  sim_param.Tf = input.value("Tf", 0.15);

  // Numerical parameters
  sim_param.Courant   = input.value("cfl", 0.45);
  sim_param.flux_name = input.value("flux_name", "Rusanov");

  // MR parameters
  sim_param.min_level     = input.value("min-level", 7);
  sim_param.max_level     = input.value("max-level", 10);
  sim_param.MR_param      = input.value("MR_param", 1e-2);
  sim_param.MR_regularity = input.value("MR_regularity", 0);

  // Output parameters
  sim_param.nfiles = input.value("nfiles", 10);

  // Restart file
  sim_param.restart_file = input.value("restart_file","");

  /*--- Allow for parsing from command line ---*/
  // Physical parameters
  app.add_option("--xL", sim_param.xL, "x Left-end of the domain")->capture_default_str()->group("Physical parameters");
  app.add_option("--xR", sim_param.xR, "x Right-end of the domain")->capture_default_str()->group("Physical parameters");

  app.add_option("--t0", sim_param.t0, "Initial time")->capture_default_str()->group("Physical parameters");
  app.add_option("--Tf", sim_param.Tf, "Final time")->capture_default_str()->group("Physical parameters");

  // Numerical parameters
  app.add_option("--cfl", sim_param.Courant, "The Courant number")->capture_default_str()->group("Numerical parameters");
  app.add_option("--flux_name", sim_param.flux_name, "Desired numerical flux")->capture_default_str()->group("Numerical parameters");

  // MR parameters
  app.add_option("--min-level", sim_param.min_level, "Minimum level of the AMR")->capture_default_str()->group("AMR parameter");
  app.add_option("--max-level", sim_param.max_level, "Maximum level of the AMR")->capture_default_str()->group("AMR parameter");
  app.add_option("--MR_param", sim_param.MR_param, "Multiresolution parameter")->capture_default_str()->group("AMR parameter");
  app.add_option("--MR_regularity", sim_param.MR_regularity, "Multiresolution regularity")->capture_default_str()->group("AMR parameter");

  // Output parameters
  app.add_option("--nfiles", sim_param.nfiles, "Number of output files")->capture_default_str()->group("Ouput");

  // Restart file
  app.add_option("--restart_file", sim_param.restart_file, "Name of the restart file")->capture_default_str()->group("Restart");

  /*--- Set and declare simulation parameters related to EOS ---*/
  EOS_Parameters<Number> eos_param;

  eos_param.gamma    = input.value("gamma", 1.4);
  eos_param.pi_infty = input.value("pi_infty", 0.0);
  eos_param.q_infty  = input.value("q_infty", 0.0);

  /*--- Allow for parsing from command line ---*/
  app.add_option("--gammma", eos_param.gamma, "gamma")->capture_default_str()->group("EOS parameters");
  app.add_option("--pi_infty", eos_param.pi_infty, "pi_infty")->capture_default_str()->group("EOS parameters");
  app.add_option("--q_infty", eos_param.q_infty, "q_infty")->capture_default_str()->group("EOS parameters");

  /*--- Set and declare simulation parameters related to initial condition ---*/
  Riemann_Parameters<Number> Riemann_param;

  Riemann_param.xd   = input.value("xd", 0.5);

  Riemann_param.rhoL = input.value("rhoL", 1.0);
  Riemann_param.pL   = input.value("pL", 0.4);
  Riemann_param.uL   = input.value("uL", -2.0);

  Riemann_param.rhoR = input.value("rhoR", 1.0);
  Riemann_param.pR   = input.value("pR", 0.4);
  Riemann_param.uR   = input.value("uR", 2.0);

  /*--- Allow for parsing from command line ---*/
  app.add_option("--xd", Riemann_param.xd, "Initial discontinuity location")->capture_default_str()->group("Initial conditions");
  app.add_option("--rhoL", Riemann_param.rhoL, "Initial density at left")->capture_default_str()->group("Initial conditions");
  app.add_option("--pL", Riemann_param.pL, "Initial pressure at left")->capture_default_str()->group("Initial conditions");
  app.add_option("--uL", Riemann_param.uL, "Initial velocity at left")->capture_default_str()->group("Initial conditions");
  app.add_option("--rhoR", Riemann_param.rhoR, "Initial density at right")->capture_default_str()->group("Initial conditions");
  app.add_option("--pR", Riemann_param.pR, "Initial pressure at right")->capture_default_str()->group("Initial conditions");
  app.add_option("--uR", Riemann_param.uR, "Initial velocity at right")->capture_default_str()->group("Initial conditions");

  /*--- Create the instance of the class to perform the simulation ---*/
  CLI11_PARSE(app, argc, argv);
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> min_corner = {sim_param.xL};
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> max_corner = {sim_param.xR};
  auto Euler_MR_sim = Euler_MR(min_corner, max_corner,
                               sim_param, eos_param,
                               Riemann_param);

  Euler_MR_sim.run(sim_param.nfiles);

  samurai::finalize();

  return 0;
}
