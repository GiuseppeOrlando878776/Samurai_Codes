// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#include <CLI/CLI.hpp>

#include <nlohmann/json.hpp>

#include "include/waves_interface.hpp"

// Main function to run the program
//
int main(int argc, char* argv[]) {
  using json = nlohmann::json;

  auto& app = samurai::initialize("Finite volume example for the waves-interface interaction", argc, argv);

  std::ifstream ifs("input.json"); // Read a JSON file
  json input = json::parse(ifs);

  /*--- Set and declare simulation parameters related to mesh, final time and Courant ---*/
  Simulation_Parameters sim_param;

  sim_param.xL = input.value("xL", 0.0);
  sim_param.xR = input.value("xR", 1.0);

  sim_param.sigma                 = input.value("sigma", 0.0);
  sim_param.eps_interface_over_dx = input.value("eps_interface_over_dx", 3.0);

  sim_param.min_level     = input.value("min-level", 10);
  sim_param.max_level     = input.value("max-level", 10);
  sim_param.MR_param      = input.value("MR_param", 1e-3);
  sim_param.MR_regularity = input.value("MR_regularity", 0);

  sim_param.Tf      = input.value("Tf", 6e-4);
  sim_param.Courant = input.value("cfl", 0.4);

  sim_param.nfiles = input.value("nfiles", 10);

  sim_param.mod_grad_alpha1_min = input.value("mod_grad_alpha1_min", 0.0);

  sim_param.apply_relaxation = input.value("apply_relaxation", true);
  sim_param.lambda           = input.value("lambda", 0.9);
  sim_param.atol_Newton      = input.value("atol_Newton", 1e-12);
  sim_param.rtol_Newton      = input.value("rtol_Newton", 1e-10);
  sim_param.max_Newton_iters = input.value("max_Newton_iters", 60);

  sim_param.atol_Newton_p_star = input.value("atol_Newton_p_star", 1e-10);
  sim_param.rtol_Newton_p_star = input.value("rtol_Newton_p_star", 1e-8);

  app.add_option("--xL", sim_param.xL, "x Left-end of the domain")->capture_default_str()->group("Simulation parameters");
  app.add_option("--xR", sim_param.xR, "x Right-end of the domain")->capture_default_str()->group("Simulation parameters");

  app.add_option("--sigma", sim_param.sigma, "Surface tension coefficient")->capture_default_str()->group("Simulation parameters");

  app.add_option("--min-level", sim_param.min_level, "Minimum level of the AMR")->capture_default_str()->group("AMR parameter");
  app.add_option("--max-level", sim_param.max_level, "Maximum level of the AMR")->capture_default_str()->group("AMR parameter");
  app.add_option("--MR_param", sim_param.MR_param, "MR parameter for adaptation")->capture_default_str()->group("AMR parameter");
  app.add_option("--MR_regularity", sim_param.MR_regularity, "MR parameter for mesh regularity")->capture_default_str()->group("AMR parameter");
  app.add_option("--nfiles", sim_param.nfiles, "Number of output files")->capture_default_str()->group("Ouput");

  app.add_option("--cfl", sim_param.Courant, "The Courant number")->capture_default_str()->group("Numerical parameters");
  app.add_option("--Tf", sim_param.Tf, "Final time")->capture_default_str()->group("Numerical parameters");
  app.add_option("--apply_relaxation", sim_param.apply_relaxation, "Apply or not relaxation")->capture_default_str()->group("Numerical paramaters");
  app.add_option("--mod_grad_alpha1_min", sim_param.mod_grad_alpha1_min,
                 "Tolerance for zero gradient volume fraction")->capture_default_str()->group("Numerical paramaters");
  app.add_option("--lambda", sim_param.lambda,
                 "Parameter for bound preserving strategy")->capture_default_str()->group("Numerical parameters");
  app.add_option("--atol_Newton", sim_param.atol_Newton,
                 "Absolute tolerance of Newton method for the relaxation")->capture_default_str()->group("Numerical parameters");
  app.add_option("--rtol_Newton", sim_param.rtol_Newton,
                 "Relative tolerance of Newton method for the relaxation")->capture_default_str()->group("Numerical parameters");
  app.add_option("--max_Newton_iters", sim_param.max_Newton_iters,
                 "Maximum number of Newton iterations")->capture_default_str()->group("Numerical parameters");
  app.add_option("--atol_Newton_p_star", sim_param.atol_Newton_p_star,
                 "Absolute tolerance of Newton method to compute p* for the exact solver")->capture_default_str()->group("Numerical parameters");
  app.add_option("--rtol_Newton_p_star", sim_param.rtol_Newton_p_star,
                 "Relative tolerance of Newton method to compute p* for the exact solver")->capture_default_str()->group("Numerical parameters");

  /*--- Set and declare simulation parameters related to EOS ---*/
  EOS_Parameters eos_param;

  eos_param.p0_phase1   = input.value("p0_phase1", 1e5);
  eos_param.rho0_phase1 = input.value("rho0_phase1", 1e3);
  eos_param.c0_phase1   = input.value("c0_phase1", 1627.52160119);

  eos_param.p0_phase2   = input.value("p0_phase2", 1e5);
  eos_param.rho0_phase2 = input.value("rho0_phase2", 1.0);
  eos_param.c0_phase2   = input.value("c0_phase2", 340.0);

  app.add_option("--p0_phase1", eos_param.p0_phase1, "p0_phase1")->capture_default_str()->group("EOS parameters");
  app.add_option("--rho0_phase1", eos_param.p0_phase1, "rho0_phase1")->capture_default_str()->group("EOS parameters");
  app.add_option("--c0_phase1", eos_param.c0_phase1, "c0_phase1")->capture_default_str()->group("EOS parameters");
  app.add_option("--p0_phase2", eos_param.p0_phase2, "p0_phase2")->capture_default_str()->group("EOS parameters");
  app.add_option("--rho0_phase2", eos_param.p0_phase2, "rho0_phase2")->capture_default_str()->group("EOS parameters");
  app.add_option("--c0_phase2", eos_param.c0_phase2, "c0_phase2")->capture_default_str()->group("EOS parameters");

  /*--- Create the instance of the class to perform the simulation ---*/
  CLI11_PARSE(app, argc, argv);
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> min_corner = {sim_param.xL};
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> max_corner = {sim_param.xR};
  auto WaveInterface_Sim = WaveInterface(min_corner, max_corner,
                                         sim_param, eos_param);

  WaveInterface_Sim.run();

  samurai::finalize();

  return 0;
}
