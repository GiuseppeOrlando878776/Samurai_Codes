// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#include <CLI/CLI.hpp>

#include <nlohmann/json.hpp>

#include "include/relaxation_6eqs.hpp"

// Main function to run the program
//
int main(int argc, char* argv[]) {
  using json = nlohmann::json;

  auto& app = samurai::initialize("Solver for 6-equation mixture-energy-consistent two-phase model", argc, argv);

  std::ifstream ifs("input.json"); // Read a JSON file
  json input = json::parse(ifs);

  /*--- Set and declare some simulation parameters ---*/
  Simulation_Parameters<double> sim_param;

  // Physical parameters
  sim_param.xL = input.value("xL", 0.0);
  sim_param.xR = input.value("xR", 1.0);

  sim_param.t0 = input.value("t0", 0.0);
  sim_param.Tf = input.value("Tf", 2.9e-5);

  sim_param.apply_pressure_relax = input.value("apply_pressure_relax", true);

  // Numerical parameters
  sim_param.Courant = input.value("cfl", 0.2);

  sim_param.apply_finite_rate_relax = input.value("apply_finite_rate_relax", false);
  sim_param.tau_p                   = input.value("tau_p", 1e-8);
  sim_param.use_exact_relax         = input.value("use_exact_relax", false);

  // Mesh parameters
  sim_param.min_level = input.value("min-level", 16);
  sim_param.max_level = input.value("max-level", 16);

  // Output parameters

  sim_param.nfiles = input.value("nfiles", 10);

  /*--- Allow for parsing from command line ---*/
  // Physical parameters
  app.add_option("--xL", sim_param.xL, "x Left-end of the domain")->capture_default_str()->group("Physical parameters");
  app.add_option("--xR", sim_param.xR, "x Right-end of the domain")->capture_default_str()->group("Physical parameters");

  app.add_option("--t0", sim_param.t0, "Initial time")->capture_default_str()->group("Physical parameters");
  app.add_option("--Tf", sim_param.Tf, "Final time")->capture_default_str()->group("Physical parameters");

  app.add_option("--apply_pressure_relax", sim_param.apply_pressure_relax,
                 "Set whether to apply or not the relaxation of the pressure")->capture_default_str()->group("Simulation parameters");

  // Numerical parameters
  app.add_option("--cfl", sim_param.Courant, "The Courant number")->capture_default_str()->group("Numerical parameters");

  app.add_option("--apply_finite_rate_relax", sim_param.apply_finite_rate_relax,
                 "Set whether to perform a finite rate mechanical relaxation")->capture_default_str()->group("Numerical parameters");
  app.add_option("--tau_p", sim_param.tau_p, "Finite rate parameter")->capture_default_str()->group("Numerical parameters");
  app.add_option("--use_exact_relax", sim_param.use_exact_relax,
                 "Use pI to obtain exact relaxation in the case of instantaneous relaxation")->capture_default_str()->group("Numerical parameters");

  // Mesh parameters
  app.add_option("--min-level", sim_param.min_level, "Minimum level of the mesh")->capture_default_str()->group("Mesh parameters");
  app.add_option("--max-level", sim_param.max_level, "Maximum level of the mesh")->capture_default_str()->group("Mesh parameters");

  // Output parameters
  app.add_option("--nfiles", sim_param.nfiles, "Number of output files")->capture_default_str()->group("Ouput");

  /*--- Set and declare simulation parameters related to EOS ---*/
  EOS_Parameters<double> eos_param;

  eos_param.gamma_1    = input.value("gamma_1", 2.43);
  eos_param.pi_infty_1 = input.value("pi_infty_1", 5.3e9);
  eos_param.q_infty_1  = input.value("q_infty_1", 0.0);
  eos_param.c_v_1      = input.value("c_v_1", 1.0);

  eos_param.gamma_2    = input.value("gamma_2", 1.62);
  eos_param.pi_infty_2 = input.value("pi_infty_2", 141e9);
  eos_param.q_infty_2  = input.value("q_infty_2", 0.0);
  eos_param.c_v_2      = input.value("c_v_2", 1.0);

  /*--- Allow for parsing from command line ---*/
  app.add_option("--gammma_1", eos_param.gamma_1, "gamma_1")->capture_default_str()->group("EOS parameters");
  app.add_option("--pi_infty_1", eos_param.pi_infty_1, "pi_infty_1")->capture_default_str()->group("EOS parameters");
  app.add_option("--q_infty_1", eos_param.q_infty_1, "q_infty_1")->capture_default_str()->group("EOS parameters");
  app.add_option("--c_v_1", eos_param.c_v_1, "c_v_1")->capture_default_str()->group("EOS parameters");

  app.add_option("--gammma_2", eos_param.gamma_2, "gamma_2")->capture_default_str()->group("EOS parameters");
  app.add_option("--pi_infty_2", eos_param.pi_infty_2, "pi_infty_2")->capture_default_str()->group("EOS parameters");
  app.add_option("--q_infty_2", eos_param.q_infty_2, "q_infty_2")->capture_default_str()->group("EOS parameters");
  app.add_option("--c_v_2", eos_param.c_v_2, "c_v_2")->capture_default_str()->group("EOS parameters");

  /*--- Set and declare simulation parameters related to initial condition ---*/
  Riemann_Parameters<double> Riemann_param;

  Riemann_param.xd = input.value("xd", 0.6);

  Riemann_param.alpha1L = input.value("alpha1L", 0.5954);
  Riemann_param.rho1L   = input.value("rho1L", 1185.0);
  Riemann_param.p1L     = input.value("p1L", 2e11);
  Riemann_param.uL      = input.value("uL", 0.0);
  Riemann_param.rho2L   = input.value("rho2L", 3622.0);
  Riemann_param.p2L     = input.value("p2L", 2e11);

  Riemann_param.alpha1R = input.value("alpha1R", 0.5954);
  Riemann_param.rho1R   = input.value("rho1R", 1185.0);
  Riemann_param.p1R     = input.value("p1R", 1e5);
  Riemann_param.uR      = input.value("uR", 0.0);
  Riemann_param.rho2R   = input.value("rho2R", 3622.0);
  Riemann_param.p2R     = input.value("p2R", 1e5);

  /*--- Allow for parsing from command line ---*/
  app.add_option("--xd", Riemann_param.xd, "Initial discontinuity location")->capture_default_str()->group("Initial conditions");

  app.add_option("--alpha1L", Riemann_param.alpha1L, "Initial volume fraction at left")->capture_default_str()->group("Initial conditions");
  app.add_option("--rho1L", Riemann_param.rho1L, "Initial density phase 1 at left")->capture_default_str()->group("Initial conditions");
  app.add_option("--p1L", Riemann_param.p1L, "Initial pressure phase 1 at left")->capture_default_str()->group("Initial conditions");
  app.add_option("--uL", Riemann_param.uL, "Initial velocity at left")->capture_default_str()->group("Initial conditions");
  app.add_option("--rho2L", Riemann_param.rho2L, "Initial density phase 2 at left")->capture_default_str()->group("Initial conditions");
  app.add_option("--p2L", Riemann_param.p2L, "Initial pressure phase 2 at left")->capture_default_str()->group("Initial conditions");

  app.add_option("--alpha1R", Riemann_param.alpha1R, "Initial volume fraction at right")->capture_default_str()->group("Initial conditions");
  app.add_option("--rho1R", Riemann_param.rho1R, "Initial density phase 1 at right")->capture_default_str()->group("Initial conditions");
  app.add_option("--p1R", Riemann_param.p1R, "Initial pressure phase 1 at right")->capture_default_str()->group("Initial conditions");
  app.add_option("--uR", Riemann_param.uR, "Initial velocity phase 1 at right")->capture_default_str()->group("Initial conditions");
  app.add_option("--rho2R", Riemann_param.rho2R, "Initial density phase 2 at right")->capture_default_str()->group("Initial conditions");
  app.add_option("--p2R", Riemann_param.p2R, "Initial pressure phase 2 at right")->capture_default_str()->group("Initial conditions");

  /*--- Create the instance of the class to perform the simulation ---*/
  CLI11_PARSE(app, argc, argv);
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> min_corner = {sim_param.xL};
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> max_corner = {sim_param.xR};
  auto Relaxation_Sim = Relaxation(min_corner, max_corner,
                                   sim_param, eos_param,
                                   Riemann_param);

  Relaxation_Sim.run(sim_param.nfiles);

  samurai::finalize();

  return 0;
}
