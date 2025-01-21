// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#include <CLI/CLI.hpp>

#include "include/relaxation_6eqs.hpp"
//#include "include/relaxation_6eqs_conservative_alpha.hpp"
//#include "include/relaxation_6eqs_internal_energy.hpp"

// Main function to run the program
//
int main(int argc, char* argv[]) {
  CLI::App app{"Solver for 6-equation mixture-energy-consistent two-phase model"};

  // Set and declare simulation parameters related to mesh, final time and Courant
  Simulation_Parameters sim_param;

  sim_param.xL = 0.0;
  sim_param.xR = 1.0;

  sim_param.min_level = 14;
  sim_param.max_level = 14;

  sim_param.Tf      = 3.2e-3;
  sim_param.Courant = 0.2;
  sim_param.nfiles  = 10;

  sim_param.apply_pressure_relax    = true;
  sim_param.apply_finite_rate_relax = false;
  sim_param.mu                      = 1e10;

  app.add_option("--cfl", sim_param.Courant, "The Courant number")->capture_default_str()->group("Simulation parameters");
  app.add_option("--Tf", sim_param.Tf, "Final time")->capture_default_str()->group("Simulation parameters");
  app.add_option("--xL", sim_param.xL, "x Left-end of the domain")->capture_default_str()->group("Simulation parameters");
  app.add_option("--xR", sim_param.xR, "x Right-end of the domain")->capture_default_str()->group("Simulation parameters");
  app.add_option("--apply_pressure_relax", sim_param.apply_pressure_relax,
                 "Set whether to apply or not the relaxation of the pressure")->capture_default_str()->group("Simulation parameters");
  app.add_option("--apply_finite_rate_relax", sim_param.apply_finite_rate_relax,
                 "Set whether to perform a finite rate mechanical relaxation")->capture_default_str()->group("Simulation parameters");
  app.add_option("--mu", sim_param.mu, "Finite rate parameter")->capture_default_str()->group("Simulation parameters");
  app.add_option("--min-level", sim_param.min_level, "Minimum level of the AMR")->capture_default_str()->group("AMR parameter");
  app.add_option("--max-level", sim_param.max_level, "Maximum level of the AMR")->capture_default_str()->group("AMR parameter");
  app.add_option("--nfiles", sim_param.nfiles, "Number of output files")->capture_default_str()->group("Ouput");

  // Set and declare simulation parameters related to EOS
  EOS_Parameters eos_param;

  eos_param.gamma_1    = 2.35;
  eos_param.pi_infty_1 = 1e9;
  eos_param.q_infty_1  = -1167e3;
  eos_param.c_v_1      = 1816.0;

  eos_param.gamma_2    = 1.43;
  eos_param.pi_infty_2 = 0.0;
  eos_param.q_infty_2  = 2030e3;
  eos_param.c_v_2      = 1040.0;

  app.add_option("--gammma_1", eos_param.gamma_1, "gamma_1")->capture_default_str()->group("EOS parameters");
  app.add_option("--pi_infty_1", eos_param.pi_infty_1, "pi_infty_1")->capture_default_str()->group("EOS parameters");
  app.add_option("--q_infty_1", eos_param.q_infty_1, "q_infty_1")->capture_default_str()->group("EOS parameters");
  app.add_option("--c_v_1", eos_param.c_v_1, "c_v_1")->capture_default_str()->group("EOS parameters");
  app.add_option("--gammma_2", eos_param.gamma_2, "gamma_2")->capture_default_str()->group("EOS parameters");
  app.add_option("--pi_infty_2", eos_param.pi_infty_2, "pi_infty_2")->capture_default_str()->group("EOS parameters");
  app.add_option("--q_infty_2", eos_param.q_infty_2, "q_infty_2")->capture_default_str()->group("EOS parameters");
  app.add_option("--c_v_2", eos_param.c_v_2, "c_v_2")->capture_default_str()->group("EOS parameters");

  // Set and declare simulation parameters related to initial condition
  Riemann_Parameters Riemann_param;

  Riemann_param.xd = 0.5;

  Riemann_param.alpha1L = 1.0 - 1e-2;
  Riemann_param.rho1L   = 1150.0;
  Riemann_param.p1L     = 1e5;
  Riemann_param.uL      = -2.0;
  Riemann_param.rho2L   = 0.63;
  Riemann_param.p2L     = 1e5;

  Riemann_param.alpha1R = 1.0 - 1e-2;
  Riemann_param.rho1R   = 1150.0;
  Riemann_param.p1R     = 1e5;
  Riemann_param.uR      = 2.0;
  Riemann_param.rho2R   = 0.63;
  Riemann_param.p2R     = 1e5;

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

  // Create the instance of the class to perform the simulation
  CLI11_PARSE(app, argc, argv);
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> min_corner = {sim_param.xL};
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> max_corner = {sim_param.xR};
  auto Relaxation_Sim = Relaxation(min_corner, max_corner,
                                   sim_param, eos_param,
                                   Riemann_param);

  Relaxation_Sim.run();

  return 0;
}
