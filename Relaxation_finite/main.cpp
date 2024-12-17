// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#include <CLI/CLI.hpp>

#include "include/BN_solver.hpp"

// Main function to run the program
//
int main(int argc, char* argv[]) {
  CLI::App app{"Suliciu-type relaxation scheme for the 1D Baer-Nunziato model"};

  // Set and declare simulation parameters related to mesh, final time and Courant
  Simulation_Parameters sim_param;

  sim_param.xL = 0.0;
  sim_param.xR = 1.0;

  sim_param.min_level     = 0;
  sim_param.max_level     = 0;
  sim_param.MR_param      = 1e-2;
  sim_param.MR_regularity = 0;

  sim_param.Tf      = 0.03;
  sim_param.Courant = 0.45;
  sim_param.dt      = 1e-8;
  sim_param.nfiles  = 10;

  sim_param.apply_relaxation = false;

  sim_param.apply_finite_rate_relaxation = true;
  sim_param.tau_u = 1.0;
  sim_param.tau_p = 1e-5;
  sim_param.tau_T = 1e-3;

  sim_param.relax_pressure    = false;
  sim_param.relax_temperature = false;

  app.add_option("--cfl", sim_param.Courant, "The Courant number")->capture_default_str()->group("Simulation parameters");
  app.add_option("--Tf", sim_param.Tf, "Final time")->capture_default_str()->group("Simulation parameters");
  app.add_option("--dt", sim_param.dt, "The time step")->capture_default_str()->group("Simulation parameters");
  app.add_option("--xL", sim_param.xL, "x Left-end of the domain")->capture_default_str()->group("Simulation parameters");
  app.add_option("--xR", sim_param.xR, "x Right-end of the domain")->capture_default_str()->group("Simulation parameters");
  app.add_option("--yL", sim_param.yL, "y Left-end of the domain")->capture_default_str()->group("Simulation parameters");
  app.add_option("--yR", sim_param.yR, "y Right-end of the domain")->capture_default_str()->group("Simulation parameters");
  app.add_option("--min-level", sim_param.min_level, "Minimum level of the AMR")->capture_default_str()->group("AMR parameter");
  app.add_option("--max-level", sim_param.max_level, "Maximum level of the AMR")->capture_default_str()->group("AMR parameter");
  app.add_option("--MR_param", sim_param.MR_param, "Multiresolution parameter")->capture_default_str()->group("AMR parameter");
  app.add_option("--MR_regularity", sim_param.MR_regularity, "Multiresolution regularity")->capture_default_str()->group("AMR parameter");
  app.add_option("--nfiles", sim_param.nfiles, "Number of output files")->capture_default_str()->group("Ouput");
  app.add_option("--apply_relaxation", sim_param.apply_relaxation,
                 "Choose whether to apply relaxation or not")->capture_default_str()->group("Simulation parameters");
  app.add_option("--apply_finite_rate_relaxation", sim_param.apply_finite_rate_relaxation,
                 "If relaxation occurs, finite rate or not")->capture_default_str()->group("Simulation parameters");
  app.add_option("--tau_u", sim_param.tau_u, "Finite rate parameter for the velocity")->capture_default_str()->group("Simulation parameters");
  app.add_option("--tau_p", sim_param.tau_p, "Finite rate parameter for the pressure")->capture_default_str()->group("Simulation parameters");
  app.add_option("--tau_T", sim_param.tau_T, "Finite rate parameter for the temperature")->capture_default_str()->group("Simulation parameters");
  app.add_option("--relax_pressure", sim_param.relax_pressure,
                 "If instantaneous relaxation, relax the pressure")->capture_default_str()->group("Simulation parameters");
  app.add_option("--relax_temperature", sim_param.relax_temperature,
                 "If instantaneous relaxation, relax the temperature (this can occur only with pressure)")->capture_default_str()->group("Simulation parameters");

  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> min_corner = {sim_param.xL};
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> max_corner = {sim_param.xR};

  // Set and declare simulation parameters related to EOS
  EOS_Parameters eos_param;

  eos_param.gamma_1    = 1.61492481180737;
  eos_param.pi_infty_1 = 3.563521398523755e8;
  eos_param.q_infty_1  = 0.0;
  eos_param.cv_1       = 1.452904592629688e3;

  eos_param.gamma_2    = 1.085507894797296;
  eos_param.pi_infty_2 = 0.0;
  eos_param.q_infty_2  = 0.0;
  eos_param.cv_2       = 4.441148752333071e3;

  app.add_option("--gammma_1", eos_param.gamma_1, "gamma_1")->capture_default_str()->group("EOS parameters");
  app.add_option("--pi_infty_1", eos_param.pi_infty_1, "pi_infty_1")->capture_default_str()->group("EOS parameters");
  app.add_option("--q_infty_1", eos_param.q_infty_1, "q_infty_1")->capture_default_str()->group("EOS parameters");
  app.add_option("--cv_1", eos_param.cv_1, "cv_1")->capture_default_str()->group("EOS parameters");
  app.add_option("--gammma_2", eos_param.gamma_2, "gamma_2")->capture_default_str()->group("EOS parameters");
  app.add_option("--pi_infty_2", eos_param.pi_infty_2, "pi_infty_2")->capture_default_str()->group("EOS parameters");
  app.add_option("--q_infty_2", eos_param.q_infty_2, "q_infty_2")->capture_default_str()->group("EOS parameters");
  app.add_option("--cv_2", eos_param.cv_2, "cv_2")->capture_default_str()->group("EOS parameters");

  // Set and declare simulation parameters related to initial condition
  Riemann_Parameters Riemann_param;

  Riemann_param.xd      = 0.5;

  Riemann_param.alpha1L = 0.3;
  Riemann_param.p1L     = 1e5;
  Riemann_param.T1L     = 363.0;
  Riemann_param.u1L     = 0.0;
  Riemann_param.p2L     = 1e5;
  Riemann_param.T2L     = 1000.0;
  Riemann_param.u2L     = 0.0;

  Riemann_param.alpha1R = 0.3;
  Riemann_param.p1R     = 1e5;
  Riemann_param.T1R     = 363.0;
  Riemann_param.u1R     = 0.0;
  Riemann_param.p2R     = 1e5;
  Riemann_param.T2R     = 1000.0;
  Riemann_param.u2R     = 0.0;

  app.add_option("--xd", Riemann_param.xd, "Initial discontinuity location")->capture_default_str()->group("Initial conditions");
  app.add_option("--alpha1L", Riemann_param.alpha1L, "Initial volume fraction at left")->capture_default_str()->group("Initial conditions");
  app.add_option("--rho1L", Riemann_param.rho1L, "Initial density phase 1 at left")->capture_default_str()->group("Initial conditions");
  app.add_option("--p1L", Riemann_param.p1L, "Initial pressure phase 1 at left")->capture_default_str()->group("Initial conditions");
  app.add_option("--T1L", Riemann_param.T1L, "Initial temperature phase 1 at left")->capture_default_str()->group("Initial conditions");
  app.add_option("--u1L", Riemann_param.u1L, "Initial horizontal velocity phase 1 at left")->capture_default_str()->group("Initial conditions");
  app.add_option("--v1L", Riemann_param.v1L, "Initial vertical velocity phase 1 at left")->capture_default_str()->group("Initial conditions");
  app.add_option("--rho2L", Riemann_param.rho2L, "Initial density phase 2 at left")->capture_default_str()->group("Initial conditions");
  app.add_option("--p2L", Riemann_param.p2L, "Initial pressure phase 2 at left")->capture_default_str()->group("Initial conditions");
  app.add_option("--T2L", Riemann_param.T2L, "Initial temperature phase 2 at left")->capture_default_str()->group("Initial conditions");
  app.add_option("--u2L", Riemann_param.u2L, "Initial horizontal velocity phase 2 at left")->capture_default_str()->group("Initial conditions");
  app.add_option("--v2L", Riemann_param.v2L, "Initial vertical velocity phase 2 at left")->capture_default_str()->group("Initial conditions");
  app.add_option("--alpha1R", Riemann_param.alpha1R, "Initial volume fraction at right")->capture_default_str()->group("Initial conditions");
  app.add_option("--rho1R", Riemann_param.rho1R, "Initial density phase 1 at right")->capture_default_str()->group("Initial conditions");
  app.add_option("--p1R", Riemann_param.p1R, "Initial pressure phase 1 at right")->capture_default_str()->group("Initial conditions");
  app.add_option("--T1R", Riemann_param.T1R, "Initial temperature phase 1 at right")->capture_default_str()->group("Initial conditions");
  app.add_option("--u1R", Riemann_param.u1R, "Initial horizotnal velocity phase 1 at right")->capture_default_str()->group("Initial conditions");
  app.add_option("--v1R", Riemann_param.v1R, "Initial vertical velocity phase 1 at right")->capture_default_str()->group("Initial conditions");
  app.add_option("--rho2R", Riemann_param.rho2R, "Initial density phase 2 at right")->capture_default_str()->group("Initial conditions");
  app.add_option("--p2R", Riemann_param.p2R, "Initial pressure phase 2 at right")->capture_default_str()->group("Initial conditions");
  app.add_option("--T2R", Riemann_param.T2R, "Initial temperature phase 2 at right")->capture_default_str()->group("Initial conditions");
  app.add_option("--u2R", Riemann_param.u2R, "Initial horizontal velocity phase 2 at right")->capture_default_str()->group("Initial conditions");
  app.add_option("--v2R", Riemann_param.v2R, "Initial vertical velocity phase 2 at right")->capture_default_str()->group("Initial conditions");

  // Create the instance of the class to perform the simulation
  CLI11_PARSE(app, argc, argv);
  auto BN_Solver_Sim = BN_Solver(min_corner, max_corner, sim_param, eos_param, Riemann_param);

  BN_Solver_Sim.run();

  return 0;
}
