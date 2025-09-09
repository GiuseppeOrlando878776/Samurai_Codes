// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#include <CLI/CLI.hpp>

#include <nlohmann/json.hpp>

#include "include/dam_break.hpp"

// Main function to run the program
//
int main(int argc, char* argv[]) {
  using json = nlohmann::json;

  auto& app = samurai::initialize("Finite volume example for the dam-break test case", argc, argv);

  #ifdef GRAVITY_IMPLICIT
    PetscInitialize(&argc, &argv, 0, nullptr);
    PetscOptionsSetValue(NULL, "-options_left", "off"); /*--- If on, Petsc will issue warnings saying that
                                                              the options managed by CLI are unused ---*/
  #endif

  std::ifstream ifs("input.json"); // Read a JSON file
  json input = json::parse(ifs);

  /*--- Set and declare some simulation parameters ---*/
  using Number = DamBreak<EquationData::dim>::Number;
  Simulation_Paramaters<Number> sim_param;

  // Physical parameters
  sim_param.xL = input.value("xL", 0.0);
  sim_param.xR = input.value("xR", 1.0);
  sim_param.yL = input.value("yL", 0.0);
  sim_param.yR = input.value("yR", 1.0);
  sim_param.zL = input.value("yL", 0.0);
  sim_param.zR = input.value("yR", 1.0);

  sim_param.t0 = input.value("t0", 0.0);
  sim_param.Tf = input.value("Tf", 0.6);

  sim_param.apply_relaxation = input.value("apply_relaxation", true);

  sim_param.L0 = input.value("L0", 0.4);
  sim_param.H0 = input.value("H0", 0.8);
  sim_param.W0 = input.value("W0", 0.4);

  // Numerical parameters
  sim_param.Courant = input.value("cfl", 0.45);

  sim_param.lambda           = input.value("lambda", 0.9);
  sim_param.atol_Newton      = input.value("atol_Newton", 1e-12);
  sim_param.rtol_Newton      = input.value("rtol_Newton", 1e-10);
  sim_param.max_Newton_iters = input.value("max_Newton_iters", 60);

  // MR parameters
  sim_param.min_level     = input.value("min-level", 4);
  sim_param.max_level     = input.value("max-level", 4);
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
  app.add_option("--yL", sim_param.yL, "y Bottom-end of the domain")->capture_default_str()->group("Physical parameters");
  app.add_option("--yR", sim_param.yR, "y Top-end of the domain")->capture_default_str()->group("Physical parameters");
  app.add_option("--zL", sim_param.zL, "z Bottom-end of the domain")->capture_default_str()->group("Physical parameters");
  app.add_option("--zR", sim_param.zR, "z Top-end of the domain")->capture_default_str()->group("Physical parameters");

  app.add_option("--t0", sim_param.t0, "Initial time")->capture_default_str()->group("Physical parameters");
  app.add_option("--Tf", sim_param.Tf, "Final time")->capture_default_str()->group("Physical parameters");

  app.add_option("--L0", sim_param.L0, "Initial length dam")->capture_default_str()->group("Physical parameters");
  app.add_option("--H0", sim_param.H0, "Initial height dam")->capture_default_str()->group("Physical parameters");
  app.add_option("--W0", sim_param.W0, "Initial width dam")->capture_default_str()->group("Physical parameters");

  // Numerical parameters
  app.add_option("--cfl", sim_param.Courant, "The Courant number")->capture_default_str()->group("Numerical parameters");

  app.add_option("--lambda", sim_param.lambda,
                 "Parameter for bound preserving strategy")->capture_default_str()->group("Numerical parameters");
  app.add_option("--atol_Newton", sim_param.atol_Newton,
                 "Absolute tolerance of Newton method for the relaxation")->capture_default_str()->group("Numerical parameters");
  app.add_option("--rtol_Newton", sim_param.rtol_Newton,
                 "Relative tolerance of Newton method for the relaxation")->capture_default_str()->group("Numerical parameters");
  app.add_option("--max_Newton_iters", sim_param.max_Newton_iters,
                 "Maximum number of Newton iterations")->capture_default_str()->group("Numerical parameters");

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

  eos_param.p0_phase1   = input.value("p0_phase1", 1e5);
  eos_param.rho0_phase1 = input.value("rho0_phase1", 1e3);
  eos_param.c0_phase1   = input.value("c0_phase1", 15.0);

  eos_param.p0_phase2   = input.value("p0_phase2", 1e5);
  eos_param.rho0_phase2 = input.value("rho0_phase2", 1.0);
  eos_param.c0_phase2   = input.value("c0_phase1", 3.4);

  app.add_option("--p0_phase1", eos_param.p0_phase1, "p0_phase1")->capture_default_str()->group("EOS parameters");
  app.add_option("--rho0_phase1", eos_param.p0_phase1, "rho0_phase1")->capture_default_str()->group("EOS parameters");
  app.add_option("--c0_phase1", eos_param.c0_phase1, "c0_phase1")->capture_default_str()->group("EOS parameters");
  app.add_option("--p0_phase2", eos_param.p0_phase2, "p0_phase2")->capture_default_str()->group("EOS parameters");
  app.add_option("--rho0_phase2", eos_param.p0_phase2, "rho0_phase2")->capture_default_str()->group("EOS parameters");
  app.add_option("--c0_phase2", eos_param.c0_phase2, "c0_phase2")->capture_default_str()->group("EOS parameters");

  /*--- Create the instance of the class to perform the simulation ---*/
  CLI11_PARSE(app, argc, argv);
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> min_corner = {sim_param.xL, sim_param.yL, sim_param.zL};
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> max_corner = {sim_param.xR, sim_param.yR, sim_param.zR};
  auto DamBreak_Sim = DamBreak(min_corner, max_corner,
                               sim_param, eos_param);

  DamBreak_Sim.run(sim_param.nfiles);

  #ifdef GRAVITY_IMPLICIT
    PetscFinalize();
  #endif

  samurai::finalize();

  return 0;
}
