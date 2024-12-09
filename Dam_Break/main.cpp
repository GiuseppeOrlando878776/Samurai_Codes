// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#include <CLI/CLI.hpp>

#include "include/dam_break.hpp"

// Main function to run the program
//
int main(int argc, char* argv[]) {
  CLI::App app{"Finite volume example for the dam-break test case"};

  #ifdef GRAVITY_IMPLICIT
    PetscInitialize(&argc, &argv, 0, nullptr);
    PetscOptionsSetValue(NULL, "-options_left", "off"); // If on, Petsc will issue warnings saying that
                                                        // the options managed by CLI are unused
  #endif

  // Set and declare simulation parameters related to mesh, final time and Courant
  Simulation_Paramaters sim_param;

  sim_param.xL = 0.0;
  sim_param.xR = 1.6;
  sim_param.yL = 0.0;
  sim_param.yR = 0.8;

  sim_param.L0 = 0.6;
  sim_param.H0 = 0.3;

  sim_param.min_level = 7;
  sim_param.max_level = 7;
  sim_param.MR_param = 1e-3;
  sim_param.MR_regularity = 0;

  sim_param.Tf = 0.6;
  sim_param.Courant = 0.4;

  sim_param.nfiles = 10;

  sim_param.apply_relaxation = true;

  app.add_option("--cfl", sim_param.Courant, "The Courant number")->capture_default_str()->group("Simulation parameters");
  app.add_option("--Tf", sim_param.Tf, "Final time")->capture_default_str()->group("Simulation parameters");
  app.add_option("--xL", sim_param.xL, "x Left-end of the domain")->capture_default_str()->group("Simulation parameters");
  app.add_option("--xR", sim_param.xR, "x Right-end of the domain")->capture_default_str()->group("Simulation parameters");
  app.add_option("--yL", sim_param.yL, "y Left-end of the domain")->capture_default_str()->group("Simulation parameters");
  app.add_option("--yR", sim_param.yR, "y Right-end of the domain")->capture_default_str()->group("Simulation parameters");
  app.add_option("--L0", sim_param.L0, "Dam initial length")->capture_default_str()->group("Simulation parameters");
  app.add_option("--H0", sim_param.H0, "Dam initial height")->capture_default_str()->group("Simulation parameters");
  app.add_option("--apply_relaxation", sim_param.apply_relaxation, "Apply or not relaxation")->capture_default_str()->group("Simulation_Paramaters");
  app.add_option("--min-level", sim_param.min_level, "Minimum level of the AMR")->capture_default_str()->group("AMR parameter");
  app.add_option("--max-level", sim_param.max_level, "Maximum level of the AMR")->capture_default_str()->group("AMR parameter");
  app.add_option("--MR_param", sim_param.MR_param, "Multiresolution parameter")->capture_default_str()->group("AMR parameter");
  app.add_option("--MR_regularity", sim_param.MR_regularity, "Multiresolution regularity")->capture_default_str()->group("AMR parameter");
  app.add_option("--nfiles", sim_param.nfiles, "Number of output files")->capture_default_str()->group("Ouput");

  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> min_corner = {sim_param.xL, sim_param.yL};
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> max_corner = {sim_param.xR, sim_param.yR};

  // Set and declare simulation parameters related to EOS
  EOS_Parameters eos_param;

  eos_param.p0_phase1   = 1e5;
  eos_param.rho0_phase1 = 1e3;
  eos_param.c0_phase1   = 15.0;

  eos_param.p0_phase2   = 1e5;
  eos_param.rho0_phase2 = 1.0;
  eos_param.c0_phase2   = 3.4;

  app.add_option("--p0_phase1", eos_param.p0_phase1, "p0_phase1")->capture_default_str()->group("EOS parameters");
  app.add_option("--rho0_phase1", eos_param.p0_phase1, "rho0_phase1")->capture_default_str()->group("EOS parameters");
  app.add_option("--c0_phase1", eos_param.c0_phase1, "c0_phase1")->capture_default_str()->group("EOS parameters");
  app.add_option("--p0_phase2", eos_param.p0_phase2, "p0_phase2")->capture_default_str()->group("EOS parameters");
  app.add_option("--rho0_phase2", eos_param.p0_phase2, "rho0_phase2")->capture_default_str()->group("EOS parameters");
  app.add_option("--c0_phase2", eos_param.c0_phase2, "c0_phase2")->capture_default_str()->group("EOS parameters");

  // Create the instance of the class to perform the simulation
  CLI11_PARSE(app, argc, argv);
  auto DamBreak_Sim = DamBreak(min_corner, max_corner,
                               sim_param, eos_param);

  DamBreak_Sim.run();

  #ifdef GRAVITY_IMPLICIT
    PetscFinalize();
  #endif

  return 0;
}
