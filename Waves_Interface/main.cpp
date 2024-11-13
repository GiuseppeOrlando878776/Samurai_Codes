// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#include <CLI/CLI.hpp>

#include "include/waves_interface.hpp"

// Main function to run the program
//
int main(int argc, char* argv[]) {
  CLI::App app{"Finite volume example for the waves-interface interaction"};

  // Set and declare simulation parameters related to mesh, final time and Courant
  Simulation_Paramaters sim_param;

  sim_param.xL = 0.0;
  sim_param.xR = 1.0;

  sim_param.min_level = 10;
  sim_param.max_level = 10;
  sim_param.MR_param = 1e-5;
  sim_param.MR_regularity = 0;

  sim_param.Tf = 6e-4;
  sim_param.Courant = 0.4;

  sim_param.nfiles = 10;

  sim_param.apply_relaxation = true;
  sim_param.eps_interface_over_dx = 3.0;

  app.add_option("--cfl", sim_param.Courant, "The Courant number")->capture_default_str()->group("Simulation parameters");
  app.add_option("--Tf", sim_param.Tf, "Final time")->capture_default_str()->group("Simulation parameters");
  app.add_option("--xL", sim_param.xL, "x Left-end of the domain")->capture_default_str()->group("Simulation parameters");
  app.add_option("--xR", sim_param.xR, "x Right-end of the domain")->capture_default_str()->group("Simulation parameters");
  app.add_option("--apply_relaxation", sim_param.apply_relaxation, "Apply or not relaxation")->capture_default_str()->group("Simulation_Paramaters");
  app.add_option("--eps_interface_over_dx", sim_param.eps_interface_over_dx,
                 "Interface thickness with respcet to Delta x")->capture_default_str()->group("Simulation_Paramaters");
  app.add_option("--min-level", sim_param.min_level, "Minimum level of the AMR")->capture_default_str()->group("AMR parameter");
  app.add_option("--max-level", sim_param.max_level, "Maximum level of the AMR")->capture_default_str()->group("AMR parameter");
  app.add_option("--MR_param", sim_param.MR_param, "Multiresolution parameter")->capture_default_str()->group("AMR parameter");
  app.add_option("--MR_regularity", sim_param.MR_regularity, "Multiresolution regularity")->capture_default_str()->group("AMR parameter");
  app.add_option("--nfiles", sim_param.nfiles, "Number of output files")->capture_default_str()->group("Ouput");

  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> min_corner = {sim_param.xL};
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> max_corner = {sim_param.xR};

  // Set and declare simulation parameters related to EOS
  EOS_Parameters eos_param;

  eos_param.p0_phase1   = 1e5;
  eos_param.rho0_phase1 = 1e3;
  eos_param.c0_phase1   = 1631.617602258568427706;

  eos_param.p0_phase2   = 1e5;
  eos_param.rho0_phase2 = 1.0;
  eos_param.c0_phase2   = 340.0;

  app.add_option("--p0_phase1", eos_param.p0_phase1, "p0_phase1")->capture_default_str()->group("EOS parameters");
  app.add_option("--rho0_phase1", eos_param.p0_phase1, "rho0_phase1")->capture_default_str()->group("EOS parameters");
  app.add_option("--c0_phase1", eos_param.c0_phase1, "c0_phase1")->capture_default_str()->group("EOS parameters");
  app.add_option("--p0_phase2", eos_param.p0_phase2, "p0_phase2")->capture_default_str()->group("EOS parameters");
  app.add_option("--rho0_phase2", eos_param.p0_phase2, "rho0_phase2")->capture_default_str()->group("EOS parameters");
  app.add_option("--c0_phase2", eos_param.c0_phase2, "c0_phase2")->capture_default_str()->group("EOS parameters");

  // Create the instance of the class to perform the simulation
  CLI11_PARSE(app, argc, argv);
  auto WaveInterface_Sim = WaveInterface(min_corner, max_corner,
                                         sim_param, eos_param);

  WaveInterface_Sim.run();

  return 0;
}
