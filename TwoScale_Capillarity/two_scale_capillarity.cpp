// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#include <CLI/CLI.hpp>

#include "include/two_scale_capillarity.hpp"

// Main function to run the program
//
int main(int argc, char* argv[]) {
  auto& app = samurai::initialize("Finite volume example for the air-blasted liquid column configuration", argc, argv);

  /*--- Set and declare simulation parameters related to mesh, final time and Courant ---*/
  Simulation_Paramaters sim_param;

  sim_param.xL = 0.0;
  sim_param.xR = 4.0;
  sim_param.yL = 0.0;
  sim_param.yR = 2.0;

  sim_param.min_level     = 8;
  sim_param.max_level     = 8;
  sim_param.MR_param      = 1e-1;
  sim_param.MR_regularity = 0;

  sim_param.R          = 0.15;
  sim_param.eps_over_R = 0.2;
  sim_param.sigma      = 1e-2;

  sim_param.Tf      = 2.5;
  sim_param.Courant = 0.4;

  sim_param.nfiles = 10;

  sim_param.eps_residual            = 1e-8;
  sim_param.mod_grad_alpha1_bar_min = 0.0;

  sim_param.apply_relaxation  = true;
  sim_param.lambda            = 0.9;
  sim_param.tol_Newton        = 1e-12;
  sim_param.max_Newton_iters  = 60;
  sim_param.tol_Newton_p_star = 1e-8;

  sim_param.mass_transfer  = false;
  sim_param.kappa          = 1.0;
  sim_param.Hmax           = 40.0;
  sim_param.alpha1d_max    = 0.5;
  sim_param.alpha1_bar_min = 0.01;
  sim_param.alpha1_bar_max = 0.1;

  app.add_option("--cfl", sim_param.Courant, "The Courant number")->capture_default_str()->group("Simulation parameters");
  app.add_option("--Tf", sim_param.Tf, "Final time")->capture_default_str()->group("Simulation parameters");
  app.add_option("--xL", sim_param.xL, "x Left-end of the domain")->capture_default_str()->group("Simulation parameters");
  app.add_option("--xR", sim_param.xR, "x Right-end of the domain")->capture_default_str()->group("Simulation parameters");
  app.add_option("--yL", sim_param.yL, "y Bottom-end of the domain")->capture_default_str()->group("Simulation parameters");
  app.add_option("--yR", sim_param.yR, "y Top-end of the domain")->capture_default_str()->group("Simulation parameters");
  app.add_option("--min-level", sim_param.min_level, "Minimum level of the AMR")->capture_default_str()->group("AMR parameter");
  app.add_option("--max-level", sim_param.max_level, "Maximum level of the AMR")->capture_default_str()->group("AMR parameter");
  app.add_option("--MR_param", sim_param.MR_param, "Multiresolution parameter")->capture_default_str()->group("AMR parameter");
  app.add_option("--MR_regularity", sim_param.MR_regularity, "Multiresolution regularity")->capture_default_str()->group("AMR parameter");
  app.add_option("--nfiles", sim_param.nfiles, "Number of output files")->capture_default_str()->group("Ouput");
  app.add_option("--R", sim_param.R, "Initial radius of the liquid column")->capture_default_str()->group("Simulation parameters");
  app.add_option("--eps_over_R", sim_param.eps_over_R,
                 "Initial interface thickness with respect to the radius")->capture_default_str()->group("Simulation parameters");
  app.add_option("--sigma", sim_param.sigma, "Surface tension coefficient")->capture_default_str()->group("Simulation parameters");
  app.add_option("--apply_relaxation", sim_param.apply_relaxation, "Apply or not relaxation")->capture_default_str()->group("Simulation_Paramaters");
  app.add_option("--eps_residual", sim_param.eps_residual, "Residual large scale volume fraction")->capture_default_str()->group("Simulation_Paramaters");
  app.add_option("--mod_grad_alpha1_bar_min", sim_param.mod_grad_alpha1_bar_min,
                 "Tolerance for zero gradient volume fraction")->capture_default_str()->group("Simulation_Paramaters");
  app.add_option("--lambda", sim_param.lambda,
                 "Parameter for bound preserving strategy")->capture_default_str()->group("Numerical parameters");
  app.add_option("--tol_Newton", sim_param.tol_Newton,
                 "Tolerance of Newton method for the relaxation")->capture_default_str()->group("Numerical parameters");
  app.add_option("--max_Newton_iters", sim_param.max_Newton_iters,
                 "Maximum number of Newton iterations")->capture_default_str()->group("Numerical parameters");
  app.add_option("--tol_Newton_p_star", sim_param.tol_Newton_p_star,
                 "Tolerance of Newton method to compute p* for the exact solver")->capture_default_str()->group("Numerical parameters");
  app.add_option("--mass_transfer", sim_param.mass_transfer,
                 "Choose whether to perform or not the mass transfer")->capture_default_str()->group("Simulation_Paramaters");
  app.add_option("--kappa", sim_param.kappa,
                 "Small-scale disperse phase raidus with rispect to maximum curvature")->capture_default_str()->group("Simulation_Paramaters");
  app.add_option("--Hmax", sim_param.Hmax,
                 "Maximum curvature before activating atomization")->capture_default_str()->group("Simulation_Paramaters");
  app.add_option("--alpha1d_max", sim_param.alpha1d_max,
                 "Maximum admitted small-scale volume fraction")->capture_default_str()->group("Numerical parameters");
  app.add_option("--alpha1_bar_min", sim_param.alpha1_bar_min,
                 "Maximum effective volume fraction for the mixture region")->capture_default_str()->group("Numerical parameters");
  app.add_option("--alpha1_bar_max", sim_param.alpha1_bar_max,
                 "Maximum effective volume fraction for the mixture region")->capture_default_str()->group("Numerical parameters");

  /*--- Set and declare simulation parameters related to EOS ---*/
  EOS_Parameters eos_param;

  eos_param.p0_phase1   = 1e5;
  eos_param.rho0_phase1 = 1e3;
  eos_param.c0_phase1   = 1e1;

  eos_param.p0_phase2   = 1e5;
  eos_param.rho0_phase2 = 1.0;
  eos_param.c0_phase2   = 1e1;

  app.add_option("--p0_phase1", eos_param.p0_phase1, "p0_phase1")->capture_default_str()->group("EOS parameters");
  app.add_option("--rho0_phase1", eos_param.p0_phase1, "rho0_phase1")->capture_default_str()->group("EOS parameters");
  app.add_option("--c0_phase1", eos_param.c0_phase1, "c0_phase1")->capture_default_str()->group("EOS parameters");
  app.add_option("--p0_phase2", eos_param.p0_phase2, "p0_phase2")->capture_default_str()->group("EOS parameters");
  app.add_option("--rho0_phase2", eos_param.p0_phase2, "rho0_phase2")->capture_default_str()->group("EOS parameters");
  app.add_option("--c0_phase2", eos_param.c0_phase2, "c0_phase2")->capture_default_str()->group("EOS parameters");

  /*--- Create the instance of the class to perform the simulation ---*/
  CLI11_PARSE(app, argc, argv);
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> min_corner = {sim_param.xL, sim_param.yL};
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> max_corner = {sim_param.xR, sim_param.yR};
  auto TwoScaleCapillarity_Sim = TwoScaleCapillarity(min_corner, max_corner,
                                                     sim_param, eos_param);

  TwoScaleCapillarity_Sim.run();

  samurai::finalize();

  return 0;
}
