// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2026
//
#include <CLI/CLI.hpp>

#include <nlohmann/json.hpp>

#include "include/two_scale_capillarity.hpp"

// Main function to run the program
//
int main(int argc, char* argv[]) {
  using json = nlohmann::json;

  auto& app = samurai::initialize("Finite volume example for air-blasted liquid column without mass transfer", argc, argv);

  json input;
  try {
    std::ifstream ifs("input.json"); // Read a JSON file
    input = json::parse(ifs);
  }
  catch(const json::parse_error& e) {
    throw std::runtime_error("Cannot parse parameter file 'input.json'. Please verify that the file is present");
  }

  /*--- Set and declare simulation parameters ---*/
  using Number = TwoScaleCapillarity<EquationData::dim>::Number;
  Simulation_Parameters<Number> sim_param;

  // Physical parameters
  sim_param.xL = input.value("xL", static_cast<double>(0.0));
  sim_param.xR = input.value("xR", static_cast<double>(4.0));
  sim_param.yL = input.value("yL", static_cast<double>(0.0));
  sim_param.yR = input.value("yR", static_cast<double>(2.0));

  sim_param.t0 = input.value("t0", static_cast<Number>(0.0));
  sim_param.Tf = input.value("Tf", static_cast<Number>(2.5));

  sim_param.sigma = input.value("sigma", static_cast<Number>(1e-2));

  sim_param.apply_relaxation = input.value("apply_relaxation", true);

  // Numerical parameters
  sim_param.num_flux_hyp = input.value("num_flux_hyp", "HLLC");

  sim_param.Courant = input.value("cfl", 0.4);

  sim_param.alpha_residual      = input.value("alpha_residual", static_cast<Number>(1e-8));
  sim_param.mod_grad_alpha1_min = input.value("mod_grad_alpha1_min", static_cast<Number>(0.0));

  sim_param.lambda           = input.value("lambda", static_cast<Number>(0.9));
  sim_param.atol_Newton      = input.value("atol_Newton", static_cast<Number>(1e-14));
  sim_param.rtol_Newton      = input.value("rtol_Newton", static_cast<Number>(1e-12));
  sim_param.max_Newton_iters = input.value("max_Newton_iters", static_cast<std::size_t>(60));

  sim_param.atol_Newton_p_star = input.value("atol_Newton_p_star", static_cast<Number>(1e-10));
  sim_param.rtol_Newton_p_star = input.value("rtol_Newton_p_star", static_cast<Number>(1e-8));

  // MR paramters
  sim_param.min_level     = input.value("min-level", static_cast<std::size_t>(8));
  sim_param.max_level     = input.value("max-level", static_cast<std::size_t>(8));
  sim_param.MR_param      = input.value("MR_param", static_cast<double>(1e-1));
  sim_param.MR_regularity = input.value("MR_regularity", static_cast<double>(0));

  // Output parameters
  sim_param.save_dir = input.value("save-dir", fs::current_path());
  sim_param.nfiles   = input.value("nfiles", static_cast<std::size_t>(10));

  // Restart file
  sim_param.restart_file = input.value("restart_file","");

  /*--- Allow for parsing from command line ---*/
  // Physical parameters
  app.add_option("--xL", sim_param.xL, "x Left-end of the domain")->capture_default_str()->group("Physical parameters");
  app.add_option("--xR", sim_param.xR, "x Right-end of the domain")->capture_default_str()->group("Physical parameters");
  app.add_option("--yL", sim_param.yL, "y Bottom-end of the domain")->capture_default_str()->group("Physical parameters");
  app.add_option("--yR", sim_param.yR, "y Top-end of the domain")->capture_default_str()->group("Physical parameters");

  app.add_option("--t0", sim_param.t0, "Initial time")->capture_default_str()->group("Physical parameters");
  app.add_option("--Tf", sim_param.Tf, "Final time")->capture_default_str()->group("Physical parameters");

  app.add_option("--sigma", sim_param.sigma, "Surface tension coefficient")->capture_default_str()->group("Physical parameters");

  app.add_option("--apply_relaxation", sim_param.apply_relaxation, "Apply or not relaxation")->capture_default_str()->group("Physical parameters");

  // Numerical parameters
  app.add_option("--num_flux_hyp", sim_param.num_flux_hyp,
                 "The numerical flux for hyperbolic subsystem")->capture_default_str()->group("Numerical parameters");

  app.add_option("--cfl", sim_param.Courant, "The Courant number")->capture_default_str()->group("Numerical parameters");

  app.add_option("--alpha_residual", sim_param.alpha_residual, "Residual large scale volume fraction")->capture_default_str()->group("Numerical parameters");
  app.add_option("--mod_grad_alpha1_min", sim_param.mod_grad_alpha1_min,
                 "Tolerance for zero gradient volume fraction")->capture_default_str()->group("Numerical parameters");

  app.add_option("--lambda", sim_param.lambda,
                 "Parameter for bound-preserving strategy")->capture_default_str()->group("Numerical parameters");
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

  // MR parameters
  app.add_option("--min-level", sim_param.min_level, "Minimum level of the AMR")->capture_default_str()->group("AMR parameter");
  app.add_option("--max-level", sim_param.max_level, "Maximum level of the AMR")->capture_default_str()->group("AMR parameter");
  app.add_option("--MR_param", sim_param.MR_param, "Multiresolution parameter")->capture_default_str()->group("AMR parameter");
  app.add_option("--MR_regularity", sim_param.MR_regularity, "Multiresolution regularity")->capture_default_str()->group("AMR parameter");

  // Output parameters
  app.add_option("--save-dir", sim_param.save_dir, "Output directory")->capture_default_str()->group("Output parameters");
  app.add_option("--nfiles", sim_param.nfiles, "Number of output files")->capture_default_str()->group("Output parameters");

  // Restart file
  app.add_option("--restart_file", sim_param.restart_file, "Name of the restart file")->capture_default_str()->group("Restart");

  /*--- Set and declare simulation parameters related to EOS ---*/
  EOS_Parameters<Number> eos_param;

  eos_param.p0_phase1   = input.value("p0_phase1", static_cast<Number>(1e5));
  eos_param.rho0_phase1 = input.value("rho0_phase1", static_cast<Number>(1e3));
  eos_param.c0_phase1   = input.value("c0_phase1", static_cast<Number>(1e1));

  eos_param.p0_phase2   = input.value("p0_phase2", static_cast<Number>(1e5));
  eos_param.rho0_phase2 = input.value("rho0_phase2", static_cast<Number>(1.0));
  eos_param.c0_phase2   = input.value("c0_phase2", static_cast<Number>(1e1));

  app.add_option("--p0_phase1", eos_param.p0_phase1, "p0_phase1")->capture_default_str()->group("EOS parameters");
  app.add_option("--rho0_phase1", eos_param.p0_phase1, "rho0_phase1")->capture_default_str()->group("EOS parameters");
  app.add_option("--c0_phase1", eos_param.c0_phase1, "c0_phase1")->capture_default_str()->group("EOS parameters");
  app.add_option("--p0_phase2", eos_param.p0_phase2, "p0_phase2")->capture_default_str()->group("EOS parameters");
  app.add_option("--rho0_phase2", eos_param.p0_phase2, "rho0_phase2")->capture_default_str()->group("EOS parameters");
  app.add_option("--c0_phase2", eos_param.c0_phase2, "c0_phase2")->capture_default_str()->group("EOS parameters");

  /*--- Create the instance of the class to perform the simulation ---*/
  // Read name and parameter file of the test case
  std::string tc_name       = input.value("test_case", "liquid_column");
  std::string tc_param_file = input.value("test_case_param", "liquid_column.json");

  app.add_option("--test_case", tc_name, "Test case configuration")->capture_default_str()->group("Physical parameters");
  app.add_option("--test_case_param", tc_param_file, "Test case parameter file")->capture_default_str()->group("Physical parameters");

  // Create the test case through factory
  auto tc = make_test_case<TwoScaleCapillarity_Traits<dim>,
                           AuxiliaryFields<TwoScaleCapillarity_Traits<dim>>>(tc_name, tc_param_file);

  CLI11_PARSE(app, argc, argv);
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> min_corner = {sim_param.xL, sim_param.yL};
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> max_corner = {sim_param.xR, sim_param.yR};
  auto TwoScaleCapillarity_Sim = TwoScaleCapillarity(min_corner, max_corner,
                                                     sim_param, eos_param,
                                                     std::move(tc));

  TwoScaleCapillarity_Sim.run(sim_param.num_flux_hyp, sim_param.nfiles);

  samurai::finalize();

  return 0;
}
