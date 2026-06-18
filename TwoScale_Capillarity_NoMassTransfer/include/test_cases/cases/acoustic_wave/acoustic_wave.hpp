// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2026
//
#pragma once

#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>

#include <nlohmann/json.hpp>

#include "test_case.hpp"

using namespace EquationData;

/**
 * @file acoustic_wave.hpp
 *
 * @brief Test case: acoustic wave propagation
 *
 * Physical configuration
 * ----------------------
 * We consider the acoustic wave propagation
 *
 * Parameter file (json)
 * ---------------------
 * All physical parameters are read from a dedicated input file whose path
 * is provided at construction time. A minimal file looks like:
 *
 * @code{.json}
 * "x_shock": 0.3 # shock location
 * "x_interface": 0.7 # interface location
 * "eps_interface_over_dx": 0.5 # interface thickness w.r.t. mesh size
 * "eps_shock_over_dx": 3.0 # 'regularized' shock thickness
 * @endcode
 *
 * Scalar parameters from SolverContext
 * -------------------------------------
 * The following keys must be present in ctx.params when setup() is called:
 *   "mod_grad_alpha1_min" - threshold for which computing the unit normal vector
 *
 * Contract with the solver
 * ------------------------
 * setup() uses the auxiliary fields and
 * ctx.params["mod_grad_alpha1_min"]
 * from the scalar map. All other context fields are ignored.
 *
 * @tparam Traits Traits struct defined in the solver header.
 * @tparam AuxFields Auxiliary fields struct defined in the solver header.
 */
template<typename Traits, typename AuxFields>
class AcousticWave final : public TestCaseBase<Traits, AuxFields> {
public:
  using Context = typename AcousticWave<Traits, AuxFields>::Context;

  using Number = typename Traits::Number;
  using Field  = typename Traits::Field;

  /**
   * Acoustic wave propagation constructor
   * @param_file name of the parameter files
   */
  explicit AcousticWave(const std::string& param_file);

  /**
    * Retrieve scalar parameters from ctx.params, then build init_fn
    * capturing the needed context references and private members.
    * ctx contains both conserved and auxiliary fields
    *
    * Required ctx.params keys: "mod_grad_alpha1_min".
    * @param ctx struct with all conserved and auxiliary fields
   */
  void setup(Context& ctx) override;

private:
  /**
   * Initialize all conserved and auxiliary fields.
   * The implementation is free to call samurai::for_each_cell, etc.
   * The mesh has already been constructed by the solver before this call.
   * @param ctx struct with all conserved and auxiliary fields
   * @param mod_grad_alpha1_min threshold of gradient of large-scale volume fraction to compute the normal
   */
  void init_variables(Context& ctx,
                      const Number mod_grad_alpha1_min);

  /**
   * Attach boundary conditions to the conserved variable field.
   *
   * @param conserved_variables field with conserved state
   */
  void apply_bcs(Field& conserved_variables);

  /**
   * Compute regularized Heavised
   *
   * @param x x-coordinate
   * @param eps regularization parameter
   */
  template<typename T = double>
  T CHeaviside(const T x, const T eps);

  Number x_shock;               /*!< Location of the shock */
  Number x_interface;           /*!< Location of the interface */
  Number eps_interface_over_dx; /*!< Interface thickness w.r.t mesh size */
  Number eps_shock_over_dx;     /*!< 'Regularized shock' thickness w.r.t mesh size */
};

// Acoustic wave propagation constructor
//
template<typename Traits, typename AuxFields>
AcousticWave<Traits, AuxFields>::AcousticWave(const std::string& param_file) {
  using json = nlohmann::json;

  try {
    std::ifstream ifs(param_file);
    json input_tc = json::parse(ifs);

    // Read with safe defaults so missing keys are not fatal
    x_shock               = input_tc.value("x_shock", static_cast<Number>(0.3));
    x_interface           = input_tc.value("x_interface", static_cast<Number>(0.7));
    eps_interface_over_dx = input_tc.value("eps_interface_over_dx", static_cast<Number>(0.5));
    eps_shock_over_dx     = input_tc.value("eps_shock_over_dx", static_cast<Number>(3.0));
  }
  catch(const json::parse_error& e) {
    // Default values in case file does not exist
    std::cerr << "AcousticWave: cannot parse parameter file '" +
                  param_file + "': " + "using default values" << std::endl;

    x_shock               = static_cast<Number>(0.3);
    x_interface           = static_cast<Number>(0.7);
    eps_interface_over_dx = static_cast<Number>(0.5);
    eps_shock_over_dx     = static_cast<Number>(3.0);
  }
}

// setup(): assemble init_fn and bc_fnfrom the solver context
//
template<typename Traits, typename AuxFields>
void AcousticWave<Traits, AuxFields>::setup(Context& ctx) {
  // Retrieve scalar parameters — throws std::out_of_range if absent,
  // with a message that names this class and the missing key.
  const Number mod_grad_alpha1_min = ctx.param("mod_grad_alpha1_min", "AcousticWave");

  // Capture ctx by reference: the solver guarantees it outlives init_fn and bc_fn.
  // Capture 'this' by pointer: AcousticWave outlives the solver.
  // Capture mod_grad_alpha1_min by values: they are lightweight scalars.
  this->init_fn = [this, &ctx, mod_grad_alpha1_min]() {
    init_variables(ctx, mod_grad_alpha1_min);
  };

  this->bc_fn = [this, &ctx]()
                {
                  apply_bcs(ctx.conserved_variables);
                };
}

// Initialize conserved and auxiliary variables
//
template<typename Traits, typename AuxFields>
void AcousticWave<Traits, AuxFields>::init_variables(Context& ctx,
                                                     const Number mod_grad_alpha1_min) {
  // Derived useful constants
  const auto dx            = ctx.mesh.cell_length(ctx.mesh.max_level());
  const auto eps_interface = eps_interface_over_dx*dx;
  const auto eps_shock     = eps_shock_over_dx*dx;

  // Initialize the volume fraction with a loop over all cells
  samurai::for_each_cell(ctx.mesh,
                         [&](const auto& cell)
                            {
                              // Set volume fraction
                              const auto center = cell.center();
                              const auto x      = static_cast<Number>(center[0]);

                              ctx.alpha1[cell] = (static_cast<Number>(1.0) - static_cast<Number>(1e-7))
                                               + (static_cast<Number>(1e-7) - static_cast<Number>(1e-4))*CHeaviside(x_interface - x, eps_interface);

                              // Set mass phase 1
                              ctx.aux.p1[cell] = static_cast<Number>(1e5);
                              if(x >= static_cast<Number>(0.45) && x <= static_cast<Number>(0.55)) {
                                ctx.aux.p1[cell] += std::sin(static_cast<Number>(2.0)*
                                                             static_cast<Number>(4.0*std::atan(1))*
                                                             (x - static_cast<Number>(0.5))/static_cast<Number>(0.1));
                              }
                              const auto rho1_loc = ctx.EOS_phase1.rho_value(ctx.aux.p1[cell]);
                              ctx.conserved_variables[cell][M1_INDEX] = ctx.alpha1[cell]*rho1_loc;

                              // Set mass phase 2
                              ctx.aux.p2[cell] = ctx.aux.p1[cell];
                              const auto rho2_loc = ctx.EOS_phase2.rho_value(ctx.aux.p2[cell]);
                              ctx.conserved_variables[cell][M2_INDEX] = (static_cast<Number>(1.0) - ctx.alpha1[cell])*rho2_loc;

                              // Save mixture pressure for post-processing
                              ctx.aux.p[cell] = ctx.alpha1[cell]*ctx.aux.p1[cell]
                                              + (static_cast<Number>(1.0) - ctx.alpha1[cell])*ctx.aux.p2[cell];

                              // Set conserved variable associated to volume fraction
                              const auto rho_loc = ctx.conserved_variables[cell][M1_INDEX]
                                                 + ctx.conserved_variables[cell][M2_INDEX];

                              ctx.conserved_variables[cell][RHO_ALPHA1_INDEX] = rho_loc*ctx.alpha1[cell];

                              // Set momentum
                              ctx.conserved_variables[cell][RHO_U_INDEX] = static_cast<Number>(0.0);

                              // Save velocity for post-processing
                              for(std::size_t d = 0; d < Field::dim; ++d) {
                                ctx.aux.vel[cell][d] = ctx.conserved_variables[cell](RHO_U_INDEX + d)/rho_loc;
                              }

                              // Compute pressure variation for post-processing
                              ctx.aux.deltap[cell] = ctx.aux.p[cell] - static_cast<Number>(1e5);

                              // Compute frozen speed of sound for post-processing
                              ctx.aux.c_frozen[cell] = std::sqrt((ctx.conserved_variables[cell][M1_INDEX]*
                                                                  ctx.EOS_phase1.c_value(rho1_loc)*ctx.EOS_phase1.c_value(rho1_loc) +
                                                                  ctx.conserved_variables[cell][M2_INDEX]*
                                                                  ctx.EOS_phase2.c_value(rho2_loc)*ctx.EOS_phase2.c_value(rho2_loc))/rho_loc);

                              // Compute Wood speed of sound for post-processing
                              ctx.aux.c_Wood[cell]  = std::sqrt(static_cast<Number>(1.0)/
                                                                (rho_loc*
                                                                 (ctx.alpha1[cell]/
                                                                  (rho1_loc*ctx.EOS_phase1.c_value(rho1_loc)*ctx.EOS_phase1.c_value(rho1_loc)) +
                                                                  (static_cast<Number>(1.0) - ctx.alpha1[cell])/
                                                                  (rho2_loc*ctx.EOS_phase2.c_value(rho2_loc)*ctx.EOS_phase2.c_value(rho2_loc)))));

                            }
                        );

  // Update geometrical quantities
  ctx.grad_alpha1.fill(static_cast<Number>(0.0));
  //ctx.grad_alpha1 = ctx.gradient(ctx.alpha1);

  samurai::for_each_cell(ctx.mesh,
                         [&](const auto& cell)
                            {
                              auto mod2_grad_alpha1_loc = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < Field::dim; ++d) {
                                mod2_grad_alpha1_loc += ctx.grad_alpha1[cell][d]*ctx.grad_alpha1[cell][d];
                              }
                              const auto mod_grad_alpha1_loc = std::sqrt(mod2_grad_alpha1_loc);

                              if(mod_grad_alpha1_loc > mod_grad_alpha1_min) {
                                ctx.normal[cell] = ctx.grad_alpha1[cell]/mod_grad_alpha1_loc;
                              }
                              else {
                                for(std::size_t d = 0; d < Field::dim; ++d) {
                                  ctx.normal[cell][d] = static_cast<Number>(nan(""));
                                }
                              }
                            }
                        );
  samurai::update_ghost_mr(ctx.normal);
  ctx.H = -ctx.divergence(ctx.normal);
}

// Apply boundary conditions
//
template<typename Traits, typename AuxFields>
void AcousticWave<Traits, AuxFields>::apply_bcs(Field& conserved_variables) {
  // Homogeneous Neumann (free outflow)
  samurai::make_bc<samurai::Neumann<1>>(conserved_variables,
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0));
}

// setup(): assemble init_fn and bc_fnfrom the solver context
//
template<typename Traits, typename AuxFields>
template<typename T>
T AcousticWave<Traits, AuxFields>::CHeaviside(const T x, const T eps) {
  if(x < -eps) {
    return static_cast<T>(0.0);
  }
  else if(x > eps) {
    return static_cast<T>(1.0);
  }

  /*const auto pi = static_cast<T>(4.0)*static_cast<T>(std::atan(1));
  return static_cast<T>(0.5)*(static_cast<T>(1.0) + x/eps + static_cast<T>(1.0)/pi*std::sin(pi*x/eps));*/

  return static_cast<T>(0.5)*(static_cast<T>(1.0) + std::tanh(static_cast<T>(8.0)*(x/eps))/std::tanh(static_cast<T>(8.0)));
}
