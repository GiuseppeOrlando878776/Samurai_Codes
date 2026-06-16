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
 * @file static_bubble.hpp
 *
 * @brief Test case: static bubble
 *
 * Physical configuration
 * ----------------------
 * A static bubble is considered. This configuration is stadardly used
 * to assess surface tension modelling and the presence of spurious
 * velocity
 *
 * Parameter file (json)
 * ---------------------
 * All physical parameters are read from a dedicated input file whose path
 * is provided at construction time. A minimal file looks like:
 *
 * @code{.json}
 * "x0": 0.0 # x-center of the bubble
 * "y0": 0.0 # y-center of the bubble
 * "R": 0.4 # radius of the bubble
 * "eps_over_R": 0.1 # initial interface thickness / R
 * @endcode
 *
 * Scalar parameters from SolverContext
 * -------------------------------------
 * The following keys must be present in ctx.params when setup() is called:
 *   "sigma"  — surface tension coefficient
 *   "alpha_residual" - 'residual' volume fraction
 *   "mod_grad_alpha1_min" - threshold for which computing the unit normal vector
 *
 * Contract with the solver
 * ------------------------
 * setup() uses the auxiliary fields and
 * ctx.params["sigma"], ctx.params["alpha_residual"], ctx.params["mod_grad_alpha1_min"]
 * from the scalar map. All other context fields are ignored.
 *
 * @tparam Traits Traits struct defined in the solver header.
 * @tparam AuxFields Auxiliary fields struct defined in the solver header.
 */
template<typename Traits, typename AuxFields>
class StaticBubble final : public TestCaseBase<Traits, AuxFields> {
public:
  using Context = typename StaticBubble<Traits, AuxFields>::Context;

  using Number = typename Traits::Number;
  using Field  = typename Traits::Field;

  /**
   * Static bubble constructor
   * @param_file name of the parameter files
   */
  explicit StaticBubble(const std::string& param_file);

  /**
    * Retrieve scalar parameters from ctx.params, then build init_fn
    * capturing the needed context references and private members.
    * ctx contains both conserved and auxiliary fields
    *
    * Required ctx.params keys: "sigma", "alpha_residual", "mod_grad_alpha1_min".
    * @param ctx struct with all conserved and auxiliary fields
   */
  void setup(Context& ctx) override;

private:
  /**
   * Initialize all conserved and auxiliary fields.
   * The implementation is free to call samurai::for_each_cell, etc.
   * The mesh has already been constructed by the solver before this call.
   * @param ctx struct with all conserved and auxiliary fields
   * @param sigma surface tension coefficient
   * @param alpha_residual 'residual' volume fraction
   * @param mod_grad_alpha1_min threshold of gradient of large-scale volume fraction to compute the normal
   */
  void init_variables(Context& ctx,
                      const Number sigma,
                      const Number alpha_residual,
                      const Number mod_grad_alpha1_min);

  Number x0, y0;     /*!< Center of the static bubble */
  Number R;          /*!< Bubble radius */
  Number eps_over_R; /*!< Interface thickness / radius */
};

// Static bubble constructor
//
template<typename Traits, typename AuxFields>
StaticBubble<Traits, AuxFields>::StaticBubble(const std::string& param_file) {
  using json = nlohmann::json;

  try {
    std::ifstream ifs(param_file);
    json input_tc = json::parse(ifs);

    // Read with safe defaults so missing keys are not fatal
    x0         = input_tc.value("x0", static_cast<Number>(0.0));
    y0         = input_tc.value("y0", static_cast<Number>(0.0));
    R          = input_tc.value("R", static_cast<Number>(0.4));
    eps_over_R = input_tc.value("eps_over_R", static_cast<Number>(0.1));
  }
  catch(const json::parse_error& e) {
    // Defaults values in case file does not exist
    std::cerr << "StaticBubble: cannot parse parameter file '" +
                  param_file + "': " + "using default values" << std::endl;

    x0         = static_cast<Number>(0.0);
    y0         = static_cast<Number>(0.0);
    R          = static_cast<Number>(0.4);
    eps_over_R = static_cast<Number>(0.1);
  }
}

// setup(): assemble init_fn from the solver context
//
template<typename Traits, typename AuxFields>
void StaticBubble<Traits, AuxFields>::setup(Context& ctx) {
  // Retrieve scalar parameters — throws std::out_of_range if absent,
  // with a message that names this class and the missing key.
  const Number sigma               = ctx.param("sigma", "StaticBubble");
  const Number alpha_residual      = ctx.param("alpha_residual", "StaticBubble");
  const Number mod_grad_alpha1_min = ctx.param("mod_grad_alpha1_min", "StaticBubble");

  // Save the radius for post-processing
  ctx.params["R"] = R;

  // Capture ctx by reference: the solver guarantees it outlives init_fn.
  // Capture 'this' by pointer: StaticBubble outlives the solver.
  // Capture sigma, alpha_residual, and mod_grad_alpha1_min by values: they are lightweight scalars.
  this->init_fn = [this, &ctx, sigma, alpha_residual, mod_grad_alpha1_min]() {
    init_variables(ctx, sigma, alpha_residual, mod_grad_alpha1_min);
  };
}

// Initialize conserved and auxiliary variables
//
template<typename Traits, typename AuxFields>
void StaticBubble<Traits, AuxFields>::init_variables(Context& ctx,
                                                     const Number sigma,
                                                     const Number alpha_residual,
                                                     const Number mod_grad_alpha1_min) {
  // Derived geometric constant
  const auto eps_R = eps_over_R*R;

  // Initialize the large-scale volume fraction to define the static bubble with a loop over all cells
  samurai::for_each_cell(ctx.mesh,
                         [&](const auto& cell)
                            {
                              // Set large-scale volume fraction
                              const auto center = cell.center();
                              const auto x      = static_cast<Number>(center[0]);
                              const auto y      = static_cast<Number>(center[1]);
                              const auto r      = std::sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));
                              const auto w      = (r >= R - static_cast<Number>(0.5)*eps_R && r <= R + static_cast<Number>(0.5)*eps_R) ?
                                                  static_cast<Number>(0.5)*
                                                  (static_cast<Number>(1.0) +
                                                   std::tanh(static_cast<Number>(-8.0)*
                                                             ((r - R + static_cast<Number>(0.5)*eps_R)/eps_R - static_cast<Number>(0.5)))) :
                                                  ((r < R - static_cast<Number>(0.5)*eps_R) ? static_cast<Number>(1.0) : static_cast<Number>(0.0));

                              ctx.alpha1[cell] = std::min(std::max(alpha_residual, w),
                                                          static_cast<Number>(1.0) - alpha_residual);
                            }
                        );

  // Update geometrical quantities
  ctx.grad_alpha1 = ctx.gradient(ctx.alpha1);

  samurai::for_each_cell(ctx.mesh,
                         [&](const auto& cell)
                            {
                              auto mod2_grad_alpha1_loc = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                mod2_grad_alpha1_loc += ctx.grad_alpha1[cell][d]*ctx.grad_alpha1[cell][d];
                              }
                              const auto mod_grad_alpha1_loc = std::sqrt(mod2_grad_alpha1_loc);

                              if(mod_grad_alpha1_loc > mod_grad_alpha1_min) {
                                ctx.normal[cell] = ctx.grad_alpha1[cell]/mod_grad_alpha1_loc;
                              }
                              else {
                                for(std::size_t d = 0; d < dim; ++d) {
                                  ctx.normal[cell][d] = static_cast<Number>(nan(""));
                                }
                              }
                            }
                        );
  samurai::update_ghost_mr(ctx.normal);
  ctx.H = -ctx.divergence(ctx.normal);

  // Loop over a cell to complete the remaining variables
  samurai::for_each_cell(ctx.mesh,
                         [&](const auto& cell)
                            {
                              // Recompute geometric locations to set partial masses
                              const auto center = cell.center();
                              const auto x      = static_cast<Number>(center[0]);
                              const auto y      = static_cast<Number>(center[1]);
                              const auto r      = std::sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));

                              // Set mass phase 1
                              ctx.aux.p1[cell] = ctx.EOS_phase2.get_p0();
                              if(r < R + eps_R) {
                                if(r >= R && r < R + eps_R && !std::isnan(ctx.H[cell])) {
                                  ctx.aux.p1[cell] += sigma*ctx.H[cell];
                                }
                                else {
                                  ctx.aux.p1[cell] += sigma/R;
                                }
                              }
                              const auto rho1_loc = ctx.EOS_phase1.rho_value(ctx.aux.p1[cell]);

                              ctx.conserved_variables[cell](M1_INDEX) = ctx.alpha1[cell]*rho1_loc;

                              // Set mass phase 2
                              ctx.aux.p2[cell]    = ctx.EOS_phase2.get_p0();
                              const auto rho2_loc = ctx.EOS_phase2.rho_value(ctx.aux.p2[cell]);

                              ctx.conserved_variables[cell](M2_INDEX) = (static_cast<Number>(1.0) - ctx.alpha1[cell])*rho2_loc;

                              // Save mixture pressure for post-processing
                              ctx.aux.p[cell] = ctx.alpha1[cell]*ctx.aux.p1[cell]
                                              + (static_cast<Number>(1.0) - ctx.alpha1[cell])*ctx.aux.p2[cell];

                              // Set conserved variable associated with volume fraction
                              const auto rho_loc = ctx.conserved_variables[cell](M1_INDEX)
                                                 + ctx.conserved_variables[cell](M2_INDEX);

                              ctx.conserved_variables[cell](RHO_ALPHA1_INDEX) = rho_loc*ctx.alpha1[cell];

                              // Set momentum
                              ctx.conserved_variables[cell](RHO_U_INDEX)     = static_cast<Number>(0.0);
                              ctx.conserved_variables[cell](RHO_U_INDEX + 1) = static_cast<Number>(0.0);

                              // Save velocity for post-processing
                              for(std::size_t d = 0; d < dim; ++d) {
                                ctx.aux.vel[cell][d] = ctx.conserved_variables[cell](RHO_U_INDEX + d)/rho_loc;
                              }
                            }
                        );
}
