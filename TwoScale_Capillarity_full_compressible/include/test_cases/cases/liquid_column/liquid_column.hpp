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
#include "user_bc.hpp"

using namespace EquationData;

/**
 * @file liquid_column.hpp
 *
 * @brief Test case: liquid column impacting a cross-flow.
 *
 * Physical configuration
 * ----------------------
 * A circular liquid column of radius R centred at (x0, y0) is embedded in
 * a gas cross-flow with horizontal velocity U0. The liquid phase has
 * horizontal velocity U1 and the mixture has vertical velocity V0.
 * The interface is smoothed over a layer of thickness eps_over_R * R
 * using a C^2 mollifier.
 *
 * Parameter file (json)
 * ---------------------
 * All physical parameters are read from a dedicated input file whose path
 * is provided at construction time. A minimal file looks like:
 *
 * @code{.json}
 * "x0": 1.0 # x-center of the column
 * "y0": 1.0 # y-center of the column
 * "U0": 6.66 # horizontal velocity of phase 2 (cross-flow)
 * "U1": 0.0 # horizontal velocity of phase 1 (liquid)
 * "V0": 0.0 # vertical velocity
 * "R": 0.15 # column radius
 * "eps_over_R": 0.2 # initial interface thickness / R
 * @endcode
 *
 * Scalar parameters from SolverContext
 * -------------------------------------
 * The following keys must be present in ctx.params when setup() is called:
 *   "sigma"  — surface tension coefficient
 *   "alpha_residual" - 'residual' volume fraction
 *   "mod_grad_alpha_l_min" - threshold for which computing the unit normal vector
 *
 * Boundary conditions
 * -------------------
 * Left  boundary: prescribed inlet (Inlet functor from user_bc.hpp).
 * Right boundary: homogeneous Neumann (free outflow) on all components.
 *
 * Contract with the solver
 * ------------------------
 * setup() uses the auxiliary fields and
 * ctx.params["sigma"], ctx.params["alpha_residual"], ctx.params["mod_grad_alpha_l_min"]
 * from the scalar map. All other context fields are ignored.
 *
 * @tparam Traits Traits struct defined in the solver header.
 * @tparam AuxFields Auxiliary fields struct defined in the solver header.
 */
template<typename Traits, typename AuxFields>
class LiquidColumn final : public TestCaseBase<Traits, AuxFields> {
public:
  using Context = typename LiquidColumn<Traits, AuxFields>::Context;

  using Number = typename Traits::Number;
  using Field  = typename Traits::Field;

  /**
   * Liquid column constructor
   * @param_file name of the parameter files
   */
  explicit LiquidColumn(const std::string& param_file);

  /**
    * Retrieve scalar parameters from ctx.params, then build init_fn and
    * bc_fn capturing the needed context references and private members.
    * ctx contains both conserved and auxiliary fields
    *
    * Required ctx.params keys: "sigma", "alpha_residual", "mod_grad_alpha_l_min".
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
   * @param mod_grad_alpha_l_min threshold of gradient of large-scale volume fraction to compute the normal
   */
  void init_variables(Context& ctx,
                      const Number sigma,
                      const Number alpha_residual,
                      const Number mod_grad_alpha_l_min);

  /**
   * Attach boundary conditions to the conserved variable field.
   *
   * Left  boundary: inlet BC (prescribed state via Inlet functor).
   * Right boundary: homogeneous Neumann on all NVARS components.
   * @param conserved_variables field with conserved state
   * @param alpha_residual 'residual' volume fraction
   */
  void apply_bcs(Field& conserved_variables,
                 const Number alpha_residual);

  Number x0, y0;     /*!< Center of the liquid column */
  Number U0, U1;     /*!< Phase velocities (horizontal) */
  Number V0;         /*!< Vertical velocity */
  Number R;          /*!< Column radius */
  Number eps_over_R; /*!< Interface thickness / radius */
};

// Liquid column constructor
//
template<typename Traits, typename AuxFields>
LiquidColumn<Traits, AuxFields>::LiquidColumn(const std::string& param_file) {
  using json = nlohmann::json;

  try {
    std::ifstream ifs(param_file);
    json input_tc = json::parse(ifs);

    // Read with safe defaults so missing keys are not fatal
    x0         = input_tc.value("x0", static_cast<Number>(1.0));
    y0         = input_tc.value("y0", static_cast<Number>(1.0));
    U0         = input_tc.value("U0", static_cast<Number>(6.66));
    U1         = input_tc.value("U1", static_cast<Number>(0.0));
    V0         = input_tc.value("V0", static_cast<Number>(0.0));
    R          = input_tc.value("R", static_cast<Number>(0.15));
    eps_over_R = input_tc.value("eps_over_R", static_cast<Number>(0.2));
  }
  catch(const json::parse_error& e) {
    // Default values in case file does not exist
    std::cerr << "LiquidColumn: cannot parse parameter file '" +
                  param_file + "': " + "using default values" << std::endl;

    x0         = static_cast<Number>(1.0);
    y0         = static_cast<Number>(1.0);
    U0         = static_cast<Number>(6.66);
    U1         = static_cast<Number>(0.0);
    V0         = static_cast<Number>(0.0);
    R          = static_cast<Number>(0.15);
    eps_over_R = static_cast<Number>(0.2);
  }
}

// setup(): assemble init_fn and bc_fn from the solver context
//
template<typename Traits, typename AuxFields>
void LiquidColumn<Traits, AuxFields>::setup(Context& ctx) {
  // Retrieve scalar parameters — throws std::out_of_range if absent,
  // with a message that names this class and the missing key.
  const Number sigma                = ctx.param("sigma", "LiquidColumn");
  const Number alpha_residual       = ctx.param("alpha_residual", "LiquidColumn");
  const Number mod_grad_alpha_l_min = ctx.param("mod_grad_alpha_l_min", "LiquidColumn");

  // Capture ctx by reference: the solver guarantees it outlives init_fn
  // and bc_fn. Capture 'this' by pointer: LiquidColumn outlives the solver.
  // Capture sigma, alpha_residual, and mod_grad_alpha_l_min by values: they are lightweight scalars.
  this->init_fn = [this, &ctx, sigma, alpha_residual, mod_grad_alpha_l_min]() {
    init_variables(ctx, sigma, alpha_residual, mod_grad_alpha_l_min);
  };

  this->bc_fn = [this, &ctx, alpha_residual]() {
    apply_bcs(ctx.conserved_variables, alpha_residual);
  };
}

// Initialize conserved and auxiliary variables
//
template<typename Traits, typename AuxFields>
void LiquidColumn<Traits, AuxFields>::init_variables(Context& ctx,
                                                     const Number sigma,
                                                     const Number alpha_residual,
                                                     const Number mod_grad_alpha_l_min) {
  // Derived geometric constant
  const auto eps_R = eps_over_R*R;

  // Initialize the large-scale volume fraction to define the liquid column with a loop over all cells
  samurai::for_each_cell(ctx.mesh,
                         [&](const auto& cell)
                            {
                              // Set large-scale volume fraction
                              const auto center = cell.center();
                              const auto x      = static_cast<Number>(center[0]);
                              const auto y      = static_cast<Number>(center[1]);
                              const auto r      = std::sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));
                              const auto w      = (r >= R && r < R + eps_R) ?
                                                  std::exp(static_cast<Number>(2.0)*
                                                           (r - R)*(r - R)/(eps_R*eps_R)*
                                                           ((r - R)*(r - R)/(eps_R*eps_R) - static_cast<Number>(3.0))/
                                                           (((r - R)*(r - R)/(eps_R*eps_R) - static_cast<Number>(1.0))*
                                                            ((r - R)*(r - R)/(eps_R*eps_R) - static_cast<Number>(1.0)))) :
                                                  ((r < R) ? static_cast<Number>(1.0) :
                                                             static_cast<Number>(0.0));

                              ctx.alpha_l[cell] = std::min(std::max(alpha_residual, w),
                                                           static_cast<Number>(1.0) - alpha_residual);
                            }
                        );

  // Update geometrical quantities
  ctx.grad_alpha_l = ctx.gradient(ctx.alpha_l);

  samurai::for_each_cell(ctx.mesh,
                         [&](const auto& cell)
                            {
                              const auto& grad_alpha_l_loc = ctx.grad_alpha_l[cell];
                              auto mod2_grad_alpha_l_loc   = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < Field::dim; ++d) {
                                mod2_grad_alpha_l_loc += grad_alpha_l_loc[d]*grad_alpha_l_loc[d];
                              }
                              const auto mod_grad_alpha_l_loc = std::sqrt(mod2_grad_alpha_l_loc);

                              if(mod_grad_alpha_l_loc > mod_grad_alpha_l_min) {
                                ctx.normal[cell] = grad_alpha_l_loc/mod_grad_alpha_l_loc;
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

  // Apply the filter to the curvature
  if(ctx.apply_filter) {
    ctx.filter.apply(ctx.H_filter, ctx.H);
    samurai::swap(ctx.H_filter, ctx.H);
  }

  // Loop over a cell to complete the remaining variables
  samurai::for_each_cell(ctx.mesh,
                         [&](const auto& cell)
                            {
                              // Set small-scale variables
                              ctx.aux.alpha_d[cell]                      = static_cast<Number>(0.0);
                              ctx.conserved_variables[cell](RHO_Z_INDEX) = static_cast<Number>(0.0);
                              const auto rho_liq_ref                     = ctx.EOS_phase_liq.get_rho0();
                              ctx.aux.Sigma_d[cell]                      = ctx.conserved_variables[cell](RHO_Z_INDEX)/std::cbrt(rho_liq_ref*rho_liq_ref);
                              ctx.conserved_variables[cell](Md_INDEX)    = ctx.aux.alpha_d[cell]*rho_liq_ref;

                              // Recompute geometric locations to set partial masses
                              const auto center = cell.center();
                              const auto x      = static_cast<Number>(center[0]);
                              const auto y      = static_cast<Number>(center[1]);
                              const auto r      = std::sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));

                              // Set mass large-scale liquid phase
                              if(r >= R + eps_R) {
                                ctx.aux.p_liq[cell] = ctx.EOS_phase_liq.get_p0();
                              }
                              else {
                                ctx.aux.p_liq[cell] = ctx.EOS_phase_gas.get_p0();
                                if(r >= R && r < R + eps_R && !std::isnan(ctx.H[cell])) {
                                  ctx.aux.p_liq[cell] += sigma*ctx.H[cell];
                                }
                                else {
                                  ctx.aux.p_liq[cell] += sigma/R;
                                }
                              }
                              const auto rho_liq_loc = ctx.EOS_phase_liq.rho_value(ctx.aux.p_liq[cell]);

                              ctx.conserved_variables[cell](Ml_INDEX) = ctx.alpha_l[cell]*rho_liq_loc;

                              // Set mass gas phase
                              ctx.aux.p_g[cell]    = ctx.EOS_phase_gas.get_p0();
                              const auto rho_g_loc = ctx.EOS_phase_gas.rho_value(ctx.aux.p_g[cell]);

                              const auto alpha_liq_loc = ctx.alpha_l[cell] + ctx.aux.alpha_d[cell];
                              const auto alpha_g_loc   = static_cast<Number>(1.0) - alpha_liq_loc;
                              ctx.conserved_variables[cell](Mg_INDEX) = alpha_g_loc*rho_g_loc;

                              // Save mixture pressure for post-processing
                              ctx.aux.p[cell] = alpha_liq_loc*ctx.aux.p_liq[cell]
                                              + alpha_g_loc*ctx.aux.p_g[cell]
                                              - static_cast<Number>(2.0/3.0)*sigma*ctx.aux.Sigma_d[cell];

                              // Set conserved variable associated with large-scale volume fraction
                              const auto rho_loc = ctx.conserved_variables[cell](Ml_INDEX)
                                                 + ctx.conserved_variables[cell](Mg_INDEX)
                                                 + ctx.conserved_variables[cell](Md_INDEX);

                              ctx.conserved_variables[cell](RHO_ALPHA_l_INDEX) = rho_loc*ctx.alpha_l[cell];

                              // Set momentum
                              ctx.conserved_variables[cell](RHO_U_INDEX)     = ctx.conserved_variables[cell](Ml_INDEX)*U1
                                                                             + ctx.conserved_variables[cell](Mg_INDEX)*U0;
                              ctx.conserved_variables[cell](RHO_U_INDEX + 1) = rho_loc*V0;

                              // Save velocity for post-processing
                              auto norm2_vel_loc = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < Field::dim; ++d) {
                                ctx.aux.vel[cell][d] = ctx.conserved_variables[cell](RHO_U_INDEX + d)/rho_loc;
                                norm2_vel_loc += ctx.aux.vel[cell][d]*ctx.aux.vel[cell][d];
                              }

                              // Save 'bar' volume fraction for post-processing
                              ctx.aux.alpha_l_bar[cell] = ctx.alpha_l[cell]/(static_cast<Number>(1.0) - ctx.aux.alpha_d[cell]);

                              // Save Mach number for post-processing
                              const auto Y_g_loc   = ctx.conserved_variables[cell](Mg_INDEX)/rho_loc;
                              const auto c_liq_loc = ctx.EOS_phase_liq.c_value(rho_liq_loc);
                              const auto c_g_loc   = ctx.EOS_phase_gas.c_value(rho_g_loc);
                              const auto cf_loc    = std::sqrt((static_cast<Number>(1.0) - Y_g_loc)*c_liq_loc*c_liq_loc +
                                                               Y_g_loc*c_g_loc*c_g_loc -
                                                               static_cast<Number>(2.0/9.0)*sigma*ctx.aux.Sigma_d[cell]/rho_loc);
                              ctx.aux.Mach[cell]   = std::sqrt(norm2_vel_loc)/cf_loc;
                            }
                        );

  // Set useful small-scale related fields
  samurai::update_ghost_mr(ctx.aux.alpha_d);
  ctx.aux.grad_alpha_d.fill(static_cast<Number>(0.0));
  ctx.gradient.apply(ctx.aux.grad_alpha_d, ctx.aux.alpha_d);

  // Set auxiliary gradient alpha_l_bar volume fraction
  samurai::update_ghost_mr(ctx.aux.alpha_l_bar);
  ctx.aux.grad_alpha_l_bar.fill(static_cast<Number>(0.0));
  ctx.gradient.apply(ctx.aux.grad_alpha_l_bar, ctx.aux.alpha_l_bar);
  samurai::for_each_cell(ctx.mesh,
                         [&](const auto& cell)
                            {
                              const auto& grad_alpha_l_bar_loc = ctx.aux.grad_alpha_l_bar[cell];
                              auto mod2_grad_alpha_l_bar_loc   = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < Field::dim; ++d) {
                                mod2_grad_alpha_l_bar_loc += grad_alpha_l_bar_loc[d]*grad_alpha_l_bar_loc[d];
                              }
                              const auto mod_grad_alpha_l_bar_loc = std::sqrt(mod2_grad_alpha_l_bar_loc);

                              if(mod_grad_alpha_l_bar_loc > mod_grad_alpha_l_min) {
                                ctx.aux.normal_bar[cell] = grad_alpha_l_bar_loc/mod_grad_alpha_l_bar_loc;
                              }
                              else {
                                for(std::size_t d = 0; d < Field::dim; ++d) {
                                  ctx.aux.normal_bar[cell][d] = static_cast<Number>(nan(""));
                                }
                              }
                            }
                        );
  samurai::update_ghost_mr(ctx.aux.normal_bar);
  ctx.aux.H_bar = -ctx.divergence(ctx.aux.normal_bar);
}

// Apply boundary conditions
//
template<typename Traits, typename AuxFields>
void LiquidColumn<Traits, AuxFields>::apply_bcs(Field& conserved_variables,
                                                const Number alpha_residual) {
  // Left boundary: prescribed inlet condition
  const samurai::DirectionVector<Field::dim> left = {-1, 0};
  samurai::make_bc<Default>(conserved_variables,
                            Inlet(conserved_variables, U0, V0, alpha_residual,
                                  static_cast<Number>(0.0),
                                  static_cast<Number>(0.0)))->on(left);

  // Right boundary: homogeneous Neumann (free outflow)
  const samurai::DirectionVector<Field::dim> right = {1, 0};
  samurai::make_bc<samurai::Neumann<1>>(conserved_variables,
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0))->on(right);
}
