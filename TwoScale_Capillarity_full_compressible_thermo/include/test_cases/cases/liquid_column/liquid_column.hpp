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
 * "p_init": initial pressure value
 * "rho_liq_init": initial liquid density value
 * "rho_g_init": initial gas density value
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
   * @param ctx struct with all conserved and auxiliary fields
   * @param sigma surface tension coefficient
   * @param alpha_residual 'residual' volume fraction
   */
  void apply_bcs(Context& ctx,
                 const Number sigma,
                 const Number alpha_residual);

  Number x0, y0;       /*!< Center of the liquid column */
  Number U0, U1;       /*!< Phase velocities (horizontal) */
  Number V0;           /*!< Vertical velocity */
  Number R;            /*!< Column radius */
  Number eps_over_R;   /*!< Interface thickness / radius */
  Number p_init;       /*!< Initial pressure */
  Number rho_liq_init; /*!< Initial density liquid phase */
  Number rho_g_init;   /*!< Initial density gas phase */
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
    x0           = input_tc.value("x0", static_cast<Number>(1.0));
    y0           = input_tc.value("y0", static_cast<Number>(1.0));
    U0           = input_tc.value("U0", static_cast<Number>(6.66));
    U1           = input_tc.value("U1", static_cast<Number>(0.0));
    V0           = input_tc.value("V0", static_cast<Number>(0.0));
    R            = input_tc.value("R", static_cast<Number>(0.15));
    eps_over_R   = input_tc.value("eps_over_R", static_cast<Number>(0.2));
    p_init       = input_tc.value("p_init", static_cast<Number>(1e5));
    rho_liq_init = input_tc.value("rho_liq_init", static_cast<Number>(1e3));
    rho_g_init   = input_tc.value("rho_g_init", static_cast<Number>(1.0));
  }
  catch(const json::parse_error& e) {
    // Default values in case file does not exist
    std::cerr << "LiquidColumn: cannot parse parameter file '" +
                  param_file + "': " + "using default values" << std::endl;

    x0           = static_cast<Number>(1.0);
    y0           = static_cast<Number>(1.0);
    U0           = static_cast<Number>(6.66);
    U1           = static_cast<Number>(0.0);
    V0           = static_cast<Number>(0.0);
    R            = static_cast<Number>(0.15);
    eps_over_R   = static_cast<Number>(0.2);
    p_init       = static_cast<Number>(1e5);
    rho_liq_init = static_cast<Number>(1e3);
    rho_g_init   = static_cast<Number>(1.0);
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

  this->bc_fn = [this, &ctx, sigma, alpha_residual]() {
    apply_bcs(ctx, sigma, alpha_residual);
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

  // Loop over a cell to complete the remaining variables
  samurai::for_each_cell(ctx.mesh,
                         [&](const auto& cell)
                            {
                              // Set small-scale variables
                              ctx.aux.alpha_d[cell]                      = static_cast<Number>(0.0);
                              ctx.conserved_variables[cell](RHO_Z_INDEX) = static_cast<Number>(0.0);
                              ctx.aux.rho_liq[cell]                      = rho_liq_init;
                              ctx.aux.Sigma_d[cell]                      = ctx.conserved_variables[cell](RHO_Z_INDEX)/
                                                                           std::cbrt(ctx.aux.rho_liq[cell]*ctx.aux.rho_liq[cell]);
                              ctx.conserved_variables[cell](Md_INDEX)    = ctx.aux.alpha_d[cell]*ctx.aux.rho_liq[cell];

                              // Recompute geometric locations to set partial masses
                              const auto center = cell.center();
                              const auto x      = static_cast<Number>(center[0]);
                              const auto y      = static_cast<Number>(center[1]);
                              const auto r      = std::sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));

                              // Set mass large-scale liquid phase
                              ctx.conserved_variables[cell](Ml_INDEX) = ctx.alpha_l[cell]*ctx.aux.rho_liq[cell];

                              // Set mass gas phase
                              ctx.aux.rho_g[cell] = rho_g_init;
                              const auto alpha_liq_loc = ctx.alpha_l[cell] + ctx.aux.alpha_d[cell];
                              const auto alpha_g_loc   = static_cast<Number>(1.0) - alpha_liq_loc;
                              ctx.conserved_variables[cell](Mg_INDEX) = alpha_g_loc*ctx.aux.rho_g[cell];

                              // Set conserved variable associated with large-scale volume fraction
                              const auto m_liq_loc = ctx.conserved_variables[cell](Ml_INDEX)
                                                   + ctx.conserved_variables[cell](Md_INDEX);
                              const auto rho_loc   = m_liq_loc + ctx.conserved_variables[cell](Mg_INDEX);

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

                              // Set total energy liquid phase
                              auto mod2_grad_alpha_l_loc = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < Field::dim; ++d) {
                                mod2_grad_alpha_l_loc += ctx.grad_alpha_l[cell][d]*ctx.grad_alpha_l[cell][d];
                              }
                              const auto mod_grad_alpha_l_loc = std::sqrt(mod2_grad_alpha_l_loc);

                              ctx.aux.p_liq[cell] = p_init;
                              if(r < R + eps_R) {
                                if(r >= R && r < R + eps_R && !std::isnan(ctx.H[cell])) {
                                  ctx.aux.p_liq[cell] += sigma*ctx.H[cell];
                                }
                                else {
                                  ctx.aux.p_liq[cell] += sigma/R;
                                }
                              }
                              const auto Y_liq_loc   = m_liq_loc/rho_loc;
                              const auto chi_liq_loc = Y_liq_loc;
                              ctx.conserved_variables[cell](Mliq_Eliq_INDEX) = ctx.conserved_variables[cell](Ml_INDEX)*
                                                                               (ctx.EOS_phase_liq.e_value_RhoP(ctx.aux.rho_liq[cell], ctx.aux.p_liq[cell]) +
                                                                                static_cast<Number>(0.5)*norm2_vel_loc +
                                                                                sigma/rho_loc*(chi_liq_loc/Y_liq_loc)*
                                                                                (mod_grad_alpha_l_loc + ctx.aux.Sigma_d[cell]));
                                                                               // TODO: Add a check in case of zero volume fraction

                              // Set total energy gas phase
                              ctx.aux.p_g[cell] = p_init;
                              const auto Y_g_loc   = static_cast<Number>(1.0) - Y_liq_loc;
                              const auto chi_g_loc = Y_g_loc;
                              ctx.conserved_variables[cell](Mg_Eg_INDEX) = ctx.conserved_variables[cell](Mg_INDEX)*
                                                                           (ctx.EOS_phase_gas.e_value_RhoP(ctx.aux.rho_g[cell], ctx.aux.p_g[cell]) +
                                                                            static_cast<Number>(0.5)*norm2_vel_loc +
                                                                            sigma/rho_loc*(chi_g_loc/Y_g_loc)*
                                                                            (mod_grad_alpha_l_loc + ctx.aux.Sigma_d[cell]));
                                                                           // TODO: Add a check in case of zero volume fraction

                              // Save mixture pressure for post-processing
                              ctx.aux.p[cell] = alpha_liq_loc*ctx.aux.p_liq[cell]
                                              + alpha_g_loc*ctx.aux.p_g[cell]
                                              - static_cast<Number>(2.0/3.0)*sigma*ctx.aux.Sigma_d[cell];

                              // Save phasic temperatures for post-processing
                              ctx.aux.T_liq[cell] = ctx.EOS_phase_liq.T_value_RhoP(ctx.aux.rho_liq[cell], ctx.aux.p_liq[cell]);
                              ctx.aux.T_g[cell]   = ctx.EOS_phase_gas.T_value_RhoP(ctx.aux.rho_g[cell], ctx.aux.p_g[cell]);

                              // Save Mach number for post-processing
                              const auto c_liq_loc = ctx.EOS_phase_liq.c_value_RhoP(ctx.aux.rho_liq[cell], ctx.aux.p_liq[cell]);
                              const auto c_g_loc   = ctx.EOS_phase_gas.c_value_RhoP(ctx.aux.rho_g[cell], ctx.aux.p_g[cell]);
                              const auto cf_loc    = std::sqrt(Y_liq_loc*c_liq_loc*c_liq_loc +
                                                               Y_g_loc*c_g_loc*c_g_loc -
                                                               static_cast<Number>(2.0/9.0)*sigma*ctx.aux.Sigma_d[cell]/rho_loc);
                              ctx.aux.Mach[cell]   = std::sqrt(norm2_vel_loc)/cf_loc;
                            }
                        );
}

// Apply boundary conditions
//
template<typename Traits, typename AuxFields>
void LiquidColumn<Traits, AuxFields>::apply_bcs(Context& ctx,
                                                const Number sigma,
                                                const Number alpha_residual) {
  // Left boundary: prescribed inlet condition
  const samurai::DirectionVector<Field::dim> left = {-1, 0};
  samurai::make_bc<Default>(ctx.conserved_variables,
                            Inlet(ctx.conserved_variables, ctx.grad_alpha_l, sigma,
                                  U0, V0, alpha_residual,
                                  static_cast<Number>(0.0),
                                  static_cast<Number>(0.0)))->on(left);

  // Right boundary: homogeneous Neumann (free outflow)
  const samurai::DirectionVector<Field::dim> right = {1, 0};
  samurai::make_bc<samurai::Neumann<1>>(ctx.conserved_variables,
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0))->on(right);
}
