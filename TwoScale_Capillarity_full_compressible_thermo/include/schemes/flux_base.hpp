// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2026
//
#pragma once

#include <samurai/schemes/fv.hpp>

#include "../eos.hpp"
#include "../utilities.hpp"

/*--- Preprocessor to define whether order 2 is desired ---*/
//#define ORDER_2

namespace samurai {
  using namespace EquationData;

  /**
   * Generic class to compute the flux between a left and right state
   */
  template<class Field>
  class Flux {
  public:
    // Definitions and sanity checks
    static_assert(Field::dim == EquationData::dim, "The spatial dimensions between Field and the parameter list do not match");
    static_assert(Field::n_comp == EquationData::NVARS, "The number of elements in the state does not correspond to the number of equations");
    #ifdef ORDER_2
      static constexpr std::size_t stencil_size = 4;
    #else
      static constexpr std::size_t stencil_size = 2;
    #endif

    using cfg = FluxConfig<SchemeType::NonLinear, stencil_size, Field, Field>;

    using Number = typename Field::value_type; // Shortcut for the arithmetic type

    /**
     * Class constructor
     * @param EOS_phase_liq_ liquid equation of state
     * @param EOS_phase_gas_ gas equation of state
     * @param sigma_ surface tension coefficient
     */
    Flux(const EOS<Number>& EOS_phase_liq_,
         const EOS<Number>& EOS_phase_gas_,
         const Number sigma_);

  protected:
    const EOS<Number>& EOS_phase_liq;
    const EOS<Number>& EOS_phase_gas;

    const Number sigma; /*!< Surface tension parameter */

    /**
     * Evaluate the 'continuous' hyperbolic flux
     * @param q state
     * @param grad_alpha_l gradient of large-scale volume fraction (needed for capillarity)
     * @param curr_d current direction
     */
    FluxValue<cfg> evaluate_conservative_hyperbolic_operator(const FluxValue<cfg>& q,
                                                             const auto& grad_alpha_l,
                                                             const std::size_t curr_d);

    /**
     * Conversion from conserved to primitive variables
     * @param cons conserved variables
     * @return prim primitive variables
     */
    FluxValue<cfg> cons2prim(const FluxValue<cfg>& cons,
                             const auto& grad_alpha_l) const;

    /**
     * Conversion from primitive to conserved variables
     * @param prim primitive variables
     * @return cons conserved variables
     */
    FluxValue<cfg> prim2cons(const FluxValue<cfg>& prim,
                             const auto& grad_alpha_l) const;
  };

  // Class constructor in order to be able to work with the equation of state
  //
  template<class Field>
  Flux<Field>::Flux(const EOS<Number>& EOS_phase_liq_,
                    const EOS<Number>& EOS_phase_gas_,
                    const Number sigma_):
    EOS_phase_liq(EOS_phase_liq_), EOS_phase_gas(EOS_phase_gas_), sigma(sigma_) {}

  // Evaluate the conservative portion of the hyperbolic part of the 'continuous' flux
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg>
  Flux<Field>::evaluate_conservative_hyperbolic_operator(const FluxValue<cfg>& q,
                                                         const auto& grad_alpha_l,
                                                         const std::size_t curr_d) {
    // Sanity check in terms of dimensions
    assert(curr_d < Field::dim);

    // Initialize the resulting variable
    FluxValue<cfg> res = q;

    // Pre-fetch some variables used multiple times in order to exploit possible vectorization
    const auto m_l      = q(Ml_INDEX);
    const auto m_g      = q(Mg_INDEX);
    const auto m_d      = q(Md_INDEX);
    const auto mliqEliq = q(Mliq_Eliq_INDEX);
    const auto mgEg     = q(Mg_Eg_INDEX);

    // Compute the current velocity
    const auto m_liq   = m_l + m_d;
    const auto rho     = m_liq + m_g;
    const auto inv_rho = static_cast<Number>(1.0)/rho;
    const auto vel_d   = q(RHO_U_INDEX + curr_d)*inv_rho;

    // Multiply the state by the velocity along the direction of interest
    res(Ml_INDEX) *= vel_d;
    res(Mg_INDEX) *= vel_d;
    res(Md_INDEX) *= vel_d;
    res(RHO_Z_INDEX) *= vel_d;
    res(RHO_ALPHA_l_INDEX) *= vel_d;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      res(RHO_U_INDEX + d) *= vel_d;
    }
    res(Mliq_Eliq_INDEX) *= vel_d;
    res(Mg_Eg_INDEX) *= vel_d;

    // Compute and add the contribution due to the pressure
    const auto alpha_l   = q(RHO_ALPHA_l_INDEX)*inv_rho;
    const auto alpha_d   = alpha_l*m_d/m_l; // TODO: Add a check in case of zero volume fraction
    const auto alpha_liq = alpha_l + alpha_d;

    auto norm2_vel = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      norm2_vel += (q(RHO_U_INDEX + d)*inv_rho)*(q(RHO_U_INDEX + d)*inv_rho);
    }

    auto mod2_grad_alpha_l = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      mod2_grad_alpha_l += grad_alpha_l[d]*grad_alpha_l[d];
    }
    const auto mod_grad_alpha_l = std::sqrt(mod2_grad_alpha_l);

    const auto rho_liq = m_liq/alpha_liq; // TODO: Add a check in case of zero volume fraction
    /*NOTE: Relation alpha_l/Y_l = (alpha_l + alpha_d)/(Y_l + Y_d) holds!!! */
    const auto Sigma_d = q(RHO_Z_INDEX)/std::cbrt(rho_liq*rho_liq);

    const auto Y_liq   = m_liq*inv_rho;
    const auto chi_liq = Y_liq;
    const auto e_liq   = mliqEliq/m_liq
                       - static_cast<Number>(0.5)*norm2_vel
                       - sigma*inv_rho*(chi_liq/Y_liq)*(Sigma_d + mod_grad_alpha_l); // TODO: Add a check in case of zero volume fraction

    const auto p_liq   = EOS_phase_liq.pres_value_Rhoe(rho_liq, e_liq);

    const auto alpha_g = static_cast<Number>(1.0) - alpha_liq;
    const auto rho_g   = m_g/alpha_g; // TODO: Add a check in case of zero volume fraction

    const auto Y_g   = static_cast<Number>(1.0) - Y_liq;
    const auto chi_g = Y_g;
    const auto e_g   = mgEg/m_g
                     - static_cast<Number>(0.5)*norm2_vel
                     - sigma*inv_rho*(chi_g/Y_g)*(Sigma_d + mod_grad_alpha_l); // TODO: Add a check in case of zero volume fraction

    const auto p_g   = EOS_phase_gas.pres_value_Rhoe(rho_g, e_g);

    const auto p = alpha_liq*p_liq
                 + alpha_g*p_g
                 - static_cast<Number>(2.0/3.0)*sigma*Sigma_d;

    res(RHO_U_INDEX + curr_d) += p;
    res(Mliq_Eliq_INDEX) += (alpha_liq*p_liq - static_cast<Number>(2.0/3.0)*sigma*chi_liq*Sigma_d)*vel_d;
    res(Mg_Eg_INDEX) += (alpha_g*p_g - static_cast<Number>(2.0/3.0)*sigma*chi_g*Sigma_d)*vel_d;

    return res;
  }

  // Conversion from conserved to primitive variables
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg>
  Flux<Field>::cons2prim(const FluxValue<cfg>& cons,
                         const auto& grad_alpha_l) const {
    FluxValue<cfg> prim;

    // Pre-fetch some variables used multiple times in order to exploit possible vectorization
    const auto m_l      = cons(Ml_INDEX);
    const auto m_g      = cons(Mg_INDEX);
    const auto m_d      = cons(Md_INDEX);
    const auto mliqEliq = cons(Mliq_Eliq_INDEX);
    const auto mgEg     = cons(Mg_Eg_INDEX);

    // Compute useful quantities
    const auto m_liq   = m_l + m_d;
    const auto rho     = m_liq + m_g;
    const auto inv_rho = static_cast<Number>(1.0)/rho;
    auto norm2_vel     = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      norm2_vel += (cons(RHO_U_INDEX + d)*inv_rho)*(cons(RHO_U_INDEX + d)*inv_rho);
    }

    auto mod2_grad_alpha_l = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      mod2_grad_alpha_l += grad_alpha_l[d]*grad_alpha_l[d];
    }
    const auto mod_grad_alpha_l = std::sqrt(mod2_grad_alpha_l);

    // Compute primitive variables
    const auto alpha_l  = cons(RHO_ALPHA_l_INDEX)*inv_rho;
    prim(ALPHA_l_INDEX) = alpha_l;

    const auto alpha_d   = alpha_l*m_d/m_l;
    prim(ALPHA_2d_INDEX) = alpha_d/(static_cast<Number>(1.0) - alpha_l);

    for(std::size_t d = 0; d < Field::dim; ++d) {
      prim(U_INDEX + d) = cons(RHO_U_INDEX + d)*inv_rho;
    }

    prim(Z_INDEX) = cons(RHO_Z_INDEX)*inv_rho;

    const auto alpha_liq = alpha_l + alpha_d;
    const auto rho_liq   = m_liq/alpha_liq; // TODO: Add a check in case of zero volume fraction
    prim(RHOl_INDEX)     = rho_liq;

    const auto Sigma_d = cons(RHO_Z_INDEX)/std::cbrt(rho_liq*rho_liq);
    const auto Y_liq   = m_liq*inv_rho;
    const auto chi_liq = Y_liq;
    const auto e_liq   = mliqEliq/m_liq
                       - static_cast<Number>(0.5)*norm2_vel
                       - sigma*inv_rho*(chi_liq/Y_liq)*(Sigma_d + mod_grad_alpha_l); // TODO: Add a check in case of zero volume fraction
    prim(Pl_INDEX)     = EOS_phase_liq.pres_value_Rhoe(rho_liq, e_liq);

    const auto rho_g = m_g/(static_cast<Number>(1.0) - alpha_liq); // TODO: Add a check in case of zero volume fraction
    prim(RHOg_INDEX) = rho_g;

    const auto Y_g   = static_cast<Number>(1.0) - Y_liq;
    const auto chi_g = Y_g;
    const auto e_g   = mgEg/m_g
                     - static_cast<Number>(0.5)*norm2_vel
                     - sigma*inv_rho*(chi_g/Y_g)*(Sigma_d + mod_grad_alpha_l); // TODO: Add a check in case of zero volume fraction
    prim(Pg_INDEX)   = EOS_phase_gas.pres_value_Rhoe(rho_g, e_g);

    return prim;
  }

  // Conversion from primitive to conserved variables
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg>
  Flux<Field>::prim2cons(const FluxValue<cfg>& prim,
                         const auto& grad_alpha_l) const {
    FluxValue<cfg> cons;

    // Pre-fetch some variables used multiple times in order to exploit possible vectorization
    const auto alpha_l = prim(ALPHA_l_INDEX);
    const auto alpha_d = prim(ALPHA_2d_INDEX)*(static_cast<Number>(1.0) - alpha_l);
    const auto rho_liq = prim(RHOl_INDEX);
    const auto rho_g   = prim(RHOg_INDEX);
    const auto p_liq   = prim(Pl_INDEX);
    const auto p_g     = prim(Pg_INDEX);

    // Compute conserved variables (except energies)
    const auto m_l = alpha_l*rho_liq;
    cons(Ml_INDEX) = m_l;

    const auto m_d = alpha_d*rho_liq;
    cons(Md_INDEX) = m_d;

    const auto alpha_liq = alpha_l + alpha_d;
    const auto m_g       = (static_cast<Number>(1.0) - alpha_liq)*rho_g;
    cons(Mg_INDEX)       = m_g;

    const auto m_liq = m_l + m_d;
    const auto rho   = m_liq + m_g;

    cons(RHO_ALPHA_l_INDEX) = rho*alpha_l;

    for(std::size_t d = 0; d < Field::dim; ++d) {
      cons(RHO_U_INDEX + d) = rho*prim(U_INDEX + d);
    }

    const auto rho_z  = rho*prim(Z_INDEX);
    cons(RHO_Z_INDEX) = rho_z;

    // Compute useful quantities
    auto norm2_vel = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      norm2_vel += prim(U_INDEX + d)*prim(U_INDEX + d);
    }

    auto mod2_grad_alpha_l = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      mod2_grad_alpha_l += grad_alpha_l[d]*grad_alpha_l[d];
    }
    const auto mod_grad_alpha_l = std::sqrt(mod2_grad_alpha_l);

    // Compute conserved variables related to energies
    const auto inv_rho = static_cast<Number>(1.0)/rho;
    const auto Sigma_d = rho_z/std::cbrt(rho_liq*rho_liq);

    const auto e_liq      = EOS_phase_liq.e_value_RhoP(rho_liq, p_liq);
    const auto Y_liq      = m_liq*inv_rho;
    const auto chi_liq    = Y_liq;
    cons(Mliq_Eliq_INDEX) = m_liq*(e_liq +
                                   static_cast<Number>(0.5)*norm2_vel +
                                   sigma*inv_rho*(chi_liq/Y_liq)*(Sigma_d + mod_grad_alpha_l)); // TODO: Add a check in case of zero volume fraction

    const auto e_g    = EOS_phase_gas.e_value_RhoP(rho_g, p_g);
    const auto Y_g    = static_cast<Number>(1.0) - Y_liq;
    const auto chi_g  = Y_g;
    cons(Mg_Eg_INDEX) = m_g*(e_g +
                             static_cast<Number>(0.5)*norm2_vel +
                             sigma*inv_rho*(chi_g/Y_g)*(Sigma_d + mod_grad_alpha_l)); // TODO: Add a check in case of zero volume fraction

    return cons;
  }

} // end namespace samurai
