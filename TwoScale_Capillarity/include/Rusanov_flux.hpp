// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#pragma once

#include "flux_base.hpp"

#define VERBOSE_FLUX

namespace samurai {
  using namespace EquationData;

  /**
    * Implementation of a Rusanov flux
    */
  template<class Field>
  class RusanovFlux: public Flux<Field> {
  public:
    using Number = Flux<Field>::Number; /*--- Define the shortcut for the arithmetic type ---*/
    using cfg    = Flux<Field>::cfg; /*--- Shortcut to specify the type of configuration
                                           for the flux (nonlinear in this case) ---*/

    RusanovFlux(const LinearizedBarotropicEOS<Number>& EOS_phase1_,
                const LinearizedBarotropicEOS<Number>& EOS_phase2_,
                const Number sigma_,
                const Number mod_grad_alpha1_bar_min_,
                const Number lambda_,
                const Number atol_Newton_,
                const Number rtol_Newton_,
                const std::size_t max_Newton_iters_); /*--- Constructor which accepts in input the equations of state of the two phases ---*/

    #ifdef ORDER_2
      template<typename Field_Scalar>
      auto make_two_scale_capillarity(const Field_Scalar& H_bar); /*--- Compute the flux over all the directions ---*/
    #else
      auto make_two_scale_capillarity(); /*--- Compute the flux over all the directions ---*/
    #endif

  private:
    FluxValue<cfg> compute_discrete_flux(const FluxValue<cfg>& qL,
                                         const FluxValue<cfg>& qR,
                                         const std::size_t curr_d); /*--- Rusanov flux along direction curr_d ---*/
  };

  // Constructor derived from the base class
  //
  template<class Field>
  RusanovFlux<Field>::RusanovFlux(const LinearizedBarotropicEOS<Number>& EOS_phase1_,
                                  const LinearizedBarotropicEOS<Number>& EOS_phase2_,
                                  const Number sigma_,
                                  const Number mod_grad_alpha1_bar_min_,
                                  const Number lambda_,
                                  const Number atol_Newton_,
                                  const Number rtol_Newton_,
                                  const std::size_t max_Newton_iters_):
    Flux<Field>(EOS_phase1_, EOS_phase2_,
                sigma_, mod_grad_alpha1_bar_min_,
                lambda_, atol_Newton_, rtol_Newton_, max_Newton_iters_) {}

  // Implementation of a Rusanov flux
  //
  template<class Field>
  FluxValue<typename RusanovFlux<Field>::cfg>
  RusanovFlux<Field>::compute_discrete_flux(const FluxValue<cfg>& qL,
                                            const FluxValue<cfg>& qR,
                                            const std::size_t curr_d) {
    /*--- Pre-fetch some variables used multiple times in order to exploit possible vectorization ---*/
    const auto m1_L             = qL(M1_INDEX);
    const auto m2_L             = qL(M2_INDEX);
    const auto m1_d_L           = qL(M1_D_INDEX);
    const auto rho_alpha1_bar_L = qL(RHO_ALPHA1_BAR_INDEX);
    const auto alpha1_d_L       = qL(ALPHA1_D_INDEX);
    const auto Sigma_d_L        = qL(SIGMA_D_INDEX);

    const auto m1_R             = qR(M1_INDEX);
    const auto m2_R             = qR(M2_INDEX);
    const auto m1_d_R           = qR(M1_D_INDEX);
    const auto rho_alpha1_bar_R = qR(RHO_ALPHA1_BAR_INDEX);
    const auto alpha1_d_R       = qR(ALPHA1_D_INDEX);
    const auto Sigma_d_R        = qR(SIGMA_D_INDEX);

    /*--- Verify if left and right state are coherent ---*/
    #ifdef VERBOSE_FLUX
      if(m1_L < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative mass large-scale phase 1 left state: " + std::to_string(m1_L)));
      }
      if(m2_L < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative mass phase 2 left state: " + std::to_string(m2_L)));
      }
      if(m1_d_L < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative mass small-scale phase 1 left state: " + std::to_string(m1_d_L)));
      }
      if(rho_alpha1_bar_L < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative large-scale volume fraction phase 1 left state: " + std::to_string(rho_alpha1_bar_L)));
      }
      if(alpha1_d_L < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative small-scale volume fraction phase 1 left state: " + std::to_string(alpha1_d_L)));
      }
      if(Sigma_d_L < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative small-scale IAD left state: " + std::to_string(Sigma_d_L)));
      }

      if(m1_R < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative mass large-scale phase 1 right state: " + std::to_string(m1_R)));
      }
      if(m2_R < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative mass phase 2 right state: " + std::to_string(m2_R)));
      }
      if(m1_d_R < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative mass small-scale phase 1 right state: " + std::to_string(m1_d_R)));
      }
      if(rho_alpha1_bar_R < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative large-scale volume fraction phase 1 right state: " + std::to_string(rho_alpha1_bar_R)));
      }
      if(alpha1_d_R < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative small-scale volume fraction phase 1 right state: " + std::to_string(alpha1_d_R)));
      }
      if(Sigma_d_R < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative small-scale IAD right state: " + std::to_string(Sigma_d_R)));
      }
    #endif

    /*--- Compute the quantities needed for the maximum eigenvalue estimate for the left state ---*/
    const auto rho_L          = m1_L + m2_L + m1_d_L;
    const auto inv_rho_L      = static_cast<Number>(1.0)/rho_L;
    const auto vel_d_L        = qL(RHO_U_INDEX + curr_d)*inv_rho_L;

    const auto alpha1_bar_L   = rho_alpha1_bar_L*inv_rho_L;
    const auto alpha1_L       = alpha1_bar_L*(static_cast<Number>(1.0) - alpha1_d_L);
    const auto rho1_L         = m1_L/alpha1_L; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto alpha2_L       = static_cast<Number>(1.0) - alpha1_L - alpha1_d_L;
    const auto rho2_L         = m2_L/alpha2_L; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rhoc_squared_L = m1_L*this->EOS_phase1.c_value(rho1_L)*this->EOS_phase1.c_value(rho1_L)
                              + m2_L*this->EOS_phase2.c_value(rho2_L)*this->EOS_phase2.c_value(rho2_L);
    const auto c_L            = std::sqrt(rhoc_squared_L*inv_rho_L)/
                                (static_cast<Number>(1.0) - alpha1_d_L);

    /*--- Compute the quantities needed for the maximum eigenvalue estimate for the right state ---*/
    const auto rho_R          = m1_R + m2_R + m1_d_R;
    const auto inv_rho_R      = static_cast<Number>(1.0)/rho_RL;
    const auto vel_d_R        = qR(RHO_U_INDEX + curr_d)*inv_rho_R;

    const auto alpha1_bar_R   = rho_alpha1_bar_R*inv_rho_R;
    const auto alpha1_R       = alpha1_bar_R*(static_cast<Number>(1.0) - alpha1_d_R);
    const auto rho1_R         = m1_R/alpha1_R; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto alpha2_R       = static_cast<Number>(1.0) - alpha1_R - alpha1_d_R;
    const auto rho2_R         = m2_R/alpha2_R; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rhoc_squared_R = m1_R*this->EOS_phase1.c_value(rho1_R)*this->EOS_phase1.c_value(rho1_R)
                              + m2_R*this->EOS_phase2.c_value(rho2_R)*this->EOS_phase2.c_value(rho2_R);
    const auto c_R            = std::sqrt(rhoc_squared_R*inv_rho_R)/
                                (static_cast<Number>(1.0) - alpha1_d_R);

    /*--- Compute the estimate of the eigenvalue ---*/
    const auto lambda = std::max(std::abs(vel_d_L) + c_L,
                                 std::abs(vel_d_R) + c_R);

    return static_cast<Number>(0.5)*
           (this->evaluate_hyperbolic_operator(qL, curr_d) +
            this->evaluate_hyperbolic_operator(qR, curr_d)) - // centered contribution
           static_cast<Number>(0.5)*lambda*(qR - qL); // upwinding contribution
  }

  // Implement the contribution of the discrete flux for all the directions.
  //
  template<class Field>
  #ifdef ORDER_2
    template<typename Field_Scalar>
    auto RusanovFlux<Field>::make_two_scale_capillarity(const Field_Scalar& H_bar)
  #else
    auto RusanovFlux<Field>::make_two_scale_capillarity()
  #endif
  {
    FluxDefinition<cfg> Rusanov_f;

    /*--- Perform the loop over each dimension to compute the flux contribution ---*/
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
         {
           static constexpr int d = decltype(integral_constant_d)::value;

           // Compute now the "discrete" flux function, in this case a Rusanov flux
           Rusanov_f[d].cons_flux_function = [&](FluxValue<cfg>& flux,
                                                 const StencilData<cfg>& data,
                                                 const StencilValues<cfg> field)
                                                 {
                                                   #ifdef ORDER_2
                                                     // MUSCL reconstruction
                                                     const FluxValue<cfg> primLL = this->cons2prim(field[0]);
                                                     const FluxValue<cfg> primL  = this->cons2prim(field[1]);
                                                     const FluxValue<cfg> primR  = this->cons2prim(field[2]);
                                                     const FluxValue<cfg> primRR = this->cons2prim(field[3]);

                                                     FluxValue<cfg> primL_recon,
                                                                    primR_recon;
                                                     this->perform_reconstruction(primLL, primL, primR, primRR,
                                                                                  primL_recon, primR_recon);

                                                     FluxValue<cfg> qL = this->prim2cons(primL_recon);
                                                     FluxValue<cfg> qR = this->prim2cons(primR_recon);

                                                     #ifdef RELAX_RECONSTRUCTION
                                                       this->relax_reconstruction(qL, H_bar[data.cells[1]][0]);
                                                       this->relax_reconstruction(qR, H_bar[data.cells[2]][0]);
                                                     #endif
                                                  #else
                                                    // Extract the states
                                                    const FluxValue<cfg> qL = field[0];
                                                    const FluxValue<cfg> qR = field[1];
                                                  #endif

                                                  // Compute the numerical flux
                                                  flux = compute_discrete_flux(qL, qR, d);
                                                };
        }
    );

    auto scheme = make_flux_based_scheme(Rusanov_f);
    scheme.set_name("Rusanov");

    return scheme;
  }

} // end of namespace
