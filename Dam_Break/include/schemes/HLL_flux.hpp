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
    * Implementation of a HLL flux
    */
  template<class Field>
  class HLLFlux: public Flux<Field> {
  public:
    using cfg = Flux<Field>::cfg;

    using Number = Flux<Field>::Number; /*--- Define the shortcut for the arithmetic type ---*/

    HLLFlux(const LinearizedBarotropicEOS<Number>& EOS_phase1_,
            const LinearizedBarotropicEOS<Number>& EOS_phase2_,
            const Number lambda_,
            const Number atol_Newton_,
            const Number rtol_Newton_,
            const std::size_t max_Newton_iters_); /*--- Constructor which accepts in input the equations of state of the two phases
                                                        (as well as other parameters for relaxation) ---*/

    virtual decltype(make_flux_based_scheme(std::declval<FluxDefinition<cfg>>())) make_flux() override;
    /*--- Compute the flux over all the faces and directions ---*/

  private:
    FluxValue<cfg> compute_discrete_flux(const FluxValue<cfg>& qL,
                                         const FluxValue<cfg>& qR,
                                         const std::size_t curr_d); /*--- HLL flux along direction d ---*/
  };

  // Constructor derived from base class
  //
  template<class Field>
  HLLFlux<Field>::HLLFlux(const LinearizedBarotropicEOS<Number>& EOS_phase1_,
                          const LinearizedBarotropicEOS<Number>& EOS_phase2_,
                          const Number lambda_,
                          const Number atol_Newton_,
                          const Number rtol_Newton_,
                          const std::size_t max_Newton_iters_):
    Flux<Field>(EOS_phase1_, EOS_phase2_,
                lambda_, atol_Newton_, rtol_Newton_, max_Newton_iters_) {}

  // Implementation of a HLL flux for the system
  //
  template<class Field>
  FluxValue<typename HLLFlux<Field>::cfg>
  HLLFlux<Field>::compute_discrete_flux(const FluxValue<cfg>& qL,
                                        const FluxValue<cfg>& qR,
                                        std::size_t curr_d) {
    /*--- Pre-fetch some variables used multiple times in order to exploit possible vectorization ---*/
    const auto m1_L      = qL(M1_INDEX);
    const auto m2_L      = qL(M2_INDEX);
    const auto rho_L     = m1_L + m2_L;
    const auto inv_rho_L = static_cast<Number>(1.0)/rho_L;
    const auto alpha1_L  = qL(RHO_ALPHA1_INDEX)*inv_rho_L;

    const auto m1_R      = qR(M1_INDEX);
    const auto m2_R      = qR(M2_INDEX);
    const auto rho_R     = m1_R + m2_R;
    const auto inv_rho_R = static_cast<Number>(1.0)/rho_R;
    const auto alpha1_R  = qR(RHO_ALPHA1_INDEX)*inv_rho_R;

    /*--- Verify if left and right state are coherent ---*/
    #ifdef VERBOSE_FLUX
      if(m1_L < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative mass phase 1 left state: " + std::to_string(m1_L)));
      }
      if(m2_L < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative mass phase 2 left state: " + std::to_string(m2_L)));
      }
      if(alpha1_L < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative volume fraction phase 1 left state: " + std::to_string(alpha1_L)));
      }
      else if(alpha1_L > static_cast<Number>(1.0)) {
        throw std::runtime_error(std::string("Exceeding volume fraction phase 1 left state: " + std::to_string(alpha1_L)));
      }

      if(m1_R < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative mass phase 1 right state: " + std::to_string(m1_R)));
      }
      if(m2_R < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative mass phase 2 right state: " + std::to_string(m2_R)));
      }
      if(alpha1_R < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative volume fraction phase 1 right state: " + std::to_string(alpha1_R)));
      }
      else if(alpha1_R > static_cast<Number>(1.0)) {
        throw std::runtime_error(std::string("Exceeding volume fraction phase 1 right state: " + std::to_string(alpha1_R)));
      }
    #endif

    /*--- Left state ---*/
    const auto vel_d_L = qL(RHO_U_INDEX + curr_d)*inv_rho_L;

    const auto rho1_L  = m1_L/alpha1_L; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rho2_L  = m2_L/(static_cast<Number>(1.0) - alpha1_L); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto Y1_L    = m1_L*inv_rho_L;
    const auto c_L     = std::sqrt(Y1_L*
                                   this->EOS_phase1.c_value(rho1_L)*
                                   this->EOS_phase1.c_value(rho1_L) +
                                   (static_cast<Number>(1.0) - Y1_L)*
                                   this->EOS_phase2.c_value(rho2_L)*
                                   this->EOS_phase2.c_value(rho2_L));

    /*--- Right state ---*/
    const auto vel_d_R = qR(RHO_U_INDEX + curr_d)*inv_rho_R;

    const auto rho1_R  = m1_R/alpha1_R; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rho2_R  = m2_R/(static_cast<Number>(1.0) - alpha1_R); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto Y1_R    = m1_R*inv_rho_R;
    const auto c_R     = std::sqrt(Y1_R*
                                   this->EOS_phase1.c_value(rho1_R)*
                                   this->EOS_phase1.c_value(rho1_R) +
                                   (static_cast<Number>(1.0) - Y1_R)*
                                   this->EOS_phase2.c_value(rho2_R)*
                                   this->EOS_phase2.c_value(rho2_R));

    /*--- Compute speeds of wave propagation ---*/
    const auto s_L = std::min(vel_d_L - c_L, vel_d_R - c_R);
    const auto s_R = std::max(vel_d_L + c_L, vel_d_R + c_R);

    /*--- Compute the flux ---*/
    if(s_L >= static_cast<Number>(0.0)) {
      return this->evaluate_continuous_flux(qL, curr_d);
    }
    else if(s_L < static_cast<Number>(0.0) &&
            s_R > static_cast<Number>(0.0)) {
      return (s_R*this->evaluate_continuous_flux(qL, curr_d) -
              s_L*this->evaluate_continuous_flux(qR, curr_d) +
              s_L*s_R*(qR - qL))/(s_R - s_L);
    }
    else if(s_R <= static_cast<Number>(0.0)) {
      return this->evaluate_continuous_flux(qR, curr_d);
    }
  }

  // Implement the contribution of the discrete flux for all the dimensions.
  //
  template<class Field>
  decltype(make_flux_based_scheme(std::declval<FluxDefinition<typename HLLFlux<Field>::cfg>>()))
  HLLFlux<Field>::make_flux() {
    FluxDefinition<cfg> HLL_flux;

    /*--- Perform the loop over each dimension to compute the flux contribution ---*/
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
         {
           static constexpr int d = decltype(integral_constant_d)::value;

           // Compute now the "discrete" flux function, in this case a HLL flux
           HLL_flux[d].cons_flux_function = [&](FluxValue<cfg>& flux,
                                                const StencilData<cfg>& /*data*/,
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
                                                    perform_reconstruction<Field, cfg>(primLL, primL, primR, primRR,
                                                                                       primL_recon, primR_recon);

                                                    const FluxValue<cfg> qL = this->prim2cons(primL_recon);
                                                    const FluxValue<cfg> qR = this->prim2cons(primR_recon);
                                                  #else
                                                    // Extract the states
                                                    const FluxValue<cfg> qL = field[0];
                                                    const FluxValue<cfg> qR = field[1];
                                                  #endif

                                                  flux = compute_discrete_flux(qL, qR, d);
                                                };
        }
    );

    auto scheme = make_flux_based_scheme(HLL_flux);
    scheme.set_name(this->get_flux_name());

    return scheme;
  }

} // end of namespace
