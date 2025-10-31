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
    using cfg = Flux<Field>::cfg;

    using Number = Flux<Field>::Number; /*--- Define the shortcut for the arithmetic type ---*/

    RusanovFlux(const EOS<Number>& EOS_); /*--- Constructor which accepts in input the equation of state ---*/

    virtual decltype(make_flux_based_scheme(std::declval<FluxDefinition<cfg>>())) make_flux() override;
    /*--- Compute the flux over all the faces and directions ---*/

  private:
    FluxValue<cfg> compute_discrete_flux(const FluxValue<cfg>& qL,
                                         const FluxValue<cfg>& qR,
                                         const std::size_t curr_d); /*--- Rusanov flux along direction curr_d ---*/
  };

  // Constructor derived from base class
  //
  template<class Field>
  RusanovFlux<Field>::RusanovFlux(const EOS<Number>& EOS_):
    Flux<Field>(EOS_) {}

  // Implementation of a Rusanov flux
  //
  template<class Field>
  FluxValue<typename RusanovFlux<Field>::cfg>
  RusanovFlux<Field>::compute_discrete_flux(const FluxValue<cfg>& qL,
                                            const FluxValue<cfg>& qR,
                                            const std::size_t curr_d) {
    /*--- Left state ---*/
    // Pre-fetch density that will used several times
    const auto rhoL     = qL(RHO_INDEX);
    const auto inv_rhoL = static_cast<Number>(1.0)/rhoL;

    // Compute auxiliary primitive variables
    const auto velL_d = qL(RHOU_INDEX + curr_d)*inv_rhoL;
    auto eL = qL(RHOE_INDEX)*inv_rhoL;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      eL -= static_cast<Number>(0.5)*
            (qL(RHOU_INDEX + d)*inv_rhoL)*(qL(RHOU_INDEX + d)*inv_rhoL);
    }
    const auto pL = this->Euler_EOS.pres_value(rhoL, eL);
    const auto cL = this->Euler_EOS.c_value(rhoL, pL);

    #ifdef VERBOSE_FLUX
      if(rhoL < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative density left state: " + std::to_string(rhoL)));
      }
      if(std::isnan(cL)) {
        throw std::runtime_error(std::string("NaN speed of sound left state"));
      }
    #endif

    /*--- Right state ---*/
    // Pre-fetch density that will used several times
    const auto rhoR     = qR(RHO_INDEX);
    const auto inv_rhoR = static_cast<Number>(1.0)/rhoR;

    // Compute auxiliary primitive variables
    auto velR_d = qR(RHOU_INDEX + curr_d)*inv_rhoR;
    auto eR = qR(RHOE_INDEX)*inv_rhoR;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      eR -= static_cast<Number>(0.5)*
            (qR(RHOU_INDEX + d)*inv_rhoR)*(qR(RHOU_INDEX + d)*inv_rhoR);
    }
    const auto pR = this->Euler_EOS.pres_value(rhoR, eR);
    const auto cR = this->Euler_EOS.c_value(rhoR, pR);

    #ifdef VERBOSE_FLUX
      if(rhoR < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative density right state: " + std::to_string(rhoR)));
      }
      if(std::isnan(cR)) {
        throw std::runtime_error(std::string("NaN speed of sound right state"));
      }
    #endif

    /*--- Compute Rusanov flux ---*/
    const auto lambda = std::max(std::abs(velL_d) + cL,
                                 std::abs(velR_d) + cR);

    return static_cast<Number>(0.5)*
           (this->evaluate_continuous_flux(qL, curr_d) +
            this->evaluate_continuous_flux(qR, curr_d)) - // centered contribution
           static_cast<Number>(0.5)*lambda*(qR - qL); // upwinding contribution
  }

  // Implement the contribution of the discrete flux for all the dimensions.
  //
  template<class Field>
  decltype(make_flux_based_scheme(std::declval<FluxDefinition<typename RusanovFlux<Field>::cfg>>()))
  RusanovFlux<Field>::make_flux() {
    FluxDefinition<cfg> Rusanov_flux;

    /*--- Perform the loop over each dimension to compute the flux contribution ---*/
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
         {
           static constexpr int d = decltype(integral_constant_d)::value;

           // Compute now the "discrete" flux function, in this case a HLL flux
           Rusanov_flux[d].cons_flux_function = [&](FluxValue<cfg>& flux,
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
                                                        perform_reconstruction<Field>(primLL, primL, primR, primRR,
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

    auto scheme = make_flux_based_scheme(Rusanov_flux);
    scheme.set_name(this->get_flux_name());

    return scheme;
  }

} // end of namespace
