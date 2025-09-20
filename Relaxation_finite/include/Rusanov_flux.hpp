// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#pragma once

#include "flux_base.hpp"

namespace samurai {
  using namespace EquationData;

  /**
    * Implementation of a Rusanov flux
    */
  template<class Field>
  class RusanovFlux: public Flux<Field> {
  public:
    using Indices = Flux<Field>::Indices; /*--- Shortcut for the indices storage ---*/
    using Number  = Flux<Field>::Number;  /*--- Shortcut for the arithmetic type ---*/
    using cfg     = Flux<Field>::cfg;     /*--- Shortcut to specify the type of configuration
                                                for the flux (nonlinear in this case) ---*/

    RusanovFlux(const EOS<Number>& EOS_phase1_,
                const EOS<Number>& EOS_phase2_); /*--- Constructor which accepts in input
                                                       the equations of state of the two phases ---*/

    auto make_flux(); /*--- Compute the flux over all the faces and directions ---*/

  private:
    FluxValue<cfg> compute_discrete_flux(const FluxValue<cfg>& qL,
                                         const FluxValue<cfg>& qR,
                                         const std::size_t curr_d); /*--- Rusanov flux along direction curr_d ---*/
  };

  // Constructor derived from base class
  //
  template<class Field>
  RusanovFlux<Field>::RusanovFlux(const EOS<Number>& EOS_phase1_,
                                  const EOS<Number>& EOS_phase2_):
    Flux<Field>(EOS_phase1_, EOS_phase2_) {}

  // Implementation of a Rusanov flux
  //
  template<class Field>
  FluxValue<typename RusanovFlux<Field>::cfg>
  RusanovFlux<Field>::compute_discrete_flux(const FluxValue<cfg>& qL,
                                            const FluxValue<cfg>& qR,
                                            std::size_t curr_d) {
    /*--- Left state ---*/
    // Pre-fetch variables that will be used several times so as to exploit possible vectorization
    // (as well as to enhance readability)
    const auto alpha1_L = qL(Indices::ALPHA1_INDEX);
    const auto m1_L     = qL(Indices::ALPHA1_RHO1_INDEX);
    const auto m1E1_L   = qL(Indices::ALPHA1_RHO1_E1_INDEX);
    const auto m2_L     = qL(Indices::ALPHA2_RHO2_INDEX);
    const auto m2E2_L   = qL(Indices::ALPHA2_RHO2_E2_INDEX);

    // Phase 1
    const auto inv_m1_L = static_cast<Number>(1.0)/m1_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto vel1_L_d = qL(Indices::ALPHA1_RHO1_U1_INDEX + curr_d)*inv_m1_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho1_L   = m1_L/alpha1_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1_L           = m1E1_L*inv_m1_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < Field::dim; ++d) {
      e1_L -= static_cast<Number>(0.5)*
              ((qL(Indices::ALPHA1_RHO1_U1_INDEX + d)*inv_m1_L)*
               (qL(Indices::ALPHA1_RHO1_U1_INDEX + d)*inv_m1_L)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto p1_L = this->EOS_phase1.pres_value_Rhoe(rho1_L, e1_L);
    const auto c1_L = this->EOS_phase1.c_value_RhoP(rho1_L, p1_L);

    // Phase 2
    const auto inv_m2_L = static_cast<Number>(1.0)/m2_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto vel2_L_d = qL(Indices::ALPHA2_RHO2_U2_INDEX + curr_d)*inv_m2_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho2_L   = m2_L/(static_cast<Number>(1.0) - alpha1_L); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2_L           = m2E2_L*inv_m2_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < Field::dim; ++d) {
      e2_L -= static_cast<Number>(0.5)*
              ((qL(Indices::ALPHA2_RHO2_U2_INDEX + d)*inv_m2_L)*
               (qL(Indices::ALPHA2_RHO2_U2_INDEX + d)*inv_m2_L)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto p2_L = this->EOS_phase2.pres_value_Rhoe(rho2_L, e2_L);
    const auto c2_L = this->EOS_phase2.c_value_RhoP(rho2_L, p2_L);

    /*--- Right state ---*/
    // Pre-fetch variables that will be used several times so as to exploit possible vectorization
    // (as well as to enhance readability)
    const auto alpha1_R = qR(Indices::ALPHA1_INDEX);
    const auto m1_R     = qR(Indices::ALPHA1_RHO1_INDEX);
    const auto m1E1_R   = qR(Indices::ALPHA1_RHO1_E1_INDEX);
    const auto m2_R     = qR(Indices::ALPHA2_RHO2_INDEX);
    const auto m2E2_R   = qR(Indices::ALPHA2_RHO2_E2_INDEX);

    // Phase 1
    const auto inv_m1_R = static_cast<Number>(1.0)/m1_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto vel1_R_d = qR(Indices::ALPHA1_RHO1_U1_INDEX + curr_d)*inv_m1_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho1_R   = m1_R/alpha1_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1_R           = m1E1_R*inv_m1_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < Field::dim; ++d) {
      e1_R -= static_cast<Number>(0.5)*
              ((qR(Indices::ALPHA1_RHO1_U1_INDEX + d)*inv_m1_R)*
               (qR(Indices::ALPHA1_RHO1_U1_INDEX + d)*inv_m1_R)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto p1_R = this->EOS_phase1.pres_value_Rhoe(rho1_R, e1_R);
    const auto c1_R = this->EOS_phase1.c_value_RhoP(rho1_R, p1_R);

    // Phase 2
    const auto inv_m2_R = static_cast<Number>(1.0)/m2_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto vel2_R_d = qR(Indices::ALPHA2_RHO2_U2_INDEX + curr_d)*inv_m2_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho2_R   = m2_R/(static_cast<Number>(1.0) - alpha1_R); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2_R           = m2E2_R*inv_m2_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < Field::dim; ++d) {
      e2_R -= static_cast<Number>(0.5)*
              ((qR(Indices::ALPHA2_RHO2_U2_INDEX + d)*inv_m2_R)*
               (qR(Indices::ALPHA2_RHO2_U2_INDEX + d)*inv_m2_R)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto p2_R = this->EOS_phase2.pres_value_Rhoe(rho2_R, e2_R);
    const auto c2_R = this->EOS_phase2.c_value_RhoP(rho2_R, p2_R);

    /*--- Compute the flux ---*/
    const auto lambda = std::max(std::max(std::abs(vel1_L_d) + c1_L, std::abs(vel1_R_d) + c1_R),
                                 std::max(std::abs(vel2_L_d) + c2_L, std::abs(vel2_R_d) + c2_R));

    return static_cast<Number>(0.5)*
           (this->evaluate_continuous_flux(qL, curr_d) +
            this->evaluate_continuous_flux(qR, curr_d)) - // centered contribution
           static_cast<Number>(0.5)*lambda*(qR - qL); // upwinding contribution
  }

  // Implement the contribution of the discrete flux for all the dimensions.
  //
  template<class Field>
  auto RusanovFlux<Field>::make_flux() {
    FluxDefinition<cfg> discrete_flux;

    /*--- Perform the loop over each dimension to compute the flux contribution ---*/
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
         {
           static constexpr int d = decltype(integral_constant_d)::value;

           // Compute now the "discrete" flux function
           discrete_flux[d].cons_flux_function = [&](FluxValue<cfg>& flux,
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
                                                         this->perform_reconstruction(primLL, primL, primR, primRR,
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

    auto scheme = make_flux_based_scheme(discrete_flux);
    scheme.set_name("Rusanov");

    return scheme;
  }

} // end of namespace
