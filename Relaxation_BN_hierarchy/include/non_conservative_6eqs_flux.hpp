// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#ifndef non_conservative_6eqs_flux_hpp
#define non_conservative_6eqs_flux_hpp

#include "flux_6eqs_base.hpp"

#define BR_ORLANDO
//#define BR_TUMOLO
//#define CENTERED

namespace samurai {
  using namespace EquationData;

  /**
    * Implementation of the non-conservative flux
    */
  template<class Field>
  class NonConservativeFlux: public Flux<Field> {
  public:
    NonConservativeFlux(const EOS<typename Field::value_type>& EOS_phase1,
                        const EOS<typename Field::value_type>& EOS_phase2); // Constructor which accepts in inputs the equations of state of the two phases

    auto make_flux(); // Compute the flux over all cells

  private:
    void compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                               const FluxValue<typename Flux<Field>::cfg>& qR,
                               const std::size_t curr_d,
                               FluxValue<typename Flux<Field>::cfg>& F_minus,
                               FluxValue<typename Flux<Field>::cfg>& F_plus); // Non-conservative flux
  };

  // Constructor derived from base class
  //
  template<class Field>
  NonConservativeFlux<Field>::NonConservativeFlux(const EOS<typename Field::value_type>& EOS_phase1,
                                                  const EOS<typename Field::value_type>& EOS_phase2):
    Flux<Field>(EOS_phase1, EOS_phase2) {}

  // Implementation of a non-conservative flux from left to right
  //
  template<class Field>
  void NonConservativeFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                         const FluxValue<typename Flux<Field>::cfg>& qR,
                                                         const std::size_t curr_d,
                                                         FluxValue<typename Flux<Field>::cfg>& F_minus,
                                                         FluxValue<typename Flux<Field>::cfg>& F_plus) {
    // Zero contribution from continuity and momentum equations
    F_minus(ALPHA1_RHO1_INDEX) = 0.0;
    F_plus(ALPHA1_RHO1_INDEX)  = 0.0;
    F_minus(ALPHA2_RHO2_INDEX) = 0.0;
    F_plus(ALPHA2_RHO2_INDEX)  = 0.0;
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      F_minus(RHO_U_INDEX + d) = 0.0;
      F_plus(RHO_U_INDEX + d) = 0.0;
    }

    // Compute velocity and mass fractions left state
    const auto rhoL = qL(ALPHA1_RHO1_INDEX) + qL(ALPHA2_RHO2_INDEX);
    const auto Y1L  = qL(ALPHA1_RHO1_INDEX)/rhoL;
    const auto Y2L  = 1.0 - Y1L;
    const auto velL = qL(RHO_U_INDEX + curr_d)/rhoL;

    // Pressure phase 1 left state
    const auto alpha1L = qL(ALPHA1_INDEX);
    const auto rho1L   = qL(ALPHA1_RHO1_INDEX)/alpha1L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1L           = qL(ALPHA1_RHO1_E1_INDEX)/qL(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e1L -= 0.5*(qL(RHO_U_INDEX + d)/rhoL)*(qL(RHO_U_INDEX + d)/rhoL);
    }
    const auto p1L = this->phase1.pres_value(rho1L, e1L);

    // Pressure phase 2 left state
    const auto rho2L = qL(ALPHA2_RHO2_INDEX)/(1.0 - alpha1L); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2L         = qL(ALPHA2_RHO2_E2_INDEX)/qL(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2L -= 0.5*(qL(RHO_U_INDEX + d)/rhoL)*(qL(RHO_U_INDEX + d)/rhoL);
    }
    const auto p2L = this->phase2.pres_value(rho2L, e2L);

    // Compute velocity and mass fractions right state
    const auto rhoR = qR(ALPHA1_RHO1_INDEX) + qR(ALPHA2_RHO2_INDEX);
    const auto Y1R  = qR(ALPHA1_RHO1_INDEX)/rhoR;
    const auto Y2R  = 1.0 - Y1R;
    const auto velR = qR(RHO_U_INDEX + curr_d)/rhoR;

    // Pressure phase 1 right state
    const auto alpha1R = qR(ALPHA1_INDEX);
    const auto rho1R   = qR(ALPHA1_RHO1_INDEX)/alpha1R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1R           = qR(ALPHA1_RHO1_E1_INDEX)/qR(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e1R -= 0.5*(qR(RHO_U_INDEX + d)/rhoR)*(qR(RHO_U_INDEX + d)/rhoR);
    }
    const auto p1R = this->phase1.pres_value(rho1R, e1R);

    // Pressure phase 2 right state
    const auto rho2R = qR(ALPHA2_RHO2_INDEX)/(1.0 - alpha1R); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2R         = qR(ALPHA2_RHO2_E2_INDEX)/qR(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2R -= 0.5*(qR(RHO_U_INDEX + d)/rhoR)*(qR(RHO_U_INDEX + d)/rhoR);
    }
    const auto p2R = this->phase2.pres_value(rho2R, e2R);

    // Build the non conservative flux (a lot of approximations to be checked here)
    #ifdef BR_ORLANDO
      #ifdef APPLY_NON_CONS_VOLUME_FRACTION
        F_minus(ALPHA1_INDEX) = (0.5*(velL*alpha1L + velR*alpha1R) -
                                 0.5*(velL + velR)*alpha1L);
        F_plus(ALPHA1_INDEX)  = (0.5*(velL*alpha1L + velR*alpha1R) -
                                 0.5*(velL + velR)*alpha1R);
      #else
        F_minus(ALPHA1_INDEX) = 0.0;
        F_plus(ALPHA1_INDEX)  = 0.0;
      #endif

      F_minus(ALPHA1_RHO1_E1_INDEX) = -(0.5*(velL*Y2L*alpha1L*p1L + velR*Y2R*alpha1R*p1R) -
                                        0.5*(velL*Y2L + velR*Y2R)*alpha1L*p1L)
                                      +(0.5*(velL*Y1L*(1.0 - alpha1L)*p2L + velR*Y1R*(1.0 - alpha1R)*p2R) -
                                        0.5*(velL*Y1L + velR*Y1R)*(1.0 - alpha1L)*p2L);
      F_plus(ALPHA1_RHO1_E1_INDEX)  = -(0.5*(velL*Y2L*alpha1L*p1L + velR*Y2R*alpha1R*p1R) -
                                        0.5*(velL*Y2L + velR*Y2R)*alpha1R*p1R)
                                      +(0.5*(velL*Y1L*(1.0 - alpha1L)*p2L + velR*Y1R*(1.0 - alpha1R)*p2R) -
                                        0.5*(velL*Y1L + velR*Y1R)*(1.0 - alpha1R)*p2R);
    #elifdef BR_TUMOLO
      #ifdef APPLY_NON_CONS_VOLUME_FRACTION
        F_minus(ALPHA1_INDEX) = (0.25*(velL + velR)*(alpha1L + alpha1R) -
                                 0.5*(velL + velR)*alpha1L);
        F_plus(ALPHA1_INDEX)  = (0.25*(velL + velR)*(alpha1L + alpha1R) -
                                 0.5*(velL + velR)*alpha1R);
      #else
        F_minus(ALPHA1_INDEX) = 0.0;
        F_plus(ALPHA1_INDEX)  = 0.0;
      #endif

      F_minus(ALPHA1_RHO1_E1_INDEX) = -(0.25*(velL*Y2L + velR*Y2R)*(alpha1L*p1L + alpha1R*p1R) -
                                        0.5*(velL*Y2L + velR*Y2R)*alpha1L*p1L)
                                      +(0.25*(velL*Y1L + velR*Y1R)*((1.0 - alpha1L)*p2L + (1.0 - alpha1R)*p2R) -
                                        0.5*(velL*Y1L + velR*Y1R)*(1.0 - alpha1L)*p2L);
      F_plus(ALPHA1_RHO1_E1_INDEX)  = -(0.25*(velL*Y2L + velR*Y2R)*(alpha1L*p1L + alpha1R*p1R) -
                                        0.5*(velL*Y2L + velR*Y2R)*alpha1R*p1R)
                                      +(0.25*(velL*Y1L + velR*Y1R)*((1.0 - alpha1L)*p2L + (1.0 - alpha1R)*p2R) -
                                        0.5*(velL*Y1L + velR*Y1R)*(1.0 - alpha1R)*p2R);
    #elifdef CENTERED
      #ifdef APPLY_NON_CONS_VOLUME_FRACTION
        F_minus(ALPHA1_INDEX) = velL*(0.5*(alpha1L + alpha1R));
        F_plus(ALPHA1_INDEX)  = velR*(0.5*(alpha1L + alpha1R));
      #else
        F_minus(ALPHA1_INDEX) = 0.0;
        F_plus(ALPHA1_INDEX)  = 0.0;
      #endif

      F_minus(ALPHA1_RHO1_E1_INDEX) = -velL*(Y2L*(0.5*(alpha1L*p1L + alpha1R*p1R)) -
                                             Y1L*(0.5*((1.0 - alpha1L)*p2L + (1.0 - alpha1R)*p2R)));
      F_plus(ALPHA1_RHO1_E1_INDEX) = -velR*(Y2R*(0.5*(alpha1L*p1L + alpha1R*p1R)) -
                                            Y1R*(0.5*((1.0 - alpha1L)*p2L + (1.0 - alpha1R)*p2R)));
    #else
      #ifdef APPLY_NON_CONS_VOLUME_FRACTION
        F_minus(ALPHA1_INDEX) = velL*(alpha1R - alpha1L);
        F_plus(ALPHA1_INDEX)  = 0.0;
      #else
        F_minus(ALPHA1_INDEX) = 0.0;
        F_plus(ALPHA1_INDEX)  = 0.0;
      #endif
      F_minus(ALPHA1_RHO1_E1_INDEX) = -velL*(Y2L*(alpha1R*p1R - alpha1L*p1L) -
                                             Y1L*((1.0 - alpha1R)*p2R - (1.0 -alpha1L)*p2L));
      F_plus(ALPHA1_RHO1_E1_INDEX)  = 0.0;
    #endif
    F_minus(ALPHA2_RHO2_E2_INDEX) = -F_minus(ALPHA1_RHO1_E1_INDEX);
    F_plus(ALPHA2_RHO2_E2_INDEX)  = -F_plus(ALPHA1_RHO1_E1_INDEX);
  }

  // Implement the contribution of the discrete flux for all the dimensions.
  //
  template<class Field>
  auto NonConservativeFlux<Field>::make_flux() {
    FluxDefinition<typename Flux<Field>::cfg> discrete_flux;

    // Perform the loop over each dimension to compute the flux contribution
    static_for<0, EquationData::dim>::apply(
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        // Compute now the "discrete" non-conservative flux function
        discrete_flux[d].flux_function = [&](auto& cells, const Field& field)
                                            {
                                              // Compute the stencil
                                              #ifdef ORDER_2
                                                const auto& left  = cells[1];
                                                const auto& right = cells[2];
                                              #else
                                                const auto& left  = cells[0];
                                                const auto& right = cells[1];
                                              #endif

                                              const FluxValue<typename Flux<Field>::cfg>& qL = field[left];
                                              const FluxValue<typename Flux<Field>::cfg>& qR = field[right];

                                              FluxValue<typename Flux<Field>::cfg> F_minus,
                                                                                   F_plus;

                                              compute_discrete_flux(qL, qR, d, F_minus, F_plus);

                                              samurai::FluxValuePair<typename Flux<Field>::cfg> flux;
                                              flux[0] = F_minus;
                                              flux[1] = -F_plus;

                                              return flux;
                                            };
      }
    );

    return make_flux_based_scheme(discrete_flux);
  }

} // end of namespace

#endif
