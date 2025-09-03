// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#ifndef non_conservative_6eqs_conservative_alpha_flux_hpp
#define non_conservative_6eqs_conservative_alpha_flux_hpp

#include "flux_6eqs_conservative_alpha_base.hpp"

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
    using Number = typename Flux<Field>::Number; /*--- Define the shortcut for the arithmetic type ---*/
    
    NonConservativeFlux(const EOS<Number>& EOS_phase1,
                        const EOS<Number>& EOS_phase2); /*--- Constructor which accepts in input
                                                              the equations of state of the two phases ---*/

    auto make_flux(); /*--- Compute the flux over all the faces and directions ---*/

  private:
    void compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                               const FluxValue<typename Flux<Field>::cfg>& qR,
                               const std::size_t curr_d,
                               FluxValue<typename Flux<Field>::cfg>& F_minus,
                               FluxValue<typename Flux<Field>::cfg>& F_plus); /*--- Non-conservative flux ---*/
  };

  // Constructor derived from base class
  //
  template<class Field>
  NonConservativeFlux<Field>::NonConservativeFlux(const EOS<Number>& EOS_phase1_,
                                                  const EOS<Number>& EOS_phase2_):
    Flux<Field>(EOS_phase1_, EOS_phase2_) {}

  // Implementation of a non-conservative flux
  //
  template<class Field>
  void NonConservativeFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                         const FluxValue<typename Flux<Field>::cfg>& qR,
                                                         const std::size_t curr_d,
                                                         FluxValue<typename Flux<Field>::cfg>& F_minus,
                                                         FluxValue<typename Flux<Field>::cfg>& F_plus) {
    /*--- Zero contribution from volume fraction, continuity, and momentum equations ---*/
    F_minus(ALPHA1_INDEX)      = static_cast<Number>(0.0);
    F_plus(ALPHA1_INDEX)       = static_cast<Number>(0.0);
    F_minus(ALPHA1_RHO1_INDEX) = static_cast<Number>(0.0);
    F_plus(ALPHA1_RHO1_INDEX)  = static_cast<Number>(0.0);
    F_minus(ALPHA2_RHO2_INDEX) = static_cast<Number>(0.0);
    F_plus(ALPHA2_RHO2_INDEX)  = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      F_minus(RHO_U_INDEX + d) = static_cast<Number>(0.0);
      F_plus(RHO_U_INDEX + d)  = static_cast<Number>(0.0);
    }

    /*--- Left state ---*/
    // Pre-fetch variables that will be used several times so as to exploit possible vectorization
    // (as well as to enhance readability)
    const auto m1_L   = qL(ALPHA1_RHO1_INDEX);
    const auto m2_L   = qL(ALPHA2_RHO2_INDEX);
    const auto m1E1_L = qL(ALPHA1_RHO1_E1_INDEX);
    const auto m2E2_L = qL(ALPHA2_RHO2_E2_INDEX);

    // Compute velocity and mass fractions
    const auto rho_L     = m1_L + m2_L;
    const auto inv_rho_L = static_cast<Number>(1.0)/rho_L;
    const auto Y1_L      = m1_L*inv_rho_L;
    const auto Y2_L      = static_cast<Number>(1.0) - Y1_L;
    const auto vel_d_L   = qL(RHO_U_INDEX + curr_d)*inv_rho_L;

    // Pressure phase 1
    const auto alpha1_L = qL(RHO_ALPHA1_INDEX)*inv_rho_L;
    const auto rho1_L   = m1_L/alpha1_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto norm2_vel_L    = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      norm2_vel_L += (qL(RHO_U_INDEX + d)*inv_rho_L)*
                     (qL(RHO_U_INDEX + d)*inv_rho_L);
    }
    const auto e1_L = m1E1_L/m1_L /*--- TODO: Add treatment for vanishing volume fraction ---*/
                    - static_cast<Number>(0.5)*norm2_vel_L;
    const auto p1_L = this->EOS_phase1.pres_value(rho1_L, e1_L);

    // Pressure phase 2
    const auto rho2_L = m2_L/(static_cast<Number>(1.0) - alpha1_L); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto e2_L   = m2E2_L/m2_L /*--- TODO: Add treatment for vanishing volume fraction ---*/
                      - static_cast<Number>(0.5)*norm2_vel_L;
    const auto p2_L   = this->EOS_phase2.pres_value(rho2_L, e2_L);

    /*--- Right state ---*/
    // Pre-fetch variables that will be used several times so as to exploit possible vectorization
    // (as well as to enhance readability)
    const auto m1_R   = qR(ALPHA1_RHO1_INDEX);
    const auto m2_R   = qR(ALPHA2_RHO2_INDEX);
    const auto m1E1_R = qR(ALPHA1_RHO1_E1_INDEX);
    const auto m2E2_R = qR(ALPHA2_RHO2_E2_INDEX);

    // Compute velocity and mass fractions
    const auto rho_R     = m1_R + m2_R;
    const auto inv_rho_R = static_cast<Number>(1.0)/rho_R;
    const auto Y1_R      = m1_R*inv_rho_R;
    const auto Y2_R      = static_cast<Number>(1.0) - Y1_R;
    const auto vel_d_R   = qR(RHO_U_INDEX + curr_d)*inv_rho_R;

    // Pressure phase 1
    const auto alpha1_R = qR(RHO_ALPHA1_INDEX)*inv_rho_R;
    const auto rho1_R   = m1_R/alpha1_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto norm2_vel_R    = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      norm2_vel_R += (qR(RHO_U_INDEX + d)*inv_rho_R)*
                     (qR(RHO_U_INDEX + d)*inv_rho_R);
    }
    const auto e1_R = m1E1_R/m1_R /*--- TODO: Add treatment for vanishing volume fraction ---*/
                    - static_cast<Number>(0.5)*norm2_vel_R;
    const auto p1_R = this->EOS_phase1.pres_value(rho1_R, e1_R);

    /*--- Pressure phase 2 right state ---*/
    const auto rho2_R = m2_R/(static_cast<Number>(1.0) - alpha1_R); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto e2_R   = m2E2_R/m2_R /*--- TODO: Add treatment for vanishing volume fraction ---*/
                      - static_cast<Number>(0.5)*norm2_vel_R;
    const auto p2_R   = this->EOS_phase2.pres_value(rho2_R, e2_R);

    /*--- Build the non conservative flux (a lot of approximations to be checked here) ---*/
    #ifdef BR_ORLANDO
      F_minus(ALPHA1_RHO1_E1_INDEX) = -(static_cast<Number>(0.5)*
                                        (vel_d_L*Y2_L*alpha1_L*p1_L + vel_d_R*Y2_R*alpha1_R*p1_R) -
                                        static_cast<Number>(0.5)*
                                        (vel_d_L*Y2_L + vel_d_R*Y2_R)*alpha1_L*p1_L)
                                      +(static_cast<Number>(0.5)*
                                        (vel_d_L*Y1_L*(static_cast<Number>(1.0) - alpha1_L)*p2_L +
                                         vel_d_R*Y1_R*(static_cast<Number>(1.0) - alpha1_R)*p2_R) -
                                        static_cast<Number>(0.5)*
                                        (vel_d_L*Y1_L + vel_d_R*Y1_R)*(static_cast<Number>(1.0) - alpha1_L)*p2_L);
      F_plus(ALPHA1_RHO1_E1_INDEX)  = -(static_cast<Number>(0.5)*
                                        (vel_d_L*Y2_L*alpha1_L*p1_L + vel_d_R*Y2_R*alpha1_R*p1_R) -
                                        static_cast<Number>(0.5)*
                                        (vel_d_L*Y2_L + vel_d_R*Y2_R)*alpha1_R*p1_R)
                                      +(static_cast<Number>(0.5)*
                                        (vel_d_L*Y1_L*(static_cast<Number>(1.0) - alpha1_L)*p2_L +
                                         vel_d_R*Y1_R*(static_cast<Number>(1.0) - alpha1_R)*p2_R) -
                                        static_cast<Number>(0.5)*
                                        (vel_d_L*Y1_L + vel_d_R*Y1_R)*(static_cast<Number>(1.0) - alpha1_R)*p2_R);
    #elifdef BR_TUMOLO
      F_minus(ALPHA1_RHO1_E1_INDEX) = -(static_cast<Number>(0.25)*
                                        (vel_d_L*Y2_L + vel_d_R*Y2_R)*(alpha1_L*p1_L + alpha1_R*p1_R) -
                                        static_cast<Number>(0.5)*
                                        (vel_d_L*Y2_L + vel_d_R*Y2_R)*alpha1_L*p1_L)
                                      +(static_cast<Number>(0.25)*
                                        (vel_d_L*Y1_L + vel_d_R*Y1_R)*
                                        ((static_cast<Number>(1.0) - alpha1_L)*p2_L +
                                         (static_cast<Number>(1.0) - alpha1_R)*p2_R) -
                                        static_cast<Number>(0.5)*
                                        (vel_d_L*Y1_L + vel_d_R*Y1_R)*(static_cast<Number>(1.0) - alpha1_L)*p2_L);
      F_plus(ALPHA1_RHO1_E1_INDEX)  = -(static_cast<Number>(0.25)*
                                        (vel_d_L*Y2_L + vel_d_R*Y2_R)*(alpha1_L*p1_L + alpha1_R*p1_R) -
                                        static_cast<Number>(0.5)*
                                        (vel_d_L*Y2_L + vel_d_R*Y2_R)*alpha1_R*p1_R)
                                      +(static_cast<Number>(0.25)*
                                        (vel_d_L*Y1_L + vel_d_R*Y1_R)*
                                        ((static_cast<Number>(1.0) - alpha1_L)*p2_L +
                                         (static_cast<Number>(1.0) - alpha1_R)*p2_R) -
                                        static_cast<Number>(0.5)*
                                        (vel_d_L*Y1_L + vel_d_R*Y1_R)*(static_cast<Number>(1.0) - alpha1_R)*p2_R);
    #elifdef CENTERED
      F_minus(ALPHA1_RHO1_E1_INDEX) = -vel_d_L*(Y2_L*(static_cast<Number>(0.5)*
                                                      (alpha1_L*p1_L + alpha1_R*p1_R)) -
                                                Y1_L*(static_cast<Number>(0.5)*
                                                      ((static_cast<Number>(1.0) - alpha1_L)*p2_L +
                                                       (static_cast<Number>(1.0) - alpha1_R)*p2_R)));
      F_plus(ALPHA1_RHO1_E1_INDEX) = -vel_d_R*(Y2_R*(static_cast<Number>(0.5)*
                                                     (alpha1_L*p1_L + alpha1_R*p1_R)) -
                                               Y1_R*(static_cast<Number>(0.5)*
                                                     ((static_cast<Number>(1.0) - alpha1_L)*p2_L +
                                                      (static_cast<Number>(1.0) - alpha1_R)*p2_R)));
    #else
      F_minus(ALPHA1_RHO1_E1_INDEX) = -vel_d_L*(Y2_L*(alpha1_R*p1_R - alpha1_L*p1_L) -
                                                Y1_L*((static_cast<Number>(1.0) - alpha1_R)*p2_R -
                                                      (static_cast<Number>(1.0) - alpha1_L)*p2_L));
      F_plus(ALPHA1_RHO1_E1_INDEX)  = static_cast<Number>(0.0);
    #endif
    F_minus(ALPHA2_RHO2_E2_INDEX) = -F_minus(ALPHA1_RHO1_E1_INDEX);
    F_plus(ALPHA2_RHO2_E2_INDEX)  = -F_plus(ALPHA1_RHO1_E1_INDEX);
  }

  // Implement the contribution of the discrete flux for all the dimensions.
  //
  template<class Field>
  auto NonConservativeFlux<Field>::make_flux() {
    /*--- Perform the loop over each dimension to compute the flux contribution ---*/
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
         {
           static constexpr int d = decltype(integral_constant_d)::value;

           // Compute now the "discrete" non-conservative flux function
           non_conservative_flux[d].flux_function = [&](samurai::FluxValuePair<typename Flux<Field>::cfg>& flux,
                                                        const StencilData<typename Flux<Field>::cfg>& /*data*/,
                                                        const StencilValues<typename Flux<Field>::cfg> field)
                                                        {
                                                          // Extract the states
                                                          #ifdef ORDER_2
                                                            const FluxValue<typename Flux<Field>::cfg>& qL = field[1];
                                                            const FluxValue<typename Flux<Field>::cfg>& qR = field[2];
                                                          #else
                                                            const FluxValue<typename Flux<Field>::cfg>& qL = field[0];
                                                            const FluxValue<typename Flux<Field>::cfg>& qR = field[1];
                                                          #endif

                                                          FluxValue<typename Flux<Field>::cfg> F_minus,
                                                                                               F_plus;

                                                          compute_discrete_flux(qL, qR, d, F_minus, F_plus);

                                                          flux[0] = F_minus;
                                                          flux[1] = -F_plus;
                                                        };
        }
    );

    auto scheme = make_flux_based_scheme(non_conservative_flux);
    scheme.set_name("Non conservative");

    return scheme;
  }

} // end of namespace

#endif
