// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#ifndef Rusanov_6eqs_flux_hpp
#define Rusanov_6eqs_flux_hpp

#include "flux_6eqs_base.hpp"

#define APPLY_NON_CONS_VOLUME_FRACTION

namespace samurai {
  using namespace EquationData;

  /**
    * Implementation of a Rusanov flux
    */
  template<class Field>
  class RusanovFlux: public Flux<Field> {
  public:
    RusanovFlux(const EOS<typename Field::value_type>& EOS_phase1_,
                const EOS<typename Field::value_type>& EOS_phase2_); /*--- Constructor which accepts in input
                                                                           the equations of state of the two phases ---*/

    auto make_flux(); /*--- Compute the flux over all the faces and directions ---*/

  private:
    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                               const FluxValue<typename Flux<Field>::cfg>& qR,
                                                               const std::size_t curr_d); /*--- Rusanov flux along direction curr_d ---*/
  };

  // Constructor derived from base class
  //
  template<class Field>
  RusanovFlux<Field>::RusanovFlux(const EOS<typename Field::value_type>& EOS_phase1_,
                                  const EOS<typename Field::value_type>& EOS_phase2_):
    Flux<Field>(EOS_phase1_, EOS_phase2_) {}

  // Implementation of a Rusanov flux
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> RusanovFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                                                 const FluxValue<typename Flux<Field>::cfg>& qR,
                                                                                 std::size_t curr_d) {
    /*--- Left state ---*/
    // Pre-fetch variables that will be used several times so as to exploit possible vectorization
    // (as well as to enhance readability)
    const auto alpha1_L = qL(ALPHA1_INDEX);
    const auto m1_L     = qL(ALPHA1_RHO1_INDEX);
    const auto m2_L     = qL(ALPHA2_RHO2_INDEX);
    const auto m1E1_L   = qL(ALPHA1_RHO1_E1_INDEX);
    const auto m2E2_L   = qL(ALPHA2_RHO2_E2_INDEX);

    // Save mixture density and velocity current direction left state
    const auto rho_L     = m1_L + m2_L;
    const auto inv_rho_L = static_cast<typename Field::value_type>(1.0)/rho_L;
    const auto vel_L_d   = qL(RHO_U_INDEX + curr_d)*inv_rho_L;

    // Phase 1
    const auto rho1_L = m1_L/alpha1_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto norm2_vel_L  = static_cast<typename Field::value_type>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      norm2_vel_L += (qL(RHO_U_INDEX + d)*inv_rho_L)*
                     (qL(RHO_U_INDEX + d)*inv_rho_L);
    }
    const auto e1_L = m1E1_L/m1_L /*--- TODO: Add treatment for vanishing volume fraction ---*/
                    - static_cast<typename Field::value_type>(0.5)*norm2_vel_L;
    const auto p1_L = this->EOS_phase1.pres_value(rho1_L, e1_L);
    const auto c1_L = this->EOS_phase1.c_value(rho1_L, p1_L);

    // Phase 2
    const auto rho2_L = m2_L/(static_cast<typename Field::value_type>(1.0) - alpha1_L); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto e2_L   = m2E2_L/m2_L /*--- TODO: Add treatment for vanishing volume fraction ---*/
                      - static_cast<typename Field::value_type>(0.5)*norm2_vel_L;
    const auto p2_L   = this->EOS_phase2.pres_value(rho2_L, e2_L);
    const auto c2_L   = this->EOS_phase2.c_value(rho2_L, p2_L);

    // Compute frozen speed of sound left state
    const auto Y1_L = m1_L*inv_rho_L;
    const auto c_L  = std::sqrt(Y1_L*c1_L*c1_L +
                                (static_cast<typename Field::value_type>(1.0) - Y1_L)*c2_L*c2_L);

    /*--- Right state ---*/
    // Pre-fetch variables that will be used several times so as to exploit possible vectorization
    // (as well as to enhance readability)
    const auto alpha1_R = qR(ALPHA1_INDEX);
    const auto m1_R     = qR(ALPHA1_RHO1_INDEX);
    const auto m2_R     = qR(ALPHA2_RHO2_INDEX);
    const auto m1E1_R   = qR(ALPHA1_RHO1_E1_INDEX);
    const auto m2E2_R   = qR(ALPHA2_RHO2_E2_INDEX);

    // Save mixture density and velocity current direction left state
    const auto rho_R     = m1_R + m2_R;
    const auto inv_rho_R = static_cast<typename Field::value_type>(1.0)/rho_R;
    const auto vel_R_d   = qR(RHO_U_INDEX + curr_d)*inv_rho_R;

    // Phase 1
    const auto rho1_R = m1_R/alpha1_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto norm2_vel_R  = static_cast<typename Field::value_type>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      norm2_vel_R += (qR(RHO_U_INDEX + d)*inv_rho_R)*
                     (qR(RHO_U_INDEX + d)*inv_rho_R);
    }
    const auto e1_R = m1E1_R/m1_R /*--- TODO: Add treatment for vanishing volume fraction ---*/
                    - static_cast<typename Field::value_type>(0.5)*norm2_vel_R;
    const auto p1_R = this->EOS_phase1.pres_value(rho1_R, e1_R);
    const auto c1_R = this->EOS_phase1.c_value(rho1_R, p1_R);

    // Phase 2
    const auto rho2_R = m2_R/(static_cast<typename Field::value_type>(1.0) - alpha1_R); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto e2_R   = m2E2_R/m2_R /*--- TODO: Add treatment for vanishing volume fraction ---*/
                      - static_cast<typename Field::value_type>(0.5)*norm2_vel_R;
    const auto p2_R   = this->EOS_phase2.pres_value(rho2_R, e2_R);
    const auto c2_R   = this->EOS_phase2.c_value(rho2_R, p2_R);

    // Compute frozen speed of sound right state
    const auto Y1_R = m1_R*inv_rho_R;
    const auto c_R  = std::sqrt(Y1_R*c1_R*c1_R +
                                (static_cast<typename Field::value_type>(1.0) - Y1_R)*c2_R*c2_R);

    /*--- Compute the numerical flux ---*/
    const auto lambda = std::max(std::abs(vel_L_d) + c_L,
                                 std::abs(vel_R_d) + c_R);

    return static_cast<typename Field::value_type>(0.5)*
           (this->evaluate_continuous_flux(qL, curr_d) +
            this->evaluate_continuous_flux(qR, curr_d)) - // centered contribution
           static_cast<typename Field::value_type>(0.5)*lambda*(qR - qL); // upwinding contribution
  }

  // Implement the contribution of the discrete flux for all the dimensions.
  //
  template<class Field>
  auto RusanovFlux<Field>::make_flux() {
    FluxDefinition<typename Flux<Field>::cfg> Rusanov_f;

    /*--- Perform the loop over each dimension to compute the flux contribution ---*/
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
         {
           static constexpr int d = decltype(integral_constant_d)::value;

           // Compute now the "discrete" flux function, in this case a Rusanov flux
           Rusanov_f[d].cons_flux_function = [&](samurai::FluxValue<typename Flux<Field>::cfg>& flux,
                                                 const StencilData<typename Flux<Field>::cfg>& /*data*/,
                                                 const StencilValues<typename Flux<Field>::cfg> field)
                                                 {
                                                   #ifdef ORDER_2
                                                     // MUSCL reconstruction
                                                     const FluxValue<typename Flux<Field>::cfg> primLL = this->cons2prim(field[0]);
                                                     const FluxValue<typename Flux<Field>::cfg> primL  = this->cons2prim(field[1]);
                                                     const FluxValue<typename Flux<Field>::cfg> primR  = this->cons2prim(field[2]);
                                                     const FluxValue<typename Flux<Field>::cfg> primRR = this->cons2prim(field[3]);

                                                     FluxValue<typename Flux<Field>::cfg> primL_recon,
                                                                                          primR_recon;
                                                     this->perform_reconstruction(primLL, primL, primR, primRR,
                                                                                  primL_recon, primR_recon);

                                                     const FluxValue<typename Flux<Field>::cfg> qL = this->prim2cons(primL_recon);
                                                     const FluxValue<typename Flux<Field>::cfg> qR = this->prim2cons(primR_recon);
                                                   #else
                                                     // Extract the states
                                                     const FluxValue<typename Flux<Field>::cfg> qL = field[0];
                                                     const FluxValue<typename Flux<Field>::cfg> qR = field[1];
                                                   #endif

                                                   flux = compute_discrete_flux(qL, qR, d);
                                                 };
        }
    );

    auto scheme = make_flux_based_scheme(Rusanov_f);
    scheme.set_name("Rusanov");

    return scheme;
  }

} // end of namespace

#endif
