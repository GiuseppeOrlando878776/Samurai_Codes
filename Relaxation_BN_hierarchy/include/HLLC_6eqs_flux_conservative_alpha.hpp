// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#pragma once

#include "flux_6eqs_conservative_alpha_base.hpp"

namespace samurai {
  using namespace EquationData;

  /**
    * Implementation of a HLLC flux (wave-propagation method)
    */
  template<class Field>
  class HLLCFlux: public Flux<Field> {
  public:
    using Number = typename Flux<Field>::Number; /*--- Define the shortcut for the arithmetic type ---*/

    HLLCFlux(const EOS<Number>& EOS_phase1,
             const EOS<Number>& EOS_phase2); /*--- Constructor which accepts in input
                                                   the equations of state of the two phases ---*/

    auto make_flux(); /*--- Compute the flux over all the faces and directions ---*/

  private:
    auto compute_middle_state(const FluxValue<typename Flux<Field>::cfg>& q,
                              const Number S,
                              const Number S_star,
                              const std::size_t curr_d) const; /*--- Compute the middle state ---*/

    void compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                               const FluxValue<typename Flux<Field>::cfg>& qR,
                               const std::size_t curr_d,
                               FluxValue<typename Flux<Field>::cfg>& H_minus,
                               FluxValue<typename Flux<Field>::cfg>& H_plus); /*--- Compute the flux in a 'non-conservative' fashion
                                                                                    (wave propagation formalism) ---*/
  };

  // Constructor derived from base class
  //
  template<class Field>
  HLLCFlux<Field>::HLLCFlux(const EOS<Number>& EOS_phase1,
                            const EOS<Number>& EOS_phase2):
    Flux<Field>(EOS_phase1, EOS_phase2) {}

  // Implement the auxliary routine that computes the middle state
  //
  template<class Field>
  auto HLLCFlux<Field>::compute_middle_state(const FluxValue<typename Flux<Field>::cfg>& q,
                                             const Number S,
                                             const Number S_star,
                                             const std::size_t curr_d) const {
    /*--- Pre-fetch variables that will be used several times so as to exploit possible vectorization
          (as well as enhance readability) ---*/
    const auto m1   = q(ALPHA1_RHO1_INDEX);
    const auto m2   = q(ALPHA2_RHO2_INDEX);
    const auto m1E1 = q(ALPHA1_RHO1_E1_INDEX);
    const auto m2E2 = q(ALPHA2_RHO2_E2_INDEX);

    /*--- Save mixture density and velocity current direction ---*/
    const auto rho     = m1 + m2;
    const auto inv_rho = static_cast<Number>(1.0)/rho;
    const auto vel_d   = q(RHO_U_INDEX + curr_d)*inv_rho;

    /*--- Phase 1 ---*/
    const auto alpha1 = q(RHO_ALPHA1_INDEX)*inv_rho;
    const auto rho1   = m1/alpha1; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto norm2_vel    = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      norm2_vel += (q(RHO_U_INDEX + d)*inv_rho)*
                   (q(RHO_U_INDEX + d)*inv_rho);
    }
    const auto e1 = m1E1/m1 /*--- TODO: Add treatment for vanishing volume fraction ---*/
                  - static_cast<Number>(0.5)*norm2_vel;
    const auto p1 = this->EOS_phase1.pres_value(rho1, e1);

    /*--- Phase 2 ---*/
    const auto rho2 = m2/(static_cast<Number>(1.0) - alpha1); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto e2   = m2E2/m2 /*--- TODO: Add treatment for vanishing volume fraction ---*/
                    - static_cast<Number>(0.5)*norm2_vel;
    const auto p2   = this->EOS_phase2.pres_value(rho2, e2);

    /*--- Compute middle state ---*/
    FluxValue<typename Flux<Field>::cfg> q_star;

    const auto m1_star           = m1*((S - vel_d)/(S - S_star));
    q_star(ALPHA1_RHO1_INDEX)    = m1_star;
    const auto m2_star           = m2*((S - vel_d)/(S - S_star));
    q_star(ALPHA2_RHO2_INDEX)    = m2_star;
    const auto rho_star          = m1_star + m2_star;
    q_star(RHO_ALPHA1_INDEX)     = rho_star*alpha1;
    q_star(RHO_U_INDEX + curr_d) = rho_star*S_star;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      if(d != curr_d) {
        q_star(RHO_U_INDEX + d) = rho_star*(q(RHO_U_INDEX + d)*inv_rho);
      }
    }
    q_star(ALPHA1_RHO1_E1_INDEX) = m1_star*(m1E1/m1 + (S_star - vel_d)*(S_star + p1/(rho1*(S - vel_d))));
                                   /*--- TODO: Add treatment for vanishing volume fraction ---*/
    q_star(ALPHA2_RHO2_E2_INDEX) = m2_star*(m2E2/m2 + (S_star - vel_d)*(S_star + p2/(rho2*(S - vel_d))));
                                   /*--- TODO: Add treatment for vanishing volume fraction ---*/

    return q_star;
  }

  // Implementation of a the numerical flux (wave-propagation)
  //
  template<class Field>
  void HLLCFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                              const FluxValue<typename Flux<Field>::cfg>& qR,
                                              const std::size_t curr_d,
                                              FluxValue<typename Flux<Field>::cfg>& H_minus,
                                              FluxValue<typename Flux<Field>::cfg>& H_plus) {
    /*--- Left state ---*/
    // Pre-fetch variables that will be used several times so as to exploit possible vectorization
    // (as well as to enhance readability)
    const auto m1_L   = qL(ALPHA1_RHO1_INDEX);
    const auto m2_L   = qL(ALPHA2_RHO2_INDEX);
    const auto m1E1_L = qL(ALPHA1_RHO1_E1_INDEX);
    const auto m2E2_L = qL(ALPHA2_RHO2_E2_INDEX);

    // Save mixture density and velocity current direction left state
    const auto rho_L     = m1_L + m2_L;
    const auto inv_rho_L = static_cast<Number>(1.0)/rho_L;
    const auto vel_L_d   = qL(RHO_U_INDEX + curr_d)*inv_rho_L;

    // Phase 1
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
    const auto c1_L = this->EOS_phase1.c_value(rho1_L, p1_L);

    // Phase 2
    const auto rho2_L = m2_L/(static_cast<Number>(1.0) - alpha1_L); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto e2_L   = m2E2_L/m2_L /*--- TODO: Add treatment for vanishing volume fraction ---*/
                      - static_cast<Number>(0.5)*norm2_vel_L;
    const auto p2_L   = this->EOS_phase2.pres_value(rho2_L, e2_L);
    const auto c2_L   = this->EOS_phase2.c_value(rho2_L, p2_L);

    // Compute frozen speed of sound and mixture pressure left state
    const auto Y1_L = m1_L*inv_rho_L;
    const auto c_L  = std::sqrt(Y1_L*c1_L*c1_L +
                                (static_cast<Number>(1.0) - Y1_L)*c2_L*c2_L);
    const auto p_L  = alpha1_L*p1_L
                    + (static_cast<Number>(1.0) - alpha1_L)*p2_L;

    /*--- Right state ---*/
    // Pre-fetch variables that will be used several times so as to exploit possible vectorization
    // (as well as to enhance readability)
    const auto m1_R   = qR(ALPHA1_RHO1_INDEX);
    const auto m2_R   = qR(ALPHA2_RHO2_INDEX);
    const auto m1E1_R = qR(ALPHA1_RHO1_E1_INDEX);
    const auto m2E2_R = qR(ALPHA2_RHO2_E2_INDEX);

    // Save mixture density and velocity current direction left state
    const auto rho_R     = m1_R + m2_R;
    const auto inv_rho_R = static_cast<Number>(1.0)/rho_R;
    const auto vel_R_d   = qR(RHO_U_INDEX + curr_d)*inv_rho_R;

    // Phase 1
    const auto alpha1_R = qR(RHO_ALPHA1_INDEX)*inv_rho_R;
    const auto rho1_R   = m1_R/alpha1_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto norm2_vel_R  = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      norm2_vel_R += (qR(RHO_U_INDEX + d)*inv_rho_R)*
                     (qR(RHO_U_INDEX + d)*inv_rho_R);
    }
    const auto e1_R = m1E1_R/m1_R /*--- TODO: Add treatment for vanishing volume fraction ---*/
                    - static_cast<Number>(0.5)*norm2_vel_R;
    const auto p1_R = this->EOS_phase1.pres_value(rho1_R, e1_R);
    const auto c1_R = this->EOS_phase1.c_value(rho1_R, p1_R);

    // Phase 2
    const auto rho2_R = m2_R/(static_cast<Number>(1.0) - alpha1_R); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto e2_R   = m2E2_R/m2_R /*--- TODO: Add treatment for vanishing volume fraction ---*/
                      - static_cast<Number>(0.5)*norm2_vel_R;
    const auto p2_R   = this->EOS_phase2.pres_value(rho2_R, e2_R);
    const auto c2_R   = this->EOS_phase2.c_value(rho2_R, p2_R);

    // Compute frozen speed of sound and mixture pressure right state
    const auto Y1_R = m1_R*inv_rho_R;
    const auto c_R  = std::sqrt(Y1_R*c1_R*c1_R +
                                (static_cast<Number>(1.0) - Y1_R)*c2_R*c2_R);
    const auto p_R  = alpha1_R*p1_R
                    + (static_cast<Number>(1.0) - alpha1_R)*p2_R;

    /*--- Compute speeds of wave propagation ---*/
    const auto s_L     = std::min(vel_L_d - c_L, vel_R_d - c_R);
    const auto s_R     = std::max(vel_L_d + c_L, vel_R_d + c_R);
    const auto s_star  = (p_R - p_L + rho_L*vel_L_d*(s_L - vel_L_d) - rho_R*vel_R_d*(s_R - vel_R_d))/
                         (rho_L*(s_L - vel_L_d) - rho_R*(s_R - vel_R_d));

    /*--- Compute intermediate states ---*/
    const auto q_star_L = compute_middle_state(qL, s_L, s_star, curr_d);
    const auto q_star_R = compute_middle_state(qR, s_R, s_star, curr_d);

    /*--- Compute the fluctuations (wave propagation formalism) ---*/
    if(s_L >= static_cast<Number>(0.0)) {
      H_minus.fill(static_cast<Number>(0.0));
      H_plus = s_R*(q_star_R - qR) + s_star*(q_star_L - q_star_R) + s_L*(qL - q_star_L);
    }
    else if(s_L < static_cast<Number>(0.0) &&
            s_star >= static_cast<Number>(0.0)) {
      H_minus = s_L*(q_star_L - qL);
      H_plus  = s_R*(q_star_R - qR) + s_star*(q_star_L - q_star_R);
    }
    else if(s_star < static_cast<Number>(0.0) &&
            s_R >= static_cast<Number>(0.0)) {
      H_minus = s_L*(q_star_L - qL) + s_star*(q_star_R - q_star_L);
      H_plus  = s_R*(q_star_R - qR);
    }
    else if(s_R < static_cast<Number>(0.0)) {
      H_minus = s_L*(q_star_L - qL) + s_star*(q_star_R - q_star_L) + s_R*(qR - q_star_R);
      H_plus.fill(static_cast<Number>(0.0));
    }
  }

  // Implement the contribution of the discrete flux for all the dimensions.
  //
  template<class Field>
  auto HLLCFlux<Field>::make_flux() {
    FluxDefinition<typename Flux<Field>::cfg> HLLC_f;

    /*--- Perform the loop over each dimension to compute the flux contribution ---*/
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
         {
           static constexpr int d = decltype(integral_constant_d)::value;

           // Compute now the "discrete" flux function, in this case a HLLC flux
           HLLC_f[d].flux_function = [&](samurai::FluxValuePair<typename Flux<Field>::cfg>& flux,
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

                                              FluxValue<typename Flux<Field>::cfg> qL = this->prim2cons(primL_recon);
                                              FluxValue<typename Flux<Field>::cfg> qR = this->prim2cons(primR_recon);
                                            #else
                                              // Extract the states
                                              const FluxValue<typename Flux<Field>::cfg>& qL = field[0];
                                              const FluxValue<typename Flux<Field>::cfg>& qR = field[1];
                                            #endif

                                            FluxValue<typename Flux<Field>::cfg> H_minus,
                                                                                 H_plus;

                                            compute_discrete_flux(qL, qR, d, H_minus, H_plus);

                                            flux[0] = H_minus;
                                            flux[1] = -H_plus;
                                         };
        }
    );

    auto scheme = make_flux_based_scheme(HLLC_f);
    scheme.set_name("HLLC wave-propagation");

    return scheme;
  }

} // end of namespace
