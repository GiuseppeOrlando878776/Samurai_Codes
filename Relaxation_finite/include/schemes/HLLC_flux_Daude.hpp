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
  using Math::operator+;
  using Math::operator-;
  using Math::operator*;

  /**
    * Implementation of a HLLC flux for the BN model following Lochon et al., JCP, 2016.
    */
  template<class Field>
  class HLLCFlux: public Flux<Field> {
  public:
    using Indices = Flux<Field>::Indices; /*--- Shortcut for the indices storage ---*/
    using Number  = Flux<Field>::Number;  /*--- Define the shortcut for the arithmetic type ---*/
    using cfg     = Flux<Field>::cfg;     /*--- Shortcut to specify the type of configuration
                                                for the flux (nonlinear in this case) ---*/

    HLLCFlux(const EOS<Number>& EOS_phase1_,
             const EOS<Number>& EOS_phase2_); /*--- Constructor which accepts in input
                                                    the equations of state of the two phases ---*/

    auto make_flux(); /*--- Compute the flux over all the faces and directions ---*/

  private:
    void compute_p_star(const Number alpha1_L, const Number alpha1_R,
                        const Number alpha2_L, const Number alpha2_R,
                        const Number vel1_L_d, const Number vel1_R_d,
                        const Number vel2_L_d, const Number vel2_R_d,
                        const Number p1_L, const Number p1_R,
                        const Number p2_L, const Number p2_R,
                        const Number inv_c1_tilde_L, const Number inv_c1_tilde_R,
                        const Number inv_c2_tilde_L, const Number inv_c2_tilde_R,
                        Number& p1_star_L, Number& p1_star_R, Number& p2_star) const; /*--- Compute p_star (approximate contact assumption) ---*/

    auto compute_middle_state(const auto& q,
                              const Number S,
                              const Number S_star,
                              const std::size_t curr_d,
                              const unsigned phase_idx) const; /*--- Compute the middle state ---*/

    auto compute_approximate_contact(const auto& q2_L,
                                     const auto& q2_R,
                                     const Number vel2_star_L_d,
                                     const Number vel2_star_R_d,
                                     const Number p2_star,
                                     const Number S2_L,
                                     const Number S2_R,
                                     const Number S1_star,
                                     const Number S2_star,
                                     const std::size_t curr_d) const; /*--- Compute the approxiamte contact state ---*/

    void compute_discrete_flux(const FluxValue<cfg>& qL,
                               const FluxValue<cfg>& qR,
                               const std::size_t curr_d,
                               FluxValue<cfg>& F_minus,
                               FluxValue<cfg>& F_plus); /*--- HLLC flux along direction curr_d ---*/
  };

  // Constructor derived from base class
  //
  template<class Field>
  HLLCFlux<Field>::HLLCFlux(const EOS<Number>& EOS_phase1_,
                            const EOS<Number>& EOS_phase2_):
    Flux<Field>(EOS_phase1_, EOS_phase2_) {}


  // Implement the auxiliary routine that computes the middle state
  //
  template<class Field>
  void HLLCFlux<Field>::compute_p_star(const Number alpha1_L, const Number alpha1_R,
                                       const Number alpha2_L, const Number alpha2_R,
                                       const Number vel1_L_d, const Number vel1_R_d,
                                       const Number vel2_L_d, const Number vel2_R_d,
                                       const Number p1_L, const Number p1_R,
                                       const Number p2_L, const Number p2_R,
                                       const Number inv_c1_tilde_L, const Number inv_c1_tilde_R,
                                       const Number inv_c2_tilde_L, const Number inv_c2_tilde_R,
                                       Number& p1_star_L, Number& p1_star_R, Number& p2_star) const {
    /*--- Compute the right-hand sides ---*/
    const auto rhs_p2_star = p1_L*inv_c1_tilde_L - vel1_L_d
                           - p1_R*inv_c1_tilde_R + vel1_R_d;

    const auto rhs_p1_star_L = alpha2_L*(p2_L*inv_c2_tilde_L - vel2_L_d - p1_L*inv_c1_tilde_L + vel1_L_d)
                             - alpha2_R*(p2_R*inv_c2_tilde_R - vel2_R_d - p1_R*inv_c1_tilde_R + vel1_R_d);

    /*--- Compute the determinant of the auxiliary matrices for Cramer's rule ---*/
    const auto a12 = inv_c1_tilde_L;
    const auto a13 = -inv_c1_tilde_R;

    const auto a21 = alpha2_L*inv_c2_tilde_L - alpha2_R*inv_c2_tilde_R;
    const auto a22 = -alpha2_L*inv_c1_tilde_L;
    const auto a23 = alpha2_R*inv_c1_tilde_R;

    const auto a31 = alpha2_L - alpha2_R;
    const auto a32 = alpha1_L;
    const auto a33 = -alpha1_R;

    /*--- Compute the determinant of the matrix ---*/
    const auto det_A     = a12*(a23*a31 - a21*a33) + a13*(a21*a32 - a22*a31);
    /*const auto det_A = (alpha2_L*inv_c2_tilde_L - alpha2_R*inv_c2_tilde_R)*
                       (alpha1_R*inv_c1_tilde_R - alpha1_L*inv_c1_tilde_L)
                     - (alpha2_L - alpha2_R)*(alpha2_L - alpha2_R)*
                       inv_c1_tilde_L*inv_c1_tilde_R;*/
    const auto inv_det_A = static_cast<Number>(1.0)/det_A;

    const auto det_A1 = rhs_p2_star*(a22*a33 - a23*a32) + rhs_p1_star_L*(a13*a32 - a12*a33);
    const auto det_A2 = rhs_p2_star*(a23*a31 - a21*a33) - rhs_p1_star_L*a13*a31;
    const auto det_A3 = rhs_p2_star*(a21*a32 - a22*a31) + rhs_p1_star_L*a12*a31;

    /*--- Result obtained through Cramer's rule ---*/
    p2_star   = det_A1*inv_det_A;
    p1_star_L = det_A2*inv_det_A;
    p1_star_R = det_A3*inv_det_A;
  }

  // Implement the auxiliary routine that computes the middle state
  //
  template<class Field>
  auto HLLCFlux<Field>::compute_middle_state(const auto& q,
                                             const Number S,
                                             const Number S_star,
                                             const std::size_t curr_d,
                                             const unsigned phase_idx) const {
    /*--- Pre-fetch variables that will be used several times so as to exploit possible vectorization.
          Same considerations of 'evaluate_phasic_continuous_flux' ---*/
    const auto alphak = q.front();
    const auto mk     = q[1];
    const auto mkEk   = q.back();

    /*--- Compute density, velocity (along the dimension) and internal energy ---*/
    const auto rhok   = mk/alphak; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto inv_mk = static_cast<Number>(1.0)/mk; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto ek           = mkEk*inv_mk; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < Field::dim; ++d) {
      ek -= static_cast<Number>(0.5)*
            ((q[d + 2]*inv_mk)*(q[d + 2]*inv_mk)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    Number pk;
    if(phase_idx == 1) {
      pk = this->EOS_phase1.pres_value_Rhoe(rhok, ek);
    }
    else if(phase_idx == 2) {
      pk = this->EOS_phase2.pres_value_Rhoe(rhok, ek);
    }
    else {
      std::cerr << "Unknown phasic index. Exiting..." << std::endl;
      exit(1);
    }
    const auto velk_d = q[curr_d + 2]*inv_mk; /*--- TODO: Add treatment for vanishing volume fraction ---*/

    /*--- Compute the corresponding middle state ---*/
    std::array<Number, Field::dim + 2> res;

    const auto mk_star = mk*((S - velk_d)/(S - S_star));

    res.front() = mk_star;
    res[curr_d + 1] = mk_star*S_star;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      if(d != curr_d) {
        res[d + 1] = mk_star*(q[d + 2]*inv_mk);
      }
    }
    res.back() = mk_star*(mkEk*inv_mk + (S_star - velk_d)*(S_star + pk/(rhok*(S - velk_d))));

    return res;
  }

  // Implement the auxiliary routine that computes the approximate contact state
  //
  template<class Field>
  auto HLLCFlux<Field>::compute_approximate_contact(const auto& q2_L,
                                                    const auto& q2_R,
                                                    const Number vel2_star_L_d,
                                                    const Number vel2_star_R_d,
                                                    const Number p2_star,
                                                    const Number S2_L,
                                                    const Number S2_R,
                                                    const Number S1_star,
                                                    const Number S2_star,
                                                    const std::size_t curr_d) const {
    std::array<Number, Field::dim + 2> res;
    if(S1_star <= S2_star) {
      const auto alpha2_M = q2_R.front();

      const auto rho2_L      = q2_L[1]/q2_L.front();
      const auto vel2_L_d    = q2_L[curr_d + 2]/q2_L[1]; /*--- TODO: Add treatment for vanishing volume fraction ---*/
      const auto rho2_star_M = rho2_L*((S2_L - vel2_L_d)/(S2_L - S2_star));;

      const auto m2_star_M = alpha2_M*rho2_star_M;

      const auto vel2_star_M_d = vel2_star_R_d;

      const auto inv_m2_R = static_cast<Number>(1.0)/q2_R[1]; /*--- TODO: Add treatment for vanishing volume fraction ---*/

      res.front() = m2_star_M;
      res[curr_d + 1] = m2_star_M*vel2_star_M_d;
      auto norm2_vel2_star_M = vel2_star_M_d*vel2_star_M_d;
      for(std::size_t d = 0; d < Field::dim; ++d) {
        if(d != curr_d) {
          const auto vel2_R_d = q2_R[d + 2]*inv_m2_R;
          res[d + 1] = m2_star_M*vel2_R_d;

          norm2_vel2_star_M += vel2_R_d*vel2_R_d;
        }
      }
      res.back() = m2_star_M*(this->EOS_phase2.e_value_RhoP(rho2_star_M, p2_star) +
                              static_cast<Number>(0.5)*norm2_vel2_star_M);
    }
    else {
      const auto alpha2_M = q2_L.front();

      const auto rho2_R      = q2_R[1]/q2_R.front();
      const auto vel2_R_d    = q2_R[curr_d + 2]/q2_R[1]; /*--- TODO: Add treatment for vanishing volume fraction ---*/
      const auto rho2_star_M = rho2_R*((S2_R - vel2_R_d)/(S2_R - S2_star));;

      const auto m2_star_M = alpha2_M*rho2_star_M;

      const auto vel2_star_M_d = vel2_star_L_d;

      const auto inv_m2_L = static_cast<Number>(1.0)/q2_L[1]; /*--- TODO: Add treatment for vanishing volume fraction ---*/

      res.front() = m2_star_M;
      res[curr_d + 1] = m2_star_M*vel2_star_M_d;
      auto norm2_vel2_star_M = vel2_star_M_d*vel2_star_M_d;
      for(std::size_t d = 0; d < Field::dim; ++d) {
        if(d != curr_d) {
          const auto vel2_L_d = q2_L[d + 2]*inv_m2_L;
          res[d + 1] = m2_star_M*vel2_L_d;

          norm2_vel2_star_M += vel2_L_d*vel2_L_d;
        }
      }
      res.back() = m2_star_M*(this->EOS_phase2.e_value_RhoP(rho2_star_M, p2_star) +
                              static_cast<Number>(0.5)*norm2_vel2_star_M);
    }

    return res;
  }

  // Implementation of a HLLC flux for the conservative portion of the system (no volume fraction)
  //
  template<class Field>
  void HLLCFlux<Field>::compute_discrete_flux(const FluxValue<cfg>& qL,
                                              const FluxValue<cfg>& qR,
                                              std::size_t curr_d,
                                              FluxValue<cfg>& F_minus,
                                              FluxValue<cfg>& F_plus) {
    /*--- Left state ---*/
    // Pre-fetch variables that will be used several times so as to exploit possible vectorization
    // (as well as to enhance readability)
    const auto alpha1_L = qL(Indices::ALPHA1_INDEX);
    const auto m1_L     = qL(Indices::ALPHA1_RHO1_INDEX);
    const auto m2_L     = qL(Indices::ALPHA2_RHO2_INDEX);
    const auto m1E1_L   = qL(Indices::ALPHA1_RHO1_E1_INDEX);
    const auto m2E2_L   = qL(Indices::ALPHA2_RHO2_E2_INDEX);

    // Verify if it is admissible
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
    #endif

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
    #ifdef VERBOSE_FLUX
      if(std::isnan(c1_L)) {
        throw std::runtime_error(std::string("NaN speed of sound phase 1 left state"));
      }
    #endif

    // Phase 2
    const auto inv_m2_L = static_cast<Number>(1.0)/m2_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto vel2_L_d = qL(Indices::ALPHA2_RHO2_U2_INDEX + curr_d)*inv_m2_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto alpha2_L = static_cast<Number>(1.0) - alpha1_L;
    const auto rho2_L   = m2_L/alpha2_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2_L           = m2E2_L*inv_m2_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < Field::dim; ++d) {
      e2_L -= static_cast<Number>(0.5)*
              ((qL(Indices::ALPHA2_RHO2_U2_INDEX + d)*inv_m2_L)*
               (qL(Indices::ALPHA2_RHO2_U2_INDEX + d)*inv_m2_L)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto p2_L = this->EOS_phase2.pres_value_Rhoe(rho2_L, e2_L);
    const auto c2_L = this->EOS_phase2.c_value_RhoP(rho2_L, p2_L);
    #ifdef VERBOSE_FLUX
      if(std::isnan(c2_L)) {
        throw std::runtime_error(std::string("NaN speed of sound phase 2 left state"));
      }
    #endif

    /*--- Right state ---*/
    // Pre-fetch variables that will be used several times so as to exploit possible vectorization
    // (as well as to enhance readability)
    const auto alpha1_R = qR(Indices::ALPHA1_INDEX);
    const auto m1_R     = qR(Indices::ALPHA1_RHO1_INDEX);
    const auto m2_R     = qR(Indices::ALPHA2_RHO2_INDEX);
    const auto m1E1_R   = qR(Indices::ALPHA1_RHO1_E1_INDEX);
    const auto m2E2_R   = qR(Indices::ALPHA2_RHO2_E2_INDEX);

    // Verify if it is admissible
    #ifdef VERBOSE_FLUX
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
    #ifdef VERBOSE_FLUX
      if(std::isnan(c1_R)) {
        throw std::runtime_error(std::string("NaN speed of sound phase 1 right state"));
      }
    #endif

    // Phase 2
    const auto inv_m2_R = static_cast<Number>(1.0)/m2_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto vel2_R_d = qR(Indices::ALPHA2_RHO2_U2_INDEX + curr_d)*inv_m2_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto alpha2_R = static_cast<Number>(1.0) - alpha1_R;
    const auto rho2_R   = m2_R/alpha2_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2_R           = m2E2_R*inv_m2_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < Field::dim; ++d) {
      e2_R -= static_cast<Number>(0.5)*
              ((qR(Indices::ALPHA2_RHO2_U2_INDEX + d)*inv_m2_R)*
               (qR(Indices::ALPHA2_RHO2_U2_INDEX + d)*inv_m2_R)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto p2_R = this->EOS_phase2.pres_value_Rhoe(rho2_R, e2_R);
    const auto c2_R = this->EOS_phase2.c_value_RhoP(rho2_R, p2_R);
    #ifdef VERBOSE_FLUX
      if(std::isnan(c2_R)) {
        throw std::runtime_error(std::string("NaN speed of sound phase 2 right state"));
      }
    #endif

    /*--- Compute speeds of wave propagation ---*/
    const auto S1_L = std::min(vel1_L_d - c1_L, vel1_R_d - c1_R);
    const auto S1_R = std::max(vel1_L_d + c1_L, vel1_R_d + c1_R);
    const auto S2_L = std::min(vel2_L_d - c2_L, vel2_R_d - c2_R);
    const auto S2_R = std::max(vel2_L_d + c2_L, vel2_R_d + c2_R);

    const auto inv_c1_tilde_L = static_cast<Number>(1.0)/(rho1_L*(S1_L - vel1_L_d));
    const auto inv_c1_tilde_R = static_cast<Number>(1.0)/(rho1_R*(S1_R - vel1_R_d));
    const auto inv_c2_tilde_L = static_cast<Number>(1.0)/(rho2_L*(S2_L - vel2_L_d));
    const auto inv_c2_tilde_R = static_cast<Number>(1.0)/(rho2_R*(S2_R - vel2_R_d));

    Number p1_star_L, p1_star_R, p2_star;
    compute_p_star(alpha1_L, alpha1_R,
                   alpha2_L, alpha2_R,
                   vel1_L_d, vel1_R_d,
                   vel2_L_d, vel2_R_d,
                   p1_L, p1_R,
                   p2_L, p2_R,
                   inv_c1_tilde_L, inv_c1_tilde_R,
                   inv_c2_tilde_L, inv_c2_tilde_R,
                   p1_star_L, p1_star_R, p2_star);

    const auto vel1_star_L_d = vel1_L_d + inv_c1_tilde_L*(p1_star_L - p1_L);
    const auto vel1_star_R_d = vel1_R_d + inv_c1_tilde_R*(p1_star_R - p1_R);
    const auto vel2_star_L_d = vel2_L_d + inv_c2_tilde_L*(p2_star - p2_L);
    const auto vel2_star_R_d = vel2_R_d + inv_c2_tilde_R*(p2_star - p2_R);

    Number S1_star, S2_star;
    if(vel1_star_L_d <= vel2_star_R_d) {
      S1_star = vel1_star_L_d;
      S2_star = vel2_star_R_d;
    }
    else {
      S1_star = vel1_star_R_d;
      S2_star = vel2_star_L_d;
    }

    /*--- Compute the flux for the conservative part ---*/
    // "Extract" the relevant variables for phase 1 and phase 2, respectively
    std::array<Number, Field::dim + 3> q1_L,
                                       q1_R,
                                       q2_L,
                                       q2_R;
    q1_L.front() = qL(Indices::ALPHA1_INDEX);
    q1_R.front() = qR(Indices::ALPHA1_INDEX);
    q2_L.front() = static_cast<Number>(1.0) - q1_L.front();
    q2_R.front() = static_cast<Number>(1.0) - q1_R.front();
    q1_L[1] = qL(Indices::ALPHA1_RHO1_INDEX);
    q1_R[1] = qR(Indices::ALPHA1_RHO1_INDEX);
    q2_L[1] = qL(Indices::ALPHA2_RHO2_INDEX);
    q2_R[1] = qR(Indices::ALPHA2_RHO2_INDEX);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      q1_L[d + 2] = qL(Indices::ALPHA1_RHO1_U1_INDEX + d);
      q1_R[d + 2] = qR(Indices::ALPHA1_RHO1_U1_INDEX + d);
      q2_L[d + 2] = qL(Indices::ALPHA2_RHO2_U2_INDEX + d);
      q2_R[d + 2] = qR(Indices::ALPHA2_RHO2_U2_INDEX + d);
    }
    q1_L.back() = qL(Indices::ALPHA1_RHO1_E1_INDEX);
    q1_R.back() = qR(Indices::ALPHA1_RHO1_E1_INDEX);
    q2_L.back() = qL(Indices::ALPHA2_RHO2_E2_INDEX);
    q2_R.back() = qR(Indices::ALPHA2_RHO2_E2_INDEX);

    // Compute the numerical flux for phase 1
    unsigned curr_phase_idx = 1;
    std::array<Number, Field::dim + 2> num_flux_phase1;
    if(S1_L >= static_cast<Number>(0.0)) {
      num_flux_phase1 = this->evaluate_phasic_continuous_flux(q1_L, curr_d, curr_phase_idx);
    }
    else if(S1_L < static_cast<Number>(0.0) &&
            S1_star >= static_cast<Number>(0.0)) {
      const auto q1_star_L = compute_middle_state(q1_L, S1_L, S1_star, curr_d, curr_phase_idx);
      decltype(num_flux_phase1) q1_L_tmp;
      std::copy(q1_L.begin() + 1, q1_L.end(), q1_L_tmp.begin());

      num_flux_phase1 = this->evaluate_phasic_continuous_flux(q1_L, curr_d, curr_phase_idx)
                      + S1_L*(q1_star_L - q1_L_tmp);
    }
    else if(S1_star < static_cast<Number>(0.0) &&
            S1_R >= static_cast<Number>(0.0)) {
      const auto q1_star_R = compute_middle_state(q1_R, S1_R, S1_star, curr_d, curr_phase_idx);
      decltype(num_flux_phase1) q1_R_tmp;
      std::copy(q1_R.begin() + 1, q1_R.end(), q1_R_tmp.begin());

      num_flux_phase1 = this->evaluate_phasic_continuous_flux(q1_R, curr_d, curr_phase_idx)
                      + S1_R*(q1_star_R - q1_R_tmp);
    }
    else if(S1_R < static_cast<Number>(0.0)) {
      num_flux_phase1 = this->evaluate_phasic_continuous_flux(q1_R, curr_d, curr_phase_idx);
    }

    // Compute the numerical flux for phase 2
    curr_phase_idx = 2;
    decltype(num_flux_phase1) num_flux_phase2;
    if(S1_star <= S2_star) {
      if(S2_L >= static_cast<Number>(0.0)) {
        num_flux_phase2 = this->evaluate_phasic_continuous_flux(q2_L, curr_d, curr_phase_idx);
      }
      else if(S2_L < static_cast<Number>(0.0) &&
              S1_star >= static_cast<Number>(0.0)) {
        const auto q2_star_L = compute_middle_state(q2_L, S2_L, S2_star, curr_d, curr_phase_idx);
        decltype(num_flux_phase2) q2_L_tmp;
        std::copy(q2_L.begin() + 1, q2_L.end(), q2_L_tmp.begin());

        num_flux_phase2 = this->evaluate_phasic_continuous_flux(q2_L, curr_d, curr_phase_idx)
                        + S2_L*(q2_star_L - q2_L_tmp);
      }
      else if(S1_star < static_cast<Number>(0.0) &&
              S2_star >= static_cast<Number>(0.0)) {
        const auto q2_star_R = compute_middle_state(q2_R, S2_R, S2_star, curr_d, curr_phase_idx);
        const auto q2_star_M = compute_approximate_contact(q2_L, q2_R,
                                                           vel2_star_L_d, vel2_star_R_d,
                                                           p2_star,
                                                           S2_L, S2_R,
                                                           S1_star, S2_star,
                                                           curr_d);
        decltype(num_flux_phase2) q2_R_tmp;
        std::copy(q2_R.begin() + 1, q2_R.end(), q2_R_tmp.begin());

        num_flux_phase2 = this->evaluate_phasic_continuous_flux(q2_R, curr_d, curr_phase_idx)
                        + S2_R*(q2_star_R - q2_R_tmp)
                        + S2_star*(q2_star_M - q2_star_R);
      }
      else if(S2_star < static_cast<Number>(0.0) &&
              S2_R >= static_cast<Number>(0.0)) {
        const auto q2_star_R = compute_middle_state(q2_R, S2_R, S2_star, curr_d, curr_phase_idx);
        decltype(num_flux_phase2) q2_R_tmp;
        std::copy(q2_R.begin() + 1, q2_R.end(), q2_R_tmp.begin());

        num_flux_phase2 = this->evaluate_phasic_continuous_flux(q2_R, curr_d, curr_phase_idx)
                        + S2_R*(q2_star_R - q2_R_tmp);
      }
      else if(S2_R < static_cast<Number>(0.0)) {
        num_flux_phase2 = this->evaluate_phasic_continuous_flux(q2_R, curr_d, curr_phase_idx);
      }
    }
    else {
      if(S2_L >= static_cast<Number>(0.0)) {
        num_flux_phase2 = this->evaluate_phasic_continuous_flux(q2_L, curr_d, curr_phase_idx);
      }
      else if(S2_L < static_cast<Number>(0.0) &&
              S2_star >= static_cast<Number>(0.0)) {
        const auto q2_star_L = compute_middle_state(q2_L, S2_L, S2_star, curr_d, curr_phase_idx);
        decltype(num_flux_phase2) q2_L_tmp;
        std::copy(q2_L.begin() + 1, q2_L.end(), q2_L_tmp.begin());

        num_flux_phase2 = this->evaluate_phasic_continuous_flux(q2_L, curr_d, curr_phase_idx)
                        + S2_L*(q2_star_L - q2_L_tmp);
      }
      else if(S2_star < static_cast<Number>(0.0) &&
              S1_star >= static_cast<Number>(0.0)) {
        const auto q2_star_L = compute_middle_state(q2_L, S2_L, S2_star, curr_d, curr_phase_idx);
        const auto q2_star_M = compute_approximate_contact(q2_L, q2_R,
                                                           vel2_star_L_d, vel2_star_R_d,
                                                           p2_star,
                                                           S2_L, S2_R,
                                                           S1_star, S2_star,
                                                           curr_d);
        decltype(num_flux_phase2) q2_L_tmp;
        std::copy(q2_L.begin() + 1, q2_L.end(), q2_L_tmp.begin());

        num_flux_phase2 = this->evaluate_phasic_continuous_flux(q2_L, curr_d, curr_phase_idx)
                        + S2_L*(q2_star_L - q2_L_tmp)
                        + S2_star*(q2_star_M - q2_star_L);
      }
      else if(S1_star < static_cast<Number>(0.0) &&
              S2_R >= static_cast<Number>(0.0)) {
        const auto q2_star_R = compute_middle_state(q2_R, S2_R, S2_star, curr_d, curr_phase_idx);
        decltype(num_flux_phase2) q2_R_tmp;
        std::copy(q2_R.begin() + 1, q2_R.end(), q2_R_tmp.begin());

        num_flux_phase2 = this->evaluate_phasic_continuous_flux(q2_R, curr_d, curr_phase_idx)
                        + S2_R*(q2_star_R - q2_R_tmp);
      }
      else if(S2_R < static_cast<Number>(0.0)) {
        num_flux_phase2 = this->evaluate_phasic_continuous_flux(q2_R, curr_d, curr_phase_idx);
      }
    }

    // Conclude computation of numerical flux
    F_minus(Indices::ALPHA1_INDEX) = static_cast<Number>(0.0);
    F_minus(Indices::ALPHA1_RHO1_INDEX) = num_flux_phase1.front();
    F_minus(Indices::ALPHA2_RHO2_INDEX) = num_flux_phase2.front();
    for(std::size_t d = 0; d < Field::dim; ++d) {
      F_minus(Indices::ALPHA1_RHO1_U1_INDEX + d) = num_flux_phase1[d + 1];
      F_minus(Indices::ALPHA2_RHO2_U2_INDEX + d) = num_flux_phase2[d + 1];
    }
    F_minus(Indices::ALPHA1_RHO1_E1_INDEX) = num_flux_phase1.back();
    F_minus(Indices::ALPHA2_RHO2_E2_INDEX) = num_flux_phase2.back();

    F_plus = F_minus;

    // Consider the contribution of non-conservative part
    if(S1_star < static_cast<Number>(0.0)) {
      F_minus(Indices::ALPHA1_INDEX) += S1_star*(alpha1_R - alpha1_L);
      F_minus(Indices::ALPHA1_RHO1_U1_INDEX + curr_d) -= (alpha1_R*p1_star_R - alpha1_L*p1_star_L);
      F_minus(Indices::ALPHA2_RHO2_U2_INDEX + curr_d) += (alpha1_R*p1_star_R - alpha1_L*p1_star_L);
      F_minus(Indices::ALPHA1_RHO1_E1_INDEX) -= S1_star*(alpha1_R*p1_star_R - alpha1_L*p1_star_L);
      F_minus(Indices::ALPHA2_RHO2_E2_INDEX) += S1_star*(alpha1_R*p1_star_R - alpha1_L*p1_star_L);
    }
    else {
      F_plus(Indices::ALPHA1_INDEX) -= S1_star*(alpha1_R - alpha1_L);
      F_plus(Indices::ALPHA1_RHO1_U1_INDEX + curr_d) += (alpha1_R*p1_star_R - alpha1_L*p1_star_L);
      F_plus(Indices::ALPHA2_RHO2_U2_INDEX + curr_d) -= (alpha1_R*p1_star_R - alpha1_L*p1_star_L);
      F_plus(Indices::ALPHA1_RHO1_E1_INDEX) += S1_star*(alpha1_R*p1_star_R - alpha1_L*p1_star_L);
      F_plus(Indices::ALPHA2_RHO2_E2_INDEX) -= S1_star*(alpha1_R*p1_star_R - alpha1_L*p1_star_L);
    }
  }

  // Implement the contribution of the discrete flux for all the dimensions.
  //
  template<class Field>
  auto HLLCFlux<Field>::make_flux() {
    FluxDefinition<cfg> HLLC_flux;

    /*--- Perform the loop over each dimension to compute the flux contribution ---*/
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
         {
           static constexpr int d = decltype(integral_constant_d)::value;

           // Compute now the "discrete" flux function, in this case a HLLC flux
           HLLC_flux[d].flux_function = [&](FluxValuePair<cfg>& flux,
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

                                                FluxValue<cfg> qL = this->prim2cons(primL_recon);
                                                FluxValue<cfg> qR = this->prim2cons(primR_recon);
                                              #else
                                                // Extract the states
                                                const FluxValue<cfg>& qL = field[0];
                                                const FluxValue<cfg>& qR = field[1];
                                              #endif

                                              FluxValue<cfg> F_minus,
                                                             F_plus;

                                              compute_discrete_flux(qL, qR, d, F_minus, F_plus);

                                              flux[0] = F_minus;
                                              flux[1] = -F_plus;
                                            };
        }
    );

    auto scheme = make_flux_based_scheme(HLLC_flux);
    scheme.set_name("HLLC");

    return scheme;
  }

} // end of namespace
