// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Khaled Saleh, 2017-2019-2024
//         Giuseppe Orlando, 2025
//
#ifndef Suliciu_scheme_hpp
#define Suliciu_scheme_hpp

#include "flux_base.hpp"

namespace samurai {
  /**
    * Implementation of the flux based on Suliciu-type relaxation
    */
  template<class Field>
  class RelaxationFlux: public Flux<Field> {
  public:
    using Indices = Flux<Field>::Indices; /*--- Shortcut for the indices storage ---*/
    using Number  = Flux<Field>::Number;  /*--- Shortcut for the arithmetic type ---*/

    RelaxationFlux(const EOS<Number>& EOS_phase1_,
                   const EOS<Number>& EOS_phase2_,
                   const Number atol_Newton_ = 1e-8,
                   const Number rtol_Newton_ = 1e-6,
                   const std::size_t max_Newton_iters_ = 60); /*--- Constructor which accepts in input
                                                                    the equations of state of the two phases ---*/

    auto make_flux(Number& c); /*--- Compute the flux over all the faces and directions.
                                     The input argument is employed to compute the Courant number ---*/

  private:
    const Number      atol_Newton;      /*--- Absolute tolerance for the Newont method of the Suliciu scheme ---*/
    const Number      rtol_Newton;      /*--- Relative tolerance for the Newont method of the Suliciu scheme ---*/
    const std::size_t max_Newton_iters; /*--- Maximum number of Newton iterations ---*/

    template<typename T>
    inline T M0(const T nu, const T Me) const;

    template<typename T>
    inline T psi(const T u_star, const T a,
                 const T alphaL, const T alphaR,
                 const T vel_diesis, const T tauL_diesis, const T tauR_diesis) const;

    template<typename T>
    inline T Psi(const T u_star,
                 const T a1, const T alpha1L, const T alpha1R, const T vel1_diesis,
                 const T a2, const T alpha2L, const T alpha2R, const T vel2_diesis,
                 const T tau2L_diesis, const T tau2R_diesis) const;

    template<typename T>
    inline T dM0_dMe(const T nu, const T Me) const;

    template<typename T>
    inline T dpsi_dustar(const T u_star, const T a,
                         const T alphaL, const T alphaR,
                         const T vel_diesis, const T tauL_diesis, const T tauR_diesis) const;

    template<typename T>
    inline T dPsi_dustar(const T u_star,
                         const T a1, const T alpha1L, const T alpha1R,
                         const T a2, const T alpha2L, const T alpha2R,
                         const T vel2_diesis, const T tau2L_diesis, const T tau2R_diesis) const;

    template<typename T>
    T Newton(const T rhs,
             const T a1, const T alpha1L, const T alpha1R, const T vel1_diesis, const T tau1L_diesis, const T tau1R_diesis,
             const T a2, const T alpha2L, const T alpha2R, const T vel2_diesis, const T tau2L_diesis, const T tau2R_diesis,
             const T cLmax, const T cRmin) const;

    template<typename T>
    void Riemann_solver_phase_vI(const T xi,
                                 const T alphaL, const T alphaR, const T tauL, const T tauR,
                                 const T wL, const T wR, const T pL, const T pR, const T EL, const T ER,
                                 const T a, const T u_star,
                                 T& alpha_m, T& tau_m, T& w_m, T& pres_m, T& E_m,
                                 T& alpha_p, T& tau_p, T& w_p, T& pres_p, T& E_p);

    template<typename T>
    void Riemann_solver_phase_pI(const T xi,
                                 const T alphaL, const T alphaR, const T tauL, const T tauR,
                                 const T wL, const T wR, const T pL, const T pR, const T EL, const T ER,
                                 const T w_diesis, const T tauL_diesis, const T tauR_diesis, const T a,
                                 T& alpha_m, T& tau_m, T& w_m, T& pres_m, T& E_m,
                                 T& alpha_p, T& tau_p, T& w_p, T& pres_p, T& E_p,
                                 T& w_star);

    void compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                               const FluxValue<typename Flux<Field>::cfg>& qR,
                               #ifdef ORDER_2
                                  const Number alpha1_L_order1,
                                  const Number alpha1_R_order1,
                               #endif
                               const std::size_t curr_d,
                               FluxValue<typename Flux<Field>::cfg>& F_minus,
                               FluxValue<typename Flux<Field>::cfg>& F_plus,
                               Number& c); /*--- Compute discrete flux ---*/
  };

  // Constructor derived from base class
  //
  template<class Field>
  RelaxationFlux<Field>::RelaxationFlux(const EOS<Number>& EOS_phase1_,
                                        const EOS<Number>& EOS_phase2_,
                                        const Number atol_Newton_,
                                        const Number rtol_Newton_,
                                        const std::size_t max_Newton_iters_):
    Flux<Field>(EOS_phase1_, EOS_phase2_),
    atol_Newton(atol_Newton_), rtol_Newton(rtol_Newton_), max_Newton_iters(max_Newton_iters_) {}

  // Implementation of the flux (F^{-} and F^{+} as in Saleh 2012 notation)
  //
  template<class Field>
  void RelaxationFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                    const FluxValue<typename Flux<Field>::cfg>& qR,
                                                    #ifdef ORDER_2
                                                      const Number alpha1_L_order1,
                                                      const Number alpha1_R_order1,
                                                    #endif
                                                    std::size_t curr_d,
                                                    FluxValue<typename Flux<Field>::cfg>& F_minus,
                                                    FluxValue<typename Flux<Field>::cfg>& F_plus,
                                                    Number& c) {
    /*--- Left state ---*/
    // Pre-fetch variables that will be used several times so as to exploit possible vectorization
    // (as well as to enhance readability)
    const auto alpha1_L = qL(Indices::ALPHA1_INDEX);
    const auto m1_L     = qL(Indices::ALPHA1_RHO1_INDEX);
    const auto m1E1_L   = qL(Indices::ALPHA1_RHO1_E1_INDEX);
    const auto m2_L     = qL(Indices::ALPHA2_RHO2_INDEX);
    const auto m2E2_L   = qL(Indices::ALPHA2_RHO2_E2_INDEX);

    // Phase 1
    const auto inv_m1_L   = static_cast<Number>(1.0)/m1_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto vel1_L_d   = qL(Indices::ALPHA1_RHO1_U1_INDEX + curr_d)*inv_m1_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho1_L     = m1_L/alpha1_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto inv_rho1_L = static_cast<Number>(1.0)/rho1_L;
    const auto E1_L       = m1E1_L*inv_m1_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1_L             = E1_L;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      e1_L -= static_cast<Number>(0.5)*
              ((qL(Indices::ALPHA1_RHO1_U1_INDEX + d)*inv_m1_L)*
               (qL(Indices::ALPHA1_RHO1_U1_INDEX + d)*inv_m1_L)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto p1_L = this->EOS_phase1.pres_value_Rhoe(rho1_L, e1_L);

    // Phase 2
    const auto alpha2_L   = static_cast<Number>(1.0) - alpha1_L;
    const auto inv_m2_L   = static_cast<Number>(1.0)/m2_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto vel2_L_d   = qL(Indices::ALPHA2_RHO2_U2_INDEX + curr_d)*inv_m2_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho2_L     = m2_L/alpha2_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto inv_rho2_L = static_cast<Number>(1.0)/rho2_L;
    const auto E2_L       = m2E2_L*inv_m2_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2_L             = E2_L;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      e2_L -= static_cast<Number>(0.5)*
              ((qL(Indices::ALPHA2_RHO2_U2_INDEX + d)*inv_m2_L)*
               (qL(Indices::ALPHA2_RHO2_U2_INDEX + d)*inv_m2_L)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto p2_L = this->EOS_phase2.pres_value_Rhoe(rho2_L, e2_L);

    /*--- Right state ---*/
    // Pre-fetch variables that will be used several times so as to exploit possible vectorization
    // (as well as to enhance readability)
    const auto alpha1_R = qR(Indices::ALPHA1_INDEX);
    const auto m1_R     = qR(Indices::ALPHA1_RHO1_INDEX);
    const auto m1E1_R   = qR(Indices::ALPHA1_RHO1_E1_INDEX);
    const auto m2_R     = qR(Indices::ALPHA2_RHO2_INDEX);
    const auto m2E2_R   = qR(Indices::ALPHA2_RHO2_E2_INDEX);

    // Phase 1
    const auto inv_m1_R   = static_cast<Number>(1.0)/m1_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto vel1_R_d   = qR(Indices::ALPHA1_RHO1_U1_INDEX + curr_d)*inv_m1_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho1_R     = m1_R/alpha1_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto inv_rho1_R = static_cast<Number>(1.0)/rho1_R;
    const auto E1_R       = m1E1_R*inv_m1_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1_R             = E1_R;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      e1_R -= static_cast<Number>(0.5)*
              ((qR(Indices::ALPHA1_RHO1_U1_INDEX + d)*inv_m1_R)*
               (qR(Indices::ALPHA1_RHO1_U1_INDEX + d)*inv_m1_R)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto p1_R = this->EOS_phase1.pres_value_Rhoe(rho1_R, e1_R);

    // Phase 2
    const auto alpha2_R   = static_cast<Number>(1.0) - alpha1_R;
    const auto inv_m2_R   = static_cast<Number>(1.0)/m2_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto vel2_R_d   = qR(Indices::ALPHA2_RHO2_U2_INDEX + curr_d)*inv_m2_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho2_R     = m2_R/alpha2_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto inv_rho2_R = static_cast<Number>(1.0)/rho2_R;
    const auto E2_R       = m2E2_R*inv_m2_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2_R             = E2_R;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      e2_R -= static_cast<Number>(0.5)*
              ((qR(Indices::ALPHA2_RHO2_U2_INDEX + d)*inv_m2_R)*
               (qR(Indices::ALPHA2_RHO2_U2_INDEX + d)*inv_m2_R)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto p2_R = this->EOS_phase2.pres_value_Rhoe(rho2_R, e2_R);

    /*--- Compute first rhs of relaxation related parameters (Whitham's approach) ---*/
    auto a1 = std::max(this->EOS_phase1.c_value_RhoP(rho1_L, p1_L)*rho1_L,
                       this->EOS_phase1.c_value_RhoP(rho1_R, p1_R)*rho1_R);
    auto a2 = std::max(this->EOS_phase2.c_value_RhoP(rho2_L, p2_L)*rho2_L,
                       this->EOS_phase2.c_value_RhoP(rho2_R, p2_R)*rho2_R);

    /*--- Compute the transport step solving a non-linear equation with the Newton method ---*/
    // Compute "diesis" state (formulas (3.21) in Saleh ESAIM 2019, starting point for subsonic wave)
    Number vel1_diesis, p1_diesis,
           tau1L_diesis = static_cast<Number>(0.0),
           tau1R_diesis = static_cast<Number>(0.0),
           inv_a1;
           /*--- NOTE: tau denotes the specific volume, i.e. the inverse of the density ---*/
    Number vel2_diesis, p2_diesis,
           tau2L_diesis = static_cast<Number>(0.0),
           tau2R_diesis = static_cast<Number>(0.0),
           inv_a2;

    const auto fact = static_cast<Number>(1.01); // Safety factor
    // Loop to be sure that tau_diesis variables are positive (theorem 3.5, Coquel et al. JCP 2017)
    unsigned int iter = 0;
    while((tau1L_diesis <= static_cast<Number>(0.0) ||
           tau1R_diesis <= static_cast<Number>(0.0)) &&
           iter < 1000) {
      iter++;
      a1 *= fact;

      inv_a1       = static_cast<Number>(1.0)/a1;
      vel1_diesis  = static_cast<Number>(0.5)*(vel1_L_d + vel1_R_d)
                   - static_cast<Number>(0.5)*(p1_R - p1_L)*inv_a1;
      p1_diesis    = static_cast<Number>(0.5)*(p1_R + p1_L)
                   - static_cast<Number>(0.5)*a1*(vel1_R_d - vel1_L_d);
      tau1L_diesis = inv_rho1_L + (vel1_diesis - vel1_L_d)*inv_a1;
      tau1R_diesis = inv_rho1_R - (vel1_diesis - vel1_R_d)*inv_a1;
    }
    if(iter == 1000) {
      std::cerr << "Maximum iterations in Suliciu flux loop: positivity of tau1" << std::endl;
      exit(1);
    }
    iter = 0;
    while((tau2L_diesis <= static_cast<Number>(0.0) ||
           tau2R_diesis <= static_cast<Number>(0.0)) &&
           iter < 1000) {
      iter++;
      a2 *= fact;

      inv_a2       = static_cast<Number>(1.0)/a2;
      vel2_diesis  = static_cast<Number>(0.5)*(vel2_L_d + vel2_R_d)
                   - static_cast<Number>(0.5)*(p2_R - p2_L)*inv_a2;
      p2_diesis    = static_cast<Number>(0.5)*(p2_R + p2_L)
                   - static_cast<Number>(0.5)*a2*(vel2_R_d - vel2_L_d);
      tau2L_diesis = inv_rho2_L + (vel2_diesis - vel2_L_d)*inv_a2;
      tau2R_diesis = inv_rho2_R - (vel2_diesis - vel2_R_d)*inv_a2;
    }
    if(iter == 1000) {
      std::cerr << "Maximum iterations in Suliciu flux loop: positivity of tau2" << std::endl;
      exit(1);
    }
    // Update of a1 and a2 so that a solution for u* surely exists
    Number rhs = static_cast<Number>(0.0),
           sup = static_cast<Number>(0.0),
           inf = static_cast<Number>(0.0);
    const auto mu = static_cast<Number>(0.02);
    Number cLmax, cRmin;
    iter = 0;
    while((rhs - inf <= mu*(sup - inf) || sup - rhs <= mu*(sup - inf)) && iter < 1000) {
      iter++;
      if(vel1_diesis - a1*tau1L_diesis > vel2_diesis - a2*tau2L_diesis &&
         vel1_diesis + a1*tau1R_diesis < vel2_diesis + a2*tau2R_diesis) {
        a1 *= fact;
        inv_a1       = static_cast<Number>(1.0)/a1;
        vel1_diesis  = static_cast<Number>(0.5)*(vel1_L_d + vel1_R_d)
                     - static_cast<Number>(0.5)*(p1_R - p1_L)*inv_a1;
        p1_diesis    = static_cast<Number>(0.5)*(p1_R + p1_L)
                     - static_cast<Number>(0.5)*a1*(vel1_R_d - vel1_L_d);
        tau1L_diesis = inv_rho1_L + (vel1_diesis - vel1_L_d)*inv_a1;
        tau1R_diesis = inv_rho1_R - (vel1_diesis - vel1_R_d)*inv_a1;
      }
      else {
        if(vel2_diesis - a2*tau2L_diesis > vel1_diesis - a1*tau1L_diesis &&
           vel2_diesis + a2*tau2R_diesis < vel1_diesis + a1*tau1R_diesis) {
          a2 *= fact;
          inv_a2       = static_cast<Number>(1.0)/a2;
          vel2_diesis  = static_cast<Number>(0.5)*(vel2_L_d + vel2_R_d)
                       - static_cast<Number>(0.5)*(p2_R - p2_L)*inv_a2;
          p2_diesis    = static_cast<Number>(0.5)*(p2_R + p2_L)
                       - static_cast<Number>(0.5)*a2*(vel2_R_d - vel2_L_d);
          tau2L_diesis = inv_rho2_L + (vel2_diesis - vel2_L_d)*inv_a2;
          tau2R_diesis = inv_rho2_R - (vel2_diesis - vel2_R_d)*inv_a2;
        }
        else {
          a1 *= fact;
          inv_a1       = static_cast<Number>(1.0)/a1;
          vel1_diesis  = static_cast<Number>(0.5)*(vel1_L_d + vel1_R_d)
                       - static_cast<Number>(0.5)*(p1_R - p1_L)*inv_a1;
          p1_diesis    = static_cast<Number>(0.5)*(p1_R + p1_L)
                       - static_cast<Number>(0.5)*a1*(vel1_R_d - vel1_L_d);
          tau1L_diesis = inv_rho1_L + (vel1_diesis - vel1_L_d)*inv_a1;
          tau1R_diesis = inv_rho1_R - (vel1_diesis - vel1_R_d)*inv_a1;

          a2 *= fact;
          inv_a2       = static_cast<Number>(1.0)/a2;
          vel2_diesis  = static_cast<Number>(0.5)*(vel2_L_d + vel2_R_d)
                       - static_cast<Number>(0.5)*(p2_R - p2_L)*inv_a2;
          p2_diesis    = static_cast<Number>(0.5)*(p2_R + p2_L)
                       - static_cast<Number>(0.5)*a2*(vel2_R_d - vel2_L_d);
          tau2L_diesis = inv_rho2_L + (vel2_diesis - vel2_L_d)*inv_a2;
          tau2R_diesis = inv_rho2_R - (vel2_diesis - vel2_R_d)*inv_a2;
        }
      }

      // Compute the rhs of the equation for u*
      rhs = -p1_diesis*(alpha1_R - alpha1_L)
            -p2_diesis*(alpha2_R - alpha2_L);

      // Limits on u* so that the relative Mach number is below one
      cLmax = std::max(vel1_diesis - a1*tau1L_diesis, vel2_diesis - a2*tau2L_diesis);
      cRmin = std::min(vel1_diesis + a1*tau1R_diesis, vel2_diesis + a2*tau2R_diesis);

      // Bounds on the function Psi
      inf = Psi(cLmax, a1, alpha1_L, alpha1_R, vel1_diesis,
                       a2, alpha2_L, alpha2_R, vel2_diesis, tau2L_diesis, tau2R_diesis);
      sup = Psi(cRmin, a1, alpha1_L, alpha1_R, vel1_diesis,
                       a2, alpha2_L, alpha2_R, vel2_diesis, tau2L_diesis, tau2R_diesis);
    }
    if(iter == 1000) {
      std::cerr << "Maximum iterations in Suliciu flux loop: failed to ensure u* exists" << std::endl;
      exit(1);
    }

    // Look for u* in the interval [cLmax, cRmin] such that Psi(u*) = rhs
    const auto uI_star = Newton(rhs, a1, alpha1_L, alpha1_R, vel1_diesis, tau1L_diesis, tau1R_diesis,
                                     a2, alpha2_L, alpha2_R, vel2_diesis, tau2L_diesis, tau2R_diesis,
                                     cLmax, cRmin);

    /*--- Compute the "fluxes" ---*/
    Number alpha1_m, tau1_m, u1_m, p1_m, E1_m,
           alpha1_p, tau1_p, u1_p, p1_p, E1_p,
           alpha2_m, tau2_m, u2_m, p2_m, E2_m, w2_m,
           alpha2_p, tau2_p, u2_p, p2_p, E2_p, w2_p,
           u2_star;
    Riemann_solver_phase_pI(-uI_star,
                            alpha2_L, alpha2_R, inv_rho2_L, inv_rho2_R, vel2_L_d - uI_star, vel2_R_d - uI_star,
                            p2_L, p2_R,
                            E2_L - (vel2_L_d - uI_star)*uI_star - static_cast<Number>(0.5)*uI_star*uI_star,
                            E2_R - (vel2_R_d - uI_star)*uI_star - static_cast<Number>(0.5)*uI_star*uI_star,
                            vel2_diesis - uI_star, tau2L_diesis, tau2R_diesis, a2,
                            alpha2_m, tau2_m, w2_m, p2_m, E2_m,
                            alpha2_p, tau2_p, w2_p, p2_p, E2_p,
                            u2_star);
    u2_m = w2_m + uI_star;
    E2_m += (u2_m - uI_star)*uI_star + static_cast<Number>(0.5)*uI_star*uI_star;
    u2_p = w2_p + uI_star;
    E2_p += (u2_p - uI_star)*uI_star + static_cast<Number>(0.5)*uI_star*uI_star;
    u2_star += uI_star;
    Riemann_solver_phase_vI(static_cast<Number>(0.0),
                            alpha1_L, alpha1_R, inv_rho1_L, inv_rho1_R, vel1_L_d, vel1_R_d, p1_L, p1_R, E1_L, E1_R,
                            a1, uI_star,
                            alpha1_m, tau1_m, u1_m, p1_m, E1_m,
                            alpha1_p, tau1_p, u1_p, p1_p, E1_p);

    /*--- Build the "fluxes" ---*/
    F_minus(Indices::ALPHA1_INDEX) = static_cast<Number>(0.0);

    const auto inv_tau1_m = static_cast<Number>(1.0)/tau1_m;
    const auto inv_tau1_p = static_cast<Number>(1.0)/tau1_p;
    const auto inv_tau2_m = static_cast<Number>(1.0)/tau2_m;
    const auto inv_tau2_p = static_cast<Number>(1.0)/tau2_p;

    F_minus(Indices::ALPHA1_RHO1_INDEX) = alpha1_m*inv_tau1_m*u1_m;
    F_minus(Indices::ALPHA1_RHO1_U1_INDEX + curr_d) = alpha1_m*inv_tau1_m*u1_m*u1_m
                                                    + alpha1_m*p1_m;
    const auto u1_star = uI_star;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      if(d != curr_d) {
        F_minus(Indices::ALPHA1_RHO1_U1_INDEX + d) = static_cast<Number>(0.5)*u1_star*(m1_L + m1_R)
                                                   - static_cast<Number>(0.5)*std::abs(u1_star)*(m1_R - m1_L);
        F_plus(Indices::ALPHA1_RHO1_U1_INDEX + d)  = F_minus(Indices::ALPHA1_RHO1_U1_INDEX + d);
      }
    }
    F_minus(Indices::ALPHA1_RHO1_E1_INDEX) = alpha1_m*inv_tau1_m*E1_m*u1_m
                                  + alpha1_m*p1_m*u1_m;

    F_minus(Indices::ALPHA2_RHO2_INDEX) = alpha2_m*inv_tau2_m*u2_m;
    F_minus(Indices::ALPHA2_RHO2_U2_INDEX + curr_d) = alpha2_m*inv_tau2_m*u2_m*u2_m
                                                    + alpha2_m*p2_m;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      if(d != curr_d) {
        F_minus(Indices::ALPHA2_RHO2_U2_INDEX + d) = static_cast<Number>(0.5)*u2_star*(m2_L + m2_R)
                                                   - static_cast<Number>(0.5)*std::abs(u2_star)*(m2_R - m2_L);
        F_plus(Indices::ALPHA2_RHO2_U2_INDEX + d)  = F_minus(Indices::ALPHA2_RHO2_U2_INDEX + d);
      }
    }
    F_minus(Indices::ALPHA2_RHO2_E2_INDEX) = alpha2_m*inv_tau2_m*E2_m*u2_m
                                  + alpha2_m*p2_m*u2_m;

    F_plus(Indices::ALPHA1_INDEX) = static_cast<Number>(0.0);

    F_plus(Indices::ALPHA1_RHO1_INDEX) = alpha1_p*inv_tau1_p*u1_p;
    F_plus(Indices::ALPHA1_RHO1_U1_INDEX + curr_d) = alpha1_p*inv_tau1_p*u1_p*u1_p
                                                   + alpha1_p*p1_p;
    F_plus(Indices::ALPHA1_RHO1_E1_INDEX) = alpha1_p*inv_tau1_p*E1_p*u1_p
                                          + alpha1_p*p1_p*u1_p;

    F_plus(Indices::ALPHA2_RHO2_INDEX) = alpha2_p*inv_tau2_p*u2_p;
    F_plus(Indices::ALPHA2_RHO2_U2_INDEX + curr_d) = alpha2_p*inv_tau2_p*u2_p*u2_p
                                                   + alpha2_p*p2_p;
    F_plus(Indices::ALPHA2_RHO2_E2_INDEX) = alpha2_p*inv_tau2_p*E2_p*u2_p
                                          + alpha2_p*p2_p*u2_p;

    /*--- Focus on non-conservative term ---*/
    #ifdef ORDER_2
      const auto alpha2_L_order1 = static_cast<Number>(1.0) - alpha1_L_order1;
      const auto alpha2_R_order1 = static_cast<Number>(1.0) - alpha1_R_order1;
      const auto pidxalpha2      = p2_diesis*(alpha2_R_order1 - alpha2_L_order1)
                                 + psi(uI_star, a2, alpha2_L_order1, alpha2_R_order1, vel2_diesis, tau2L_diesis, tau2R_diesis);
    #else
      const auto pidxalpha2 = p2_diesis*(alpha2_R - alpha2_L)
                            + psi(uI_star, a2, alpha2_L, alpha2_R, vel2_diesis, tau2L_diesis, tau2R_diesis);
    #endif

    if(uI_star < static_cast<Number>(0.0)) {
      #ifdef ORDER_2
        F_minus(Indices::ALPHA1_INDEX) -= -uI_star*(alpha1_R_order1 - alpha1_L_order1);
      #else
        F_minus(Indices::ALPHA1_INDEX) -= -uI_star*(alpha1_R - alpha1_L);
      #endif

      F_minus(Indices::ALPHA1_RHO1_U1_INDEX + curr_d) -= -pidxalpha2;
      F_minus(Indices::ALPHA1_RHO1_E1_INDEX) -= -uI_star*pidxalpha2;

      F_minus(Indices::ALPHA2_RHO2_U2_INDEX + curr_d) -= pidxalpha2;
      F_minus(Indices::ALPHA2_RHO2_E2_INDEX) -= uI_star*pidxalpha2;
    }
    else {
      #ifdef ORDER_2
        F_plus(Indices::ALPHA1_INDEX) += -uI_star*(alpha1_R_order1 - alpha1_L_order1);
      #else
        F_plus(Indices::ALPHA1_INDEX) += -uI_star*(alpha1_R - alpha1_L);
      #endif

      F_plus(Indices::ALPHA1_RHO1_U1_INDEX + curr_d) += -pidxalpha2;
      F_plus(Indices::ALPHA1_RHO1_E1_INDEX) += -uI_star*pidxalpha2;

      F_plus(Indices::ALPHA2_RHO2_U2_INDEX + curr_d) += pidxalpha2;
      F_plus(Indices::ALPHA2_RHO2_E2_INDEX) += uI_star*pidxalpha2;
    }

    c = std::max(c, std::max(std::max(std::abs(vel1_L_d - a1*inv_rho1_L), std::abs(vel1_R_d + a1*inv_rho1_R)),
                             std::max(std::abs(vel2_L_d - a2*inv_rho2_L), std::abs(vel2_R_d + a2*inv_rho2_R))));
  }

  // Implement the contribution of the discrete flux for all the directions.
  //
  template<class Field>
  auto RelaxationFlux<Field>::make_flux(Number& c) {
    FluxDefinition<typename Flux<Field>::cfg> discrete_flux;

    // Perform the loop over each dimension to compute the flux contribution
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
         {
           static constexpr int d = decltype(integral_constant_d)::value;

           // Compute now the "discrete" non-conservative flux function
           discrete_flux[d].flux_function = [&](samurai::FluxValuePair<typename Flux<Field>::cfg>& flux,
                                                const StencilData<typename Flux<Field>::cfg>& /*data*/,
                                                const StencilValues<typename Flux<Field>::cfg> field)
                                                {
                                                  FluxValue<typename Flux<Field>::cfg> F_minus,
                                                                                       F_plus;

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

                                                    compute_discrete_flux(qL, qR,
                                                                          field[1](Indices::ALPHA1_INDEX), field[2](Indices::ALPHA1_INDEX),
                                                                          d, F_minus, F_plus, c);
                                                  #else
                                                    // Extract state
                                                    const FluxValue<typename Flux<Field>::cfg>& qL = field[0];
                                                    const FluxValue<typename Flux<Field>::cfg>& qR = field[1];

                                                    compute_discrete_flux(qL, qR, d, F_minus, F_plus, c);
                                                  #endif

                                                  flux[0] = F_minus;
                                                  flux[1] = -F_plus;
                                                };
        }
    );

    auto scheme = make_flux_based_scheme(discrete_flux);
    scheme.set_name("Suliciu");

    return scheme;
  }

  //////////////////////////////////////////////////////////////
  /*---- FOCUS NOW ON THE AUXILIARY FUNCTIONS ---*/
  /////////////////////////////////////////////////////////////

  // Implement M0 function (3.312 Saleh 2012, 3.30 Saleh ESAIM 2019)
  //
  template<class Field>
  template<typename T>
  inline T RelaxationFlux<Field>::M0(const T nu, const T Me) const {
    return static_cast<T>(4.0)/(nu + static_cast<T>(1.0))*
           Me/((static_cast<T>(1.0) + Me*Me)*
               (static_cast<T>(1.0) +
                std::sqrt(std::abs(static_cast<T>(1.0) -
                                   static_cast<T>(4.0)*nu/
                                   ((nu + static_cast<T>(1.0))*(nu + static_cast<T>(1.0)))*
                                   static_cast<T>(4.0)*Me*Me/
                                   ((static_cast<T>(1.0) + Me*Me)*
                                    (static_cast<T>(1.0) + Me*Me))))));
  }

  // Implement psi function (Saleh 2012 ??)
  //
  template<class Field>
  template<typename T>
  inline T RelaxationFlux<Field>::psi(const T u_star, const T a,
                                      const T alphaL, const T alphaR,
                                      const T vel_diesis, const T tauL_diesis, const T tauR_diesis) const {
    if(u_star <= vel_diesis) {
      return a*(alphaL + alphaR)*(u_star - vel_diesis) +
             static_cast<T>(2.0)*a*a*alphaL*tauL_diesis*
             M0(alphaL/alphaR, (vel_diesis - u_star)/(a*tauL_diesis));
    }

    return -psi(-u_star, a, alphaR, alphaL, -vel_diesis, tauR_diesis, tauL_diesis);
  }

  // Implement Psi function (3.3.15 Saleh 2012 ??)
  //
  template<class Field>
  template<typename T>
  inline T RelaxationFlux<Field>::Psi(const T u_star,
                                      const T a1, const T alpha1L, const T alpha1R, const T vel1_diesis,
                                      const T a2, const T alpha2L, const T alpha2R, const T vel2_diesis,
                                      const T tau2L_diesis, const T tau2R_diesis) const {
    return a1*(alpha1L + alpha1R)*(u_star - vel1_diesis) +
           psi(u_star, a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis);
  }

  // Implement the derivative of M0 w.r.t Me for the Newton method
  //
  template<class Field>
  template<typename T>
  inline T RelaxationFlux<Field>::dM0_dMe(const T nu, const T Me) const {
    const T w = (static_cast<T>(1.0) - Me)/
                (static_cast<T>(1.0) + Me);

    return static_cast<T>(4.0)/(nu + static_cast<T>(1.0))*
           w/((static_cast<T>(1.0) + w*w)*(static_cast<T>(1.0) + w*w))*
           (static_cast<T>(1.0) + w)*(static_cast<T>(1.0) + w)/
           (static_cast<T>(1.0) - static_cast<T>(4.0)*nu/((nu + static_cast<T>(1.0))*(nu + static_cast<T>(1.0)))*
                                  (static_cast<T>(1.0) - w*w)*(static_cast<T>(1.0) - w*w)/
                                  ((static_cast<T>(1.0) + w*w)*(static_cast<T>(1.0) + w*w)) +
                                  std::sqrt(std::abs(static_cast<T>(1.0) -
                                                     static_cast<T>(4.0)*nu/
                                                     ((nu + static_cast<T>(1.0))*(nu + static_cast<T>(1.0)))*
                                                     (static_cast<T>(1.0) - w*w)*(static_cast<T>(1.0) - w*w)/
                                                     ((static_cast<T>(1.0) + w*w)*(static_cast<T>(1.0) + w*w)))));
  }

  // Implement the derivative of psi w.r.t. u* for the Newton method
  //
  template<class Field>
  template<typename T>
  inline T RelaxationFlux<Field>::dpsi_dustar(const T u_star, const T a,
                                              const T alphaL, const T alphaR,
                                              const T vel_diesis, const T tauL_diesis, const T tauR_diesis) const {
    if(u_star <= vel_diesis) {
      return a*(alphaL + alphaR) -
             static_cast<T>(2.0)*a*alphaL*
             dM0_dMe(alphaL/alphaR, (vel_diesis - u_star)/(a*tauL_diesis));
    }

    return a*(alphaL + alphaR) -
           static_cast<T>(2.0)*a*alphaR*
           dM0_dMe(alphaR/alphaL, (vel_diesis - u_star)/(a*tauR_diesis));
  }

  // Implement the derivative of Psi w.r.t. u* for the Newton method
  //
  template<class Field>
  template<typename T>
  inline T RelaxationFlux<Field>::dPsi_dustar(const T u_star,
                                              const T a1, const T alpha1L, const T alpha1R,
                                              const T a2, const T alpha2L, const T alpha2R,
                                              const T vel2_diesis, const T tau2L_diesis, const T tau2R_diesis) const {
    return a1*(alpha1L + alpha1R) + dpsi_dustar(u_star, a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis);
  }

  // Newton method to compute u*
  //
  template<class Field>
  template<typename T>
  T RelaxationFlux<Field>::Newton(const T rhs,
                                  const T a1, const T alpha1L, const T alpha1R, const T vel1_diesis, const T tau1L_diesis, const T tau1R_diesis,
                                  const T a2, const T alpha2L, const T alpha2R, const T vel2_diesis, const T tau2L_diesis, const T tau2R_diesis,
                                  const T cLmax, const T cRmin) const {
    if(alpha1L == alpha1R) {
      return vel1_diesis;
    }
    else {
      unsigned int iter = 0;
      const T xl = std::max(vel1_diesis - a1*tau1L_diesis, vel2_diesis - a2*tau2L_diesis);
      const T xr = std::min(vel1_diesis + a1*tau1R_diesis, vel2_diesis + a2*tau2R_diesis);

      T u_star = static_cast<T>(0.5)*(xl + xr);

      T du = -(Psi(u_star, a1, alpha1L, alpha1R, vel1_diesis,
                           a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis) - rhs)/
              (dPsi_dustar(u_star, a1, alpha1L, alpha1R,
                                   a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis));
      du = std::max(std::min(du, static_cast<T>(0.9)*(cRmin - u_star)), static_cast<T>(0.9)*(cLmax - u_star));

      while(iter < max_Newton_iters &&
            //std::abs(Psi(u_star, a1, alpha1L, alpha1R, vel1_diesis,
            //                     a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis) - rhs) > atol_Newton &&
            std::abs(du) > atol_Newton + std::abs(u_star)*rtol_Newton) {
        ++iter;

        u_star += du;

        du = -(Psi(u_star, a1, alpha1L, alpha1R, vel1_diesis,
                           a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis) - rhs)/
              (dPsi_dustar(u_star, a1, alpha1L, alpha1R,
                                   a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis));
        du = std::max(std::min(du, static_cast<T>(0.9)*(cRmin - u_star)), static_cast<T>(0.9)*(cLmax - u_star));
      }

      // Safety check
      if(iter == max_Newton_iters) {
        std::cout << "Newton method not converged in Suliciu-type scheme." << std::endl;
        exit(1);
      }

      return u_star;
    }
  }

  // Riemann solver for the phase associated to the interfacial velocity
  //
  template<class Field>
  template<typename T>
  void RelaxationFlux<Field>::Riemann_solver_phase_vI(const T xi,
                                                      const T alphaL, const T alphaR, const T tauL, const T tauR,
                                                      const T wL, const T wR, const T pL, const T pR, const T EL, const T ER,
                                                      const T a, const T u_star,
                                                      T& alpha_m, T& tau_m, T& w_m, T& pres_m, T& E_m,
                                                      T& alpha_p, T& tau_p, T& w_p, T& pres_p, T& E_p) {
    if(xi < wL - a*tauL) {
      alpha_m = alphaL;
      tau_m   = tauL;
      w_m     = wL;
      pres_m  = pL;
      E_m     = EL;

      alpha_p = alphaL;
      tau_p   = tauL;
      w_p     = wL;
      pres_p  = pL;
      E_p     = EL;
    }
    else {
      if(xi == wL - a*tauL) {
        alpha_m = alphaL;
        tau_m   = tauL;
        w_m     = wL;
        pres_m  = pL;
        E_m     = EL;

        alpha_p = alphaL;
        tau_p   = tauL + static_cast<T>(1.0)/a*(u_star - wL);
        w_p     = u_star;
        pres_p  = pL + a*(wL - u_star);
        E_p     = EL - static_cast<T>(1.0)/a*(pres_p*w_p - pL*wL);
      }
      else {
        if(xi > wL - a*tauL && xi < u_star) {
          alpha_m = alphaL;
          tau_m   = tauL + static_cast<T>(1.0)/a*(u_star - wL);
          w_m     = u_star;
          pres_m  = pL + a*(wL - u_star);
          E_m     = EL - static_cast<T>(1.0)/a*(pres_m*w_m - pL*wL);

          alpha_p = alphaL;
          tau_p   = tauL + static_cast<T>(1.0)/a*(u_star - wL);
          w_p     = u_star;
          pres_p  = pL + a*(wL - u_star);
          E_p     = EL - static_cast<T>(1.0)/a*(pres_p*w_p - pL*wL);
        }
        else {
          if(xi == u_star) {
            alpha_m = alphaL;
            tau_m   = tauL + static_cast<T>(1.0)/a*(u_star - wL);
            w_m     = u_star;
            pres_m  = pL + a*(wL - u_star);
            E_m     = EL - static_cast<T>(1.0)/a*(pres_m*w_m - pL*wL);

            alpha_p = alphaR;
            tau_p   = tauR - static_cast<T>(1.0)/a*(u_star - wR);
            w_p     = u_star;
            pres_p  = pR - a*(wR - u_star);
            E_p     = ER + static_cast<T>(1.0)/a*(pres_p*w_p - pR*wR);
          }
          else {
            if(xi > u_star && xi < wR + a*tauR)	{
              alpha_m = alphaR;
              tau_m   = tauR - static_cast<T>(1.0)/a*(u_star - wR);
              w_m     = u_star;
              pres_m  = pR - a*(wR - u_star);
              E_m     = ER + static_cast<T>(1.0)/a*(pres_m*w_m - pR*wR);

              alpha_p = alphaR;
              tau_p   = tauR - static_cast<T>(1.0)/a*(u_star - wR);
              w_p     = u_star;
              pres_p  = pR - a*(wR - u_star);
              E_p     = ER + static_cast<T>(1.0)/a*(pres_p*w_p - pR*wR);
            }
            else {
              if(xi == wR + a*tauR) {
                alpha_m = alphaR;
                tau_m   = tauR - static_cast<T>(1.0)/a*(u_star - wR);
                w_m     = u_star;
                pres_m  = pR - a*(wR - u_star);
                E_m     = ER + static_cast<T>(1.0)/a*(pres_m*w_m - pR*wR);

                alpha_p = alphaR;
                tau_p   = tauR;
                w_p     = wR;
                pres_p  = pR;
                E_p     = ER;
              }
              else {
                alpha_m = alphaR;
                tau_m   = tauR;
                w_m     = wR;
                pres_m  = pR;
                E_m     = ER;

                alpha_p = alphaR;
                tau_p   = tauR;
                w_p     = wR;
                pres_p  = pR;
                E_p     = ER;
              }
            }
          }
        }
      }
    }
  }

  // Riemann solver for the phase associated to the interfacial pressure
  //
  template<class Field>
  template<typename T>
  void RelaxationFlux<Field>::Riemann_solver_phase_pI(const T xi,
                                                      const T alphaL, const T alphaR, const T tauL, const T tauR,
                                                      const T wL, const T wR, const T pL, const T pR, const T EL, const T ER,
                                                      const T w_diesis, const T tauL_diesis, const T tauR_diesis, const T a,
                                                      T& alpha_m, T& tau_m, T& w_m, T& pres_m, T& E_m,
                                                      T& alpha_p, T& tau_p, T& w_p, T& pres_p, T& E_p,
                                                      T& w_star) {
    const T nu  = alphaL/alphaR;
    const T ML  = wL/(a*tauL);
    const T MdL = w_diesis/(a*tauL_diesis);

    T M;
    T Mzero;
    const T mu = static_cast<T>(0.9);
    const T t  = tauR_diesis/tauL_diesis;

    if(w_diesis > static_cast<T>(0.0)) {
      if(ML < static_cast<T>(1.0)) {
         /*--- Configuration <1,2> subsonic.
               Computation of M which parametrises the whole solution ---*/
        Mzero = static_cast<T>(4.0)/(nu + static_cast<T>(1.0))*
                MdL/((static_cast<T>(1.0) + MdL*MdL)*
                     (static_cast<T>(1.0) +
                      std::sqrt(std::abs(static_cast<T>(1.0) -
                                         static_cast<T>(4.0)*nu/((nu + static_cast<T>(1.0))*(nu + static_cast<T>(1.0)))*
                                         static_cast<T>(4.0)*MdL*MdL/((static_cast<T>(1.0) + MdL*MdL)*(static_cast<T>(1.0) + MdL*MdL))))));

        if(mu*tauR_diesis <= tauR_diesis + tauL_diesis*(MdL + nu*Mzero)/(1.+nu*Mzero)) {
          M = Mzero;
        }
        else {
          /*--- Add the required amount of energy dissipation ---*/
          M = static_cast<T>(1.0)/nu*(MdL + t*(static_cast<T>(1.0) - mu))/
                                     (static_cast<T>(1.0) - t*(static_cast<T>(1.0) - mu));
        }
      }

      if(xi < wL - a*tauL) {
        alpha_m = alphaL;
        tau_m   = tauL;
        w_m     = wL;
        pres_m  = pL;
        E_m     = EL;

        alpha_p = alphaL;
        tau_p   = tauL;
        w_p     = wL;
        pres_p  = pL;
        E_p     = EL;
      }
      else {
        if(xi == wL - a*tauL) {
          alpha_m = alphaL;
          tau_m   = tauL;
          w_m     = wL;
          pres_m  = pL;
          E_m     = EL;

          alpha_p = alphaL;
          tau_p   = tauL_diesis*(static_cast<T>(1.0) - MdL)/(static_cast<T>(1.0) - M);
          w_p     = a*M*tau_p;
          pres_p  = pL + a*(wL - w_p);
          E_p     = EL - static_cast<T>(1.0)/a*(pres_p*w_p - pL*wL);
        }
        else {
          if(xi > wL - a*tauL && xi < static_cast<T>(0.0)) {
            alpha_m = alphaL;
            tau_m   = tauL_diesis*(static_cast<T>(1.0) - MdL)/(static_cast<T>(1.0) - M);
            w_m     = a*M*tau_m;
            pres_m  = pL + a*(wL - w_m);
            E_m     = EL - static_cast<T>(1.0)/a*(pres_m*w_m - pL*wL);

            alpha_p = alphaL;
            tau_p   = tauL_diesis*(static_cast<T>(1.0) - MdL)/(static_cast<T>(1.0) - M);
            w_p     = a*M*tau_p;
            pres_p  = pL + a*(wL - w_p);
            E_p     = EL - static_cast<T>(1.0)/a*(pres_p*w_p - pL*wL);
          }
          else {
            if(xi == static_cast<T>(0.0)) {
              alpha_m = alphaL;
              tau_m   = tauL_diesis*(static_cast<T>(1.0) - MdL)/(static_cast<T>(1.0) - M);
              w_m     = a*M*tau_m;
              pres_m  = pL + a*(wL - w_m);
              E_m     = EL - static_cast<T>(1.0)/a*(pres_m*w_m - pL*wL);

              alpha_p = alphaR;
              tau_p   = tauL_diesis*(static_cast<T>(1.0) + MdL)/(static_cast<T>(1.0) + nu*M);
              w_p     = nu*a*M*tau_p;
              pres_p  = pL + a*a*(tauL - tau_p);
              E_p     = E_m - (pres_p*tau_p - pres_m*tau_m);
            }
            else {
              if(xi > static_cast<T>(0.0) &&
                 xi < nu*a*M*tauL_diesis*(static_cast<T>(1.0) + MdL)/(static_cast<T>(1.0) + nu*M)) {
                /*--- Computations of E_m and E_p ---*/
                alpha_m = alphaL;
                tau_m   = tauL_diesis*(static_cast<T>(1.0) - MdL)/(static_cast<T>(1.0) - M);
                w_m     = a*M*tau_m;
                pres_m  = pL + a*(wL - w_m);
                E_m     = EL - static_cast<T>(1.0)/a*(pres_m*w_m - pL*wL);

                alpha_p = alphaR;
                tau_p   = tauL_diesis*(static_cast<T>(1.0) + MdL)/(static_cast<T>(1.0) + nu*M);
                w_p     = nu*a*M*tau_p;
                pres_p  = pL + a*a*(tauL - tau_p);
                E_p     = E_m - (pres_p*tau_p - pres_m*tau_m);

                /*--- Compute the real states ---*/
                alpha_m = alphaR;
                tau_m   = tauL_diesis*(static_cast<T>(1.0) + MdL)/(static_cast<T>(1.0) + nu*M);
                w_m     = nu*a*M*tau_m;
                pres_m  = pL + a*a*(tauL - tau_m);
                E_m     = E_p;

                alpha_p = alphaR;
                tau_p   = tauL_diesis*(static_cast<T>(1.0) + MdL)/(static_cast<T>(1.0) + nu*M);
                w_p     = nu*a*M*tau_p;
                pres_p  = pL + a*a*(tauL - tau_p);
              }
              else {
                if(xi == nu*a*M*tauL_diesis*(static_cast<T>(1.0) + MdL)/(static_cast<T>(1.0) + nu*M)) {
                  /*--- Computations of E_m and E_p ---*/
                  alpha_m = alphaL;
                  tau_m   = tauL_diesis*(static_cast<T>(1.0) - MdL)/(static_cast<T>(1.0) - M);
                  w_m     = a*M*tau_m;
                  pres_m  = pL + a*(wL - w_m);
                  E_m     = EL - static_cast<T>(1.0)/a*(pres_m*w_m - pL*wL);

                  alpha_p = alphaR;
                  tau_p   = tauL_diesis*(static_cast<T>(1.0) + MdL)/(static_cast<T>(1.0) + nu*M);
                  w_p     = nu*a*M*tau_p;
                  pres_p  = pL + a*a*(tauL - tau_p);
                  E_p     = E_m - (pres_p*tau_p - pres_m*tau_m);

                  /*--- Compute the real states ---*/
                  alpha_m = alphaR;
                  tau_m   = tauL_diesis*(static_cast<T>(1.0) + MdL)/(static_cast<T>(1.0) + nu*M);
                  w_m     = nu*a*M*tau_m;
                  pres_m  = pL + a*a*(tauL - tau_m);
                  E_m     = E_p;

                  alpha_p = alphaR;
                  tau_p   = tauR_diesis + tauL_diesis*(MdL - nu*M)/(static_cast<T>(1.0) + nu*M);
                  w_p     = nu*a*M*tauL_diesis*(static_cast<T>(1.0) + MdL)/(static_cast<T>(1.0) + nu*M);
                  pres_p  = pR - a*(wR - w_p);
                  E_p     = ER - static_cast<T>(1.0)/a*(pR*wR - pres_p*w_p);
                }
                else {
                  if(xi > nu*a*M*tauL_diesis*(static_cast<T>(1.0) + MdL)/(static_cast<T>(1.0) + nu*M) &&
                     xi < wR + a*tauR) {
                    alpha_m = alphaR;
                    tau_m   = tauR_diesis + tauL_diesis*(MdL - nu*M)/(static_cast<T>(1.0) + nu*M);
                    w_m     = nu*a*M*tauL_diesis*(static_cast<T>(1.0) + MdL)/(static_cast<T>(1.0) + nu*M);
                    pres_m  = pR - a*(wR - w_m);
                    E_m     = ER - static_cast<T>(1.0)/a*(pR*wR - pres_m*w_m);

                    alpha_p = alphaR;
                    tau_p   = tauR_diesis + tauL_diesis*(MdL - nu*M)/(static_cast<T>(1.0) + nu*M);
                    w_p     = nu*a*M*tauL_diesis*(static_cast<T>(1.0) + MdL)/(static_cast<T>(1.0) + nu*M);
                    pres_p  = pR - a*(wR - w_p);
                    E_p     = ER + static_cast<T>(1.0)/a*(pres_p*w_p - pR*wR);
                  }
                  else {
                    if(xi == wR + a*tauR) {
                      alpha_m = alphaR;
                      tau_m   = tauR_diesis + tauL_diesis*(MdL - nu*M)/(static_cast<T>(1.0) + nu*M);
                      w_m     = nu*a*M*tauL_diesis*(static_cast<T>(1.0) + MdL)/(static_cast<T>(1.0) + nu*M);
                      pres_m  = pR - a*(wR - w_m);
                      E_m     = ER + static_cast<T>(1.0)/a*(pres_m*w_m - pR*wR);

                      alpha_p = alphaR;
                      tau_p   = tauR;
                      w_p     = wR;
                      pres_p  = pR;
                      E_p     = ER;
                    }
                    else {
                      alpha_m = alphaR;
                      tau_m   = tauR;
                      w_m     = wR;
                      pres_m  = pR;
                      E_m     = ER;

                      alpha_p = alphaR;
                      tau_p   = tauR;
                      w_p     = wR;
                      pres_p  = pR;
                      E_p     = ER;
                    }
                  }
                }
              }
            }
          }
        }
      }
      w_star = nu*a*M*tauL_diesis*(static_cast<T>(1.0) + MdL)/(static_cast<T>(1.0) + nu*M);
    }
    else {
      if(w_diesis < static_cast<T>(0.0)) {
        Riemann_solver_phase_pI(-xi,
                                alphaR, alphaL, tauR, tauL, -wR, -wL, pR, pL, ER, EL,
                                -w_diesis, tauR_diesis, tauL_diesis, a,
                                alpha_p, tau_p, w_p, pres_p, E_p,
                                alpha_m, tau_m, w_m, pres_m, E_m,
                                w_star);
        w_m = -w_m;
        w_p = -w_p;
        w_star = -w_star;
      }
      else {
        w_star = static_cast<T>(0.0);
        if(xi < wL - a*tauL) {
          alpha_m = alphaL;
          tau_m   = tauL;
          w_m     = wL;
          pres_m  = pL;
          E_m     = EL;

          alpha_p = alphaL;
          tau_p   = tauL;
          w_p     = wL;
          pres_p  = pL;
          E_p     = EL;
        }
        else {
          if(xi == wL - a*tauL) {
            alpha_m = alphaL;
            tau_m   = tauL;
            w_m     = wL;
            pres_m  = pL;
            E_m     = EL;

            alpha_p = alphaL;
            tau_p   = tauL_diesis;
            w_p     = static_cast<T>(0.0);
            pres_p  = pL + a*(wL - w_p);
            E_p     = EL - static_cast<T>(1.0)/a*(pres_p*w_p - pL*wL);
          }
          else {
            if(xi > wL - a*tauL && xi < static_cast<T>(0.0)) {
              alpha_m = alphaL;
              tau_m   = tauL_diesis;
              w_m     = static_cast<T>(0.0);
              pres_m  = pL + a*(wL - w_m);
              E_m     = EL - static_cast<T>(1.0)/a*(pres_m*w_m - pL*wL);

              alpha_p = alphaL;
              tau_p   = tauL_diesis;
              w_p     = static_cast<T>(0.0);
              pres_p  = pL + a*(wL - w_p);
              E_p     = EL - static_cast<T>(1.0)/a*(pres_p*w_p - pL*wL);
            }
            else {
              if(xi == static_cast<T>(0.0)) {
                alpha_m = alphaL;
                tau_m   = tauL_diesis;
                w_m     = static_cast<T>(0.0);
                pres_m  = pL + a*(wL - w_m);
                E_m     = EL - static_cast<T>(1.0)/a*(pres_m*w_m - pL*wL);

                alpha_p = alphaR;
                tau_p   = tauR_diesis;
                w_p     = static_cast<T>(0.0);
                pres_p  = pR - a*(wR - w_p);
                E_p     = ER + static_cast<T>(1.0)/a*(pres_p*w_p - pR*wR);
              }
              else {
                if(xi > static_cast<T>(0.0) && xi < wR + a*tauR) {
                  alpha_m = alphaR;
                  tau_m   = tauR_diesis;
                  w_m     = static_cast<T>(0.0);
                  pres_m  = pR - a*(wR - w_m);
                  E_m     = ER + static_cast<T>(1.0)/a*(pres_m*w_m - pR*wR);

                  alpha_p = alphaR;
                  tau_p   = tauR_diesis;
                  w_p     = static_cast<T>(0.0);
                  pres_p  = pR - a*(wR-w_p);
                  E_p     = ER + static_cast<T>(1.0)/a*(pres_p*w_p - pR*wR);
                }
                else {
                  if(xi == wR + a*tauR) {
                    alpha_m = alphaR;
                    tau_m   = tauR_diesis;
                    w_m     = static_cast<T>(0.0);
                    pres_m  = pR - a*(wR - w_m);
                    E_m     = ER + static_cast<T>(1.0)/a*(pres_m*w_m - pR*wR);

                    alpha_p = alphaR;
                    tau_p   = tauR;
                    w_p     = wR;
                    pres_p  = pR;
                    E_p     = ER;
                  }
                  else {
                    alpha_m = alphaR;
                    tau_m   = tauR;
                    w_m     = wR;
                    pres_m  = pR;
                    E_m     = ER;

                    alpha_p = alphaR;
                    tau_p   = tauR;
                    w_p     = wR;
                    pres_p  = pR;
                    E_p     = ER;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

} // end of namespace

#endif
