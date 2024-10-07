// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#ifndef Exact_Godunov_flux_hpp
#define Exact_Godunov_flux_hpp

#include "flux_base.hpp"

namespace samurai {
  using namespace EquationData;

  /**
    * Implementation of a Godunov flux
    */
  template<class Field>
  class GodunovFlux: public Flux<Field> {
  public:
    GodunovFlux(const LinearizedBarotropicEOS<>& EOS_phase1,
                const LinearizedBarotropicEOS<>& EOS_phase2,
                const double sigma_,
                const double sigma_relax_,
                const double eps_,
                const double mod_grad_alpha1_bar_min_); // Constructor which accepts in inputs the equations of state of the two phases

    #ifdef ORDER_2
      template<typename Gradient, typename Field_Scalar>
      auto make_two_scale_capillarity(const Gradient& grad_alpha1_bar,
                                      const Field_Scalar& H); // Compute the flux over all the directions
    #else
      template<typename Gradient>
      auto make_two_scale_capillarity(const Gradient& grad_alpha1_bar); // Compute the flux over all the directions
    #endif

  private:
    template<typename Gradient>
    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                               const FluxValue<typename Flux<Field>::cfg>& qR,
                                                               const std::size_t curr_d,
                                                               const Gradient& grad_alpha1_barL,
                                                               const Gradient& grad_alpha1_barR,
                                                               const bool is_discontinuous); // Godunov flux for the along direction curr_d

    void solve_alpha1_d_fan(const typename Field::value_type rhs,
                            typename Field::value_type& alpha1_d); // Newton method to compute alpha1_d for the fan

    void solve_p_star(const FluxValue<typename Flux<Field>::cfg>& qL,
                      const FluxValue<typename Flux<Field>::cfg>& qR,
                      const typename Field::value_type dvel_d,
                      const typename Field::value_type vel_d_L,
                      const typename Field::value_type p0_L,
                      const typename Field::value_type p0_R,
                      typename Field::value_type& p_star); // Newton method to compute p* in the exact solver for the hyperbolic part
  };

  // Constructor derived from the base class
  //
  template<class Field>
  GodunovFlux<Field>::GodunovFlux(const LinearizedBarotropicEOS<>& EOS_phase1,
                                  const LinearizedBarotropicEOS<>& EOS_phase2,
                                  const double sigma_,
                                  const double sigma_relax_,
                                  const double eps_,
                                  const double grad_alpha1_bar_min_):
    Flux<Field>(EOS_phase1, EOS_phase2, sigma_, sigma_relax_, eps_, grad_alpha1_bar_min_) {}

  // Compute small-scale volume fraction for the fan through Newton-Rapson method
  //
  template<class Field>
  void GodunovFlux<Field>::solve_alpha1_d_fan(const typename Field::value_type rhs,
                                              typename Field::value_type& alpha1_d) {
    const double tol            = 1e-12; // Tolerance of the Newton method
    const double lambda         = 0.9;   // Parameter for bound preserving strategy
    const std::size_t max_iters = 60;    // Maximum number of Newton iterations

    typename Field::value_type dalpha1_d = std::numeric_limits<typename Field::value_type>::infinity();

    const auto alpha1_d_0 = alpha1_d;
    auto F_alpha1_d       = 1.0/(1.0 - alpha1_d) + std::log((alpha1_d/alpha1_d_0)*((1.0 - alpha1_d_0)/(1.0 - alpha1_d)));

    // Loop of Newton method
    std::size_t Newton_iter = 0;
    while(Newton_iter < max_iters && alpha1_d > this->eps && 1.0 - alpha1_d > this->eps &&
          std::abs(F_alpha1_d - rhs) > tol*std::abs(rhs) && std::abs(dalpha1_d)/alpha1_d > tol) {
      Newton_iter++;

      // Unmodified Newton-Rapson increment
      auto dF_dalpha1_d = 1.0/((1.0 - alpha1_d)*(1.0 - alpha1_d)*alpha1_d);
      dalpha1_d         = -(F_alpha1_d - rhs)/dF_dalpha1_d;

      // Bound preserving increment
      dalpha1_d = (dalpha1_d < 0.0) ? std::max(dalpha1_d, -lambda*alpha1_d) : std::min(dalpha1_d, lambda*(1.0 - alpha1_d));

      if(alpha1_d + dalpha1_d < 0.0 || alpha1_d + dalpha1_d > 1.0) {
        std::cerr << "Bounds exceeding value for small-scale volume fraction in the Newton method at fan" << std::endl;
        exit(1);
      }
      else {
        alpha1_d += dalpha1_d;
      }

      // Newton cycle diverged
      if(Newton_iter == max_iters) {
        std::cout << "Netwon method not converged to compute small-scale volume fraction in the fan" << std::endl;
        exit(1);
      }

      // Update function for which we seek the zero
      F_alpha1_d = 1.0/(1.0 - alpha1_d) + std::log((alpha1_d/alpha1_d_0)*((1.0 - alpha1_d_0)/(1.0 - alpha1_d)));
    }
  }

  // Compute p* through Newton-Rapson method
  //
  template<class Field>
  void GodunovFlux<Field>::solve_p_star(const FluxValue<typename Flux<Field>::cfg>& qL,
                                        const FluxValue<typename Flux<Field>::cfg>& qR,
                                        const typename Field::value_type dvel_d,
                                        const typename Field::value_type vel_d_L,
                                        const typename Field::value_type p0_L,
                                        const typename Field::value_type p0_R,
                                        typename Field::value_type& p_star) {
    const double tol            = 1e-8; // Tolerance of the Newton method
    const double lambda         = 0.9;  // Parameter for bound preserving strategy
    const std::size_t max_iters = 100;  // Maximum number of Newton iterations

    typename Field::value_type dp_star = std::numeric_limits<typename Field::value_type>::infinity();

    // Left state useful variables
    const auto rho_L        = qL(M1_INDEX) + qL(M2_INDEX) + qL(M1_D_INDEX);
    const auto alpha1_bar_L = qL(RHO_ALPHA1_BAR_INDEX)/rho_L;
    const auto alpha1_L     = alpha1_bar_L*(1.0 - qL(ALPHA1_D_INDEX));
    const auto rho1_L       = (alpha1_L > this->eps) ? qL(M1_INDEX)/alpha1_L : nan("");
    const auto alpha2_L     = 1.0 - alpha1_L - qL(ALPHA1_D_INDEX);
    const auto rho2_L       = (alpha2_L > this->eps) ? qL(M2_INDEX)/alpha2_L : nan("");
    const auto c_squared_L  = qL(M1_INDEX)*this->phase1.c_value(rho1_L)*this->phase1.c_value(rho1_L)
                            + qL(M2_INDEX)*this->phase2.c_value(rho2_L)*this->phase2.c_value(rho2_L);
    const auto c_L          = std::sqrt(c_squared_L/rho_L)/(1.0 - qL(ALPHA1_D_INDEX));
    const auto p_bar_L      = (alpha1_L > this->eps && alpha2_L > this->eps) ?
                              alpha1_bar_L*this->phase1.pres_value(rho1_L) + (1.0 - alpha1_bar_L)*this->phase2.pres_value(rho2_L) :
                              ((alpha1_L < this->eps) ? this->phase2.pres_value(rho2_L) : this->phase1.pres_value(rho1_L));

    // Right state useful variables
    const auto rho_R        = qR(M1_INDEX) + qR(M2_INDEX) + qR(M1_D_INDEX);
    const auto alpha1_bar_R = qR(RHO_ALPHA1_BAR_INDEX)/rho_R;
    const auto alpha1_R     = alpha1_bar_R*(1.0 - qR(ALPHA1_D_INDEX));
    const auto rho1_R       = (alpha1_R > this->eps) ? qR(M1_INDEX)/alpha1_R : nan("");
    const auto alpha2_R     = 1.0 - alpha1_R - qR(ALPHA1_D_INDEX);
    const auto rho2_R       = (alpha2_R > this->eps) ? qR(M2_INDEX)/alpha2_R : nan("");
    const auto c_squared_R  = qR(M1_INDEX)*this->phase1.c_value(rho1_R)*this->phase1.c_value(rho1_R)
                            + qR(M2_INDEX)*this->phase2.c_value(rho2_R)*this->phase2.c_value(rho2_R);
    const auto c_R          = std::sqrt(c_squared_R/rho_R)/(1.0 - qR(ALPHA1_D_INDEX));

    const auto p_bar_R      = (alpha1_R > this->eps && alpha2_R > this->eps) ?
                              alpha1_bar_R*this->phase1.pres_value(rho1_R) + (1.0 - alpha1_bar_R)*this->phase2.pres_value(rho2_R) :
                              ((alpha1_R < this->eps) ? this->phase2.pres_value(rho2_R) : this->phase1.pres_value(rho1_R));

    if(p_star <= p0_L || p_bar_L <= p0_L) {
      std::cerr << "Non-admissible value for the pressure at the beginning of the Newton moethod to compute p* in Godunov solver" << std::endl;
      exit(1);
    }

    auto F_p_star = dvel_d;
    if(p_star <= p_bar_L) {
      F_p_star += c_L*(1.0 - qL(ALPHA1_D_INDEX))*std::log((p_bar_L - p0_L)/(p_star - p0_L));
    }
    else {
      F_p_star -= std::sqrt(1.0 - qL(ALPHA1_D_INDEX))*(p_star - p_bar_L)/std::sqrt(rho_L*(p_star - p0_L));
    }
    if(p_star <= p_bar_R) {
      F_p_star += c_R*(1.0 - qR(ALPHA1_D_INDEX))*std::log((p_bar_R - p0_R)/(p_star - p0_R));
    }
    else {
      F_p_star -= std::sqrt(1.0 - qR(ALPHA1_D_INDEX))*(p_star - p_bar_R)/std::sqrt(rho_R*(p_star - p0_R));
    }

    // Loop of Newton method
    std::size_t Newton_iter = 0;
    while(Newton_iter < max_iters && std::abs(F_p_star) > tol*std::abs(vel_d_L) && std::abs(dp_star/p_star) > tol) {
      Newton_iter++;

      // Unmodified Newton-Rapson increment
      typename Field::value_type dF_p_star;
      if(p_star <= p_bar_L) {
        dF_p_star = c_L*(1.0 - qL(ALPHA1_D_INDEX))/(p0_L - p_star);
      }
      else {
        dF_p_star = std::sqrt(1.0 - qL(ALPHA1_D_INDEX))*(2.0*p0_L - p_star - p_bar_L)/
                    (2.0*(p_star - p0_L)*std::sqrt(rho_L*(p_star - p0_L)));
      }
      if(p_star <= p_bar_R) {
        dF_p_star += c_R*(1.0 - qR(ALPHA1_D_INDEX))/(p0_R - p_star);
      }
      else {
        dF_p_star += std::sqrt(1.0 - qR(ALPHA1_D_INDEX))*(2.0*p0_R - p_star - p_bar_R)/
                     (2.0*(p_star - p0_R)*std::sqrt(rho_R*(p_star - p0_R)));
      }
      dp_star = -F_p_star/dF_p_star;

      // Bound preserving increment
      dp_star = std::max(dp_star, lambda*(std::max(p0_L, p0_R) - p_star));

      if(p_star + dp_star <= p0_L) {
        std::cerr << "Non-admissible value for the pressure in the Newton moethod to compute p* in Godunov solver" << std::endl;
        exit(1);
      }
      else {
        p_star += dp_star;
      }

      // Newton cycle diverged
      if(Newton_iter == max_iters) {
        std::cout << "Netwon method not converged to compute p* in the Godunov solver" << std::endl;
        exit(1);
      }

      // Update function for which we seek the zero
      F_p_star = dvel_d;
      if(p_star <= p_bar_L) {
        F_p_star += c_L*(1.0 - qL(ALPHA1_D_INDEX))*std::log((p_bar_L - p0_L)/(p_star - p0_L));
      }
      else {
        F_p_star -= std::sqrt(1.0 - qL(ALPHA1_D_INDEX))*(p_star - p_bar_L)/std::sqrt(rho_L*(p_star - p0_L));
      }
      if(p_star <= p_bar_R) {
        F_p_star += c_R*(1.0 - qR(ALPHA1_D_INDEX))*std::log((p_bar_R - p0_R)/(p_star - p0_R));
      }
      else {
        F_p_star -= std::sqrt(1.0 - qR(ALPHA1_D_INDEX))*(p_star - p_bar_R)/std::sqrt(rho_R*(p_star - p0_R));
      }
    }
  }

  // Implementation of a Godunov flux
  //
  template<class Field>
  template<typename Gradient>
  FluxValue<typename Flux<Field>::cfg> GodunovFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                                                 const FluxValue<typename Flux<Field>::cfg>& qR,
                                                                                 const std::size_t curr_d,
                                                                                 const Gradient& grad_alpha1_barL,
                                                                                 const Gradient& grad_alpha1_barR,
                                                                                 const bool is_discontinuous) {
    // Compute the intermediate state (either shock or rarefaction)
    FluxValue<typename Flux<Field>::cfg> q_star = qL;

    if(is_discontinuous) {
      // Left state useful variables
      const auto rho_L        = qL(M1_INDEX) + qL(M2_INDEX) + qL(M1_D_INDEX);
      const auto vel_d_L      = qL(RHO_U_INDEX + curr_d)/rho_L;
      const auto alpha1_bar_L = qL(RHO_ALPHA1_BAR_INDEX)/rho_L;
      const auto alpha1_L     = alpha1_bar_L*(1.0 - qL(ALPHA1_D_INDEX));
      const auto rho1_L       = (alpha1_L > this->eps) ? qL(M1_INDEX)/alpha1_L : nan("");
      const auto alpha2_L     = 1.0 - alpha1_L - qL(ALPHA1_D_INDEX);
      const auto rho2_L       = (alpha2_L > this->eps) ? qL(M2_INDEX)/alpha2_L : nan("");
      const auto c_squared_L  = qL(M1_INDEX)*this->phase1.c_value(rho1_L)*this->phase1.c_value(rho1_L)
                              + qL(M2_INDEX)*this->phase2.c_value(rho2_L)*this->phase2.c_value(rho2_L);
      const auto c_L          = std::sqrt(c_squared_L/rho_L)/(1.0 - qL(ALPHA1_D_INDEX));

      // Right state useful variables
      const auto rho_R        = qR(M1_INDEX) + qR(M2_INDEX) + qR(M1_D_INDEX);
      const auto vel_d_R      = qR(RHO_U_INDEX + curr_d)/rho_R;
      const auto alpha1_bar_R = qR(RHO_ALPHA1_BAR_INDEX)/rho_R;
      const auto alpha1_R     = alpha1_bar_R*(1.0 - qR(ALPHA1_D_INDEX));
      const auto rho1_R       = (alpha1_R > this->eps) ? qR(M1_INDEX)/alpha1_R : nan("");
      const auto alpha2_R     = 1.0 - alpha1_R - qR(ALPHA1_D_INDEX);
      const auto rho2_R       = (alpha2_R > this->eps) ? qR(M2_INDEX)/alpha2_R : nan("");
      const auto c_squared_R  = qR(M1_INDEX)*this->phase1.c_value(rho1_R)*this->phase1.c_value(rho1_R)
                              + qR(M2_INDEX)*this->phase2.c_value(rho2_R)*this->phase2.c_value(rho2_R);
      const auto c_R          = std::sqrt(c_squared_R/rho_R)/(1.0 - qR(ALPHA1_D_INDEX));

      // Compute p*
      const auto p_bar_L = (alpha1_L > this->eps && alpha2_L > this->eps) ?
                            alpha1_bar_L*this->phase1.pres_value(rho1_L) + (1.0 - alpha1_bar_L)*this->phase2.pres_value(rho2_L) :
                           ((alpha1_L < this->eps) ? this->phase2.pres_value(rho2_L) : this->phase1.pres_value(rho1_L));
      const auto p_bar_R = (alpha1_R > this->eps && alpha2_R > this->eps) ?
                            alpha1_bar_R*this->phase1.pres_value(rho1_R) + (1.0 - alpha1_bar_R)*this->phase2.pres_value(rho2_R) :
                           ((alpha1_R < this->eps) ? this->phase2.pres_value(rho2_R) : this->phase1.pres_value(rho1_R));

      const auto p0_L = p_bar_L - rho_L*c_L*c_L;
      const auto p0_R = p_bar_R - rho_R*c_R*c_R;

      auto p_star = std::max(0.5*(p_bar_L + p_bar_R),
                             std::max(p0_L, p0_R) + 0.1*std::abs(std::max(p0_L, p0_R)));
      solve_p_star(qL, qR, vel_d_L - vel_d_R, vel_d_L, p0_L, p0_R, p_star);

      // Compute u*
      const auto u_star = (p_star <= p_bar_L) ? vel_d_L + c_L*(1.0 - qL(ALPHA1_D_INDEX))*std::log((p_bar_L - p0_L)/(p_star - p0_L)) :
                                                vel_d_L - std::sqrt(1.0 - qL(ALPHA1_D_INDEX))*(p_star - p_bar_L)/std::sqrt(rho_L*(p_star - p0_L));

      // Left "connecting state"
      if(u_star > 0.0) {
        // 1-wave left shock
        if(p_star > p_bar_L) {
          const auto r = 1.0 + (1.0 - qL(ALPHA1_D_INDEX))/
                               (qL(ALPHA1_D_INDEX) + (rho_L*c_L*c_L*(1.0 - qL(ALPHA1_D_INDEX)))/(p_star - p_bar_L));

          const auto m1_L_star       = qL(M1_INDEX)*r;
          const auto m2_L_star       = qL(M2_INDEX)*r;
          const auto alpha1_d_L_star = qL(ALPHA1_D_INDEX)*r;
          const auto m1_d_L_star     = (qL(ALPHA1_D_INDEX) > this->eps) ? alpha1_d_L_star*(qL(M1_D_INDEX)/qL(ALPHA1_D_INDEX)) : 0.0;
          const auto Sigma_d_L_star  = (qL(ALPHA1_D_INDEX) > this->eps) ? alpha1_d_L_star*(qL(SIGMA_D_INDEX)/qL(ALPHA1_D_INDEX)) : 0.0;
          const auto rho_L_star      = m1_L_star + m2_L_star + m1_d_L_star;

          auto s_L = nan("");
          if(r > 1) {
            s_L = u_star + (vel_d_L - u_star)/(1.0 - r);
          }
          else if (r == 1) {
            s_L = u_star + (vel_d_L - u_star)*(-std::numeric_limits<double>::infinity());
          }

          // If left of left shock, q* = qL, already assigned.
          // If right of left shock, is the computed state
          if(!std::isnan(s_L) && s_L < 0.0) {
            q_star(M1_INDEX)             = m1_L_star;
            q_star(M2_INDEX)             = m2_L_star;
            q_star(M1_D_INDEX)           = m1_d_L_star;
            q_star(ALPHA1_D_INDEX)       = alpha1_d_L_star;
            q_star(SIGMA_D_INDEX)        = Sigma_d_L_star;
            q_star(RHO_ALPHA1_BAR_INDEX) = rho_L_star*alpha1_bar_L;
            if(curr_d == 0) {
              q_star(RHO_U_INDEX)     = rho_L_star*u_star;
              q_star(RHO_U_INDEX + 1) = rho_L_star*(qL(RHO_U_INDEX + 1)/rho_L);
            }
            else if(curr_d == 1) {
              q_star(RHO_U_INDEX)     = rho_L_star*(qL(RHO_U_INDEX)/rho_L);
              q_star(RHO_U_INDEX + 1) = rho_L_star*u_star;
            }
          }
        }
        // 3-waves left fan
        else {
          // Left of the left fan is qL, already assigned. Now we need to check if we are in
          // the left fan or at the right of the left fan
          const auto alpha1_d_L_star = 1.0 - 1.0/(1.0 + qL(ALPHA1_D_INDEX)/(1.0 - qL(ALPHA1_D_INDEX))*
                                                        std::exp((vel_d_L - u_star)/(c_L*(1.0 - qL(ALPHA1_D_INDEX)))));
          const auto sH_L            = vel_d_L - c_L;
          const auto sT_L            = u_star - c_L*(1.0 + qL(ALPHA1_D_INDEX)*std::exp((vel_d_L - u_star)/(c_L*(1.0 - qL(ALPHA1_D_INDEX)))));

          // Compute state in the left fan
          if(sH_L < 0.0 && sT_L > 0.0) {
            auto alpha1_d_L_fan = qL(ALPHA1_D_INDEX);
            solve_alpha1_d_fan(vel_d_L/(c_L*(1.0 - qL(ALPHA1_D_INDEX))), alpha1_d_L_fan);

            const auto m1_L_fan      = (1.0 - alpha1_d_L_fan)*(qL(M1_INDEX)/(1.0 - qL(ALPHA1_D_INDEX)))*
                                       std::exp((vel_d_L - c_L*(1.0 - qL(ALPHA1_D_INDEX))/(1.0 - alpha1_d_L_fan))/(c_L*(1.0 - qL(ALPHA1_D_INDEX))));
            const auto m2_L_fan      = (1.0 - alpha1_d_L_fan)*(qL(M2_INDEX)/(1.0 - qL(ALPHA1_D_INDEX)))*
                                       std::exp((vel_d_L - c_L*(1.0 - qL(ALPHA1_D_INDEX))/(1.0 - alpha1_d_L_fan))/(c_L*(1.0 - qL(ALPHA1_D_INDEX))));
            const auto m1_d_L_fan    = (qL(ALPHA1_D_INDEX) > this->eps) ? alpha1_d_L_fan*(qL(M1_D_INDEX)/qL(ALPHA1_D_INDEX)) : 0.0;
            const auto Sigma_d_L_fan = (qL(ALPHA1_D_INDEX) > this->eps) ? alpha1_d_L_fan*(qL(SIGMA_D_INDEX)/qL(ALPHA1_D_INDEX)) : 0.0;
            const auto rho_L_fan     = m1_L_fan + m2_L_fan + m1_d_L_fan;

            q_star(M1_INDEX)             = m1_L_fan;
            q_star(M2_INDEX)             = m2_L_fan;
            q_star(M1_D_INDEX)           = m1_d_L_fan;
            q_star(ALPHA1_D_INDEX)       = alpha1_d_L_fan;
            q_star(SIGMA_D_INDEX)        = Sigma_d_L_fan;
            q_star(RHO_ALPHA1_BAR_INDEX) = rho_L_fan*alpha1_bar_L;
            if(curr_d == 0) {
              q_star(RHO_U_INDEX)     = rho_L_fan*(c_L*(1.0 - qL(ALPHA1_D_INDEX))/(1.0 - alpha1_d_L_fan));
              q_star(RHO_U_INDEX + 1) = rho_L_fan*(qL(RHO_U_INDEX + 1)/rho_L);
            }
            else if(curr_d == 1) {
              q_star(RHO_U_INDEX)     = rho_L_fan*(qL(RHO_U_INDEX)/rho_L);
              q_star(RHO_U_INDEX + 1) = rho_L_fan*(c_L*(1.0 - qL(ALPHA1_D_INDEX))/(1.0 - alpha1_d_L_fan));
            }
          }
          // Right of the left fan. Compute the state
          else if(sH_L < 0.0 && sT_L <= 0.0) {
            const auto m1_L_star      = (1.0 - alpha1_d_L_star)*(qL(M1_INDEX)/(1.0 - qL(ALPHA1_D_INDEX)))*
                                        std::exp((vel_d_L - u_star)/(c_L*(1.0 - qL(ALPHA1_D_INDEX))));
            const auto m2_L_star      = (1.0 - alpha1_d_L_star)*(qL(M2_INDEX)/(1.0 - qL(ALPHA1_D_INDEX)))*
                                        std::exp((vel_d_L - u_star)/(c_L*(1.0 - qL(ALPHA1_D_INDEX))));
            const auto m1_d_L_star    = (qL(ALPHA1_D_INDEX) > this->eps) ? alpha1_d_L_star*(qL(M1_D_INDEX)/qL(ALPHA1_D_INDEX)) : 0.0;
            const auto Sigma_d_L_star = (qL(ALPHA1_D_INDEX) > this->eps) ? alpha1_d_L_star*(qL(SIGMA_D_INDEX)/qL(ALPHA1_D_INDEX)) : 0.0;
            const auto rho_L_star     = m1_L_star + m2_L_star + m1_d_L_star;

            q_star(M1_INDEX)             = m1_L_star;
            q_star(M2_INDEX)             = m2_L_star;
            q_star(M1_D_INDEX)           = m1_d_L_star;
            q_star(ALPHA1_D_INDEX)       = alpha1_d_L_star;
            q_star(SIGMA_D_INDEX)        = Sigma_d_L_star;
            q_star(RHO_ALPHA1_BAR_INDEX) = rho_L_star*alpha1_bar_L;
            if(curr_d == 0) {
              q_star(RHO_U_INDEX)     = rho_L_star*u_star;
              q_star(RHO_U_INDEX + 1) = rho_L_star*(qL(RHO_U_INDEX + 1)/rho_L);
            }
            else if(curr_d == 1) {
              q_star(RHO_U_INDEX)     = rho_L_star*(qL(RHO_U_INDEX)/rho_L);
              q_star(RHO_U_INDEX + 1) = rho_L_star*u_star;
            }
          }
        }
      }
      // Right "connecting state"
      else {
        // 1-wave right shock
        if(p_star > p_bar_R) {
          const auto r = 1.0 + (1.0 - qR(ALPHA1_D_INDEX))/
                               (qR(ALPHA1_D_INDEX) + (rho_R*c_R*c_R*(1.0 - qR(ALPHA1_D_INDEX)))/(p_star - p_bar_R));

          const auto m1_R_star       = qR(M1_INDEX)*r;
          const auto m2_R_star       = qR(M2_INDEX)*r;
          const auto alpha1_d_R_star = qR(ALPHA1_D_INDEX)*r;
          const auto m1_d_R_star     = (qR(ALPHA1_D_INDEX) > this->eps) ? alpha1_d_R_star*(qR(M1_D_INDEX)/qR(ALPHA1_D_INDEX)) : 0.0;
          const auto Sigma_d_R_star  = (qR(ALPHA1_D_INDEX) > this->eps) ? alpha1_d_R_star*(qR(SIGMA_D_INDEX)/qR(ALPHA1_D_INDEX)) : 0.0;
          const auto rho_R_star      = m1_R_star + m2_R_star + m1_d_R_star;

          auto s_R = nan("");
          if(r > 1) {
            s_R = u_star + (vel_d_R - u_star)/(1.0 - r);
          }
          else if(r == 1) {
            s_R = u_star + (vel_d_R - u_star)/(-std::numeric_limits<double>::infinity());
          }

          // If right of right shock, the state is qR
          if(std::isnan(s_R) || s_R < 0.0) {
            q_star = qR;
          }
          // Left of right shock, compute the state
          else {
            q_star(M1_INDEX)             = m1_R_star;
            q_star(M2_INDEX)             = m2_R_star;
            q_star(M1_D_INDEX)           = m1_d_R_star;
            q_star(ALPHA1_D_INDEX)       = alpha1_d_R_star;
            q_star(SIGMA_D_INDEX)        = Sigma_d_R_star;
            q_star(RHO_ALPHA1_BAR_INDEX) = rho_R_star*alpha1_bar_R;
            if(curr_d == 0) {
              q_star(RHO_U_INDEX)     = rho_R_star*u_star;
              q_star(RHO_U_INDEX + 1) = rho_R_star*(qR(RHO_U_INDEX + 1)/rho_R);
            }
            else if(curr_d == 1) {
              q_star(RHO_U_INDEX)     = rho_R_star*(qR(RHO_U_INDEX)/rho_R);
              q_star(RHO_U_INDEX + 1) = rho_R_star*u_star;
            }
          }
        }
        // 3-waves right fan
        else {
          auto alpha1_d_R_star = 1.0;
          const auto sH_R      = vel_d_R + c_R;
          auto sT_R            = std::numeric_limits<double>::infinity();
          if(-(vel_d_R - u_star)/(c_R*(1.0 - qR(ALPHA1_D_INDEX))) < 100.0) {
            alpha1_d_R_star = 1.0 - 1.0/(1.0 + qR(ALPHA1_D_INDEX)/(1.0 - qR(ALPHA1_D_INDEX))*
                                               std::exp(-(vel_d_R - u_star)/(c_R*(1.0 - qR(ALPHA1_D_INDEX)))));
            sT_R            = u_star + c_R*(1.0 + qR(ALPHA1_D_INDEX)*std::exp((vel_d_R - u_star)/(c_R*(1.0 - qR(ALPHA1_D_INDEX)))));
          }

          // Right of right fan is qR
          if(sH_R < 0.0) {
            q_star = qR;
          }
          // Compute the state in the right fan
          else if(sH_R >= 0.0 && sT_R < 0.0) {
            auto alpha1_d_R_fan = qR(ALPHA1_D_INDEX);
            solve_alpha1_d_fan(-vel_d_R/(c_R*(1.0 - qL(ALPHA1_D_INDEX))), alpha1_d_R_fan);

            const auto m1_R_fan      = (1.0 - alpha1_d_R_fan)*(qR(M1_INDEX)/(1.0 - qR(ALPHA1_D_INDEX)))*
                                       std::exp(-(vel_d_R + c_R*(1.0 - qR(ALPHA1_D_INDEX))/(1.0 - alpha1_d_R_fan))/(c_R*(1.0 - qR(ALPHA1_D_INDEX))));
            const auto m2_R_fan      = (1.0 - alpha1_d_R_fan)*(qR(M2_INDEX)/(1.0 - qR(ALPHA1_D_INDEX)))*
                                       std::exp(-(vel_d_R + c_R*(1.0 - qR(ALPHA1_D_INDEX))/(1.0 - alpha1_d_R_fan))/(c_R*(1.0 - qR(ALPHA1_D_INDEX))));
            const auto m1_d_R_fan    = (qR(ALPHA1_D_INDEX) > this->eps) ? alpha1_d_R_fan*(qR(M1_D_INDEX)/qR(ALPHA1_D_INDEX)) : 0.0;
            const auto Sigma_d_R_fan = (qR(ALPHA1_D_INDEX) > this->eps) ? alpha1_d_R_fan*(qR(SIGMA_D_INDEX)/qR(ALPHA1_D_INDEX)) : 0.0;
            const auto rho_R_fan     = m1_R_fan + m2_R_fan + m1_d_R_fan;

            q_star(M1_INDEX)             = m1_R_fan;
            q_star(M2_INDEX)             = m2_R_fan;
            q_star(M1_D_INDEX)           = m1_d_R_fan;
            q_star(ALPHA1_D_INDEX)       = alpha1_d_R_fan;
            q_star(SIGMA_D_INDEX)        = Sigma_d_R_fan;
            q_star(RHO_ALPHA1_BAR_INDEX) = rho_R_fan*alpha1_bar_R;
            if(curr_d == 0) {
              q_star(RHO_U_INDEX)     = rho_R_fan*(-c_R*(1.0 - qR(ALPHA1_D_INDEX))/(1.0 - alpha1_d_R_fan));
              q_star(RHO_U_INDEX + 1) = rho_R_fan*(qR(RHO_U_INDEX + 1)/rho_R);
            }
            else if(curr_d == 1) {
              q_star(RHO_U_INDEX)     = rho_R_fan*(qR(RHO_U_INDEX)/rho_R);
              q_star(RHO_U_INDEX + 1) = rho_R_fan*(-c_R*(1.0 - qR(ALPHA1_D_INDEX))/(1.0 - alpha1_d_R_fan));
            }
          }
          // Compute state at the left of the right fan
          else {
            const auto m1_R_star      = (1.0 - alpha1_d_R_star)*(qR(M1_INDEX)/(1.0 - qR(ALPHA1_D_INDEX)))*
                                        std::exp(-(vel_d_R - u_star)/(c_R*(1.0 - qR(ALPHA1_D_INDEX))));
            const auto m2_R_star      = (1.0 - alpha1_d_R_star)*(qR(M2_INDEX)/(1.0 - qR(ALPHA1_D_INDEX)))*
                                        std::exp(-(vel_d_R - u_star)/(c_R*(1.0 - qR(ALPHA1_D_INDEX))));
            const auto m1_d_R_star    = (qR(ALPHA1_D_INDEX) > this->eps) ? alpha1_d_R_star*(qR(M1_D_INDEX)/qR(ALPHA1_D_INDEX)) : 0.0;
            const auto Sigma_d_R_star = (qR(ALPHA1_D_INDEX) > this->eps) ? alpha1_d_R_star*(qR(SIGMA_D_INDEX)/qR(ALPHA1_D_INDEX)) : 0.0;
            const auto rho_R_star     = m1_R_star + m2_R_star + m1_d_R_star;

            q_star(M1_INDEX)             = m1_R_star;
            q_star(M2_INDEX)             = m2_R_star;
            q_star(M1_D_INDEX)           = m1_d_R_star;
            q_star(ALPHA1_D_INDEX)       = alpha1_d_R_star;
            q_star(SIGMA_D_INDEX)        = Sigma_d_R_star;
            q_star(RHO_ALPHA1_BAR_INDEX) = rho_R_star*alpha1_bar_R;
            if(curr_d == 0) {
              q_star(RHO_U_INDEX)     = rho_R_star*u_star;
              q_star(RHO_U_INDEX + 1) = rho_R_star*(qR(RHO_U_INDEX + 1)/rho_R);
            }
            else if(curr_d == 1) {
              q_star(RHO_U_INDEX)     = rho_R_star*(qR(RHO_U_INDEX)/rho_R);
              q_star(RHO_U_INDEX + 1) = rho_R_star*u_star;
            }
          }
        }
      }
    }

    // Compute the hyperbolic contribution to the flux
    FluxValue<typename Flux<Field>::cfg> res = this->evaluate_hyperbolic_operator(q_star, curr_d);

    // Add the contribution due to surface tension
    const auto mod_grad_alpha1_barL = std::sqrt(xt::sum(grad_alpha1_barL*grad_alpha1_barL)());
    const auto mod_grad_alpha1_barR = std::sqrt(xt::sum(grad_alpha1_barR*grad_alpha1_barR)());

    if(mod_grad_alpha1_barL > this->mod_grad_alpha1_bar_min &&
       mod_grad_alpha1_barR > this->mod_grad_alpha1_bar_min) {
      const auto nL = grad_alpha1_barL/mod_grad_alpha1_barL;
      const auto nR = grad_alpha1_barR/mod_grad_alpha1_barR;

      if(curr_d == 0) {
        res(RHO_U_INDEX) += 0.5*this->sigma*((nL(0)*nL(0) - 1.0)*mod_grad_alpha1_barL +
                                             (nR(0)*nR(0) - 1.0)*mod_grad_alpha1_barR);
        res(RHO_U_INDEX + 1) += 0.5*this->sigma*(nL(0)*nL(1)*mod_grad_alpha1_barL +
                                                 nR(0)*nR(1)*mod_grad_alpha1_barR);
      }
      else if(curr_d == 1) {
        res(RHO_U_INDEX) += 0.5*this->sigma*(nL(0)*nL(1)*mod_grad_alpha1_barL +
                                             nR(0)*nR(1)*mod_grad_alpha1_barR);
        res(RHO_U_INDEX + 1) += 0.5*this->sigma*((nL(1)*nL(1) - 1.0)*mod_grad_alpha1_barL +
                                                 (nR(1)*nR(1) - 1.0)*mod_grad_alpha1_barR);
      }
    }

    return res;
  }

  // Implement the contribution of the discrete flux for all the directions.
  //
  template<class Field>
  #ifdef ORDER_2
    template<typename Gradient, typename Field_Scalar>
    auto GodunovFlux<Field>::make_two_scale_capillarity(const Gradient& grad_alpha1_bar,
                                                        const Field_Scalar& H)
  #else
    template<typename Gradient>
    auto GodunovFlux<Field>::make_two_scale_capillarity(const Gradient& grad_alpha1_bar)
  #endif
  {
    FluxDefinition<typename Flux<Field>::cfg> Godunov_f;

    // Perform the loop over each dimension to compute the flux contribution
    static_for<0, EquationData::dim>::apply(
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        // Compute now the "discrete" flux function, in this case a Godunov flux
        Godunov_f[d].cons_flux_function = [&](auto& cells, const Field& field)
                                          {
                                            #ifdef ORDER_2
                                              // Compute the stencil
                                              const auto& left_left   = cells[0];
                                              const auto& left        = cells[1];
                                              const auto& right       = cells[2];
                                              const auto& right_right = cells[3];

                                              // MUSCL reconstruction
                                              const FluxValue<typename Flux<Field>::cfg> primLL = this->cons2prim(field[left_left]);
                                              const FluxValue<typename Flux<Field>::cfg> primL  = this->cons2prim(field[left]);
                                              const FluxValue<typename Flux<Field>::cfg> primR  = this->cons2prim(field[right]);
                                              const FluxValue<typename Flux<Field>::cfg> primRR = this->cons2prim(field[right_right]);

                                              FluxValue<typename Flux<Field>::cfg> primL_recon,
                                                                                   primR_recon;
                                              this->perform_reconstruction(primLL, primL, primR, primRR,
                                                                           primL_recon, primR_recon);

                                              FluxValue<typename Flux<Field>::cfg> qL = this->prim2cons(primL_recon);
                                              FluxValue<typename Flux<Field>::cfg> qR = this->prim2cons(primR_recon);

                                              #ifdef RELAX_RECONSTRUCTION
                                                this->relax_reconstruction(qL, H[left]);
                                                this->relax_reconstruction(qR, H[right]);
                                              #endif
                                            #else
                                              // Compute the stencil and extract state
                                              const auto& left  = cells[0];
                                              const auto& right = cells[1];

                                              const FluxValue<typename Flux<Field>::cfg> qL = field[left];
                                              const FluxValue<typename Flux<Field>::cfg> qR = field[right];
                                            #endif

                                            // Check if we are at a cell with discontinuity in the state. This is not sufficient to say that the
                                            // flux is equal to the 'continuous' one because of surface tension, which involves gradients,
                                            // namely a non-local info
                                            bool is_discontinuous = false;
                                            for(std::size_t comp = 0; comp < Field::size; ++comp) {
                                              if(qL(comp) != qR(comp)) {
                                                is_discontinuous = true;
                                              }

                                              if(is_discontinuous)
                                                break;
                                            }

                                            // Compute the numerical flux
                                            return compute_discrete_flux(qL, qR, d,
                                                                         grad_alpha1_bar[left], grad_alpha1_bar[right],
                                                                         is_discontinuous);
                                          };
    });

    return make_flux_based_scheme(Godunov_f);
  }

} // end of namespace

#endif
