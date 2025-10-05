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
    * Implementation of a Godunov flux
    */
  template<class Field>
  class GodunovFlux: public Flux<Field> {
  public:
    using Number = Flux<Field>::Number; /*--- Define the shortcut for the arithmetic type ---*/
    using cfg    = Flux<Field>::cfg;    /*--- Shortcut to specify the type of configuration
                                              for the flux (nonlinear in this case) ---*/

    GodunovFlux(const LinearizedBarotropicEOS<Number>& EOS_phase1_,
                const LinearizedBarotropicEOS<Number>& EOS_phase2_,
                const Number sigma_,
                const Number mod_grad_alpha1_min_,
                const Number lambda_,
                const Number atol_Newton_,
                const Number rtol_Newton_,
                const std::size_t max_Newton_iters_,
                const Number atol_Newton_p_star_ = static_cast<Number>(1e-10),
                const Number rtol_Newton_p_star_ = static_cast<Number>(1e-8));
                /*--- Constructor which accepts in inputs the equations of state of the two phases ---*/

    #ifdef ORDER_2
      template<typename Field_Scalar>
      auto make_flux(const Field_Scalar& H); /*--- Compute the flux over all the directions ---*/
    #else
      auto make_flux(); /*--- Compute the flux over all the directions ---*/
    #endif

  private:
    const Number atol_Newton_p_star; /*--- Absolute tolerance of the Newton method to compute p_star ---*/
    const Number rtol_Newton_p_star; /*--- Relative tolerance of the Newton method to compute p_star ---*/

    FluxValue<cfg> compute_discrete_flux(const FluxValue<cfg>& qL,
                                         const FluxValue<cfg>& qR,
                                         const std::size_t curr_d); /*--- Godunov flux for the along direction curr_d ---*/

    void solve_p_star(const FluxValue<cfg>& qL,
                      const FluxValue<cfg>& qR,
                      const Number dvel_d,
                      const Number vel_d_L,
                      const Number p0_L,
                      const Number p0_R,
                      Number& p_star); /*--- Newton method to compute p* in the exact solver for the hyperbolic part ---*/
  };

  // Constructor derived from the base class
  //
  template<class Field>
  GodunovFlux<Field>::GodunovFlux(const LinearizedBarotropicEOS<Number>& EOS_phase1_,
                                  const LinearizedBarotropicEOS<Number>& EOS_phase2_,
                                  const Number sigma_,
                                  const Number mod_grad_alpha1_min_,
                                  const Number lambda_,
                                  const Number atol_Newton_,
                                  const Number rtol_Newton_,
                                  const std::size_t max_Newton_iters_,
                                  const Number atol_Newton_p_star_,
                                  const Number rtol_Newton_p_star_):
    Flux<Field>(EOS_phase1_, EOS_phase2_,
                sigma_, mod_grad_alpha1_min_,
                lambda_, atol_Newton_, rtol_Newton_, max_Newton_iters_),
    atol_Newton_p_star(atol_Newton_p_star_), rtol_Newton_p_star(rtol_Newton_p_star_) {}

  // Compute p* through Newton-Rapson method
  //
  template<class Field>
  void GodunovFlux<Field>::solve_p_star(const FluxValue<cfg>& qL,
                                        const FluxValue<cfg>& qR,
                                        const Number dvel_d,
                                        const Number vel_d_L,
                                        const Number p0_L,
                                        const Number p0_R,
                                        Number& p_star) {
    Number dp_star = std::numeric_limits<Number>::infinity();

    /*--- Pre-fetch some variables used multiple times in order to exploit possible vectorization ---*/
    const auto m1_L = qL(M1_INDEX);
    const auto m2_L = qL(M2_INDEX);

    const auto m1_R = qR(M1_INDEX);
    const auto m2_R = qR(M2_INDEX);

    /*--- Left state useful variables ---*/
    const auto rho_L          = m1_L + m2_L;
    const auto inv_rho_L      = static_cast<Number>(1.0)/rho_L;
    const auto alpha1_L       = qL(RHO_ALPHA1_INDEX)*inv_rho_L;
    const auto rho1_L         = m1_L/alpha1_L; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto alpha2_L       = static_cast<Number>(1.0) - alpha1_L;
    const auto rho2_L         = m2_L/alpha2_L; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rhoc_squared_L = m1_L*this->EOS_phase1.c_value(rho1_L)*this->EOS_phase1.c_value(rho1_L)
                              + m2_L*this->EOS_phase2.c_value(rho2_L)*this->EOS_phase2.c_value(rho2_L);
    const auto c_L            = std::sqrt(rhoc_squared_L*inv_rho_L);
    const auto p_L            = alpha1_L*this->EOS_phase1.pres_value(rho1_L)
                              + alpha2_L*this->EOS_phase2.pres_value(rho2_L);

    /*--- Right state useful variables ---*/
    const auto rho_R          = m1_R + m2_R;
    const auto inv_rho_R      = static_cast<Number>(1.0)/rho_R;
    const auto alpha1_R       = qR(RHO_ALPHA1_INDEX)*inv_rho_R;
    const auto rho1_R         = m1_R/alpha1_R; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto alpha2_R       = static_cast<Number>(1.0) - alpha1_R;
    const auto rho2_R         = m2_R/alpha2_R; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rhoc_squared_R = m1_R*this->EOS_phase1.c_value(rho1_R)*this->EOS_phase1.c_value(rho1_R)
                              + m2_R*this->EOS_phase2.c_value(rho2_R)*this->EOS_phase2.c_value(rho2_R);
    const auto c_R            = std::sqrt(rhoc_squared_R*inv_rho_R);
    const auto p_R            = alpha1_R*this->EOS_phase1.pres_value(rho1_R)
                              + alpha2_R*this->EOS_phase2.pres_value(rho2_R);

    if(p_star <= p0_L || p_L <= p0_L) {
      throw std::runtime_error("Non-admissible value for the pressure at the beginning of the Newton method to compute p* in Godunov solver");
    }

    auto F_p_star = dvel_d;
    if(p_star <= p_L) {
      F_p_star += c_L*std::log((p_L - p0_L)/(p_star - p0_L));
    }
    else {
      F_p_star -= (p_star - p_L)/std::sqrt(rho_L*(p_star - p0_L));
    }
    if(p_star <= p_R) {
      F_p_star += c_R*std::log((p_R - p0_R)/(p_star - p0_R));
    }
    else {
      F_p_star -= (p_star - p_R)/std::sqrt(rho_R*(p_star - p0_R));
    }

    /*--- Loop of Newton method to compute p* ---*/
    std::size_t Newton_iter = 0;
    while(Newton_iter < this->max_Newton_iters &&
          std::abs(F_p_star) > this->atol_Newton_p_star + this->rtol_Newton_p_star*std::abs(vel_d_L) &&
          std::abs(dp_star) > this->atol_Newton_p_star + this->rtol_Newton_p_star*std::abs(p_star)) {
      Newton_iter++;

      // Unmodified Newton-Rapson increment
      Number dF_p_star;
      if(p_star <= p_L) {
        dF_p_star = c_L/(p0_L - p_star);
      }
      else {
        dF_p_star = (static_cast<Number>(2.0)*p0_L - p_star - p_L)/
                    (static_cast<Number>(2.0)*(p_star - p0_L)*std::sqrt(rho_L*(p_star - p0_L)));
      }
      if(p_star <= p_R) {
        dF_p_star += c_R/(p0_R - p_star);
      }
      else {
        dF_p_star += (static_cast<Number>(2.0)*p0_R - p_star - p_R)/
                     (static_cast<Number>(2.0)*(p_star - p0_R)*std::sqrt(rho_R*(p_star - p0_R)));
      }
      dp_star = -F_p_star/dF_p_star;

      // Bound preserving increment
      dp_star = std::max(dp_star, this->lambda*(std::max(p0_L, p0_R) - p_star));

      if(p_star + dp_star <= p0_L) {
        throw std::runtime_error("Non-admissible value for the pressure in the Newton method to compute p* in Godunov solver");
      }
      else {
        p_star += dp_star;
      }

      // Newton cycle diverged
      if(Newton_iter == this->max_Newton_iters) {
        throw std::runtime_error("Netwon method not converged to compute p* in the Godunov solver");
      }

      // Update function for which we seek the zero
      F_p_star = dvel_d;
      if(p_star <= p_L) {
        F_p_star += c_L*std::log((p_L - p0_L)/(p_star - p0_L));
      }
      else {
        F_p_star -= (p_star - p_L)/std::sqrt(rho_L*(p_star - p0_L));
      }
      if(p_star <= p_R) {
        F_p_star += c_R*std::log((p_R - p0_R)/(p_star - p0_R));
      }
      else {
        F_p_star -= (p_star - p_R)/std::sqrt(rho_R*(p_star - p0_R));
      }
    }
  }

  // Implementation of a Godunov flux
  //
  template<class Field>
  GodunovFlux<Field>::FluxValue<cfg>
  GodunovFlux<Field>::compute_discrete_flux(const FluxValue<cfg>& qL,
                                            const FluxValue<cfg>& qR,
                                            const std::size_t curr_d) {
    /*--- Pre-fetch some variables used multiple times in order to exploit possible vectorization ---*/
    const auto m1_L         = qL(M1_INDEX);
    const auto m2_L         = qL(M2_INDEX);
    const auto rho_alpha1_L = qL(RHO_ALPHA1_INDEX);

    const auto m1_R         = qR(M1_INDEX);
    const auto m2_R         = qR(M2_INDEX);
    const auto rho_alpha1_R = qR(RHO_ALPHA1_INDEX);

    /*--- Verify if left and right state are coherent ---*/
    #ifdef VERBOSE_FLUX
      if(m1_L < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative mass phase 1 left state: " + std::to_string(m1_L)));
      }
      if(m2_L < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative mass phase 2 left state: " + std::to_string(m2_L)));
      }
      if(rho_alpha1_L < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative volume fraction phase 1 left state: " + std::to_string(rho_alpha1_L)));
      }

      if(m1_R < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative mass phase 1 right state: " + std::to_string(m1_R)));
      }
      if(m2_R < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative mass phase 2 right state: " + std::to_string(m2_R)));
      }
      if(rho_alpha1_R < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative volume fraction phase 1 right state: " + std::to_string(rho_alpha1_R)));
      }
    #endif

    /*--- Compute the intermediate state (either shock or rarefaction) ---*/
    FluxValue<cfg> q_star = qL;

    // Left state useful variables
    const auto rho_L          = m1_L + m2_L;
    const auto inv_rho_L      = static_cast<Number>(1.0)/rho_L;
    const auto vel_d_L        = qL(RHO_U_INDEX + curr_d)*inv_rho_L;
    const auto alpha1_L       = rho_alpha1_L*inv_rho_L;
    const auto rho1_L         = m1_L/alpha1_L; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto alpha2_L       = static_cast<Number>(1.0) - alpha1_L;
    const auto rho2_L         = m2_L/alpha2_L; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rhoc_squared_L = m1_L*this->EOS_phase1.c_value(rho1_L)*this->EOS_phase1.c_value(rho1_L)
                              + m2_L*this->EOS_phase2.c_value(rho2_L)*this->EOS_phase2.c_value(rho2_L);
    #ifdef VERBOSE_FLUX
      if(rho_L < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative density left state: " + std::to_string(rho_L)));
      }
      if(rhoc_squared_L*inv_rho_L < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative square speed of sound left state: " + std::to_string(rhoc_squared_L*inv_rho_L)));
      }
    #endif
    const auto c_L = std::sqrt(rhoc_squared_L*inv_rho_L);

    // Right state useful variables
    const auto rho_R          = m1_R + m2_R;
    const auto inv_rho_R      = static_cast<Number>(1.0)/rho_R;
    const auto vel_d_R        = qR(RHO_U_INDEX + curr_d)*inv_rho_R;
    const auto alpha1_R       = rho_alpha1_R*invrho_R;
    const auto rho1_R         = m1_R/alpha1_R; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto alpha2_R       = static_cast<Number>(1.0) - alpha1_R;
    const auto rho2_R         = m2_R/alpha2_R; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rhoc_squared_R = m1_R*this->EOS_phase1.c_value(rho1_R)*this->EOS_phase1.c_value(rho1_R)
                              + m2_R*this->EOS_phase2.c_value(rho2_R)*this->EOS_phase2.c_value(rho2_R);
    #ifdef VERBOSE_FLUX
      if(rho_R < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative density right state: " + std::to_string(rho_R)));
      }
      if(rhoc_squared_R*inv_rho_R < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative square speed of sound right state: " + std::to_string(rhoc_squared_R*inv_rho_R)));
      }
    #endif
    const auto c_R = std::sqrt(rhoc_squared_R*inv_rho_R);

    // Compute p*
    const auto p_L = alpha1_L*this->EOS_phase1.pres_value(rho1_L)
                   + alpha2_L*this->EOS_phase2.pres_value(rho2_L);
    const auto p_R = alpha1_R*this->EOS_phase1.pres_value(rho1_R)
                   + alpha2_R*this->EOS_phase2.pres_value(rho2_R);

    const auto p0_L = p_L - rho_L*c_L*c_L;
    const auto p0_R = p_R - rho_R*c_R*c_R;

    auto p_star = std::max(static_cast<Number>(0.5)*(p_L + p_R),
                           std::max(p0_L, p0_R) + static_cast<Number>(0.1)*std::abs(std::max(p0_L, p0_R)));
    solve_p_star(qL, qR, vel_d_L - vel_d_R, vel_d_L, p0_L, p0_R, p_star);

    // Compute u*
    const auto u_star = (p_star <= p_L) ? vel_d_L + c_L*std::log((p_L - p0_L)/(p_star - p0_L)) :
                                          vel_d_L - (p_star - p_L)/std::sqrt(rho_L*(p_star - p0_L));

    // Left "connecting state"
    if(u_star > static_cast<Number>(0.0)) {
      // 1-wave left shock
      if(p_star > p_L) {
        const auto r = static_cast<Number>(1.0)
                     + static_cast<Number>(1.0)/(rho_L*c_L*c_L/(p_star - p_L));

        const auto m1_L_star  = m1_L*r;
        const auto m2_L_star  = m2_L*r;
        const auto rho_L_star = m1_L_star + m2_L_star;

        auto s_L = nan("");
        if(r > static_cast<Number>(1.0)) {
          s_L = u_star
              + (vel_d_L - u_star)/(static_cast<Number>(1.0) - r);
        }
        else if(r == static_cast<Number>(1.0)) {
          s_L = u_star
              + (vel_d_L - u_star)*(-std::numeric_limits<Number>::infinity());
        }

        // If left of left shock, q* = qL, already assigned.
        // If right of left shock, is the computed state
        if(!std::isnan(s_L) && s_L < static_cast<Number>(0.0)) {
          q_star(M1_INDEX)             = m1_L_star;
          q_star(M2_INDEX)             = m2_L_star;
          q_star(RHO_ALPHA1_INDEX)     = rho_L_star*alpha1_L;
          q_star(RHO_U_INDEX + curr_d) = rho_L_star*u_star;
          for(std::size_t d = 0; d < Field::dim; ++d) {
            if(d != curr_d) {
              q_star(RHO_U_INDEX + d) = rho_L_star*(qL(RHO_U_INDEX + d)*inv_rho_L);
            }
          }
        }
      }
      // 3-wave left fan
      else {
        // Left of the left fan is qL, already assigned. Now we need to check if we are in
        // the left fan or at the right of the left fan
        const auto sH_L = vel_d_L - c_L;
        const auto sT_L = u_star - c_L;

        // Compute state in the left fan
        if(sH_L < static_cast<Number>(0.0) &&
           sT_L > static_cast<Number>(0.0)) {
          const auto m1_L_fan  = m1_L*std::exp((vel_d_L - c_L)/c_L);
          const auto m2_L_fan  = m2_L*std::exp((vel_d_L - c_L)/c_L);
          const auto rho_L_fan = m1_L_fan + m2_L_fan;

          q_star(M1_INDEX)             = m1_L_fan;
          q_star(M2_INDEX)             = m2_L_fan;
          q_star(RHO_ALPHA1_INDEX)     = rho_L_fan*alpha1_L;
          q_star(RHO_U_INDEX + curr_d) = rho_L_fan*c_L;
          for(std::size_t d = 0; d < Field::dim; ++d) {
            if(d != curr_d) {
              q_star(RHO_U_INDEX + d) = rho_L_fan*(qL(RHO_U_INDEX + d)*inv_rho_L);
            }
          }
        }
        // Right of the left fan. Compute the state
        else if(sH_L < static_cast<Number>(0.0) &&
                sT_L <= static_cast<Number>(0.0)) {
          const auto m1_L_star  = m1_L*std::exp((vel_d_L - u_star)/c_L);
          const auto m2_L_star  = m2_L*std::exp((vel_d_L - u_star)/c_L);
          const auto rho_L_star = m1_L_star + m2_L_star;

          q_star(M1_INDEX)             = m1_L_star;
          q_star(M2_INDEX)             = m2_L_star;
          q_star(RHO_ALPHA1_INDEX)     = rho_L_star*alpha1_L;
          q_star(RHO_U_INDEX + curr_d) = rho_L_star*u_star;
          for(std::size_t d = 0; d < Field::dim; ++d) {
            if(d != curr_d) {
              q_star(RHO_U_INDEX + d) = rho_L_star*(qL(RHO_U_INDEX + d)*inv_rho_L);
            }
          }
        }
      }
    }
    // Right "connecting state"
    else {
      // 1-wave right shock
      if(p_star > p_R) {
        const auto r = static_cast<Number>(1.0)
                     + static_cast<Number>(1.0)/(rho_R*c_R*c_R/(p_star - p_R));

        const auto m1_R_star  = m1_R*r;
        const auto m2_R_star  = m2_R*r;
        const auto rho_R_star = m1_R_star + m2_R_star;

        auto s_R = nan("");
        if(r > static_cast<Number>(1.0)) {
          s_R = u_star
              + (vel_d_R - u_star)/(static_cast<Number>(1.0) - r);
        }
        else if(r == static_cast<Number>(1.0)) {
          s_R = u_star
              + (vel_d_R - u_star)/(-std::numeric_limits<Number>::infinity());
        }

        // If right of right shock, the state is qR
        if(std::isnan(s_R) || s_R < static_cast<Number>(0.0)) {
          q_star = qR;
        }
        // Left of right shock, compute the state
        else {
          q_star(M1_INDEX)             = m1_R_star;
          q_star(M2_INDEX)             = m2_R_star;
          q_star(RHO_ALPHA1_INDEX)     = rho_R_star*alpha1_R;
          q_star(RHO_U_INDEX + curr_d) = rho_R_star*u_star;
          for(std::size_t d = 0; d < Field::dim; ++d) {
            if(d != curr_d) {
              q_star(RHO_U_INDEX + d) = rho_R_star*(qR(RHO_U_INDEX + d)*inv_rho_R);
            }
          }
        }
      }
      // 3-wave right fan
      else {
        const auto sH_R = vel_d_R + c_R;
        auto sT_R       = std::numeric_limits<Number>::infinity();
        if(-(vel_d_R - u_star)/c_R < static_cast<Number>(100.0)) {
          sT_R = u_star + c_R;
        }

        // Right of right fan is qR
        if(sH_R < static_cast<Number>(0.0)) {
          q_star = qR;
        }
        // Compute the state in the right fan
        else if(sH_R >= static_cast<Number>(0.0) &&
                sT_R < static_cast<Number>(0.0)) {
          const auto m1_R_fan  = m1_R*std::exp(-(vel_d_R + c_R)/c_R);
          const auto m2_R_fan  = m2_R*std::exp(-(vel_d_R + c_R)/c_R);
          const auto rho_R_fan = m1_R_fan + m2_R_fan;

          q_star(M1_INDEX)             = m1_R_fan;
          q_star(M2_INDEX)             = m2_R_fan;
          q_star(RHO_ALPHA1_INDEX)     = rho_R_fan*alpha1_R;
          q_star(RHO_U_INDEX + curr_d) = -rho_R_fan*c_R;
          for(std::size_t d = 0; d < Field::dim; ++d) {
            if(d != curr_d) {
              q_star(RHO_U_INDEX + d) = rho_R_fan*(qR(RHO_U_INDEX + d)*inv_rho_R);
            }
          }
        }
        // Compute state at the left of the right fan
        else {
          const auto m1_R_star  = m1_R*std::exp(-(vel_d_R - u_star)/c_R);
          const auto m2_R_star  = m2_R*std::exp(-(vel_d_R - u_star)/c_R);
          const auto rho_R_star = m1_R_star + m2_R_star ;

          q_star(M1_INDEX)             = m1_R_star;
          q_star(M2_INDEX)             = m2_R_star;
          q_star(RHO_ALPHA1_INDEX)     = rho_R_star*alpha1_R;
          q_star(RHO_U_INDEX + curr_d) = rho_R_star*u_star;
          for(std::size_t d = 0; d < Field::dim; ++d) {
            if(d != curr_d) {
              q_star(RHO_U_INDEX + d) = rho_R_star*(qR(RHO_U_INDEX + d)*inv_rho_R);
            }
          }
        }
      }
    }

    /*--- Compute the hyperbolic contribution to the flux ---*/
    return this->evaluate_hyperbolic_operator(q_star, curr_d);
  }

  // Implement the contribution of the discrete flux for all the directions.
  //
  template<class Field>
  #ifdef ORDER_2
    template<typename Field_Scalar>
    auto GodunovFlux<Field>::make_flux(const Field_Scalar& H)
  #else
    auto GodunovFlux<Field>::make_flux()
  #endif
  {
    FluxDefinition<cfg> Godunov_f;

    /*--- Perform the loop over each dimension to compute the flux contribution ---*/
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
         {
           static constexpr int d = decltype(integral_constant_d)::value;

           // Compute now the "discrete" flux function, in this case a Godunov flux
           Godunov_f[d].cons_flux_function = [&](FluxValue<cfg>& flux,
                                                 const StencilData<cfg>& data,
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

                                                     FluxValue<cfg> qL = this->prim2cons(primL_recon);
                                                     FluxValue<cfg> qR = this->prim2cons(primR_recon);

                                                     #ifdef RELAX_RECONSTRUCTION
                                                       this->relax_reconstruction(qL, H[data.cells[1]][0]);
                                                       this->relax_reconstruction(qR, H[data.cells[2]][0]);
                                                     #endif
                                                  #else
                                                    // Extract the states
                                                    const FluxValue<cfg> qL = field[0];
                                                    const FluxValue<cfg> qR = field[1];
                                                  #endif

                                                  flux = compute_discrete_flux(qL, qR, d);
                                                };
        }
    );

    auto scheme = make_flux_based_scheme(Godunov_f);
    scheme.set_name("Godunov exact");

    return scheme;
  }

} // end of namespace
