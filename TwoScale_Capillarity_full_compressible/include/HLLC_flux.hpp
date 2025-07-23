// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#ifndef HLLC_flux_hpp
#define HLLC_flux_hpp

#include "flux_base.hpp"

#define VERBOSE_FLUX

namespace samurai {
  using namespace EquationData;

  /**
    * Implementation of a HLLC flux
    */
  template<class Field>
  class HLLCFlux: public Flux<Field> {
  public:
    HLLCFlux(const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase_liq_,
             const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase_gas_,
             const typename Field::value_type sigma_,
             const typename Field::value_type mod_grad_alpha_l_min_,
             const typename Field::value_type lambda_,
             const typename Field::value_type atol_Newton_,
             const typename Field::value_type rtol_Newton_,
             const std::size_t max_Newton_iters_); /*--- Constructor which accepts in inputs the equations of state of the two phases ---*/

    #ifdef ORDER_2
      template<typename Field_Scalar>
      auto make_two_scale_capillarity(const Field_Scalar& H); /*--- Compute the flux over all the directions ---*/
    #else
      auto make_two_scale_capillarity(); /*--- Compute the flux over all the directions ---*/
    #endif

  private:
    auto compute_middle_state(const FluxValue<typename Flux<Field>::cfg>& q,
                              const typename Field::value_type S,
                              const typename Field::value_type S_star,
                              const std::size_t curr_d) const; /*--- Compute the middle state ---*/

    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                               const FluxValue<typename Flux<Field>::cfg>& qR,
                                                               const std::size_t curr_d); /*--- HLLC flux for the along direction curr_d ---*/
  };

  // Constructor derived from the base class
  //
  template<class Field>
  HLLCFlux<Field>::HLLCFlux(const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase_liq_,
                            const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase_gas_,
                            const typename Field::value_type sigma_,
                            const typename Field::value_type mod_grad_alpha_l_min_,
                            const typename Field::value_type lambda_,
                            const typename Field::value_type atol_Newton_,
                            const typename Field::value_type rtol_Newton_,
                            const std::size_t max_Newton_iters_):
    Flux<Field>(EOS_phase_liq_, EOS_phase_gas_,
                sigma_, mod_grad_alpha_l_min_,
                lambda_, atol_Newton_, rtol_Newton_, max_Newton_iters_) {}

  // Implement the auxiliary routine that computes the middle state
  //
  template<class Field>
  auto HLLCFlux<Field>::compute_middle_state(const FluxValue<typename Flux<Field>::cfg>& q,
                                             const typename Field::value_type S,
                                             const typename Field::value_type S_star,
                                             const std::size_t curr_d) const {
    /*--- Save velocity current direction ---*/
    const auto rho   = q(Ml_INDEX) + q(Mg_INDEX) + q(Md_INDEX);
    const auto vel_d = q(RHO_U_INDEX + curr_d)/rho;

    /*--- Compute middle state ---*/
    FluxValue<typename Flux<Field>::cfg> q_star;

    q_star(Ml_INDEX)             = q(Ml_INDEX)*((S - vel_d)/(S - S_star));
    q_star(Mg_INDEX)             = q(Mg_INDEX)*((S - vel_d)/(S - S_star));
    q_star(Md_INDEX)             = q(Md_INDEX)*((S - vel_d)/(S - S_star));
    const auto rho_star          = q_star(Ml_INDEX) + q_star(Mg_INDEX) + q_star(Md_INDEX);
    q_star(RHO_ALPHA_l_INDEX)    = rho_star*(q(RHO_ALPHA_l_INDEX)/rho);
    q_star(RHO_Z_INDEX)          = rho_star*(q(RHO_Z_INDEX)/rho);
    q_star(RHO_U_INDEX + curr_d) = rho_star*S_star;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      if(d != curr_d) {
        q_star(RHO_U_INDEX + d) = rho_star*(q(RHO_U_INDEX + d)/rho);
      }
    }

    return q_star;
  }

  // Implementation of a HLLC flux
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> HLLCFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                                              const FluxValue<typename Flux<Field>::cfg>& qR,
                                                                              const std::size_t curr_d) {
    /*--- Verify if left and right state are coherent ---*/
    #ifdef VERBOSE_FLUX
      if(qL(Ml_INDEX) < static_cast<typename Field::value_type>(0.0)) {
        throw std::runtime_error(std::string("Negative mass large-scale liquid left state: " + std::to_string(qL(Ml_INDEX))));
      }
      if(qL(Mg_INDEX) < static_cast<typename Field::value_type>(0.0)) {
        throw std::runtime_error(std::string("Negative mass gas left state: " + std::to_string(qL(Mg_INDEX))));
      }
      if(qL(Md_INDEX) < static_cast<typename Field::value_type>(-1e-15)) {
        throw std::runtime_error(std::string("Negative mass small-scale liquid left state: " + std::to_string(qL(Md_INDEX))));
      }
      if(qL(RHO_ALPHA_l_INDEX) < static_cast<typename Field::value_type>(0.0)) {
        throw std::runtime_error(std::string("Negative volume fraction large-scale liquid left state: " + std::to_string(qL(RHO_ALPHA_l_INDEX))));
      }
      if(qL(RHO_Z_INDEX) < static_cast<typename Field::value_type>(-1e-15)) {
        throw std::runtime_error(std::string("Negative interface area small-scale liquid left state: " + std::to_string(qL(RHO_Z_INDEX))));
      }

      if(qR(Ml_INDEX) < static_cast<typename Field::value_type>(0.0)) {
        throw std::runtime_error(std::string("Negative mass large-scale liquid right state: " + std::to_string(qR(Ml_INDEX))));
      }
      if(qR(Mg_INDEX) < static_cast<typename Field::value_type>(0.0)) {
        throw std::runtime_error(std::string("Negative mass gas right state: " + std::to_string(qR(Mg_INDEX))));
      }
      if(qR(Md_INDEX) < static_cast<typename Field::value_type>(-1e-15)) {
        throw std::runtime_error(std::string("Negative mass small-scale liquid right state: " + std::to_string(qR(Md_INDEX))));
      }
      if(qR(RHO_ALPHA_l_INDEX) < static_cast<typename Field::value_type>(0.0)) {
        throw std::runtime_error(std::string("Negative volume fraction large-scale liquid right state: " + std::to_string(qR(RHO_ALPHA_l_INDEX))));
      }
      if(qR(RHO_Z_INDEX) < static_cast<typename Field::value_type>(-1e-15)) {
        throw std::runtime_error(std::string("Negative interface area small-scale liquid right state: " + std::to_string(qR(RHO_Z_INDEX))));
      }
    #endif

    /*--- Compute the quantities needed for the maximum eigenvalue estimate for the left state ---*/
    const auto rho_L     = qL(Ml_INDEX) + qL(Mg_INDEX) + qL(Md_INDEX);
    const auto vel_d_L   = qL(RHO_U_INDEX + curr_d)/rho_L;

    const auto alpha_l_L = qL(RHO_ALPHA_l_INDEX)/rho_L;
    const auto alpha_d_L = alpha_l_L*qL(Md_INDEX)/qL(Ml_INDEX); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto alpha_g_L = static_cast<typename Field::value_type>(1.0) - alpha_l_L - alpha_d_L;

    const auto Y_g_L     = qL(Mg_INDEX)/rho_L;
    const auto rho_liq_L = (qL(Ml_INDEX) + qL(Md_INDEX))/(alpha_l_L + alpha_d_L); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rho_g_L   = qL(Mg_INDEX)/alpha_g_L; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto Sigma_d_L = qL(RHO_Z_INDEX)/std::pow(rho_liq_L, static_cast<typename Field::value_type>(2.0/3.0));
    const auto c_L       = std::sqrt((static_cast<typename Field::value_type>(1.0) - Y_g_L)*
                                     this->EOS_phase_liq.c_value(rho_liq_L)*
                                     this->EOS_phase_liq.c_value(rho_liq_L) +
                                     Y_g_L*
                                     this->EOS_phase_gas.c_value(rho_g_L)*
                                     this->EOS_phase_gas.c_value(rho_g_L) -
                                     static_cast<typename Field::value_type>(2.0/9.0)*this->sigma*Sigma_d_L/rho_L);

    /*--- Compute the quantities needed for the maximum eigenvalue estimate for the right state ---*/
    const auto rho_R     = qR(Ml_INDEX) + qR(Mg_INDEX) + qR(Md_INDEX);
    const auto vel_d_R   = qR(RHO_U_INDEX + curr_d)/rho_R;

    const auto alpha_l_R = qR(RHO_ALPHA_l_INDEX)/rho_R;
    const auto alpha_d_R = alpha_l_R*qR(Md_INDEX)/qR(Ml_INDEX); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto alpha_g_R = static_cast<typename Field::value_type>(1.0) - alpha_l_R - alpha_d_R;

    const auto Y_g_R     = qR(Mg_INDEX)/rho_R;
    const auto rho_liq_R = (qR(Ml_INDEX) + qR(Md_INDEX))/(alpha_l_R + alpha_d_R); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rho_g_R   = qR(Mg_INDEX)/alpha_g_R; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto Sigma_d_R = qR(RHO_Z_INDEX)/std::pow(rho_liq_R, static_cast<typename Field::value_type>(2.0/3.0));
    const auto c_R       = std::sqrt((static_cast<typename Field::value_type>(1.0) - Y_g_R)*
                                     this->EOS_phase_liq.c_value(rho_liq_R)*
                                     this->EOS_phase_liq.c_value(rho_liq_R) +
                                     Y_g_R*
                                     this->EOS_phase_gas.c_value(rho_g_R)*
                                     this->EOS_phase_gas.c_value(rho_g_R) -
                                     static_cast<typename Field::value_type>(2.0/9.0)*this->sigma*Sigma_d_R/rho_R);

    /*--- Compute speeds of wave propagation ---*/
    const auto s_L    = std::min(vel_d_L - c_L, vel_d_R - c_R);
    const auto s_R    = std::max(vel_d_L + c_L, vel_d_R + c_R);
    const auto p_L    = (alpha_l_L + alpha_d_L)*this->EOS_phase_liq.pres_value(rho_liq_L)
                      + alpha_g_L*this->EOS_phase_gas.pres_value(rho_g_L)
                      - static_cast<typename Field::value_type>(2.0/3.0)*this->sigma*Sigma_d_L;
    const auto p_R    = (alpha_l_R + alpha_d_R)*this->EOS_phase_liq.pres_value(rho_liq_R)
                      + alpha_g_R*this->EOS_phase_gas.pres_value(rho_g_R)
                      - static_cast<typename Field::value_type>(2.0/3.0)*this->sigma*Sigma_d_R;
    const auto s_star = (p_R - p_L + rho_L*vel_d_L*(s_L - vel_d_L) - rho_R*vel_d_R*(s_R - vel_d_R))/
                        (rho_L*(s_L - vel_d_L) - rho_R*(s_R - vel_d_R));

    /*--- Compute intermediate states ---*/
    const auto q_star_L = compute_middle_state(qL, s_L, s_star, curr_d);
    const auto q_star_R = compute_middle_state(qR, s_R, s_star, curr_d);

    if(q_star_L(Md_INDEX) < static_cast<typename Field::value_type>(-1e-15)) {
      throw std::runtime_error("Negative mass small-scale left star state");
    }
    if(q_star_R(Md_INDEX) < static_cast<typename Field::value_type>(-1e-15)) {
      throw std::runtime_error("Negative mass small-scale right star state");
    }

    /*--- Compute the flux ---*/
    if(s_L >= static_cast<typename Field::value_type>(0.0)) {
      return this->evaluate_hyperbolic_operator(qL, curr_d);
    }
    else if(s_L < static_cast<typename Field::value_type>(0.0) &&
            s_star >= static_cast<typename Field::value_type>(0.0)) {
      return this->evaluate_hyperbolic_operator(qL, curr_d) + s_L*(q_star_L - qL);
    }
    else if(s_star < static_cast<typename Field::value_type>(0.0) &&
            s_R >= static_cast<typename Field::value_type>(0.0)) {
      return this->evaluate_hyperbolic_operator(qR, curr_d) + s_R*(q_star_R - qR);
    }
    else if(s_R < static_cast<typename Field::value_type>(0.0)) {
      return this->evaluate_hyperbolic_operator(qR, curr_d);
    }
  }

  // Implement the contribution of the discrete flux for all the directions.
  //
  template<class Field>
  #ifdef ORDER_2
    template<typename Field_Scalar>
    auto HLLCFlux<Field>::make_two_scale_capillarity(const Field_Scalar& H)
  #else
    auto HLLCFlux<Field>::make_two_scale_capillarity()
  #endif
  {
    FluxDefinition<typename Flux<Field>::cfg> HLLC_f;

    /*--- Perform the loop over each dimension to compute the flux contribution ---*/
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
         {
           static constexpr int d = decltype(integral_constant_d)::value;

           // Compute now the "discrete" flux function, in this case a HLLC flux
           HLLC_f[d].cons_flux_function = [&](samurai::FluxValue<typename Flux<Field>::cfg>& flux,
                                              const StencilData<typename Flux<Field>::cfg>& data,
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

                                                  #ifdef RELAX_RECONSTRUCTION
                                                    this->relax_reconstruction(qL, H[data.cells[1]][0]);
                                                    this->relax_reconstruction(qR, H[data.cells[2]][0]);
                                                  #endif
                                                #else
                                                  // Extract the states
                                                  const FluxValue<typename Flux<Field>::cfg> qL = field[0];
                                                  const FluxValue<typename Flux<Field>::cfg> qR = field[1];
                                                #endif

                                                flux = compute_discrete_flux(qL, qR, d);
                                              };
        }
    );

    auto scheme = make_flux_based_scheme(HLLC_f);
    scheme.set_name("HLLC");

    return scheme;
  }

} // end of namespace

#endif
