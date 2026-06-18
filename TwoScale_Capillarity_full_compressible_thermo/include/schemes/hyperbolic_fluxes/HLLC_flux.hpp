// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2026
//
#pragma once

#include "../flux_base.hpp"

#define DEBUG_FLUX

namespace samurai {
  using namespace EquationData;

  /**
   * Implementation of a HLLC flux
   */
  template<class Field>
  class HLLCFlux: public Flux<Field> {
  public:
    using Number = Flux<Field>::Number; // Define the shortcut for the arithmetic type
    using cfg    = Flux<Field>::cfg;    // Shortcut to specify the type of configuration
                                        // for the flux (nonlinear in this case)

    /**
     * Class constructor
     * @param EOS_phase_liq_ liquid equation of state
     * @param EOS_phase_gas_ gas equation of state
     * @param sigma_ surface tension coefficient
     */
    HLLCFlux(const EOS<Number>& EOS_phase_liq_,
             const EOS<Number>& EOS_phase_gas_,
             const Number sigma_);

    /**
     * Compute the flux over all the directions
     * @param grad_alpha_l gradient of large-scale volume fraction
     */
    template<class Field_Vect>
    auto make_two_scale_capillarity(const Field_Vect& grad_alpha_l);

  private:
    /**
     * Compute middle state for HLLC flux
     * @param q current state
     * @param grad_alpha_l gradient of large-scale volume fraction
     * @param S estimate of speed of wave propagation
     * @param S_star, estimate of speed of star wave propagation
     * @param curr_d current direction
     * @return q_star, i.e. the middle state
     */
    FluxValue<cfg> compute_middle_state(const FluxValue<cfg>& q,
                                        const auto& grad_alpha_l,
                                        const Number S,
                                        const Number S_star,
                                        const std::size_t curr_d) const;

    /**
     * HLLC flux
     * @param qL left state
     * @param qR right state
     * @param grad_alpha_l_L left gradient of large-scale volume fraction
     * @param grad_alpha_l_R right gradient of large-scale volume fraction
     * @param curr_d current direction
     */
    FluxValue<cfg> compute_discrete_flux(const FluxValue<cfg>& qL,
                                         const FluxValue<cfg>& qR,
                                         const auto& grad_alpha_l_L,
                                         const auto& grad_alpha_l_R,
                                         const std::size_t curr_d);
  };

  // Constructor derived from the base class
  //
  template<class Field>
  HLLCFlux<Field>::HLLCFlux(const EOS<Number>& EOS_phase_liq_,
                            const EOS<Number>& EOS_phase_gas_,
                            const Number sigma_):
    Flux<Field>(EOS_phase_liq_, EOS_phase_gas_, sigma_) {}

  // Implement the auxiliary routine that computes the middle state
  //
  template<class Field>
  FluxValue<typename HLLCFlux<Field>::cfg>
  HLLCFlux<Field>::compute_middle_state(const FluxValue<cfg>& q,
                                        const auto& grad_alpha_l,
                                        const Number S,
                                        const Number S_star,
                                        const std::size_t curr_d) const {
    // Pre-fetch some variables used multiple times in order to exploit possible vectorization
    const auto m_l      = q(Ml_INDEX);
    const auto m_g      = q(Mg_INDEX);
    const auto m_d      = q(Md_INDEX);
    const auto rho_z    = q(RHO_Z_INDEX);
    const auto mliqEliq = q(Mliq_Eliq_INDEX);
    const auto mgEg     = q(Mg_Eg_INDEX);

    // Save velocity current direction
    const auto rho     = m_l + m_g + m_d;
    const auto inv_rho = static_cast<Number>(1.0)/rho;
    const auto vel_d   = q(RHO_U_INDEX + curr_d)*inv_rho;

    // Compute middle state
    FluxValue<cfg> q_star;

    const auto u_star = (S - vel_d)/(S - S_star);

    const auto m_l_star          = m_l*u_star;
    q_star(Ml_INDEX)             = m_l_star;
    const auto m_g_star          = m_g*u_star;
    q_star(Mg_INDEX)             = m_g_star;
    const auto m_d_star          = m_d*u_star;
    q_star(Md_INDEX)             = m_d_star;
    const auto rho_star          = m_l_star + m_g_star + m_d_star;
    const auto alpha_l           = q(RHO_ALPHA_l_INDEX)*inv_rho;
    q_star(RHO_ALPHA_l_INDEX)    = rho_star*alpha_l;
    q_star(RHO_Z_INDEX)          = rho_star*(rho_z*inv_rho);
    q_star(RHO_U_INDEX + curr_d) = rho_star*S_star;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      if(d != curr_d) {
        q_star(RHO_U_INDEX + d) = rho_star*(q(RHO_U_INDEX + d)*inv_rho);
      }
    }

    // Compute contribution related to total energies. The difficulty is that
    // we need to compute pi_k which depends on p_k and requires therefore
    // to compute the thermodynamic internal energy, meaning that we need to
    // remove the contribution due to large-scale gradient
    const auto m_liq = m_l + m_d;
    auto norm2_vel   = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      norm2_vel += (q(RHO_U_INDEX + d)*inv_rho)*(q(RHO_U_INDEX + d)*inv_rho);
    }

    auto mod2_grad_alpha_l = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      mod2_grad_alpha_l += grad_alpha_l[d]*grad_alpha_l[d];
    }
    const auto mod_grad_alpha_l = std::sqrt(mod2_grad_alpha_l);

    // Compute rho_liq
    const auto alpha_d   = alpha_l*m_d/m_l; // TODO: Add a check in case of zero volume fraction
    const auto alpha_liq = alpha_l + alpha_d;
    const auto rho_liq   = m_liq/alpha_liq; // TODO: Add a check in case of zero volume fraction

    const auto Sigma_d = rho_z/std::cbrt(rho_liq*rho_liq);

    // Compute p_liq
    const auto Y_liq   = m_liq*inv_rho;
    const auto chi_liq = Y_liq;
    const auto e_liq   = mliqEliq/m_liq
                       - static_cast<Number>(0.5)*norm2_vel
                       - this->sigma*inv_rho*(chi_liq/Y_liq)*(Sigma_d + mod_grad_alpha_l);

    const auto p_liq = this->EOS_phase_liq.pres_value_Rhoe(rho_liq, e_liq);

    const auto pi_liq = p_liq - static_cast<Number>(2.0/3.0)*this->sigma*chi_liq*Sigma_d/alpha_liq;
                        // TODO: Add a check in case of zero volume fraction

    q_star(Mliq_Eliq_INDEX) = (m_l_star + m_d_star)*
                              (mliqEliq/m_liq + (S_star - vel_d)*(S_star + pi_liq/(rho_liq*(S - vel_d))));

    // Compute rho_g
    const auto alpha_g = static_cast<Number>(1.0) - alpha_liq;
    const auto rho_g   = m_g/alpha_g; // TODO: Add a check in case of zero volume fraction

    // Compute p_g
    const auto Y_g   = static_cast<Number>(1.0) - Y_liq;
    const auto chi_g = Y_g;
    const auto e_g   = mgEg/m_g
                     - static_cast<Number>(0.5)*norm2_vel
                     - this->sigma*inv_rho*(chi_g/Y_g)*(Sigma_d + mod_grad_alpha_l);

    const auto p_g = this->EOS_phase_gas.pres_value_Rhoe(rho_g, e_g);

    const auto pi_g = p_g - static_cast<Number>(2.0/3.0)*this->sigma*chi_g*Sigma_d/alpha_g;
                      // TODO: Add a check in case of zero volume fraction

    q_star(Mg_Eg_INDEX) = m_g_star*
                          (mgEg/m_g + (S_star - vel_d)*(S_star + pi_g/(rho_g*(S - vel_d))));

    return q_star;
  }

  // Implementation of a HLLC flux
  //
  template<class Field>
  FluxValue<typename HLLCFlux<Field>::cfg>
  HLLCFlux<Field>::compute_discrete_flux(const FluxValue<cfg>& qL,
                                         const FluxValue<cfg>& qR,
                                         const auto& grad_alpha_l_L,
                                         const auto& grad_alpha_l_R,
                                         const std::size_t curr_d) {
    // Pre-fetch some variables used multiple times in order to exploit possible vectorization
    const auto m_l_L         = qL(Ml_INDEX);
    const auto m_g_L         = qL(Mg_INDEX);
    const auto m_d_L         = qL(Md_INDEX);
    const auto rho_alpha_l_L = qL(RHO_ALPHA_l_INDEX);
    const auto mliqEliq_L    = qL(Mliq_Eliq_INDEX);
    const auto mgEg_L        = qL(Mg_Eg_INDEX);

    const auto m_l_R         = qR(Ml_INDEX);
    const auto m_g_R         = qR(Mg_INDEX);
    const auto m_d_R         = qR(Md_INDEX);
    const auto rho_alpha_l_R = qR(RHO_ALPHA_l_INDEX);
    const auto mliqEliq_R    = qR(Mliq_Eliq_INDEX);
    const auto mgEg_R        = qR(Mg_Eg_INDEX);

    // Verify if left and right state are coherent
    // Compute c_liq_L
    const auto m_liq_L   = m_l_L + m_d_L;
    const auto rho_L     = m_liq_L + m_g_L;
    const auto inv_rho_L = static_cast<Number>(1.0)/rho_L;

    auto norm2_vel_L = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      norm2_vel_L += (qL(RHO_U_INDEX + d)*inv_rho_L)*(qL(RHO_U_INDEX + d)*inv_rho_L);
    }

    auto mod2_grad_alpha_l_L = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      mod2_grad_alpha_l_L += grad_alpha_l_L[d]*grad_alpha_l_L[d];
    }
    const auto mod_grad_alpha_l_L = std::sqrt(mod2_grad_alpha_l_L);

    const auto alpha_l_L   = rho_alpha_l_L*inv_rho_L;
    const auto alpha_d_L   = alpha_l_L*m_d_L/m_l_L; // TODO: Add a check in case of zero volume fraction
    const auto alpha_liq_L = alpha_l_L + alpha_d_L;
    const auto rho_liq_L   = m_liq_L/alpha_liq_L;
    const auto Sigma_d_L   = qL(RHO_Z_INDEX)/std::cbrt(rho_liq_L*rho_liq_L);

    const auto Y_liq_L   = m_liq_L*inv_rho_L;
    const auto chi_liq_L = Y_liq_L;
    const auto e_liq_L   = mliqEliq_L/m_liq_L
                         - static_cast<Number>(0.5)*norm2_vel_L
                         - this->sigma*inv_rho_L*(chi_liq_L/Y_liq_L)*(Sigma_d_L + mod_grad_alpha_l_L);
                         // TODO: Add a check in case of zero volume fraction

    const auto p_liq_L = this->EOS_phase_liq.pres_value_Rhoe(rho_liq_L, e_liq_L);

    const auto c_liq_L = this->EOS_phase_liq.c_value_RhoP(rho_liq_L, p_liq_L);

    // Compute c_g_L
    const auto alpha_g_L = static_cast<Number>(1.0) - alpha_liq_L;
    const auto rho_g_L   = m_g_L/alpha_g_L;

    const auto Y_g_L   = static_cast<Number>(1.0) - Y_liq_L;
    const auto chi_g_L = Y_g_L;
    const auto e_g_L   = mgEg_L/m_g_L
                       - static_cast<Number>(0.5)*norm2_vel_L
                       - this->sigma*inv_rho_L*(chi_g_L/Y_g_L)*(Sigma_d_L + mod_grad_alpha_l_L);
                       // TODO: Add a check in case of zero volume fraction

    const auto p_g_L = this->EOS_phase_gas.pres_value_Rhoe(rho_g_L, e_g_L);

    const auto c_g_L = this->EOS_phase_gas.c_value_RhoP(rho_g_L, p_g_L);

    // Compute c_liq_R
    const auto m_liq_R   = m_l_R + m_d_R;
    const auto rho_R     = m_liq_R + m_g_R;
    const auto inv_rho_R = static_cast<Number>(1.0)/rho_R;

    auto norm2_vel_R = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      norm2_vel_R += (qR(RHO_U_INDEX + d)*inv_rho_R)*(qR(RHO_U_INDEX + d)*inv_rho_R);
    }

    auto mod2_grad_alpha_l_R = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      mod2_grad_alpha_l_R += grad_alpha_l_R[d]*grad_alpha_l_R[d];
    }
    const auto mod_grad_alpha_l_R = std::sqrt(mod2_grad_alpha_l_R);

    const auto alpha_l_R   = rho_alpha_l_R*inv_rho_R;
    const auto alpha_d_R   = alpha_l_R*m_d_R/m_l_R; // TODO: Add a check in case of zero volume fraction
    const auto alpha_liq_R = alpha_l_R + alpha_d_R;
    const auto rho_liq_R   = m_liq_R/alpha_liq_R;
    const auto Sigma_d_R   = qR(RHO_Z_INDEX)/std::cbrt(rho_liq_R*rho_liq_R);

    const auto Y_liq_R   = m_liq_R*inv_rho_R;
    const auto chi_liq_R = Y_liq_R;
    const auto e_liq_R   = mliqEliq_R/m_liq_R
                         - static_cast<Number>(0.5)*norm2_vel_R
                         - this->sigma*inv_rho_R*(chi_liq_R/Y_liq_R)*(Sigma_d_R + mod_grad_alpha_l_R);
                         // TODO: Add a check in case of zero volume fraction

    const auto p_liq_R = this->EOS_phase_liq.pres_value_Rhoe(rho_liq_R, e_liq_R);

    const auto c_liq_R = this->EOS_phase_liq.c_value_RhoP(rho_liq_R, p_liq_R);

    // Compute c_g_R
    const auto alpha_g_R = static_cast<Number>(1.0) - alpha_liq_R;
    const auto rho_g_R   = m_g_R/alpha_g_R;

    const auto Y_g_R   = static_cast<Number>(1.0) - Y_liq_R;
    const auto chi_g_R = Y_g_R;
    const auto e_g_R   = mgEg_R/m_g_R
                       - static_cast<Number>(0.5)*norm2_vel_R
                       - this->sigma*inv_rho_R*(chi_g_R/Y_g_R)*(Sigma_d_R + mod_grad_alpha_l_R);
                       // TODO: Add a check in case of zero volume fraction

    const auto p_g_R = this->EOS_phase_gas.pres_value_Rhoe(rho_g_R, e_g_R);

    const auto c_g_R = this->EOS_phase_gas.c_value_RhoP(rho_g_R, p_g_R);

    // Perform the check
    #ifdef DEBUG_FLUX
      if(m_l_L < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative mass large-scale liquid left state: " + std::to_string(m_l_L)));
      }
      if(m_g_L < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative mass gas left state: " + std::to_string(m_g_L)));
      }
      if(m_d_L < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative mass small-scale liquid left state: " + std::to_string(m_d_L)));
      }
      if(rho_alpha_l_L < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative volume fraction large-scale liquid left state: " + std::to_string(alpha_l_L)));
      }
      if(Sigma_d_L < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative interface area small-scale liquid left state: " + std::to_string(Sigma_d_L)));
      }
      if(std::isnan(c_liq_L)) {
        throw std::runtime_error(std::string("Non admissible liquid pressure left state: " + std::to_string(p_liq_L)));
      }
      if(std::isnan(c_g_L)) {
        std::cout << e_g_L << std::endl;
        std::cout << m_g_L << std::endl;
        std::cout << norm2_vel_L << std::endl;
        throw std::runtime_error(std::string("Non admissible gas pressure left state: " + std::to_string(p_g_L)));
      }

      if(m_l_R < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative mass large-scale liquid right state: " + std::to_string(m_l_R)));
      }
      if(m_g_R < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative mass gas right state: " + std::to_string(m_g_R)));
      }
      if(m_d_R < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative mass small-scale liquid right state: " + std::to_string(m_d_R)));
      }
      if(alpha_l_R < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative volume fraction large-scale liquid right state: " + std::to_string(alpha_l_R)));
      }
      if(Sigma_d_R < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative interface area small-scale liquid right state: " + std::to_string(Sigma_d_R)));
      }
      if(std::isnan(c_liq_R)) {
        throw std::runtime_error(std::string("Non admissible liquid pressure right state: " + std::to_string(p_liq_R)));
      }
      if(std::isnan(c_g_R)) {
        throw std::runtime_error(std::string("Non admissible gas pressure right state: " + std::to_string(p_g_R)));
      }
    #endif

    // Compute the quantities needed for the maximum eigenvalue estimate for the left state
    const auto vel_d_L = qL(RHO_U_INDEX + curr_d)*inv_rho_L;
    const auto cf_L    = std::sqrt(Y_liq_L*c_liq_L*c_liq_L +
                                   Y_g_L*c_g_L*c_g_L -
                                   static_cast<Number>(2.0/9.0)*this->sigma*Sigma_d_L*inv_rho_L);

    // Compute the quantities needed for the maximum eigenvalue estimate for the right state
    const auto vel_d_R = qR(RHO_U_INDEX + curr_d)*inv_rho_R;
    const auto cf_R    = std::sqrt(Y_liq_R*c_liq_R*c_liq_R +
                                   Y_g_R*c_g_R*c_g_R -
                                   static_cast<Number>(2.0/9.0)*this->sigma*Sigma_d_R*inv_rho_R);

    // Compute speeds of wave propagation
    const auto s_L    = std::min(vel_d_L - cf_L, vel_d_R - cf_R);
    const auto s_R    = std::max(vel_d_L + cf_L, vel_d_R + cf_R);
    const auto p_L    = alpha_liq_L*p_liq_L
                      + alpha_g_L*p_g_L
                      - static_cast<Number>(2.0/3.0)*this->sigma*Sigma_d_L;
    const auto p_R    = alpha_liq_R*p_liq_R
                      + alpha_g_R*p_g_R
                      - static_cast<Number>(2.0/3.0)*this->sigma*Sigma_d_R;
    const auto s_star = (p_R - p_L + rho_L*vel_d_L*(s_L - vel_d_L) - rho_R*vel_d_R*(s_R - vel_d_R))/
                        (rho_L*(s_L - vel_d_L) - rho_R*(s_R - vel_d_R));

    // Compute intermediate states
    const auto q_star_L = compute_middle_state(qL, grad_alpha_l_L, s_L, s_star, curr_d);
    const auto q_star_R = compute_middle_state(qR, grad_alpha_l_R, s_R, s_star, curr_d);

    // Compute the flux
    if(s_L >= static_cast<Number>(0.0)) {
      return this->evaluate_conservative_hyperbolic_operator(qL, grad_alpha_l_L, curr_d);
    }
    else if(s_L < static_cast<Number>(0.0) &&
            s_star >= static_cast<Number>(0.0)) {
      return this->evaluate_conservative_hyperbolic_operator(qL, grad_alpha_l_L, curr_d) + s_L*(q_star_L - qL);
    }
    else if(s_star < static_cast<Number>(0.0) &&
            s_R >= static_cast<Number>(0.0)) {
      return this->evaluate_conservative_hyperbolic_operator(qR, grad_alpha_l_R, curr_d) + s_R*(q_star_R - qR);
    }
    else if(s_R < static_cast<Number>(0.0)) {
      return this->evaluate_conservative_hyperbolic_operator(qR, grad_alpha_l_R, curr_d);
    }
  }

  // Implement the contribution of the discrete flux for all the directions.
  //
  template<class Field>
  template<class Field_Vect>
  auto HLLCFlux<Field>::make_two_scale_capillarity(const Field_Vect& grad_alpha_l)
  {
    FluxDefinition<cfg> HLLC_f;

    // Perform the loop over each dimension to compute the flux contribution
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
         {
           static constexpr int d = decltype(integral_constant_d)::value;

           // Compute now the "discrete" flux function, in this case a HLLC flux
           HLLC_f[d].cons_flux_function = [&](FluxValue<cfg>& flux,
                                              const StencilData<cfg>& data,
                                              const StencilValues<cfg>& field)
                                              {
                                                #ifdef ORDER_2
                                                  // MUSCL reconstruction
                                                  const auto grad_alpha_l_LL = grad_alpha_l[data.cells[0]];
                                                  const auto grad_alpha_l_L  = grad_alpha_l[data.cells[1]];
                                                  const auto grad_alpha_l_R  = grad_alpha_l[data.cells[2]];
                                                  const auto grad_alpha_l_RR = grad_alpha_l[data.cells[3]];

                                                  const FluxValue<cfg> primLL = this->cons2prim(field[0], grad_alpha_l_LL);
                                                  const FluxValue<cfg> primL  = this->cons2prim(field[1], grad_alpha_l_L);
                                                  const FluxValue<cfg> primR  = this->cons2prim(field[2], grad_alpha_l_R);
                                                  const FluxValue<cfg> primRR = this->cons2prim(field[3], grad_alpha_l_RR);

                                                  FluxValue<cfg> primL_recon,
                                                                 primR_recon;
                                                  Utilities::perform_reconstruction<Field>(primLL, primL, primR, primRR,
                                                                                           primL_recon, primR_recon);

                                                  /* NOTE: Perform the reconstruction on w = grad\alpha_{l}. This is maybe where
                                                  a 'mixed' formulation differs from a formulation in which we keep \grad\alpha_{l} that
                                                  we suitably approximate, e.g., as finite difference of \alpha_{l}.
                                                  In the mixed formulation, I should reconstruct the auxiliary variable for 'coherence',
                                                  while, keeping \grad\alpha_{l}, I should recompute its aprpoximation starting from the
                                                  reconstructed values. The 'issue' somewhat is that I do not have all the reconstructed
                                                  values to computed the gradient. Suppose I am on face i+1/2,j: I have access to
                                                  \alpha_{l_{i+1,j}} and \alpha_{l_{i-1,j}} so as to compute
                                                  (\alpha_{l_{i+1,j}} - \alpha_{l_{i-1,j}})/dx as approximation of \partial_{x}\alpha_{l_{i+1/2,j}},
                                                  but what about the approximation of \partial_{y}\alpha_{l_{i+1/2,j}}? I do not have access, e.g., to
                                                  \alpha_{l_{j+1,i}} reconstructed. With the first approach obviously, we 'decouple' w from \alpha_{l},
                                                  in the sense that it is no longer computed directly as \grad\alpha_{l} */
                                                  /*decltype(grad_alpha_l_L) grad_alpha_l_L_flux,
                                                                           grad_alpha_l_R_flux;
                                                  Utilities::perform_reconstruction<Field_Vect>(grad_alpha_l_LL, grad_alpha_l_L,
                                                                                                grad_alpha_l_R, grad_alpha_l_RR,
                                                                                                grad_alpha_l_L_flux, grad_alpha_l_R_flux);*/

                                                  /*FluxValue<cfg> qL = this->prim2cons(primL_recon, grad_alpha_l_L_flux;
                                                  FluxValue<cfg> qR = this->prim2cons(primR_recon, grad_alpha_l_R_flux);*/
                                                  FluxValue<cfg> qL = this->prim2cons(primL_recon, grad_alpha_l_L);
                                                  FluxValue<cfg> qR = this->prim2cons(primR_recon, grad_alpha_l_R);
                                                #else
                                                  // Extract the states
                                                  const FluxValue<cfg>& qL = field[0];
                                                  const FluxValue<cfg>& qR = field[1];

                                                  /*const auto& grad_alpha_l_L_flux = grad_alpha_l[data.cells[0]];
                                                  const auto& grad_alpha_l_R_flux = grad_alpha_l[data.cells[1]];*/
                                                  const auto& grad_alpha_l_L = grad_alpha_l[data.cells[0]];
                                                  const auto& grad_alpha_l_R = grad_alpha_l[data.cells[1]];
                                                #endif

                                                //flux = compute_discrete_flux(qL, qR, grad_alpha_l_L_flux, grad_alpha_l_R_flux, d);
                                                flux = compute_discrete_flux(qL, qR, grad_alpha_l_L, grad_alpha_l_R, d);
                                              };
        }
    );

    auto scheme = make_flux_based_scheme(HLLC_f);
    scheme.set_name("HLLC");

    return scheme;
  }

} // end of namespace
