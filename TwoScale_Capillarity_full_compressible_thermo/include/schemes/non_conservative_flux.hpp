// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2026
//
#pragma once

#include "flux_base.hpp"

namespace samurai {
  /**
    * Implementation of the non-conservative flux
    */
  template<class Field>
  class NonConservativeFlux: public Flux<Field> {
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
    NonConservativeFlux(const EOS<Number>& EOS_phase_liq_,
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
     * Non-conservative flux
     * @param qL left state
     * @param qR right state
     * @param grad_alpha_l_L left gradient of large-scale volume fraction
     * @param grad_alpha_l_R right gradient of large-scale volume fraction
     * @param curr_d current direction
     * @return T_minus 'internal' contribution of non-conservative flux
     * @return T_plus 'external' contribution of non-conservative flux
     */
    void compute_discrete_flux(const FluxValue<cfg>& qL,
                               const FluxValue<cfg>& qR,
                               const auto& grad_alpha_l_L,
                               const auto& grad_alpha_l_R,
                               const std::size_t curr_d,
                               FluxValue<cfg>& T_minus,
                               FluxValue<cfg>& T_plus);
  };

  // Constructor derived from base class
  //
  template<class Field>
  NonConservativeFlux<Field>::NonConservativeFlux(const EOS<Number>& EOS_phase_liq_,
                                                  const EOS<Number>& EOS_phase_gas_,
                                                  const Number sigma_):
    Flux<Field>(EOS_phase_liq_, EOS_phase_gas_, sigma_) {}

  // Implementation of a non-conservative flux
  //
  template<class Field>
  void NonConservativeFlux<Field>::compute_discrete_flux(const FluxValue<cfg>& qL,
                                                         const FluxValue<cfg>& qR,
                                                         const auto& grad_alpha_l_L,
                                                         const auto& grad_alpha_l_R,
                                                         const std::size_t curr_d,
                                                         FluxValue<cfg>& T_minus,
                                                         FluxValue<cfg>& T_plus) {
    // Zero contribution from continuity and momentum equations
    // (as well as large-scale interface and small-scale interfacial area density)
    T_minus(Ml_INDEX)          = static_cast<Number>(0.0);
    T_plus(Ml_INDEX)           = static_cast<Number>(0.0);
    T_minus(Mg_INDEX)          = static_cast<Number>(0.0);
    T_plus(Mg_INDEX)           = static_cast<Number>(0.0);
    T_minus(Md_INDEX)          = static_cast<Number>(0.0);
    T_plus(Md_INDEX)           = static_cast<Number>(0.0);
    T_minus(RHO_Z_INDEX)       = static_cast<Number>(0.0);
    T_plus(RHO_Z_INDEX)        = static_cast<Number>(0.0);
    T_minus(RHO_ALPHA_l_INDEX) = static_cast<Number>(0.0);
    T_plus(RHO_ALPHA_l_INDEX)  = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      T_minus(RHO_U_INDEX + d) = static_cast<Number>(0.0);
      T_plus(RHO_U_INDEX + d)  = static_cast<Number>(0.0);
    }

    /*--- Left state ---*/
    // Pre-fetch variables that will be used several times so as to exploit possible vectorization
    // (as well as to enhance readability)
    const auto m_l_L         = qL(Ml_INDEX);
    const auto m_g_L         = qL(Mg_INDEX);
    const auto m_d_L         = qL(Md_INDEX);
    const auto rho_alpha_l_L = qL(RHO_ALPHA_l_INDEX);
    const auto mliqEliq_L    = qL(Mliq_Eliq_INDEX);
    const auto mgEg_L        = qL(Mg_Eg_INDEX);

    // Compute velocity along current direction
    const auto m_liq_L   = m_l_L + m_d_L;
    const auto rho_L     = m_liq_L + m_g_L;
    const auto inv_rho_L = static_cast<Number>(1.0)/rho_L;
    const auto vel_d_L   = qL(RHO_U_INDEX + curr_d)*inv_rho_L;

    // Density liquid phase
    const auto alpha_l_L   = rho_alpha_l_L*inv_rho_L;
    const auto alpha_d_L   = alpha_l_L*m_d_L/m_l_L; // TODO: Add a check in case of zero volume fraction
    const auto alpha_liq_L = alpha_l_L + alpha_d_L;
    const auto rho_liq_L   = m_liq_L/alpha_liq_L; // TODO: Add a check in case of zero volume fraction

    const auto Sigma_d_L = qL(RHO_Z_INDEX)/std::cbrt(rho_liq_L*rho_liq_L);

    // Pressure liquid phase
    auto norm2_vel_L  = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      norm2_vel_L += (qL(RHO_U_INDEX + d)*inv_rho_L)*(qL(RHO_U_INDEX + d)*inv_rho_L);
    }

    auto mod2_grad_alpha_l_L = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      mod2_grad_alpha_l_L += grad_alpha_l_L[d]*grad_alpha_l_L[d];
    }
    const auto mod_grad_alpha_l_L = std::sqrt(mod2_grad_alpha_l_L);

    const auto Y_liq_L   = m_liq_L*inv_rho_L;
    const auto chi_liq_L = Y_liq_L;
    const auto e_liq_L   = mliqEliq_L/m_liq_L
                         - static_cast<Number>(0.5)*norm2_vel_L
                         - this->sigma*inv_rho_L*(chi_liq_L/Y_liq_L)*(Sigma_d_L + mod_grad_alpha_l_L);
                         // TODO: Add a check in case of zero volume fraction
    const auto p_liq_L   = this->EOS_phase_liq.pres_value_Rhoe(rho_liq_L, e_liq_L);

    // Augmented pressure liquid phase
    const auto pi_liq_L = p_liq_L - static_cast<Number>(2.0/3.0)*this->sigma*chi_liq_L*Sigma_d_L/alpha_liq_L;
    // TODO: Add a check in case of zero volume fraction

    // Density gas phase
    const auto alpha_g_L = static_cast<Number>(1.0) - alpha_liq_L;
    const auto rho_g_L   = m_g_L/alpha_g_L;

    // Pressure gas phase
    const auto Y_g_L   = static_cast<Number>(1.0) - Y_liq_L;
    const auto chi_g_L = Y_g_L;
    const auto e_g_L   = mgEg_L/m_g_L
                       - static_cast<Number>(0.5)*norm2_vel_L
                       - this->sigma*inv_rho_L*(chi_g_L/Y_g_L)*(Sigma_d_L + mod_grad_alpha_l_L);
                       /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto p_g_L   = this->EOS_phase_gas.pres_value_Rhoe(rho_g_L, e_g_L);

    // Augmented pressure gas phase
    const auto pi_g_L = p_g_L - static_cast<Number>(2.0/3.0)*this->sigma*chi_g_L*Sigma_d_L/alpha_g_L;
    // TODO: Add a check in case of zero volume fraction

    /*--- Right state ---*/
    // Pre-fetch variables that will be used several times so as to exploit possible vectorization
    // (as well as to enhance readability)
    const auto m_l_R         = qR(Ml_INDEX);
    const auto m_g_R         = qR(Mg_INDEX);
    const auto m_d_R         = qR(Md_INDEX);
    const auto rho_alpha_l_R = qR(RHO_ALPHA_l_INDEX);
    const auto mliqEliq_R    = qR(Mliq_Eliq_INDEX);
    const auto mgEg_R        = qR(Mg_Eg_INDEX);

    // Compute velocity along current direction
    const auto m_liq_R   = m_l_R + m_d_R;
    const auto rho_R     = m_liq_R + m_g_R;
    const auto inv_rho_R = static_cast<Number>(1.0)/rho_R;
    const auto vel_d_R   = qR(RHO_U_INDEX + curr_d)*inv_rho_R;

    // Density liquid phase
    const auto alpha_l_R   = rho_alpha_l_R*inv_rho_R;
    const auto alpha_d_R   = alpha_l_R*m_d_R/m_l_R; // TODO: Add a check in case of zero volume fraction
    const auto alpha_liq_R = alpha_l_R + alpha_d_R;
    const auto rho_liq_R   = m_liq_R/alpha_liq_R; // TODO: Add a check in case of zero volume fraction

    const auto Sigma_d_R = qR(RHO_Z_INDEX)/std::cbrt(rho_liq_R*rho_liq_R);

    // Pressure liquid phase
    auto norm2_vel_R  = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      norm2_vel_R += (qR(RHO_U_INDEX + d)*inv_rho_R)*(qR(RHO_U_INDEX + d)*inv_rho_R);
    }

    auto mod2_grad_alpha_l_R = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      mod2_grad_alpha_l_R += grad_alpha_l_R[d]*grad_alpha_l_R[d];
    }
    const auto mod_grad_alpha_l_R = std::sqrt(mod2_grad_alpha_l_R);

    const auto Y_liq_R   = m_liq_R*inv_rho_R;
    const auto chi_liq_R = Y_liq_R;
    const auto e_liq_R   = mliqEliq_R/m_liq_R
                         - static_cast<Number>(0.5)*norm2_vel_R
                         - this->sigma*inv_rho_R*(chi_liq_R/Y_liq_R)*(Sigma_d_R + mod_grad_alpha_l_R);
                         // TODO: Add a check in case of zero volume fraction
    const auto p_liq_R   = this->EOS_phase_liq.pres_value_Rhoe(rho_liq_R, e_liq_R);

    // Augmented pressure liquid phase
    const auto pi_liq_R = p_liq_R - static_cast<Number>(2.0/3.0)*this->sigma*chi_liq_R*Sigma_d_R/alpha_liq_R;
    // TODO: Add a check in case of zero volume fraction

    // Density gas phase
    const auto alpha_g_R = static_cast<Number>(1.0) - alpha_liq_R;
    const auto rho_g_R   = m_g_R/alpha_g_R;

    // Pressure gas phase
    const auto Y_g_R   = static_cast<Number>(1.0) - Y_liq_R;
    const auto chi_g_R = Y_g_R;
    const auto e_g_R   = mgEg_R/m_g_R
                       - static_cast<Number>(0.5)*norm2_vel_R
                       - this->sigma*inv_rho_R*(chi_g_R/Y_g_R)*(Sigma_d_R + mod_grad_alpha_l_R);
                       // TODO: Add a check in case of zero volume fraction
    const auto p_g_R   = this->EOS_phase_gas.pres_value_Rhoe(rho_g_R, e_g_R);

    // Augmented pressure gas phase
    const auto pi_g_R = p_g_R - static_cast<Number>(2.0/3.0)*this->sigma*chi_g_R*Sigma_d_R/alpha_g_R;
    // TODO: Add a check in case of zero volume fraction

    /*--- Build the non conservative flux ---*/
    T_minus(Mliq_Eliq_INDEX) = -(static_cast<Number>(0.5)*
                                 (vel_d_L*Y_g_L*alpha_liq_L*pi_liq_L +
                                  vel_d_R*Y_g_R*alpha_liq_R*pi_liq_R) -
                                 static_cast<Number>(0.5)*
                                 (vel_d_L*Y_g_L + vel_d_R*Y_g_R)*
                                 alpha_liq_L*pi_liq_L)
                              +(static_cast<Number>(0.5)*
                                (vel_d_L*Y_liq_L*alpha_g_L*pi_g_L +
                                 vel_d_R*Y_liq_R*alpha_g_R*pi_g_R) -
                                static_cast<Number>(0.5)*
                                (vel_d_L*Y_liq_L + vel_d_R*Y_liq_R)*
                                alpha_g_L*pi_g_L);
    T_plus(Mliq_Eliq_INDEX)  = -(static_cast<Number>(0.5)*
                                 (vel_d_L*Y_g_L*alpha_liq_L*pi_liq_L +
                                  vel_d_R*Y_g_R*alpha_liq_R*pi_liq_R) -
                                 static_cast<Number>(0.5)*
                                 (vel_d_L*Y_g_L + vel_d_R*Y_g_R)*
                                 alpha_liq_R*pi_liq_R)
                               +(static_cast<Number>(0.5)*
                                 (vel_d_L*Y_liq_L*alpha_g_L*pi_g_L +
                                  vel_d_R*Y_liq_R*alpha_g_R*pi_g_R) -
                                 static_cast<Number>(0.5)*
                                 (vel_d_L*Y_liq_L + vel_d_R*Y_liq_R)*
                                 alpha_g_R*pi_g_R);

    T_minus(Mg_Eg_INDEX) = -T_minus(Mliq_Eliq_INDEX);
    T_plus(Mg_Eg_INDEX)  = -T_plus(Mliq_Eliq_INDEX);
  }

  // Implement the contribution of the discrete flux for all the dimensions.
  //
  template<class Field>
  template<class Field_Vect>
  auto NonConservativeFlux<Field>::make_two_scale_capillarity(const Field_Vect& grad_alpha_l)
  {
    FluxDefinition<cfg> non_conservative_flux;

    // Perform the loop over each dimension to compute the flux contribution
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
         {
           static constexpr int d = decltype(integral_constant_d)::value;

           // Compute now the "discrete" non-conservative flux function
           non_conservative_flux[d].flux_function = [&](FluxValuePair<cfg>& flux,
                                                        const StencilData<cfg>& data,
                                                        const StencilValues<cfg> field)
                                                        {
                                                          #ifdef ORDER_2
                                                            // MUSCL recsontruction
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
                                                            a mixed formulation differs from a formulation in which we keep \grad\alpha_{l} that
                                                            we suitably approximate, e.g., as finite difference of \alpha_{l}.
                                                            In the mixed formulation, I should reconstruct the auxiliary variable for 'coherence',
                                                            while, keeping \grad\alpha_{l}, I should recompute its aprpoximation starting from the
                                                            reconstructed values. The 'issue' somewhat is that I do not have all the reconstructed
                                                            values to computed the gradient. Suppose I am on face i+1/2,j: I have access to
                                                            \alpha_{l_{i+1,j}} and \alpha_{l_{i-1,j}} so as to compute
                                                            (\alpha_{l_{i+1,j}} - \alpha_{l_{i-1,j}})/dx as approximation of \partial_{x}\alpha_{l_{i+1/2,j}},
                                                            but what about the approximation of \partial_{y}\alpha_{l_{i+1/2,j}}? I do not have access, e.g., to
                                                            \alpha_{l_{j+1,i}} reconstructed. With the first approach obviously, we 'decouple' w from \alpha_{l},
                                                            in the sense that it is no onleger computed directly as \grad\alpha_{l} */
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

                                                          FluxValue<cfg> T_minus,
                                                                         T_plus;

                                                          /*compute_discrete_flux(qL, qR,
                                                                                grad_alpha_l_L_flux, grad_alpha_l_R_flux,
                                                                                d, T_minus, T_plus);*/
                                                          compute_discrete_flux(qL, qR,
                                                                                grad_alpha_l_L, grad_alpha_l_R,
                                                                                d, T_minus, T_plus);

                                                          flux[0] = T_minus;
                                                          flux[1] = -T_plus;
                                                        };
        }
    );

    auto scheme = make_flux_based_scheme(non_conservative_flux);
    scheme.set_name("Non-conservative flux");

    return scheme;
  }

} // end of namespace
