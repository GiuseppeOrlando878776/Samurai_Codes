// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#ifndef Rusanov_flux_hpp
#define Rusanov_flux_hpp

#include "flux_base.hpp"

#define VERBOSE_FLUX

namespace samurai {
  using namespace EquationData;

  /**
    * Implementation of a Rusanov flux
    */
  template<class Field>
  class RusanovFlux: public Flux<Field> {
  public:
    using Number = Flux<Field>::Number; /*--- Define the shortcut for the arithmetic type ---*/
    using cfg    = Flux<Field>::cfg; /*--- Shortcut to specify the type of configuration
                                           for the flux (nonlinear in this case) ---*/

    RusanovFlux(const LinearizedBarotropicEOS<Number>& EOS_phase_liq_,
                const LinearizedBarotropicEOS<Number>& EOS_phase_gas_,
                const Number sigma_,
                const Number mod_grad_alpha_l_min_,
                const Number lambda_,
                const Number atol_Newton_,
                const Number rtol_Newton_,
                const std::size_t max_Newton_iters_); /*--- Constructor which accepts in input the equations of state of the two phases ---*/

    #ifdef ORDER_2
      template<typename Field_Scalar>
      auto make_two_scale_capillarity(const Field_Scalar& H); /*--- Compute the flux over all the directions ---*/
    #else
      auto make_two_scale_capillarity(); /*--- Compute the flux over all the directions ---*/
    #endif

  private:
    FluxValue<cfg> compute_discrete_flux(const FluxValue<cfg>& qL,
                                         const FluxValue<cfg>& qR,
                                         const std::size_t curr_d); /*--- Rusanov flux along direction curr_d ---*/
  };

  // Constructor derived from the base class
  //
  template<class Field>
  RusanovFlux<Field>::RusanovFlux(const LinearizedBarotropicEOS<Number>& EOS_phase_liq_,
                                  const LinearizedBarotropicEOS<Number>& EOS_phase_gas_,
                                  const Number sigma_,
                                  const Number mod_grad_alpha_l_min_,
                                  const Number lambda_,
                                  const Number atol_Newton_,
                                  const Number rtol_Newton_,
                                  const std::size_t max_Newton_iters_):
    Flux<Field>(EOS_phase_liq_, EOS_phase_gas_,
                sigma_, mod_grad_alpha_l_min_,
                lambda_, atol_Newton_, rtol_Newton_, max_Newton_iters_) {}

  // Implementation of a Rusanov flux
  //
  template<class Field>
  FluxValue<RusanovFlux<Field>::cfg>
  RusanovFlux<Field>::compute_discrete_flux(const FluxValue<cfg>& qL,
                                            const FluxValue<cfg>& qR,
                                            const std::size_t curr_d) {
    /*--- Pre-fetch some variables used multiple times in order to exploit possible vectorization ---*/
    const auto m_l_L         = qL(Ml_INDEX);
    const auto m_g_L         = qL(Mg_INDEX);
    const auto m_d_L         = qL(Md_INDEX);
    const auto rho_alpha_l_L = qL(RHO_ALPHA_l_INDEX);
    const auto rho_z_L       = qL(RHO_Z_INDEX);

    const auto m_l_R         = qR(Ml_INDEX);
    const auto m_g_R         = qR(Mg_INDEX);
    const auto m_d_R         = qR(Md_INDEX);
    const auto rho_alpha_l_R = qR(RHO_ALPHA_l_INDEX);
    const auto rho_z_R       = qR(RHO_Z_INDEX);

    /*--- Verify if left and right state are coherent ---*/
    #ifdef VERBOSE_FLUX
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
        throw std::runtime_error(std::string("Negative volume fraction large-scale liquid left state: " + std::to_string(rho_alpha_l_L)));
      }
      if(rho_z_L < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative interface area small-scale liquid left state: " + std::to_string(rho_z_L)));
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
      if(rho_alpha_l_R < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative volume fraction large-scale liquid right state: " + std::to_string(rho_alpha_l_R)));
      }
      if(rho_z_R < static_cast<Number>(0.0)) {
        throw std::runtime_error(std::string("Negative interface area small-scale liquid right state: " + std::to_string(rho_z_R)));
      }
    #endif

    /*--- Compute the quantities needed for the maximum eigenvalue estimate for the left state ---*/
    const auto rho_L     = m_l_L + m_g_L + m_d_L;
    const auto inv_rho_L = static_cast<Number>(1.0)/rho_L;
    const auto vel_d_L   = qL(RHO_U_INDEX + curr_d)*inv_rho_L;

    const auto alpha_l_L = rho_alpha_l_L*inv_rho_L;
    const auto alpha_d_L = alpha_l_L*m_d_L/m_l_L; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto alpha_g_L = static_cast<Number>(1.0) - alpha_l_L - alpha_d_L;

    const auto Y_g_L     = m_g_L*inv_rho_L;
    const auto rho_liq_L = (m_l_L + m_d_L)/(alpha_l_L + alpha_d_L); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rho_g_L   = m_g_L/alpha_g_L; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto Sigma_d_L = rho_z_L/std::cbrt(rho_liq_L*rho_liq_L);
    const auto c_L       = std::sqrt((static_cast<Number>(1.0) - Y_g_L)*
                                     this->EOS_phase_liq.c_value(rho_liq_L)*
                                     this->EOS_phase_liq.c_value(rho_liq_L) +
                                     Y_g_L*
                                     this->EOS_phase_gas.c_value(rho_g_L)*
                                     this->EOS_phase_gas.c_value(rho_g_L) -
                                     static_cast<Number>(2.0/9.0)*sigma*Sigma_d_L*inv_rho_L);

    /*--- Compute the quantities needed for the maximum eigenvalue estimate for the right state ---*/
    const auto rho_R     = m_l_R + m_g_R + m_d_R;
    const auto inv_rho_R = static_cast<Number>(1.0)/rho_R;
    const auto vel_d_R   = qR(RHO_U_INDEX + curr_d)*inv_rho_R;

    const auto alpha_l_R = rho_alpha_l_R*inv_rho_R;
    const auto alpha_d_R = alpha_l_R*m_d_R/m_l_R; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto alpha_g_R = static_cast<Number>(1.0) - alpha_l_R - alpha_d_R;

    const auto Y_g_R     = m_g_R*inv_rho_R;
    const auto rho_liq_R = (m_l_R + m_d_R)/(alpha_l_R + alpha_d_R); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rho_g_R   = m_g_R/alpha_g_R; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto Sigma_d_R = rho_z_R/std::cbrt(rho_liq_R*rho_liq_R);
    const auto c_R       = std::sqrt((static_cast<Number>(1.0) - Y_g_R)*
                                     EOS_phase_liq.c_value(rho_liq_R)*
                                     EOS_phase_liq.c_value(rho_liq_R) +
                                     Y_g_R*EOS_phase_gas.c_value(rho_g_R)*
                                     EOS_phase_gas.c_value(rho_g_R) -
                                     static_cast<Number>(2.0/9.0)*sigma*Sigma_d_R*inv_rho_R);

    /*--- Compute the estimate of the eigenvalue ---*/
    const auto lambda = std::max(std::abs(vel_d_L) + c_L,
                                 std::abs(vel_d_R) + c_R);

    return static_cast<Number>(0.5)*
           (this->evaluate_hyperbolic_operator(qL, curr_d) +
            this->evaluate_hyperbolic_operator(qR, curr_d)) - // centered contribution
           static_cast<Number>(0.5)*lambda*(qR - qL); // upwinding contribution
  }

  // Implement the contribution of the discrete flux for all the directions.
  //
  template<class Field>
  #ifdef ORDER_2
    template<typename Field_Scalar>
    auto RusanovFlux<Field>::make_two_scale_capillarity(const Field_Scalar& H_bar)
  #else
    auto RusanovFlux<Field>::make_two_scale_capillarity()
  #endif
  {
    FluxDefinition<cfg> Rusanov_f;

    /*--- Perform the loop over each dimension to compute the flux contribution ---*/
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
         {
           static constexpr int d = decltype(integral_constant_d)::value;

           // Compute now the "discrete" flux function, in this case a Rusanov flux
           Rusanov_f[d].cons_flux_function = [&](FluxValue<cfg>& flux,
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

    auto scheme = make_flux_based_scheme(Rusanov_f);
    scheme.set_name("Rusanov");

    return scheme;
  }

} // end of namespace

#endif
