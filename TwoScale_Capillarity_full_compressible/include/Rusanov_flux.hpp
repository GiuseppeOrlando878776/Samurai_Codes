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
    RusanovFlux(const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase_liq_,
                const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase_gas_,
                const double sigma_,
                const double mod_grad_alpha_l_min_,
                const double lambda_,
                const double atol_Newton_,
                const double rtol_Newton_,
                const std::size_t max_Newton_iters_); /*--- Constructor which accepts in input the equations of state of the two phases ---*/

    #ifdef ORDER_2
      template<typename Field_Scalar>
      auto make_two_scale_capillarity(const Field_Scalar& H); /*--- Compute the flux over all the directions ---*/
    #else
      auto make_two_scale_capillarity(); /*--- Compute the flux over all the directions ---*/
    #endif

  private:
    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                               const FluxValue<typename Flux<Field>::cfg>& qR,
                                                               const std::size_t curr_d); /*--- Rusanov flux along direction curr_d ---*/
  };

  // Constructor derived from the base class
  //
  template<class Field>
  RusanovFlux<Field>::RusanovFlux(const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase_liq_,
                                  const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase_gas_,
                                  const double sigma_,
                                  const double mod_grad_alpha_l_min_,
                                  const double lambda_,
                                  const double atol_Newton_,
                                  const double rtol_Newton_,
                                  const std::size_t max_Newton_iters_):
    Flux<Field>(EOS_phase_liq_, EOS_phase_gas_,
                sigma_, mod_grad_alpha_l_min_,
                lambda_, atol_Newton_, rtol_Newton_, max_Newton_iters_) {}

  // Implementation of a Rusanov flux
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> RusanovFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                                                 const FluxValue<typename Flux<Field>::cfg>& qR,
                                                                                 const std::size_t curr_d) {
    /*--- Verify if left and right state are coherent ---*/
    #ifdef VERBOSE_FLUX
      if(qL(Ml_INDEX) < 0.0) {
        throw std::runtime_error(std::string("Negative mass large-scale liquid left state: " + std::to_string(qL(Ml_INDEX))));
      }
      if(qL(Mg_INDEX) < 0.0) {
        throw std::runtime_error(std::string("Negative mass gas left state: " + std::to_string(qL(Mg_INDEX))));
      }
      if(qL(Md_INDEX) < 0.0) {
        throw std::runtime_error(std::string("Negative mass small-scale liquid left state: " + std::to_string(qL(Md_INDEX))));
      }
      if(qL(RHO_ALPHA_l_INDEX) < 0.0) {
        throw std::runtime_error(std::string("Negative volume fraction large-scale liquid left state: " + std::to_string(qL(RHO_ALPHA_l_INDEX))));
      }

      if(qR(Ml_INDEX) < 0.0) {
        throw std::runtime_error(std::string("Negative mass large-scale liquid right state: " + std::to_string(qR(Ml_INDEX))));
      }
      if(qR(Mg_INDEX) < 0.0) {
        throw std::runtime_error(std::string("Negative mass gas right state: " + std::to_string(qR(Mg_INDEX))));
      }
      if(qR(Md_INDEX) < 0.0) {
        throw std::runtime_error(std::string("Negative mass small-scale liquid right state: " + std::to_string(qR(Md_INDEX))));
      }
      if(qR(RHO_ALPHA_l_INDEX) < 0.0) {
        throw std::runtime_error(std::string("Negative volume fraction large-scale liquid right state: " + std::to_string(qR(RHO_ALPHA_l_INDEX))));
      }
    #endif

    /*--- Compute the quantities needed for the maximum eigenvalue estimate for the left state ---*/
    const auto rho_L     = qL(Ml_INDEX) + qL(Mg_INDEX) + qL(Md_INDEX);
    const auto vel_d_L   = qL(RHO_U_INDEX + curr_d)/rho_L;

    const auto alpha_l_L = qL(RHO_ALPHA_l_INDEX)/rho_L;
    const auto alpha_d_L = alpha_l_L*qL(Md_INDEX)/qL(Ml_INDEX); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto alpha_g_L = 1.0 - alpha_l_L - alpha_d_L;

    const auto Y_g_L     = qL(Mg_INDEX)/rho_L;
    const auto rho_liq_L = (qL(Ml_INDEX) + qL(Md_INDEX))/(alpha_l_L + alpha_d_L); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rho_g_L   = qL(Mg_INDEX)/alpha_g_L; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto Sigma_d_L = qL(RHO_Z_INDEX)/std::pow(rho_liq_L, 2.0/3.0);
    const auto c_L       = std::sqrt((1.0 - Y_g_L)*
                                     this->EOS_phase_liq.c_value(rho_liq_L)*
                                     this->EOS_phase_liq.c_value(rho_liq_L) +
                                     Y_g*
                                     this->EOS_phase_gas.c_value(rho_g_L)*
                                     this->EOS_phase_gas.c_value(rho_g_L) -
                                     2.0/9.0*sigma*Sigma_d_L/rho_L);

    /*--- Compute the quantities needed for the maximum eigenvalue estimate for the right state ---*/
    const auto rho_R     = qR(Ml_INDEX) + qR(Mg_INDEX) + qR(Md_INDEX);
    const auto vel_d_R   = qR(RHO_U_INDEX + curr_d)/rho_R;

    const auto alpha_l_R = qR(RHO_ALPHA_l_INDEX)/rho_R;
    const auto alpha_d_R = alpha_l_R*qR(Md_INDEX)/qR(Ml_INDEX); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto alpha_g_R = 1.0 - alpha_l_R - alpha_d_R;

    const auto Y_g_R     = qR(Mg_INDEX)/rho_R;
    const auto rho_liq_R = (qR(Ml_INDEX) + qR(Md_INDEX))/(alpha_l_R + alpha_d_R); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rho_g_R   = qR(Mg_INDEX)/alpha_g_R; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto Sigma_d_R = qR(RHO_Z_INDEX)/std::pow(rho_liq_R, 2.0/3.0);
    const auto c_R       = std::sqrt((1.0 - Y_g_R)*
                                     EOS_phase_liq.c_value(rho_liq_R)*
                                     EOS_phase_liq.c_value(rho_liq_R) +
                                     Y_g*EOS_phase_gas.c_value(rho_g_R)*
                                     EOS_phase_gas.c_value(rho_g_R) -
                                     2.0/9.0*sigma*Sigma_d_R/rho_R);

    /*--- Compute the estimate of the eigenvalue ---*/
    const auto lambda = std::max(std::abs(vel_d_L) + c_L,
                                 std::abs(vel_d_R) + c_R);

    return 0.5*(this->evaluate_hyperbolic_operator(qL, curr_d) +
                this->evaluate_hyperbolic_operator(qR, curr_d)) - // centered contribution
           0.5*lambda*(qR - qL); // upwinding contribution
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
    FluxDefinition<typename Flux<Field>::cfg> Rusanov_f;

    /*--- Perform the loop over each dimension to compute the flux contribution ---*/
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        // Compute now the "discrete" flux function, in this case a Rusanov flux
        Rusanov_f[d].cons_flux_function = [&](samurai::FluxValue<typename Flux<Field>::cfg>& flux,
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
                                                    this->relax_reconstruction(qL, H_bar[data.cells[1]][0]);
                                                    this->relax_reconstruction(qR, H_bar[data.cells[2]][0]);
                                                  #endif
                                                #else
                                                  // Extract the states
                                                  const FluxValue<typename Flux<Field>::cfg> qL = field[0];
                                                  const FluxValue<typename Flux<Field>::cfg> qR = field[1];
                                                #endif

                                                // Compute the numerical flux
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
