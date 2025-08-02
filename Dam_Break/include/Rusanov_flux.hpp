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
    RusanovFlux(const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase1_,
                const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase2_,
                const typename Field::value_type lambda_,
                const typename Field::value_type atol_Newton_,
                const typename Field::value_type rtol_Newton_,
                const std::size_t max_Newton_iters_); /*--- Constructor which accepts in input the equations of state of the two phases ---*/

    auto make_flux(); /*--- Compute the flux over all the directions ---*/

  private:
    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                               const FluxValue<typename Flux<Field>::cfg>& qR,
                                                               const std::size_t curr_d); /*--- Rusanov flux along direction curr_d ---*/
  };

  // Constructor derived from the base class
  //
  template<class Field>
  RusanovFlux<Field>::RusanovFlux(const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase1_,
                                  const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase2_,
                                  const typename Field::value_type lambda_,
                                  const typename Field::value_type atol_Newton_,
                                  const typename Field::value_type rtol_Newton_,
                                  const std::size_t max_Newton_iters_):
    Flux<Field>(EOS_phase1_, EOS_phase2_,
                lambda_, atol_Newton_, rtol_Newton_, max_Newton_iters_) {}

  // Implementation of a Rusanov flux
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> RusanovFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                                                 const FluxValue<typename Flux<Field>::cfg>& qR,
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
      if(m1_L < static_cast<typename Field::value_type>(0.0)) {
        throw std::runtime_error(std::string("Negative mass phase 1 left state: " + std::to_string(m1_L)));
      }
      if(m2_L < static_cast<typename Field::value_type>(0.0)) {
        throw std::runtime_error(std::string("Negative mass phase 2 left state: " + std::to_string(m2_L)));
      }
      if(rho_alpha1_L < static_cast<typename Field::value_type>(0.0)) {
        throw std::runtime_error(std::string("Negative volume fraction phase 1 left state: " + std::to_string(rho_alpha1_L)));
      }

      if(m1_R < static_cast<typename Field::value_type>(0.0)) {
        throw std::runtime_error(std::string("Negative mass phase 1 right state: " + std::to_string(m1_R)));
      }
      if(m2_R < static_cast<typename Field::value_type>(0.0)) {
        throw std::runtime_error(std::string("Negative mass phase 2 right state: " + std::to_string(m2_R)));
      }
      if(rho_alpha1_R < static_cast<typename Field::value_type>(0.0)) {
        throw std::runtime_error(std::string("Negative volume fraction phase 1 right state: " + std::to_string(rho_alpha1_R)));
      }
    #endif

    /*--- Compute the quantities needed for the maximum eigenvalue estimate for the left state ---*/
    const auto rho_L     = m1_L + m2_L;
    const auto inv_rho_L = static_cast<typename Field::value_type>(1.0)/rho_L;
    const auto vel_d_L   = qL(RHO_U_INDEX + curr_d)*inv_rho_L;

    const auto alpha1_L  = rho_alpha1_L*inv_rho_L;
    const auto rho1_L    = m1_L/alpha1_L; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rho2_L    = m2_L/(static_cast<typename Field::value_type>(1.0) - alpha1_L); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto Y1_L      = m1_L*inv_rho_L;
    const auto c_L       = std::sqrt(Y1_L*
                                     this->EOS_phase1.c_value(rho1_L)*
                                     this->EOS_phase1.c_value(rho1_L) +
                                     (static_cast<typename Field::value_type>(1.0) - Y1_L)*
                                     this->EOS_phase2.c_value(rho2_L)*
                                     this->EOS_phase2.c_value(rho2_L));

    /*--- Compute the quantities needed for the maximum eigenvalue estimate for the right state ---*/
    const auto rho_R     = m1_R + m2_R;
    const auto inv_rho_R = static_cast<typename Field::value_type>(1.0)/rho_R;
    const auto vel_d_R   = qR(RHO_U_INDEX + curr_d)*inv_rho_R;

    const auto alpha1_R  = rho_alpha1_R*inv_rho_R;
    const auto rho1_R    = m1_R/alpha1_R; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rho2_R    = m2_R/(static_cast<typename Field::value_type>(1.0) - alpha1_R); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto Y1_R      = m1_R*inv_rho_R;
    const auto c_R       = std::sqrt(Y1_R*
                                     this->EOS_phase1.c_value(rho1_R)*
                                     this->EOS_phase1.c_value(rho1_R) +
                                     (static_cast<typename Field::value_type>(1.0) - Y1_R)*
                                     this->EOS_phase2.c_value(rho2_R)*
                                     this->EOS_phase2.c_value(rho2_R));

    /*--- Compute the estimate of the eigenvalue ---*/
    const auto lambda = std::max(std::abs(vel_d_L) + c_L,
                                 std::abs(vel_d_R) + c_R);

    return static_cast<typename Field::value_type>(0.5)*
           (this->evaluate_continuous_flux(qL, curr_d) +
            this->evaluate_continuous_flux(qR, curr_d)) - // centered contribution
           static_cast<typename Field::value_type>(0.5)*lambda*(qR - qL); // upwinding contribution
  }

  // Implement the contribution of the discrete flux for all the directions.
  //
  template<class Field>
  auto RusanovFlux<Field>::make_flux() {
    FluxDefinition<typename Flux<Field>::cfg> Rusanov_f;

    /*--- Perform the loop over each dimension to compute the flux contribution ---*/
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
         {
           static constexpr int d = decltype(integral_constant_d)::value;

           // Compute now the "discrete" flux function, in this case a Rusanov flux
           Rusanov_f[d].cons_flux_function = [&](samurai::FluxValue<typename Flux<Field>::cfg>& flux,
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

                                                     #ifdef RELAX_RECONSTRUCTION
                                                       this->relax_reconstruction(qL);
                                                       this->relax_reconstruction(qR);
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

    auto scheme = make_flux_based_scheme(Rusanov_f);
    scheme.set_name("Rusanov");

    return scheme;
  }

} // end of namespace

#endif
