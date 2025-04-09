// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#ifndef Rusanov_flux_hpp
#define Rusanov_flux_hpp

#include "flux_base.hpp"

//#define VERBOSE_FLUX

namespace samurai {
  using namespace EquationData;

  /**
    * Implementation of a Rusanov flux
    */
  template<class Field>
  class RusanovFlux: public Flux<Field> {
  public:
    RusanovFlux(const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase1,
                const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase2,
                const double sigma_,
                const double mod_grad_alpha1_min_,
                const double lambda_ = 0.9,
                const double atol_Newton_ = 1e-12,
                const double rtol_Newton_ = 1e-10,
                const std::size_t max_Newton_iters_ = 60); /*--- Constructor which accepts in input the equations of state of the two phases ---*/

    #ifdef ORDER_2
      template<typename Field_Scalar>
      auto make_flux(const Field_Scalar& H); /*--- Compute the flux over all the directions ---*/
    #else
      auto make_flux(); /*--- Compute the flux over all the directions ---*/
    #endif

  private:
    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                               const FluxValue<typename Flux<Field>::cfg>& qR,
                                                               const std::size_t curr_d); /*--- Rusanov flux along direction curr_d ---*/
  };

  // Constructor derived from the base class
  //
  template<class Field>
  RusanovFlux<Field>::RusanovFlux(const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase1,
                                  const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase2,
                                  const double sigma_,
                                  const double mod_grad_alpha1_min_,
                                  const double lambda_,
                                  const double atol_Newton_,
                                  const double rtol_Newton_,
                                  const std::size_t max_Newton_iters_):
    Flux<Field>(EOS_phase1, EOS_phase2, sigma_, mod_grad_alpha1_min_, lambda_, atol_Newton_, rtol_Newton_, max_Newton_iters_) {}

  // Implementation of a Rusanov flux
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> RusanovFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                                                 const FluxValue<typename Flux<Field>::cfg>& qR,
                                                                                 const std::size_t curr_d) {
    /*--- Compute the quantities needed for the maximum eigenvalue estimate for the left state ---*/
    const auto rho_L       = qL(M1_INDEX) + qL(M2_INDEX);
    const auto vel_d_L     = qL(RHO_U_INDEX + curr_d)/rho_L;

    const auto alpha1_L    = qL(RHO_ALPHA1_INDEX)/rho_L;
    const auto rho1_L      = qL(M1_INDEX)/alpha1_L; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rho2_L      = qL(M2_INDEX)/(1.0 - alpha1_L); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto c_squared_L = qL(M1_INDEX)*this->phase1.c_value(rho1_L)*this->phase1.c_value(rho1_L)
                           + qL(M2_INDEX)*this->phase2.c_value(rho2_L)*this->phase2.c_value(rho2_L);
    const auto c_L         = std::sqrt(c_squared_L/rho_L);

    /*--- Compute the quantities needed for the maximum eigenvalue estimate for the right state ---*/
    const auto rho_R       = qR(M1_INDEX) + qR(M2_INDEX);
    const auto vel_d_R     = qR(RHO_U_INDEX + curr_d)/rho_R;

    const auto alpha1_R    = qR(RHO_ALPHA1_INDEX)/rho_R;;
    const auto rho1_R      = qR(M1_INDEX)/alpha1_R; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rho2_R      = qR(M2_INDEX)/(1.0 - alpha1_R); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto c_squared_R = qR(M1_INDEX)*this->phase1.c_value(rho1_R)*this->phase1.c_value(rho1_R)
                           + qR(M2_INDEX)*this->phase2.c_value(rho2_R)*this->phase2.c_value(rho2_R);
    const auto c_R         = std::sqrt(c_squared_R/rho_R);

    /*--- Compute the estimate of the eigenvalue considering also the surface tension contribution ---*/
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
    auto RusanovFlux<Field>::make_flux(const Field_Scalar& H)
  #else
    auto RusanovFlux<Field>::make_flux()
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
                                                    this->relax_reconstruction(qL, H[data.cells[1]]);
                                                    this->relax_reconstruction(qR, H[data.cells[2]]);
                                                  #endif
                                                #else
                                                  // Extract the states
                                                  const FluxValue<typename Flux<Field>::cfg> qL = field[0];
                                                  const FluxValue<typename Flux<Field>::cfg> qR = field[1];
                                                #endif

                                                // Compute the numerical flux
                                                flux = compute_discrete_flux(qL, qR, d);
                                          };
    });

    return make_flux_based_scheme(Rusanov_f);
  }

} // end of namespace

#endif
