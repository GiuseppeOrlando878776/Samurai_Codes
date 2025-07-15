// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#ifndef SurfaceTension_flux_hpp
#define SurfaceTension_flux_hpp

#include "flux_base.hpp"

namespace samurai {
  using namespace EquationData;

  /**
    * Implementation of the surface tensino contribution
    */
  template<class Field>
  class SurfaceTensionFlux: public Flux<Field> {
  public:
    SurfaceTensionFlux(const LinearizedBarotropicEOS<>& EOS_phase_liq_,
                       const LinearizedBarotropicEOS<>& EOS_phase_gas_,
                       const double sigma_,
                       const double mod_grad_alpha_l_min_,
                       const double lambda_,
                       const double atol_Newton_,
                       const double rtol_Newton_,
                       const std::size_t max_Newton_iters_); /*--- Constructor which accepts in input the equations of state of the two phases ---*/

    template<typename Gradient>
    auto make_two_scale_capillarity(const Gradient& grad_alpha_l); /*--- Compute the flux over all the directions ---*/

  private:
    template<typename Gradient>
    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux(const Gradient& grad_alpha_l_L,
                                                               const Gradient& grad_alpha_l_R,
                                                               const std::size_t curr_d); /*--- Surface tension contribution along direction curr_d ---*/
  };

  // Constructor derived from the base class
  //
  template<class Field>
  SurfaceTensionFlux<Field>::SurfaceTensionFlux(const LinearizedBarotropicEOS<>& EOS_phase_liq_,
                                                const LinearizedBarotropicEOS<>& EOS_phase_gas_,
                                                const double sigma_,
                                                const double mod_grad_alpha_l_min_,
                                                const double lambda_,
                                                const double atol_Newton_,
                                                const double rtol_Newton_,
                                                const std::size_t max_Newton_iters_):
    Flux<Field>(EOS_phase_liq_, EOS_phase_gas_,
                sigma_, mod_grad_alpha_l_min_,
                lambda_, atol_Newton_, rtol_Newton_, max_Newton_iters_) {}

  // Implementation of the surface tension contribution
  //
  template<class Field>
  template<typename Gradient>
  FluxValue<typename Flux<Field>::cfg> SurfaceTensionFlux<Field>::compute_discrete_flux(const Gradient& grad_alpha_l_L,
                                                                                        const Gradient& grad_alpha_l_R,
                                                                                        const std::size_t curr_d) {
    return 0.5*(this->evaluate_surface_tension_operator(grad_alpha_l_L, curr_d) +
                this->evaluate_surface_tension_operator(grad_alpha_l_R, curr_d));
  }

  // Implement the contribution of the discrete flux for all the directions.
  //
  template<class Field>
  template<typename Gradient>
  auto SurfaceTensionFlux<Field>::make_two_scale_capillarity(const Gradient& grad_alpha_l) {
    FluxDefinition<typename Flux<Field>::cfg> SurfaceTension_f;

    /*--- Perform the loop over each dimension to compute the flux contribution ---*/
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        // Compute now the "discrete" flux function
        SurfaceTension_f[d].cons_flux_function = [&](samurai::FluxValue<typename Flux<Field>::cfg>& flux,
                                                     const StencilData<typename Flux<Field>::cfg>& data,
                                                     const StencilValues<typename Flux<Field>::cfg> /*field*/)
                                                     {
                                                        // Compute the numerical flux
                                                        #ifdef ORDER_2
                                                          flux = compute_discrete_flux(grad_alpha_l[data.cells[1]], grad_alpha_l[data.cells[2]], d);
                                                        #else
                                                          flux = compute_discrete_flux(grad_alpha_l[data.cells[0]], grad_alpha_l[data.cells[1]], d);
                                                        #endif
                                                     };
    });

    auto scheme = make_flux_based_scheme(SurfaceTension_f);
    scheme.set_name("Surface tension");

    return scheme;
  }

} // end of namespace

#endif
