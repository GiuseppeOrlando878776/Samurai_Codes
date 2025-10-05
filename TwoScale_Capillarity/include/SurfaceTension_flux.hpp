// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#pragma once

#include "flux_base.hpp"

namespace samurai {
  using namespace EquationData;

  /**
    * Implementation of the surface tensino contribution
    */
  template<class Field>
  class SurfaceTensionFlux: public Flux<Field> {
  public:
    using Number = Flux<Field>::Number; /*--- Define the shortcut for the arithmetic type ---*/
    using cfg    = Flux<Field>::cfg; /*--- Shortcut to specify the type of configuration
                                           for the flux (nonlinear in this case) ---*/

    SurfaceTensionFlux(const LinearizedBarotropicEOS<Number>& EOS_phase1_,
                       const LinearizedBarotropicEOS<Number>& EOS_phase2_,
                       const Number sigma_,
                       const Number mod_grad_alpha1_bar_min_,
                       const Number lambda_,
                       const Number atol_Newton_,
                       const Number rtol_Newton_,
                       const std::size_t max_Newton_iters_); /*--- Constructor which accepts in input the equations of state of the two phases ---*/

    template<typename Gradient>
    auto make_two_scale_capillarity(const Gradient& grad_alpha1_bar); /*--- Compute the flux over all the directions ---*/

  private:
    template<typename Gradient>
    FluxValue<cfg> compute_discrete_flux(const Gradient& grad_alpha1_bar_L,
                                         const Gradient& grad_alpha1_bar_R,
                                         const std::size_t curr_d); /*--- Surface tension contribution along direction curr_d ---*/
  };

  // Constructor derived from the base class
  //
  template<class Field>
  SurfaceTensionFlux<Field>::SurfaceTensionFlux(const LinearizedBarotropicEOS<Number>& EOS_phase1_,
                                                const LinearizedBarotropicEOS<Number>& EOS_phase2_,
                                                const Number sigma_,
                                                const Number mod_grad_alpha1_bar_min_,
                                                const Number lambda_,
                                                const Number atol_Newton_,
                                                const Number rtol_Newton_,
                                                const std::size_t max_Newton_iters_):
    Flux<Field>(EOS_phase1_, EOS_phase2_,
                sigma_, mod_grad_alpha1_bar_min_,
                lambda_, atol_Newton_, rtol_Newton_, max_Newton_iters_) {}

  // Implementation of the surface tension contribution
  //
  template<class Field>
  template<typename Gradient>
  FluxValue<typename SurfaceTensionFlux<Field>::cfg>
  SurfaceTensionFlux<Field>::compute_discrete_flux(const Gradient& grad_alpha1_bar_L,
                                                   const Gradient& grad_alpha1_bar_R,
                                                   const std::size_t curr_d) {
    return static_cast<Number>(0.5)*
           (this->evaluate_surface_tension_operator(grad_alpha1_bar_L, curr_d) +
            this->evaluate_surface_tension_operator(grad_alpha1_bar_R, curr_d));
  }

  // Implement the contribution of the discrete flux for all the directions.
  //
  template<class Field>
  template<typename Gradient>
  auto SurfaceTensionFlux<Field>::make_two_scale_capillarity(const Gradient& grad_alpha1_bar) {
    FluxDefinition<cfg> SurfaceTension_f;

    /*--- Perform the loop over each dimension to compute the flux contribution ---*/
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
         {
           static constexpr int d = decltype(integral_constant_d)::value;

           // Compute now the "discrete" flux function
           SurfaceTension_f[d].cons_flux_function = [&](FluxValue<cfg>& flux,
                                                        const StencilData<cfg>& data,
                                                        const StencilValues<cfg> /*field*/)
                                                        {
                                                          // Compute the numerical flux
                                                          #ifdef ORDER_2
                                                            flux = compute_discrete_flux(grad_alpha1_bar[data.cells[1]],
                                                                                         grad_alpha1_bar[data.cells[2]],
                                                                                         d);
                                                          #else
                                                            flux = compute_discrete_flux(grad_alpha1_bar[data.cells[0]],
                                                                                         grad_alpha1_bar[data.cells[1]],
                                                                                         d);
                                                          #endif
                                                        };
        }
    );

    auto scheme = make_flux_based_scheme(SurfaceTension_f);
    scheme.set_name("Surface tension");

    return scheme;
  }

} // end of namespace
