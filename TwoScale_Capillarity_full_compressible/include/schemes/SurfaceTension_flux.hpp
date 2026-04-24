// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2026
//
#pragma once

#include "flux_base.hpp"

namespace samurai {
  using namespace EquationData;

  /**
    * Implementation of the surface tensino contribution
    */
  template<class Field, class Field_Vect>
  class SurfaceTensionFlux: public Flux<Field> {
  public:
    static_assert(Field_Vect::n_comp == Field::dim, "The spatial dimensions between Field_Vect and Field do not match");

    using Number = Flux<Field>::Number; /*--- Define the shortcut for the arithmetic type ---*/
    using cfg_st = Flux<Field>::template cfg_st<Field_Vect>; /*--- Shortcut to specify the type of configuration
                                                                   for the flux (nonlinear in this case with input_size different than output_size) ---*/

    SurfaceTensionFlux(const LinearizedBarotropicEOS<Number>& EOS_phase_liq_,
                       const LinearizedBarotropicEOS<Number>& EOS_phase_gas_,
                       const Number sigma_,
                       const Number lambda_,
                       const Number atol_Newton_,
                       const Number rtol_Newton_,
                       const std::size_t max_Newton_iters_); /*--- Constructor which accepts in input the equations of state of the two phases ---*/

    auto make_two_scale_capillarity(); /*--- Compute the flux over all the directions ---*/

  private:
    FluxValue<cfg_st> compute_discrete_flux(const auto& grad_alpha_l_L,
                                            const auto& grad_alpha_l_R,
                                            const std::size_t curr_d); /*--- Surface tension contribution along direction curr_d ---*/
  };

  // Constructor derived from the base class
  //
  template<class Field, class Field_Vect>
  SurfaceTensionFlux<Field, Field_Vect>::SurfaceTensionFlux(const LinearizedBarotropicEOS<Number>& EOS_phase_liq_,
                                                            const LinearizedBarotropicEOS<Number>& EOS_phase_gas_,
                                                            const Number sigma_,
                                                            const Number lambda_,
                                                            const Number atol_Newton_,
                                                            const Number rtol_Newton_,
                                                            const std::size_t max_Newton_iters_):
    Flux<Field>(EOS_phase_liq_, EOS_phase_gas_, sigma_,
                lambda_, atol_Newton_, rtol_Newton_, max_Newton_iters_) {}

  // Implementation of the surface tension contribution
  //
  template<class Field, class Field_Vect>
  FluxValue<typename SurfaceTensionFlux<Field, Field_Vect>::cfg_st>
  SurfaceTensionFlux<Field, Field_Vect>::compute_discrete_flux(const auto& grad_alpha_l_L,
                                                               const auto& grad_alpha_l_R,
                                                               const std::size_t curr_d) {
    return static_cast<Number>(0.5)*
           (this->template evaluate_surface_tension_operator<Field_Vect>(grad_alpha_l_L, curr_d) +
            this->template evaluate_surface_tension_operator<Field_Vect>(grad_alpha_l_R, curr_d));
  }

  // Implement the contribution of the discrete flux for all the directions.
  //
  template<class Field, class Field_Vect>
  auto SurfaceTensionFlux<Field, Field_Vect>::make_two_scale_capillarity() {
    FluxDefinition<cfg_st> SurfaceTension_f;

    /*--- Perform the loop over each dimension to compute the flux contribution ---*/
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
         {
           static constexpr int d = decltype(integral_constant_d)::value;

           // Compute now the "discrete" flux function
           SurfaceTension_f[d].cons_flux_function = [&](FluxValue<cfg_st>& flux,
                                                        const StencilData<cfg_st>& /*data*/,
                                                        const StencilValues<cfg_st> field)
                                                        {
                                                          // Compute the numerical flux
                                                          #ifdef ORDER_2
                                                            flux = compute_discrete_flux(field[1],
                                                                                         field[2],
                                                                                         d);
                                                          #else
                                                            flux = compute_discrete_flux(field[0],
                                                                                         field[1],
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
