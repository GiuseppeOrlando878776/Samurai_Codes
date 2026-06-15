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
   * Implementation of the surface tension contribution
   */
  template<class Field, class Field_Vect>
  class SurfaceTensionFlux: public Flux<Field> {
  public:
    static_assert(Field_Vect::n_comp == Field::dim, "The spatial dimensions between Field_Vect and Field do not match");

    using Number = Flux<Field>::Number; // Define the shortcut for the arithmetic type
    using cfg_st = FluxConfig<SchemeType::NonLinear, Flux<Field>::stencil_size, Field, Field_Vect>; // Shortcut to specify the type of configuration
                                                                                                    // for the flux (nonlinear in this case
                                                                                                    // with input_size different than output_size)

    /**
     * Class constructor
     * @param EOS_phase_liq_ liquid equation of state
     * @param EOS_phase_gas_ gas equation of state
     * @param sigma_ surface tension coefficient
     */
    SurfaceTensionFlux(const EOS<Number>& EOS_phase_liq_,
                       const EOS<Number>& EOS_phase_gas_,
                       const Number sigma_);

    /**
     * Compute the flux over all the directions
     */
    auto make_two_scale_capillarity();

  private:
    /**
     * 'Continuous' surface tension contribution
     * @param grad_alpha_l gradient of large-scale volume fraction
     * @param curr_d current direction
     */
    FluxValue<cfg_st> evaluate_surface_tension_operator(const auto& grad_alpha_l,
                                                        const std::size_t curr_d);

    /**
     * Surface tension contribution
     * @param grad_alpha_l_L left state
     * @param grad_alpha_l_R right state
     * @param curr_d current direction
     */
    FluxValue<cfg_st> compute_discrete_flux(const auto& grad_alpha_l_L,
                                            const auto& grad_alpha_l_R,
                                            const std::size_t curr_d);
  };

  // Constructor derived from the base class
  //
  template<class Field, class Field_Vect>
  SurfaceTensionFlux<Field, Field_Vect>::SurfaceTensionFlux(const EOS<Number>& EOS_phase_liq_,
                                                            const EOS<Number>& EOS_phase_gas_,
                                                            const Number sigma_):
    Flux<Field>(EOS_phase_liq_, EOS_phase_gas_, sigma_) {}

  // Evaluate the surface tension operator
  //
  /* NOTE: It is worth to remark that no contribution of the
           surface tension in the energy balances will be included. This is related
           to the fact that for the capillarity we are considering a change of
           variables and resolving using the local augmented internal energies
           for the sake of simplicity. */
  template<class Field, class Field_Vect>
  FluxValue<typename SurfaceTensionFlux<Field, Field_Vect>::cfg_st>
  SurfaceTensionFlux<Field, Field_Vect>::evaluate_surface_tension_operator(const auto& grad_alpha_l,
                                                                           const std::size_t curr_d) {
    // Sanity check in terms of dimensions
    assert(curr_d < Field::dim);

    // Initialize the resulting variable
    FluxValue<cfg_st> res;

    // Set to zero all the contributions
    res.fill(static_cast<Number>(0.0));

    // Add the contribution due to surface tension
    auto mod2_grad_alpha_l = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      mod2_grad_alpha_l += grad_alpha_l[d]*grad_alpha_l[d];
    }
    const auto mod_grad_alpha_l = std::sqrt(mod2_grad_alpha_l);

    const auto n  = grad_alpha_l/(mod_grad_alpha_l + static_cast<Number>(1e-10));
    const auto nx = n(0);
    const auto ny = n(1);

    if(curr_d == 0) {
      res(RHO_U_INDEX) += this->sigma*(nx*nx - static_cast<Number>(1.0))*mod_grad_alpha_l;
      res(RHO_U_INDEX + 1) += this->sigma*nx*ny*mod_grad_alpha_l;
    }
    else if(curr_d == 1) {
      res(RHO_U_INDEX) += this->sigma*nx*ny*mod_grad_alpha_l;
      res(RHO_U_INDEX + 1) += this->sigma*(ny*ny - static_cast<Number>(1.0))*mod_grad_alpha_l;
    }

    return res;
  }

  // Implementation of the surface tension contribution
  //
  template<class Field, class Field_Vect>
  FluxValue<typename SurfaceTensionFlux<Field, Field_Vect>::cfg_st>
  SurfaceTensionFlux<Field, Field_Vect>::compute_discrete_flux(const auto& grad_alpha_l_L,
                                                               const auto& grad_alpha_l_R,
                                                               const std::size_t curr_d) {
    return static_cast<Number>(0.5)*
           (this->evaluate_surface_tension_operator(grad_alpha_l_L, curr_d) +
            this->evaluate_surface_tension_operator(grad_alpha_l_R, curr_d));
  }

  // Implement the contribution of the discrete flux for all the directions.
  //
  template<class Field, class Field_Vect>
  auto SurfaceTensionFlux<Field, Field_Vect>::make_two_scale_capillarity() {
    FluxDefinition<cfg_st> SurfaceTension_f;

    // Perform the loop over each dimension to compute the flux contribution
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
