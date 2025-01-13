// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
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
    SurfaceTensionFlux(const LinearizedBarotropicEOS<>& EOS_phase1,
                       const LinearizedBarotropicEOS<>& EOS_phase2,
                       const double sigma_,
                       const double mod_grad_alpha1_min_); // Constructor which accepts in inputs the equations of state of the two phases

    template<typename Gradient>
    auto make_two_scale_capillarity(const Gradient& grad_alpha1); // Compute the flux over all the directions

  private:
    template<typename Gradient>
    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux(const Gradient& grad_alpha1L,
                                                               const Gradient& grad_alpha1R,
                                                               const std::size_t curr_d); // Surface tension contribution along direction curr_d
  };

  // Constructor derived from the base class
  //
  template<class Field>
  SurfaceTensionFlux<Field>::SurfaceTensionFlux(const LinearizedBarotropicEOS<>& EOS_phase1,
                                                const LinearizedBarotropicEOS<>& EOS_phase2,
                                                const double sigma_,
                                                const double grad_alpha1_min_):
    Flux<Field>(EOS_phase1, EOS_phase2, sigma_, grad_alpha1_min_) {}

  // Implementation of the surface tension contribution
  //
  template<class Field>
  template<typename Gradient>
  FluxValue<typename Flux<Field>::cfg> SurfaceTensionFlux<Field>::compute_discrete_flux(const Gradient& grad_alpha1L,
                                                                                        const Gradient& grad_alpha1R,
                                                                                        const std::size_t curr_d) {
    return 0.5*(this->evaluate_surface_tension_operator(grad_alpha1L, curr_d) +
                this->evaluate_surface_tension_operator(grad_alpha1R, curr_d));
  }

  // Implement the contribution of the discrete flux for all the directions.
  //
  template<class Field>
  template<typename Gradient>
  auto SurfaceTensionFlux<Field>::make_two_scale_capillarity(const Gradient& grad_alpha1) {
    FluxDefinition<typename Flux<Field>::cfg> SurfaceTension_f;

    // Perform the loop over each dimension to compute the flux contribution
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        // Compute now the "discrete" flux function
        SurfaceTension_f[d].cons_flux_function = [&](auto& cells, const Field& field)
                                                    {
                                                      // Compute the stencil
                                                      #ifdef ORDER_2
                                                        const auto& left  = cells[1];
                                                        const auto& right = cells[2];
                                                      #else
                                                        const auto& left  = cells[0];
                                                        const auto& right = cells[1];
                                                      #endif

                                                      // Compute the numerical flux
                                                      return compute_discrete_flux(grad_alpha1[left], grad_alpha1[right], d);
                                                    };
    });

    return make_flux_based_scheme(SurfaceTension_f);
  }

} // end of namespace

#endif
