// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#pragma once

#include "flux_base.hpp"

#include "Suliciu_base/eos.hpp"
#include "Suliciu_base/Riemannsol.hpp"
#include "Suliciu_base/flux_numeriques.hpp"

namespace samurai {
  /**
    * Implementation of the flux based on Suliciu-type relaxation
    */
  template<class Field>
  class RelaxationFlux {
  public:
    using Indices = Flux<Field>::Indices; /*--- Shortcut for the indices storage ---*/
    using Number  = Flux<Field>::Number;  /*--- Shortcut for the arithmetic type ---*/
    using cfg     = Flux<Field>::cfg;     /*--- Shortcut to specify the type of configuration
                                                for the flux (nonlinear in this case) ---*/

    RelaxationFlux() = default; /*--- Default constructor ---*/

    auto make_flux(Number& c); /*--- Compute the flux over all faces and directions.
                                     The input argument is employed to compute the Courant number ---*/

  private:
    template<class Other>
    Other FluxValue_to_Other(const FluxValue<cfg>& q); /*--- Conversion from Samurai to other ("analogous")
                                                             data structure for the state ---*/

    template<class Other>
    FluxValue<cfg> Other_to_FluxValue(const Other& q); /*--- Conversion from other ("analogous")
                                                             data structure for the state to Samurai ---*/

    void compute_discrete_flux(const FluxValue<cfg>& qL,
                               const FluxValue<cfg>& qR,
                               const std::size_t curr_d,
                               FluxValue<cfg>& F_minus,
                               FluxValue<cfg>& F_plus,
                               Number& c); /*--- Compute discrete flux ---*/
  };

  // Conversion from Samurai to other ("analogous") data structure for the state
  //
  template<class Field>
  template<class Other>
  Other RelaxationFlux<Field>::FluxValue_to_Other(const FluxValue<cfg>& q) {
    return Other(q(Indices::ALPHA1_INDEX),
                 q(Indices::ALPHA1_RHO1_INDEX),
                 q(Indices::ALPHA1_RHO1_U1_INDEX),
                 q(Indices::ALPHA1_RHO1_E1_INDEX),
                 q(Indices::ALPHA2_RHO2_INDEX),
                 q(Indices::ALPHA2_RHO2_U2_INDEX),
                 q(Indices::ALPHA2_RHO2_E2_INDEX));
  }

  // Conversion from Samurai to other ("analogous") data structure for the state
  //
  template<class Field>
  template<class Other>
  FluxValue<typename RelaxationFlux<Field>::cfg>
  RelaxationFlux<Field>::Other_to_FluxValue(const Other& q) {
    FluxValue<cfg> res;

    res(Indices::ALPHA1_INDEX)	       = q.al1;
    res(Indices::ALPHA1_RHO1_INDEX)	   = q.alrho1;
    res(Indices::ALPHA1_RHO1_U1_INDEX) = q.alrhou1;
    res(Indices::ALPHA1_RHO1_E1_INDEX) = q.alrhoE1;
    res(Indices::ALPHA2_RHO2_INDEX)	   = q.alrho2;
    res(Indices::ALPHA2_RHO2_U2_INDEX) = q.alrhou2;
    res(Indices::ALPHA2_RHO2_E2_INDEX) = q.alrhoE2;

    return res;
  }

  // Implementation of the Suliciu type flux
  //
  template<class Field>
  void RelaxationFlux<Field>::compute_discrete_flux(const FluxValue<cfg>& qL,
                                                    const FluxValue<cfg>& qR,
                                                    [[maybe_unused]] std::size_t curr_d,
                                                    FluxValue<cfg>& F_minus,
                                                    FluxValue<cfg>& F_plus,
                                                    Number& c) {
    /*--- Employ "Saleh et al." functions to compute the Suliciu "flux" ---*/
    int Newton_iter;
    int dissip;
    const double eps = 1e-7;
    auto fW_minus = Etat();
    auto fW_plus  = Etat();
    c = std::max(c, flux_relax(FluxValue_to_Other<Etat>(qL), FluxValue_to_Other<Etat>(qR),
                               fW_minus, fW_plus,
                               Newton_iter, dissip, eps));

    /*--- Conversion from Etat to FluxValue<cfg> ---*/
    F_minus = Other_to_FluxValue<Etat>(fW_minus);
    F_plus  = Other_to_FluxValue<Etat>(fW_plus);
  }

  // Implement the contribution of the discrete flux for all the directions.
  //
  template<class Field>
  auto RelaxationFlux<Field>::make_flux(double& c) {
    FluxDefinition<cfg> Suliciu_flux;

    /*--- Perform the loop over each dimension to compute the flux contribution ---*/
    static_for<0, EquationData::dim>::apply(
      [&](auto integral_constant_d)
         {
           static constexpr int d = decltype(integral_constant_d)::value;

           // Compute now the "discrete" non-conservative flux function
           Suliciu_flux[d].flux_function = [&](FluxValuePair<cfg>& flux,
                                               const StencilData<cfg>& /*data*/,
                                               const StencilValues<cfg> field)
                                               {
                                                 // Extract the states (no reoncstruction implemented)
                                                 #ifdef ORDER_2
                                                   const auto& qL = field[1];
                                                   const auto& qR = field[2];
                                                 #else
                                                   const auto& qL = field[0];
                                                   const auto& qR = field[1];

                                                 FluxValue<cfg> F_minus,
                                                                F_plus;

                                                 compute_discrete_flux(qL, qR, d, F_minus, F_plus, c);

                                                 flux[0] = F_minus;
                                                 flux[1] = -F_plus;

                                                 return flux;
                                               };
        }
    );

    auto scheme = make_flux_based_scheme(Suliciu_flux);
    scheme.set_name("Suliciu");

    return scheme;
  }

} // end of namespace
