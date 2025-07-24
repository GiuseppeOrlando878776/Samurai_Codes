// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
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
    RusanovFlux(const EOS<typename Field::value_type>& EOS_); /*--- Constructor which accepts in inputs the equation of state ---*/

    auto make_flux(); /*--- Compute the flux over all the faces and directions ---*/

  private:
    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                               const FluxValue<typename Flux<Field>::cfg>& qR,
                                                               const std::size_t curr_d); /*--- Rusanov flux along direction d ---*/
  };

  // Constructor derived from base class
  //
  template<class Field>
  RusanovFlux<Field>::RusanovFlux(const EOS<typename Field::value_type>& EOS_):
    Flux<Field>(EOS_) {}

  // Implementation of a Rusanov flux
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> RusanovFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                                                 const FluxValue<typename Flux<Field>::cfg>& qR,
                                                                                 std::size_t curr_d) {
    /*--- Left state ---*/
    const auto velL_d = qL(RHOU_INDEX + curr_d)/qL(RHO_INDEX);
    auto eL = qL(RHOE_INDEX)/qL(RHO_INDEX);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      eL -= static_cast<typename Field::value_type>(0.5)*
            (qL(RHOU_INDEX + d)/qL(RHO_INDEX))*(qL(RHOU_INDEX + d)/qL(RHO_INDEX));
    }
    const auto pL = this->Euler_EOS.pres_value(qL(RHO_INDEX), eL);
    const auto cL = this->Euler_EOS.c_value(qL(RHO_INDEX), pL);

    #ifdef VERBOSE_FLUX
      if(qL(RHO_INDEX) < static_cast<typename Field::value_type>(0.0)) {
        throw std::runtime_error(std::string("Negative density left state: " + std::to_string(qL(RHO_INDEX))));
      }
      if(pL < static_cast<typename Field::value_type>(0.0)) {
        throw std::runtime_error(std::string("Negative pressure left state: " + std::to_string(pL)));
      }
    #endif

    /*--- Right state ---*/
    const auto velR_d = qR(RHOU_INDEX + curr_d)/qR(RHO_INDEX);
    auto eR = qR(RHOE_INDEX)/qR(RHO_INDEX);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      eR -= static_cast<typename Field::value_type>(0.5)*
            (qR(RHOU_INDEX + d)/qR(RHO_INDEX))*(qR(RHOU_INDEX + d)/qR(RHO_INDEX));
    }
    const auto pR = this->Euler_EOS.pres_value(qR(RHO_INDEX), eR);
    const auto cR = this->Euler_EOS.c_value(qR(RHO_INDEX), pR);

    #ifdef VERBOSE_FLUX
      if(qR(RHO_INDEX) < static_cast<typename Field::value_type>(0.0)) {
        throw std::runtime_error(std::string("Negative density right state: " + std::to_string(qR(RHO_INDEX))));
      }
      if(pR < static_cast<typename Field::value_type>(0.0)) {
        throw std::runtime_error(std::string("Negative pressure right state: " + std::to_string(pR)));
      }
    #endif

    // Compute Rusanov flux
    const auto lambda = std::max(std::abs(velL_d) + cL, std::abs(velR_d) + cR);

    return static_cast<typename Field::value_type>(0.5)*
           (this->evaluate_continuous_flux(qL, curr_d) +
            this->evaluate_continuous_flux(qR, curr_d)) - // centered contribution
           static_cast<typename Field::value_type>(0.5)*lambda*(qR - qL); // upwinding contribution
  }

  // Implement the contribution of the discrete flux for all the dimensions.
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
