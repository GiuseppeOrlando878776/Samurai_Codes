// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#ifndef HLL_6eqs_flux_hpp
#define HLL_6eqs_flux_hpp

#include "flux_6eqs_base.hpp"

namespace samurai {
  using namespace EquationData;

  /**
    * Implementation of a HLL flux
    */
  template<class Field>
  class HLLFlux: public Flux<Field> {
  public:
    HLLFlux(const EOS<typename Field::value_type>& EOS_phase1,
            const EOS<typename Field::value_type>& EOS_phase2); // Constructor which accepts in input the equations of state of the two phases

    auto make_flux(); // Compute the flux over all cells

  private:
    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                               const FluxValue<typename Flux<Field>::cfg>& qR,
                                                               const std::size_t curr_d); // HLLC flux along direction d
  };

  // Constructor derived from base class
  //
  template<class Field>
  HLLFlux<Field>::HLLFlux(const EOS<typename Field::value_type>& EOS_phase1,
                          const EOS<typename Field::value_type>& EOS_phase2):
    Flux<Field>(EOS_phase1, EOS_phase2) {}

  // Implementation of a HLL flux
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> HLLFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                                             const FluxValue<typename Flux<Field>::cfg>& qR,
                                                                             std::size_t curr_d) {
    // Save mixture density and velocity current direction left state
    const auto rhoL   = qL(ALPHA1_RHO1_INDEX) + qL(ALPHA2_RHO2_INDEX);
    const auto velL_d = qL(RHO_U_INDEX + curr_d)/rhoL;

    // Left state phase 1
    const auto alpha1L = qL(ALPHA1_INDEX);
    const auto rho1L   = qL(ALPHA1_RHO1_INDEX)/alpha1L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1L           = qL(ALPHA1_RHO1_E1_INDEX)/qL(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d  = 0; d < Field::dim; ++d) {
      e1L -= 0.5*(qL(RHO_U_INDEX + d)/rhoL)*(qL(RHO_U_INDEX + d)/rhoL);
    }
    const auto p1L = this->phase1.pres_value(rho1L, e1L);
    const auto c1L = this->phase1.c_value(rho1L, p1L);

    // Left state phase 2
    const auto rho2L = qL(ALPHA2_RHO2_INDEX)/(1.0 - alpha1L); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2L         = qL(ALPHA2_RHO2_E2_INDEX)/qL(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < Field::dim; ++d) {
      e2L -= 0.5*(qL(RHO_U_INDEX + d)/rhoL)*(qL(RHO_U_INDEX + d)/rhoL);
    }
    const auto p2L = this->phase2.pres_value(rho2L, e2L);
    const auto c2L = this->phase2.c_value(rho2L, p2L);

    // Compute frozen speed of sound and mixture pressure left state
    const auto Y1L = qL(ALPHA1_RHO1_INDEX)/rhoL;
    const auto cL  = std::sqrt(Y1L*c1L*c1L + (1.0 - Y1L)*c2L*c2L);

    // Save mixture density and velocity current direction right state
    const auto rhoR   = qR(ALPHA1_RHO1_INDEX) + qR(ALPHA2_RHO2_INDEX);
    const auto velR_d = qR(RHO_U_INDEX + curr_d)/rhoR;

    // Right state phase 1
    const auto alpha1R = qR(ALPHA1_INDEX);
    const auto rho1R   = qR(ALPHA1_RHO1_INDEX)/alpha1R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1R           = qR(ALPHA1_RHO1_E1_INDEX)/qR(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < Field::dim; ++d) {
      e1R -= 0.5*(qR(RHO_U_INDEX + d)/rhoR)*(qR(RHO_U_INDEX + d)/rhoR);
    }
    const auto p1R = this->phase1.pres_value(rho1R, e1R);
    const auto c1R = this->phase1.c_value(rho1R, p1R);

    // Right state phase 2
    const auto rho2R = qR(ALPHA2_RHO2_INDEX)/(1.0 - alpha1R); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2R         = qR(ALPHA2_RHO2_E2_INDEX)/qR(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < Field::dim; ++d) {
      e2R -= 0.5*(qR(RHO_U_INDEX + d)/rhoR)*(qR(RHO_U_INDEX + d)/rhoR);
    }
    const auto p2R = this->phase2.pres_value(rho2R, e2R);
    const auto c2R = this->phase2.c_value(rho2R, p2R);

    // Compute frozen speed of sound and mixture pressure right state
    const auto Y1R = qR(ALPHA1_RHO1_INDEX)/rhoR;
    const auto cR  = std::sqrt(Y1R*c1R*c1R + (1.0 - Y1R)*c2R*c2R);

    // Compute speeds of wave propagation
    const auto sL = std::min(0.0, std::min(velL_d - cL, velR_d - cR));
    const auto sR = std::max(0.0, std::max(velL_d + cL, velR_d + cR));

    return (sR*this->evaluate_continuous_flux(qL, curr_d) - sL*this->evaluate_continuous_flux(qR, curr_d))/(sR - sL) + // centered contribution
           (sR*sL/(sR - sL))*(qR - qL); // upwinding contribution
  }

  // Implement the contribution of the discrete flux for all the dimensions.
  //
  template<class Field>
  auto HLLFlux<Field>::make_flux() {
    FluxDefinition<typename Flux<Field>::cfg> discrete_flux;

    /*--- Perform the loop over each dimension to compute the flux contribution ---*/
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        // Compute now the "discrete" flux function
        discrete_flux[d].cons_flux_function = [&](samurai::FluxValue<typename Flux<Field>::cfg>& flux,
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
                                                      const FluxValue<typename Flux<Field>::cfg>& qL = field[0];
                                                      const FluxValue<typename Flux<Field>::cfg>& qR = field[1];
                                                    #endif

                                                    flux = compute_discrete_flux(qL, qR, d);
                                                  };
      }
    );

    return make_flux_based_scheme(discrete_flux);
  }

} // end of namespace

#endif
