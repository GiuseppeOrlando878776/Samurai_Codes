// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#ifndef HLL_flux_hpp
#define HLL_flux_hpp

#include "flux_base.hpp"

namespace samurai {
  using namespace EquationData;

  /**
    * Implementation of a HLL flux for the Euler system
    */
  template<class Field>
  class HLLFlux: public Flux<Field> {
  public:
    HLLFlux(const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase1,
            const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase2); // Constructor which accepts in inputs the equation of state of the two phases

    auto make_flux(); // Compute the flux over all cells

  private:
    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                               const FluxValue<typename Flux<Field>::cfg>& qR,
                                                               const std::size_t curr_d); // Rusanov flux along direction d
  };

  // Constructor derived from base class
  //
  template<class Field>
  HLLFlux<Field>::HLLFlux(const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase1,
                          const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase2):
    Flux<Field>(EOS_phase1, EOS_phase2) {}

  // Implementation of a HLL flux for the system
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg>
  HLLFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                         const FluxValue<typename Flux<Field>::cfg>& qR,
                                         std::size_t curr_d) {
    // Left state
    const auto rho_L       = qL(M1_INDEX) + qL(M2_INDEX);
    const auto vel_d_L     = qL(RHO_U_INDEX + curr_d)/rho_L;

    const auto alpha1_L    = qL(RHO_ALPHA1_INDEX)/rho_L;
    const auto rho1_L      = qL(M1_INDEX)/alpha1_L; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rho2_L      = qL(M2_INDEX)/(1.0 - alpha1_L); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto c_squared_L = qL(M1_INDEX)*this->phase1.c_value(rho1_L)*this->phase1.c_value(rho1_L)
                           + qL(M2_INDEX)*this->phase2.c_value(rho2_L)*this->phase2.c_value(rho2_L);
    const auto c_L         = std::sqrt(c_squared_L/rho_L);

    // Right state
    const auto rho_R       = qR(M1_INDEX) + qR(M2_INDEX);
    const auto vel_d_R     = qR(RHO_U_INDEX + curr_d)/rho_R;

    const auto alpha1_R    = qR(RHO_ALPHA1_INDEX)/rho_R;
    const auto rho1_R      = qR(M1_INDEX)/alpha1_R; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rho2_R      = qR(M2_INDEX)/(1.0 - alpha1_R); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto c_squared_R = qR(M1_INDEX)*this->phase1.c_value(rho1_R)*this->phase1.c_value(rho1_R)
                           + qR(M2_INDEX)*this->phase2.c_value(rho2_R)*this->phase2.c_value(rho2_R);
    const auto c_R         = std::sqrt(c_squared_R/rho_R);

    // Compute speeds of wave propagation
    const auto s_L    = std::min(vel_d_L - c_L, vel_d_R - c_R);
    const auto s_R    = std::max(vel_d_L + c_L, vel_d_R + c_R);

    /*--- Compute the flux ---*/
    if(s_L >= 0.0) {
      return this->evaluate_continuous_flux(qL, curr_d);
    }
    else if(s_L < 0.0 && s_R > 0.0) {
      return (s_R*this->evaluate_continuous_flux(qL, curr_d) -
              s_L*this->evaluate_continuous_flux(qR, curr_d) +
              s_L*s_R*(qR - qL))/(s_R - s_L);
    }
    else if(s_R <= 0.0) {
      return this->evaluate_continuous_flux(qR, curr_d);
    }
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
        discrete_flux[d].cons_flux_function = [&](auto& cells, const Field& field)
                                              {
                                                #ifdef ORDER_2
                                                  // Compute the stencil
                                                  const auto& left_left   = cells[0];
                                                  const auto& left        = cells[1];
                                                  const auto& right       = cells[2];
                                                  const auto& right_right = cells[3];

                                                  // MUSCL reconstruction
                                                  const FluxValue<typename Flux<Field>::cfg> primLL = this->cons2prim(field[left_left]);
                                                  const FluxValue<typename Flux<Field>::cfg> primL  = this->cons2prim(field[left]);
                                                  const FluxValue<typename Flux<Field>::cfg> primR  = this->cons2prim(field[right]);
                                                  const FluxValue<typename Flux<Field>::cfg> primRR = this->cons2prim(field[right_right]);

                                                  FluxValue<typename Flux<Field>::cfg> primL_recon,
                                                                                       primR_recon;
                                                  this->perform_reconstruction(primLL, primL, primR, primRR,
                                                                               primL_recon, primR_recon);

                                                  FluxValue<typename Flux<Field>::cfg> qL = this->prim2cons(primL_recon);
                                                  FluxValue<typename Flux<Field>::cfg> qR = this->prim2cons(primR_recon);
                                                #else
                                                  // Compute the stencil and extract state
                                                  const auto& left  = cells[0];
                                                  const auto& right = cells[1];

                                                  const FluxValue<typename Flux<Field>::cfg>& qL = field[left];
                                                  const FluxValue<typename Flux<Field>::cfg>& qR = field[right];
                                                #endif

                                                return compute_discrete_flux(qL, qR, d);
                                              };
      }
    );

    return make_flux_based_scheme(discrete_flux);
  }

} // end of namespace

#endif
