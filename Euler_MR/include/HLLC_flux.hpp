// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#ifndef HLLC_flux_hpp
#define HLLC_flux_hpp

#include "flux_base.hpp"

#define VERBOSE_FLUX

namespace samurai {
  using namespace EquationData;

  /**
    * Implementation of a HLLC flux for the Euler system
    */
  template<class Field>
  class HLLCFlux: public Flux<Field> {
  public:
    HLLCFlux(const EOS<typename Field::value_type>& EOS_); // Constructor which accepts in inputs the equation of state

    auto make_flux(); // Compute the flux over all cells

  private:
    auto compute_middle_state(const FluxValue<typename Flux<Field>::cfg>& q,
                              const typename Field::value_type S,
                              const typename Field::value_type S_star,
                              const std::size_t curr_d) const; // Compute the middle state

    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                               const FluxValue<typename Flux<Field>::cfg>& qR,
                                                               const std::size_t curr_d); // Rusanov flux along direction d
  };

  // Constructor derived from base class
  //
  template<class Field>
  HLLCFlux<Field>::HLLCFlux(const EOS<typename Field::value_type>& EOS_):
    Flux<Field>(EOS_) {}

  // Implement the auxiliary routine that computes the middle state
  //
  template<class Field>
  auto HLLCFlux<Field>::compute_middle_state(const FluxValue<typename Flux<Field>::cfg>& q,
                                             const typename Field::value_type S,
                                             const typename Field::value_type S_star,
                                             const std::size_t curr_d) const {
    // Compute the pressure
    auto e = q(RHOE_INDEX)/q(RHO_INDEX);
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e -= 0.5*(q(RHOU_INDEX + d)/q(RHO_INDEX))*(q(RHOU_INDEX + d)/q(RHO_INDEX));
    }
    const auto p = this->Euler_EOS.pres_value(q(RHO_INDEX), e);

    // Compute middle state
    FluxValue<typename Flux<Field>::cfg> q_star;

    const auto vel_d = q(RHOU_INDEX + curr_d)/q(RHO_INDEX);
    q_star(RHO_INDEX) = q(RHO_INDEX)*((S - vel_d)/(S - S_star));
    q_star(RHOU_INDEX) = q(RHO_INDEX)*((S - vel_d)/(S - S_star))*S_star;
    if(EquationData::dim > 1) {
      for(std::size_t d = 0; d < dim && d != curr_d; ++d) {
        q_star(RHOU_INDEX + d) = q(RHO_INDEX)*((S - vel_d)/(S - S_star))*(q(RHOU_INDEX + d)/q(RHO_INDEX));
      }
    }
    q_star(RHOE_INDEX) = q(RHO_INDEX)*((S - vel_d)/(S - S_star))*
                         (q(RHOE_INDEX)/q(RHO_INDEX) + (S_star - vel_d)*(S_star + p/(q(RHO_INDEX)*(S - vel_d))));

    return q_star;
  }

  // Implementation of a HLLC flux for the system
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg>
  HLLCFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                         const FluxValue<typename Flux<Field>::cfg>& qR,
                                         std::size_t curr_d) {
    // Left state
    const auto velL_d = qL(RHOU_INDEX + curr_d)/qL(RHO_INDEX);
    auto eL = qL(RHOE_INDEX)/qL(RHO_INDEX);
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      eL -= 0.5*(qL(RHOU_INDEX + d)/qL(RHO_INDEX))*(qL(RHOU_INDEX + d)/qL(RHO_INDEX));
    }
    const auto pL = this->Euler_EOS.pres_value(qL(RHO_INDEX), eL);
    const auto cL = this->Euler_EOS.c_value(qL(RHO_INDEX), pL);

    #ifdef VERBOSE_FLUX
      if(qL(RHO_INDEX) < 0.0) {
        throw std::runtime_error(std::string("Negative density left state: " + std::to_string(qL(RHO_INDEX))));
      }
      if(pL < 0.0) {
        throw std::runtime_error(std::string("Negative pressure left state: " + std::to_string(pL)));
      }
    #endif

    // Right state
    const auto velR_d = qR(RHOU_INDEX + curr_d)/qR(RHO_INDEX);
    auto eR = qR(RHOE_INDEX)/qR(RHO_INDEX);
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      eR -= 0.5*(qR(RHOU_INDEX + d)/qR(RHO_INDEX))*(qR(RHOU_INDEX + d)/qR(RHO_INDEX));
    }
    const auto pR = this->Euler_EOS.pres_value(qR(RHO_INDEX), eR);
    const auto cR = this->Euler_EOS.c_value(qR(RHO_INDEX), pR);

    #ifdef VERBOSE_FLUX
      if(qR(RHO_INDEX) < 0.0) {
        throw std::runtime_error(std::string("Negative density right state: " + std::to_string(qR(RHO_INDEX))));
      }
      if(pR < 0.0) {
        throw std::runtime_error(std::string("Negative pressure right state: " + std::to_string(pR)));
      }
    #endif

    // Compute speeds of wave propagation
    const auto sL     = std::min(velL_d - cL, velR_d - cR);
    const auto sR     = std::max(velL_d + cL, velR_d + cR);
    const auto s_star = (pR - pL + qL(RHO_INDEX)*velL_d*(sL - velL_d) - qR(RHO_INDEX)*velR_d*(sR - velR_d))/
                        (qL(RHO_INDEX)*(sL - velL_d) - qR(RHO_INDEX)*(sR - velR_d));

    /*--- Compute intermediate states ---*/
    auto q_star_L = compute_middle_state(qL, sL, s_star, curr_d);
    auto q_star_R = compute_middle_state(qR, sR, s_star, curr_d);

    /*--- Compute the flux ---*/
    if(sL >= 0.0) {
      return this->evaluate_continuous_flux(qL, curr_d);
    }
    else if(sL < 0.0 && s_star >= 0.0) {
      return this->evaluate_continuous_flux(qL, curr_d) + sL*(q_star_L - qL);
    }
    else if(s_star < 0.0 && sR >= 0.0) {
      return this->evaluate_continuous_flux(qR, curr_d) + sR*(q_star_R - qR);
    }
    else if(sR < 0.0) {
      return this->evaluate_continuous_flux(qR, curr_d);
    }
  }

  // Implement the contribution of the discrete flux for all the dimensions.
  //
  template<class Field>
  auto HLLCFlux<Field>::make_flux() {
    FluxDefinition<typename Flux<Field>::cfg> discrete_flux;

    /*--- Perform the loop over each dimension to compute the flux contribution ---*/
    static_for<0, EquationData::dim>::apply(
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
