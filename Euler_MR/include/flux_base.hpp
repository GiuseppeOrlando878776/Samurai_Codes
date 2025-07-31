// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#ifndef flux_base_hpp
#define flux_base_hpp

#pragma once
#include <samurai/schemes/fv.hpp>

#include "eos.hpp"

//#define ORDER_2

namespace EquationData {
  static constexpr std::size_t dim = 1; /*--- Spatial dimension. It would be ideal to be able to get it
                                              direclty from Field, but I need to move the definition of these indices ---*/

  /*--- Declare suitable static variables for the sake of generalities in the indices ---*/
  static constexpr std::size_t RHO_INDEX  = 0;
  static constexpr std::size_t RHOU_INDEX = 1;
  static constexpr std::size_t RHOE_INDEX = RHOU_INDEX + dim;

  static constexpr std::size_t NVARS = RHOE_INDEX + 1;

  /*--- Use auxiliary variables for the indices also for primitive variables for the sake of generality ---*/
  static constexpr std::size_t U_INDEX = RHOU_INDEX;
  static constexpr std::size_t P_INDEX = RHOE_INDEX;
}

namespace samurai {
  using namespace EquationData;

  /**
    * Generic class to compute the flux between a left and right state
    */
  template<class Field>
  class Flux {
  public:
    // Definitions and sanity checks
    static constexpr std::size_t field_size = Field::n_comp;
    static_assert(field_size == EquationData::NVARS, "The number of elements in the state does not correpsond to the number of equations");
    static_assert(Field::dim == EquationData::dim, "The spatial dimesions between Field and EquationData do not match");
    static constexpr std::size_t output_field_size = field_size;
    #ifdef ORDER_2
      static constexpr std::size_t stencil_size = 4;
    #else
      static constexpr std::size_t stencil_size = 2;
    #endif

    using cfg = FluxConfig<SchemeType::NonLinear, output_field_size, stencil_size, Field>;

    Flux(const EOS<typename Field::value_type>& EOS_); /*--- Constructor which accepts in inputs the equation of state ---*/

  protected:
    const EOS<typename Field::value_type>& Euler_EOS; /*--- Pass it by reference because pure virtual (not so nice, maybe moving to pointers) ---*/

    FluxValue<cfg> evaluate_continuous_flux(const FluxValue<cfg>& q,
                                            const std::size_t curr_d); /*--- Evaluate the 'continuous' flux for the state q
                                                                             along direction curr_d ---*/

    #ifdef ORDER_2
      FluxValue<cfg> cons2prim(const FluxValue<cfg>& cons) const; /*--- Conversion from conservative to primitive variables ---*/

      FluxValue<cfg> prim2cons(const FluxValue<cfg>& prim) const; /*--- Conversion from primitive to conservative variables ---*/

      void perform_reconstruction(const FluxValue<cfg>& primLL,
                                  const FluxValue<cfg>& primL,
                                  const FluxValue<cfg>& primR,
                                  const FluxValue<cfg>& primRR,
                                  FluxValue<cfg>& primL_recon,
                                  FluxValue<cfg>& primR_recon); /*--- Reconstruction for second order scheme ---*/
    #endif
  };

  // Class constructor in order to be able to work with the equation of state
  //
  template<class Field>
  Flux<Field>::Flux(const EOS<typename Field::value_type>& EOS_): Euler_EOS(EOS_) {}

  // Evaluate the 'continuous flux' along direction 'curr_d'
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg>
  Flux<Field>::evaluate_continuous_flux(const FluxValue<cfg>& q,
                                        const std::size_t curr_d) {
    /*--- Sanity check in terms of dimensions ---*/
    assert(curr_d < Field::dim);

    /*--- Initialize with the state ---*/
    FluxValue<cfg> res = q;

    /*--- Pre-fetch density that will be used several times ---*/
    const auto rho     = q(RHO_INDEX);
    const auto inv_rho = static_cast<typename Field::value_type>(1.0)/rho;

    /*--- Start computing the flux ---*/
    const auto vel_d = q(RHOU_INDEX + curr_d)/q(RHO_INDEX);
    res(RHO_INDEX) *= vel_d;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      res(RHOU_INDEX + d) *= vel_d;
    }
    res(RHOE_INDEX) *= vel_d;

    /*--- Compute the pressure ---*/
    auto e = q(RHOE_INDEX)*inv_rho;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      e -= static_cast<typename Field::value_type>(0.5)*
           (q(RHOU_INDEX + d)*inv_rho)*(q(RHOU_INDEX + d)*inv_rho);
    }
    const auto p = this->Euler_EOS.pres_value(rho, e);

    /*--- Add the pressure contribution to the momentum equation and energy equation ---*/
    res(RHOU_INDEX + curr_d) += p;
    res(RHOE_INDEX) += p*vel_d;

    return res;
  }

  // Implement functions for second order scheme
  //
  #ifdef ORDER_2
    // Conversion from conserved to primitive variables
    //
    template<class Field>
    FluxValue<typename Flux<Field>::cfg> Flux<Field>::cons2prim(const FluxValue<cfg>& cons) const {
      /*--- Create a state to store the primitive variables ---*/
      FluxValue<cfg> prim;

      /*--- Compute primitive variables ---*/
      const auto rho     = prim(RHO_INDEX);
      const autp inv_rho = static_cast<typename Field::value_type>(1.0)/rho;
      prim(RHO_INDEX)    = rho;
      for(std::size_t d = 0; d < Field::dim; ++d) {
        prim(U_INDEX + d) = cons(RHO_U_INDEX + d)*inv_rho;
      }
      auto e = cons(RHOE_INDEX)*inv_rho;
      for(std::size_t d = 0; d < Field::dim; ++d) {
        e -= static_cast<typename Field::value_type>(0.5)*
             (prim(U_INDEX + d)*prim(U_INDEX + d));
      }
      prim(P_INDEX) = this->Euler_EOS.pres_value(rho, e);

      /*--- Return primitive variables ---*/
      return prim;
    }

    // Conversion from primitive to conserved variables
    //
    template<class Field>
    FluxValue<typename Flux<Field>::cfg> Flux<Field>::prim2cons(const FluxValue<cfg>& prim) const {
      /*--- Create a suitable variable to save the conserved variables ---*/
      FluxValue<cfg> cons;

      /*--- Compute conserved variables ---*/
      const auto rho  = prim(RHO_INDEX);
      const auto p    = prim(P_INDEX);
      cons(RHO_INDEX) = rho;
      for(std::size_t d = 0; d < Field::dim; ++d) {
        cons(RHOU_INDEX + d) = rho*prim(U_INDEX + d);
      }
      auto E = this->Euler_EOS.e_value(rho, p);
      for(std::size_t d = 0; d < Field::dim; ++d) {
        E += static_cast<typename Field::value_type>(0.5)*
             (prim(U_INDEX + d)*prim(U_INDEX + d));
      }
      cons(RHOE_INDEX) = rho*E;

      /*--- Return conserved variables ---*/
      return cons;
    }

    // Perform reconstruction for order 2 scheme
    //
    template<class Field>
    void Flux<Field>::perform_reconstruction(const FluxValue<cfg>& primLL,
                                             const FluxValue<cfg>& primL,
                                             const FluxValue<cfg>& primR,
                                             const FluxValue<cfg>& primRR,
                                             FluxValue<cfg>& primL_recon,
                                             FluxValue<cfg>& primR_recon) {
      /*--- Initialize with the original state ---*/
      primL_recon = primL;
      primR_recon = primR;

      /*--- Perform the reconstruction ---*/
      const auto beta = static_cast<typename Field::value_type>(1.0); // MINMOD limiter
      for(std::size_t comp = 0; comp < Field::n_comp; ++comp) {
        if(primR(comp) - primL(comp) > static_cast<typename Field::value_type>(0.0)) {
          primL_recon(comp) += static_cast<typename Field::value_type>(0.5)*
                               std::max(static_cast<typename Field::value_type>(0.0),
                                        std::max(std::min(beta*(primL(comp) - primLL(comp)),
                                                          primR(comp) - primL(comp)),
                                                 std::min(primL(comp) - primLL(comp),
                                                          beta*(primR(comp) - primL(comp)))));
        }
        else if(primR(comp) - primL(comp) < static_cast<typename Field::value_type>(0.0)) {
          primL_recon(comp) += static_cast<typename Field::value_type>(0.5)*
                               std::min(static_cast<typename Field::value_type>(0.0),
                                        std::min(std::max(beta*(primL(comp) - primLL(comp)),
                                                          primR(comp) - primL(comp)),
                                                 std::max(primL(comp) - primLL(comp),
                                                          beta*(primR(comp) - primL(comp)))));
        }

        if(primRR(comp) - primR(comp) > static_cast<typename Field::value_type>(0.0)) {
          primR_recon(comp) -= static_cast<typename Field::value_type>(0.5)*
                               std::max(static_cast<typename Field::value_type>(0.0),
                                        std::max(std::min(beta*(primR(comp) - primL(comp)),
                                                          primRR(comp) - primR(comp)),
                                                 std::min(primR(comp) - primL(comp),
                                                          beta*(primRR(comp) - primR(comp)))));
        }
        else if(primRR(comp) - primR(comp) < static_cast<typename Field::value_type>(0.0)) {
          primR_recon(comp) -= static_cast<typename Field::value_type>(0.5)*
                               std::min(static_cast<typename Field::value_type>(0.0),
                                        std::min(std::max(beta*(primR(comp) - primL(comp)),
                                                          primRR(comp) - primR(comp)),
                                                 std::max(primR(comp) - primL(comp),
                                                          beta*(primRR(comp) - primR(comp)))));
        }
      }
    }
  #endif

} // end namespace samurai

#endif
