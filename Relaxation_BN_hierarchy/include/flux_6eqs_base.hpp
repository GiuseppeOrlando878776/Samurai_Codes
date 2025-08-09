// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#ifndef flux_6eqs_base_hpp
#define flux_6eqs_base_hpp

#pragma once
#include <samurai/schemes/fv.hpp>

#include "eos.hpp"

//#define ORDER_2

namespace EquationData {
  static constexpr std::size_t dim = 1; /*--- Spatial dimension. It would be ideal to be able to get it
                                              direclty from Field, but I need to move the definition of these indices ---*/

  /*--- Declare suitable static variables for the sake of generalities in the indices ---*/
  static constexpr std::size_t ALPHA1_INDEX         = 0;
  static constexpr std::size_t ALPHA1_RHO1_INDEX    = 1;
  static constexpr std::size_t ALPHA2_RHO2_INDEX    = 2;
  static constexpr std::size_t RHO_U_INDEX          = 3;
  static constexpr std::size_t ALPHA1_RHO1_E1_INDEX = RHO_U_INDEX + dim;
  static constexpr std::size_t ALPHA2_RHO2_E2_INDEX = ALPHA1_RHO1_E1_INDEX + 1;

  static constexpr std::size_t NVARS = ALPHA2_RHO2_E2_INDEX + 1;

  /*--- Use auxiliary variables for the indices also for primitive variables for the sake of generality ---*/
  static constexpr std::size_t RHO1_INDEX = ALPHA1_RHO1_INDEX;
  static constexpr std::size_t RHO2_INDEX = ALPHA2_RHO2_INDEX;
  static constexpr std::size_t U_INDEX    = RHO_U_INDEX;
  static constexpr std::size_t P1_INDEX   = ALPHA1_RHO1_E1_INDEX;
  static constexpr std::size_t P2_INDEX   = ALPHA2_RHO2_E2_INDEX;
}

namespace samurai {
  using namespace EquationData;

  /**
    * Generic class to compute the flux between a left and right state
    */
  template<class Field>
  class Flux {
  public:
    /*--- Definitions and sanity checks ---*/
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

    Flux(const EOS<typename Field::value_type>& EOS_phase1_,
         const EOS<typename Field::value_type>& EOS_phase2_); /*--- Constructor which accepts in input
                                                                    the equations of state of the two phases ---*/

  protected:
    const EOS<typename Field::value_type>& EOS_phase1; /*--- Pass it by reference because pure virtual (not so nice, maybe moving to pointers) ---*/
    const EOS<typename Field::value_type>& EOS_phase2; /*--- Pass it by reference because pure virtual (not so nice, maybe moving to pointers) ---*/

    FluxValue<cfg> evaluate_continuous_flux(const FluxValue<cfg>& q, const std::size_t curr_d); /*--- Evaluate the 'continuous' flux for the state q
                                                                                                      along direction curr_d ---*/

    #ifdef ORDER_2
      FluxValue<cfg> cons2prim(const FluxValue<cfg>& cons) const; /*--- Conversion from conserved to primitive variables ---*/

      FluxValue<cfg> prim2cons(const FluxValue<cfg>& prim) const; /*--- Conversion from primitive to conserved variables ---*/

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
  Flux<Field>::Flux(const EOS<typename Field::value_type>& EOS_phase1_,
                    const EOS<typename Field::value_type>& EOS_phase2_):
    EOS_phase1(EOS_phase1_), EOS_phase2(EOS_phase2_) {}

  // Evaluate the 'continuous flux' along direction 'curr_d'
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::evaluate_continuous_flux(const FluxValue<cfg>& q, const std::size_t curr_d) {
    /*--- Sanity check in terms of dimensions ---*/
    assert(curr_d < Field::dim);

    /*--- Initialize with the state ---*/
    FluxValue<cfg> res = q;

    /*--- Pre-fetch variables that will be used several times so as to exploit possible vectorization ---*/
    const auto m1 = q(ALPHA1_RHO1_INDEX);
    const auto m2 = q(ALPHA2_RHO2_INDEX);

    /*--- Save the mixture density and the velocity along the direction of interest ---*/
    const auto rho     = m1 + m2;
    const auto inv_rho = static_cast<typename Field::value_type>(1.0)/rho;
    const auto vel_d   = q(RHO_U_INDEX + curr_d)*inv_rho;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      res(RHO_U_INDEX + d) *= vel_d;
    }

    /*--- Compute density and pressure of phase 1 ---*/
    const auto alpha1 = q(ALPHA1_INDEX);
    const auto rho1   = m1/alpha1; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto norm2_vel    = static_cast<typename Field::value_type>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      norm2_vel += (q(RHO_U_INDEX + d)*inv_rho)*
                   (q(RHO_U_INDEX + d)*inv_rho);
    }
    const auto m1E1 = q(ALPHA1_RHO1_E1_INDEX);
    const auto e1   = m1E1/m1 /*--- TODO: Add treatment for vanishing volume fraction ---*/
                    - static_cast<typename Field::value_type>(0.5)*norm2_vel;
    const auto p1   = EOS_phase1.pres_value(rho1, e1);

    /*--- Compute the flux for the equations "associated" to phase 1 ---*/
    res(ALPHA1_INDEX) = 0.0;
    res(ALPHA1_RHO1_INDEX) *= vel_d;
    res(ALPHA1_RHO1_E1_INDEX) *= vel_d;
    res(ALPHA1_RHO1_E1_INDEX) += alpha1*p1*vel_d;

    /*--- Compute density and pressure of phase 2 ---*/
    const auto rho2 = m2/(static_cast<typename Field::value_type>(1.0) - alpha1); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto m2E2 = q(ALPHA2_RHO2_E2_INDEX);
    const auto e2   = m2E2/m2 /*--- TODO: Add treatment for vanishing volume fraction ---*/
                    - static_cast<typename Field::value_type>(0.5)*norm2_vel;
    const auto p2   = EOS_phase2.pres_value(rho2, e2);

    /*--- Compute the flux for the equations "associated" to phase 2 ---*/
    res(ALPHA2_RHO2_INDEX) *= vel_d;
    res(ALPHA2_RHO2_E2_INDEX) *= vel_d;
    res(ALPHA2_RHO2_E2_INDEX) += (static_cast<typename Field::value_type>(1.0) - alpha1)*p2*vel_d;

    /*--- Add the mixture pressure contribution to the momentum equation ---*/
    res(RHO_U_INDEX + curr_d) += (alpha1*p1 +
                                  (static_cast<typename Field::value_type>(1.0) - alpha1)*p2);

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

      /*--- Pre-fetch variables that will be used several times so as to exploit possible vectorization ---*/
      const auto alpha1 = cons(ALPHA1_INDEX);
      const auto m1     = cons(ALPHA1_RHO1_INDEX);
      const auto m2     = cons(ALPHA2_RHO2_INDEX);
      const auto m1E1   = cons(ALPHA1_RHO1_E1_INDEX);
      const auto m2E2   = cons(ALPHA2_RHO2_E2_INDEX);

      /*--- Start with phase 1 ---*/
      prim(ALPHA1_INDEX) = alpha1;
      const auto rho1    = m1/alpha1; /*--- TODO: Add treatment for vanishing volume fraction ---*/
      prim(RHO1_INDEX)   = rho1;
      const auto rho     = m1 + m2;
      const auto inv_rho = static_cast<typename Field::value_type>(1.0)/rho;
      auto norm2_vel     = static_cast<typename Field::value_type>(0.0);
      for(std::size_t d = 0; d < Field::dim; ++d) {
        prim(U_INDEX + d) = cons(RHO_U_INDEX + d)*inv_rho;
        norm2_vel += prim(U_INDEX + d)*prim(U_INDEX + d);
      }
      const auto e1  = m1E1/m1 /*--- TODO: Add treatment for vanishing volume fraction ---*/
                     - static_cast<typename Field::value_type>(0.5)*norm2_vel;
      prim(P1_INDEX) = EOS_phase1.pres_value(rho1, e1);

      /*--- Proceed with phase 2 ---*/
      const auto rho2  = m2/(static_cast<typename Field::value_type>(1.0) - alpha1); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      prim(RHO2_INDEX) = rho2;
      const auto e2    = m2E2/m2 /*--- TODO: Add treatment for vanishing volume fraction ---*/
                       - 0.5*norm2_vel;
      prim(P2_INDEX)   = EOS_phase2.pres_value(rho2, e2);

      /*--- Return primitive variables ---*/
      return prim;
    }

    // Conversion from primitive to conserved variables
    //
    template<class Field>
    FluxValue<typename Flux<Field>::cfg> Flux<Field>::prim2cons(const FluxValue<cfg>& prim) const {
      /*--- Create a suitable variable to save the conserved variables ---*/
      FluxValue<cfg> cons;

      /*--- Pre-fetch variables that will be used several times so as to exploit possible vectorization ---*/
      const auto alpha1 = prim(ALPHA1_INDEX);
      const auto rho1   = prim(RHO1_INDEX);
      const auto rho2   = prim(RHO2_INDEX);
      const auto p1     = prim(P1_INDEX);
      const auto p2     = prim(P2_INDEX);

      /*--- Start with phase 1 ---*/
      cons(ALPHA1_INDEX)      = alpha1;
      const auto m1           = alpha1*rho1;
      cons(ALPHA1_RHO1_INDEX) = m1;
      auto norm2_vel          = static_cast<typename Field::value_type>(0.0);
      for(std::size_t d = 0; d < Field::dim; ++d) {
        norm2_vel += prim(U_INDEX + d)*prim(U_INDEX + d);
      }
      const auto E1 = EOS_phase1.e_value(rho1, p1)
                    + static_cast<typename Field::value_type>(0.5)*norm2_vel;
      cons(ALPHA1_RHO1_E1_INDEX) = m1*E1;

      /*--- Proceed with phase 2 ---*/
      const auto m2              = (static_cast<typename Field::value_type>(1.0) - alpha1)*rho2;
      cons(ALPHA2_RHO2_INDEX)    = m2;
      const auto E2              = EOS_phase2.e_value(rho2, p2)
                                 + static_cast<typename Field::value_type>(0.5)*norm2_vel;
      cons(ALPHA2_RHO2_E2_INDEX) = m2*E2;

      /*--- Update momentum ---*/
      const auto rho = m1 + m2;
      for(std::size_t d = 0; d < Field::dim; ++d) {
        cons(RHO_U_INDEX + d) = rho*prim(U_INDEX + d);
      }

      /*--- Return computed conserved variables ---*/
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
