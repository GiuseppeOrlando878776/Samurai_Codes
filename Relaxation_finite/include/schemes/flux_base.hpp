// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#pragma once

#include <samurai/schemes/fv.hpp>

#include "../eos.hpp"
#include "../utilities.hpp"

//#define ORDER_2

namespace samurai {
  /**
    * Generic class to compute the flux between a left and right state
    */
  template<class Field>
  class Flux {
  public:
    /*--- Definitions and sanity checks ---*/
    using Indices = EquationData<Field::dim>;
    static_assert(Field::n_comp == Indices::NVARS, "The number of elements in the state does not correpsond to the number of equations");
    #ifdef ORDER_2
      static constexpr std::size_t stencil_size = 4;
    #else
      static constexpr std::size_t stencil_size = 2;
    #endif

    using cfg = FluxConfig<SchemeType::NonLinear, stencil_size, Field, Field>;

    using Number = typename Field::value_type; /*--- Shortcut for the arithmetic type ---*/

    Flux(const EOS<Number>& EOS_phase1_,
         const EOS<Number>& EOS_phase2_); /*--- Constructor which accepts in input
                                                the equations of state of the two phases ---*/

  protected:
    const EOS<Number>& EOS_phase1; /*--- Pass it by reference because pure virtual (not so nice, maybe moving to pointers) ---*/
    const EOS<Number>& EOS_phase2; /*--- Pass it by reference because pure virtual (not so nice, maybe moving to pointers) ---*/

    FluxValue<cfg> evaluate_continuous_flux(const FluxValue<cfg>& q,
                                            const std::size_t curr_d); /*--- Evaluate the 'continuous' flux for the state q
                                                                             along direction curr_d ---*/

    auto evaluate_phasic_continuous_flux(const auto& q,
                                         const std::size_t curr_d,
                                         const unsigned phase_idx); /*--- Evaluate the 'continuous' flux
                                                                          of phase 'phase_idx' for the state q
                                                                          along direction curr_d ---*/

    #ifdef ORDER_2
      FluxValue<cfg> cons2prim(const FluxValue<cfg>& cons) const; /*--- Conversion from conserved to primitive variables ---*/

      FluxValue<cfg> prim2cons(const FluxValue<cfg>& prim) const; /*--- Conversion from primitive to conserved variables ---*/
    #endif
  };

  // Class constructor in order to be able to work with the equation of state
  //
  template<class Field>
  Flux<Field>::Flux(const EOS<Number>& EOS_phase1_,
                    const EOS<Number>& EOS_phase2_):
    EOS_phase1(EOS_phase1_), EOS_phase2(EOS_phase2_) {}

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

    /*--- Pre-fetch variables that will be used several times so as to exploit possible vectorization ---*/
    const auto alpha1 = q(Indices::ALPHA1_INDEX);
    const auto m1     = q(Indices::ALPHA1_RHO1_INDEX);
    const auto m1E1   = q(Indices::ALPHA1_RHO1_E1_INDEX);
    const auto m2     = q(Indices::ALPHA2_RHO2_INDEX);
    const auto m2E2   = q(Indices::ALPHA2_RHO2_E2_INDEX);

    /*--- Compute density, velocity (along the dimension) and internal energy of phase 1 ---*/
    const auto rho1   = m1/alpha1; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto inv_m1 = static_cast<Number>(1.0)/m1; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1           = m1E1*inv_m1; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < Field::dim; ++d) {
      e1 -= static_cast<Number>(0.5)*
            ((q(Indices::ALPHA1_RHO1_U1_INDEX + d)*inv_m1)*
             (q(Indices::ALPHA1_RHO1_U1_INDEX + d)*inv_m1)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pres1  = EOS_phase1.pres_value_Rhoe(rho1, e1);
    const auto vel1_d = q(Indices::ALPHA1_RHO1_U1_INDEX + curr_d)*inv_m1; /*--- TODO: Add treatment for vanishing volume fraction ---*/

    /*--- Compute the flux for the equations "associated" to phase 1 ---*/
    res(Indices::ALPHA1_INDEX) = static_cast<Number>(0.0);
    res(Indices::ALPHA1_RHO1_INDEX) *= vel1_d;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      res(Indices::ALPHA1_RHO1_U1_INDEX + d) *= vel1_d;
    }
    res(Indices::ALPHA1_RHO1_U1_INDEX + curr_d) += alpha1*pres1;
    res(Indices::ALPHA1_RHO1_E1_INDEX) *= vel1_d;
    res(Indices::ALPHA1_RHO1_E1_INDEX) += alpha1*pres1*vel1_d;

    /*--- Compute density, velocity (along the dimension) and internal energy of phase 2 ---*/
    const auto alpha2 = static_cast<Number>(1.0) - alpha1;
    const auto rho2   = m2/alpha2; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto inv_m2 = static_cast<Number>(1.0)/m2; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2           = m2E2*inv_m2; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < Field::dim; ++d) {
      e2 -= static_cast<Number>(0.5)*
            ((q(Indices::ALPHA2_RHO2_U2_INDEX + d)*inv_m2)*
             (q(Indices::ALPHA2_RHO2_U2_INDEX + d)*inv_m2)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pres2  = EOS_phase2.pres_value_Rhoe(rho2, e2);
    const auto vel2_d = q(Indices::ALPHA2_RHO2_U2_INDEX + curr_d)*inv_m2; /*--- TODO: Add treatment for vanishing volume fraction ---*/

    /*--- Compute the flux for the equations "associated" to phase 2 ---*/
    res(Indices::ALPHA2_RHO2_INDEX) *= vel2_d;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      res(Indices::ALPHA2_RHO2_U2_INDEX + d) *= vel2_d;
    }
    res(Indices::ALPHA2_RHO2_U2_INDEX + curr_d) += alpha2*pres2;
    res(Indices::ALPHA2_RHO2_E2_INDEX) *= vel2_d;
    res(Indices::ALPHA2_RHO2_E2_INDEX) += alpha2*pres2*vel2_d;

    return res;
  }

  // Evaluate the 'phasic continuous flux' along direction 'curr_d'
  //
  template<class Field>
  auto Flux<Field>::evaluate_phasic_continuous_flux(const auto& q,
                                                    const std::size_t curr_d,
                                                    const unsigned phase_idx) {
    /*--- Sanity check in terms of dimensions ---*/
    assert(curr_d < Field::dim);

    /*--- Pre-fetch variables that will be used several times so as to exploit possible vectorization.
          The order is hard-coded, it can be made more general following the same appraoch in EquationData.
          Since this is supposed to be likely temporary, we do not worry to much about that for the moment ---*/
    const auto alphak = q.front();
    const auto mk     = q[1];
    const auto mkEk   = q.back();

    /*--- Compute density, velocity (along the dimension) and internal energy ---*/
    const auto rhok   = mk/alphak; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto inv_mk = static_cast<Number>(1.0)/mk; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto ek           = mkEk*inv_mk; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < Field::dim; ++d) {
      ek -= static_cast<Number>(0.5)*
            ((q[d + 2]*inv_mk)*(q[d + 2]*inv_mk)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    Number pk;
    if(phase_idx == 1) {
      pk = EOS_phase1.pres_value_Rhoe(rhok, ek);
    }
    else if(phase_idx == 2) {
      pk = EOS_phase2.pres_value_Rhoe(rhok, ek);
    }
    else {
      std::cerr << "Unknown phasic index. Exiting..." << std::endl;
      exit(1);
    }
    const auto velk_d = q[curr_d + 2]*inv_mk; /*--- TODO: Add treatment for vanishing volume fraction ---*/

    /*--- Compute the flux for the equations "associated" to phase k ---*/
    std::array<Number, Field::dim + 2> res;
    res.front() = mk*velk_d;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      res[d + 1] = q[d + 2]*velk_d;
    }
    res[curr_d + 1] += alphak*pk;
    res.back() = (mkEk + alphak*pk)*velk_d;

    return res;
  }

  // Implement functions for second order scheme
  //
  #ifdef ORDER_2
    // Conversion from conserved to primitive variables
    //
    template<class Field>
    FluxValue<typename Flux<Field>::cfg> Flux<Field>::cons2prim(const FluxValue<cfg>& cons) const {
      /*--- Create a suitable variable to set primitive variables ---*/
      FluxValue<cfg> prim;

      /*--- Pre-fetch variables that will be used several times so as to exploit possible vectorization ---*/
      const auto alpha1 = cons(Indices::ALPHA1_INDEX);
      const auto m1     = cons(Indices::ALPHA1_RHO1_INDEX);
      const auto m1E1   = cons(Indices::ALPHA1_RHO1_E1_INDEX);
      const auto m2     = cons(Indices::ALPHA2_RHO2_INDEX);
      const auto m2E2   = cons(Indices::ALPHA2_RHO2_E2_INDEX);

      /*--- Start with phase 1 ---*/
      prim(Indices::ALPHA1_INDEX) = alpha1;
      prim(Indices::RHO1_INDEX)   = m1/alpha1; /*--- TODO: Add treatment for vanishing volume fraction ---*/
      const auto inv_m1  = static_cast<Number>(1.0)/m1; /*--- TODO: Add treatment for vanishing volume fraction ---*/
      auto e1 = m1E1*inv_m1; /*--- TODO: Add treatment for vanishing volume fraction ---*/
      for(std::size_t d = 0; d < Field::dim; ++d) {
        prim(Indices::U1_INDEX + d) = cons(Indices::ALPHA1_RHO1_U1_INDEX + d)*inv_m1; /*--- TODO: Add treatment for vanishing volume fraction ---*/
        e1 -= static_cast<Number>(0.5)*
              (prim(Indices::U1_INDEX + d)*prim(Indices::U1_INDEX + d));
      }
      prim(Indices::P1_INDEX) = EOS_phase1.pres_value_Rhoe(prim(Indices::RHO1_INDEX), e1);

      /*--- Proceed with phase 2 ---*/
      prim(Indices::RHO2_INDEX) = m2/(static_cast<Number>(1.0) - alpha1); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      const auto inv_m2 = static_cast<Number>(1.0)/m2; /*--- TODO: Add treatment for vanishing volume fraction ---*/
      auto e2 = m2E2*inv_m2; /*--- TODO: Add treatment for vanishing volume fraction ---*/
      for(std::size_t d = 0; d < Field::dim; ++d) {
        prim(Indices::U2_INDEX + d) = cons(Indices::ALPHA2_RHO2_U2_INDEX + d)*inv_m2; /*--- TODO: Add treatment for vanishing volume fraction ---*/
        e2 -= static_cast<Number>(0.5)*
              (prim(Indices::U2_INDEX + d)*prim(Indices::U2_INDEX + d));
      }
      prim(Indices::P2_INDEX) = EOS_phase2.pres_value_Rhoe(prim(Indices::RHO2_INDEX), e2);

      /*--- Return computed primitive variables ---*/
      return prim;
    }

    // Conversion from primitive to conserved variables
    //
    template<class Field>
    FluxValue<typename Flux<Field>::cfg> Flux<Field>::prim2cons(const FluxValue<cfg>& prim) const {
      /*--- Create a suitable variable to save the conserved variables ---*/
      FluxValue<cfg> cons;

      /*--- Pre-fetch variables that will be used several times so as to exploit possible vectorization ---*/
      const auto alpha1 = prim(Indices::ALPHA1_INDEX);
      const auto rho1   = prim(Indices::RHO1_INDEX);
      const auto p1     = prim(Indices::P1_INDEX);
      const auto rho2   = prim(Indices::RHO2_INDEX);
      const auto p2     = prim(Indices::P2_INDEX);

      /*--- Start with phase 1 ---*/
      cons(Indices::ALPHA1_INDEX) = alpha1;
      const auto m1 = alpha1*rho1;
      cons(Indices::ALPHA1_RHO1_INDEX) = m1;
      auto E1 = EOS_phase1.e_value_RhoP(rho1, p1);
      for(std::size_t d = 0; d < Field::dim; ++d) {
        cons(Indices::ALPHA1_RHO1_U1_INDEX + d) = m1*prim(Indices::U1_INDEX + d);
        E1 += static_cast<Number>(0.5)*
              (prim(Indices::U1_INDEX + d)*prim(Indices::U1_INDEX + d));
      }
      cons(Indices::ALPHA1_RHO1_E1_INDEX) = m1*E1;

      /*--- Proceed with phase 2 ---*/
      const auto m2 = (static_cast<Number>(1.0) - alpha1)*rho2;
      cons(Indices::ALPHA2_RHO2_INDEX) = m2;
      auto E2 = EOS_phase2.e_value_RhoP(rho2, p2);
      for(std::size_t d = 0; d < Field::dim; ++d) {
        cons(Indices::ALPHA2_RHO2_U2_INDEX + d) = m2*prim(Indices::U2_INDEX + d);
        E2 += static_cast<Number>(0.5)*
              (prim(Indices::U2_INDEX + d)*prim(Indices::U2_INDEX + d));
      }
      cons(Indices::ALPHA2_RHO2_E2_INDEX) = m2*E2;

      /*--- Return computed conserved variables ---*/
      return cons;
    }
  #endif

} // end namespace samurai
