// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#pragma once

#include <samurai/schemes/fv.hpp>

#include "../utilities.hpp"
#include "../eos.hpp"

//#define ORDER_2

namespace EquationData {
  static constexpr std::size_t dim = 1; /*--- Spatial dimension. It would be ideal to be able to get it
                                              directly from Field, but I need to move the definition of these indices ---*/

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
    static_assert(Field::dim == EquationData::dim, "The spatial dimesions between Field and EquationData do not match");
    static_assert(Field::n_comp == EquationData::NVARS, "The number of elements in the state does not correspond to the number of equations");
    #ifdef ORDER_2
      static constexpr std::size_t stencil_size = 4;
    #else
      static constexpr std::size_t stencil_size = 2;
    #endif

    using cfg = FluxConfig<SchemeType::NonLinear, stencil_size, Field, Field>;

    using Number = typename Field::value_type; /*--- Define the shortcut for the arithmetic type ---*/

    Flux(const EOS<Number>& EOS_); /*--- Constructor which accepts in inputs the equation of state ---*/

    virtual ~Flux() {} /*--- Virtual destructor (because pure virtual class) ---*/

    inline void set_flux_name(const std::string& flux_name_); /*--- Set the name of the numerical flux ---*/

    inline std::string get_flux_name() const; /*--- Get the name of the numerical flux ---*/

    virtual decltype(make_flux_based_scheme(std::declval<FluxDefinition<cfg>>())) make_flux() = 0;
    /*--- Compute the flux over all the faces and directions ---*/

  protected:
    const EOS<Number>& Euler_EOS; /*--- Pass it by reference because pure virtual (not so nice, maybe moving to pointers) ---*/

    FluxValue<cfg> evaluate_continuous_flux(const FluxValue<cfg>& q,
                                            const std::size_t curr_d); /*--- Evaluate the 'continuous' flux for the state q
                                                                             along direction curr_d ---*/

    #ifdef ORDER_2
      FluxValue<cfg> cons2prim(const FluxValue<cfg>& cons) const; /*--- Conversion from conservative to primitive variables ---*/

      FluxValue<cfg> prim2cons(const FluxValue<cfg>& prim) const; /*--- Conversion from primitive to conservative variables ---*/
    #endif

  private:
    std::string flux_name; /*--- Name of the numerical flux ---*/
  };

  // Class constructor in order to be able to work with the equation of state
  //
  template<class Field>
  Flux<Field>::Flux(const EOS<Number>& EOS_): Euler_EOS(EOS_) {}

  // Set the name of the numerical flux
  //
  template<class Field>
  inline void Flux<Field>::set_flux_name(const std::string& flux_name_) {
    flux_name = flux_name_;
  }

  // Get the name of the numerical flux
  //
  template<class Field>
  inline std::string Flux<Field>::get_flux_name() const {
    return flux_name;
  }

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
    const auto inv_rho = static_cast<Number>(1.0)/rho;

    /*--- Start computing the flux ---*/
    const auto vel_d = q(RHOU_INDEX + curr_d)*inv_rho;
    res(RHO_INDEX) *= vel_d;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      res(RHOU_INDEX + d) *= vel_d;
    }
    res(RHOE_INDEX) *= vel_d;

    /*--- Compute the pressure ---*/
    auto e = q(RHOE_INDEX)*inv_rho;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      e -= static_cast<Number>(0.5)*
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
    FluxValue<typename Flux<Field>::cfg>
    Flux<Field>::cons2prim(const FluxValue<cfg>& cons) const {
      /*--- Create a state to store the primitive variables ---*/
      FluxValue<cfg> prim;

      /*--- Compute primitive variables ---*/
      const auto rho     = prim(RHO_INDEX);
      const auto inv_rho = static_cast<Number>(1.0)/rho;
      prim(RHO_INDEX)    = rho;
      for(std::size_t d = 0; d < Field::dim; ++d) {
        prim(U_INDEX + d) = cons(RHO_U_INDEX + d)*inv_rho;
      }
      auto e = cons(RHOE_INDEX)*inv_rho;
      for(std::size_t d = 0; d < Field::dim; ++d) {
        e -= static_cast<Number>(0.5)*
             (prim(U_INDEX + d)*prim(U_INDEX + d));
      }
      prim(P_INDEX) = this->Euler_EOS.pres_value(rho, e);

      /*--- Return primitive variables ---*/
      return prim;
    }

    // Conversion from primitive to conserved variables
    //
    template<class Field>
    FluxValue<typename Flux<Field>::cfg>
    Flux<Field>::prim2cons(const FluxValue<cfg>& prim) const {
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
        E += static_cast<Number>(0.5)*
             (prim(U_INDEX + d)*prim(U_INDEX + d));
      }
      cons(RHOE_INDEX) = rho*E;

      /*--- Return conserved variables ---*/
      return cons;
    }
  #endif

} // end namespace samurai
