// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#pragma once
#include <samurai/schemes/fv.hpp>

#include "barotropic_eos.hpp"

/*--- Preprocessor to define whether order 2 is desired ---*/
#define ORDER_2

/*--- Preprocessor to define whether relaxation is desired after reconstruction for order 2 ---*/
#ifdef ORDER_2
  #define RELAX_RECONSTRUCTION
#endif

/**
 * Useful parameters and enumerators
 */
namespace EquationData {
  static constexpr std::size_t dim = 3; /*--- Spatial dimension. It would be ideal to be able to get it
                                              direclty from Field, but I need to move the definition of these indices ---*/

  /*--- Declare suitable static variables for the sake of generalities in the indices ---*/
  static constexpr std::size_t M1_INDEX         = 0;
  static constexpr std::size_t M2_INDEX         = 1;
  static constexpr std::size_t RHO_ALPHA1_INDEX = 2;
  static constexpr std::size_t RHO_U_INDEX      = 3;

  /*--- Save also the total number of (scalar) variables ---*/
  static constexpr std::size_t NVARS = 3 + dim;

  /*--- Use auxiliary variables for the indices also for primitive variables for the sake of generality ---*/
  static constexpr std::size_t ALPHA1_INDEX = RHO_ALPHA1_INDEX;
  static constexpr std::size_t U_INDEX      = RHO_U_INDEX;
  static constexpr std::size_t P1_INDEX     = M1_INDEX;
  static constexpr std::size_t P2_INDEX     = M2_INDEX;
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
    static_assert(field_size == EquationData::NVARS, "The number of elements in the state does not correspond to the number of equations");
    static_assert(Field::dim == EquationData::dim, "The spatial dimensions between Field and the parameter list do not match");
    static constexpr std::size_t output_field_size = field_size;
    #ifdef ORDER_2
      static constexpr std::size_t stencil_size = 4;
    #else
      static constexpr std::size_t stencil_size = 2;
    #endif

    using cfg    = FluxConfig<SchemeType::NonLinear, output_field_size, stencil_size, Field>;
    using Number = typename Field::value_type; /*--- Define the shortcut for the arithmetic type ---*/

    Flux(const LinearizedBarotropicEOS<Number>& EOS_phase1_,
         const LinearizedBarotropicEOS<Number>& EOS_phase2_,
         const Number lambda_ = static_cast<Number>(0.9),
         const Number atol_Newton_ = static_cast<Number>(1e-14),
         const Number rtol_Newton_ = static_cast<Number>(1e-12),
         const std::size_t max_Newton_iters_ = 60); /*--- Constructor which accepts in input the equations of state of the two phases ---*/

  protected:
    const LinearizedBarotropicEOS<Number>& EOS_phase1;
    const LinearizedBarotropicEOS<Number>& EOS_phase2;

    const Number      lambda;           /*--- Parameter for bound preserving strategy ---*/
    const Number      atol_Newton;      /*--- Absolute tolerance Newton method relaxation ---*/
    const Number      rtol_Newton;      /*--- Relative tolerance Newton method relaxation ---*/
    const std::size_t max_Newton_iters; /*--- Maximum number of Newton iterations ---*/

    FluxValue<cfg> evaluate_continuous_flux(const FluxValue<cfg>& q,
                                            const std::size_t curr_d); /*--- Evaluate the 'continuous' flux for the state q
                                                                             along direction curr_d ---*/

    FluxValue<cfg> cons2prim(const FluxValue<cfg>& cons) const; /*--- Conversion from conserved to primitive variables ---*/

    FluxValue<cfg> prim2cons(const FluxValue<cfg>& prim) const; /*--- Conversion from primitive to conserved variables ---*/

    #ifdef ORDER_2
      void perform_reconstruction(const FluxValue<cfg>& primLL,
                                  const FluxValue<cfg>& primL,
                                  const FluxValue<cfg>& primR,
                                  const FluxValue<cfg>& primRR,
                                  FluxValue<cfg>& primL_recon,
                                  FluxValue<cfg>& primR_recon); /*--- Reconstruction for second order scheme ---*/

      #ifdef RELAX_RECONSTRUCTION
        template<typename State>
        void perform_Newton_step_relaxation(State conserved_variables,
                                            Number& dalpha1,
                                            Number& alpha1,
                                            bool& relaxation_applied); /*--- Perform a Newton step relaxation for a state vector
                                                                             (it is not a real space dependent procedure,
                                                                              but I would need to be able to do it inside the flux location
                                                                              for MUSCL reconstruction) ---*/

        void relax_reconstruction(FluxValue<cfg>& q); /*--- Relax reconstructed state ---*/
      #endif
    #endif
  };

  // Class constructor in order to be able to work with the equation of state
  //
  template<class Field>
  Flux<Field>::Flux(const LinearizedBarotropicEOS<Number>& EOS_phase1_,
                    const LinearizedBarotropicEOS<Number>& EOS_phase2_,
                    const Number lambda_,
                    const Number atol_Newton_,
                    const Number rtol_Newton_,
                    const std::size_t max_Newton_iters_):
    EOS_phase1(EOS_phase1_), EOS_phase2(EOS_phase2_),
    lambda(lambda_), atol_Newton(atol_Newton_), rtol_Newton(rtol_Newton_),
    max_Newton_iters(max_Newton_iters_) {}

  // Evaluate the 'continuous flux'
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::evaluate_continuous_flux(const FluxValue<cfg>& q,
                                                                             const std::size_t curr_d) {
    /*--- Sanity check in terms of dimensions ---*/
    assert(curr_d < Field::dim);

    /*--- Initialize the resulting variable ---*/
    FluxValue<cfg> res = q;

    /*--- Pre-fetch some variables used multiple times in order to exploit possible vectorization ---*/
    const auto m1 = q(M1_INDEX);
    const auto m2 = q(M2_INDEX);

    /*--- Compute the current density and velocity ---*/
    const auto rho     = m1 + m2;
    const auto inv_rho = static_cast<Number>(1.0)/rho;
    const auto vel_d   = q(RHO_U_INDEX + curr_d)*inv_rho;

    /*--- Multiply the state by the velocity along the direction of interest ---*/
    res(M1_INDEX) *= vel_d;
    res(M2_INDEX) *= vel_d;
    res(RHO_ALPHA1_INDEX) *= vel_d;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      res(RHO_U_INDEX + d) *= vel_d;
    }

    /*--- Compute and add the contribution due to the pressure ---*/
    const auto alpha1 = q(RHO_ALPHA1_INDEX)*inv_rho;
    const auto rho1   = m1/alpha1; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto p1     = EOS_phase1.pres_value(rho1);

    const auto p2     = EOS_phase2.pres_value(m2/(static_cast<Number>(1.0) - alpha1));
                        /*--- TODO: Add a check in case of zero volume fraction ---*/

    const auto p      = alpha1*p1
                      + (static_cast<Number>(1.0) - alpha1)*p2;

    res(RHO_U_INDEX + curr_d) += p;

    return res;
  }

  // Conversion from conserved to primitive variables
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::cons2prim(const FluxValue<cfg>& cons) const {
    FluxValue<cfg> prim = cons;

    /*--- Pre-fetch some variables used multiple times in order to exploit possible vectorization ---*/
    const auto m1 = cons(M1_INDEX);
    const auto m2 = cons(M2_INDEX);

    /*--- Compute primitive variables ---*/
    const auto rho     = m1 + m2;
    const auto inv_rho = static_cast<Number>(1.0)/rho;
    const auto alpha1  = cons(RHO_ALPHA1_INDEX)*inv_rho;
    prim(ALPHA1_INDEX) = alpha1;

    for(std::size_t d = 0; d < Field::dim; ++d) {
      prim(U_INDEX + d) = cons(RHO_U_INDEX + d)*inv_rho;
    }

    const auto rho1 = m1/alpha1; /*--- TODO: Add a check in case of zero volume fraction ---*/
    prim(P1_INDEX)  = EOS_phase1.pres_value(rho1);
    const auto rho2 = m2/(static_cast<Number>(1.0) - alpha1);
                      /*--- TODO: Add a check in case of zero volume fraction ---*/
    prim(P2_INDEX)  = EOS_phase2.pres_value(rho2);

    return prim;
  }

  // Conversion from primitive to conserved variables
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::prim2cons(const FluxValue<cfg>& prim) const {
    FluxValue<cfg> cons;

    /*--- Pre-fetch some variables used multiple times in order to exploit possible vectorization ---*/
    const auto alpha1 = prim(ALPHA1_INDEX);

    /*--- Compute conserved variables ---*/
    const auto rho1 = EOS_phase1.rho_value(prim(P1_INDEX));
    const auto m1   = alpha1*rho1;
    cons(M1_INDEX)  = m1;

    const auto rho2 = EOS_phase2.rho_value(prim(P2_INDEX));
    const auto m2   = (static_cast<Number>(1.0) - alpha1)*rho2;
    cons(M2_INDEX)  = m2;

    const auto rho         = m1 + m2;
    cons(RHO_ALPHA1_INDEX) = rho*prim(ALPHA1_INDEX);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      cons(RHO_U_INDEX + d) = rho*prim(U_INDEX + d);
    }

    return cons;
  }

  // Perform reconstruction for order 2 scheme
  //
  #ifdef ORDER_2
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
      const auto beta = static_cast<Number>(1.0); // MINMOD limiter
      for(std::size_t comp = 0; comp < Field::n_comp; ++comp) {
        if(primR(comp) - primL(comp) > static_cast<Number>(0.0)) {
          primL_recon(comp) += static_cast<Number>(0.5)*
                               std::max(static_cast<Number>(0.0),
                                        std::max(std::min(beta*(primL(comp) - primLL(comp)),
                                                          primR(comp) - primL(comp)),
                                                 std::min(primL(comp) - primLL(comp),
                                                          beta*(primR(comp) - primL(comp)))));
        }
        else if(primR(comp) - primL(comp) < static_cast<Number>(0.0)) {
          primL_recon(comp) += static_cast<Number>(0.5)*
                               std::min(static_cast<Number>(0.0),
                                        std::min(std::max(beta*(primL(comp) - primLL(comp)),
                                                          primR(comp) - primL(comp)),
                                                 std::max(primL(comp) - primLL(comp),
                                                          beta*(primR(comp) - primL(comp)))));
        }

        if(primRR(comp) - primR(comp) > static_cast<Number>(0.0)) {
          primR_recon(comp) -= static_cast<Number>(0.5)*
                               std::max(static_cast<Number>(0.0),
                                        std::max(std::min(beta*(primR(comp) - primL(comp)),
                                                          primRR(comp) - primR(comp)),
                                                 std::min(primR(comp) - primL(comp),
                                                          beta*(primRR(comp) - primR(comp)))));
        }
        else if(primRR(comp) - primR(comp) < static_cast<Number>(0.0)) {
          primR_recon(comp) -= static_cast<Number>(0.5)*
                               std::min(static_cast<Number>(0.0),
                                        std::min(std::max(beta*(primR(comp) - primL(comp)),
                                                          primRR(comp) - primR(comp)),
                                                 std::max(primR(comp) - primL(comp),
                                                          beta*(primRR(comp) - primR(comp)))));
        }
      }
    }
  #endif

  // Relax reconstruction
  //
  #ifdef ORDER_2
    #ifdef RELAX_RECONSTRUCTION
      // Perform a Newton step relaxation for a single vector state (i.e. a single cell)
      //
      template<class Field>
      template<typename State>
      void Flux<Field>::perform_Newton_step_relaxation(State conserved_variables,
                                                       Number& dalpha1,
                                                       Number& alpha1,
                                                       bool& relaxation_applied) {
        /*--- Pre-fetch some variables used multiple times in order to exploit possible vectorization ---*/
        const auto m1 = conserved_variables(M1_INDEX);
        const auto m2 = conserved_variables(M2_INDEX);

        /*--- Update auxiliary values affected by the nonlinear function for which we seek a zero ---*/
        const auto rho1 = m1/alpha1; /*--- TODO: Add a check in case of zero volume fraction ---*/
        const auto p1   = EOS_phase1.pres_value(rho1);

        const auto rho2 = m2/(static_cast<Number>(1.0) - alpha1); /*--- TODO: Add a check in case of zero volume fraction ---*/
        const auto p2   = EOS_phase2.pres_value(rho2);

        /*--- Compute the nonlinear function for which we seek the zero (basically the pressure equilibrium) ---*/
        const auto F = p1 - p2;

        /*--- Perform the relaxation only where really needed ---*/
        if(!std::isnan(F) && std::abs(F) > atol_Newton + rtol_Newton*EOS_phase1.get_p0() &&
           std::abs(dalpha1) > atol_Newton) {
          relaxation_applied = true;

          /*--- Compute the derivative w.r.t volume fraction recalling that for a barotropic EOS dp/drho = c^2 ---*/
          const auto dF_dalpha1 = -m1/(alpha1*alpha1)*
                                  EOS_phase1.c_value(rho1)*EOS_phase1.c_value(rho1)
                                  -m2/((static_cast<Number>(1.0) - alpha1)*
                                       (static_cast<Number>(1.0) - alpha1))*
                                  EOS_phase2.c_value(rho2)*EOS_phase2.c_value(rho2);

          /*--- Compute the volume fraction update with a bound-preserving strategy ---*/
          dalpha1 = -F/dF_dalpha1;
          if(dalpha1 > static_cast<Number>(0.0)) {
            dalpha1 = std::min(dalpha1,
                               lambda*(static_cast<Number>(1.0) - alpha1));
          }
          else if(dalpha1 < static_cast<Number>(0.0)) {
            dalpha1 = std::max(dalpha1, -lambda*alpha1);
          }

          #ifdef VERBOSE
            if(alpha1 + dalpha1 < static_cast<Number>(0.0) ||
               alpha1 + dalpha1 > static_cast<Number>(1.0)) {
              // We should never arrive here thanks to the bound-preserving strategy. Added only for the sake of safety
              throw std::runtime_error("Bounds exceeding value for the volume fraction inside Newton step");
            }
          #endif
          alpha1 += dalpha1;
        }

        /*--- Update the vector of conserved variables ---*/
        conserved_variables(RHO_ALPHA1_INDEX) = (m1 + m2)*alpha1;
      }

      // Relax reconstructed state if desired
      //
      template<class Field>
      void Flux<Field>::relax_reconstruction(FluxValue<cfg>& q) {
        /*--- Declare and set relevant parameters ---*/
        std::size_t Newton_iter = 0;
        bool relaxation_applied = true;

        auto dalpha1 = std::numeric_limits<Number>::infinity();
        auto alpha1  = q(RHO_ALPHA1_INDEX)/(q(M1_INDEX) + q(M2_INDEX));

        /*--- Apply Newton method ---*/
        while(relaxation_applied == true) {
          relaxation_applied = false;
          Newton_iter++;

          this->perform_Newton_step_relaxation(q, dalpha1, alpha1, relaxation_applied);

          // Newton cycle diverged
          if(Newton_iter > max_Newton_iters && relaxation_applied == true) {
            std::cout << "Netwon method not converged in the relaxation after MUSCL" << std::endl;
            exit(1);
          }
        }
      }
    #endif
  #endif

} // end namespace samurai
