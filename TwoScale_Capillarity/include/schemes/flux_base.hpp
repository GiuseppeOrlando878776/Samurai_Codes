// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2026
//
#pragma once

#include <samurai/schemes/fv.hpp>

#include "../barotropic_eos.hpp"
#include "../utilities.hpp"

/*--- Preprocessor to define whether order 2 is desired ---*/
#define ORDER_2

/*--- Preprocessor to define whether relaxation is desired after reconstruction for order 2 ---*/
#ifdef ORDER_2
  #define RELAX_RECONSTRUCTION
#endif

namespace samurai {
  using namespace EquationData;

  /**
   * Generic class to compute the flux between a left and right state
   */
  template<class Field>
  class Flux {
  public:
    // Definitions and sanity checks
    static_assert(Field::dim == EquationData::dim, "The spatial dimensions between Field and the parameter list do not match");
    static_assert(Field::n_comp == EquationData::NVARS, "The number of elements in the state does not correspond to the number of equations");
    #ifdef ORDER_2
      static constexpr std::size_t stencil_size = 4;
    #else
      static constexpr std::size_t stencil_size = 2;
    #endif

    using cfg = FluxConfig<SchemeType::NonLinear, stencil_size, Field, Field>;

    template<class Field_Vect>
    using cfg_st = FluxConfig<SchemeType::NonLinear, stencil_size, Field, Field_Vect>;

    using Number = typename Field::value_type; // Shortcut for the arithmetic type

    /**
     * Class constructor
     * @param EOS_phase_1_ phase 1 equation of state
     * @param EOS_phase_2_ phase 2 equation of state
     * @param sigma_ surface tension coefficient
     * @param lambda_ bound-preserving parameter
     * @param atol_Newton_ absolute tolerance for dual-time stepping
     * @param rtol_Newton_ relative tolerance for dual-time stepping
     * @param max_Newton_iters_ maximum number of iterations for dual-time stepping
     */
    Flux(const LinearizedBarotropicEOS<Number>& EOS_phase1_,
         const LinearizedBarotropicEOS<Number>& EOS_phase2_,
         const Number sigma_,
         const Number lambda_ = static_cast<Number>(0.9),
         const Number atol_Newton_ = static_cast<Number>(1e-14),
         const Number rtol_Newton_ = static_cast<Number>(1e-12),
         const std::size_t max_Newton_iters_ = 60);

  protected:
    const LinearizedBarotropicEOS<Number>& EOS_phase1;
    const LinearizedBarotropicEOS<Number>& EOS_phase2;

    const Number sigma; /*!< Surface tension parameter */

    const Number      lambda;           /*!< Parameter for bound-preserving strategy */
    const Number      atol_Newton;      /*!< Absolute tolerance Newton method relaxation */
    const Number      rtol_Newton;      /*!< Relative tolerance Newton method relaxation */
    const std::size_t max_Newton_iters; /*!< Maximum number of Newton iterations */

    /**
     * Evaluate the 'continuous' flux
     * @param q state
     * @param curr_d current direction
     * @param grad_alpha1_bar gradient of 'effective' large-scale volume fraction (needed for capillarity)
     */
    FluxValue<cfg> evaluate_continuous_flux(const FluxValue<cfg>& q,
                                            const std::size_t curr_d,
                                            const auto& grad_alpha1_bar);

    /**
     * Evaluate the hyperbolic operator
     * @param q state
     * @param curr_d current direction
     */
    FluxValue<cfg> evaluate_hyperbolic_operator(const FluxValue<cfg>& q,
                                                const std::size_t curr_d);

    /**
     * Evaluate the surface tension operator
     * @param grad_alpha1_bar gradient of effective large-scale volume fraction
     * @param curr_d current direction
     */
    template<class Field_Vect>
    FluxValue<cfg_st<Field_Vect>> evaluate_surface_tension_operator(const auto& grad_alpha1_bar,
                                                                    const std::size_t curr_d);

    /**
     * Conversion from conserved to primitive variables
     * @param cons conserved variables
     * @return prim primitive variables
     */
    FluxValue<cfg> cons2prim(const FluxValue<cfg>& cons) const;

    /**
     * Conversion from primitive to conserved variables
     * @param prim primitive variables
     * @return cons conserved variables
     */
    FluxValue<cfg> prim2cons(const FluxValue<cfg>& prim) const;

    #ifdef RELAX_RECONSTRUCTION
      /**
       * Perform a Newton step relaxation for a state vector
         (it is not a real space dependent procedure, but I would need to be able
          to do it inside the flux location for MUSCL reconstruction)
       * @param conserved_variables state
       * @param H_bar large-scale curvature
       * @param dalpha1_bar variation of large-scale volume fraction
       * @param alpha1_bar large-scale volume fraction
       * @param relaxation_applied flag to check whether relaxation has been applied or we converged
       */
      void perform_Newton_step_relaxation(auto conserved_variables,
                                          const Number H_bar,
                                          Number& dalpha1_bar,
                                          Number& alpha1_bar,
                                          bool& relaxation_applied);

      /**
       * Relax reconstructed state
       * @param q state
       * @param H_bar large-scale curvature
       */
      void relax_reconstruction(FluxValue<cfg>& q,
                                const Number H_bar);
    #endif
  };

  // Class constructor in order to be able to work with the equation of state
  //
  template<class Field>
  Flux<Field>::Flux(const LinearizedBarotropicEOS<Number>& EOS_phase1_,
                    const LinearizedBarotropicEOS<Number>& EOS_phase2_,
                    const Number sigma_,
                    const Number lambda_,
                    const Number atol_Newton_,
                    const Number rtol_Newton_,
                    const std::size_t max_Newton_iters_):
    EOS_phase1(EOS_phase1_), EOS_phase2(EOS_phase2_), sigma(sigma_),
    lambda(lambda_), atol_Newton(atol_Newton_), rtol_Newton(rtol_Newton_),
    max_Newton_iters(max_Newton_iters_) {}

  // Evaluate the 'continuous flux'
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg>
  Flux<Field>::evaluate_continuous_flux(const FluxValue<cfg>& q,
                                        const std::size_t curr_d,
                                        const auto& grad_alpha1_bar) {
    // Initialize the resulting variable with the hyperbolic operator
    FluxValue<cfg> res = this->evaluate_hyperbolic_operator(q, curr_d);

    // Add the contribution due to surface tension
    res += this->evaluate_surface_tension_operator(grad_alpha1_bar, curr_d);

    return res;
  }

  // Evaluate the hyperbolic part of the 'continuous' flux
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg>
  Flux<Field>::evaluate_hyperbolic_operator(const FluxValue<cfg>& q,
                                            const std::size_t curr_d) {
    // Sanity check in terms of dimensions
    assert(curr_d < Field::dim);

    // Initialize the resulting variable
    FluxValue<cfg> res = q;

    // Pre-fetch some variables used multiple times in order to exploit possible vectorization
    const auto m1   = q(M1_INDEX);
    const auto m2   = q(M2_INDEX);
    const auto m1_d = q(M1_D_INDEX);

    // Compute the current velocity
    const auto rho     = m1 + m2 + m1_d;
    const auto inv_rho = static_cast<Number>(1.0)/rho;
    const auto vel_d   = q(RHO_U_INDEX + curr_d)*inv_rho;

    // Multiply the state the velocity along the direction of interest
    res(M1_INDEX) *= vel_d;
    res(M2_INDEX) *= vel_d;
    res(M1_D_INDEX) *= vel_d;
    res(ALPHA1_D_INDEX) *= vel_d;
    res(SIGMA_D_INDEX) *= vel_d;
    res(RHO_ALPHA1_BAR_INDEX) *= vel_d;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      res(RHO_U_INDEX + d) *= vel_d;
    }

    // Compute and add the contribution due to the pressure
    const auto alpha1_bar = q(RHO_ALPHA1_BAR_INDEX)*inv_rho;
    const auto alpha1_d   = q(ALPHA1_D_INDEX);
    const auto alpha1     = alpha1_bar*(static_cast<Number>(1.0) - alpha1_d);
    const auto rho1       = m1/alpha1; // TODO: Add a check in case of zero volume fraction
    const auto p1         = EOS_phase1.pres_value(rho1);

    const auto alpha2     = static_cast<Number>(1.0) - alpha1 - alpha1_d;
    const auto rho2       = m2/alpha2; // TODO: Add a check in case of zero volume fraction
    const auto p2         = EOS_phase2.pres_value(rho2);

    const auto p_bar      = alpha1_bar*p1
                          + (static_cast<Number>(1.0) - alpha1_bar)*p2;

    res(RHO_U_INDEX + curr_d) += p_bar;

    return res;
  }

  // Evaluate the surface tension operator
  //
  template<class Field>
  template<class Field_Vect>
  FluxValue<typename Flux<Field>::template cfg_st<Field_Vect>>
  Flux<Field>::evaluate_surface_tension_operator(const auto& grad_alpha1_bar,
                                                 const std::size_t curr_d) {
    // Sanity check in terms of dimensions
    assert(curr_d < Field::dim);

    // Initialize the resulting variable
    FluxValue<cfg> res;

    // Set to zero all the contributions
    res.fill(static_cast<Number>(0.0));

    // Add the contribution due to surface tension
    //const auto mod_grad_alpha1_bar = std::sqrt(xt::sum(grad_alpha1_bar*grad_alpha1_bar)());
    auto mod2_grad_alpha1_bar = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      mod2_grad_alpha1_bar += grad_alpha1_bar[d]*grad_alpha1_bar[d];
    }
    const auto mod_grad_alpha1_bar = std::sqrt(mod2_grad_alpha1_bar);

    const auto n  = grad_alpha1_bar/(mod_grad_alpha1_bar + static_cast<Number>(1e-10));
    const auto nx = n(0);
    const auto ny = n(1);

    if(curr_d == 0) {
      res(RHO_U_INDEX) += sigma*(nx*nx - static_cast<Number>(1.0))*mod_grad_alpha1_bar;
      res(RHO_U_INDEX + 1) += sigma*nx*ny*mod_grad_alpha1_bar;
    }
    else if(curr_d == 1) {
      res(RHO_U_INDEX) += sigma*nx*ny*mod_grad_alpha1_bar;
      res(RHO_U_INDEX + 1) += sigma*(ny*ny - static_cast<Number>(1.0))*mod_grad_alpha1_bar;
    }

    return res;
  }

  // Conversion from conserved to primitive variables
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::cons2prim(const FluxValue<cfg>& cons) const {
    FluxValue<cfg> prim;

    // Pre-fetch some variables used multiple times in order to exploit possible vectorization
    const auto m1       = cons(M1_INDEX);
    const auto m2       = cons(M2_INDEX);
    const auto m1_d     = cons(M1_D_INDEX);
    const auto alpha1_d = cons(ALPHA1_D_INDEX);

    // Compute the primitive variables
    const auto rho         = m1 + m2 + m1_d;
    const auto inv_rho     = static_cast<Number>(1.0)/rho;
    const auto alpha1_bar  = cons(RHO_ALPHA1_BAR_INDEX)*inv_rho;
    prim(ALPHA1_BAR_INDEX) = alpha1_bar;
    const auto alpha1      = alpha1_bar*(static_cast<Number>(1.0) - alpha1_d);
    prim(P1_INDEX)         = EOS_phase1.pres_value(m1/alpha1); // TODO: Add a check in case of zero volume fraction
    prim(P2_INDEX)         = EOS_phase2.pres_value(m2/(static_cast<Number>(1.0) - alpha1 - alpha1_d));
                             // TODO: Add a check in case of zero volume fraction
    for(std::size_t d = 0; d < Field::dim; ++d) {
      prim(U_INDEX + d) = cons(RHO_U_INDEX + d)*inv_rho;
    }
    prim(M1_D_INDEX)     = m1_d;
    prim(ALPHA1_D_INDEX) = alpha1_d;
    prim(SIGMA_D_INDEX)  = cons(SIGMA_D_INDEX);

    return prim;
  }

  // Conversion from primitive to conserved variables
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::prim2cons(const FluxValue<cfg>& prim) const {
    FluxValue<cfg> cons;

    // Pre-fetch some variables used multiple times in order to exploit possible vectorization
    const auto alpha1_bar = prim(ALPHA1_BAR_INDEX);
    const auto alpha1_d   = prim(ALPHA1_D_INDEX);
    const auto m1_d       = prim(M1_D_INDEX);

    // Compute the conserved variables
    const auto alpha1          = alpha1_bar*(static_cast<Number>(1.0) - alpha1_d);
    const auto m1              = alpha1*EOS_phase1.rho_value(prim(P1_INDEX));
    cons(M1_INDEX)             = m1;
    const auto m2              = (static_cast<Number>(1.0) - alpha1 - alpha1_d)*
                                 EOS_phase2.rho_value(prim(P2_INDEX));
    cons(M2_INDEX)             = m2;
    cons(M1_D_INDEX)           = m1_d;
    const auto rho             = m1 + m2 + m1_d;
    cons(RHO_ALPHA1_BAR_INDEX) = rho*alpha1_bar;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      cons(RHO_U_INDEX + d) = rho*prim(U_INDEX + d);
    }
    cons(ALPHA1_D_INDEX) = alpha1_d;
    cons(SIGMA_D_INDEX)  = prim(SIGMA_D_INDEX);

    return cons;
  }

  // Relax reconstruction
  //
  #ifdef RELAX_RECONSTRUCTION
    // Perform a Newton step relaxation for a single vector state (i.e. a single cell) without mass transfer
    //
    template<class Field>
    void Flux<Field>::perform_Newton_step_relaxation(auto conserved_variables,
                                                     const Number H_bar,
                                                     Number& dalpha1_bar,
                                                     Number& alpha1_bar,
                                                     bool& relaxation_applied) {
      if(!std::isnan(H_bar)) {
        // Pre-fetch some variables used multiple times in order to exploit possible vectorization
        const auto m1       = conserved_variables(M1_INDEX);
        const auto m2       = conserved_variables(M2_INDEX);
        const auto m1_d     = conserved_variables(M1_D_INDEX);
        const auto alpha1_d = conserved_variables(ALPHA1_D_INDEX);

        // Update auxiliary values affected by the nonlinear function for which we seek a zero
        const auto alpha1 = alpha1_bar*(static_cast<Number>(1.0) - alpha1_d);
        const auto rho1   = m1/alpha1; // TODO: Add a check in case of zero volume fraction
        const auto p1     = EOS_phase1.pres_value(rho1);

        const auto alpha2 = static_cast<Number>(1.0) - alpha1 - alpha1_d;
        const auto rho2   = m2/alpha2; // TODO: Add a check in case of zero volume fraction
        const auto p2     = EOS_phase2.pres_value(rho2);

        // Compute the nonlinear function for which we seek the zero (basically the local Laplace law)
        const auto F = (static_cast<Number>(1.0) - alpha1_d)*(p1 - p2)
                     - sigma*H_bar;

        // Perform the relaxation only where really needed
        if(std::abs(F) > atol_Newton + rtol_Newton*std::min(EOS_phase1.get_p0(), sigma*std::abs(H_bar)) &&
           std::abs(dalpha1_bar) > atol_Newton) {
          relaxation_applied = true;

          // Compute the derivative w.r.t large-scale volume fraction recalling that for a barotropic EOS dp/drho = c^2
          const auto c1 = EOS_phase1.c_value(rho1);
          const auto c2 = EOS_phase2.c_value(rho2);

          const auto alpha2_bar = static_cast<Number>(1.0) - alpha1_bar;

          const auto dF_dalpha1_bar = -m1/(alpha1_bar*alpha1_bar)*
                                      c1*c1
                                      -m2/(alpha2_bar*alpha2_bar)*
                                      c2*c2;

          // Compute the large-scale volume fraction update
          dalpha1_bar = -F/dF_dalpha1_bar;

          if(dalpha1_bar > static_cast<Number>(0.0)) {
            dalpha1_bar = std::min(dalpha1_bar, lambda*alpha2_bar);
          }
          else if(dalpha1_bar < static_cast<Number>(0.0)) {
            dalpha1_bar = std::max(dalpha1_bar, -lambda*alpha1_bar);
          }

          #ifdef DEBUG_FLUX
            if(alpha1_bar + dalpha1_bar < static_cast<Number>(0.0) ||
               alpha1_bar + dalpha1_bar > static_cast<Number>(1.0)) {
              // I should never get here. Added only for the sake of safety!!
              throw std::runtime_error("Bounds exceeding value for large-scale volume fraction inside Newton step of reconstruction");
            }
          #endif
          alpha1_bar += dalpha1_bar;
        }

        // Update the vector of conserved variables (probably not the optimal choice since I need this update only at the end of the Newton loop,
        // but the most coherent one thinking about the transfer of mass)
        conserved_variables(RHO_ALPHA1_BAR_INDEX) = (m1 + m2 + m1_d)*alpha1_bar;
      }
    }

    // Relax the reconstruction
    //
    template<class Field>
    void Flux<Field>::relax_reconstruction(FluxValue<cfg>& q,
                                           const Number H_bar) {
      // Declare and set relevant parameters
      bool relaxation_applied = true;

      Number dalpha1_bar = std::numeric_limits<Number>::infinity();
      Number alpha1_bar  = q(RHO_ALPHA1_BAR_INDEX)/
                           (q(M1_INDEX) + q(M2_INDEX) + q(M1_D_INDEX));

      // Apply Newton method
      for(std::size_t Newton_iter = 1; Newton_iter <= max_Newton_iters; ++Newton_iter) {
        relaxation_applied = false;

        try {
          this->perform_Newton_step_relaxation(q, H_bar, dalpha1_bar, alpha1_bar, relaxation_applied);
        }
        catch(std::exception& e) {
          std::cerr << e.what() << std::endl;
          exit(1);
        }

        // Converged: no further relaxation requested
        if(!relaxation_applied) {
          break;
        }
      }

      // Newton cycle diverged
      if(relaxation_applied) {
        std::cerr << "Netwon method not converged in the relaxation after MUSCL" << std::endl;
        exit(1);
      }
    }
  #endif

} // end namespace samurai
