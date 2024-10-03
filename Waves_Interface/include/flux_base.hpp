// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#ifndef flux_base_hpp
#define flux_base_hpp

#pragma once
#include <samurai/schemes/fv.hpp>

#include "barotropic_eos.hpp"

// Preprocessor to define whether order 2 is desired
#define ORDER_2

// Preprocessor to define whether relaxation is desired after reconstruction for order 2
#ifdef ORDER_2
  #define RELAX_RECONSTRUCTION
#endif

/**
 * Useful parameters and enumerators
 */
namespace EquationData {
  // Declare spatial dimension
  static constexpr std::size_t dim = 1;

  // Declare some parameters related to EOS.
  static constexpr double p0_phase1   = 1e5;
  static constexpr double p0_phase2   = 1e5;

  static constexpr double rho0_phase1 = 1e3;
  static constexpr double rho0_phase2 = 1.0;

  static constexpr double c0_phase1   = 1631.617602258568427706;
  static constexpr double c0_phase2   = 340.0;

  // Use auxiliary variables for the indices for the sake of generality
  static constexpr std::size_t M1_INDEX         = 0;
  static constexpr std::size_t M2_INDEX         = 1;
  static constexpr std::size_t RHO_ALPHA1_INDEX = 2;
  static constexpr std::size_t RHO_U_INDEX      = 3;

  static constexpr std::size_t ALPHA1_INDEX = RHO_ALPHA1_INDEX;

  // Save also the total number of (scalar) variables
  static constexpr std::size_t NVARS = 3 + dim;
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
    static constexpr std::size_t field_size = Field::size;
    static_assert(field_size == EquationData::NVARS, "The number of elements in the state does not correspond to the number of equations");
    static_assert(Field::dim == EquationData::dim, "The spatial dimensions between Field and the parameter list do not match");
    static constexpr std::size_t output_field_size = field_size;
    #ifdef ORDER_2
      static constexpr std::size_t stencil_size = 4;
    #else
      static constexpr std::size_t stencil_size = 2;
    #endif

    using cfg = FluxConfig<SchemeType::NonLinear, output_field_size, stencil_size, Field>;

    Flux(const BarotropicEOS<>& EOS_phase1,
         const BarotropicEOS<>& EOS_phase2,
         const double eps_); // Constructor which accepts in inputs the equations of state of the two phases

    template<typename State>
    void perform_Newton_step_relaxation(std::unique_ptr<State> conserved_variables,
                                        typename Field::value_type& dalpha1, typename Field::value_type& alpha1,
                                        typename Field::value_type& rho, bool& relaxation_applied,
                                        const double tol = 1e-12, const double lambda = 0.9); // Perform a Newton step relaxation for a state vector
                                                                                              // (it is not a real space dependent procedure,
                                                                                              // but I would need to be able to do it inside the flux location
                                                                                              // for MUSCL reconstruction)

  protected:
    const BarotropicEOS<>& phase1;
    const BarotropicEOS<>& phase2;

    const double eps; // Tolerance of pure phase to set NaNs

    FluxValue<cfg> evaluate_continuous_flux(const FluxValue<cfg>& q,
                                            const std::size_t curr_d); // Evaluate the 'continuous' flux for the state q along direction curr_d

    FluxValue<cfg> cons2prim(const FluxValue<cfg>& cons) const; // Conversion from conservative to primitive variables

    FluxValue<cfg> prim2cons(const FluxValue<cfg>& prim) const; // Conversion from primitive to conservative variables

    #ifdef ORDER_2
      void perform_reconstruction(const FluxValue<cfg>& primLL,
                                  const FluxValue<cfg>& primL,
                                  const FluxValue<cfg>& primR,
                                  const FluxValue<cfg>& primRR,
                                  FluxValue<cfg>& primL_recon,
                                  FluxValue<cfg>& primR_recon); // Reconstruction for second order scheme

      #ifdef RELAX_RECONSTRUCTION
        void relax_reconstruction(FluxValue<cfg>& q); // Relax reconstructed state
      #endif
    #endif
  };

  // Class constructor in order to be able to work with the equation of state
  //
  template<class Field>
  Flux<Field>::Flux(const BarotropicEOS<>& EOS_phase1,
                    const BarotropicEOS<>& EOS_phase2,
                    const double eps_): phase1(EOS_phase1), phase2(EOS_phase2), eps(eps_) {}

  // Evaluate the 'continuous flux'
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::evaluate_continuous_flux(const FluxValue<cfg>& q,
                                                                             const std::size_t curr_d) {
    // Sanity check in terms of dimensions
    assert(curr_d < EquationData::dim);

    // Initialize the resulting variable
    FluxValue<cfg> res = q;

    // Compute the current density and velocity
    const auto rho   = q(M1_INDEX) + q(M2_INDEX);
    const auto vel_d = q(RHO_U_INDEX + curr_d)/rho;

    // Multiply the state by the velocity along the direction of interest
    res(M1_INDEX) *= vel_d;
    res(M2_INDEX) *= vel_d;
    res(RHO_ALPHA1_INDEX) *= vel_d;
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      res(RHO_U_INDEX + d) *= vel_d;
    }

    // Compute and add the contribution due to the pressure
    const auto alpha1 = q(RHO_ALPHA1_INDEX)/rho;
    const auto rho1   = (alpha1 > eps) ? q(M1_INDEX)/alpha1 : nan("");
    const auto p1     = phase1.pres_value(rho1);

    const auto alpha2 = 1.0 - alpha1;
    const auto rho2   = (alpha2 > eps) ? q(M2_INDEX)/alpha2 : nan("");
    const auto p2     = phase2.pres_value(rho2);

    const auto p      = (alpha1 > eps && alpha2 > eps) ?
                        alpha1*p1 + alpha2*p2 :
                        ((alpha1 < eps) ? p2 : p1);

    res(RHO_U_INDEX + curr_d) += p;

    return res;
  }

  // Conversion from conserved to primitive variables
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::cons2prim(const FluxValue<cfg>& cons) const {
    // Create a copy of the state to save the output
    FluxValue<cfg> prim = cons;

    // Apply conversion only to the volume fraction
    prim(ALPHA1_INDEX) = cons(RHO_ALPHA1_INDEX)/(cons(M1_INDEX) + cons(M2_INDEX));

    return prim;
  }

  // Conversion from primitive to conserved variables
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::prim2cons(const FluxValue<cfg>& prim) const {
    // Create a copy of the state to save the output
    FluxValue<cfg> cons = prim;

    // Apply conversion only to the mixture density times volume fraction
    cons(RHO_ALPHA1_INDEX) = (cons(M1_INDEX) + cons(M2_INDEX))*prim(ALPHA1_INDEX);

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
      // Initialize with the original state
      primL_recon = primL;
      primR_recon = primR;

      // Perform the reconstruction.
      const double beta = 1.0; // MINMOD limiter
      for(std::size_t comp = 0; comp < Field::size; ++comp) {
        if(primR(comp) - primL(comp) > 0.0) {
          primL_recon(comp) += 0.5*std::max(0.0, std::max(std::min(beta*(primL(comp) - primLL(comp)),
                                                                   primR(comp) - primL(comp)),
                                                          std::min(primL(comp) - primLL(comp),
                                                                   beta*(primR(comp) - primL(comp)))));
        }
        else if(primR(comp) - primL(comp) < 0.0) {
          primL_recon(comp) += 0.5*std::min(0.0, std::min(std::max(beta*(primL(comp) - primLL(comp)),
                                                                   primR(comp) - primL(comp)),
                                                          std::max(primL(comp) - primLL(comp),
                                                                   beta*(primR(comp) - primL(comp)))));
        }

        if(primRR(comp) - primR(comp) > 0.0) {
          primR_recon(comp) -= 0.5*std::max(0.0, std::max(std::min(beta*(primR(comp) - primL(comp)),
                                                                   primRR(comp) - primR(comp)),
                                                          std::min(primR(comp) - primL(comp),
                                                                   beta*(primRR(comp) - primR(comp)))));
        }
        else if(primRR(comp) - primR(comp) < 0.0) {
          primR_recon(comp) -= 0.5*std::min(0.0, std::min(std::max(beta*(primR(comp) - primL(comp)),
                                                                   primRR(comp) - primR(comp)),
                                                          std::max(primR(comp) - primL(comp),
                                                                   beta*(primRR(comp) - primR(comp)))));
        }
      }
    }
  #endif

  // Perform a Newton step relaxation for a single vector state (i.e. a single cell)
  //
  template<class Field>
  template<typename State>
  void Flux<Field>::perform_Newton_step_relaxation(std::unique_ptr<State> conserved_variables,
                                                   typename Field::value_type& dalpha1, typename Field::value_type& alpha1,
                                                   typename Field::value_type& rho, bool& relaxation_applied,
                                                   const double tol, const double lambda) {
    // Reinitialization of partial masses in case of evanascent volume fraction
    if(alpha1 < eps) {
      (*conserved_variables)(M1_INDEX) = alpha1*EquationData::rho0_phase1;
    }
    if(1.0 - alpha1 < eps) {
      (*conserved_variables)(M2_INDEX) = (1.0 - alpha1)*EquationData::rho0_phase2;
    }

    // Update auxiliary values affected by the nonlinear function for which we seek a zero
    const auto rho1   = (alpha1 > eps) ? (*conserved_variables)(M1_INDEX)/alpha1 : nan("");
    const auto p1     = phase1.pres_value(rho1);

    const auto alpha2 = 1.0 - alpha1;
    const auto rho2   = (alpha2 > eps) ? (*conserved_variables)(M2_INDEX)/alpha2 : nan("");
    const auto p2     = phase2.pres_value(rho2);

    // Compute the nonlinear function for which we seek the zero (basically the Laplace law)
    const auto F = p1 - p2;

    // Perform the relaxation only where really needed
    if(!std::isnan(F) && std::abs(F) > tol*EquationData::p0_phase1 && std::abs(dalpha1) > tol && alpha1 > eps && alpha2 > eps) {
      relaxation_applied = true;

      // Compute the derivative w.r.t volume fraction recalling that for a barotropic EOS dp/drho = c^2
      const auto dF_dalpha1 = -(*conserved_variables)(M1_INDEX)/(alpha1*alpha1)*
                               phase1.c_value(rho1)*phase1.c_value(rho1)
                              -(*conserved_variables)(M2_INDEX)/(alpha2*alpha2)*
                               phase2.c_value(rho2)*phase2.c_value(rho2);

      // Compute the volume fraction update with a bound-preserving strategy
      dalpha1 = -F/dF_dalpha1;
      if(dalpha1 > 0.0) {
        dalpha1 = std::min(dalpha1, lambda*alpha2);
      }
      else if(dalpha1 < 0.0) {
        dalpha1 = std::max(dalpha1, -lambda*alpha1);
      }

      if(alpha1 + dalpha1 < 0.0 || alpha1 + dalpha1 > 1.0) {
        // We should never arrive here thakns to the bound-preserving strategy. Added only for the sake fo safety
        std::cerr << "Bounds exceeding value for the volume fraction inside Newton step " << std::endl;
      }
      else {
        alpha1 += dalpha1;
      }
    }

    // Update the vector of conserved variables
    rho = (*conserved_variables)(M1_INDEX)
        + (*conserved_variables)(M2_INDEX);
    (*conserved_variables)(RHO_ALPHA1_INDEX) = rho*alpha1;
  }

  // Relax reconstruction
  //
  #ifdef ORDER_2
    #ifdef RELAX_RECONSTRUCTION
      template<class Field>
      void Flux<Field>::relax_reconstruction(FluxValue<cfg>& q) {
        // Declare and set relevant parameters
        const double tol    = 1e-12; // Tolerance of the Newton method
        const double lambda = 0.9;   // Parameter for bound preserving strategy
        std::size_t Newton_iter = 0;
        bool relaxation_applied = true;
        typename Field::value_type dalpha1 = std::numeric_limits<typename Field::value_type>::infinity();
        typename Field::value_type alpha1  = q(RHO_ALPHA1_INDEX)/(q(M1_INDEX) + q(M2_INDEX));
        typename Field::value_type rho     = q(M1_INDEX) + q(M2_INDEX);

        // Apply Newton method
        while(relaxation_applied == true) {
          relaxation_applied = false;
          Newton_iter++;

          this->perform_Newton_step_relaxation(std::make_unique<FluxValue<cfg>>(q),
                                               dalpha1, alpha1, rho,
                                               relaxation_applied, tol, lambda);

          // Newton cycle diverged
          if(Newton_iter > 60) {
            std::cout << "Netwon method not converged in the relaxation after MUSCL" << std::endl;
            exit(1);
          }
        }
      }
    #endif
  #endif

} // end namespace samurai

#endif
