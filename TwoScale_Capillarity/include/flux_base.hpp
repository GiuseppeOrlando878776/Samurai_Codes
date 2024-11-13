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
//#define ORDER_2

// Preprocessor to define whether relaxation is desired after reconstruction for order 2
#ifdef ORDER_2
  #define RELAX_RECONSTRUCTION
#endif

/**
 * Useful parameters and enumerators
 */
namespace EquationData {
  // Declare spatial dimension
  static constexpr std::size_t dim = 2;

  // Use auxiliary variables for the indices for the sake of generality
  static constexpr std::size_t M1_INDEX             = 0;
  static constexpr std::size_t M2_INDEX             = 1;
  static constexpr std::size_t M1_D_INDEX           = 2;
  static constexpr std::size_t ALPHA1_D_INDEX       = 3;
  static constexpr std::size_t SIGMA_D_INDEX        = 4;
  static constexpr std::size_t RHO_ALPHA1_BAR_INDEX = 5;
  static constexpr std::size_t RHO_U_INDEX          = 6;

  // Save also the total number of (scalar) variables
  static constexpr std::size_t NVARS = 6 + dim;

  // Use auxiliary variables for the indices also for primitive variables for the sake of generality
  static constexpr std::size_t ALPHA1_BAR_INDEX = RHO_ALPHA1_BAR_INDEX;
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

    Flux(const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase1,
         const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase2,
         const double sigma_,
         const double eps_,
         const double mod_grad_alpha1_bar_min_,
         const bool mass_transfer_,
         const double kappa_,
         const double Hmax_,
         const double alpha1d_max_ = 0.5,
         const double lambda_ = 0.9,
         const double tol_Newton_ = 1e-12,
         const std::size_t max_Newton_iters_ = 60); // Constructor which accepts in inputs the equations of state of the two phases

    template<typename State, typename Gradient>
    void perform_Newton_step_relaxation(std::unique_ptr<State> conserved_variables,
                                        const typename Field::value_type H_bar,
                                        typename Field::value_type& dalpha1_bar,
                                        typename Field::value_type& alpha1_bar,
                                        const Gradient& grad_alpha1_bar,
                                        bool& relaxation_applied, const bool mass_transfer_NR); // Perform a Newton step relaxation for a state vector
                                                                                                // (it is not a real space dependent procedure,
                                                                                                // but I would need to be able to do it inside the flux location
                                                                                                // for MUSCL reconstruction)

  protected:
    const LinearizedBarotropicEOS<typename Field::value_type>& phase1;
    const LinearizedBarotropicEOS<typename Field::value_type>& phase2;

    const bool mass_transfer; // Set whether to perform or not mass transfer
    const double kappa; // Size of disperse phase particles
    const double Hmax; // Maximume curvature before atomization

    const double sigma;                   // Surfaace tension parameter
    const double eps;                     // Tolerance of pure phase to set NaNs
    const double mod_grad_alpha1_bar_min; // Tolerance to compute the unit normal

    const double alpha1d_max;           // Maximum admitted small-scale volume fraction
    const double lambda;                // Parameter for bound preserving strategy
    const double tol_Newton;            // Tolerance Newton method relaxation
    const std::size_t max_Newton_iters; // Maximum Newton iterations

    template<typename Gradient>
    FluxValue<cfg> evaluate_continuous_flux(const FluxValue<cfg>& q,
                                            const std::size_t curr_d,
                                            const Gradient& grad_alpha1_bar); // Evaluate the 'continuous' flux for the state q along direction curr_d

    FluxValue<cfg> evaluate_hyperbolic_operator(const FluxValue<cfg>& q,
                                                const std::size_t curr_d); // Evaluate the hyperbolic operator for the state q along direction curr_d

    template<typename Gradient>
    FluxValue<cfg> evaluate_surface_tension_operator(const Gradient& grad_alpha1_bar,
                                                     const std::size_t curr_d); // Evaluate the surface tension operator for the state q along direction curr_d

    FluxValue<cfg> cons2prim(const FluxValue<cfg>& cons) const; // Conversion from conserved to primitive variables

    FluxValue<cfg> prim2cons(const FluxValue<cfg>& prim) const; // Conversion from conserved to primitive variables

    #ifdef ORDER_2
      void perform_reconstruction(const FluxValue<cfg>& primLL,
                                  const FluxValue<cfg>& primL,
                                  const FluxValue<cfg>& primR,
                                  const FluxValue<cfg>& primRR,
                                  FluxValue<cfg>& primL_recon,
                                  FluxValue<cfg>& primR_recon); // Reconstruction for second order scheme

      #ifdef RELAX_RECONSTRUCTION
        template<typename Gradient>
        void relax_reconstruction(FluxValue<cfg>& q,
                                  const typename Field::value_type H_bar,
                                  const Gradient& grad_alpha1_bar); // Relax reconstructed state
      #endif
    #endif
  };

  // Class constructor in order to be able to work with the equation of state
  //
  template<class Field>
  Flux<Field>::Flux(const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase1,
                    const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase2,
                    const double sigma_,
                    const double eps_,
                    const double mod_grad_alpha1_bar_min_,
                    const bool mass_transfer_,
                    const double kappa_,
                    const double Hmax_,
                    const double alpha1d_max_,
                    const double lambda_,
                    const double tol_Newton_,
                    const std::size_t max_Newton_iters_):
    phase1(EOS_phase1), phase2(EOS_phase2),
    mass_transfer(mass_transfer_), kappa(kappa_), Hmax(Hmax_),
    sigma(sigma_), eps(eps_), mod_grad_alpha1_bar_min(mod_grad_alpha1_bar_min_),
    alpha1d_max(alpha1d_max_), lambda(lambda_),
    tol_Newton(tol_Newton_), max_Newton_iters(max_Newton_iters_) {}

  // Evaluate the 'continuous flux'
  //
  template<class Field>
  template<typename Gradient>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::evaluate_continuous_flux(const FluxValue<cfg>& q,
                                                                             const std::size_t curr_d,
                                                                             const Gradient& grad_alpha1_bar) {
    // Sanity check in terms of dimensions
    assert(curr_d < Field::dim);

    // Initialize the resulting variable with the hyperbolic operator
    FluxValue<cfg> res = this->evaluate_hyperbolic_operator(q, curr_d);

    // Add the contribution due to surface tension
    res += this->evaluate_surface_tension_operator(grad_alpha1_bar, curr_d);

    return res;
  }

  // Evaluate the hyperbolic part of the 'continuous' flux
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::evaluate_hyperbolic_operator(const FluxValue<cfg>& q,
                                                                                 const std::size_t curr_d) {
    // Sanity check in terms of dimensions
    assert(curr_d < Field::dim);

    // Initialize the resulting variable
    FluxValue<cfg> res = q;

    // Compute the current velocity
    const auto rho   = q(M1_INDEX) + q(M2_INDEX) + q(M1_D_INDEX);
    const auto vel_d = q(RHO_U_INDEX + curr_d)/rho;

    // Multiply the state the velcoity along the direction of interest
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
    const auto alpha1_bar = q(RHO_ALPHA1_BAR_INDEX)/rho;
    const auto alpha1     = alpha1_bar*(1.0 - q(ALPHA1_D_INDEX));
    const auto rho1       = (alpha1 > eps) ? q(M1_INDEX)/alpha1 : nan("");
    const auto p1         = phase1.pres_value(rho1);

    const auto alpha2     = 1.0 - alpha1 - q(ALPHA1_D_INDEX);
    const auto rho2       = (alpha2 > eps) ? q(M2_INDEX)/alpha2 : nan("");
    const auto p2         = phase2.pres_value(rho2);

    const auto p_bar      = (alpha1 > eps && alpha2 > eps) ?
                            alpha1_bar*p1 + (1.0 - alpha1_bar)*p2 :
                            ((alpha1 < eps) ? p2 : p1);

    res(RHO_U_INDEX + curr_d) += p_bar;

    return res;
  }

  // Evaluate the surface tension operator
  //
  template<class Field>
  template<typename Gradient>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::evaluate_surface_tension_operator(const Gradient& grad_alpha1_bar,
                                                                                      const std::size_t curr_d) {
    // Sanity check in terms of dimensions
    assert(curr_d < Field::dim);

    // Initialize the resulting variable
    FluxValue<cfg> res;

    // Set to zero all the contributions
    res(M1_INDEX) = 0.0;
    res(M2_INDEX) = 0.0;
    res(RHO_ALPHA1_BAR_INDEX) = 0.0;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      res(RHO_U_INDEX + d) = 0.0;
    }

    // Add the contribution due to surface tension
    const auto mod_grad_alpha1_bar = std::sqrt(xt::sum(grad_alpha1_bar*grad_alpha1_bar)());

    if(mod_grad_alpha1_bar > mod_grad_alpha1_bar_min) {
      const auto n = grad_alpha1_bar/mod_grad_alpha1_bar;

      if(curr_d == 0) {
        res(RHO_U_INDEX) += sigma*(n(0)*n(0) - 1.0)*mod_grad_alpha1_bar;
        res(RHO_U_INDEX + 1) += sigma*n(0)*n(1)*mod_grad_alpha1_bar;
      }
      else if(curr_d == 1) {
        res(RHO_U_INDEX) += sigma*n(0)*n(1)*mod_grad_alpha1_bar;
        res(RHO_U_INDEX + 1) += sigma*(n(1)*n(1) - 1.0)*mod_grad_alpha1_bar;
      }
    }

    return res;
  }

  // Conversion from conserved to primitive variables
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::cons2prim(const FluxValue<cfg>& cons) const {
    // Create a copy of the state to save the output
    FluxValue<cfg> prim = cons;

    // Apply conversion only to the large-scale volume fraction
    prim(ALPHA1_BAR_INDEX) = cons(RHO_ALPHA1_BAR_INDEX)/
                             (cons(M1_INDEX) + cons(M2_INDEX) + cons(M1_D_INDEX));

    return prim;
  }

  // Conversion from primitive to conserved variables
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::prim2cons(const FluxValue<cfg>& prim) const {
    // Create a copy of the state to save the output
    FluxValue<cfg> cons = prim;

    // Apply conversion only to the mixture density times volume fraction
    cons(RHO_ALPHA1_BAR_INDEX) = (prim(M1_INDEX) + prim(M2_INDEX) + prim(M1_D_INDEX))*prim(ALPHA1_BAR_INDEX);

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
      const double beta = 1.0; /*--- MINMOD limiter ---*/
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
  template<typename State, typename Gradient>
  void Flux<Field>::perform_Newton_step_relaxation(std::unique_ptr<State> conserved_variables,
                                                   const typename Field::value_type H_bar,
                                                   typename Field::value_type& dalpha1_bar,
                                                   typename Field::value_type& alpha1_bar,
                                                   const Gradient& grad_alpha1_bar,
                                                   bool& relaxation_applied, const bool mass_transfer_NR) {
    // Reinitialization of partial masses in case of evanascent volume fraction
    if(alpha1_bar < eps) {
      (*conserved_variables)(M1_INDEX) = alpha1_bar*phase1.get_rho0();
    }
    if(1.0 - alpha1_bar < eps) {
      (*conserved_variables)(M2_INDEX) = (1.0 - alpha1_bar)*phase2.get_rho0();
    }

    const auto rho = (*conserved_variables)(M1_INDEX)
                   + (*conserved_variables)(M2_INDEX)
                   + (*conserved_variables)(M1_D_INDEX);

    // Update auxiliary values affected by the nonlinear function for which we seek a zero
    const auto alpha1 = alpha1_bar*(1.0 - (*conserved_variables)(ALPHA1_D_INDEX));
    const auto rho1   = (alpha1 > eps) ? (*conserved_variables)(M1_INDEX)/alpha1 : nan("");
    const auto p1     = phase1.pres_value(rho1);

    const auto alpha2 = 1.0 - alpha1 - (*conserved_variables)(ALPHA1_D_INDEX);
    const auto rho2   = (alpha2 > eps) ? (*conserved_variables)(M2_INDEX)/alpha2 : nan("");
    const auto p2     = phase2.pres_value(rho2);

    const auto rho1d  = ((*conserved_variables)(M1_D_INDEX) > eps && (*conserved_variables)(ALPHA1_D_INDEX) > eps) ?
                        (*conserved_variables)(M1_D_INDEX)/(*conserved_variables)(ALPHA1_D_INDEX) : phase1.get_rho0();

    // Prepare for mass transfer if desired
    typename Field::value_type H_lim;
    if(mass_transfer_NR) {
      if(3.0/(kappa*rho1d)*rho1 - (1.0 - alpha1_bar)/(1.0 - (*conserved_variables)(ALPHA1_D_INDEX)) > 0.0 &&
         alpha1_bar > 1e-2 && alpha1_bar < 1e-1 &&
         -grad_alpha1_bar[0]*(*conserved_variables)(RHO_U_INDEX)
         -grad_alpha1_bar[1]*(*conserved_variables)(RHO_U_INDEX + 1) > 0.0 &&
         (*conserved_variables)(ALPHA1_D_INDEX) < alpha1d_max) {
        H_lim = std::min(H_bar, Hmax);
      }
      else {
        H_lim = H_bar;
      }
    }
    else {
      H_lim = H_bar;
    }

    const auto dH = H_bar - H_lim;  //TODO: Initialize this outside and check if the maximum of dH
                                    //at previous iteration is greater than a tolerance (1e-7 in Arthur's code).
                                    //On the other hand, update geoemtry should in principle always be necessary,
                                    //but seems to lead to issues if called at every Newton iteration

    // Compute the nonlinear function for which we seek the zero (basically the Laplace law)
    const auto F = (1.0 - (*conserved_variables)(ALPHA1_D_INDEX))*(p1 - p2)
                 - sigma*H_lim;

    // Perform the relaxation only where really needed
    if(!std::isnan(F) && std::abs(F) > tol_Newton*std::min(phase1.get_p0(), sigma*H_lim) && std::abs(dalpha1_bar) > tol_Newton &&
       alpha1_bar > eps && 1.0 - alpha1_bar > eps) {
      relaxation_applied = true;

      // Compute the derivative w.r.t large scale volume fraction recalling that for a barotropic EOS dp/drho = c^2
      const auto dF_dalpha1_bar = -(*conserved_variables)(M1_INDEX)/(alpha1_bar*alpha1_bar)*
                                   phase1.c_value(rho1)*phase1.c_value(rho1)
                                  -(*conserved_variables)(M2_INDEX)/((1.0 - alpha1_bar)*(1.0 - alpha1_bar))*
                                   phase2.c_value(rho2)*phase2.c_value(rho2);

      // Compute the pseudo time step starting as initial guess from the ideal unmodified Newton method
      auto dtau_ov_epsilon = std::numeric_limits<typename Field::value_type>::infinity();

      /*--- Bound preserving condition for m1, velocity and small-scale volume fraction --*/
      if(dH > 0.0 && !std::isnan(rho1)) {
        /*--- Bound preserving condition for m1 ---*/
        dtau_ov_epsilon = lambda*(alpha1*(1.0 - alpha1_bar))/(sigma*dH);
        if(dtau_ov_epsilon < 0.0) {
          throw std::runtime_error("Negative time step found after relaxation of mass of large-scale phase 1");
        }

        /*--- Bound preserving for the velocity ---*/
        const auto mom_dot_vel = ((*conserved_variables)(RHO_U_INDEX)*(*conserved_variables)(RHO_U_INDEX) +
                                  (*conserved_variables)(RHO_U_INDEX + 1)*(*conserved_variables)(RHO_U_INDEX + 1))/rho;
        const auto fac         = std::max(3.0/(kappa*rho1d)*(rho1/(1.0 - alpha1_bar)) -
                                          1.0/(1.0 - (*conserved_variables)(ALPHA1_D_INDEX)), 0.0);
        if(fac > 0.0) {
          auto dtau_ov_epsilon_tmp = mom_dot_vel/(Hmax*dH*fac*sigma*sigma);
          dtau_ov_epsilon          = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
          if(dtau_ov_epsilon < 0.0) {
            throw std::runtime_error("Negative time step found after relaxation of velocity");
            exit(1);
          }
        }

        /*--- Bound preserving for the small-scale volume fraction ---*/
        auto dtau_ov_epsilon_tmp = lambda*(alpha1d_max - (*conserved_variables)(ALPHA1_D_INDEX))*(1.0 - alpha1_bar)*rho1d/
                                   (rho1*sigma*dH);
        dtau_ov_epsilon          = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
        if((*conserved_variables)(ALPHA1_D_INDEX) > 0.0 && (*conserved_variables)(ALPHA1_D_INDEX) < alpha1d_max) {
          dtau_ov_epsilon_tmp = (*conserved_variables)(ALPHA1_D_INDEX)*(1.0 - alpha1_bar)*rho1d/
                                (rho1*sigma*dH);

          dtau_ov_epsilon     = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
        }
        if(dtau_ov_epsilon < 0.0) {
          throw std::runtime_error("Negative time step found after relaxation of small-scale volume fraction");
        }
      }

      /*--- Bound preserving condition for large-scale volume fraction ---*/
      const auto dF_dalpha1d   = p2 - p1
                               + phase1.c_value(rho1)*phase1.c_value(rho1)*rho1
                               - phase2.c_value(rho2)*phase2.c_value(rho2)*rho2;
      const auto dF_dm1        = phase1.c_value(rho1)*phase1.c_value(rho1)/alpha1_bar;
      const auto R             = dF_dalpha1d/rho1d - dF_dm1;
      const auto a             = rho1*sigma*dH*R/
                                 ((1.0 - alpha1_bar)*(1.0 - (*conserved_variables)(ALPHA1_D_INDEX)));
      /*--- Upper bound ---*/
      auto b                   = (F + lambda*(1.0 - alpha1_bar)*dF_dalpha1_bar)/
                                 (1.0 - (*conserved_variables)(ALPHA1_D_INDEX));
      auto D                   = b*b - 4.0*a*(-lambda*(1.0 - alpha1_bar));
      auto dtau_ov_epsilon_tmp = std::numeric_limits<double>::infinity();
      if(D > 0.0 && (a > 0.0 || (a < 0.0 && b > 0.0))) {
        dtau_ov_epsilon_tmp = 0.5*(-b + std::sqrt(D))/a;
      }
      if(a == 0.0 && b > 0.0) {
        dtau_ov_epsilon_tmp = lambda*(1.0 - alpha1_bar)/b;
      }
      dtau_ov_epsilon = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
      /*--- Lower bound ---*/
      dtau_ov_epsilon_tmp = std::numeric_limits<double>::infinity();
      b                   = (F - lambda*alpha1_bar*dF_dalpha1_bar)/
                            (1.0 - (*conserved_variables)(ALPHA1_D_INDEX));
      D                   = b*b - 4.0*a*(lambda*alpha1_bar);
      if(D > 0.0 && (a < 0.0 || (a > 0.0 && b < 0.0))) {
        dtau_ov_epsilon_tmp = 0.5*(-b - std::sqrt(D))/a;
      }
      if(a == 0.0 && b < 0.0) {
        dtau_ov_epsilon_tmp = -lambda*alpha1_bar/b;
      }
      dtau_ov_epsilon = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
      if(dtau_ov_epsilon < 0.0) {
        throw std::runtime_error("Negative time step found after relaxation of large-scale volume fraction");
        exit(1);
      }

      // Compute the effective variation of the variables
      if(std::isinf(dtau_ov_epsilon)) {
        // If we are in this branch we do not have mass transfer
        // and we do not have other restrictions on the bounds of large scale volume fraction
        dalpha1_bar = -F/dF_dalpha1_bar;
      }
      else {
        const auto dm1 = -dtau_ov_epsilon/(1.0 - alpha1_bar)*
                          ((*conserved_variables)(M1_INDEX)/(alpha1_bar*(1.0 - (*conserved_variables)(ALPHA1_D_INDEX))))*
                          sigma*dH;

        const auto num_dalpha1_bar = dtau_ov_epsilon/(1.0 - (*conserved_variables)(ALPHA1_D_INDEX));
        const auto den_dalpha1_bar = 1.0 - num_dalpha1_bar*dF_dalpha1_bar;
        dalpha1_bar                = (num_dalpha1_bar/den_dalpha1_bar)*(F - dm1*R);

        if(dm1 > 0.0) {
          throw std::runtime_error("Negative sign of mass transfer inside Newton step");
        }
        else {
          (*conserved_variables)(M1_INDEX) += dm1;
          if((*conserved_variables)(M1_INDEX) < 0.0) {
            throw std::runtime_error("Negative mass of large-scale phase 1 inside Newton step");
          }

          (*conserved_variables)(M1_D_INDEX) -= dm1;
          if((*conserved_variables)(M1_D_INDEX) < 0.0) {
            throw std::runtime_error("Negative mass of small-scale phase 1 inside Newton step");
          }
        }

        if((*conserved_variables)(ALPHA1_D_INDEX) - dm1/rho1d > 1.0) {
          throw std::runtime_error("Exceeding value for small-scale volume fraction inside Newton step");
        }
        else {
          (*conserved_variables)(ALPHA1_D_INDEX) -= dm1/rho1d;
        }

        (*conserved_variables)(SIGMA_D_INDEX) -= dm1*3.0*Hmax/(kappa*rho1d);
      }

      if(alpha1_bar + dalpha1_bar < 0.0 || alpha1_bar + dalpha1_bar > 1.0) {
        throw std::runtime_error("Bounds exceeding value for large-scale volume fraction inside Newton step");
      }
      else {
        alpha1_bar += dalpha1_bar;
      }

      if(dH > 0.0) {
        const auto fac = std::max(3.0/(kappa*rho1d)*(rho1/(1.0 - alpha1_bar)) -
                                  1.0/(1.0 - (*conserved_variables)(ALPHA1_D_INDEX)), 0.0);

        double drho_fac = 0.0;
        const auto mom_squared = (*conserved_variables)(RHO_U_INDEX)*(*conserved_variables)(RHO_U_INDEX)
                               + (*conserved_variables)(RHO_U_INDEX + 1)*(*conserved_variables)(RHO_U_INDEX + 1);
        if(mom_squared > 0.0) {
           drho_fac = dtau_ov_epsilon*
                      sigma*sigma*dH*fac*H_lim*rho/mom_squared;
        }

        for(std::size_t d = 0; d < Field::dim; ++d) {
          (*conserved_variables)(RHO_U_INDEX + d) -= drho_fac*(*conserved_variables)(RHO_U_INDEX + d);
        }
      }
    }

    // Update "conservative counter part" of large-scale volume fraction.
    // Do it outside because this can change either because of relaxation of
    // alpha1_bar or because of change of rho for evanescent volume fractions.
    (*conserved_variables)(RHO_ALPHA1_BAR_INDEX) = rho*alpha1_bar;
  }

  // Relax reconstruction
  //
  #ifdef ORDER_2
    #ifdef RELAX_RECONSTRUCTION
      template<class Field>
      template<typename Gradient>
      void Flux<Field>::relax_reconstruction(FluxValue<cfg>& q,
                                             const typename Field::value_type H_bar,
                                             const Gradient& grad_alpha1_bar) {
        // Declare and set relevant parameters
        std::size_t Newton_iter = 0;
        bool relaxation_applied = true;
        bool mass_transfer_NR   = false;

        typename Field::value_type dalpha1_bar = std::numeric_limits<typename Field::value_type>::infinity();
        typename Field::value_type alpha1_bar  = q(RHO_ALPHA1_BAR_INDEX)/
                                                 (q(M1_INDEX) + q(M2_INDEX) + q(M1_D_INDEX));

        // Apply Newton method
        while(relaxation_applied == true) {
          relaxation_applied = false;
          Newton_iter++;

          this->perform_Newton_step_relaxation(std::make_unique<FluxValue<cfg>>(q),
                                               H_bar, dalpha1_bar, alpha1_bar, grad_alpha1_bar,
                                               relaxation_applied, mass_transfer_NR);

          // Newton cycle diverged
          if(Newton_iter > max_Newton_iters) {
            std::cerr << "Netwon method not converged in the relaxation after MUSCL" << std::endl;
            exit(1);
          }
        }
      }
    #endif
  #endif

} // end namespace samurai

#endif
