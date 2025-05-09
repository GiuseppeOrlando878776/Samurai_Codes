// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#ifndef flux_base_hpp
#define flux_base_hpp

#pragma once
#include <samurai/schemes/fv.hpp>

#include "barotropic_eos.hpp"

/*--- Preprocessor to define whether order 2 is desired ---*/
#define ORDER_2

/*--- Preprocessor to define whether relaxation is desired after reconstruction for order 2 ---*/
#ifdef ORDER_2
  //#define RELAX_RECONSTRUCTION
#endif

/**
 * Useful parameters and enumerators
 */
namespace EquationData {
  /*--- Declare spatial dimension ---*/
  static constexpr std::size_t dim = 2;

  /*--- Use auxiliary variables for the indices for the sake of generality ---*/
  static constexpr std::size_t M1_INDEX         = 0;
  static constexpr std::size_t M2_INDEX         = 1;
  static constexpr std::size_t M1_D_INDEX       = 2;
  static constexpr std::size_t RHO_Z_INDEX      = 3;
  static constexpr std::size_t RHO_ALPHA1_INDEX = 4;
  static constexpr std::size_t RHO_U_INDEX      = 5;

  /*--- Save also the total number of (scalar) variables ---*/
  static constexpr std::size_t NVARS = 5 + dim;

  /*--- Use auxiliary variables for the indices also for primitive variables for the sake of generality ---*/
  static constexpr std::size_t ALPHA1_INDEX            = RHO_ALPHA1_INDEX;
  static constexpr std::size_t P1_INDEX                = M1_INDEX;
  static constexpr std::size_t P2_INDEX                = M2_INDEX;
  static constexpr std::size_t ALPHA1_D_2_INDEX        = M1_D_INDEX;
  static constexpr std::size_t SIGMA_OV_ALPHA1_D_INDEX = RHO_Z_INDEX;
  static constexpr std::size_t U_INDEX                 = RHO_U_INDEX;
  /*static constexpr std::size_t P1_D_INDEX   = M1_D_INDEX;
  static constexpr std::size_t Z_INDEX      = RHO_Z_INDEX;*/

  template<typename Field>
  typename Field::value_type compute_rho1_d_local_Laplace(const typename Field::value_type m1_d,
                                                          const typename Field::value_type m2,
                                                          const typename Field::value_type m1,
                                                          const typename Field::value_type alpha1,
                                                          const typename Field::value_type rho_z,
                                                          const double sigma,
                                                          const BarotropicEOS<typename Field::value_type>& EOS_phase1,
                                                          const BarotropicEOS<typename Field::value_type>& EOS_phase2,
                                                          const double atol_Newton = 1e-14, const double rtol_Newton = 1e-12,
                                                          const std::size_t max_Newton_iters = 60, const double lambda = 0.9) {
    auto rho1_d = m1/alpha1; /*--- TODO: Add a check in case of zero volume fraction ---*/
    if(rho1_d < 0.0) {
      throw std::runtime_error("Negative initial guess to compute small-scale Laplace law");
    }

    /*--- Compute the nonlinear function for which we seek the zero (basically the Laplace law) ---*/
    auto p1_d       = EOS_phase1.pres_value(rho1_d);
    auto p2         = EOS_phase2.pres_value(m2/(1.0 - alpha1 - m1_d/rho1_d));
    auto F          = m1_d*(p1_d - p2) - 2.0/3.0*sigma*rho_z*std::pow(rho1_d, 1.0/3.0);
    const auto F_0  = F;
    auto drho1_d    = atol_Newton + 1.0;

    std::size_t iter = 0;
    for(iter = 0; iter < max_Newton_iters; ++iter) {
      if(std::abs(F) > atol_Newton + rtol_Newton*std::abs(F_0) && std::abs(drho1_d) > atol_Newton) {
        // Compute the derivative w.r.t rho1_d
        const auto rho2       = m2/(1.0 - alpha1 - m1_d/rho1_d); /*--- TODO: Add a check in case of zero volume fraction ---*/
        const auto c1_d       = EOS_phase1.c_value(rho1_d);
        const auto c2         = EOS_phase2.c_value(rho2);
        const auto dF_drho1_d = m1_d*(c1_d*c1_d + c2*c2*rho2*rho2*m1/(m2*rho1_d*rho1_d))
                              - 2.0/9.0*sigma*rho_z*std::pow(rho1_d, -2.0/3.0); /*--- TODO: Add a check in case of zero volume fraction ---*/

        // Compute the rho1_d update
        drho1_d = -F/dF_drho1_d;
        if(drho1_d < 0.0) {
          drho1_d = std::max(drho1_d, -lambda*rho1_d);
        }

        rho1_d += drho1_d;

        p1_d = EOS_phase1.pres_value(rho1_d);
        p2   = EOS_phase2.pres_value(m2/(1.0 - alpha1 - m1_d/rho1_d));
        F    = m1_d*(p1_d - p2) - 2.0/3.0*sigma*rho_z*std::pow(rho1_d, 1.0/3.0);
      }
      else {
        break;
      }
    }
    if(iter == max_Newton_iters) {
      throw std::runtime_error("Non-converging Newton method to compute small-scale Laplace law");
    }

    return rho1_d;
  }
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

    using cfg = FluxConfig<SchemeType::NonLinear, output_field_size, stencil_size, Field>;

    Flux(const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase1_,
         const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase2_,
         const double sigma_,
         const double mod_grad_alpha1_min_,
         const double lambda_ = 0.9,
         const double atol_Newton_ = 1e-14,
         const double rtol_Newton_ = 1e-12,
         const std::size_t max_Newton_iters_ = 60); /*--- Constructor which accepts in input the equations of state of the two phases ---*/

  protected:
    const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase1;
    const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase2;

    const double sigma; /*--- Surface tension parameter ---*/

    const double mod_grad_alpha1_min; /*--- Tolerance to compute the unit normal ---*/

    const double      lambda;           /*--- Parameter for bound preserving strategy ---*/
    const double      atol_Newton;      /*--- Absolute tolerance Newton method relaxation ---*/
    const double      rtol_Newton;      /*--- Relative tolerance Newton method relaxation ---*/
    const std::size_t max_Newton_iters; /*--- Maximum number of Newton iterations ---*/

    template<typename Gradient>
    FluxValue<cfg> evaluate_continuous_flux(const FluxValue<cfg>& q,
                                            const std::size_t curr_d,
                                            const Gradient& grad_alpha1); /*--- Evaluate the 'continuous' flux for the state q along direction curr_d ---*/

    FluxValue<cfg> evaluate_hyperbolic_operator(const FluxValue<cfg>& q,
                                                const std::size_t curr_d); /*--- Evaluate the hyperbolic operator for the state q along direction curr_d --*/

    template<typename Gradient>
    FluxValue<cfg> evaluate_surface_tension_operator(const Gradient& grad_alpha1,
                                                     const std::size_t curr_d); /*--- Evaluate the surface tension operator for the state q along direction curr_d ---*/

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
        void perform_Newton_step_relaxation(std::unique_ptr<State> conserved_variables,
                                            const typename Field::value_type H,
                                            typename Field::value_type& dalpha1,
                                            typename Field::value_type& alpha1,
                                            bool& relaxation_applied); /*--- Perform a Newton step relaxation for a state vector
                                                                             (it is not a real space dependent procedure,
                                                                              but I would need to be able to do it inside the flux location
                                                                              for MUSCL reconstruction) ---*/

        void relax_reconstruction(FluxValue<cfg>& q,
                                  const typename Field::value_type H); /*--- Relax reconstructed state ---*/
      #endif
    #endif
  };

  // Class constructor in order to be able to work with the equation of state
  //
  template<class Field>
  Flux<Field>::Flux(const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase1_,
                    const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase2_,
                    const double sigma_,
                    const double mod_grad_alpha1_min_,
                    const double lambda_,
                    const double atol_Newton_,
                    const double rtol_Newton_,
                    const std::size_t max_Newton_iters_):
    EOS_phase1(EOS_phase1_), EOS_phase2(EOS_phase2_),
    sigma(sigma_), mod_grad_alpha1_min(mod_grad_alpha1_min_),
    lambda(lambda_), atol_Newton(atol_Newton_), rtol_Newton(rtol_Newton_),
    max_Newton_iters(max_Newton_iters_) {}

  // Evaluate the 'continuous flux'
  //
  template<class Field>
  template<typename Gradient>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::evaluate_continuous_flux(const FluxValue<cfg>& q,
                                                                             const std::size_t curr_d,
                                                                             const Gradient& grad_alpha1) {
    /*--- Initialize the resulting variable with the hyperbolic operator ---*/
    FluxValue<cfg> res = this->evaluate_hyperbolic_operator(q, curr_d);

    /*--- Add the contribution due to surface tension ---*/
    res += this->evaluate_surface_tension_operator(grad_alpha1, curr_d);

    return res;
  }

  // Evaluate the hyperbolic part of the 'continuous' flux
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::evaluate_hyperbolic_operator(const FluxValue<cfg>& q,
                                                                                 const std::size_t curr_d) {
    /*--- Sanity check in terms of dimensions ---*/
    assert(curr_d < Field::dim);

    /*--- Initialize the resulting variable ---*/
    FluxValue<cfg> res = q;

    /*--- Compute the current velocity ---*/
    const auto rho   = q(M1_INDEX) + q(M2_INDEX) + q(M1_D_INDEX);
    const auto vel_d = q(RHO_U_INDEX + curr_d)/rho;

    /*--- Multiply the state the velcoity along the direction of interest ---*/
    res(M1_INDEX) *= vel_d;
    res(M2_INDEX) *= vel_d;
    res(M1_D_INDEX) *= vel_d;
    res(RHO_Z_INDEX) *= vel_d;
    res(RHO_ALPHA1_INDEX) *= vel_d;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      res(RHO_U_INDEX + d) *= vel_d;
    }

    /*--- Compute and add the contribution due to the pressure ---*/
    const auto alpha1   = q(RHO_ALPHA1_INDEX)/rho;
    const auto rho1     = q(M1_INDEX)/alpha1; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto p1       = EOS_phase1.pres_value(rho1);

    typename Field::value_type rho1_d;
    try {
      rho1_d = compute_rho1_d_local_Laplace<Field>(q(M1_D_INDEX), q(M2_INDEX), q(M1_INDEX), alpha1, q(RHO_Z_INDEX),
                                                   sigma, EOS_phase1, EOS_phase2, atol_Newton, rtol_Newton, max_Newton_iters, lambda);
    }
    catch(const std::exception& e) {
      std::cerr << "Small-scale error when evaluating operator" << std::endl;
      std::cout << q << std::endl;
      std::cerr << e.what() << std::endl;
      exit(1);
    }
    const auto alpha1_d = q(M1_D_INDEX)/rho1_d;
    const auto alpha2   = 1.0 - alpha1 - alpha1_d;
    const auto rho2     = q(M2_INDEX)/alpha2; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto p2       = EOS_phase2.pres_value(rho2);

    const auto p        = alpha1*p1 + (1.0 - alpha1)*p2;

    res(RHO_U_INDEX + curr_d) += p;

    return res;
  }

  // Evaluate the surface tension operator
  //
  template<class Field>
  template<typename Gradient>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::evaluate_surface_tension_operator(const Gradient& grad_alpha1,
                                                                                      const std::size_t curr_d) {
    /*--- Sanity check in terms of dimensions ---*/
    assert(curr_d < Field::dim);

    /*--- Initialize the resulting variable ---*/
    FluxValue<cfg> res;

    // Set to zero all the contributions
    res(M1_INDEX) = 0.0;
    res(M2_INDEX) = 0.0;
    res(RHO_ALPHA1_INDEX) = 0.0;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      res(RHO_U_INDEX + d) = 0.0;
    }
    res(M1_D_INDEX) = 0.0;
    res(RHO_Z_INDEX) = 0.0;

    /*--- Add the contribution due to surface tension ---*/
    const auto mod_grad_alpha1 = std::sqrt(xt::sum(grad_alpha1*grad_alpha1)());

    if(mod_grad_alpha1 > mod_grad_alpha1_min) {
      const auto n = grad_alpha1/mod_grad_alpha1;

      if(curr_d == 0) {
        res(RHO_U_INDEX) += sigma*(n(0)*n(0) - 1.0)*mod_grad_alpha1;
        res(RHO_U_INDEX + 1) += sigma*n(0)*n(1)*mod_grad_alpha1;
      }
      else if(curr_d == 1) {
        res(RHO_U_INDEX) += sigma*n(0)*n(1)*mod_grad_alpha1;
        res(RHO_U_INDEX + 1) += sigma*(n(1)*n(1) - 1.0)*mod_grad_alpha1;
      }
    }

    return res;
  }

  // Conversion from conserved to primitive variables
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::cons2prim(const FluxValue<cfg>& cons) const {
    FluxValue<cfg> prim;

    const auto rho     = cons(M1_INDEX) + cons(M2_INDEX) + cons(M1_D_INDEX);
    prim(ALPHA1_INDEX) = cons(RHO_ALPHA1_INDEX)/rho;
    prim(P1_INDEX)     = EOS_phase1.pres_value(cons(M1_INDEX)/prim(ALPHA1_INDEX)); /*--- TODO: Add a check in case of zero volume fraction ---*/
    typename Field::value_type rho1_d;
    try {
      rho1_d = compute_rho1_d_local_Laplace<Field>(cons(M1_D_INDEX), cons(M2_INDEX), cons(M1_INDEX), prim(ALPHA1_INDEX), cons(RHO_Z_INDEX),
                                                   sigma, EOS_phase1, EOS_phase2, atol_Newton, rtol_Newton, max_Newton_iters, lambda);
    }
    catch(const std::exception& e) {
      std::cerr << "Small-scale error in cons2prim" << std::endl;
      std::cout << cons << std::endl;
      std::cerr << e.what() << std::endl;
      exit(1);
    }
    if(rho1_d < 0.0) {
      std::cerr << "Negative rho1_d in cons2prim" << std::endl;
    }
    const auto alpha1_d = cons(M1_D_INDEX)/rho1_d;
    prim(P2_INDEX)      = EOS_phase2.pres_value(cons(M2_INDEX)/(1.0 - prim(ALPHA1_INDEX) - alpha1_d));
                         /*--- TODO: Add a check in case of zero volume fraction ---*/
    for(std::size_t d = 0; d < Field::dim; ++d) {
      prim(U_INDEX + d) = cons(RHO_U_INDEX + d)/rho;
    }
    /*prim(P1_D_INDEX) = EOS_phase1.pres_value(rho1_d);
      prim(Z_INDEX)    = cons(RHO_Z_INDEX)/rho;*/
    prim(ALPHA1_D_2_INDEX) = alpha1_d/(1.0 - prim(ALPHA1_INDEX));
    const auto p1_d = EOS_phase1.pres_value(rho1_d);
    prim(SIGMA_OV_ALPHA1_D_INDEX) = 1.5*(p1_d - prim(P2_INDEX))/sigma;

    if(rho1_d < 0.0) {
      std::cerr << "Negative rho1_d in cons2prim" << std::endl;
    }
    if(prim(SIGMA_OV_ALPHA1_D_INDEX) < -1e-12) {
      std::cerr << "Negative \Delta p = p1_d - p2 in cons2prim = " << p1_d - prim(P2_INDEX) << std::endl;
    }

    /*if(std::isnan(prim(Z_INDEX))) {
      std::cerr << "NaN in the cons2prim Z" << std::endl;
      std::cerr << "rho z = " << cons(RHO_Z_INDEX) << std::endl;
      std::cerr << "rho = " << rho << std::endl;
      throw std::runtime_error("NaN in the cons2prim");
    }*/

    return prim;
  }

  // Conversion from primitive to conserved variables
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::prim2cons(const FluxValue<cfg>& prim) const {
    FluxValue<cfg> cons;

    cons(M1_INDEX) = prim(ALPHA1_INDEX)*EOS_phase1.rho_value(prim(P1_INDEX));

    /*--- Update alpha1_d thanks to local Laplace law ---*/
    /*const auto rho1_d      = EOS_phase1.rho_value(prim(P1_D_INDEX));
    const auto rho2        = EOS_phase2.rho_value(prim(P2_INDEX));
    const auto Delta_p     = prim(P1_D_INDEX) - prim(P2_INDEX);
    const auto alpha1_d    = 2.0*sigma*prim(Z_INDEX)*
                             (cons(M1_INDEX) + (1.0 - prim(ALPHA1_INDEX))*rho2)/
                             (std::abs(3.0*std::pow(rho1_d, 2.0/3.0)*Delta_p + 2.0*sigma*prim(Z_INDEX)*(rho2 - rho1_d)) + 1e-13);*/
                             /*--- Added tolerance because division by zero occurs when alpha1_d = 0 (\Delta p = 0, z = 0) ---*/
    /*if(prim(ALPHA1_D_INDEX) < 0.0) {
      std::cerr << "Negative alpha1_d in prim2cons" << std::endl;
    }
    if(rho1_d < 0.0) {
      std::cerr << "Negative rho1_d in prim2cons" << std::endl;
    }
    if(Delta_p < 0.0) {
      std::cerr << "Negative \Delta p = p1_d - p2 in prim2cons" << std::endl;
    }*/
    const auto rho2        = EOS_phase2.rho_value(prim(P2_INDEX));
    const auto alpha1_d    = (1.0 - prim(ALPHA1_INDEX))*prim(ALPHA1_D_2_INDEX);
    cons(M2_INDEX)         = (1.0 - prim(ALPHA1_INDEX) - alpha1_d)*rho2;
    const auto p1_d        = prim(P2_INDEX) + 2.0/3.0*sigma*prim(SIGMA_OV_ALPHA1_D_INDEX);
    const auto rho1_d      = EOS_phase1.rho_value(p1_d);
    cons(M1_D_INDEX)       = alpha1_d*rho1_d;
    const auto rho         = cons(M1_INDEX) + cons(M2_INDEX) + cons(M1_D_INDEX);
    cons(RHO_ALPHA1_INDEX) = rho*prim(ALPHA1_INDEX);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      cons(RHO_U_INDEX + d) = rho*prim(U_INDEX + d);
    }
    if(prim(SIGMA_OV_ALPHA1_D_INDEX) < 0.0) {
      std::cerr << "Negative Sigma_d/alpha1_d in cons2prim" << std::endl;
    }
    if(rho1_d < 0.0) {
      std::cerr << "Negative rho1_d in cons2prim" << std::endl;
    }
    if(alpha1_d < 0.0) {
      std::cerr << "Negative alpha1_d in prim2cons" << std::endl;
    }
    cons(RHO_Z_INDEX) = std::pow(rho1_d, 2.0/3.0)*prim(SIGMA_OV_ALPHA1_D_INDEX)*alpha1_d;

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
      const double beta = 1.0; // MINMOD limiter
      for(std::size_t comp = 0; comp < Field::n_comp; ++comp) {
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

  // Relax reconstruction
  //
  #ifdef ORDER_2
    #ifdef RELAX_RECONSTRUCTION
      // Perform a Newton step relaxation for a single vector state (i.e. a single cell) without mass transfer
      //
      template<class Field>
      template<typename State>
      void Flux<Field>::perform_Newton_step_relaxation(std::unique_ptr<State> conserved_variables,
                                                       const typename Field::value_type H,
                                                       typename Field::value_type& dalpha1,
                                                       typename Field::value_type& alpha1,
                                                       bool& relaxation_applied) {
        if(!std::isnan(H)) {
          /*--- Update auxiliary values affected by the nonlinear function for which we seek a zero ---*/
          const auto rho1 = (*conserved_variables)(M1_INDEX)/alpha1; /*--- TODO: Add a check in case of zero volume fraction ---*/
          const auto p1   = EOS_phase1.pres_value(rho1);

          typename Field::value_type rho1_d;
          try {
            rho1_d = compute_rho1_d_local_Laplace<Field>((*conserved_variables)(M1_D_INDEX),
                                                         (*conserved_variables)(M2_INDEX),
                                                         (*conserved_variables)(M1_INDEX), alpha1, (*conserved_variables)(RHO_Z_INDEX),
                                                         sigma, EOS_phase1, EOS_phase2, atol_Newton, rtol_Newton, max_Newton_iters, lambda);
          }
          catch(const std::exception& e) {
            std::cerr << e.what() << std::endl;
            exit(1);
          }
          const auto alpha1_d = (*conserved_variables)(M1_D_INDEX)/rho1_d;
          const auto alpha2   = 1.0 - alpha1 - alpha1_d;
          const auto rho2     = (*conserved_variables)(M2_INDEX)/alpha2; /*--- TODO: Add a check in case of zero volume fraction ---*/
          const auto p2       = EOS_phase2.pres_value(rho2);

          /*--- Compute the nonlinear function for which we seek the zero (basically the Laplace law) ---*/
          const auto F = (p1 - p2) - sigma*H;

          /*--- Perform the relaxation only where really needed ---*/
          if(std::abs(F) > atol_Newton + rtol_Newton*std::min(EOS_phase1.get_p0(), sigma*std::abs(H)) &&
             std::abs(dalpha1) > atol_Newton) {
            relaxation_applied = true;

            // Compute the derivative w.r.t large-scale volume fraction recalling that for a barotropic EOS dp/drho = c^2
            const auto dF_dalpha1 = -(*conserved_variables)(M1_INDEX)/(alpha1*alpha1)*
                                     EOS_phase1.c_value(rho1)*EOS_phase1.c_value(rho1)
                                    -(*conserved_variables)(M2_INDEX)/(alpha2*alpha2)*
                                     EOS_phase2.c_value(rho2)*EOS_phase2.c_value(rho2);

            // Compute the large-scale volume fraction update
            dalpha1 = -F/dF_dalpha1;
            if(dalpha1 > 0.0) {
              dalpha1 = std::min(dalpha1, lambda*(1.0 - alpha1));
            }
            else if(dalpha1 < 0.0) {
              dalpha1 = std::max(dalpha1, -lambda*alpha1);
            }

            if(alpha1 + dalpha1 < 0.0 || alpha1 + dalpha1 > 1.0) {
              throw std::runtime_error("Bounds exceeding value for large-scale liquid volume fraction inside Newton step of reconstruction");
            }
            else {
              alpha1 += dalpha1;
            }
          }

          /*--- Update the vector of conserved variables (probably not the optimal choice since I need this update only at the end of the Newton loop,
                but the most coherent one thinking about the transfer of mass) ---*/
          const auto rho = (*conserved_variables)(M1_INDEX)
                         + (*conserved_variables)(M2_INDEX)
                         + (*conserved_variables)(M1_D_INDEX);
          (*conserved_variables)(RHO_ALPHA1_INDEX) = rho*alpha1;
          /*--- TODO: This approach is not compatible with small-scale Laplace law, to be reanalyzed.... ----*/
        }
      }

      // Relax the reconstruction
      //
      template<class Field>
      void Flux<Field>::relax_reconstruction(FluxValue<cfg>& q,
                                             const typename Field::value_type H) {
        /*--- Declare and set relevant parameters ---*/
        std::size_t Newton_iter = 0;
        bool relaxation_applied = true;

        typename Field::value_type dalpha1 = std::numeric_limits<typename Field::value_type>::infinity();
        typename Field::value_type alpha1  = q(RHO_ALPHA1_INDEX)/
                                             (q(M1_INDEX) + q(M2_INDEX) + q(M1_D_INDEX));

        /*--- Apply Newton method ---*/
        while(relaxation_applied == true) {
          relaxation_applied = false;
          Newton_iter++;

          try {
            this->perform_Newton_step_relaxation(std::make_unique<FluxValue<cfg>>(q),
                                                 H, dalpha1, alpha1, relaxation_applied);
          }
          catch(std::exception& e) {
            std::cerr << e.what() << std::endl;
            exit(1);
          }

          // Newton cycle diverged
          if(Newton_iter > max_Newton_iters && relaxation_applied == true) {
            std::cerr << "Netwon method not converged in the relaxation after MUSCL" << std::endl;
            exit(1);
          }
        }
      }
    #endif
  #endif

} // end namespace samurai

#endif
