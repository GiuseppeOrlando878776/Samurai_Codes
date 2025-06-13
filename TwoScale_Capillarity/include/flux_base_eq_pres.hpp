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
  #define RELAX_RECONSTRUCTION
#endif

/**
 * Useful parameters and enumerators
 */
namespace EquationData {
  /*--- Declare spatial dimension ---*/
  static constexpr std::size_t dim = 2;

  /*--- Use auxiliary variables for the indices for the sake of generality ---*/
  static constexpr std::size_t M1_INDEX             = 0;
  static constexpr std::size_t M2_INDEX             = 1;
  static constexpr std::size_t M1_D_INDEX           = 2;
  static constexpr std::size_t ALPHA1_D_INDEX       = 3;
  static constexpr std::size_t SIGMA_D_INDEX        = 4;
  static constexpr std::size_t RHO_ALPHA1_BAR_INDEX = 5;
  static constexpr std::size_t RHO_U_INDEX          = 6;

  /*--- Save also the total number of (scalar) variables ---*/
  static constexpr std::size_t NVARS = 6 + dim;

  /*--- Use auxiliary variables for the indices also for primitive variables for the sake of generality ---*/
  static constexpr std::size_t ALPHA1_BAR_INDEX = RHO_ALPHA1_BAR_INDEX;
  static constexpr std::size_t P1_INDEX         = M1_INDEX;
  static constexpr std::size_t P2_INDEX         = M2_INDEX;
  static constexpr std::size_t U_INDEX          = RHO_U_INDEX;

  /*--- Use auxiliary indices also to distinguish bewteen the type of relaxation ---*/
  static constexpr std::size_t PRESSURE_EQUILIBRIUM = 0;
  static constexpr std::size_t LOCAL_LAPLACE        = 1;
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
         const double mod_grad_alpha1_bar_min_,
         const double lambda_ = 0.9,
         const double atol_Newton_ = 1e-14,
         const double rtol_Newton_ = 1e-12,
         const std::size_t max_Newton_iters_ = 60); /*--- Constructor which accepts in input the equations of state of the two phases ---*/

  protected:
    const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase1;
    const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase2;

    const double sigma; /*--- Surface tension parameter ---*/

    const double mod_grad_alpha1_bar_min; /*--- Tolerance to compute the unit normal ---*/

    const double      lambda;           /*--- Parameter for bound preserving strategy ---*/
    const double      atol_Newton;      /*--- Absolute tolerance Newton method relaxation ---*/
    const double      rtol_Newton;      /*--- Relative tolerance Newton method relaxation ---*/
    const std::size_t max_Newton_iters; /*--- Maximum number of Newton iterations ---*/

    template<typename Gradient>
    FluxValue<cfg> evaluate_continuous_flux(const FluxValue<cfg>& q,
                                            const std::size_t curr_d,
                                            const Gradient& grad_alpha1_bar); /*--- Evaluate the 'continuous' flux for the state q along direction curr_d ---*/

    FluxValue<cfg> evaluate_hyperbolic_operator(const FluxValue<cfg>& q,
                                                const std::size_t curr_d); /*--- Evaluate the hyperbolic operator for the state q along direction curr_d --*/

    template<typename Gradient>
    FluxValue<cfg> evaluate_surface_tension_operator(const Gradient& grad_alpha1_bar,
                                                     const std::size_t curr_d); /*--- Evaluate the surface tension operator
                                                                                      for the state q along direction curr_d ---*/

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
                                            const typename Field::value_type H_bar,
                                            typename Field::value_type& dalpha1_bar,
                                            typename Field::value_type& alpha1_bar,
                                            bool& relaxation_applied); /*--- Perform a Newton step relaxation for a state vector
                                                                             (it is not a real space dependent procedure,
                                                                              but I would need to be able to do it inside the flux location
                                                                              for MUSCL reconstruction) ---*/

        void relax_reconstruction(FluxValue<cfg>& q,
                                  const typename Field::value_type H_bar); /*--- Relax reconstructed state ---*/
      #endif
    #endif
  };

  // Class constructor in order to be able to work with the equation of state
  //
  template<class Field>
  Flux<Field>::Flux(const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase1_,
                    const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase2_,
                    const double sigma_,
                    const double mod_grad_alpha1_bar_min_,
                    const double lambda_,
                    const double atol_Newton_,
                    const double rtol_Newton_,
                    const std::size_t max_Newton_iters_):
    EOS_phase1(EOS_phase1_), EOS_phase2(EOS_phase2_),
    sigma(sigma_), mod_grad_alpha1_bar_min(mod_grad_alpha1_bar_min_),
    lambda(lambda_), atol_Newton(atol_Newton_), rtol_Newton(rtol_Newton_),
    max_Newton_iters(max_Newton_iters_) {}

  // Evaluate the 'continuous flux'
  //
  template<class Field>
  template<typename Gradient>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::evaluate_continuous_flux(const FluxValue<cfg>& q,
                                                                             const std::size_t curr_d,
                                                                             const Gradient& grad_alpha1_bar) {
    /*--- Initialize the resulting variable with the hyperbolic operator ---*/
    FluxValue<cfg> res = this->evaluate_hyperbolic_operator(q, curr_d);

    /*--- Add the contribution due to surface tension ---*/
    res += this->evaluate_surface_tension_operator(grad_alpha1_bar, curr_d);

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
    res(ALPHA1_D_INDEX) *= vel_d;
    res(SIGMA_D_INDEX) *= vel_d;
    res(RHO_ALPHA1_BAR_INDEX) *= vel_d;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      res(RHO_U_INDEX + d) *= vel_d;
    }

    /*--- Compute and add the contribution due to the pressure ---*/
    const auto alpha1_bar = q(RHO_ALPHA1_BAR_INDEX)/rho;
    const auto alpha1     = alpha1_bar*(1.0 - q(ALPHA1_D_INDEX));
    const auto rho1       = q(M1_INDEX)/alpha1; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto p1         = EOS_phase1.pres_value(rho1);

    const auto alpha2     = 1.0 - alpha1 - q(ALPHA1_D_INDEX);
    const auto rho2       = q(M2_INDEX)/alpha2; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto p2         = EOS_phase2.pres_value(rho2);

    const auto p_bar      = alpha1_bar*p1 + (1.0 - alpha1_bar)*p2;

    res(RHO_U_INDEX + curr_d) += p_bar;

    return res;
  }

  // Evaluate the surface tension operator
  //
  template<class Field>
  template<typename Gradient>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::evaluate_surface_tension_operator(const Gradient& grad_alpha1_bar,
                                                                                      const std::size_t curr_d) {
    /*--- Sanity check in terms of dimensions ---*/
    assert(curr_d < Field::dim);

    /*--- Initialize the resulting variable ---*/
    FluxValue<cfg> res;

    // Set to zero all the contributions
    res(M1_INDEX) = 0.0;
    res(M2_INDEX) = 0.0;
    res(RHO_ALPHA1_BAR_INDEX) = 0.0;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      res(RHO_U_INDEX + d) = 0.0;
    }
    res(M1_D_INDEX) = 0.0;
    res(ALPHA1_D_INDEX) = 0.0;
    res(SIGMA_D_INDEX) = 0.0;

    /*--- Add the contribution due to surface tension ---*/
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
    FluxValue<cfg> prim;

    prim(ALPHA1_BAR_INDEX) = cons(RHO_ALPHA1_BAR_INDEX)/
                             (cons(M1_INDEX) + cons(M2_INDEX) + cons(M1_D_INDEX));
    const auto alpha1      = prim(ALPHA1_BAR_INDEX)*(1.0 - cons(ALPHA1_D_INDEX));
    prim(P1_INDEX)         = EOS_phase1.pres_value(cons(M1_INDEX)/alpha1); /*--- TODO: Add a check in case of zero volume fraction ---*/
    prim(P2_INDEX)         = EOS_phase2.pres_value(cons(M2_INDEX)/(1.0 - alpha1 - cons(ALPHA1_D_INDEX))); /*--- TODO: Add a check in case of zero volume fraction ---*/
    for(std::size_t d = 0; d < Field::dim; ++d) {
      prim(U_INDEX + d) = cons(RHO_U_INDEX + d)/
                          (cons(M1_INDEX) + cons(M2_INDEX) + cons(M1_D_INDEX));
    }
    prim(M1_D_INDEX)       = cons(M1_D_INDEX);
    prim(ALPHA1_D_INDEX)   = cons(ALPHA1_D_INDEX);
    prim(SIGMA_D_INDEX)    = cons(SIGMA_D_INDEX);

    return prim;
  }

  // Conversion from primitive to conserved variables
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::prim2cons(const FluxValue<cfg>& prim) const {
    FluxValue<cfg> cons;

    const auto alpha1          = prim(ALPHA1_BAR_INDEX)*(1.0 - prim(ALPHA1_D_INDEX));
    cons(M1_INDEX)             = alpha1*EOS_phase1.rho_value(prim(P1_INDEX));
    cons(M2_INDEX)             = (1.0 - alpha1 - prim(ALPHA1_D_INDEX))*EOS_phase2.rho_value(prim(P2_INDEX));
    cons(M1_D_INDEX)           = prim(M1_D_INDEX);
    cons(RHO_ALPHA1_BAR_INDEX) = (cons(M1_INDEX) + cons(M2_INDEX) + cons(M1_D_INDEX))*
                                 prim(ALPHA1_BAR_INDEX);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      cons(RHO_U_INDEX + d) = (cons(M1_INDEX) + cons(M2_INDEX) + cons(M1_D_INDEX))*
                              prim(U_INDEX + d);
    }
    cons(ALPHA1_D_INDEX)       = prim(ALPHA1_D_INDEX);
    cons(SIGMA_D_INDEX)        = prim(SIGMA_D_INDEX);

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
                                                       const typename Field::value_type H_bar,
                                                       typename Field::value_type& dalpha1_bar,
                                                       typename Field::value_type& alpha1_bar,
                                                       bool& relaxation_applied) {
        /*--- Update auxiliary values affected by the nonlinear function for which we seek a zero ---*/
        const auto alpha1 = alpha1_bar*(1.0 - (*conserved_variables)(ALPHA1_D_INDEX));
        const auto rho1   = (*conserved_variables)(M1_INDEX)/alpha1; /*--- TODO: Add a check in case of zero volume fraction ---*/
        const auto p1     = EOS_phase1.pres_value(rho1);

        const auto alpha2 = 1.0 - alpha1 - (*conserved_variables)(ALPHA1_D_INDEX);
        const auto rho2   = (*conserved_variables)(M2_INDEX)/alpha2; /*--- TODO: Add a check in case of zero volume fraction ---*/
        const auto p2     = EOS_phase2.pres_value(rho2);

        /*--- Compute the nonlinear function for which we seek the zero (basically the Laplace law) ---*/
        const auto F = std::isnan(H_bar) ? p1 - p2 :
                                           (1.0 - (*conserved_variables)(ALPHA1_D_INDEX))*(p1 - p2) - sigma*H_bar;

        /*--- Perform the relaxation only where really needed ---*/
        if(std::abs(F) > atol_Newton + rtol_Newton*std::min(EOS_phase1.get_p0(), sigma*std::abs(H_bar)) &&
           std::abs(dalpha1_bar) > atol_Newton) {
          relaxation_applied = true;

          // Compute the derivative w.r.t large-scale volume fraction recalling that for a barotropic EOS dp/drho = c^2
          const auto dF_dalpha1_bar = -(*conserved_variables)(M1_INDEX)/(alpha1_bar*alpha1_bar)*
                                       EOS_phase1.c_value(rho1)*EOS_phase1.c_value(rho1)
                                      -(*conserved_variables)(M2_INDEX)/((1.0 - alpha1_bar)*(1.0 - alpha1_bar))*
                                       EOS_phase2.c_value(rho2)*EOS_phase2.c_value(rho2);

          // Compute the large-scale volume fraction update
          dalpha1_bar = -F/dF_dalpha1_bar;
          if(dalpha1_bar > 0.0) {
            dalpha1_bar = std::min(dalpha1_bar, lambda*(1.0 - alpha1_bar));
          }
          else if(dalpha1_bar < 0.0) {
            dalpha1_bar = std::max(dalpha1_bar, -lambda*alpha1_bar);
          }

          if(alpha1_bar + dalpha1_bar < 0.0 || alpha1_bar + dalpha1_bar > 1.0) {
            throw std::runtime_error("Bounds exceeding value for large-scale volume fraction inside Newton step of reconstruction");
          }
          else {
            alpha1_bar += dalpha1_bar;
          }
        }

        /*--- Update the vector of conserved variables (probably not the optimal choice since I need this update only at the end of the Newton loop,
              but the most coherent one thinking about the transfer of mass) ---*/
        const auto rho = (*conserved_variables)(M1_INDEX)
                       + (*conserved_variables)(M2_INDEX)
                       + (*conserved_variables)(M1_D_INDEX);
        (*conserved_variables)(RHO_ALPHA1_BAR_INDEX) = rho*alpha1_bar;
      }

      // Relax the reconstruction
      //
      template<class Field>
      void Flux<Field>::relax_reconstruction(FluxValue<cfg>& q,
                                             const typename Field::value_type H_bar) {
        /*--- Declare and set relevant parameters ---*/
        std::size_t Newton_iter = 0;
        bool relaxation_applied = true;

        typename Field::value_type dalpha1_bar = std::numeric_limits<typename Field::value_type>::infinity();
        typename Field::value_type alpha1_bar  = q(RHO_ALPHA1_BAR_INDEX)/
                                                 (q(M1_INDEX) + q(M2_INDEX) + q(M1_D_INDEX));

        /*--- Apply Newton method ---*/
        while(relaxation_applied == true) {
          relaxation_applied = false;
          Newton_iter++;

          try {
            this->perform_Newton_step_relaxation(std::make_unique<FluxValue<cfg>>(q),
                                                 H_bar, dalpha1_bar, alpha1_bar, relaxation_applied);
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
