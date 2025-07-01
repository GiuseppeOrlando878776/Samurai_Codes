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
  static constexpr std::size_t Ml_INDEX          = 0;
  static constexpr std::size_t Mg_INDEX          = 1;
  static constexpr std::size_t Md_INDEX          = 2;
  static constexpr std::size_t RHO_Z_INDEX       = 3;
  static constexpr std::size_t RHO_ALPHA_l_INDEX = 4;
  static constexpr std::size_t RHO_U_INDEX       = 5;

  /*--- Save also the total number of (scalar) variables ---*/
  static constexpr std::size_t NVARS = 5 + dim;

  /*--- Use auxiliary variables for the indices also for primitive variables for the sake of generality ---*/
  static constexpr std::size_t ALPHA_l_INDEX = RHO_ALPHA_l_INDEX;
  static constexpr std::size_t U_INDEX       = RHO_U_INDEX;
  static constexpr std::size_t Z_INDEX       = RHO_Z_INDEX;
  static constexpr std::size_t Pl_INDEX      = Ml_INDEX;
  static constexpr std::size_t Pg_INDEX      = Mg_INDEX;
  static constexpr std::size_t ALPHA_d_INDEX = Md_INDEX;
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

    Flux(const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase_liq_,
         const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase_gas_,
         const double sigma_,
         const double mod_grad_alpha_l_min_,
         const double lambda_ = 0.9,
         const double atol_Newton_ = 1e-14,
         const double rtol_Newton_ = 1e-12,
         const std::size_t max_Newton_iters_ = 60); /*--- Constructor which accepts in input the equations of state of the two phases ---*/

  protected:
    const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase_liq;
    const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase_gas;

    const double sigma; /*--- Surface tension parameter ---*/

    const double mod_grad_alpha_l_min; /*--- Tolerance to compute the unit normal ---*/

    const double      lambda;           /*--- Parameter for bound preserving strategy ---*/
    const double      atol_Newton;      /*--- Absolute tolerance Newton method relaxation ---*/
    const double      rtol_Newton;      /*--- Relative tolerance Newton method relaxation ---*/
    const std::size_t max_Newton_iters; /*--- Maximum number of Newton iterations ---*/

    template<typename Gradient>
    FluxValue<cfg> evaluate_continuous_flux(const FluxValue<cfg>& q,
                                            const std::size_t curr_d,
                                            const Gradient& grad_alpha_l); /*--- Evaluate the 'continuous' flux for the state q along direction curr_d ---*/

    FluxValue<cfg> evaluate_hyperbolic_operator(const FluxValue<cfg>& q,
                                                const std::size_t curr_d); /*--- Evaluate the hyperbolic operator for the state q along direction curr_d --*/

    template<typename Gradient>
    FluxValue<cfg> evaluate_surface_tension_operator(const Gradient& grad_alpha_l,
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
                                            typename Field::value_type& dalpha_l,
                                            typename Field::value_type& alpha_l,
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
  Flux<Field>::Flux(const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase_liq_,
                    const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase_gas_,
                    const double sigma_,
                    const double mod_grad_alpha_l_min_,
                    const double lambda_,
                    const double atol_Newton_,
                    const double rtol_Newton_,
                    const std::size_t max_Newton_iters_):
    EOS_phase_liq(EOS_phase_liq_), EOS_phase_gas(EOS_phase_gas_),
    sigma(sigma_), mod_grad_alpha_l_min(mod_grad_alpha_l_min_),
    lambda(lambda_), atol_Newton(atol_Newton_), rtol_Newton(rtol_Newton_),
    max_Newton_iters(max_Newton_iters_) {}

  // Evaluate the 'continuous flux'
  //
  template<class Field>
  template<typename Gradient>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::evaluate_continuous_flux(const FluxValue<cfg>& q,
                                                                             const std::size_t curr_d,
                                                                             const Gradient& grad_alpha_l) {
    /*--- Initialize the resulting variable with the hyperbolic operator ---*/
    FluxValue<cfg> res = this->evaluate_hyperbolic_operator(q, curr_d);

    /*--- Add the contribution due to surface tension ---*/
    res += this->evaluate_surface_tension_operator(grad_alpha_l, curr_d);

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
    const auto rho   = q(Ml_INDEX) + q(Mg_INDEX) + q(Md_INDEX);
    const auto vel_d = q(RHO_U_INDEX + curr_d)/rho;

    /*--- Multiply the state the velcoity along the direction of interest ---*/
    res(Ml_INDEX) *= vel_d;
    res(Mg_INDEX) *= vel_d;
    res(Md_INDEX) *= vel_d;
    res(RHO_Z_INDEX) *= vel_d;
    res(RHO_ALPHA_l_INDEX) *= vel_d;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      res(RHO_U_INDEX + d) *= vel_d;
    }

    /*--- Compute and add the contribution due to the pressure ---*/
    const auto alpha_l = q(RHO_ALPHA_l_INDEX)/rho;
    const auto alpha_d = alpha_l*q(Md_INDEX)/q(Ml_INDEX); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto alpha_g = 1.0 - alpha_l - alpha_d;

    const auto rho_liq = (q(Ml_INDEX) + q(Md_INDEX))/(alpha_l + alpha_d); /*--- TODO: Add a check in case of zero volume fraction ---*/
    /*--- Relation alpha_l/Y_l = (alpha_l + alpha_d)/(Y_l + Y_d) holds!!! ---*/
    const auto p_liq   = EOS_phase_liq.pres_value(rho_liq);
    const auto rho_g   = q(Mg_INDEX)/alpha_g; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto p_g     = EOS_phase_gas.pres_value(rho_g);

    const auto Sigma_d = q(RHO_Z_INDEX)/std::pow(rho_liq, 2.0/3.0);

    const auto p       = (alpha_l + alpha_d)*p_liq + alpha_g*p_g - 2.0/3.0*sigma*Sigma_d;

    res(RHO_U_INDEX + curr_d) += p;

    return res;
  }

  // Evaluate the surface tension operator
  //
  template<class Field>
  template<typename Gradient>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::evaluate_surface_tension_operator(const Gradient& grad_alpha_l,
                                                                                      const std::size_t curr_d) {
    /*--- Sanity check in terms of dimensions ---*/
    assert(curr_d < Field::dim);

    /*--- Initialize the resulting variable ---*/
    FluxValue<cfg> res;

    // Set to zero all the contributions
    res(Ml_INDEX) = 0.0;
    res(Mg_INDEX) = 0.0;
    res(RHO_ALPHA_l_INDEX) = 0.0;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      res(RHO_U_INDEX + d) = 0.0;
    }
    res(Md_INDEX) = 0.0;
    res(RHO_Z_INDEX) = 0.0;

    /*--- Add the contribution due to surface tension ---*/
    const auto mod_grad_alpha_l = std::sqrt(xt::sum(grad_alpha_l*grad_alpha_l)());

    if(mod_grad_alpha_l > mod_grad_alpha_l_min) {
      const auto n = grad_alpha_l/mod_grad_alpha_l;

      if(curr_d == 0) {
        res(RHO_U_INDEX) += sigma*(n(0)*n(0) - 1.0)*mod_grad_alpha_l;
        res(RHO_U_INDEX + 1) += sigma*n(0)*n(1)*mod_grad_alpha_l;
      }
      else if(curr_d == 1) {
        res(RHO_U_INDEX) += sigma*n(0)*n(1)*mod_grad_alpha_l;
        res(RHO_U_INDEX + 1) += sigma*(n(1)*n(1) - 1.0)*mod_grad_alpha_l;
      }
    }

    return res;
  }

  // Conversion from conserved to primitive variables
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::cons2prim(const FluxValue<cfg>& cons) const {
    FluxValue<cfg> prim;

    const auto rho      = cons(Ml_INDEX) + cons(Mg_INDEX) + cons(Md_INDEX);
    prim(ALPHA_l_INDEX) = cons(RHO_ALPHA_l_INDEX)/rho;
    prim(ALPHA_d_INDEX) = prim(ALPHA_l_INDEX)*cons(Md_INDEX)/cons(Ml_INDEX);
    const auto rho_liq  = (cons(Ml_INDEX) + cons(Md_INDEX))/
                          (prim(ALPHA_l_INDEX) + prim(ALPHA_d_INDEX)); /*--- TODO: Add a check in case of zero volume fraction ---*/
    prim(Pl_INDEX)      = EOS_phase_liq.pres_value(rho_liq);
    const auto rho_g    = cons(Mg_INDEX)/
                          (1.0 - prim(ALPHA_l_INDEX) - prim(ALPHA_d_INDEX)); /*--- TODO: Add a check in case of zero volume fraction ---*/
    prim(Pg_INDEX)      = EOS_phase_gas.pres_value(rho_g);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      prim(U_INDEX + d) = cons(RHO_U_INDEX + d)/rho;
    }
    prim(Z_INDEX) = cons(RHO_Z_INDEX)/rho;

    return prim;
  }

  // Conversion from primitive to conserved variables
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::prim2cons(const FluxValue<cfg>& prim) const {
    FluxValue<cfg> cons;

    const auto rho_liq      = EOS_phase_liq.rho_value(prim(Pl_INDEX));
    const auto rho_g        = EOS_phase_gas.rho_value(prim(Pg_INDEX));
    cons(Ml_INDEX)          = prim(ALPHA_l_INDEX)*rho_liq;
    cons(Md_INDEX)          = prim(ALPHA_d_INDEX)*rho_liq;
    cons(Mg_INDEX)          = (1.0 - prim(ALPHA_l_INDEX) - prim(ALPHA_d_INDEX))*rho_g;
    const auto rho          = cons(Ml_INDEX) + cons(Mg_INDEX) + cons(Md_INDEX);
    cons(RHO_ALPHA_l_INDEX) = rho*prim(ALPHA_l_INDEX);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      cons(RHO_U_INDEX + d) = rho*prim(U_INDEX + d);
    }
    cons(RHO_Z_INDEX) = rho*prim(Z_INDEX);

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
                                                       typename Field::value_type& dalpha_l,
                                                       typename Field::value_type& alpha_l,
                                                       bool& relaxation_applied) {
        if(!std::isnan(H)) {
          /*--- Update auxiliary values affected by the nonlinear function for which we seek a zero ---*/
          const auto alpha_d = alpha_l*(*conserved_variables)(Md_INDEX)/(*conserved_variables)(Ml_INDEX);
          const auto alpha_g = 1.0 - alpha_l - alpha_d;

          const auto rho_liq = ((*conserved_variables)(Ml_INDEX) + (*conserved_variables)(Md_INDEX))/
                               (alpha_l + alpha_d); /*--- TODO: Add a check in case of zero volume fraction ---*/
          const auto p_liq   = EOS_phase_liq.pres_value(rho_liq);

          const auto rho_g   = (*conserved_variables)(Mg_INDEX)/alpha_g; /*--- TODO: Add a check in case of zero volume fraction ---*/
          const auto p_g     = EOS_phase_gas.pres_value(rho_g);

          /*--- Compute the nonlinear function for which we seek the zero (basically the Laplace law) ---*/
          const auto delta_p = p_liq - p_g;
          const auto F_LS    = (*conserved_variables)(Ml_INDEX)*(delta_p - sigma*H);
          const auto aux_SS  = 2.0/3.0*sigma*
                               (*conserved_variables)(RHO_Z_INDEX)*
                               std::pow((*conserved_variables)(Ml_INDEX), 1.0/3.0);
          const auto F_SS    = (*conserved_variables)(Md_INDEX)*delta_p
                             - std::pow(alpha_l, -1.0/3.0)*aux_SS;
          const auto F       = F_LS + F_SS;

          /*--- Perform the relaxation only where really needed ---*/
          if(std::abs(F) > atol_Newton + rtol_Newton*std::min(EOS_phase_liq.get_p0(), sigma*std::abs(H)) &&
             std::abs(dalpha_l) > atol_Newton) {
            relaxation_applied = true;

            // Compute the derivative w.r.t large-scale volume fraction recalling that for a barotropic EOS dp/drho = c^2
            const auto ddelta_p_dalpha_l = -(*conserved_variables)(Ml_INDEX)/(alpha_l*alpha_l)*
                                           EOS_phase_liq.c_value(rho_liq)*EOS_phase_liq.c_value(rho_liq)
                                           -(*conserved_variables)(Mg_INDEX)/(alpha_g*alpha_g)*
                                           EOS_phase_gas.c_value(rho_g)*EOS_phase_gas.c_value(rho_g)*
                                           ((*conserved_variables)(Ml_INDEX) + (*conserved_variables)(Md_INDEX))/
                                           (*conserved_variables)(Ml_INDEX);
            const auto dF_LS_dalpha_l    = (*conserved_variables)(Ml_INDEX)*ddelta_p_dalpha_l;
            const auto dF_SS_dalpha_l    = (*conserved_variables)(Md_INDEX)*ddelta_p_dalpha_l
                                         + 1.0/3.0*std::pow(alpha_l, -4.0/3.0)*aux_SS;
            const auto dF_dalpha_l       = dF_LS_dalpha_l + dF_SS_dalpha_l;

            // Compute the large-scale volume fraction update
            dalpha_l = -F/dF_dalpha_l;
            if(dalpha_l > 0.0) {
              dalpha_l = std::min(dalpha_l, lambda*(1.0 - alpha_l));
            }
            else if(dalpha_l < 0.0) {
              dalpha_l = std::max(dalpha_l, -lambda*alpha_l);
            }

            if(alpha_l + dalpha_l < 0.0 || alpha_l + dalpha_l > 1.0) {
              throw std::runtime_error("Bounds exceeding value for large-scale liquid volume fraction inside Newton step of reconstruction");
            }
            else {
              alpha_l += dalpha_l;
            }
          }

          /*--- Update the vector of conserved variables (probably not the optimal choice since I need this update only at the end of the Newton loop,
                but the most coherent one thinking about the transfer of mass) ---*/
          const auto rho = (*conserved_variables)(Ml_INDEX)
                         + (*conserved_variables)(Mg_INDEX)
                         + (*conserved_variables)(Md_INDEX);
          (*conserved_variables)(RHO_ALPHA_l_INDEX) = rho*alpha_l;
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

        typename Field::value_type dalpha_l = std::numeric_limits<typename Field::value_type>::infinity();
        typename Field::value_type alpha_l  = q(RHO_ALPHA_l_INDEX)/
                                              (q(Ml_INDEX) + q(Mg_INDEX) + q(Md_INDEX));

        /*--- Apply Newton method ---*/
        while(relaxation_applied == true) {
          relaxation_applied = false;
          Newton_iter++;

          try {
            this->perform_Newton_step_relaxation(std::make_unique<FluxValue<cfg>>(q),
                                                 H, dalpha_l, alpha_l, relaxation_applied);
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
