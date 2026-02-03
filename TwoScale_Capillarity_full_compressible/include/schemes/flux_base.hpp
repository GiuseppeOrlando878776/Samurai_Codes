// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
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
    static_assert(Field::dim == EquationData::dim, "The spatial dimensions between Field and the parameter list do not match");
    static_assert(Field::n_comp == EquationData::NVARS, "The number of elements in the state does not correspond to the number of equations");
    #ifdef ORDER_2
      static constexpr std::size_t stencil_size = 4;
    #else
      static constexpr std::size_t stencil_size = 2;
    #endif

    using cfg = FluxConfig<SchemeType::NonLinear, stencil_size, Field, Field>;

    using Number = typename Field::value_type; /*--- Shortcut for the arithmetic type ---*/

    Flux(const LinearizedBarotropicEOS<Number>& EOS_phase_liq_,
         const LinearizedBarotropicEOS<Number>& EOS_phase_gas_,
         const Number sigma_,
         const Number mod_grad_alpha_l_min_,
         const Number lambda_ = static_cast<Number>(0.9),
         const Number atol_Newton_ = static_cast<Number>(1e-14),
         const Number rtol_Newton_ = static_cast<Number>(1e-12),
         const std::size_t max_Newton_iters_ = 60); /*--- Constructor which accepts in input the equations of state of the two phases ---*/

  protected:
    const LinearizedBarotropicEOS<Number>& EOS_phase_liq;
    const LinearizedBarotropicEOS<Number>& EOS_phase_gas;

    const Number sigma; /*--- Surface tension parameter ---*/

    const Number mod_grad_alpha_l_min; /*--- Tolerance to compute the unit normal ---*/

    const Number      lambda;           /*--- Parameter for bound preserving strategy ---*/
    const Number      atol_Newton;      /*--- Absolute tolerance Newton method relaxation ---*/
    const Number      rtol_Newton;      /*--- Relative tolerance Newton method relaxation ---*/
    const std::size_t max_Newton_iters; /*--- Maximum number of Newton iterations ---*/

    template<typename Gradient>
    FluxValue<cfg> evaluate_continuous_flux(const FluxValue<cfg>& q,
                                            const std::size_t curr_d,
                                            const Gradient& grad_alpha_l); /*--- Evaluate the 'continuous' flux for the state q
                                                                                 along direction curr_d ---*/

    FluxValue<cfg> evaluate_hyperbolic_operator(const FluxValue<cfg>& q,
                                                const std::size_t curr_d); /*--- Evaluate the hyperbolic operator for the state q
                                                                                 along direction curr_d --*/

    template<typename Gradient>
    FluxValue<cfg> evaluate_surface_tension_operator(const Gradient& grad_alpha_l,
                                                     const std::size_t curr_d); /*--- Evaluate the surface tension operator for the state q
                                                                                      along direction curr_d ---*/

    FluxValue<cfg> cons2prim(const FluxValue<cfg>& cons) const; /*--- Conversion from conserved to primitive variables ---*/

    FluxValue<cfg> prim2cons(const FluxValue<cfg>& prim) const; /*--- Conversion from primitive to conserved variables ---*/

    #ifdef RELAX_RECONSTRUCTION
      template<typename State>
      void perform_Newton_step_relaxation(State conserved_variables,
                                          const Number H,
                                          Number& dalpha_l,
                                          Number& alpha_l,
                                          bool& relaxation_applied); /*--- Perform a Newton step relaxation for a state vector
                                                                           (it is not a real space dependent procedure,
                                                                            but I would need to be able to do it inside the flux location
                                                                            for MUSCL reconstruction) ---*/

      void relax_reconstruction(FluxValue<cfg>& q,
                                const Number H); /*--- Relax reconstructed state ---*/
    #endif
  };

  // Class constructor in order to be able to work with the equation of state
  //
  template<class Field>
  Flux<Field>::Flux(const LinearizedBarotropicEOS<Number>& EOS_phase_liq_,
                    const LinearizedBarotropicEOS<Number>& EOS_phase_gas_,
                    const Number sigma_,
                    const Number mod_grad_alpha_l_min_,
                    const Number lambda_,
                    const Number atol_Newton_,
                    const Number rtol_Newton_,
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

    /*--- Pre-fetch some variables used multiple times in order to exploit possible vectorization ---*/
    const auto m_l = q(Ml_INDEX);
    const auto m_g = q(Mg_INDEX);
    const auto m_d = q(Md_INDEX);

    /*--- Compute the current velocity ---*/
    const auto rho     = m_l + m_g + m_d;
    const auto inv_rho = static_cast<Number>(1.0)/rho;
    const auto vel_d   = q(RHO_U_INDEX + curr_d)*inv_rho;

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
    const auto alpha_l = q(RHO_ALPHA_l_INDEX)*inv_rho;
    const auto alpha_d = alpha_l*m_d/m_l; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto alpha_g = static_cast<Number>(1.0) - alpha_l - alpha_d;

    const auto rho_liq = (m_l + m_d)/(alpha_l + alpha_d); /*--- TODO: Add a check in case of zero volume fraction ---*/
    /*--- Relation alpha_l/Y_l = (alpha_l + alpha_d)/(Y_l + Y_d) holds!!! ---*/
    const auto p_liq   = EOS_phase_liq.pres_value(rho_liq);
    const auto rho_g   = m_g/alpha_g; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto p_g     = EOS_phase_gas.pres_value(rho_g);

    const auto Sigma_d = q(RHO_Z_INDEX)/std::cbrt(rho_liq*rho_liq);

    const auto p       = (alpha_l + alpha_d)*p_liq
                       + alpha_g*p_g
                       - static_cast<Number>(2.0/3.0)*sigma*Sigma_d;

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
    res.fill(static_cast<Number>(0.0));

    /*--- Add the contribution due to surface tension ---*/
    //const auto mod_grad_alpha_l = std::sqrt(xt::sum(grad_alpha_l*grad_alpha_l)());
    auto mod2_grad_alpha_l = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      mod2_grad_alpha_l += grad_alpha_l[d]*grad_alpha_l[d];
    }
    const auto mod_grad_alpha_l = std::sqrt(mod2_grad_alpha_l);

    if(mod_grad_alpha_l > mod_grad_alpha_l_min) {
      const auto n  = grad_alpha_l/mod_grad_alpha_l;
      const auto nx = n(0);
      const auto ny = n(1);

      if(curr_d == 0) {
        res(RHO_U_INDEX) += sigma*(nx*nx - static_cast<Number>(1.0))*mod_grad_alpha_l;
        res(RHO_U_INDEX + 1) += sigma*nx*ny*mod_grad_alpha_l;
      }
      else if(curr_d == 1) {
        res(RHO_U_INDEX) += sigma*nx*ny*mod_grad_alpha_l;
        res(RHO_U_INDEX + 1) += sigma*(ny*ny - static_cast<Number>(1.0))*mod_grad_alpha_l;
      }
    }

    return res;
  }

  // Conversion from conserved to primitive variables
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::cons2prim(const FluxValue<cfg>& cons) const {
    FluxValue<cfg> prim;

    /*--- Pre-fetch some variables used multiple times in order to exploit possible vectorization ---*/
    const auto m_l = cons(Ml_INDEX);
    const auto m_g = cons(Mg_INDEX);
    const auto m_d = cons(Md_INDEX);

    /*--- Compute primitive variables ---*/
    const auto rho      = m_l + m_g + m_d;
    const auto inv_rho  = static_cast<Number>(1.0)/rho;
    const auto alpha_l  = cons(RHO_ALPHA_l_INDEX)*inv_rho;
    prim(ALPHA_l_INDEX) = alpha_l;
    const auto alpha_d  = alpha_l*m_d/m_l;
    prim(ALPHA_d_INDEX) = alpha_d;

    const auto rho_liq  = (m_l + m_d)/(alpha_l + alpha_d); /*--- TODO: Add a check in case of zero volume fraction ---*/
    prim(Pl_INDEX)      = EOS_phase_liq.pres_value(rho_liq);
    const auto rho_g    = m_g/(static_cast<Number>(1.0) - alpha_l - alpha_d);
                          /*--- TODO: Add a check in case of zero volume fraction ---*/
    prim(Pg_INDEX)      = EOS_phase_gas.pres_value(rho_g);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      prim(U_INDEX + d) = cons(RHO_U_INDEX + d)*inv_rho;
    }
    prim(Z_INDEX) = cons(RHO_Z_INDEX)*inv_rho;

    return prim;
  }

  // Conversion from primitive to conserved variables
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::prim2cons(const FluxValue<cfg>& prim) const {
    FluxValue<cfg> cons;

    /*--- Pre-fetch some variables used multiple times in order to exploit possible vectorization ---*/
    const auto alpha_l = prim(ALPHA_l_INDEX);
    const auto alpha_d = prim(ALPHA_d_INDEX);

    /*--- Compute conserved variables ---*/
    const auto rho_liq      = EOS_phase_liq.rho_value(prim(Pl_INDEX));
    const auto m_l          = alpha_l*rho_liq;
    cons(Ml_INDEX)          = m_l;
    const auto m_d          = alpha_d*rho_liq;
    cons(Md_INDEX)          = m_d;

    const auto rho_g        = EOS_phase_gas.rho_value(prim(Pg_INDEX));
    const auto m_g          = (static_cast<Number>(1.0) - alpha_l - alpha_d)*rho_g;
    cons(Mg_INDEX)          = m_g;

    const auto rho          = m_l + m_g + m_d;
    cons(RHO_ALPHA_l_INDEX) = rho*prim(ALPHA_l_INDEX);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      cons(RHO_U_INDEX + d) = rho*prim(U_INDEX + d);
    }
    cons(RHO_Z_INDEX) = rho*prim(Z_INDEX);

    return cons;
  }

  // Relax reconstruction
  //
  #ifdef RELAX_RECONSTRUCTION
    // Perform a Newton step relaxation for a single vector state (i.e. a single cell) without mass transfer
    //
    template<class Field>
    template<typename State>
    void Flux<Field>::perform_Newton_step_relaxation(State conserved_variables,
                                                     const Number H,
                                                     Number& dalpha_l,
                                                     Number& alpha_l,
                                                     bool& relaxation_applied) {
      if(!std::isnan(H)) {
        /*--- Pre-fetch some variables used multiple times in order to exploit possible vectorization ---*/
        const auto m_l = conserved_variables(Ml_INDEX);
        const auto m_g = conserved_variables(Mg_INDEX);
        const auto m_d = conserved_variables(Md_INDEX);

        /*--- Update auxiliary values affected by the nonlinear function for which we seek a zero ---*/
        const auto alpha_d = alpha_l*m_d/m_l;
        const auto alpha_g = static_cast<Number>(1.0) - alpha_l - alpha_d;

        const auto rho_liq = (m_l + m_d)/(alpha_l + alpha_d); /*--- TODO: Add a check in case of zero volume fraction ---*/
        const auto p_liq   = EOS_phase_liq.pres_value(rho_liq);

        const auto rho_g   = m_g/alpha_g; /*--- TODO: Add a check in case of zero volume fraction ---*/
        const auto p_g     = EOS_phase_gas.pres_value(rho_g);

        /*--- Compute the nonlinear function for which we seek the zero (basically the Laplace law) ---*/
        const auto delta_p = p_liq - p_g;
        const auto F_LS    = m_l*(delta_p - sigma*H);
        const auto aux_SS  = static_cast<Number>(2.0/3.0)*sigma*
                             conserved_variables(RHO_Z_INDEX)*std::cbrt(m_l);
        const auto F_SS    = m_d*delta_p - (static_cast<Number>(1.0)/std::cbrt(alpha_l))*aux_SS; /*--- TODO: Add a check in case of zero volume fraction ---*/
        const auto F       = F_LS + F_SS;

        /*--- Perform the relaxation only where really needed ---*/
        if(std::abs(F) > atol_Newton + rtol_Newton*std::min(EOS_phase_liq.get_p0(), sigma*std::abs(H)) &&
           std::abs(dalpha_l) > atol_Newton) {
          relaxation_applied = true;

          // Compute the derivative w.r.t large-scale volume fraction recalling that for a barotropic EOS dp/drho = c^2
          const auto ddelta_p_dalpha_l = -m_l/(alpha_l*alpha_l)*
                                         EOS_phase_liq.c_value(rho_liq)*EOS_phase_liq.c_value(rho_liq)
                                         -m_g/(alpha_g*alpha_g)*
                                         EOS_phase_gas.c_value(rho_g)*EOS_phase_gas.c_value(rho_g)*
                                         (m_l + m_d)/m_l;
          const auto dF_LS_dalpha_l    = m_l*ddelta_p_dalpha_l;
          const auto dF_SS_dalpha_l    = m_d*ddelta_p_dalpha_l
                                       + static_cast<Number>(1.0/3.0)*
                                         std::pow(alpha_l, static_cast<Number>(-4.0/3.0))*aux_SS;
          const auto dF_dalpha_l       = dF_LS_dalpha_l + dF_SS_dalpha_l;

          // Compute the large-scale volume fraction update
          dalpha_l = -F/dF_dalpha_l;
          if(dalpha_l > static_cast<Number>(0.0)) {
            dalpha_l = std::min(dalpha_l, lambda*(static_cast<Number>(1.0) - alpha_l));
          }
          else if(dalpha_l < static_cast<Number>(0.0)) {
            dalpha_l = std::max(dalpha_l, -lambda*alpha_l);
          }

          if(alpha_l + dalpha_l < static_cast<Number>(0.0) ||
             alpha_l + dalpha_l > static_cast<Number>(1.0)) {
            // I should never get here. Added only for the sake of safety!!
            throw std::runtime_error("Bounds exceeding value for large-scale liquid volume fraction inside Newton step of reconstruction");
          }
          else {
            alpha_l += dalpha_l;
          }
        }

        /*--- Update the vector of conserved variables
              (probably not the optimal choice since I need this update only at the end of the Newton loop,
               but the most coherent one thinking about the transfer of mass) ---*/
        conserved_variables(RHO_ALPHA_l_INDEX) = (m_l + m_g + m_d)*alpha_l;
      }
    }

    // Relax the reconstruction
    //
    template<class Field>
    void Flux<Field>::relax_reconstruction(FluxValue<cfg>& q,
                                           const Number H) {
      /*--- Declare and set relevant parameters ---*/
      std::size_t Newton_iter = 0;
      bool relaxation_applied = true;

      auto dalpha_l = std::numeric_limits<Number>::infinity();
      auto alpha_l  = q(RHO_ALPHA_l_INDEX)/
                      (q(Ml_INDEX) + q(Mg_INDEX) + q(Md_INDEX));

      /*--- Apply Newton method ---*/
      while(relaxation_applied == true) {
        relaxation_applied = false;
        Newton_iter++;

        try {
          this->perform_Newton_step_relaxation(q, H, dalpha_l, alpha_l, relaxation_applied);
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

} // end namespace samurai
