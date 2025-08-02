// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#ifndef HLLC_flux_hpp
#define HLLC_flux_hpp

#include "flux_base.hpp"

#define VERBOSE_FLUX

namespace samurai {
  using namespace EquationData;

  /**
    * Implementation of a HLLC flux
    */
  template<class Field>
  class HLLCFlux: public Flux<Field> {
  public:
    HLLCFlux(const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase1_,
             const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase2_,
             const typename Field::value_type lambda_,
             const typename Field::value_type atol_Newton_,
             const typename Field::value_type rtol_Newton_,
             const std::size_t max_Newton_iters_); /*--- Constructor which accepts in input the equations of state of the two phases ---*/

    auto make_flux(); /*--- Compute the flux over all the directions ---*/

  private:
    auto compute_middle_state(const FluxValue<typename Flux<Field>::cfg>& q,
                              const typename Field::value_type S,
                              const typename Field::value_type S_star,
                              const std::size_t curr_d) const; /*--- Compute the middle state ---*/

    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                               const FluxValue<typename Flux<Field>::cfg>& qR,
                                                               const std::size_t curr_d); /*--- HLLC flux along direction d ---*/
  };

  // Constructor derived from base class
  //
  template<class Field>
  HLLCFlux<Field>::HLLCFlux(const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase1_,
                            const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase2_,
                            const typename Field::value_type lambda_,
                            const typename Field::value_type atol_Newton_,
                            const typename Field::value_type rtol_Newton_,
                            const std::size_t max_Newton_iters_):
    Flux<Field>(EOS_phase1_, EOS_phase2_,
                lambda_, atol_Newton_, rtol_Newton_, max_Newton_iters_) {}

  // Implement the auxiliary routine that computes the middle state
  //
  template<class Field>
  auto HLLCFlux<Field>::compute_middle_state(const FluxValue<typename Flux<Field>::cfg>& q,
                                             const typename Field::value_type S,
                                             const typename Field::value_type S_star,
                                             const std::size_t curr_d) const {
    /*--- Pre-fetch some variables used multiple times in order to exploit possible vectorization ---*/
    const auto m1 = q(M1_INDEX);
    const auto m2 = q(M2_INDEX);

    /*--- Compute middle state ---*/
    const auto rho     = m1 + m2;
    const auto inv_rho = static_cast<typename Field::value_type>(1.0)/rho;
    const auto vel_d   = q(RHO_U_INDEX + curr_d)*inv_rho;
    const auto alpha1  = q(RHO_ALPHA1_INDEX)*inv_rho;

    FluxValue<typename Flux<Field>::cfg> q_star;
    q_star(M1_INDEX)             = m1*((S - vel_d)/(S - S_star));
    q_star(M2_INDEX)             = m2*((S - vel_d)/(S - S_star));
    const auto rho_star          = q_star(M1_INDEX) + q_star(M2_INDEX);
    q_star(RHO_ALPHA1_INDEX)     = rho_star*alpha1;
    q_star(RHO_U_INDEX + curr_d) = rho_star*S_star;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      if(d != curr_d) {
        q_star(RHO_U_INDEX + d) = rho_star*(q(RHO_U_INDEX + d)*inv_rho);
      }
    }

    return q_star;
  }

  // Implementation of a HLLC flux for the system
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg>
  HLLCFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                         const FluxValue<typename Flux<Field>::cfg>& qR,
                                         std::size_t curr_d) {
    /*--- Pre-fetch some variables used multiple times in order to exploit possible vectorization ---*/
    const auto m1_L         = qL(M1_INDEX);
    const auto m2_L         = qL(M2_INDEX);
    const auto rho_alpha1_L = qL(RHO_ALPHA1_INDEX);

    const auto m1_R         = qR(M1_INDEX);
    const auto m2_R         = qR(M2_INDEX);
    const auto rho_alpha1_R = qR(RHO_ALPHA1_INDEX);

    /*--- Verify if left and right state are coherent ---*/
    #ifdef VERBOSE_FLUX
      if(m1_L < static_cast<typename Field::value_type>(0.0)) {
        throw std::runtime_error(std::string("Negative mass phase 1 left state: " + std::to_string(m1_L)));
      }
      if(m2_L < static_cast<typename Field::value_type>(0.0)) {
        throw std::runtime_error(std::string("Negative mass phase 2 left state: " + std::to_string(m2_L)));
      }
      if(rho_alpha1_L < static_cast<typename Field::value_type>(0.0)) {
        throw std::runtime_error(std::string("Negative volume fraction phase 1 left state: " + std::to_string(rho_alpha1_L)));
      }

      if(m1_R < static_cast<typename Field::value_type>(0.0)) {
        throw std::runtime_error(std::string("Negative mass phase 1 right state: " + std::to_string(m1_R)));
      }
      if(m2_R < static_cast<typename Field::value_type>(0.0)) {
        throw std::runtime_error(std::string("Negative mass phase 2 right state: " + std::to_string(m2_R)));
      }
      if(rho_alpha1_R < static_cast<typename Field::value_type>(0.0)) {
        throw std::runtime_error(std::string("Negative volume fraction phase 1 right state: " + std::to_string(rho_alpha1_R)));
      }
    #endif

    /*--- Left state ---*/
    const auto rho_L     = m1_L + m2_L;
    const auto inv_rho_L = static_cast<typename Field::value_type>(1.0)/rho_L;
    const auto vel_d_L   = qL(RHO_U_INDEX + curr_d)*inv_rho_L;

    const auto alpha1_L  = rho_alpha1_L*inv_rho_L;
    const auto rho1_L    = m1_L/alpha1_L; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rho2_L    = m2_L/(static_cast<typename Field::value_type>(1.0) - alpha1_L); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto Y1_L      = m1_L*inv_rho_L;
    const auto c_L       = std::sqrt(Y1_L*
                                     this->EOS_phase1.c_value(rho1_L)*
                                     this->EOS_phase1.c_value(rho1_L) +
                                     (static_cast<typename Field::value_type>(1.0) - Y1_L)*
                                     this->EOS_phase2.c_value(rho2_L)*
                                     this->EOS_phase2.c_value(rho2_L));
    const auto p1_L      = this->EOS_phase1.pres_value(rho1_L);
    const auto p2_L      = this->EOS_phase2.pres_value(rho2_L);
    const auto p_L       = alpha1_L*p1_L
                         + (static_cast<typename Field::value_type>(1.0) - alpha1_L)*p2_L;

    /*--- Right state ---*/
    const auto rho_R     = m1_R + m2_R;
    const auto inv_rho_R = static_cast<typename Field::value_type>(1.0)/rho_R;
    const auto vel_d_R   = qR(RHO_U_INDEX + curr_d)*inv_rho_R;

    const auto alpha1_R  = rho_alpha1_R*inv_rho_R;
    const auto rho1_R    = m1_R/alpha1_R; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rho2_R    = m2_R/(static_cast<typename Field::value_type>(1.0) - alpha1_R); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto Y1_R      = m1_R*inv_rho_R;
    const auto c_R       = std::sqrt(Y1_R*
                                     this->EOS_phase1.c_value(rho1_R)*
                                     this->EOS_phase1.c_value(rho1_R) +
                                     (static_cast<typename Field::value_type>(1.0) - Y1_R)*
                                     this->EOS_phase2.c_value(rho2_R)*
                                     this->EOS_phase2.c_value(rho2_R));
    const auto p1_R      = this->EOS_phase1.pres_value(rho1_R);
    const auto p2_R      = this->EOS_phase2.pres_value(rho2_R);
    const auto p_R       = alpha1_R*p1_R
                         + (static_cast<typename Field::value_type>(1.0) - alpha1_R)*p2_R;

    /*--- Compute speeds of wave propagation ---*/
    const auto s_L    = std::min(vel_d_L - c_L, vel_d_R - c_R);
    const auto s_R    = std::max(vel_d_L + c_L, vel_d_R + c_R);
    const auto s_star = (p_R - p_L + rho_L*vel_d_L*(s_L - vel_d_L) - rho_R*vel_d_R*(s_R - vel_d_R))/
                        (rho_L*(s_L - vel_d_L) - rho_R*(s_R - vel_d_R));

    /*--- Compute intermediate states ---*/
    auto q_star_L = compute_middle_state(qL, s_L, s_star, curr_d);
    auto q_star_R = compute_middle_state(qR, s_R, s_star, curr_d);

    /*--- Compute the flux ---*/
    if(s_L >= static_cast<typename Field::value_type>(0.0)) {
      return this->evaluate_continuous_flux(qL, curr_d);
    }
    else if(s_L < static_cast<typename Field::value_type>(0.0) &&
            s_star >= static_cast<typename Field::value_type>(0.0)) {
      return this->evaluate_continuous_flux(qL, curr_d) + s_L*(q_star_L - qL);
    }
    else if(s_star < static_cast<typename Field::value_type>(0.0) &&
            s_R >= static_cast<typename Field::value_type>(0.0)) {
      return this->evaluate_continuous_flux(qR, curr_d) + s_R*(q_star_R - qR);
    }
    else if(s_R < static_cast<typename Field::value_type>(0.0)) {
      return this->evaluate_continuous_flux(qR, curr_d);
    }
  }

  // Implement the contribution of the discrete flux for all the dimensions.
  //
  template<class Field>
  auto HLLCFlux<Field>::make_flux() {
    FluxDefinition<typename Flux<Field>::cfg> HLLC_f;

    /*--- Perform the loop over each dimension to compute the flux contribution ---*/
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
         {
           static constexpr int d = decltype(integral_constant_d)::value;

           // Compute now the "discrete" flux function, in this case a HLLC flux
           HLLC_f[d].cons_flux_function = [&](samurai::FluxValue<typename Flux<Field>::cfg>& flux,
                                              const StencilData<typename Flux<Field>::cfg>& /*data*/,
                                              const StencilValues<typename Flux<Field>::cfg> field)
                                              {
                                                #ifdef ORDER_2
                                                  // MUSCL reconstruction
                                                  const FluxValue<typename Flux<Field>::cfg> primLL = this->cons2prim(field[0]);
                                                  const FluxValue<typename Flux<Field>::cfg> primL  = this->cons2prim(field[1]);
                                                  const FluxValue<typename Flux<Field>::cfg> primR  = this->cons2prim(field[2]);
                                                  const FluxValue<typename Flux<Field>::cfg> primRR = this->cons2prim(field[3]);

                                                  FluxValue<typename Flux<Field>::cfg> primL_recon,
                                                                                       primR_recon;
                                                  this->perform_reconstruction(primLL, primL, primR, primRR,
                                                                               primL_recon, primR_recon);

                                                  const FluxValue<typename Flux<Field>::cfg> qL = this->prim2cons(primL_recon);
                                                  const FluxValue<typename Flux<Field>::cfg> qR = this->prim2cons(primR_recon);
                                                #else
                                                  // Extract the states
                                                  const FluxValue<typename Flux<Field>::cfg> qL = field[0];
                                                  const FluxValue<typename Flux<Field>::cfg> qR = field[1];
                                                #endif

                                                flux = compute_discrete_flux(qL, qR, d);
                                              };
        }
    );

    auto scheme = make_flux_based_scheme(HLLC_f);
    scheme.set_name("HLLC");

    return scheme;
  }

} // end of namespace

#endif
