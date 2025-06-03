// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#ifndef HLLC_flux_hpp
#define HLLC_flux_hpp

#include "flux_base.hpp"

//#define VERBOSE_FLUX

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
             const double sigma_,
             const double mod_grad_alpha1_min_,
             const double lambda_,
             const double atol_Newton_,
             const double rtol_Newton_,
             const std::size_t max_Newton_iters_); /*--- Constructor which accepts in inputs the equations of state of the two phases ---*/

    #ifdef ORDER_2
      template<typename Field_Scalar>
      auto make_flux(const Field_Scalar& H); /*--- Compute the flux over all the directions ---*/
    #else
      auto make_flux(); /*--- Compute the flux over all the directions ---*/
    #endif

  private:
    auto compute_middle_state(const FluxValue<typename Flux<Field>::cfg>& q,
                              const typename Field::value_type S,
                              const typename Field::value_type S_star,
                              const std::size_t curr_d) const; /*--- Compute the middle state ---*/

    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                               const FluxValue<typename Flux<Field>::cfg>& qR,
                                                               const std::size_t curr_d); /*--- HLLC flux for the along direction curr_d ---*/
  };

  // Constructor derived from the base class
  //
  template<class Field>
  HLLCFlux<Field>::HLLCFlux(const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase1_,
                            const LinearizedBarotropicEOS<typename Field::value_type>& EOS_phase2_,
                            const double sigma_,
                            const double mod_grad_alpha1_min_,
                            const double lambda_,
                            const double atol_Newton_,
                            const double rtol_Newton_,
                            const std::size_t max_Newton_iters_):
    Flux<Field>(EOS_phase1_, EOS_phase2_, sigma_, mod_grad_alpha1_min_, lambda_, atol_Newton_, rtol_Newton_, max_Newton_iters_) {}

  // Implement the auxliary routine that computes the middle state
  //
  template<class Field>
  auto HLLCFlux<Field>::compute_middle_state(const FluxValue<typename Flux<Field>::cfg>& q,
                                             const typename Field::value_type S,
                                             const typename Field::value_type S_star,
                                             const std::size_t curr_d) const {
    /*--- Save velocity current direction ---*/
    const auto rho   = q(M1_INDEX) + q(M2_INDEX);
    const auto vel_d = q(RHO_U_INDEX + curr_d)/rho;

    /*--- Compute middle state ---*/
    FluxValue<typename Flux<Field>::cfg> q_star;

    q_star(M1_INDEX)             = q(M1_INDEX)*((S - vel_d)/(S - S_star));
    q_star(M2_INDEX)             = q(M2_INDEX)*((S - vel_d)/(S - S_star));
    const auto rho_star          = q_star(M1_INDEX) + q_star(M2_INDEX);
    q_star(RHO_ALPHA1_INDEX)     = rho_star*(q(RHO_ALPHA1_INDEX)/rho);
    q_star(RHO_U_INDEX + curr_d) = rho_star*S_star;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      if(d != curr_d) {
        q_star(RHO_U_INDEX + d) = rho_star*(q(RHO_U_INDEX + d)/rho);
      }
    }

    return q_star;
  }

  // Implementation of a HLLC flux
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> HLLCFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                                              const FluxValue<typename Flux<Field>::cfg>& qR,
                                                                              const std::size_t curr_d) {
    /*--- Verify if left and right state are coherent ---*/
    #ifdef VERBOSE_FLUX
      if(qL(M1_INDEX) < 0.0) {
        throw std::runtime_error(std::string("Negative mass phase 1 left state: " + std::to_string(qL(M1_INDEX))));
      }
      if(qL(M2_INDEX) < 0.0) {
        throw std::runtime_error(std::string("Negative mass phase 2 left state: " + std::to_string(qL(M2_INDEX))));
      }
      if(qL(RHO_ALPHA1_INDEX) < 0.0) {
        throw std::runtime_error(std::string("Negative volume fraction phase 1 left state: " + std::to_string(qL(RHO_ALPHA1_INDEX))));
      }

      if(qR(M1_INDEX) < 0.0) {
        throw std::runtime_error(std::string("Negative mass phase 1 right state: " + std::to_string(qR(M1_INDEX))));
      }
      if(qR(M2_INDEX) < 0.0) {
        throw std::runtime_error(std::string("Negative mass phase 2 right state: " + std::to_string(qR(M2_INDEX))));
      }
      if(qR(RHO_ALPHA1_INDEX) < 0.0) {
        throw std::runtime_error(std::string("Negative volume fraction phase 1 left state: " + std::to_string(qL(RHO_ALPHA1_INDEX))));
      }
    #endif

    /*--- Compute the quantities needed for the maximum eigenvalue estimate for the left state ---*/
    const auto rho_L          = qL(M1_INDEX) + qL(M2_INDEX);
    const auto vel_d_L        = qL(RHO_U_INDEX + curr_d)/rho_L;

    const auto alpha1_L       = qL(RHO_ALPHA1_INDEX)/rho_L;
    const auto rho1_L         = qL(M1_INDEX)/alpha1_L; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rho2_L         = qL(M2_INDEX)/(1.0 - alpha1_L); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rhoc_squared_L = qL(M1_INDEX)*this->EOS_phase1.c_value(rho1_L)*this->EOS_phase1.c_value(rho1_L)
                              + qL(M2_INDEX)*this->EOS_phase2.c_value(rho2_L)*this->EOS_phase2.c_value(rho2_L);
    #ifdef VERBOSE_FLUX
      if(rho_L < 0.0) {
        throw std::runtime_error(std::string("Negative density left state: " + std::to_string(rho_L)));
      }
      if(rhoc_squared_L/rho_L < 0.0) {
        throw std::runtime_error(std::string("Negative square speed of sound left state: " + std::to_string(rhoc_squared_L/rho_L)));
      }
    #endif
    const auto c_L = std::sqrt(rhoc_squared_L/rho_L);

    /*--- Compute the quantities needed for the maximum eigenvalue estimate for the right state ---*/
    const auto rho_R          = qR(M1_INDEX) + qR(M2_INDEX);
    const auto vel_d_R        = qR(RHO_U_INDEX + curr_d)/rho_R;

    const auto alpha1_R       = qR(RHO_ALPHA1_INDEX)/rho_R;;
    const auto rho1_R         = qR(M1_INDEX)/alpha1_R; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rho2_R         = qR(M2_INDEX)/(1.0 - alpha1_R); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rhoc_squared_R = qR(M1_INDEX)*this->EOS_phase1.c_value(rho1_R)*this->EOS_phase1.c_value(rho1_R)
                              + qR(M2_INDEX)*this->EOS_phase2.c_value(rho2_R)*this->EOS_phase2.c_value(rho2_R);
    #ifdef VERBOSE_FLUX
      if(rho_R < 0.0) {
        throw std::runtime_error(std::string("Negative density right state: " + std::to_string(rho_R)));
      }
      if(rhoc_squared_R/rho_R < 0.0) {
        throw std::runtime_error(std::string("Negative square speed of sound right state: " + std::to_string(rhoc_squared_R/rho_R)));
      }
    #endif
    const auto c_R = std::sqrt(rhoc_squared_R/rho_R);

    /*--- Compute speeds of wave propagation ---*/
    const auto s_L    = std::min(vel_d_L - c_L, vel_d_R - c_R);
    const auto s_R    = std::max(vel_d_L + c_L, vel_d_R + c_R);
    const auto p_L    = alpha1_L*this->EOS_phase1.pres_value(rho1_L)
                      + (1.0 - alpha1_L)*this->EOS_phase2.pres_value(rho2_L);
    const auto p_R    = alpha1_R*this->EOS_phase1.pres_value(rho1_R)
                      + (1.0 - alpha1_R)*this->EOS_phase2.pres_value(rho2_R);
    const auto s_star = (p_R - p_L + rho_L*vel_d_L*(s_L - vel_d_L) - rho_R*vel_d_R*(s_R - vel_d_R))/
                        (rho_L*(s_L - vel_d_L) - rho_R*(s_R - vel_d_R));

    /*--- Compute intermediate states ---*/
    const auto q_star_L = compute_middle_state(qL, s_L, s_star, curr_d);
    const auto q_star_R = compute_middle_state(qR, s_R, s_star, curr_d);

    /*--- Compute the flux ---*/
    if(s_L >= 0.0) {
      return this->evaluate_hyperbolic_operator(qL, curr_d);
    }
    else if(s_L < 0.0 && s_star >= 0.0) {
      return this->evaluate_hyperbolic_operator(qL, curr_d) + s_L*(q_star_L - qL);
    }
    else if(s_star < 0.0 && s_R >= 0.0) {
      return this->evaluate_hyperbolic_operator(qR, curr_d) + s_R*(q_star_R - qR);
    }
    else if(s_R < 0.0) {
      return this->evaluate_hyperbolic_operator(qR, curr_d);
    }
  }

  // Implement the contribution of the discrete flux for all the directions.
  //
  template<class Field>
  #ifdef ORDER_2
    template<typename Field_Scalar>
    auto HLLCFlux<Field>::make_flux(const Field_Scalar& H)
  #else
    auto HLLCFlux<Field>::make_flux()
  #endif
  {
    FluxDefinition<typename Flux<Field>::cfg> HLLC_f;

    /*--- Perform the loop over each dimension to compute the flux contribution ---*/
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        // Compute now the "discrete" flux function, in this case a HLLC flux
        HLLC_f[d].cons_flux_function = [&](samurai::FluxValue<typename Flux<Field>::cfg>& flux,
                                           const StencilData<typename Flux<Field>::cfg>& data,
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

                                               FluxValue<typename Flux<Field>::cfg> qL = this->prim2cons(primL_recon);
                                               FluxValue<typename Flux<Field>::cfg> qR = this->prim2cons(primR_recon);

                                               #ifdef RELAX_RECONSTRUCTION
                                                 this->relax_reconstruction(qL, H[data.cells[1]][0]);
                                                 this->relax_reconstruction(qR, H[data.cells[2]][0]);
                                               #endif
                                             #else
                                               // Extract the states
                                               const FluxValue<typename Flux<Field>::cfg> qL = field[0];
                                               const FluxValue<typename Flux<Field>::cfg> qR = field[1];
                                             #endif

                                             flux = compute_discrete_flux(qL, qR, d);
                                           };
    });

    auto scheme = make_flux_based_scheme(HLLC_f);
    scheme.set_name("HLLC");

    return scheme;
  }

} // end of namespace

#endif
