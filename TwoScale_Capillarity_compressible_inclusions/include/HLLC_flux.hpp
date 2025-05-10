// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
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
             const double sigma_,
             const double mod_grad_alpha1_min_,
             const double lambda_,
             const double atol_Newton_,
             const double rtol_Newton_,
             const std::size_t max_Newton_iters_); /*--- Constructor which accepts in inputs the equations of state of the two phases ---*/

    #ifdef ORDER_2
      template<typename Field_Scalar>
      auto make_two_scale_capillarity(const Field_Scalar& H); /*--- Compute the flux over all the directions ---*/
    #else
      auto make_two_scale_capillarity(); /*--- Compute the flux over all the directions ---*/
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
    Flux<Field>(EOS_phase1_, EOS_phase2_,
                sigma_, mod_grad_alpha1_min_,
                lambda_, atol_Newton_, rtol_Newton_, max_Newton_iters_) {}

  // Implement the auxliary routine that computes the middle state
  //
  template<class Field>
  auto HLLCFlux<Field>::compute_middle_state(const FluxValue<typename Flux<Field>::cfg>& q,
                                             const typename Field::value_type S,
                                             const typename Field::value_type S_star,
                                             const std::size_t curr_d) const {
    /*--- Save velocity current direction ---*/
    const auto rho   = q(M1_INDEX) + q(M2_INDEX) + q(M1_D_INDEX);
    const auto vel_d = q(RHO_U_INDEX + curr_d)/rho;

    /*--- Compute middle state ---*/
    FluxValue<typename Flux<Field>::cfg> q_star;

    q_star(M1_INDEX)             = q(M1_INDEX)*((S - vel_d)/(S - S_star));
    q_star(M2_INDEX)             = q(M2_INDEX)*((S - vel_d)/(S - S_star));
    q_star(M1_D_INDEX)           = q(M1_D_INDEX)*((S - vel_d)/(S - S_star));
    const auto rho_star          = q_star(M1_INDEX) + q_star(M2_INDEX) + q_star(M1_D_INDEX);
    q_star(RHO_ALPHA1_INDEX)     = rho_star*(q(RHO_ALPHA1_INDEX)/rho);
    q_star(RHO_Z_INDEX)          = rho_star*(q(RHO_Z_INDEX)/rho);
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
        throw std::runtime_error(std::string("Negative mass large-scale liquid left state: " + std::to_string(qL(M1_INDEX))));
      }
      if(qL(M2_INDEX) < 0.0) {
        throw std::runtime_error(std::string("Negative mass gas left state: " + std::to_string(qL(M2_INDEX))));
      }
      if(qL(M1_D_INDEX) < -1e-15) {
        throw std::runtime_error(std::string("Negative mass small-scale liquid left state: " + std::to_string(qL(M1_D_INDEX))));
      }
      if(qL(RHO_ALPHA1_INDEX) < 0.0) {
        throw std::runtime_error(std::string("Negative volume fraction large-scale liquid left state: " + std::to_string(qL(RHO_ALPHA1_INDEX))));
      }
      if(qL(RHO_Z_INDEX) < -1e-15) {
        throw std::runtime_error(std::string("Negative interface area small-scale liquid left state: " + std::to_string(qL(RHO_Z_INDEX))));
      }

      if(qR(M1_INDEX) < 0.0) {
        throw std::runtime_error(std::string("Negative mass large-scale liquid right state: " + std::to_string(qR(M1_INDEX))));
      }
      if(qR(M2_INDEX) < 0.0) {
        throw std::runtime_error(std::string("Negative mass gas right state: " + std::to_string(qR(M2_INDEX))));
      }
      if(qR(M1_D_INDEX) < -1e-15) {
        throw std::runtime_error(std::string("Negative mass small-scale liquid right state: " + std::to_string(qR(M1_D_INDEX))));
      }
      if(qR(RHO_ALPHA1_INDEX) < 0.0) {
        throw std::runtime_error(std::string("Negative volume fraction large-scale liquid right state: " + std::to_string(qR(RHO_ALPHA1_INDEX))));
      }
      if(qR(RHO_Z_INDEX) < -1e-15) {
        throw std::runtime_error(std::string("Negative interface area small-scale liquid right state: " + std::to_string(qR(RHO_Z_INDEX))));
      }
    #endif

    /*--- Compute the quantities needed for the maximum eigenvalue estimate for the left state ---*/
    const auto rho_L          = qL(M1_INDEX) + qL(M2_INDEX) + qL(M1_D_INDEX);
    const auto vel_d_L        = qL(RHO_U_INDEX + curr_d)/rho_L;

    const auto alpha1_L       = qL(RHO_ALPHA1_INDEX)/rho_L;
    const auto rho1_L         = qL(M1_INDEX)/alpha1_L; /*--- TODO: Add a check in case of zero volume fraction ---*/
    typename Field::value_type rho1_d_L;
    try {
      rho1_d_L = compute_rho1_d_local_Laplace<Field>(qL(M1_D_INDEX), qL(M2_INDEX), qL(M1_INDEX), alpha1_L, qL(RHO_Z_INDEX),
                                                     this->sigma, this->EOS_phase1, this->EOS_phase2,
                                                     this->atol_Newton, this->rtol_Newton, this->max_Newton_iters, this->lambda);
    }
    catch(const std::exception& e) {
      std::cerr << "Small-scale error when computing rho1_d_L" << std::endl;
      std::cout << qL << std::endl;
      throw std::runtime_error(e.what());
    }
    const auto alpha1_d_L     = qL(M1_D_INDEX)/rho1_d_L;
    const auto alpha2_L       = 1.0 - alpha1_L - alpha1_d_L;
    const auto rho2_L         = qL(M2_INDEX)/alpha2_L; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rhoc_squared_L = qL(M1_INDEX)*this->EOS_phase1.c_value(rho1_L)*this->EOS_phase1.c_value(rho1_L)
                              + ((1.0 - alpha1_L)/(alpha2_L))*((1.0 - alpha1_L)/(alpha2_L))*
                                qL(M2_INDEX)*this->EOS_phase2.c_value(rho2_L)*this->EOS_phase2.c_value(rho2_L);
    const auto c_L            = std::sqrt(rhoc_squared_L/rho_L);

    /*--- Compute the quantities needed for the maximum eigenvalue estimate for the right state ---*/
    const auto rho_R          = qR(M1_INDEX) + qR(M2_INDEX) + qR(M1_D_INDEX);
    const auto vel_d_R        = qR(RHO_U_INDEX + curr_d)/rho_R;

    const auto alpha1_R       = qR(RHO_ALPHA1_INDEX)/rho_R;
    const auto rho1_R         = qR(M1_INDEX)/alpha1_R; /*--- TODO: Add a check in case of zero volume fraction ---*/
    typename Field::value_type rho1_d_R;
    try {
      rho1_d_R = compute_rho1_d_local_Laplace<Field>(qR(M1_D_INDEX), qR(M2_INDEX), qR(M1_INDEX), alpha1_R, qR(RHO_Z_INDEX),
                                                     this->sigma, this->EOS_phase1, this->EOS_phase2,
                                                     this->atol_Newton, this->rtol_Newton, this->max_Newton_iters, this->lambda);
    }
    catch(const std::exception& e) {
      std::cerr << "Small-scale error when computing rho1_d_R" << std::endl;
      std::cerr << qR << std::endl;
      throw std::runtime_error(e.what());
    }
    const auto alpha1_d_R     = qR(M1_D_INDEX)/rho1_d_R;
    const auto alpha2_R       = 1.0 - alpha1_R - alpha1_d_R;
    const auto rho2_R         = qR(M2_INDEX)/alpha2_R; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rhoc_squared_R = qR(M1_INDEX)*this->EOS_phase1.c_value(rho1_R)*this->EOS_phase1.c_value(rho1_R)
                              + ((1.0 - alpha1_R)/(alpha2_R))*((1.0 - alpha1_R)/(alpha2_R))*
                                qR(M2_INDEX)*this->EOS_phase2.c_value(rho2_R)*this->EOS_phase2.c_value(rho2_R);
    const auto c_R            = std::sqrt(rhoc_squared_R/rho_R);

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

    if(q_star_L(M1_D_INDEX) < -1e-15) {
      throw std::runtime_error("Negative mass small-scale left star state");
    }
    if(q_star_R(M1_D_INDEX) < -1e-15) {
      throw std::runtime_error("Negative mass small-scale right star state");
    }

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
    auto HLLCFlux<Field>::make_two_scale_capillarity(const Field_Scalar& H)
  #else
    auto HLLCFlux<Field>::make_two_scale_capillarity()
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
                                               /*FluxValue<typename Flux<Field>::cfg> primLL;
                                               try {
                                                 primLL = this->cons2prim(field[0]);
                                               }
                                               catch(const std::exception& e) {
                                                 std::cout << e.what() << std::endl;
                                                 std::cout << data.cells[0] << std::endl;
                                               }
                                               FluxValue<typename Flux<Field>::cfg> primL;
                                               try {
                                                 primL = this->cons2prim(field[1]);
                                               }
                                               catch(const std::exception& e) {
                                                 std::cout << e.what() << std::endl;
                                                 std::cout << data.cells[1] << std::endl;
                                               }
                                               FluxValue<typename Flux<Field>::cfg> primR;
                                               try {
                                                 primR = this->cons2prim(field[2]);
                                               }
                                               catch(const std::exception& e) {
                                                 std::cout << e.what() << std::endl;
                                                 std::cout << data.cells[2] << std::endl;
                                               }
                                               FluxValue<typename Flux<Field>::cfg> primRR;
                                               try {
                                                 primRR = this->cons2prim(field[3]);
                                               }
                                               catch(const std::exception& e) {
                                                 std::cout << e.what() << std::endl;
                                                 std::cout << data.cells[3] << std::endl;
                                               }*/
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
