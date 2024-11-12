// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#ifndef Rusanov_flux_hpp
#define Rusanov_flux_hpp

#include "flux_base.hpp"

//#define VERBOSE_FLUX

namespace samurai {
  using namespace EquationData;

  /**
    * Implementation of a Rusanov flux
    */
  template<class Field>
  class RusanovFlux: public Flux<Field> {
  public:
    RusanovFlux(const LinearizedBarotropicEOS<>& EOS_phase1,
                const LinearizedBarotropicEOS<>& EOS_phase2,
                const double sigma_,
                const double eps_,
                const double mod_grad_alpha1_min_); // Constructor which accepts in inputs the equations of state of the two phases

    #ifdef ORDER_2
      template<typename Field_Scalar>
      auto make_two_scale_capillarity(const Field_Scalar& H); // Compute the flux over all the directions
    #else
      auto make_two_scale_capillarity(); // Compute the flux over all the directions
    #endif

  private:
    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                               const FluxValue<typename Flux<Field>::cfg>& qR,
                                                               const std::size_t curr_d); // Rusanov flux along direction curr_d
  };

  // Constructor derived from the base class
  //
  template<class Field>
  RusanovFlux<Field>::RusanovFlux(const LinearizedBarotropicEOS<>& EOS_phase1,
                                  const LinearizedBarotropicEOS<>& EOS_phase2,
                                  const double sigma_,
                                  const double eps_,
                                  const double grad_alpha1_min_):
    Flux<Field>(EOS_phase1, EOS_phase2, sigma_, eps_, grad_alpha1_min_) {}

  // Implementation of a Rusanov flux
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> RusanovFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                                                 const FluxValue<typename Flux<Field>::cfg>& qR,
                                                                                 const std::size_t curr_d) {
    // Verify if left and right state are coherent
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

    // Compute the quantities needed for the maximum eigenvalue estimate for the left state
    const auto rho_L       = qL(M1_INDEX) + qL(M2_INDEX);
    const auto vel_d_L     = qL(RHO_U_INDEX + curr_d)/rho_L;

    const auto alpha1_L    = qL(RHO_ALPHA1_INDEX)/rho_L;;
    const auto rho1_L      = qL(M1_INDEX)/alpha1_L;
    const auto alpha2_L    = 1.0 - alpha1_L;
    const auto rho2_L      = qL(M2_INDEX)/alpha2_L;
    const auto c_squared_L = qL(M1_INDEX)*this->phase1.c_value(rho1_L)*this->phase1.c_value(rho1_L)
                           + qL(M2_INDEX)*this->phase2.c_value(rho2_L)*this->phase2.c_value(rho2_L);
    #ifdef VERBOSE_FLUX
      if(rho_L < 0.0) {
        throw std::runtime_error(std::string("Negative density left state: " + std::to_string(rho_L)));
      }
      if(c_squared_L/rho_L < 0.0) {
        throw std::runtime_error(std::string("Negative square speed of sound left state: " + std::to_string(c_squared_L/rho_L)));
      }
    #endif
    const auto c_L = std::sqrt(c_squared_L/rho_L);

    // Compute the quantities needed for the maximum eigenvalue estimate for the right state
    const auto rho_R       = qR(M1_INDEX) + qR(M2_INDEX);
    const auto vel_d_R     = qR(RHO_U_INDEX + curr_d)/rho_R;

    const auto alpha1_R    = qR(RHO_ALPHA1_INDEX)/rho_R;;
    const auto rho1_R      = qR(M1_INDEX)/alpha1_R;
    const auto alpha2_R    = 1.0 - alpha1_R;
    const auto rho2_R      = qR(M2_INDEX)/alpha2_R;
    const auto c_squared_R = qR(M1_INDEX)*this->phase1.c_value(rho1_R)*this->phase1.c_value(rho1_R)
                           + qR(M2_INDEX)*this->phase2.c_value(rho2_R)*this->phase2.c_value(rho2_R);
    #ifdef VERBOSE_FLUX
      if(rho_R < 0.0) {
        throw std::runtime_error(std::string("Negative density right state: " + std::to_string(rho_R)));
      }
      if(c_squared_R/rho_R < 0.0) {
        throw std::runtime_error(std::string("Negative square speed of sound right state: " + std::to_string(c_squared_R/rho_R)));
      }
    #endif
    const auto c_R = std::sqrt(c_squared_R/rho_R);

    // Compute the estimate of the eigenvalue considering also the surface tension contribution
    const auto lambda = std::max(std::abs(vel_d_L) + c_L,
                                 std::abs(vel_d_R) + c_R);

    return 0.5*(this->evaluate_hyperbolic_operator(qL, curr_d) +
                this->evaluate_hyperbolic_operator(qR, curr_d)) - // centered contribution
           0.5*lambda*(qR - qL); // upwinding contribution
  }

  // Implement the contribution of the discrete flux for all the directions.
  //
  template<class Field>
  #ifdef ORDER_2
    template<typename Field_Scalar>
    auto RusanovFlux<Field>::make_two_scale_capillarity(const Field_Scalar& H)
  #else
    auto RusanovFlux<Field>::make_two_scale_capillarity()
  #endif
  {
    FluxDefinition<typename Flux<Field>::cfg> Rusanov_f;

    // Perform the loop over each dimension to compute the flux contribution
    static_for<0, EquationData::dim>::apply(
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        // Compute now the "discrete" flux function, in this case a Rusanov flux
        Rusanov_f[d].cons_flux_function = [&](auto& cells, const Field& field)
                                          {
                                            #ifdef ORDER_2
                                              // Compute the stencil
                                              const auto& left_left   = cells[0];
                                              const auto& left        = cells[1];
                                              const auto& right       = cells[2];
                                              const auto& right_right = cells[3];

                                              // MUSCL reconstruction
                                              const FluxValue<typename Flux<Field>::cfg> primLL = this->cons2prim(field[left_left]);
                                              const FluxValue<typename Flux<Field>::cfg> primL  = this->cons2prim(field[left]);
                                              const FluxValue<typename Flux<Field>::cfg> primR  = this->cons2prim(field[right]);
                                              const FluxValue<typename Flux<Field>::cfg> primRR = this->cons2prim(field[right_right]);

                                              FluxValue<typename Flux<Field>::cfg> primL_recon,
                                                                                   primR_recon;
                                              this->perform_reconstruction(primLL, primL, primR, primRR,
                                                                           primL_recon, primR_recon);

                                              FluxValue<typename Flux<Field>::cfg> qL = this->prim2cons(primL_recon);
                                              FluxValue<typename Flux<Field>::cfg> qR = this->prim2cons(primR_recon);

                                              #ifdef RELAX_RECONSTRUCTION
                                                this->relax_reconstruction(qL, H[left]);
                                                this->relax_reconstruction(qR, H[right]);
                                              #endif
                                            #else
                                              // Compute the stencil and extract state
                                              const auto& left  = cells[0];
                                              const auto& right = cells[1];

                                              const FluxValue<typename Flux<Field>::cfg> qL = field[left];
                                              const FluxValue<typename Flux<Field>::cfg> qR = field[right];
                                            #endif

                                            // Compute the numerical flux
                                            return compute_discrete_flux(qL, qR, d);
                                          };
    });

    return make_flux_based_scheme(Rusanov_f);
  }

} // end of namespace

#endif
