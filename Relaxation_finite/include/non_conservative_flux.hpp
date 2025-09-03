// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#ifndef non_conservative_flux_hpp
#define non_conservative_flux_hpp

//#define BR
#define CENTERED

//#define PERFORM_RECON

namespace samurai {
  using namespace EquationData;

  /**
    * Implementation of the non-conservative flux
    */
  template<class Field>
  class NonConservativeFlux: public Flux<Field> {
  public:
    using Number = Flux<Field>::Number; /*--- Shortcut for the arithmetic type ---*/

    NonConservativeFlux(const EOS<Number>& EOS_phase1_,
                        const EOS<Number>& EOS_phase2_); /*--- Constructor which accepts in input
                                                               the equations of state of the two phases ---*/

    auto make_flux(); /*--- Compute the flux over all the faces and directions ---*/

  private:
    void compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                               const FluxValue<typename Flux<Field>::cfg>& qR,
                               const std::size_t curr_d,
                               FluxValue<typename Flux<Field>::cfg>& F_minus,
                               FluxValue<typename Flux<Field>::cfg>& F_plus); /*--- Non-conservative flux ---*/
  };

  // Constructor derived from base class
  //
  template<class Field>
  NonConservativeFlux<Field>::NonConservativeFlux(const EOS<Number>& EOS_phase1_,
                                                  const EOS<Number>& EOS_phase2_):
    Flux<Field>(EOS_phase1_, EOS_phase2_) {}

  // Implementation of a non-conservative flux
  //
  template<class Field>
  void NonConservativeFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                         const FluxValue<typename Flux<Field>::cfg>& qR,
                                                         const std::size_t curr_d,
                                                         FluxValue<typename Flux<Field>::cfg>& F_minus,
                                                         FluxValue<typename Flux<Field>::cfg>& F_plus) {
    /*--- Zero contribution from continuity equations ---*/
    F_minus(ALPHA1_RHO1_INDEX) = static_cast<Number>(0.0);
    F_minus(ALPHA2_RHO2_INDEX) = static_cast<Number>(0.0);
    F_plus(ALPHA1_RHO1_INDEX)  = static_cast<Number>(0.0);
    F_plus(ALPHA2_RHO2_INDEX)  = static_cast<Number>(0.0);

    /*--- Left state ---*/
    // Pre-fetch variables that will be used several times so as to exploit possible vectorization
    // (as well as to enhance readability)
    const auto alpha1_L = qL(ALPHA1_INDEX);
    const auto m1_L     = qL(ALPHA1_RHO1_INDEX);
    const auto m2_L     = qL(ALPHA2_RHO2_INDEX);
    const auto m2E2_L   = qL(ALPHA2_RHO2_E2_INDEX);

    // Interface velocity and interface pressure computed from left state
    const auto inv_m1_L = static_cast<Number>(1.0)/m1_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto inv_m2_L = static_cast<Number>(1.0)/m2_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto velI_L   = qL(ALPHA1_RHO1_U1_INDEX + curr_d)*inv_m1_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho2_L   = m2_L/(static_cast<Number>(1.0) - alpha1_L); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2_L           = m2E2_L*inv_m2_L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < Field::dim; ++d) {
      e2_L -= static_cast<Number>(0.5)*
              ((qL(ALPHA2_RHO2_U2_INDEX + d)*inv_m2_L)*
               (qL(ALPHA2_RHO2_U2_INDEX + d)*inv_m2_L)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pI_L = this->EOS_phase2.pres_value_Rhoe(rho2_L, e2_L);

    /*--- Right state ---*/
    // Pre-fetch variables that will be used several times so as to exploit possible vectorization
    // (as well as to enhance readability)
    const auto alpha1_R = qR(ALPHA1_INDEX);
    const auto m1_R     = qR(ALPHA1_RHO1_INDEX);
    const auto m2_R     = qR(ALPHA2_RHO2_INDEX);
    const auto m2E2_R   = qR(ALPHA2_RHO2_E2_INDEX);

    // Interface velocity and interface pressure computed from right state
    const auto inv_m1_R = static_cast<Number>(1.0)/m1_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto inv_m2_R = static_cast<Number>(1.0)/m2_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto velI_R   = qR(ALPHA1_RHO1_U1_INDEX + curr_d)*inv_m1_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho2_R   = m2_R/(static_cast<Number>(1.0) - alpha1_R); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2_R           = m2E2_R*inv_m2_R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < Field::dim; ++d) {
      e2_R -= static_cast<Number>(0.5)*
              ((qR(ALPHA2_RHO2_U2_INDEX + d)*inv_m2_R)*
               (qR(ALPHA2_RHO2_U2_INDEX + d)*inv_m2_R)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pI_R = this->EOS_phase2.pres_value_Rhoe(rho2_R, e2_R);

    /*--- Build the non conservative flux ---*/
    #ifdef BR
      F_minus(ALPHA1_INDEX) = static_cast<Number>(0.5)*
                              (velI_L*alpha1_L + velI_R*alpha1_R) -
                              static_cast<Number>(0.5)*
                              (velI_L + velI_R)*alpha1_L;
      F_plus(ALPHA1_INDEX)  = static_cast<Number>(0.5)*
                              (velI_L*alpha1_L + velI_R*alpha1_R) -
                              static_cast<Number>(0.5)*
                              (velI_L + velI_R)*alpha1_R;

      F_minus(ALPHA1_RHO1_U1_INDEX + curr_d) = -(static_cast<Number>(0.5)*
                                                 (pI_L*alpha1_L + pI_R*alpha1_R) -
                                                 static_cast<Number>(0.5)*
                                                 (pI_L + pI_R)*alpha1_L);
      F_plus(ALPHA1_RHO1_U1_INDEX + curr_d)  = -(static_cast<Number>(0.5)*
                                                 (pI_L*alpha1_L + pI_R*alpha1_R) -
                                                 static_cast<Number>(0.5)*
                                                 (pI_L + pI_R)*alpha1_R);

      F_minus(ALPHA1_RHO1_E1_INDEX) = -(static_cast<Number>(0.5)*
                                        (pI_L*velI_L*alpha1_L + pI_R*velI_R*alpha1_R) -
                                        static_cast<Number>(0.5)*
                                        (pI_L*velI_L + pI_R*velI_R)*alpha1_L);
      F_plus(ALPHA1_RHO1_E1_INDEX)  = -(static_cast<Number>(0.5)*
                                        (pI_L*velI_L*alpha1_L + pI_R*velI_R*alpha1_R) -
                                        static_cast<Number>(0.5)*
                                        (pI_L*velI_L + pI_R*velI_R)*alpha1_R);
    #elifdef CENTERED
      F_minus(ALPHA1_INDEX) = velI_L*
                              (static_cast<Number>(0.5)*
                               (alpha1_L + alpha1_R));
      F_plus(ALPHA1_INDEX)  = velI_R*
                              (static_cast<Number>(0.5)*
                               (alpha1_L + alpha1_R));

      F_minus(ALPHA1_RHO1_U1_INDEX + curr_d) = -pI_L*
                                                (static_cast<Number>(0.5)*
                                                 (alpha1_L + alpha1_R));
      F_plus(ALPHA1_RHO1_U1_INDEX + curr_d)  = -pI_R*
                                                (static_cast<Number>(0.5)*
                                                 (alpha1_L + alpha1_R));

      F_minus(ALPHA1_RHO1_E1_INDEX) = -velI_L*pI_L*
                                       (static_cast<Number>(0.5)*
                                        (alpha1_L + alpha1_R));
      F_plus(ALPHA1_RHO1_E1_INDEX)  = -velI_R*pI_R*
                                       (static_cast<Number>(0.5)*
                                        (alpha1_L + alpha1_R));
    #endif

    F_minus(ALPHA2_RHO2_U2_INDEX + curr_d) = -F_minus(ALPHA1_RHO1_U1_INDEX + curr_d);
    F_plus(ALPHA2_RHO2_U2_INDEX + curr_d)  = -F_plus(ALPHA1_RHO1_U1_INDEX + curr_d);

    F_minus(ALPHA2_RHO2_E2_INDEX) = -F_minus(ALPHA1_RHO1_E1_INDEX);
    F_plus(ALPHA2_RHO2_E2_INDEX)  = -F_plus(ALPHA1_RHO1_E1_INDEX);
  }

  // Implement the contribution of the discrete flux for all the dimensions.
  //
  template<class Field>
  auto NonConservativeFlux<Field>::make_flux() {
    FluxDefinition<typename Flux<Field>::cfg> discrete_flux;

    /*--- Perform the loop over each dimension to compute the flux contribution ---*/
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
         {
           static constexpr int d = decltype(integral_constant_d)::value;

           // Compute now the "discrete" non-conservative flux function
           discrete_flux[d].flux_function = [&](samurai::FluxValuePair<typename Flux<Field>::cfg>& flux,
                                                const StencilData<typename Flux<Field>::cfg>& /*data*/,
                                                const StencilValues<typename Flux<Field>::cfg> field)
                                                {
                                                  #ifdef ORDER_2
                                                    #ifdef PERFORM_RECON
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
                                                    #else
                                                      // Extract the states
                                                      const FluxValue<typename Flux<Field>::cfg>& qL = field[1];
                                                      const FluxValue<typename Flux<Field>::cfg>& qR = field[2];
                                                    #endif
                                                  #else
                                                    // Extract the states
                                                    const FluxValue<typename Flux<Field>::cfg>& qL = field[0];
                                                    const FluxValue<typename Flux<Field>::cfg>& qR = field[1];
                                                  #endif

                                                  FluxValue<typename Flux<Field>::cfg> F_minus,
                                                                                       F_plus;

                                                  compute_discrete_flux(qL, qR, d, F_minus, F_plus);

                                                  flux[0] = F_minus;
                                                  flux[1] = -F_plus;
                                                };
        }
    );

    auto scheme = make_flux_based_scheme(discrete_flux);
    scheme.set_name("Non conservative");

    return scheme;
  }

} // end of namespace

#endif
