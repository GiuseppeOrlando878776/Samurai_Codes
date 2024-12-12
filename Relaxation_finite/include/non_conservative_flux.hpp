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
    NonConservativeFlux(const EOS<>& EOS_phase1, const EOS<>& EOS_phase2); // Constructor which accepts in inputs the equations of state of the two phases

    auto make_flux(); // Compute the flux over all cells

  private:
    void compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                               const FluxValue<typename Flux<Field>::cfg>& qR,
                               const std::size_t curr_d,
                               FluxValue<typename Flux<Field>::cfg>& F_minus,
                               FluxValue<typename Flux<Field>::cfg>& F_plus); // Non-conservative flux
  };

  // Constructor derived from base class
  //
  template<class Field>
  NonConservativeFlux<Field>::NonConservativeFlux(const EOS<>& EOS_phase1, const EOS<>& EOS_phase2):
    Flux<Field>(EOS_phase1, EOS_phase2) {}

  // Implementation of a non-conservative flux from left to right
  //
  template<class Field>
  void NonConservativeFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                         const FluxValue<typename Flux<Field>::cfg>& qR,
                                                         const std::size_t curr_d,
                                                         FluxValue<typename Flux<Field>::cfg>& F_minus,
                                                         FluxValue<typename Flux<Field>::cfg>& F_plus) {
    // Zero contribution from continuity equations
    F_minus(ALPHA1_RHO1_INDEX) = 0.0;
    F_minus(ALPHA2_RHO2_INDEX) = 0.0;
    F_plus(ALPHA1_RHO1_INDEX)  = 0.0;
    F_plus(ALPHA2_RHO2_INDEX)  = 0.0;

    // Interfacial velocity and interfacial pressure computed from left state
    const auto velIL = qL(ALPHA1_RHO1_U1_INDEX + curr_d)/qL(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho2L = qL(ALPHA2_RHO2_INDEX)/(1.0 - qL(ALPHA1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2L         = qL(ALPHA2_RHO2_E2_INDEX)/qL(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < Field::dim; ++d) {
      e2L -= 0.5*((qL(ALPHA2_RHO2_U2_INDEX + d)/qL(ALPHA2_RHO2_INDEX))*
                  (qL(ALPHA2_RHO2_U2_INDEX + d)/qL(ALPHA2_RHO2_INDEX))); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pIL   = this->phase2.pres_value_Rhoe(rho2L, e2L);

    // Interfacial velocity and interfacial pressure computed from right state
    const auto velIR = qR(ALPHA1_RHO1_U1_INDEX + curr_d)/qR(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho2R = qR(ALPHA2_RHO2_INDEX)/(1.0 - qR(ALPHA1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2R         = qR(ALPHA2_RHO2_E2_INDEX)/qR(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < Field::dim; ++d) {
      e2R -= 0.5*((qR(ALPHA2_RHO2_U2_INDEX + d)/qR(ALPHA2_RHO2_INDEX))*
                  (qR(ALPHA2_RHO2_U2_INDEX + d)/qR(ALPHA2_RHO2_INDEX))); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pIR   = this->phase2.pres_value_Rhoe(rho2R, e2R);

    /*--- Build the non conservative flux ---*/
    #ifdef BR
      F_minus(ALPHA1_INDEX) = (0.5*(velIL*qL(ALPHA1_INDEX) + velIR*qR(ALPHA1_INDEX)) -
                               0.5*(velIL + velIR)*qL(ALPHA1_INDEX));
      F_plus(ALPHA1_INDEX) = (0.5*(velIL*qL(ALPHA1_INDEX) + velIR*qR(ALPHA1_INDEX)) -
                              0.5*(velIL + velIR)*qR(ALPHA1_INDEX));

      F_minus(ALPHA1_RHO1_U1_INDEX + curr_d) = -(0.5*(pIL*qL(ALPHA1_INDEX) + pIR*qR(ALPHA1_INDEX)) -
                                                 0.5*(pIL + pIR)*qL(ALPHA1_INDEX));
      F_plus(ALPHA1_RHO1_U1_INDEX + curr_d)  = -(0.5*(pIL*qL(ALPHA1_INDEX) + pIR*qR(ALPHA1_INDEX)) -
                                                 0.5*(pIL + pIR)*qR(ALPHA1_INDEX));

      F_minus(ALPHA1_RHO1_E1_INDEX) = -(0.5*(pIL*velIL*qL(ALPHA1_INDEX) + pIR*velIR*qR(ALPHA1_INDEX)) -
                                        0.5*(pIL*velIL + pIR*velIR)*qL(ALPHA1_INDEX));
      F_plus(ALPHA1_RHO1_E1_INDEX)  = -(0.5*(pIL*velIL*qL(ALPHA1_INDEX) + pIR*velIR*qR(ALPHA1_INDEX)) -
                                        0.5*(pIL*velIL + pIR*velIR)*qR(ALPHA1_INDEX));
    #elifdef CENTERED
      F_minus(ALPHA1_INDEX) = velIL*(0.5*(qL(ALPHA1_INDEX) + qR(ALPHA1_INDEX)));
      F_plus(ALPHA1_INDEX)  = velIR*(0.5*(qL(ALPHA1_INDEX) + qR(ALPHA1_INDEX)));

      F_minus(ALPHA1_RHO1_U1_INDEX + curr_d) = -pIL*(0.5*(qL(ALPHA1_INDEX) + qR(ALPHA1_INDEX)));
      F_plus(ALPHA1_RHO1_U1_INDEX + curr_d)  = -pIR*(0.5*(qL(ALPHA1_INDEX) + qR(ALPHA1_INDEX)));

      F_minus(ALPHA1_RHO1_E1_INDEX) = -velIL*pIL*(0.5*(qL(ALPHA1_INDEX) + qR(ALPHA1_INDEX)));
      F_plus(ALPHA1_RHO1_E1_INDEX)  = -velIR*pIR*(0.5*(qL(ALPHA1_INDEX) + qR(ALPHA1_INDEX)));
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

    // Perform the loop over each dimension to compute the flux contribution
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        // Compute now the "discrete" non-conservative flux function
        discrete_flux[d].flux_function = [&](auto& cells, const Field& field)
                                            {
                                              #ifdef ORDER_2
                                                #ifdef PERFORM_RECON
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
                                                #else
                                                  // Compute the stencil
                                                  const auto& left  = cells[1];
                                                  const auto& right = cells[2];

                                                  const FluxValue<typename Flux<Field>::cfg>& qL = field[left];
                                                  const FluxValue<typename Flux<Field>::cfg>& qR = field[right];
                                                #endif
                                              #else
                                                // Compute the stencil and extract state
                                                const auto& left  = cells[0];
                                                const auto& right = cells[1];

                                                const FluxValue<typename Flux<Field>::cfg>& qL = field[left];
                                                const FluxValue<typename Flux<Field>::cfg>& qR = field[right];
                                              #endif

                                              FluxValue<typename Flux<Field>::cfg> F_minus,
                                                                                   F_plus;

                                              compute_discrete_flux(qL, qR, d, F_minus, F_plus);

                                              samurai::FluxValuePair<typename Flux<Field>::cfg> flux;
                                              flux[0] = F_minus;
                                              flux[1] = -F_plus;

                                              return flux;
                                            };
      }
    );

    return make_flux_based_scheme(discrete_flux);
  }

} // end of namespace

#endif
