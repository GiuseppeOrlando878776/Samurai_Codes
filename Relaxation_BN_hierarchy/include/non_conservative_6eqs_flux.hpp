#ifndef non_conservative_6eqs_flux_hpp
#define non_conservative_6eqs_flux_hpp

#include "flux_6eqs_base.hpp"

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
    // Zero contribution from continuity and momentum equations
    F_minus(ALPHA1_RHO1_INDEX) = 0.0;
    F_plus(ALPHA1_RHO1_INDEX)  = 0.0;
    F_minus(ALPHA2_RHO2_INDEX) = 0.0;
    F_plus(ALPHA2_RHO2_INDEX)  = 0.0;
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      F_minus(RHO_U_INDEX + d) = 0.0;
      F_plus(RHO_U_INDEX + d) = 0.0;
    }

    // Compute velocity and mass fractions left state
    const auto rhoL = qL(ALPHA1_RHO1_INDEX) + qL(ALPHA2_RHO2_INDEX);
    const auto Y1L  = qL(ALPHA1_RHO1_INDEX)/rhoL;
    const auto Y2L  = 1.0 - Y1L;
    const auto velL = qL(RHO_U_INDEX + curr_d)/rhoL;

    // Pressure phase 1 left state
    const auto alpha1L = qL(ALPHA1_INDEX);
    const auto rho1L   = qL(ALPHA1_RHO1_INDEX)/alpha1L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1L           = qL(ALPHA1_RHO1_E1_INDEX)/qL(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e1L -= 0.5*(qL(RHO_U_INDEX + d)/rhoL)*(qL(RHO_U_INDEX + d)/rhoL);
    }
    const auto p1L = this->phase1.pres_value(rho1L, e1L);

    // Pressure phase 2 left state
    const auto alpha2L = 1.0 - alpha1L;
    const auto rho2L   = qL(ALPHA2_RHO2_INDEX)/alpha2L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2L           = qL(ALPHA2_RHO2_E2_INDEX)/qL(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2L -= 0.5*(qL(RHO_U_INDEX + d)/rhoL)*(qL(RHO_U_INDEX + d)/rhoL);
    }
    const auto p2L = this->phase2.pres_value(rho2L, e2L);

    // Compute velocity and mass fractions right state
    const auto rhoR = qR(ALPHA1_RHO1_INDEX) + qR(ALPHA2_RHO2_INDEX);
    const auto Y1R  = qR(ALPHA1_RHO1_INDEX)/rhoR;
    const auto Y2R  = 1.0 - Y1R;
    const auto velR = qR(RHO_U_INDEX + curr_d)/rhoR;

    // Pressure phase 1 right state
    const auto alpha1R = qR(ALPHA1_INDEX);
    const auto rho1R   = qR(ALPHA1_RHO1_INDEX)/alpha1R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1R           = qR(ALPHA1_RHO1_E1_INDEX)/qR(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e1R -= 0.5*(qR(RHO_U_INDEX + d)/rhoR)*(qR(RHO_U_INDEX + d)/rhoR);
    }
    const auto p1R = this->phase1.pres_value(rho1R, e1R);

    // Pressure phase 2 right state
    const auto alpha2R = 1.0 - alpha1R;
    const auto rho2R   = qR(ALPHA2_RHO2_INDEX)/alpha2R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2R           = qR(ALPHA2_RHO2_E2_INDEX)/qR(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2R -= 0.5*(qR(RHO_U_INDEX + d)/rhoR)*(qR(RHO_U_INDEX + d)/rhoR);
    }
    const auto p2R = this->phase2.pres_value(rho2R, e2R);

    // Build the non conservative flux (a lot of approximations to be checked here)
    F_minus(ALPHA1_INDEX) = (0.5*(velL*qL(ALPHA1_INDEX) + velR*qR(ALPHA1_INDEX)) -
                             0.5*(velL + velR)*qL(ALPHA1_INDEX));
    F_plus(ALPHA1_INDEX)  = (0.5*(velL*qL(ALPHA1_INDEX) + velR*qR(ALPHA1_INDEX)) -
                             0.5*(velL + velR)*qR(ALPHA1_INDEX));

    F_minus(ALPHA1_RHO1_E1_INDEX) = -(0.5*(velL*Y2L*alpha1L*p1L + velR*Y2R*alpha1R*p1R) -
                                      0.5*(velL*Y2L + velR*Y2R)*alpha1L*p1L)
                                    +(0.5*(velL*Y1L*alpha2L*p2L + velR*Y1R*alpha2R*p2R) -
                                      0.5*(velL*Y1L + velR*Y1R)*alpha2L*p2L);
    F_plus(ALPHA1_RHO1_E1_INDEX)  = -(0.5*(velL*Y2L*alpha1L*p1L + velR*Y2R*alpha1R*p1R) -
                                      0.5*(velL*Y2L + velR*Y2R)*alpha1R*p1R)
                                    +(0.5*(velL*Y1L*alpha2L*p2L + velR*Y1R*alpha2R*p2R) -
                                      0.5*(velL*Y1L + velR*Y1R)*alpha2R*p2R);

    F_minus(ALPHA2_RHO2_E2_INDEX) = -F_minus(ALPHA1_RHO1_E1_INDEX);
    F_plus(ALPHA2_RHO2_E2_INDEX)  = -F_plus(ALPHA1_RHO1_E1_INDEX);
  }

  // Implement the contribution of the discrete flux for all the dimensions.
  //
  template<class Field>
  auto NonConservativeFlux<Field>::make_flux() {
    FluxDefinition<typename Flux<Field>::cfg> discrete_flux;

    // Perform the loop over each dimension to compute the flux contribution
    static_for<0, EquationData::dim>::apply(
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        // Compute now the "discrete" non-conservative flux function
        discrete_flux[d].flux_function = [&](auto& cells, const Field& field)
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
