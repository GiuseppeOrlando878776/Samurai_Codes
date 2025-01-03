#ifndef Suliciu_scheme_interface_hpp
#define Suliciu_scheme_interface_hpp

#include "flux_base.hpp"

#include "Suliciu_base/eos.hpp"
#include "Suliciu_base/Riemannsol.hpp"
#include "Suliciu_base/flux_numeriques.hpp"

namespace samurai {
  using namespace EquationData;

  /**
    * Implementation of the flux based on Suliciu-type relaxation
    */
  template<class Field>
  class RelaxationFlux {
  public:
    RelaxationFlux() = default; // Constructor which accepts in inputs the equations of state of the two phases

    auto make_flux(double& c); // Compute the flux over all cells.
                               // The input argument is employed to compute the Courant number

  private:
    template<class Other>
    Other FluxValue_to_Other(const FluxValue<typename Flux<Field>::cfg>& q); // Conversion from Samurai to other ("analogous") data structure for the state

    template<class Other>
    FluxValue<typename Flux<Field>::cfg> Other_to_FluxValue(const Other& q); // Conversion from other ("analogous") data structure for the state to Samurai

    void compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                               const FluxValue<typename Flux<Field>::cfg>& qR,
                               const std::size_t curr_d,
                               FluxValue<typename Flux<Field>::cfg>& F_minus,
                               FluxValue<typename Flux<Field>::cfg>& F_plus,
                               double& c); // Compute discrete flux
  };

  // Conversion from Samurai to other ("analogous") data structure for the state
  //
  template<class Field>
  template<class Other>
  Other RelaxationFlux<Field>::FluxValue_to_Other(const FluxValue<typename Flux<Field>::cfg>& q) {
    return Other(q(ALPHA1_INDEX),
                 q(ALPHA1_RHO1_INDEX), q(ALPHA1_RHO1_U1_INDEX), q(ALPHA1_RHO1_E1_INDEX),
                 q(ALPHA2_RHO2_INDEX), q(ALPHA2_RHO2_U2_INDEX), q(ALPHA2_RHO2_E2_INDEX));
  }

  // Conversion from Samurai to other ("analogous") data structure for the state
  //
  template<class Field>
  template<class Other>
  FluxValue<typename Flux<Field>::cfg> RelaxationFlux<Field>::Other_to_FluxValue(const Other& q) {
    FluxValue<typename Flux<Field>::cfg> res;

    res(ALPHA1_INDEX)	        = q.al1;
    res(ALPHA1_RHO1_INDEX)	  = q.alrho1;
    res(ALPHA1_RHO1_U1_INDEX)	= q.alrhou1;
    res(ALPHA1_RHO1_E1_INDEX) = q.alrhoE1;
    res(ALPHA2_RHO2_INDEX)	  = q.alrho2;
    res(ALPHA2_RHO2_U2_INDEX)	= q.alrhou2;
    res(ALPHA2_RHO2_E2_INDEX) = q.alrhoE2;

    return res;
  }

  // Implementation of the Suliciu type flux
  //
  template<class Field>
  void RelaxationFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                    const FluxValue<typename Flux<Field>::cfg>& qR,
                                                    std::size_t curr_d,
                                                    FluxValue<typename Flux<Field>::cfg>& F_minus,
                                                    FluxValue<typename Flux<Field>::cfg>& F_plus,
                                                    double& c) {
    // Employ "Saleh et al." functions to compute the Suliciu "flux"
    int Newton_iter;
    int dissip;
    const double eps = 1e-7;
    auto fW_minus = Etat();
    auto fW_plus  = Etat();
    c = std::max(c, flux_relax(FluxValue_to_Other<Etat>(qL), FluxValue_to_Other<Etat>(qR), fW_minus, fW_plus, Newton_iter, dissip, eps));

    // Conversion from Etat to FluxValue<typename Flux<Field>::cfg>
    F_minus = Other_to_FluxValue<Etat>(fW_minus);
    F_plus  = Other_to_FluxValue<Etat>(fW_plus);
  }

  // Implement the contribution of the discrete flux for all the directions.
  //
  template<class Field>
  auto RelaxationFlux<Field>::make_flux(double& c) {
    FluxDefinition<typename Flux<Field>::cfg> discrete_flux;

    // Perform the loop over each dimension to compute the flux contribution
    static_for<0, EquationData::dim>::apply(
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        // Compute now the "discrete" non-conservative flux function
        discrete_flux[d].flux_function = [&](auto& cells, const Field& field)
                                            {
                                              const auto& left  = cells[0];
                                              const auto& right = cells[1];

                                              const auto& qL = field[left];
                                              const auto& qR = field[right];

                                              FluxValue<typename Flux<Field>::cfg> F_minus,
                                                                                   F_plus;

                                              compute_discrete_flux(qL, qR, d, F_minus, F_plus, c);

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
