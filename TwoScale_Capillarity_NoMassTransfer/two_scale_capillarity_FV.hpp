#pragma once
#include <samurai/schemes/fv.hpp>

#define ORDER_2

namespace EquationData {
  // Declare spatial dimension
  static constexpr std::size_t dim = 2; // Spatial dimension. It would be ideal to be able to get it
                                        // direclty from Field, but I need to move the definition of these indices

  // Declare parameters related to surface tension coefficient
  static constexpr double sigma = 1e-2;

  // Declare some parameters related to EOS.
  static constexpr double p0_phase1   = 1e5;
  static constexpr double p0_phase2   = 1e5;

  static constexpr double rho0_phase1 = 1e3;
  static constexpr double rho0_phase2 = 1.0;

  static constexpr double c0_phase1   = 1e1;
  static constexpr double c0_phase2   = 1e1;

  // Use auxiliary variables for the indices for the sake of generality
  static constexpr std::size_t M1_INDEX             = 0;
  static constexpr std::size_t M2_INDEX             = 1;
  static constexpr std::size_t RHO_ALPHA1_BAR_INDEX = 2;
  static constexpr std::size_t RHO_U_INDEX          = 3;

  static constexpr std::size_t ALPHA1_BAR_INDEX = RHO_ALPHA1_BAR_INDEX;

  // Save also the total number of (scalar) variables
  static constexpr std::size_t NVARS = 3 + dim;
}


namespace samurai {
  using namespace EquationData;

  /**
    * Generic class to compute the flux between a left and right state
    */
  template<class Field>
  class Flux {
  public:
    // Definitions and sanity checks
    static constexpr std::size_t field_size = Field::size;
    static_assert(field_size == EquationData::NVARS, "The number of elements in the state does not correpsond to the number of equations");
    static_assert(Field::dim == EquationData::dim, "The spatial dimesions do not match");
    static constexpr std::size_t output_field_size = field_size;
    #ifdef ORDER_2
      static constexpr std::size_t stencil_size = 4;
    #else
      static constexpr std::size_t stencil_size = 2;
    #endif

    using cfg = FluxConfig<SchemeType::NonLinear, output_field_size, stencil_size, Field>;

    Flux(const BarotropicEOS<>& EOS_phase1,
         const BarotropicEOS<>& EOS_phase2,
         const double eps_,
         const double mod_grad_alpha1_bar_min_); // Constructor which accepts in inputs the equations of state of the two phases

    FluxValue<cfg> evaluate_continuous_flux(const FluxValue<cfg>& q,
                                            const std::size_t curr_d,
                                            const auto& grad_alpha1_bar); // Evaluate the 'continuous' flux for the state q along direction curr_d

    FluxValue<cfg> evaluate_hyperbolic_operator(const FluxValue<cfg>& q,
                                                const std::size_t curr_d); // Evaluate the hyperbolic operator for the state q along direction curr_d

    void perform_Newton_step_relaxation(auto conserved_variables, const auto H,
                                        auto& dalpha1_bar, auto& alpha1_bar, bool& relaxation_applied,
                                        const double tol = 1e-12, const double lambda = 0.9); // Perform a Newton step relaxation for a state vector
                                                                                              // (it is not a real space dependent procedure,
                                                                                              // but I would need to be able to do it inside the flux location
                                                                                              // for MUSCL reconstruction)

  protected:
    const BarotropicEOS<>& phase1;
    const BarotropicEOS<>& phase2;

    const double eps;                     // Tolerance of pure phase to set NaNs
    const double mod_grad_alpha1_bar_min; // Tolerance to compute the unit normal

    FluxValue<cfg> cons2prim(const FluxValue<cfg>& cons) const; // Conversion from conservative to primitive variables

    FluxValue<cfg> prim2cons(const FluxValue<cfg>& prim) const; // Conversion from primitive to conservative variabless
  };

  // Class constructor in order to be able to work with the equation of state
  //
  template<class Field>
  Flux<Field>::Flux(const BarotropicEOS<>& EOS_phase1,
                    const BarotropicEOS<>& EOS_phase2,
                    const double eps_,
                    const double mod_grad_alpha1_bar_min_): phase1(EOS_phase1), phase2(EOS_phase2), eps(eps_), mod_grad_alpha1_bar_min(mod_grad_alpha1_bar_min_) {}

  // Evaluate the 'continuous flux'
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::evaluate_continuous_flux(const FluxValue<cfg>& q,
                                                                             const std::size_t curr_d,
                                                                             const auto& grad_alpha1_bar) {
    // Sanity check in terms of dimensions
    assert(curr_d < EquationData::dim);

    FluxValue<cfg> res = q;

    // Compute the current velocity
    const auto rho   = q(M1_INDEX) + q(M2_INDEX);
    const auto vel_d = q(RHO_U_INDEX + curr_d)/rho;

    // Multiply the state the velcoity along the direction of interest
    res(M1_INDEX) *= vel_d;
    res(M2_INDEX) *= vel_d;
    res(RHO_ALPHA1_BAR_INDEX) *= vel_d;
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      res(RHO_U_INDEX + d) *= vel_d;
    }

    // Compute and add the contribution due to the pressure
    const auto alpha1_bar = q(RHO_ALPHA1_BAR_INDEX)/rho;
    const auto alpha1     = alpha1_bar;
    const auto rho1       = (alpha1 > eps) ? q(M1_INDEX)/alpha1 : nan("");
    const auto p1         = phase1.pres_value(rho1);

    const auto alpha2     = 1.0 - alpha1;
    const auto rho2       = (alpha2 > eps) ? q(M2_INDEX)/alpha2 : nan("");
    const auto p2         = phase2.pres_value(rho2);

    const auto p_bar      = (alpha1 > eps && alpha2 > eps) ?
                            alpha1_bar*p1 + (1.0 - alpha1_bar)*p2 :
                            ((alpha1 < eps) ? p2 : p1);

    res(RHO_U_INDEX + curr_d) += p_bar;

    // Add the contribution due to surface tension
    const auto mod_grad_alpha1_bar = std::sqrt(xt::sum(grad_alpha1_bar*grad_alpha1_bar)());

    if(mod_grad_alpha1_bar > mod_grad_alpha1_bar_min) {
      const auto n = grad_alpha1_bar/mod_grad_alpha1_bar;

      if(curr_d == 0) {
        res(RHO_U_INDEX) += EquationData::sigma*(n(0)*n(0) - 1.0)*mod_grad_alpha1_bar;
        res(RHO_U_INDEX + 1) += EquationData::sigma*n(0)*n(1)*mod_grad_alpha1_bar;
      }

      if(curr_d == 1) {
        res(RHO_U_INDEX) += EquationData::sigma*n(0)*n(1)*mod_grad_alpha1_bar;
        res(RHO_U_INDEX + 1) += EquationData::sigma*(n(1)*n(1) - 1.0)*mod_grad_alpha1_bar;
      }
    }

    return res;
  }

  // Evaluate the hyperbolic part of the 'continuous' flux
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::evaluate_hyperbolic_operator(const FluxValue<cfg>& q,
                                                                                 const std::size_t curr_d) {
    // Sanity check in terms of dimensions
    assert(curr_d < EquationData::dim);

    FluxValue<cfg> res = q;

    // Compute the current velocity
    const auto rho   = q(M1_INDEX) + q(M2_INDEX);
    const auto vel_d = q(RHO_U_INDEX + curr_d)/rho;

    // Multiply the state the velcoity along the direction of interest
    res(M1_INDEX) *= vel_d;
    res(M2_INDEX) *= vel_d;
    res(RHO_ALPHA1_BAR_INDEX) *= vel_d;
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      res(RHO_U_INDEX + d) *= vel_d;
    }

    // Compute and add the contribution due to the pressure
    const auto alpha1_bar = q(RHO_ALPHA1_BAR_INDEX)/rho;
    const auto alpha1     = alpha1_bar;
    const auto rho1       = (alpha1 > eps) ? q(M1_INDEX)/alpha1 : nan("");
    const auto p1         = phase1.pres_value(rho1);

    const auto alpha2     = 1.0 - alpha1;
    const auto rho2       = (alpha2 > eps) ? q(M2_INDEX)/alpha2 : nan("");
    const auto p2         = phase2.pres_value(rho2);

    const auto p_bar      = (alpha1 > eps && alpha2 > eps) ?
                            alpha1_bar*p1 + (1.0 - alpha1_bar)*p2 :
                            ((alpha1 < eps) ? p2 : p1);

    res(RHO_U_INDEX + curr_d) += p_bar;

    return res;
  }

  // Perform a Newton step relaxation for a single vector state (i.e. a single cell)
  //
  template<class Field>
  void Flux<Field>::perform_Newton_step_relaxation(auto conserved_variables, const auto H,
                                                   auto& dalpha1_bar, auto& alpha1_bar, bool& relaxation_applied,
                                                   const double tol, const double lambda) {

    // Reinitialization of partial masses in case of evanascent volume fraction
    if(alpha1_bar < eps) {
      (*conserved_variables)(M1_INDEX) = alpha1_bar*EquationData::rho0_phase1;
    }
    if(1.0 - alpha1_bar < eps) {
      (*conserved_variables)(M2_INDEX) = (1.0 - alpha1_bar)*EquationData::rho0_phase2;
    }

    const auto rho = (*conserved_variables)(M1_INDEX)
                   + (*conserved_variables)(M2_INDEX);

    // Update auxiliary values affected by the nonlinear function for which we seek a zero
    const auto alpha1 = alpha1_bar;
    const auto rho1   = (alpha1 > eps) ? (*conserved_variables)(M1_INDEX)/alpha1 : nan("");
    const auto p1     = phase1.pres_value(rho1);

    const auto alpha2 = 1.0 - alpha1;
    const auto rho2   = (alpha2 > eps) ? (*conserved_variables)(M2_INDEX)/alpha2 : nan("");
    const auto p2     = phase2.pres_value(rho2);

    // Compute the nonlinear function for which we seek the zero (basically the Laplace law)
    const auto F = p1 - p2 - EquationData::sigma*H;

    // Perform the relaxation only where really needed
    if(!std::isnan(F) && std::abs(F) > tol*std::min(EquationData::p0_phase1, EquationData::sigma*H) && std::abs(dalpha1_bar) > tol &&
       alpha1_bar > eps && 1.0 - alpha1_bar > eps) {
      relaxation_applied = true;

      // Compute the derivative w.r.t large-scale volume fraction recalling that for a barotropic EOS dp/drho = c^2
      const auto dF_dalpha1_bar = -(*conserved_variables)(M1_INDEX)/(alpha1_bar*alpha1_bar)*
                                   phase1.c_value(rho1)*phase1.c_value(rho1)
                                  -(*conserved_variables)(M2_INDEX)/((1.0 - alpha1_bar)*(1.0 - alpha1_bar))*
                                   phase2.c_value(rho2)*phase2.c_value(rho2);

      // Compute the large-scale volume fraction update
      dalpha1_bar = -F/dF_dalpha1_bar;
      if(dalpha1_bar > 0.0) {
        dalpha1_bar = std::min(dalpha1_bar, lambda*(1.0 - alpha1_bar));
      }
      else if(dalpha1_bar < 0.0) {
        dalpha1_bar = std::max(dalpha1_bar, -lambda*alpha1_bar);
      }

      if(alpha1_bar + dalpha1_bar < 0.0 || alpha1_bar + dalpha1_bar > 1.0) {
        std::cerr << "Bounds exceeding value for large-scale volume fraction inside Newton step " << std::endl;
      }
      else {
        alpha1_bar += dalpha1_bar;
      }
    }

    // Update the vector of conserved variables
    (*conserved_variables)(RHO_ALPHA1_BAR_INDEX) = rho*alpha1_bar;
  }

  // Conversion from conserved to primitive variables
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::cons2prim(const FluxValue<cfg>& cons) const {

    FluxValue<cfg> prim = cons;

    prim(M1_INDEX)         = cons(M1_INDEX);
    prim(M2_INDEX)         = cons(M2_INDEX);
    prim(ALPHA1_BAR_INDEX) = cons(RHO_ALPHA1_BAR_INDEX)/(cons(M1_INDEX) + cons(M2_INDEX));
    prim(RHO_U_INDEX)      = cons(RHO_U_INDEX);

    return prim;
  }

  // Conversion from primitive to conserved variables
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::prim2cons(const FluxValue<cfg>& prim) const {

    FluxValue<cfg> cons = prim;

    cons(M1_INDEX)             = prim(M1_INDEX);
    cons(M2_INDEX)             = prim(M2_INDEX);
    cons(RHO_ALPHA1_BAR_INDEX) = (prim(M1_INDEX) + prim(M2_INDEX))*prim(ALPHA1_BAR_INDEX);
    cons(RHO_U_INDEX)          = prim(RHO_U_INDEX);

    return cons;
  }


  /**
    * Implementation of a Rusanov flux
    */
  template<class Field>
  class RusanovFlux: public Flux<Field> {
  public:
    RusanovFlux(const BarotropicEOS<>& EOS_phase1,
                const BarotropicEOS<>& EOS_phase2,
                const double eps_,
                const double mod_grad_alpha1_bar_min_); // Constructor which accepts in inputs the equations of state of the two phases

    #ifdef ORDER_2
      auto make_two_scale_capillarity(const auto& grad_alpha1_bar, const auto& H); // Compute the flux over all cells
    #else
      auto make_two_scale_capillarity(const auto& grad_alpha1_bar); // Compute the flux over all cells
    #endif

  private:
    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                               const FluxValue<typename Flux<Field>::cfg>& qR,
                                                               const std::size_t curr_d,
                                                               const auto& grad_alpha1_barL,
                                                               const auto& grad_alpha1_barR); // Rusanov flux along direction curr_d
  };

  // Constructor derived from the base class
  //
  template<class Field>
  RusanovFlux<Field>::RusanovFlux(const BarotropicEOS<>& EOS_phase1,
                                  const BarotropicEOS<>& EOS_phase2,
                                  const double eps_,
                                  const double grad_alpha1_bar_min_): Flux<Field>(EOS_phase1, EOS_phase2, eps_, grad_alpha1_bar_min_) {}

  // Implementation of a Rusanov flux
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> RusanovFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                                                 const FluxValue<typename Flux<Field>::cfg>& qR,
                                                                                 const std::size_t curr_d,
                                                                                 const auto& grad_alpha1_barL,
                                                                                 const auto& grad_alpha1_barR) {
    // Compute the quantities needed for the maximum eigenvalue estimate for the left state
    const auto rho_L        = qL(M1_INDEX) + qL(M2_INDEX);
    const auto vel_d_L      = qL(RHO_U_INDEX + curr_d)/rho_L;

    const auto alpha1_bar_L = qL(RHO_ALPHA1_BAR_INDEX)/rho_L;
    const auto alpha1_L     = alpha1_bar_L;
    const auto rho1_L       = (alpha1_L > this->eps) ? qL(M1_INDEX)/alpha1_L : nan("");
    const auto alpha2_L     = 1.0 - alpha1_L;
    const auto rho2_L       = (alpha2_L > this->eps) ? qL(M2_INDEX)/alpha2_L : nan("");
    const auto c_squared_L  = qL(M1_INDEX)*this->phase1.c_value(rho1_L)*this->phase1.c_value(rho1_L)
                            + qL(M2_INDEX)*this->phase2.c_value(rho2_L)*this->phase2.c_value(rho2_L);
    const auto c_L          = std::sqrt(c_squared_L/rho_L);
    const auto r_L          = EquationData::sigma*std::sqrt(xt::sum(grad_alpha1_barL*grad_alpha1_barL)())/(rho_L*c_L*c_L);

    // Compute the quantities needed for the maximum eigenvalue estimate for the right state
    const auto rho_R        = qR(M1_INDEX) + qR(M2_INDEX);
    const auto vel_d_R      = qR(RHO_U_INDEX + curr_d)/rho_R;

    const auto alpha1_bar_R = qR(RHO_ALPHA1_BAR_INDEX)/rho_R;
    const auto alpha1_R     = alpha1_bar_R;
    const auto rho1_R       = (alpha1_R > this->eps) ? qR(M1_INDEX)/alpha1_R : nan("");
    const auto alpha2_R     = 1.0 - alpha1_R;
    const auto rho2_R       = (alpha2_R > this->eps) ? qR(M2_INDEX)/alpha2_R : nan("");
    const auto c_squared_R  = qR(M1_INDEX)*this->phase1.c_value(rho1_R)*this->phase1.c_value(rho1_R)
                            + qR(M2_INDEX)*this->phase2.c_value(rho2_R)*this->phase2.c_value(rho2_R);
    const auto c_R          = std::sqrt(c_squared_R/rho_R);
    const auto r_R          = EquationData::sigma*std::sqrt(xt::sum(grad_alpha1_barR*grad_alpha1_barR)())/(rho_R*c_R*c_R);

    // Compute the estimate of the eigenvalue considering also the surface tension contribution
    const auto lambda = std::max(std::abs(vel_d_L) + c_L*(1.0 + 0.125*r_L),
                                 std::abs(vel_d_R) + c_R*(1.0 + 0.125*r_R));

    return 0.5*(this->evaluate_continuous_flux(qL, curr_d, grad_alpha1_barL) +
                this->evaluate_continuous_flux(qR, curr_d, grad_alpha1_barR)) - // centered contribution
           0.5*lambda*(qR - qL); // upwinding contribution
  }

  // Implement the contribution of the discrete flux for all the cells in the mesh.
  //
  template<class Field>
  #ifdef ORDER_2
  auto RusanovFlux<Field>::make_two_scale_capillarity(const auto& grad_alpha1_bar, const auto& H)
  #else
  auto RusanovFlux<Field>::make_two_scale_capillarity(const auto& grad_alpha1_bar)
  #endif
  {
    FluxDefinition<typename Flux<Field>::cfg> Rusanov_f;

    // Perform the loop over each dimension to compute the flux contribution
    static_for<0, EquationData::dim>::apply(
      // First, we need a function to compute the "continuous" flux
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

                                              const double beta = 1.0;
                                              auto primL_recon = primL;
                                              auto primR_recon = primR;
                                              for(std::size_t comp = 0; comp < Field::size; ++comp) {
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

                                              FluxValue<typename Flux<Field>::cfg> qL = this->prim2cons(primL_recon);
                                              FluxValue<typename Flux<Field>::cfg> qR = this->prim2cons(primR_recon);

                                              // Relax the interpolated state
                                              const double tol    = 1e-12; /*--- Tolerance of the Newton method ---*/
                                              const double lambda = 0.9;   /*--- Parameter for bound preserving strategy ---*/

                                              // Start with left state
                                              std::size_t Newton_iter = 0;
                                              bool relaxation_applied = true;
                                              double dalpha1_barL     = std::numeric_limits<double>::infinity();
                                              double alpha1_barL      = qL(RHO_ALPHA1_BAR_INDEX)/
                                                                        (qL(M1_INDEX) + qL(M2_INDEX));
                                              while(relaxation_applied == true) {
                                                relaxation_applied = false;
                                                Newton_iter++;

                                                this->perform_Newton_step_relaxation(std::make_unique<decltype(qL)>(qL), H[left],
                                                                                     dalpha1_barL, alpha1_barL, relaxation_applied, tol, lambda);

                                                // Newton cycle diverged
                                                if(Newton_iter > 60) {
                                                  std::cout << "Netwon method not converged in the relaxation of left state after MUSCL" << std::endl;
                                                  exit(1);
                                                }
                                              }

                                              // Focus now on right state
                                              Newton_iter         = 0;
                                              relaxation_applied  = true;
                                              double dalpha1_barR = std::numeric_limits<double>::infinity();
                                              double alpha1_barR  = qR(RHO_ALPHA1_BAR_INDEX)/
                                                                    (qR(M1_INDEX) + qR(M2_INDEX));
                                              while(relaxation_applied == true) {
                                                relaxation_applied = false;
                                                Newton_iter++;

                                                this->perform_Newton_step_relaxation(std::make_unique<decltype(qR)>(qR), H[right],
                                                                                     dalpha1_barR, alpha1_barR, relaxation_applied, tol, lambda);

                                                // Newton cycle diverged
                                                if(Newton_iter > 60) {
                                                  std::cout << "Netwon method not converged in the relaxation of right state after MUSCL" << std::endl;
                                                  exit(1);
                                                }
                                              }
                                            #else
                                              // Compute the stencil and extract state
                                              const auto& left  = cells[0];
                                              const auto& right = cells[1];

                                              const FluxValue<typename Flux<Field>::cfg> qL = field[left];
                                              const FluxValue<typename Flux<Field>::cfg> qR = field[right];
                                            #endif

                                            // Compute the numerical flux
                                            return compute_discrete_flux(qL, qR, d,
                                                                         grad_alpha1_bar[left], grad_alpha1_bar[right]);
                                          };
    });

    return make_flux_based_scheme(Rusanov_f);
  }


  /**
    * Implementation of a Godunov flux
    */
  template<class Field>
  class GodunovFlux: public Flux<Field> {
  public:
    GodunovFlux(const BarotropicEOS<>& EOS_phase1,
                const BarotropicEOS<>& EOS_phase2,
                const double eps_,
                const double mod_grad_alpha1_bar_min_); // Constructor which accepts in inputs the equations of state of the two phases

    #ifdef ORDER_2
      auto make_two_scale_capillarity(const auto& grad_alpha1_bar, const auto& H); // Compute the flux over all cells
    #else
      auto make_two_scale_capillarity(const auto& grad_alpha1_bar); // Compute the flux over all cells
    #endif

  private:
    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                               const FluxValue<typename Flux<Field>::cfg>& qR,
                                                               const std::size_t curr_d,
                                                               const auto& grad_alpha1_barL,
                                                               const auto& grad_alpha1_barR,
                                                               const bool is_discontinuous); // Godunov flux for the along direction curr_d

    void solve_p_star(const auto& qL, const auto& qR,
                      const double dvel_d, const double vel_d_L,
                      const double p0_L, const double p0_R, double& p_star); // Newton method to compute p* in the exact solver for the hyperbolic part
  };

  // Constructor derived from the base class
  //
  template<class Field>
  GodunovFlux<Field>::GodunovFlux(const BarotropicEOS<>& EOS_phase1,
                                  const BarotropicEOS<>& EOS_phase2,
                                  const double eps_,
                                  const double grad_alpha1_bar_min_): Flux<Field>(EOS_phase1, EOS_phase2, eps_, grad_alpha1_bar_min_) {}

  // Compute p* through Newton-Rapson method
  //
  template<class Field>
  void GodunovFlux<Field>::solve_p_star(const auto& qL, const auto& qR,
                                        const double dvel_d, const double vel_d_L,
                                        const double p0_L, const double p0_R, double& p_star) {
    const double tol            = 1e-8; /*--- Tolerance of the Newton method ---*/
    const double lambda         = 0.9;  /*--- Parameter for bound preserving strategy ---*/
    const std::size_t max_iters = 100;  /*--- Maximum number of Newton iterations ----*/

    double dp_star = std::numeric_limits<double>::infinity();

    // Left state useful variables
    const auto rho_L        = qL(M1_INDEX) + qL(M2_INDEX);
    const auto alpha1_bar_L = qL(RHO_ALPHA1_BAR_INDEX)/rho_L;
    const auto alpha1_L     = alpha1_bar_L;
    const auto rho1_L       = (alpha1_L > this->eps) ? qL(M1_INDEX)/alpha1_L : nan("");
    const auto alpha2_L     = 1.0 - alpha1_L;
    const auto rho2_L       = (alpha2_L > this->eps) ? qL(M2_INDEX)/alpha2_L : nan("");
    const auto c_squared_L  = qL(M1_INDEX)*this->phase1.c_value(rho1_L)*this->phase1.c_value(rho1_L)
                            + qL(M2_INDEX)*this->phase2.c_value(rho2_L)*this->phase2.c_value(rho2_L);
    const auto c_L          = std::sqrt(c_squared_L/rho_L);
    const auto p_bar_L      = (alpha1_L > this->eps && alpha2_L > this->eps) ?
                              alpha1_bar_L*this->phase1.pres_value(rho1_L) + (1.0 - alpha1_bar_L)*this->phase2.pres_value(rho2_L) :
                              ((alpha1_L < this->eps) ? this->phase2.pres_value(rho2_L) : this->phase1.pres_value(rho1_L));

    // Right state useful variables
    const auto rho_R        = qR(M1_INDEX) + qR(M2_INDEX);
    const auto alpha1_bar_R = qR(RHO_ALPHA1_BAR_INDEX)/rho_R;
    const auto alpha1_R     = alpha1_bar_R;
    const auto rho1_R       = (alpha1_R > this->eps) ? qR(M1_INDEX)/alpha1_R : nan("");
    const auto alpha2_R     = 1.0 - alpha1_R;
    const auto rho2_R       = (alpha2_R > this->eps) ? qR(M2_INDEX)/alpha2_R : nan("");
    const auto c_squared_R  = qR(M1_INDEX)*this->phase1.c_value(rho1_R)*this->phase1.c_value(rho1_R)
                            + qR(M2_INDEX)*this->phase2.c_value(rho2_R)*this->phase2.c_value(rho2_R);
    const auto c_R          = std::sqrt(c_squared_R/rho_R);
    const auto p_bar_R      = (alpha1_R > this->eps && alpha2_R > this->eps) ?
                              alpha1_bar_R*this->phase1.pres_value(rho1_R) + (1.0 - alpha1_bar_R)*this->phase2.pres_value(rho2_R) :
                              ((alpha1_R < this->eps) ? this->phase2.pres_value(rho2_R) : this->phase1.pres_value(rho1_R));

    if(p_star <= p0_L || p_bar_L <= p0_L) {
      std::cerr << "Non-admissible value for the pressure at the beginning of the Newton method to compute p* in Godunov solver" << std::endl;
      exit(1);
    }

    double F_p_star = dvel_d;
    if(p_star <= p_bar_L) {
      F_p_star += c_L*std::log((p_bar_L - p0_L)/(p_star - p0_L));
    }
    else {
      F_p_star -= (p_star - p_bar_L)/std::sqrt(rho_L*(p_star - p0_L));
    }
    if(p_star <= p_bar_R) {
      F_p_star += c_R*std::log((p_bar_R - p0_R)/(p_star - p0_R));
    }
    else {
      F_p_star -= (p_star - p_bar_R)/std::sqrt(rho_R*(p_star - p0_R));
    }

    // Loop of Newton method to compute p*
    std::size_t Newton_iter = 0;
    while(Newton_iter < max_iters && std::abs(F_p_star) > tol*std::abs(vel_d_L) && std::abs(dp_star)/std::abs(p_star) > tol) {
      Newton_iter++;

      // Unmodified Newton-Rapson increment
      double dF_p_star;
      if(p_star <= p_bar_L) {
        dF_p_star = c_L/(p0_L - p_star);
      }
      else {
        dF_p_star = (2.0*p0_L - p_star - p_bar_L)/
                    (2.0*(p_star - p0_L)*std::sqrt(rho_L*(p_star - p0_L)));
      }
      if(p_star <= p_bar_R) {
        dF_p_star += c_R/(p0_R - p_star);
      }
      else {
        dF_p_star += (2.0*p0_R - p_star - p_bar_R)/
                     (2.0*(p_star - p0_R)*std::sqrt(rho_R*(p_star - p0_R)));
      }
      double ddF_p_star;
      if(p_star <= p_bar_L) {
        ddF_p_star = c_L/((p0_L - p_star)*(p0_L - p_star));
      }
      else {
        ddF_p_star = (-4.0*p0_L + p_star + 3.0*p_bar_L)/
                     (4.0*(p_star - p0_L)*(p_star - p0_L)*std::sqrt(rho_L*(p_star - p0_L)));
      }
      if(p_star <= p_bar_R) {
        ddF_p_star += c_R/((p0_R - p_star)*(p0_R - p_star));
      }
      else {
        ddF_p_star += (-4.0*p0_R + p_star + 3.0*p_bar_R)/
                      (4.0*(p_star - p0_R)*(p_star - p0_R)*std::sqrt(rho_R*(p_star - p0_R)));
      }
      dp_star = -2.0*F_p_star*dF_p_star/(2.0*dF_p_star*dF_p_star - F_p_star*ddF_p_star);

      // Bound preserving increment
      dp_star = std::max(dp_star, lambda*(std::max(p0_L, p0_R) - p_star));

      if(p_star + dp_star <= p0_L) {
        std::cerr << "Non-admissible value for the pressure in the Newton method to compute p* in Godunov solver" << std::endl;
        exit(1);
      }
      else {
        p_star += dp_star;
      }

      // Newton cycle diverged
      if(Newton_iter == max_iters) {
        std::cout << "Netwon method not converged to compute p* in the Godunov solver" << std::endl;
        exit(1);
      }

      // Update function for which we seek the zero
      F_p_star = dvel_d;
      if(p_star <= p_bar_L) {
        F_p_star += c_L*std::log((p_bar_L - p0_L)/(p_star - p0_L));
      }
      else {
        F_p_star -= (p_star - p_bar_L)/std::sqrt(rho_L*(p_star - p0_L));
      }
      if(p_star <= p_bar_R) {
        F_p_star += c_R*std::log((p_bar_R - p0_R)/(p_star - p0_R));
      }
      else {
        F_p_star -= (p_star - p_bar_R)/std::sqrt(rho_R*(p_star - p0_R));
      }
    }
  }

  // Implementation of a Godunov flux
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> GodunovFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                                                 const FluxValue<typename Flux<Field>::cfg>& qR,
                                                                                 const std::size_t curr_d,
                                                                                 const auto& grad_alpha1_barL,
                                                                                 const auto& grad_alpha1_barR,
                                                                                 const bool is_discontinuous) {
    // Compute the intermediate state (either shock or rarefaction)
    FluxValue<typename Flux<Field>::cfg> q_star = qL;

    if(is_discontinuous) {
      // Left state useful variables
      const auto rho_L        = qL(M1_INDEX) + qL(M2_INDEX);
      const auto vel_d_L      = qL(RHO_U_INDEX + curr_d)/rho_L;
      const auto alpha1_bar_L = qL(RHO_ALPHA1_BAR_INDEX)/rho_L;
      const auto alpha1_L     = alpha1_bar_L;
      const auto rho1_L       = (alpha1_L > this->eps) ? qL(M1_INDEX)/alpha1_L : nan("");
      const auto alpha2_L     = 1.0 - alpha1_L;
      const auto rho2_L       = (alpha2_L > this->eps) ? qL(M2_INDEX)/alpha2_L : nan("");
      const auto c_squared_L  = qL(M1_INDEX)*this->phase1.c_value(rho1_L)*this->phase1.c_value(rho1_L)
                              + qL(M2_INDEX)*this->phase2.c_value(rho2_L)*this->phase2.c_value(rho2_L);
      const auto c_L          = std::sqrt(c_squared_L/rho_L);

      const auto p0_L         = EquationData::p0_phase1
                              - alpha1_bar_L*EquationData::rho0_phase1*EquationData::c0_phase1*EquationData::c0_phase1
                              - (1.0 - alpha1_bar_L)*EquationData::rho0_phase2*EquationData::c0_phase2*EquationData::c0_phase2;

      // Right state useful variables
      const auto rho_R        = qR(M1_INDEX) + qR(M2_INDEX);
      const auto vel_d_R      = qR(RHO_U_INDEX + curr_d)/rho_R;
      const auto alpha1_bar_R = qR(RHO_ALPHA1_BAR_INDEX)/rho_R;
      const auto alpha1_R     = alpha1_bar_R;
      const auto rho1_R       = (alpha1_R > this->eps) ? qR(M1_INDEX)/alpha1_R : nan("");
      const auto alpha2_R     = 1.0 - alpha1_R;
      const auto rho2_R       = (alpha2_R > this->eps) ? qR(M2_INDEX)/alpha2_R : nan("");
      const auto c_squared_R  = qR(M1_INDEX)*this->phase1.c_value(rho1_R)*this->phase1.c_value(rho1_R)
                              + qR(M2_INDEX)*this->phase2.c_value(rho2_R)*this->phase2.c_value(rho2_R);
      const auto c_R          = std::sqrt(c_squared_R/rho_R);

      const auto p0_R         = EquationData::p0_phase1
                              - alpha1_bar_R*EquationData::rho0_phase1*EquationData::c0_phase1*EquationData::c0_phase1
                              - (1.0 - alpha1_bar_R)*EquationData::rho0_phase2*EquationData::c0_phase2*EquationData::c0_phase2;

      // Compute p*
      const auto p_bar_L = (alpha1_L > this->eps && alpha2_L > this->eps) ?
                            alpha1_bar_L*this->phase1.pres_value(rho1_L) + (1.0 - alpha1_bar_L)*this->phase2.pres_value(rho2_L) :
                           ((alpha1_L < this->eps) ? this->phase2.pres_value(rho2_L) : this->phase1.pres_value(rho1_L));
      const auto p_bar_R = (alpha1_R > this->eps && alpha2_R > this->eps) ?
                            alpha1_bar_R*this->phase1.pres_value(rho1_R) + (1.0 - alpha1_bar_R)*this->phase2.pres_value(rho2_R) :
                           ((alpha1_R < this->eps) ? this->phase2.pres_value(rho2_R) : this->phase1.pres_value(rho1_R));
      auto p_star        = std::max(0.5*(p_bar_L + p_bar_R),
                                    std::max(p0_L, p0_R) + 0.1*std::abs(std::max(p0_L, p0_R)));
      solve_p_star(qL, qR, vel_d_L - vel_d_R, vel_d_L, p0_L, p0_R, p_star);

      // Compute u*
      const auto u_star = (p_star <= p_bar_L) ? vel_d_L + c_L*std::log((p_bar_L - p0_L)/(p_star - p0_L)) :
                                                vel_d_L - (p_star - p_bar_L)/std::sqrt(rho_L*(p_star - p0_L));

      // Left "connecting state"
      if(u_star > 0.0) {
        // 1-wave left shock
        if(p_star > p_bar_L) {
          const auto r = 1.0 + 1.0/((rho_L*c_L*c_L)/(p_star - p_bar_L));

          const auto m1_L_star  = qL(M1_INDEX)*r;
          const auto m2_L_star  = qL(M2_INDEX)*r;
          const auto rho_L_star = m1_L_star + m2_L_star;

          auto s_L = nan("");
          if(r > 1) {
            s_L = u_star + (vel_d_L - u_star)/(1.0 - r);
          }
          else if (r == 1) {
            s_L = u_star + (vel_d_L - u_star)*(-std::numeric_limits<double>::infinity());
          }

          // If left of left shock, q* = qL, already assigned.
          // If right of left shock, is the computed state
          if(!std::isnan(s_L) && s_L < 0.0) {
            q_star(M1_INDEX)             = m1_L_star;
            q_star(M2_INDEX)             = m2_L_star;
            q_star(RHO_ALPHA1_BAR_INDEX) = rho_L_star*alpha1_bar_L;
            if(curr_d == 0) {
              q_star(RHO_U_INDEX)     = rho_L_star*u_star;
              q_star(RHO_U_INDEX + 1) = rho_L_star*(qL(RHO_U_INDEX + 1)/rho_L);
            }
            else if(curr_d == 1) {
              q_star(RHO_U_INDEX)     = rho_L_star*(qL(RHO_U_INDEX)/rho_L);
              q_star(RHO_U_INDEX + 1) = rho_L_star*u_star;
            }
          }
        }
        // 3-waves left fan
        else {
          // Left of the left fan is qL, already assigned. Now we need to check if we are in
          // the left fan or at the right of the left fan
          const auto sH_L = vel_d_L - c_L;
          const auto sT_L = u_star - c_L;

          // Compute state in the left fan
          if(sH_L < 0.0 && sT_L > 0.0) {
            const auto m1_L_fan  = qL(M1_INDEX)*std::exp((vel_d_L - c_L)/c_L);
            const auto m2_L_fan  = qL(M2_INDEX)*std::exp((vel_d_L - c_L)/c_L);
            const auto rho_L_fan = m1_L_fan + m2_L_fan;

            q_star(M1_INDEX)             = m1_L_fan;
            q_star(M2_INDEX)             = m2_L_fan;
            q_star(RHO_ALPHA1_BAR_INDEX) = rho_L_fan*alpha1_bar_L;
            if(curr_d == 0) {
              q_star(RHO_U_INDEX)     = rho_L_fan*c_L; //TODO: Check this sign of the velocity
              q_star(RHO_U_INDEX + 1) = rho_L_fan*(qL(RHO_U_INDEX + 1)/rho_L);
            }
            else if(curr_d == 1) {
              q_star(RHO_U_INDEX)     = rho_L_fan*(qL(RHO_U_INDEX)/rho_L);
              q_star(RHO_U_INDEX + 1) = rho_L_fan*c_L;
            }
          }
          // Right of the left fan. Compute the state
          else if(sH_L < 0.0 && sT_L <= 0.0) {
            const auto m1_L_star  = qL(M1_INDEX)*std::exp((vel_d_L - u_star)/c_L);
            const auto m2_L_star  = qL(M2_INDEX)*std::exp((vel_d_L - u_star)/c_L);
            const auto rho_L_star = m1_L_star + m2_L_star;

            q_star(M1_INDEX)             = m1_L_star;
            q_star(M2_INDEX)             = m2_L_star;
            q_star(RHO_ALPHA1_BAR_INDEX) = rho_L_star*alpha1_bar_L;
            if(curr_d == 0) {
              q_star(RHO_U_INDEX)     = rho_L_star*u_star;
              q_star(RHO_U_INDEX + 1) = rho_L_star*(qL(RHO_U_INDEX + 1)/rho_L);
            }
            else if(curr_d == 1) {
              q_star(RHO_U_INDEX)     = rho_L_star*(qL(RHO_U_INDEX)/rho_L);
              q_star(RHO_U_INDEX + 1) = rho_L_star*u_star;
            }
          }
        }
      }
      // Right "connecting state"
      else {
        // 1-wave right shock
        if(p_star > p_bar_R) {
          const auto r = 1.0 + 1.0/((rho_R*c_R*c_R)/(p_star - p_bar_R));

          const auto m1_R_star  = qR(M1_INDEX)*r;
          const auto m2_R_star  = qR(M2_INDEX)*r;
          const auto rho_R_star = m1_R_star + m2_R_star;

          auto s_R = nan("");
          if(r > 1) {
            s_R = u_star + (vel_d_R - u_star)/(1.0 - r);
          }
          else if(r == 1) {
            s_R = u_star + (vel_d_R - u_star)/(-std::numeric_limits<double>::infinity()); //TODO: Understand why / and not * infinity
          }

          // If right of right shock, the state is qR
          if(std::isnan(s_R) || s_R < 0.0) {
            q_star = qR;
          }
          // Left of right shock, compute the state
          else {
            q_star(M1_INDEX)             = m1_R_star;
            q_star(M2_INDEX)             = m2_R_star;
            q_star(RHO_ALPHA1_BAR_INDEX) = rho_R_star*alpha1_bar_R;
            if(curr_d == 0) {
              q_star(RHO_U_INDEX)     = rho_R_star*u_star;
              q_star(RHO_U_INDEX + 1) = rho_R_star*(qR(RHO_U_INDEX + 1)/rho_R);
            }
            else if(curr_d == 1) {
              q_star(RHO_U_INDEX)     = rho_R_star*(qR(RHO_U_INDEX)/rho_R);
              q_star(RHO_U_INDEX + 1) = rho_R_star*u_star;
            }
          }
        }
        // 3-waves right fan
        else {
          const auto sH_R = vel_d_R + c_R;
          auto sT_R       = std::numeric_limits<double>::infinity();
          if(-(vel_d_R - u_star)/c_R < 100.0) {
            sT_R = u_star + c_R; // TODO: Check this sign of vel_d_R - u_star
          }

          // Right of right fan is qR
          if(sH_R < 0.0) {
            q_star = qR;
          }
          // Compute the state in the right fan
          else if(sH_R >= 0.0 && sT_R < 0.0) {
            const auto m1_R_fan  = qR(M1_INDEX)*std::exp(-(vel_d_R + c_R)/c_R);
            const auto m2_R_fan  = qR(M2_INDEX)*std::exp(-(vel_d_R + c_R)/c_R);
            const auto rho_R_fan = m1_R_fan + m2_R_fan;

            q_star(M1_INDEX)             = m1_R_fan;
            q_star(M2_INDEX)             = m2_R_fan;
            q_star(RHO_ALPHA1_BAR_INDEX) = rho_R_fan*alpha1_bar_R;
            if(curr_d == 0) {
              q_star(RHO_U_INDEX)     = -rho_R_fan*c_R; // TODO: Check this sign of the velocity
              q_star(RHO_U_INDEX + 1) = rho_R_fan*(qR(RHO_U_INDEX + 1)/rho_R);
            }
            else if(curr_d == 1) {
              q_star(RHO_U_INDEX)     = rho_R_fan*(qR(RHO_U_INDEX)/rho_R);
              q_star(RHO_U_INDEX + 1) = -rho_R_fan*c_R;
            }
          }
          // Compute state at the left of the right fan
          else {
            const auto m1_R_star  = qR(M1_INDEX)*std::exp(-(vel_d_R - u_star)/c_R);
            const auto m2_R_star  = qR(M2_INDEX)*std::exp(-(vel_d_R - u_star)/c_R);
            const auto rho_R_star = m1_R_star + m2_R_star ;

            q_star(M1_INDEX)             = m1_R_star;
            q_star(M2_INDEX)             = m2_R_star;
            q_star(RHO_ALPHA1_BAR_INDEX) = rho_R_star*alpha1_bar_R;
            if(curr_d == 0) {
              q_star(RHO_U_INDEX)     = rho_R_star*u_star;
              q_star(RHO_U_INDEX + 1) = rho_R_star*(qR(RHO_U_INDEX + 1)/rho_R);
            }
            else if(curr_d == 1) {
              q_star(RHO_U_INDEX)     = rho_R_star*(qR(RHO_U_INDEX)/rho_R);
              q_star(RHO_U_INDEX + 1) = rho_R_star*u_star;
            }
          }
        }
      }
    }

    // Compute the hyperbolic contribution to the flux
    FluxValue<typename Flux<Field>::cfg> res = this->evaluate_hyperbolic_operator(q_star, curr_d);

    // Add the contribution due to surface tension
    const auto mod_grad_alpha1_barL = std::sqrt(xt::sum(grad_alpha1_barL*grad_alpha1_barL)());
    const auto mod_grad_alpha1_barR = std::sqrt(xt::sum(grad_alpha1_barR*grad_alpha1_barR)());

    if(mod_grad_alpha1_barL > this->mod_grad_alpha1_bar_min &&
       mod_grad_alpha1_barR > this->mod_grad_alpha1_bar_min) {
      const auto nL = grad_alpha1_barL/mod_grad_alpha1_barL;
      const auto nR = grad_alpha1_barR/mod_grad_alpha1_barR;

      if(curr_d == 0) {
        res(RHO_U_INDEX) += 0.5*EquationData::sigma*((nL(0)*nL(0) - 1.0)*mod_grad_alpha1_barL +
                                                     (nR(0)*nR(0) - 1.0)*mod_grad_alpha1_barR);
        res(RHO_U_INDEX + 1) += 0.5*EquationData::sigma*(nL(0)*nL(1)*mod_grad_alpha1_barL +
                                                         nR(0)*nR(1)*mod_grad_alpha1_barR);
      }

      if(curr_d == 1) {
        res(RHO_U_INDEX) += 0.5*EquationData::sigma*(nL(0)*nL(1)*mod_grad_alpha1_barL +
                                                     nR(0)*nR(1)*mod_grad_alpha1_barR);
        res(RHO_U_INDEX + 1) += 0.5*EquationData::sigma*((nL(1)*nL(1) - 1.0)*mod_grad_alpha1_barL +
                                                         (nR(1)*nR(1) - 1.0)*mod_grad_alpha1_barR);
      }
    }

    return res;
  }

  // Implement the contribution of the discrete flux for all the cells in the mesh.
  //
  template<class Field>
  #ifdef ORDER_2
  auto GodunovFlux<Field>::make_two_scale_capillarity(const auto& grad_alpha1_bar, const auto& H)
  #else
  auto GodunovFlux<Field>::make_two_scale_capillarity(const auto& grad_alpha1_bar)
  #endif
  {
    FluxDefinition<typename Flux<Field>::cfg> Godunov_f;

    // Perform the loop over each dimension to compute the flux contribution
    static_for<0, EquationData::dim>::apply(
      // First, we need a function to compute the "continuous" flux
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        // Compute now the "discrete" flux function, in this case a Rusanov flux
        Godunov_f[d].cons_flux_function = [&](auto& cells, const Field& field)
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

                                              const double beta = 1.0;
                                              auto primL_recon = primL;
                                              auto primR_recon = primR;
                                              for(std::size_t comp = 0; comp < Field::size; ++comp) {
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

                                              FluxValue<typename Flux<Field>::cfg> qL = this->prim2cons(primL_recon);
                                              FluxValue<typename Flux<Field>::cfg> qR = this->prim2cons(primR_recon);

                                              // Relax the interpolated state
                                              const double tol    = 1e-12; /*--- Tolerance of the Newton method ---*/
                                              const double lambda = 0.9;   /*--- Parameter for bound preserving strategy ---*/

                                              // Start with left state
                                              std::size_t Newton_iter = 0;
                                              bool relaxation_applied = true;
                                              double dalpha1_barL     = std::numeric_limits<double>::infinity();
                                              double alpha1_barL      = qL(RHO_ALPHA1_BAR_INDEX)/
                                                                        (qL(M1_INDEX) + qL(M2_INDEX));
                                              while(relaxation_applied == true) {
                                                relaxation_applied = false;
                                                Newton_iter++;

                                                this->perform_Newton_step_relaxation(std::make_unique<decltype(qL)>(qL), H[left],
                                                                                     dalpha1_barL, alpha1_barL, relaxation_applied, tol, lambda);

                                                // Newton cycle diverged
                                                if(Newton_iter > 60) {
                                                  std::cout << "Netwon method not converged in the relaxation of left state after MUSCL" << std::endl;
                                                  exit(1);
                                                }
                                              }

                                              // Focus now on right state
                                              Newton_iter         = 0;
                                              relaxation_applied  = true;
                                              double dalpha1_barR = std::numeric_limits<double>::infinity();
                                              double alpha1_barR  = qR(RHO_ALPHA1_BAR_INDEX)/
                                                                    (qR(M1_INDEX) + qR(M2_INDEX));
                                              while(relaxation_applied == true) {
                                                relaxation_applied = false;
                                                Newton_iter++;

                                                this->perform_Newton_step_relaxation(std::make_unique<decltype(qR)>(qR), H[right],
                                                                                     dalpha1_barR, alpha1_barR, relaxation_applied, tol, lambda);

                                                // Newton cycle diverged
                                                if(Newton_iter > 60) {
                                                  std::cout << "Netwon method not converged in the relaxation of right state after MUSCL" << std::endl;
                                                  exit(1);
                                                }
                                              }
                                            #else
                                              // Compute the stencil and extract state
                                              const auto& left  = cells[0];
                                              const auto& right = cells[1];

                                              const FluxValue<typename Flux<Field>::cfg> qL = field[left];
                                              const FluxValue<typename Flux<Field>::cfg> qR = field[right];
                                            #endif

                                            // Check if we are at a cell with discontinuity in the state. This is not sufficient to say that the
                                            // flux is equal to the 'continuous' one because of surface tension, which involves gradients,
                                            // namely a non-local info
                                            bool is_discontinuous = false;
                                            for(std::size_t comp = 0; comp < Field::size; ++comp) {
                                              if(qL(comp) != qR(comp)) {
                                                is_discontinuous = true;
                                              }

                                              if(is_discontinuous)
                                                break;
                                            }

                                            // Compute the numerical flux
                                            return compute_discrete_flux(qL, qR, d,
                                                                         grad_alpha1_bar[left], grad_alpha1_bar[right],
                                                                         is_discontinuous);
                                          };
    });

    return make_flux_based_scheme(Godunov_f);
  }

} // end namespace samurai
