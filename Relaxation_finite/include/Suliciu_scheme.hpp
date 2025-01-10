#ifndef Suliciu_scheme_hpp
#define Suliciu_scheme_hpp

#include "flux_base.hpp"

namespace samurai {
  using namespace EquationData;

  /**
    * Implementation of the flux based on Suliciu-type relaxation
    */
  template<class Field>
  class RelaxationFlux: public Flux<Field> {
  public:
    RelaxationFlux(const EOS<>& EOS_phase1, const EOS<>& EOS_phase2); // Constructor which accepts in inputs the equations of state of the two phases

    auto make_flux(double& c); // Compute the flux over all cells.
                               // The input argument is employed to compute the Courant number

  private:
    template<typename T>
    inline T M0(const T nu, const T Me) const;

    template<typename T>
    inline T psi(const T u_star, const T a,
                 const T alphaL, const T alphaR,
                 const T vel_diesis, const T tauL_diesis, const T tauR_diesis) const;

    template<typename T>
    inline T Psi(const T u_star,
                 const T a1, const T alpha1L, const T alpha1R, const T vel1_diesis,
                 const T a2, const T alpha2L, const T alpha2R, const T vel2_diesis,
                 const T tau2L_diesis, const T tau2R_diesis) const;

    template<typename T>
    inline T dM0_dMe(const T nu, const T Me) const;

    template<typename T>
    inline T dpsi_dustar(const T u_star, const T a,
                         const T alphaL, const T alphaR,
                         const T vel_diesis, const T tauL_diesis, const T tauR_diesis) const;

    template<typename T>
    inline T dPsi_dustar(const T u_star,
                         const T a1, const T alpha1L, const T alpha1R,
                         const T a2, const T alpha2L, const T alpha2R,
                         const T vel2_diesis, const T tau2L_diesis, const T tau2R_diesis) const;

    template<typename T>
    T Newton(const T rhs,
             const T a1, const T alpha1L, const T alpha1R, const T vel1_diesis, const T tau1L_diesis, const T tau1R_diesis,
             const T a2, const T alpha2L, const T alpha2R, const T vel2_diesis, const T tau2L_diesis, const T tau2R_diesis,
             const double eps) const;

    template<typename T>
    void Riemann_solver_phase_vI(const T xi,
                                 const T alphaL, const T alphaR, const T tauL, const T tauR,
                                 const T wL, const T wR, const T pL, const T pR, const T EL, const T ER,
                                 const T a, const T u_star,
                                 T& alpha_m, T& tau_m, T& w_m, T& pres_m, T& E_m,
                                 T& alpha_p, T& tau_p, T& w_p, T& pres_p, T& E_p);

    template<typename T>
    void Riemann_solver_phase_pI(const T xi,
                                 const T alphaL, const T alphaR, const T tauL, const T tauR,
                                 const T wL, const T wR, const T pL, const T pR, const T EL, const T ER,
                                 const T w_diesis, const T tauL_diesis, const T tauR_diesis, const T a,
                                 T& alpha_m, T& tau_m, T& w_m, T& pres_m, T& E_m,
                                 T& alpha_p, T& tau_p, T& w_p, T& pres_p, T& E_p,
                                 T& w_star);

    #ifdef ORDER_2
      void compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                 const FluxValue<typename Flux<Field>::cfg>& qR,
                                 const typename Field::value_type alpha1L_order1,
                                 const typename Field::value_type alpha1R_order1,
                                 const std::size_t curr_d,
                                 FluxValue<typename Flux<Field>::cfg>& F_minus,
                                 FluxValue<typename Flux<Field>::cfg>& F_plus,
                                 double& c); // Compute discrete flux
    #else
      void compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                 const FluxValue<typename Flux<Field>::cfg>& qR,
                                 const std::size_t curr_d,
                                 FluxValue<typename Flux<Field>::cfg>& F_minus,
                                 FluxValue<typename Flux<Field>::cfg>& F_plus,
                                 double& c); // Compute discrete flux
    #endif
  };

  // Constructor derived from base class
  //
  template<class Field>
  RelaxationFlux<Field>::RelaxationFlux(const EOS<>& EOS_phase1, const EOS<>& EOS_phase2):
    Flux<Field>(EOS_phase1, EOS_phase2) {}

  // Implementation of the flux (F^{-} and F^{+} as in Saleh 2012 notation)
  //
  #ifdef ORDER_2
    template<class Field>
    void RelaxationFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                      const FluxValue<typename Flux<Field>::cfg>& qR,
                                                      const typename Field::value_type alpha1L_order1,
                                                      const typename Field::value_type alpha1R_order1,
                                                      std::size_t curr_d,
                                                      FluxValue<typename Flux<Field>::cfg>& F_minus,
                                                      FluxValue<typename Flux<Field>::cfg>& F_plus,
                                                      double& c) {
      // Compute the relevant variables from left state for phase 1
      const auto alpha1L = qL(ALPHA1_INDEX);
      const auto rho1L   = qL(ALPHA1_RHO1_INDEX)/alpha1L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
      const auto vel1L_d = qL(ALPHA1_RHO1_U1_INDEX + curr_d)/qL(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      const auto E1L     = qL(ALPHA1_RHO1_E1_INDEX)/qL(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      auto e1L           = E1L;
      for(std::size_t d = 0; d < Field::dim; ++d) {
        e1L -= 0.5*((qL(ALPHA1_RHO1_U1_INDEX + d)/qL(ALPHA1_RHO1_INDEX))*
                    (qL(ALPHA1_RHO1_U1_INDEX + d)/qL(ALPHA1_RHO1_INDEX))); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      }
      const auto p1L = this->phase1.pres_value_Rhoe(rho1L, e1L);

      // Compute the relevant variables from right state for phase 1
      const auto alpha1R = qR(ALPHA1_INDEX);
      const auto rho1R   = qR(ALPHA1_RHO1_INDEX)/alpha1R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
      const auto vel1R_d = qR(ALPHA1_RHO1_U1_INDEX + curr_d)/qR(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      const auto E1R     = qR(ALPHA1_RHO1_E1_INDEX)/qR(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      auto e1R           = E1R;
      for(std::size_t d = 0; d < Field::dim; ++d) {
        e1R -= 0.5*((qR(ALPHA1_RHO1_U1_INDEX + d)/qR(ALPHA1_RHO1_INDEX))*
                    (qR(ALPHA1_RHO1_U1_INDEX + d)/qR(ALPHA1_RHO1_INDEX))); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      }
      const auto p1R = this->phase1.pres_value_Rhoe(rho1R, e1R);

      // Compute the relevant variables from left state for phase 2
      const auto alpha2L = 1.0 - alpha1L;
      const auto rho2L   = qL(ALPHA2_RHO2_INDEX)/alpha2L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
      const auto vel2L_d = qL(ALPHA2_RHO2_U2_INDEX + curr_d)/qL(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      const auto E2L     = qL(ALPHA2_RHO2_E2_INDEX)/qL(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      auto e2L           = E2L;
      for(std::size_t d = 0; d < Field::dim; ++d) {
        e2L -= 0.5*((qL(ALPHA2_RHO2_U2_INDEX + d)/qL(ALPHA2_RHO2_INDEX))*
                    (qL(ALPHA2_RHO2_U2_INDEX + d)/qL(ALPHA2_RHO2_INDEX))); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      }
      const auto p2L = this->phase2.pres_value_Rhoe(rho2L, e2L);

      // Compute the relevant variables from right state for phase 2
      const auto alpha2R = 1.0 - alpha1R;
      const auto rho2R   = qR(ALPHA2_RHO2_INDEX)/alpha2R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
      const auto vel2R_d = qR(ALPHA2_RHO2_U2_INDEX + curr_d)/qR(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      const auto E2R     = qR(ALPHA2_RHO2_E2_INDEX)/qR(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      auto e2R           = E2R;
      for(std::size_t d = 0; d < Field::dim; ++d) {
        e2R -= 0.5*((qR(ALPHA2_RHO2_U2_INDEX + d)/qR(ALPHA2_RHO2_INDEX))*
                    (qR(ALPHA2_RHO2_U2_INDEX + d)/qR(ALPHA2_RHO2_INDEX))); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      }
      const auto p2R = this->phase2.pres_value_Rhoe(rho2R, e2R);

      // Compute first rhs of relaxation related parameters (Whitham's approach)
      auto a1 = std::max(this->phase1.c_value_RhoP(rho1L, p1L)*rho1L,
                         this->phase1.c_value_RhoP(rho1R, p1R)*rho1R);
      auto a2 = std::max(this->phase2.c_value_RhoP(rho2L, p2L)*rho2L,
                         this->phase2.c_value_RhoP(rho2R, p2R)*rho2R);

      /*--- Compute the transport step solving a non-linear equation with the Newton method ---*/

      // Compute "diesis" state (formulas (3.21) in Saleh ESAIM 2019, starting point for subsonic wave)
      using field_type = decltype(a2);
      field_type vel1_diesis, p1_diesis, tau1L_diesis = 0.0, tau1R_diesis = 0.0; /*--- NOTE: tau denotes the specific volume, i.e. the inverse of the density ---*/
      field_type vel2_diesis, p2_diesis, tau2L_diesis = 0.0, tau2R_diesis = 0.0;

      const double fact = 1.01; // Safety factor
      // Loop to be sure that tau_diesis variables are positive (theorem 3.5, Coquel et al. JCP 2017)
      while(tau1L_diesis <= 0.0 || tau1R_diesis <= 0.0) {
        a1 *= fact;
        vel1_diesis  = 0.5*(vel1L_d + vel1R_d) - 0.5*(p1R - p1L)/a1;
        p1_diesis    = 0.5*(p1R + p1L) - 0.5*a1*(vel1R_d - vel1L_d);
        tau1L_diesis = 1.0/rho1L + (vel1_diesis - vel1L_d)/a1;
        tau1R_diesis = 1.0/rho1R - (vel1_diesis - vel1R_d)/a1;
      }
      while(tau2L_diesis <= 0.0 || tau2R_diesis <= 0.0) {
        a2 *= fact;
        vel2_diesis  = 0.5*(vel2L_d + vel2R_d) - 0.5*(p2R - p2L)/a2;
        p2_diesis    = 0.5*(p2R + p2L) - 0.5*a2*(vel2R_d - vel2L_d);
        tau2L_diesis = 1.0/rho2L + (vel2_diesis - vel2L_d)/a2;
        tau2R_diesis = 1.0/rho2R - (vel2_diesis - vel2R_d)/a2;
      }

      // Update of a1 and a2 so that a solution for u* surely exists
      field_type rhs = 0.0, sup = 0.0, inf = 0.0;
      const double mu = 0.02;
      while(rhs - inf <= mu*(sup - inf) || sup - rhs <= mu*(sup - inf)) {
        if(vel1_diesis - a1*tau1L_diesis > vel2_diesis - a2*tau2L_diesis &&
           vel1_diesis + a1*tau1R_diesis < vel2_diesis + a2*tau2R_diesis) {
          a1 *= fact;
          vel1_diesis  = 0.5*(vel1L_d + vel1R_d) - 0.5/a1*(p1R - p1L);
          p1_diesis    = 0.5*(p1R + p1L) - 0.5*a1*(vel1R_d - vel1L_d);
          tau1L_diesis = 1.0/rho1L + 1.0/a1*(vel1_diesis - vel1L_d);
          tau1R_diesis = 1.0/rho1R - 1.0/a1*(vel1_diesis - vel1R_d);
        }
        else {
          if(vel2_diesis - a2*tau2L_diesis > vel1_diesis - a1*tau1L_diesis &&
             vel2_diesis + a2*tau2R_diesis < vel1_diesis + a1*tau1R_diesis) {
            a2 *= fact;
            vel2_diesis  = 0.5*(vel2L_d + vel2R_d) - 0.5/a2*(p2R - p2L);
            p2_diesis    = 0.5*(p2R + p2L) - 0.5*a2*(vel2R_d - vel2L_d);
            tau2L_diesis = 1.0/rho2L + 1.0/a2*(vel2_diesis - vel2L_d);
            tau2R_diesis = 1.0/rho2R - 1.0/a2*(vel2_diesis - vel2R_d);
          }
          else {
            a1 *= fact;
            vel1_diesis  = 0.5*(vel1L_d + vel1R_d) - 0.5/a1*(p1R - p1L);
            p1_diesis    = 0.5*(p1R + p1L) - 0.5*a1*(vel1R_d - vel1L_d);
            tau1L_diesis = 1.0/rho1L + 1.0/a1*(vel1_diesis - vel1L_d);
            tau1R_diesis = 1.0/rho1R - 1.0/a1*(vel1_diesis - vel1R_d);

            a2 *= fact;
            vel2_diesis  = 0.5*(vel2L_d + vel2R_d) - 0.5/a2*(p2R - p2L);
            p2_diesis    = 0.5*(p2R + p2L) - 0.5*a2*(vel2R_d - vel2L_d);
            tau2L_diesis = 1.0/rho2L + 1.0/a2*(vel2_diesis - vel2L_d);
            tau2R_diesis = 1.0/rho2R - 1.0/a2*(vel2_diesis - vel2R_d);
          }
        }

        // Compute the rhs of the equation for u*
        rhs = -p1_diesis*(alpha1R-alpha1L) -p2_diesis*(alpha2R-alpha2L);

        // Limits on u* so that the relative Mach number is below one
        const auto cLmax = std::max(vel1_diesis - a1*tau1L_diesis, vel2_diesis - a2*tau2L_diesis);
        const auto cRmin = std::min(vel1_diesis + a1*tau1R_diesis, vel2_diesis + a2*tau2R_diesis);

        // Bounds on the function Psi
        inf = Psi(cLmax, a1, alpha1L, alpha1R, vel1_diesis,
                         a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis);
        sup = Psi(cRmin, a1, alpha1L, alpha1R, vel1_diesis,
                         a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis);

      }

      // Look for u* in the interval [cLmax, cRmin] such that Psi(u*) = rhs
      const double eps   = 1e-7;
      const auto uI_star = Newton(rhs, a1, alpha1L, alpha1R, vel1_diesis, tau1L_diesis, tau1R_diesis,
                                       a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis, eps);

      // Compute the "fluxes"
      field_type alpha1_m, tau1_m, u1_m, p1_m, E1_m,
                 alpha1_p, tau1_p, u1_p, p1_p, E1_p,
                 alpha2_m, tau2_m, u2_m, p2_m, E2_m, w2_m,
                 alpha2_p, tau2_p, u2_p, p2_p, E2_p, w2_p,
                 u2_star;
      Riemann_solver_phase_pI(-uI_star,
                              alpha2L, alpha2R, 1.0/rho2L, 1.0/rho2R, vel2L_d - uI_star, vel2R_d - uI_star,
                              p2L, p2R, E2L - (vel2L_d - uI_star)*uI_star - 0.5*uI_star*uI_star, E2R - (vel2R_d - uI_star)*uI_star - 0.5*uI_star*uI_star,
                              vel2_diesis - uI_star, tau2L_diesis, tau2R_diesis, a2,
                              alpha2_m, tau2_m, w2_m, p2_m, E2_m,
                              alpha2_p, tau2_p, w2_p, p2_p, E2_p,
                              u2_star);
      u2_m = w2_m + uI_star;
      E2_m += (u2_m - uI_star)*uI_star + 0.5*uI_star*uI_star;
      u2_p = w2_p + uI_star;
      E2_p += (u2_p - uI_star)*uI_star + 0.5*uI_star*uI_star;
      u2_star += uI_star;
      Riemann_solver_phase_vI(0.0,
                              alpha1L, alpha1R, 1.0/rho1L, 1.0/rho1R, vel1L_d, vel1R_d, p1L, p1R, E1L, E1R,
                              a1, uI_star,
                              alpha1_m, tau1_m, u1_m, p1_m, E1_m,
                              alpha1_p, tau1_p, u1_p, p1_p, E1_p);

      // Build the "fluxes"
      F_minus(ALPHA1_INDEX) = 0.0;

      F_minus(ALPHA1_RHO1_INDEX)             = alpha1_m/tau1_m*u1_m;
      F_minus(ALPHA1_RHO1_U1_INDEX + curr_d) = alpha1_m/tau1_m*u1_m*u1_m + alpha1_m*p1_m;
      const auto u1_star = uI_star;
      for(std::size_t d = 0; d < Field::dim; ++d) {
        if(d != curr_d) {
          F_minus(ALPHA1_RHO1_U1_INDEX + d) = 0.5*u1_star*(qL(ALPHA1_RHO1_INDEX) + qR(ALPHA1_RHO1_INDEX))
                                            - 0.5*std::abs(u1_star)*(qR(ALPHA1_RHO1_INDEX) - qL(ALPHA1_RHO1_INDEX));
          F_plus(ALPHA1_RHO1_U1_INDEX + d) = F_minus(ALPHA1_RHO1_U1_INDEX + d);
        }
      }
      F_minus(ALPHA1_RHO1_E1_INDEX)          = alpha1_m/tau1_m*E1_m*u1_m + alpha1_m*p1_m*u1_m;

      F_minus(ALPHA2_RHO2_INDEX)             = alpha2_m/tau2_m*u2_m;
      F_minus(ALPHA2_RHO2_U2_INDEX + curr_d) = alpha2_m/tau2_m*u2_m*u2_m + alpha2_m*p2_m;
      for(std::size_t d = 0; d < Field::dim; ++d) {
        if(d != curr_d) {
          F_minus(ALPHA2_RHO2_U2_INDEX + d) = 0.5*u2_star*(qL(ALPHA2_RHO2_INDEX) + qR(ALPHA1_RHO1_U1_INDEX))
                                            - 0.5*std::abs(u2_star)*(qR(ALPHA2_RHO2_INDEX) - qL(ALPHA2_RHO2_INDEX));
          F_plus(ALPHA2_RHO2_U2_INDEX + d) = F_minus(ALPHA2_RHO2_U2_INDEX + d);
        }
      }
      F_minus(ALPHA2_RHO2_E2_INDEX) = alpha2_m/tau2_m*E2_m*u2_m + alpha2_m*p2_m*u2_m;

      F_plus(ALPHA1_INDEX) = 0.0;

      F_plus(ALPHA1_RHO1_INDEX)             = alpha1_p/tau1_p*u1_p;
      F_plus(ALPHA1_RHO1_U1_INDEX + curr_d) = alpha1_p/tau1_p*u1_p*u1_p + alpha1_p*p1_p;
      F_plus(ALPHA1_RHO1_E1_INDEX)          = alpha1_p/tau1_p*E1_p*u1_p + alpha1_p*p1_p*u1_p;

      F_plus(ALPHA2_RHO2_INDEX)             = alpha2_p/tau2_p*u2_p;
      F_plus(ALPHA2_RHO2_U2_INDEX + curr_d) = alpha2_p/tau2_p*u2_p*u2_p + alpha2_p*p2_p;
      F_plus(ALPHA2_RHO2_E2_INDEX)          = alpha2_p/tau2_p*E2_p*u2_p + alpha2_p*p2_p*u2_p;

      // Focus on non-conservative term
      const auto pidxalpha2 = p2_diesis*((1.0 - alpha1R_order1) - (1.0 - alpha1L_order1))
                            + psi(uI_star, a2, 1.0 - alpha1L_order1, 1.0 - alpha1R_order1, vel2_diesis, tau2L_diesis, tau2R_diesis);

      if(uI_star < 0.0) {
        F_minus(ALPHA1_INDEX) -= -uI_star*(alpha1R_order1 - alpha1L_order1);

        F_minus(ALPHA1_RHO1_U1_INDEX + curr_d) -= -pidxalpha2;
        F_minus(ALPHA1_RHO1_E1_INDEX) -= -uI_star*pidxalpha2;

        F_minus(ALPHA2_RHO2_U2_INDEX + curr_d) -= pidxalpha2;
        F_minus(ALPHA2_RHO2_E2_INDEX) -= uI_star*pidxalpha2;
      }
      else {
        F_plus(ALPHA1_INDEX) += -uI_star*(alpha1R_order1 - alpha1L_order1);

        F_plus(ALPHA1_RHO1_U1_INDEX + curr_d) += -pidxalpha2;
        F_plus(ALPHA1_RHO1_E1_INDEX) += -uI_star*pidxalpha2;

        F_plus(ALPHA2_RHO2_U2_INDEX + curr_d) += pidxalpha2;
        F_plus(ALPHA2_RHO2_E2_INDEX) += uI_star*pidxalpha2;
      }

      c = std::max(c, std::max(std::max(std::abs(vel1L_d - a1/rho1L), std::abs(vel1R_d + a1/rho1R)),
                               std::max(std::abs(vel2L_d - a2/rho2L), std::abs(vel2R_d + a2/rho2R))));
    }
  #else
    template<class Field>
    void RelaxationFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                      const FluxValue<typename Flux<Field>::cfg>& qR,
                                                      std::size_t curr_d,
                                                      FluxValue<typename Flux<Field>::cfg>& F_minus,
                                                      FluxValue<typename Flux<Field>::cfg>& F_plus,
                                                      double& c) {
      // Compute the relevant variables from left state for phase 1
      const auto alpha1L = qL(ALPHA1_INDEX);
      const auto rho1L   = qL(ALPHA1_RHO1_INDEX)/alpha1L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
      const auto vel1L_d = qL(ALPHA1_RHO1_U1_INDEX + curr_d)/qL(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      const auto E1L     = qL(ALPHA1_RHO1_E1_INDEX)/qL(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      auto e1L           = E1L;
      for(std::size_t d = 0; d < Field::dim; ++d) {
        e1L -= 0.5*((qL(ALPHA1_RHO1_U1_INDEX + d)/qL(ALPHA1_RHO1_INDEX))*
                    (qL(ALPHA1_RHO1_U1_INDEX + d)/qL(ALPHA1_RHO1_INDEX))); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      }
      const auto p1L = this->phase1.pres_value_Rhoe(rho1L, e1L);

      // Compute the relevant variables from right state for phase 1
      const auto alpha1R = qR(ALPHA1_INDEX);
      const auto rho1R   = qR(ALPHA1_RHO1_INDEX)/alpha1R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
      const auto vel1R_d = qR(ALPHA1_RHO1_U1_INDEX + curr_d)/qR(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      const auto E1R     = qR(ALPHA1_RHO1_E1_INDEX)/qR(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      auto e1R           = E1R;
      for(std::size_t d = 0; d < Field::dim; ++d) {
        e1R -= 0.5*((qR(ALPHA1_RHO1_U1_INDEX + d)/qR(ALPHA1_RHO1_INDEX))*
                    (qR(ALPHA1_RHO1_U1_INDEX + d)/qR(ALPHA1_RHO1_INDEX))); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      }
      const auto p1R = this->phase1.pres_value_Rhoe(rho1R, e1R);

      // Compute the relevant variables from left state for phase 2
      const auto alpha2L = 1.0 - alpha1L;
      const auto rho2L   = qL(ALPHA2_RHO2_INDEX)/alpha2L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
      const auto vel2L_d = qL(ALPHA2_RHO2_U2_INDEX + curr_d)/qL(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      const auto E2L     = qL(ALPHA2_RHO2_E2_INDEX)/qL(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      auto e2L           = E2L;
      for(std::size_t d = 0; d < Field::dim; ++d) {
        e2L -= 0.5*((qL(ALPHA2_RHO2_U2_INDEX + d)/qL(ALPHA2_RHO2_INDEX))*
                    (qL(ALPHA2_RHO2_U2_INDEX + d)/qL(ALPHA2_RHO2_INDEX))); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      }
      const auto p2L = this->phase2.pres_value_Rhoe(rho2L, e2L);

      // Compute the relevant variables from right state for phase 2
      const auto alpha2R = 1.0 - alpha1R;
      const auto rho2R   = qR(ALPHA2_RHO2_INDEX)/alpha2R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
      const auto vel2R_d = qR(ALPHA2_RHO2_U2_INDEX + curr_d)/qR(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      const auto E2R     = qR(ALPHA2_RHO2_E2_INDEX)/qR(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      auto e2R           = E2R;
      for(std::size_t d = 0; d < Field::dim; ++d) {
        e2R -= 0.5*((qR(ALPHA2_RHO2_U2_INDEX + d)/qR(ALPHA2_RHO2_INDEX))*
                    (qR(ALPHA2_RHO2_U2_INDEX + d)/qR(ALPHA2_RHO2_INDEX))); /*--- TODO: Add treatment for vanishing volume fraction ---*/
      }
      const auto p2R = this->phase2.pres_value_Rhoe(rho2R, e2R);

      // Compute first rhs of relaxation related parameters (Whitham's approach)
      auto a1 = std::max(this->phase1.c_value_RhoP(rho1L, p1L)*rho1L,
                         this->phase1.c_value_RhoP(rho1R, p1R)*rho1R);
      auto a2 = std::max(this->phase2.c_value_RhoP(rho2L, p2L)*rho2L,
                         this->phase2.c_value_RhoP(rho2R, p2R)*rho2R);

      /*--- Compute the transport step solving a non-linear equation with the Newton method ---*/

      // Compute "diesis" state (formulas (3.21) in Saleh ESAIM 2019, starting point for subsonic wave)
      using field_type = decltype(a2);
      field_type vel1_diesis, p1_diesis, tau1L_diesis = 0.0, tau1R_diesis = 0.0; /*--- NOTE: tau denotes the specific volume, i.e. the inverse of the density ---*/
      field_type vel2_diesis, p2_diesis, tau2L_diesis = 0.0, tau2R_diesis = 0.0;

      const double fact = 1.01; // Safety factor
      // Loop to be sure that tau_diesis variables are positive (theorem 3.5, Coquel et al. JCP 2017)
      while(tau1L_diesis <= 0.0 || tau1R_diesis <= 0.0) {
        a1 *= fact;
        vel1_diesis  = 0.5*(vel1L_d + vel1R_d) - 0.5*(p1R - p1L)/a1;
        p1_diesis    = 0.5*(p1R + p1L) - 0.5*a1*(vel1R_d - vel1L_d);
        tau1L_diesis = 1.0/rho1L + (vel1_diesis - vel1L_d)/a1;
        tau1R_diesis = 1.0/rho1R - (vel1_diesis - vel1R_d)/a1;
      }
      while(tau2L_diesis <= 0.0 || tau2R_diesis <= 0.0) {
        a2 *= fact;
        vel2_diesis  = 0.5*(vel2L_d + vel2R_d) - 0.5*(p2R - p2L)/a2;
        p2_diesis    = 0.5*(p2R + p2L) - 0.5*a2*(vel2R_d - vel2L_d);
        tau2L_diesis = 1.0/rho2L + (vel2_diesis - vel2L_d)/a2;
        tau2R_diesis = 1.0/rho2R - (vel2_diesis - vel2R_d)/a2;
      }

      // Update of a1 and a2 so that a solution for u* surely exists
      field_type rhs = 0.0, sup = 0.0, inf = 0.0;
      const double mu = 0.02;
      while(rhs - inf <= mu*(sup - inf) || sup - rhs <= mu*(sup - inf)) {
        if(vel1_diesis - a1*tau1L_diesis > vel2_diesis - a2*tau2L_diesis &&
           vel1_diesis + a1*tau1R_diesis < vel2_diesis + a2*tau2R_diesis) {
          a1 *= fact;
          vel1_diesis  = 0.5*(vel1L_d + vel1R_d) - 0.5/a1*(p1R - p1L);
          p1_diesis    = 0.5*(p1R + p1L) - 0.5*a1*(vel1R_d - vel1L_d);
          tau1L_diesis = 1.0/rho1L + 1.0/a1*(vel1_diesis - vel1L_d);
          tau1R_diesis = 1.0/rho1R - 1.0/a1*(vel1_diesis - vel1R_d);
        }
        else {
          if(vel2_diesis - a2*tau2L_diesis > vel1_diesis - a1*tau1L_diesis &&
             vel2_diesis + a2*tau2R_diesis < vel1_diesis + a1*tau1R_diesis) {
            a2 *= fact;
            vel2_diesis  = 0.5*(vel2L_d + vel2R_d) - 0.5/a2*(p2R - p2L);
            p2_diesis    = 0.5*(p2R + p2L) - 0.5*a2*(vel2R_d - vel2L_d);
            tau2L_diesis = 1.0/rho2L + 1.0/a2*(vel2_diesis - vel2L_d);
            tau2R_diesis = 1.0/rho2R - 1.0/a2*(vel2_diesis - vel2R_d);
          }
          else {
            a1 *= fact;
            vel1_diesis  = 0.5*(vel1L_d + vel1R_d) - 0.5/a1*(p1R - p1L);
            p1_diesis    = 0.5*(p1R + p1L) - 0.5*a1*(vel1R_d - vel1L_d);
            tau1L_diesis = 1.0/rho1L + 1.0/a1*(vel1_diesis - vel1L_d);
            tau1R_diesis = 1.0/rho1R - 1.0/a1*(vel1_diesis - vel1R_d);

            a2 *= fact;
            vel2_diesis  = 0.5*(vel2L_d + vel2R_d) - 0.5/a2*(p2R - p2L);
            p2_diesis    = 0.5*(p2R + p2L) - 0.5*a2*(vel2R_d - vel2L_d);
            tau2L_diesis = 1.0/rho2L + 1.0/a2*(vel2_diesis - vel2L_d);
            tau2R_diesis = 1.0/rho2R - 1.0/a2*(vel2_diesis - vel2R_d);
          }
        }

        // Compute the rhs of the equation for u*
        rhs = -p1_diesis*(alpha1R-alpha1L) -p2_diesis*(alpha2R-alpha2L);

        // Limits on u* so that the relative Mach number is below one
        const auto cLmax = std::max(vel1_diesis - a1*tau1L_diesis, vel2_diesis - a2*tau2L_diesis);
        const auto cRmin = std::min(vel1_diesis + a1*tau1R_diesis, vel2_diesis + a2*tau2R_diesis);

        // Bounds on the function Psi
        inf = Psi(cLmax, a1, alpha1L, alpha1R, vel1_diesis,
                         a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis);
        sup = Psi(cRmin, a1, alpha1L, alpha1R, vel1_diesis,
                         a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis);

      }

      // Look for u* in the interval [cLmax, cRmin] such that Psi(u*) = rhs
      const double eps   = 1e-7;
      const auto uI_star = Newton(rhs, a1, alpha1L, alpha1R, vel1_diesis, tau1L_diesis, tau1R_diesis,
                                       a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis, eps);

      // Compute the "fluxes"
      field_type alpha1_m, tau1_m, u1_m, p1_m, E1_m,
                 alpha1_p, tau1_p, u1_p, p1_p, E1_p,
                 alpha2_m, tau2_m, u2_m, p2_m, E2_m, w2_m,
                 alpha2_p, tau2_p, u2_p, p2_p, E2_p, w2_p,
                 u2_star;
      Riemann_solver_phase_pI(-uI_star,
                              alpha2L, alpha2R, 1.0/rho2L, 1.0/rho2R, vel2L_d - uI_star, vel2R_d - uI_star,
                              p2L, p2R, E2L - (vel2L_d - uI_star)*uI_star - 0.5*uI_star*uI_star, E2R - (vel2R_d - uI_star)*uI_star - 0.5*uI_star*uI_star,
                              vel2_diesis - uI_star, tau2L_diesis, tau2R_diesis, a2,
                              alpha2_m, tau2_m, w2_m, p2_m, E2_m,
                              alpha2_p, tau2_p, w2_p, p2_p, E2_p,
                              u2_star);
      u2_m = w2_m + uI_star;
      E2_m += (u2_m - uI_star)*uI_star + 0.5*uI_star*uI_star;
      u2_p = w2_p + uI_star;
      E2_p += (u2_p - uI_star)*uI_star + 0.5*uI_star*uI_star;
      u2_star += uI_star;
      Riemann_solver_phase_vI(0.0,
                              alpha1L, alpha1R, 1.0/rho1L, 1.0/rho1R, vel1L_d, vel1R_d, p1L, p1R, E1L, E1R,
                              a1, uI_star,
                              alpha1_m, tau1_m, u1_m, p1_m, E1_m,
                              alpha1_p, tau1_p, u1_p, p1_p, E1_p);

      // Build the "fluxes"
      F_minus(ALPHA1_INDEX) = 0.0;

      F_minus(ALPHA1_RHO1_INDEX)             = alpha1_m/tau1_m*u1_m;
      F_minus(ALPHA1_RHO1_U1_INDEX + curr_d) = alpha1_m/tau1_m*u1_m*u1_m + alpha1_m*p1_m;
      const auto u1_star = uI_star;
      for(std::size_t d = 0; d < Field::dim; ++d) {
        if(d != curr_d) {
          F_minus(ALPHA1_RHO1_U1_INDEX + d) = 0.5*u1_star*(qL(ALPHA1_RHO1_INDEX) + qR(ALPHA1_RHO1_INDEX))
                                            - 0.5*std::abs(u1_star)*(qR(ALPHA1_RHO1_INDEX) - qL(ALPHA1_RHO1_INDEX));
          F_plus(ALPHA1_RHO1_U1_INDEX + d) = F_minus(ALPHA1_RHO1_U1_INDEX + d);
        }
      }
      F_minus(ALPHA1_RHO1_E1_INDEX)          = alpha1_m/tau1_m*E1_m*u1_m + alpha1_m*p1_m*u1_m;

      F_minus(ALPHA2_RHO2_INDEX)             = alpha2_m/tau2_m*u2_m;
      F_minus(ALPHA2_RHO2_U2_INDEX + curr_d) = alpha2_m/tau2_m*u2_m*u2_m + alpha2_m*p2_m;
      for(std::size_t d = 0; d < Field::dim; ++d) {
        if(d != curr_d) {
          F_minus(ALPHA2_RHO2_U2_INDEX + d) = 0.5*u2_star*(qL(ALPHA2_RHO2_INDEX) + qR(ALPHA1_RHO1_U1_INDEX))
                                            - 0.5*std::abs(u2_star)*(qR(ALPHA2_RHO2_INDEX) - qL(ALPHA2_RHO2_INDEX));
          F_plus(ALPHA2_RHO2_U2_INDEX + d) = F_minus(ALPHA2_RHO2_U2_INDEX + d);
        }
      }
      F_minus(ALPHA2_RHO2_E2_INDEX) = alpha2_m/tau2_m*E2_m*u2_m + alpha2_m*p2_m*u2_m;

      F_plus(ALPHA1_INDEX) = 0.0;

      F_plus(ALPHA1_RHO1_INDEX)             = alpha1_p/tau1_p*u1_p;
      F_plus(ALPHA1_RHO1_U1_INDEX + curr_d) = alpha1_p/tau1_p*u1_p*u1_p + alpha1_p*p1_p;
      F_plus(ALPHA1_RHO1_E1_INDEX)          = alpha1_p/tau1_p*E1_p*u1_p + alpha1_p*p1_p*u1_p;

      F_plus(ALPHA2_RHO2_INDEX)             = alpha2_p/tau2_p*u2_p;
      F_plus(ALPHA2_RHO2_U2_INDEX + curr_d) = alpha2_p/tau2_p*u2_p*u2_p + alpha2_p*p2_p;
      F_plus(ALPHA2_RHO2_E2_INDEX)          = alpha2_p/tau2_p*E2_p*u2_p + alpha2_p*p2_p*u2_p;

      // Focus on non-conservative term
      const auto pidxalpha2 = p2_diesis*(alpha2R - alpha2L) + psi(uI_star, a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis);

      if(uI_star < 0.0) {
        F_minus(ALPHA1_INDEX) -= -uI_star*(alpha1R - alpha1L);

        F_minus(ALPHA1_RHO1_U1_INDEX + curr_d) -= -pidxalpha2;
        F_minus(ALPHA1_RHO1_E1_INDEX) -= -uI_star*pidxalpha2;

        F_minus(ALPHA2_RHO2_U2_INDEX + curr_d) -= pidxalpha2;
        F_minus(ALPHA2_RHO2_E2_INDEX) -= uI_star*pidxalpha2;
      }
      else {
        F_plus(ALPHA1_INDEX) += -uI_star*(alpha1R - alpha1L);

        F_plus(ALPHA1_RHO1_U1_INDEX + curr_d) += -pidxalpha2;
        F_plus(ALPHA1_RHO1_E1_INDEX) += -uI_star*pidxalpha2;

        F_plus(ALPHA2_RHO2_U2_INDEX + curr_d) += pidxalpha2;
        F_plus(ALPHA2_RHO2_E2_INDEX) += uI_star*pidxalpha2;
      }

      c = std::max(c, std::max(std::max(std::abs(vel1L_d - a1/rho1L), std::abs(vel1R_d + a1/rho1R)),
                               std::max(std::abs(vel2L_d - a2/rho2L), std::abs(vel2R_d + a2/rho2R))));
    }
  #endif

  // Implement the contribution of the discrete flux for all the directions.
  //
  template<class Field>
  auto RelaxationFlux<Field>::make_flux(double& c) {
    FluxDefinition<typename Flux<Field>::cfg> discrete_flux;

    // Perform the loop over each dimension to compute the flux contribution
    static_for<0, Field::dim>::apply(
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        // Compute now the "discrete" non-conservative flux function
        discrete_flux[d].flux_function = [&](auto& cells, const Field& field)
                                            {
                                              FluxValue<typename Flux<Field>::cfg> F_minus,
                                                                                   F_plus;

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

                                                compute_discrete_flux(qL, qR,
                                                                      field[left](ALPHA1_INDEX), field[right](ALPHA1_INDEX),
                                                                      d, F_minus, F_plus, c);
                                              #else
                                                // Compute the stencil and extract state
                                                const auto& left  = cells[0];
                                                const auto& right = cells[1];

                                                const FluxValue<typename Flux<Field>::cfg>& qL = field[left];
                                                const FluxValue<typename Flux<Field>::cfg>& qR = field[right];

                                                compute_discrete_flux(qL, qR, d, F_minus, F_plus, c);
                                              #endif

                                              samurai::FluxValuePair<typename Flux<Field>::cfg> flux;
                                              flux[0] = F_minus;
                                              flux[1] = -F_plus;

                                              return flux;
                                            };
      }
    );

    return make_flux_based_scheme(discrete_flux);
  }

  // Implement M0 function (3.312 Saleh 2012, 3.30 Saleh ESAIM 2019)
  //
  template<class Field>
  template<typename T>
  inline T RelaxationFlux<Field>::M0(const T nu, const T Me) const {
    return 4.0/(nu + 1.0)*
           Me/((1.0 + Me*Me)*
               (1.0 + std::sqrt(std::abs(1.0 - 4.0*nu/((nu + 1.0)*(nu + 1.0))*
                                               4.0*Me*Me/((1.0 + Me*Me)*(1.0 + Me*Me))))));
  }

  // Implement psi function (Saleh 2012 ??)
  //
  template<class Field>
  template<typename T>
  inline T RelaxationFlux<Field>::psi(const T u_star, const T a,
                                      const T alphaL, const T alphaR,
                                      const T vel_diesis, const T tauL_diesis, const T tauR_diesis) const {
    if(u_star <= vel_diesis) {
      return a*(alphaL + alphaR)*(u_star - vel_diesis) +
             2.0*a*a*alphaL*tauL_diesis*M0(alphaL/alphaR, (vel_diesis - u_star)/(a*tauL_diesis));
    }

    return -psi(-u_star, a, alphaR, alphaL, -vel_diesis, tauR_diesis, tauL_diesis);
  }

  // Implement Psi function (3.3.15 Saleh 2012 ??)
  //
  template<class Field>
  template<typename T>
  inline T RelaxationFlux<Field>::Psi(const T u_star,
                                      const T a1, const T alpha1L, const T alpha1R, const T vel1_diesis,
                                      const T a2, const T alpha2L, const T alpha2R, const T vel2_diesis,
                                      const T tau2L_diesis, const T tau2R_diesis) const {
    return a1*(alpha1L + alpha1R)*(u_star - vel1_diesis) +
           psi(u_star, a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis);
  }

  // Implement the derivative of M0 w.r.t Me for the Newton method
  //
  template<class Field>
  template<typename T>
  inline T RelaxationFlux<Field>::dM0_dMe(const T nu, const T Me) const {
    const T w = (1.0 - Me)/(1.0 + Me);

    return 4.0/(nu + 1.0)*w/((1.0 + w*w)*(1.0 + w*w))*(1.0 + w)*(1.0 + w)/
           (1.0 - 4.0*nu/((nu + 1.0)*(nu + 1.0))*(1.0 - w*w)*(1.0 - w*w)/((1.0 + w*w)*(1.0 + w*w)) +
            std::sqrt(std::abs(1.0 - 4.0*nu/((nu + 1.0)*(nu + 1.0))*(1.0 - w*w)*(1.0 - w*w)/((1.0 + w*w)*(1.0 + w*w)))));
  }

  // Implement the derivative of psi w.r.t. u* for the Newton method
  //
  template<class Field>
  template<typename T>
  inline T RelaxationFlux<Field>::dpsi_dustar(const T u_star, const T a,
                                              const T alphaL, const T alphaR,
                                              const T vel_diesis, const T tauL_diesis, const T tauR_diesis) const {
    if(u_star <= vel_diesis) {
      return a*(alphaL + alphaR) -
             2.0*a*alphaL*dM0_dMe(alphaL/alphaR, (vel_diesis - u_star)/(a*tauL_diesis));
    }

    return a*(alphaL + alphaR) -
           2.0*a*alphaR*dM0_dMe(alphaR/alphaL, (vel_diesis - u_star)/(a*tauR_diesis));
  }

  // Implement the derivative of Psi w.r.t. u* for the Newton method
  //
  template<class Field>
  template<typename T>
  inline T RelaxationFlux<Field>::dPsi_dustar(const T u_star,
                                              const T a1, const T alpha1L, const T alpha1R,
                                              const T a2, const T alpha2L, const T alpha2R,
                                              const T vel2_diesis, const T tau2L_diesis, const T tau2R_diesis) const {
    return a1*(alpha1L + alpha1R) + dpsi_dustar(u_star, a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis);
  }

  // Newton method to compute u*
  //
  template<class Field>
  template<typename T>
  T RelaxationFlux<Field>::Newton(const T rhs,
                                  const T a1, const T alpha1L, const T alpha1R, const T vel1_diesis, const T tau1L_diesis, const T tau1R_diesis,
                                  const T a2, const T alpha2L, const T alpha2R, const T vel2_diesis, const T tau2L_diesis, const T tau2R_diesis,
                                  const double eps) const {
    if(alpha1L == alpha1R) {
      return vel1_diesis;
    }
    else {
      unsigned int iter = 0;
      const T xl = std::max(vel1_diesis - a1*tau1L_diesis, vel2_diesis - a2*tau2L_diesis);
      const T xr = std::min(vel1_diesis + a1*tau1R_diesis, vel2_diesis + a2*tau2R_diesis);

      T u_star = 0.5*(xl + xr);

      T du = -(Psi(u_star, a1, alpha1L, alpha1R, vel1_diesis,
                           a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis) - rhs)/
              (dPsi_dustar(u_star, a1, alpha1L, alpha1R,
                                   a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis));

      while(iter < 50 &&
            std::abs(Psi(u_star, a1, alpha1L, alpha1R, vel1_diesis,
                                 a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis) - rhs) > eps &&
            std::abs(du) > eps) {
        ++iter;

        u_star += du;

        du = -(Psi(u_star, a1, alpha1L, alpha1R, vel1_diesis,
                           a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis) - rhs)/
              (dPsi_dustar(u_star, a1, alpha1L, alpha1R,
                                   a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis));
      }

      // Safety check
      if(iter == 50) {
        std::cout << "Newton method not converged." << std::endl;
        exit(0);
      }

      return u_star;
    }
  }

  // Riemann solver for the phase associated to the interfacial velocity
  //
  template<class Field>
  template<typename T>
  void RelaxationFlux<Field>::Riemann_solver_phase_vI(const T xi,
                                                      const T alphaL, const T alphaR, const T tauL, const T tauR,
                                                      const T wL, const T wR, const T pL, const T pR, const T EL, const T ER,
                                                      const T a, const T u_star,
                                                      T& alpha_m, T& tau_m, T& w_m, T& pres_m, T& E_m,
                                                      T& alpha_p, T& tau_p, T& w_p, T& pres_p, T& E_p) {
    if(xi < wL-a*tauL) {
      alpha_m = alphaL;
      tau_m   = tauL;
      w_m     = wL;
      pres_m  = pL;
      E_m     = EL;

      alpha_p = alphaL;
      tau_p   = tauL;
      w_p     = wL;
      pres_p  = pL;
      E_p     = EL;
    }
    else {
      if(xi == wL-a*tauL) {
        alpha_m = alphaL;
        tau_m   = tauL;
        w_m     = wL;
        pres_m  = pL;
        E_m     = EL;

        alpha_p = alphaL;
        tau_p   = tauL + 1./a*(u_star - wL);
        w_p     = u_star;
        pres_p  = pL + a*(wL - u_star);
        E_p     = EL - 1.0/a*(pres_p*w_p - pL*wL);
      }
      else {
        if(xi > wL - a*tauL && xi < u_star) {
          alpha_m = alphaL;
          tau_m   = tauL + 1.0/a*(u_star - wL);
          w_m     = u_star;
          pres_m  = pL + a*(wL - u_star);
          E_m     = EL - 1.0/a*(pres_m*w_m - pL*wL);

          alpha_p = alphaL;
          tau_p   = tauL + 1.0/a*(u_star - wL);
          w_p     = u_star;
          pres_p  = pL + a*(wL - u_star);
          E_p     = EL - 1.0/a*(pres_p*w_p - pL*wL);
        }
        else {
          if(xi == u_star) {
            alpha_m = alphaL;
            tau_m   = tauL + 1.0/a*(u_star - wL);
            w_m     = u_star;
            pres_m  = pL + a*(wL - u_star);
            E_m     = EL - 1.0/a*(pres_m*w_m - pL*wL);

            alpha_p = alphaR;
            tau_p   = tauR - 1.0/a*(u_star - wR);
            w_p     = u_star;
            pres_p  = pR - a*(wR - u_star);
            E_p     = ER + 1.0/a*(pres_p*w_p - pR*wR);
          }
          else {
            if(xi > u_star && xi < wR + a*tauR)	{
              alpha_m = alphaR;
              tau_m   = tauR - 1.0/a*(u_star - wR);
              w_m     = u_star;
              pres_m  = pR - a*(wR - u_star);
              E_m     = ER + 1.0/a*(pres_m*w_m - pR*wR);

              alpha_p = alphaR;
              tau_p   = tauR - 1.0/a*(u_star - wR);
              w_p     = u_star;
              pres_p  = pR - a*(wR - u_star);
              E_p     = ER + 1.0/a*(pres_p*w_p - pR*wR);
            }
            else {
              if(xi == wR + a*tauR) {
                alpha_m = alphaR;
                tau_m   = tauR - 1.0/a*(u_star - wR);
                w_m     = u_star;
                pres_m  = pR - a*(wR - u_star);
                E_m     = ER + 1.0/a*(pres_m*w_m - pR*wR);

                alpha_p = alphaR;
                tau_p   = tauR;
                w_p     = wR;
                pres_p  = pR;
                E_p     = ER;
              }
              else {
                alpha_m = alphaR;
                tau_m   = tauR;
                w_m     = wR;
                pres_m  = pR;
                E_m     = ER;

                alpha_p = alphaR;
                tau_p   = tauR;
                w_p     = wR;
                pres_p  = pR;
                E_p     = ER;
              }
            }
          }
        }
      }
    }
  }

  // Riemann solver for the phase associated to the interfacial pressure
  //
  template<class Field>
  template<typename T>
  void RelaxationFlux<Field>::Riemann_solver_phase_pI(const T xi,
                                                      const T alphaL, const T alphaR, const T tauL, const T tauR,
                                                      const T wL, const T wR, const T pL, const T pR, const T EL, const T ER,
                                                      const T w_diesis, const T tauL_diesis, const T tauR_diesis, const T a,
                                                      T& alpha_m, T& tau_m, T& w_m, T& pres_m, T& E_m,
                                                      T& alpha_p, T& tau_p, T& w_p, T& pres_p, T& E_p,
                                                      T& w_star) {
    const T nu  = alphaL/alphaR;
    const T ML  = wL/(a*tauL);
    const T MdL = w_diesis/(a*tauL_diesis);

    T M;
    T Mzero;
    const T mu = 0.9;
    const T t  = tauR_diesis/tauL_diesis;

    if(w_diesis > 0.0) {
      if(ML <  1.0) {
         /*--- Configuration <1,2> subsonic.
               Computation of M which parametrisez the whole solution ---*/
        Mzero = 4.0/(nu + 1.0)*
                MdL/((1.0 + MdL*MdL)*
                     (1.0 + std::sqrt(std::abs(1.0 - 4.0*nu/((nu + 1.0)*(nu + 1.0))*
                                                     4.0*MdL*MdL/((1.0 + MdL*MdL)*(1.0 + MdL*MdL))))));

        if(mu*tauR_diesis <= tauR_diesis + tauL_diesis*(MdL + nu*Mzero)/(1.+nu*Mzero)) {
          M = Mzero;
        }
        else {
          /*--- Add the required amount of energy dissipation ---*/
          M = 1.0/nu*(MdL + t*(1.0 - mu))/(1.0 - t*(1.0 - mu));
        }
      }

      if(xi < wL - a*tauL) {
        alpha_m = alphaL;
        tau_m   = tauL;
        w_m     = wL;
        pres_m  = pL;
        E_m     = EL;

        alpha_p = alphaL;
        tau_p   = tauL;
        w_p     = wL;
        pres_p  = pL;
        E_p     = EL;
      }
      else {
        if(xi == wL - a*tauL) {
          alpha_m = alphaL;
          tau_m   = tauL;
          w_m     = wL;
          pres_m  = pL;
          E_m     = EL;

          alpha_p = alphaL;
          tau_p   = tauL_diesis*(1.0 - MdL)/(1.0 - M);
          w_p     = a*M*tau_p;
          pres_p  = pL + a*(wL - w_p);
          E_p     = EL - 1.0/a*(pres_p*w_p - pL*wL);
        }
        else {
          if(xi > wL - a*tauL && xi < 0.0) {
            alpha_m = alphaL;
            tau_m   = tauL_diesis*(1.0 - MdL)/(1.0 - M);
            w_m     = a*M*tau_m;
            pres_m  = pL + a*(wL - w_m);
            E_m     = EL - 1.0/a*(pres_m*w_m - pL*wL);

            alpha_p = alphaL;
            tau_p   = tauL_diesis*(1.0 - MdL)/(1.0 - M);
            w_p     = a*M*tau_p;
            pres_p  = pL + a*(wL - w_p);
            E_p     = EL - 1.0/a*(pres_p*w_p - pL*wL);
          }
          else {
            if(xi == 0.0) {
              alpha_m = alphaL;
              tau_m   = tauL_diesis*(1.0 - MdL)/(1.0 - M);
              w_m     = a*M*tau_m;
              pres_m  = pL + a*(wL - w_m);
              E_m     = EL - 1.0/a*(pres_m*w_m - pL*wL);

              alpha_p = alphaR;
              tau_p   = tauL_diesis*(1.0 + MdL)/(1.0 + nu*M);
              w_p     = nu*a*M*tau_p;
              pres_p  = pL + a*a*(tauL - tau_p);
              E_p     = E_m - (pres_p*tau_p - pres_m*tau_m);
            }
            else {
              if(xi > 0.0 && xi < nu*a*M*tauL_diesis*(1.0 + MdL)/(1.0 + nu*M)) {
                /*--- Computations of E_m and E_p ---*/
                alpha_m = alphaL;
                tau_m   = tauL_diesis*(1.0 - MdL)/(1.0 - M);
                w_m     = a*M*tau_m;
                pres_m  = pL + a*(wL - w_m);
                E_m     = EL - 1.0/a*(pres_m*w_m - pL*wL);

                alpha_p = alphaR;
                tau_p   = tauL_diesis*(1.0 + MdL)/(1.0 + nu*M);
                w_p     = nu*a*M*tau_p;
                pres_p  = pL + a*a*(tauL - tau_p);
                E_p     = E_m - (pres_p*tau_p - pres_m*tau_m);

                /*--- Compute the real states ---*/
                alpha_m = alphaR;
                tau_m   = tauL_diesis*(1.0 + MdL)/(1.0 + nu*M);
                w_m     = nu*a*M*tau_m;
                pres_m  = pL + a*a*(tauL - tau_m);
                E_m     = E_p;

                alpha_p = alphaR;
                tau_p   = tauL_diesis*(1.0 + MdL)/(1.0 + nu*M);
                w_p     = nu*a*M*tau_p;
                pres_p  = pL + a*a*(tauL - tau_p);
              }
              else {
                if(xi == nu*a*M*tauL_diesis*(1.0 + MdL)/(1.0 + nu*M)) {
                  /*--- Computations of E_m and E_p ---*/
                  alpha_m = alphaL;
                  tau_m   = tauL_diesis*(1.0 - MdL)/(1.0 - M);
                  w_m     = a*M*tau_m;
                  pres_m  = pL + a*(wL - w_m);
                  E_m     = EL - 1.0/a*(pres_m*w_m - pL*wL);

                  alpha_p = alphaR;
                  tau_p   = tauL_diesis*(1.0 + MdL)/(1.0 + nu*M);
                  w_p     = nu*a*M*tau_p;
                  pres_p  = pL + a*a*(tauL - tau_p);
                  E_p     = E_m - (pres_p*tau_p - pres_m*tau_m);

                  /*--- Compute the real states ---*/
                  alpha_m = alphaR;
                  tau_m   = tauL_diesis*(1.0 + MdL)/(1.0 + nu*M);
                  w_m     = nu*a*M*tau_m;
                  pres_m  = pL + a*a*(tauL - tau_m);
                  E_m     = E_p;

                  alpha_p = alphaR;
                  tau_p   = tauR_diesis + tauL_diesis*(MdL - nu*M)/(1.0 + nu*M);
                  w_p     = nu*a*M*tauL_diesis*(1.0 + MdL)/(1.0 + nu*M);
                  pres_p  = pR - a*(wR - w_p);
                  E_p     = ER - 1.0/a*(pR*wR - pres_p*w_p);
                }
                else {
                  if(xi > nu*a*M*tauL_diesis*(1.0 + MdL)/(1.0 + nu*M) && xi < wR + a*tauR) {
                    alpha_m = alphaR;
                    tau_m   = tauR_diesis + tauL_diesis*(MdL - nu*M)/(1.0 + nu*M);
                    w_m     = nu*a*M*tauL_diesis*(1.0 + MdL)/(1.0 + nu*M);
                    pres_m  = pR - a*(wR - w_m);
                    E_m     = ER - 1.0/a*(pR*wR - pres_m*w_m);

                    alpha_p = alphaR;
                    tau_p   = tauR_diesis + tauL_diesis*(MdL - nu*M)/(1.0 + nu*M);
                    w_p     = nu*a*M*tauL_diesis*(1.0 + MdL)/(1.0 + nu*M);
                    pres_p  = pR - a*(wR - w_p);
                    E_p     = ER + 1.0/a*(pres_p*w_p - pR*wR);
                  }
                  else {
                    if(xi == wR + a*tauR) {
                      alpha_m = alphaR;
                      tau_m   = tauR_diesis + tauL_diesis*(MdL - nu*M)/(1.0 + nu*M);
                      w_m     = nu*a*M*tauL_diesis*(1.0 + MdL)/(1.0 + nu*M);
                      pres_m  = pR - a*(wR - w_m);
                      E_m     = ER + 1.0/a*(pres_m*w_m - pR*wR);

                      alpha_p = alphaR;
                      tau_p   = tauR;
                      w_p     = wR;
                      pres_p  = pR;
                      E_p     = ER;
                    }
                    else {
                      alpha_m = alphaR;
                      tau_m   = tauR;
                      w_m     = wR;
                      pres_m  = pR;
                      E_m     = ER;

                      alpha_p = alphaR;
                      tau_p   = tauR;
                      w_p     = wR;
                      pres_p  = pR;
                      E_p     = ER;
                    }
                  }
                }
              }
            }
          }
        }
      }
      w_star = nu*a*M*tauL_diesis*(1.0 + MdL)/(1.0 + nu*M);
    }
    else {
      if(w_diesis < 0.0) {
        Riemann_solver_phase_pI(-xi,
                                alphaR, alphaL, tauR, tauL, -wR, -wL, pR, pL, ER, EL,
                                -w_diesis, tauR_diesis, tauL_diesis, a,
                                alpha_p, tau_p, w_p, pres_p, E_p,
                                alpha_m, tau_m, w_m, pres_m, E_m,
                                w_star);
        w_m = -w_m;
        w_p = -w_p;
        w_star = -w_star;
      }
      else {
        w_star = 0.0;
        if(xi < wL - a*tauL) {
          alpha_m = alphaL;
          tau_m   = tauL;
          w_m     = wL;
          pres_m  = pL;
          E_m     = EL;

          alpha_p = alphaL;
          tau_p   = tauL;
          w_p     = wL;
          pres_p  = pL;
          E_p     = EL;
        }
        else {
          if(xi == wL - a*tauL) {
            alpha_m = alphaL;
            tau_m   = tauL;
            w_m     = wL;
            pres_m  = pL;
            E_m     = EL;

            alpha_p = alphaL;
            tau_p   = tauL_diesis;
            w_p     = 0.0;
            pres_p  = pL + a*(wL - w_p);
            E_p     = EL - 1.0/a*(pres_p*w_p - pL*wL);
          }
          else {
            if(xi > wL - a*tauL && xi < 0.0) {
              alpha_m = alphaL;
              tau_m   = tauL_diesis;
              w_m     = 0.0;
              pres_m  = pL + a*(wL - w_m);
              E_m     = EL - 1.0/a*(pres_m*w_m - pL*wL);

              alpha_p = alphaL;
              tau_p   = tauL_diesis;
              w_p     = 0.0;
              pres_p  = pL + a*(wL - w_p);
              E_p     = EL - 1.0/a*(pres_p*w_p - pL*wL);
            }
            else {
              if(xi == 0.0) {
                alpha_m = alphaL;
                tau_m   = tauL_diesis;
                w_m     = 0.0;
                pres_m  = pL + a*(wL - w_m);
                E_m     = EL - 1.0/a*(pres_m*w_m - pL*wL);

                alpha_p = alphaR;
                tau_p   = tauR_diesis;
                w_p     = 0.0;
                pres_p  = pR - a*(wR - w_p);
                E_p     = ER + 1.0/a*(pres_p*w_p - pR*wR);
              }
              else {
                if(xi > 0.0 && xi < wR + a*tauR) {
                  alpha_m = alphaR;
                  tau_m   = tauR_diesis;
                  w_m     = 0.0;
                  pres_m  = pR - a*(wR - w_m);
                  E_m     = ER + 1.0/a*(pres_m*w_m - pR*wR);

                  alpha_p = alphaR;
                  tau_p   = tauR_diesis;
                  w_p     = 0.0;
                  pres_p  = pR - a*(wR-w_p);
                  E_p     = ER + 1.0/a*(pres_p*w_p - pR*wR);
                }
                else {
                  if(xi == wR + a*tauR) {
                    alpha_m = alphaR;
                    tau_m   = tauR_diesis;
                    w_m     = 0.0;
                    pres_m  = pR - a*(wR - w_m);
                    E_m     = ER + 1.0/a*(pres_m*w_m - pR*wR);

                    alpha_p = alphaR;
                    tau_p   = tauR;
                    w_p     = wR;
                    pres_p  = pR;
                    E_p     = ER;
                  }
                  else {
                    alpha_m = alphaR;
                    tau_m   = tauR;
                    w_m     = wR;
                    pres_m  = pR;
                    E_m     = ER;

                    alpha_p = alphaR;
                    tau_p   = tauR;
                    w_p     = wR;
                    pres_p  = pR;
                    E_p     = ER;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

} // end of namespace

#endif
