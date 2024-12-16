// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#include <samurai/algorithm/update.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

#include <filesystem>
namespace fs = std::filesystem;

#include <samurai/mr/adapt.hpp>

#include "Rusanov_flux.hpp"
#include "non_conservative_flux.hpp"
#include "Suliciu_scheme.hpp"

#include "containers.hpp"

#define SULICIU_RELAXATION
//#define RUSANOV_FLUX

// Specify the use of this namespace where we just store the indices
// and some parameters related to the equations of state
using namespace EquationData;

// This is the class for the simulation of a BN model
//
template<std::size_t dim>
class BN_Solver {
public:
  using Config = samurai::MRConfig<dim, 2>;

  BN_Solver() = default; // Default constructor. This will do nothing
                         // and basically will never be used

  BN_Solver(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
            const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
            const Simulation_Parameters& sim_param,
            const EOS_Parameters& eos_param,
            const Riemann_Parameters& Riemann_param); // Class constrcutor with the arguments related
                                                      // to the grid and to the physics.

  void run(); // Function which actually executes the temporal loop

  template<class... Variables>
  void save(const fs::path& path,
            const std::string& filename,
            const std::string& suffix,
            const Variables&... fields); // Routine to save the results

private:
  /*--- Now we declare some relevant variables ---*/
  const samurai::Box<double, dim> box;

  samurai::MRMesh<Config> mesh; // Variable to store the mesh

  using Field        = samurai::Field<decltype(mesh), double, EquationData::NVARS, false>;
  using Field_Scalar = samurai::Field<decltype(mesh), double, 1, false>;
  using Field_Vect   = samurai::Field<decltype(mesh), double, dim, false>;

  double Tf;  // Final time of the simulation
  double cfl; // Courant number of the simulation so as to compute the time step
  double dt;  // Time-step (modified according to CFL)

  std::size_t nfiles; // Number of files desired for output

  bool apply_relaxation; // Set whether to apply or not the pressure relaxation

  bool apply_finite_rate_relaxation; // Set if finite rate relaxation is desired
  double eps_u; // Relaxation parameter for the velocity
  double eps_p; // Relaxation parameter for the pressure
  double eps_T; // Relaxation parameter for the temperature

  Field conserved_variables; // The variable which stores the conserved variables,
                             // namely the varialbes for which we solve a PDE system

  const SG_EOS<typename Field::value_type> EOS_phase1; // Equation of state of phase 1
  const SG_EOS<typename Field::value_type> EOS_phase2; // Equation of state of phase 2

  #ifdef SULICIU_RELAXATION
    samurai::RelaxationFlux<Field> numerical_flux; // function to compute the numerical flux
                                                   // (this is necessary to call 'make_flux')
  #elifdef RUSANOV_FLUX
    samurai::RusanovFlux<Field> numerical_flux_cons; // function to compute the numerical flux for the conservative part
                                                     // (this is necessary to call 'make_flux')

    samurai::NonConservativeFlux<Field> numerical_flux_non_cons; // function to compute the numerical flux for the non-conservative part
                                                                 // (this is necessary to call 'make_flux')
  #endif

  /*-- Now we declare a bunch of fields which depend from the state, but it is useful
       to have it for the output ---*/
  Field_Scalar rho,
               p,
               rho1,
               p1,
               c1,
               rho2,
               p2,
               c2,
               alpha2,
               T1,
               T2;

  Field_Vect vel1,
             vel2;

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  void init_variables(const Riemann_Parameters& Riemann_param); // Routine to initialize the variables (both conserved and auxiliary, this is problem dependent)

  void update_auxiliary_fields(); // Routine to update auxiliary fields for output and time step update

  #ifdef RUSANOV_FLUX
    double get_max_lambda() const; // Compute the estimate of the maximum eigenvalue
  #endif

  using Matrix_Relaxation = std::array<std::array<typename Field::value_type, 2>, 2>;
  using Vector_Relaxation = std::array<typename Field::value_type, 2>;
  Matrix_Relaxation A_relax; // Matrix associated to the relaxation
  Vector_Relaxation S_relax; // Vector associated to the source term

  Matrix_Relaxation Jac_update;     // Matrix associated to the Jacobian of the Newton method to update consered variables
  Matrix_Relaxation inv_Jac_update; // Inverse of the Jacobian of the Newton method to update consered variables

  void perform_relaxation_finite_rate(); // Routine to perform the finite-rate relaxation (following Jomée 2023)

  void perform_instantaneous_relaxation(); // Routine to perform instantaneous relaxation (velocity and pressure-temperature)

  template<typename State>
  void compute_coefficients_source_relaxation(const State& q,
                                              Matrix_Relaxation& A, Vector_Relaxation& S); // Compute the coefficients
                                                                                             // and the source term for the relaxation
};

/*--- START WITH CLASS CONSTRUCTOR ---*/

// Implement class constructor
//
template<std::size_t dim>
BN_Solver<dim>::BN_Solver(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                          const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                          const Simulation_Parameters& sim_param,
                          const EOS_Parameters& eos_param,
                          const Riemann_Parameters& Riemann_param):
  box(min_corner, max_corner), mesh(box, sim_param.min_level, sim_param.max_level, {false}),
  Tf(sim_param.Tf), cfl(sim_param.Courant), dt(), nfiles(sim_param.nfiles),
  apply_relaxation(sim_param.apply_relaxation),
  apply_finite_rate_relaxation(sim_param.apply_finite_rate_relaxation),
  eps_u(sim_param.eps_u), eps_p(sim_param.eps_p), eps_T(sim_param.eps_T),
  EOS_phase1(eos_param.gamma_1, eos_param.pi_infty_1, eos_param.q_infty_1, eos_param.cv_1),
  EOS_phase2(eos_param.gamma_2, eos_param.pi_infty_2, eos_param.q_infty_2, eos_param.cv_2),
  #ifdef SULICIU_RELAXATION
    numerical_flux(EOS_phase1, EOS_phase2)
  #elifdef RUSANOV_FLUX
    numerical_flux_cons(EOS_phase1, EOS_phase2),
    numerical_flux_non_cons(EOS_phase1, EOS_phase2)
  #endif
  {
    init_variables(Riemann_param);
  }

// Initialization of conserved and auxiliary variables
//
template<std::size_t dim>
void BN_Solver<dim>::init_variables(const Riemann_Parameters& Riemann_param) {
  // Create conserved and auxiliary fields
  conserved_variables = samurai::make_field<typename Field::value_type, EquationData::NVARS>("conserved", mesh);

  rho  = samurai::make_field<typename Field::value_type, 1>("rho", mesh);
  p    = samurai::make_field<typename Field::value_type, 1>("p", mesh);

  rho1 = samurai::make_field<typename Field::value_type, 1>("rho1", mesh);
  p1   = samurai::make_field<typename Field::value_type, 1>("p1", mesh);
  c1   = samurai::make_field<typename Field::value_type, 1>("c1", mesh);

  rho2 = samurai::make_field<typename Field::value_type, 1>("rho2", mesh);
  p2   = samurai::make_field<typename Field::value_type, 1>("p2", mesh);
  c2   = samurai::make_field<typename Field::value_type, 1>("c2", mesh);

  vel1 = samurai::make_field<typename Field::value_type, dim>("vel1", mesh);
  vel2 = samurai::make_field<typename Field::value_type, dim>("vel2", mesh);

  alpha2 = samurai::make_field<typename Field::value_type, 1>("alpha2", mesh);
  T1     = samurai::make_field<typename Field::value_type, 1>("T1", mesh);
  T2     = samurai::make_field<typename Field::value_type, 1>("T2", mesh);

  // Initialize the fields with a loop over all cells
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           const auto center = cell.center();
                           const double x    = center[0];

                           if(x <= Riemann_param.xd) {
                             conserved_variables[cell][ALPHA1_INDEX] = Riemann_param.alpha1L;

                             rho1[cell] = Riemann_param.rho1L;
                             vel1[cell] = Riemann_param.u1L;
                             p1[cell]   = Riemann_param.p1L;

                             rho2[cell] = Riemann_param.rho2L;
                             vel2[cell] = Riemann_param.u2L;
                             p2[cell]   = Riemann_param.p2L;
                           }
                           else {
                             conserved_variables[cell][ALPHA1_INDEX] = Riemann_param.alpha1R;

                             rho1[cell] = Riemann_param.rho1R;
                             vel1[cell] = Riemann_param.u1R;
                             p1[cell]   = Riemann_param.p1R;

                             rho2[cell] = Riemann_param.rho2R;
                             vel2[cell] = Riemann_param.u2R;
                             p2[cell]   = Riemann_param.p2R;
                           }

                           conserved_variables[cell][ALPHA1_RHO1_INDEX]    = conserved_variables[cell][ALPHA1_INDEX]*rho1[cell];
                           conserved_variables[cell][ALPHA1_RHO1_U1_INDEX] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*vel1[cell];
                           const auto e1 = EOS_phase1.e_value_RhoP(rho1[cell], p1[cell]);
                           conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*
                                                                             (e1 + 0.5*vel1[cell]*vel1[cell]);

                           conserved_variables[cell][ALPHA2_RHO2_INDEX]    = (1.0 - conserved_variables[cell][ALPHA1_INDEX])*rho2[cell];
                           conserved_variables[cell][ALPHA2_RHO2_U2_INDEX] = conserved_variables[cell][ALPHA2_RHO2_INDEX]*vel2[cell];
                           const auto e2 = EOS_phase2.e_value_RhoP(rho2[cell], p2[cell]);
                           conserved_variables[cell][ALPHA2_RHO2_E2_INDEX] = conserved_variables[cell][ALPHA2_RHO2_INDEX]*
                                                                             (e2 + 0.5*vel2[cell]*vel2[cell]);

                           c1[cell] = EOS_phase1.c_value_RhoP(rho1[cell], p1[cell]);
                           T1[cell] = EOS_phase1.T_value_RhoP(rho1[cell], p1[cell]);

                           c2[cell] = EOS_phase2.c_value_RhoP(rho2[cell], p2[cell]);
                           T2[cell] = EOS_phase2.T_value_RhoP(rho2[cell], p2[cell]);

                           alpha2[cell] = 1.0 - conserved_variables[cell][ALPHA1_INDEX];

                           rho[cell] = conserved_variables[cell][ALPHA1_RHO1_INDEX]
                                     + conserved_variables[cell][ALPHA2_RHO2_INDEX];

                           p[cell] = conserved_variables[cell][ALPHA1_INDEX]*p1[cell]
                                   + (1.0 - conserved_variables[cell][ALPHA1_INDEX])*p2[cell];
                         });

  samurai::make_bc<samurai::Neumann<1>>(conserved_variables, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
}

/*--- AUXILIARY ROUTINES ---*/

// Compute the estimate of the maximum eigenvalue for CFL condition
//
#ifdef RUSANOV_FLUX
  template<std::size_t dim>
  double BN_Solver<dim>::get_max_lambda() const {
    double res = 0.0;

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                             res = std::max(std::max(std::abs(vel1[cell]) + c1[cell],
                                                     std::abs(vel2[cell]) + c2[cell]),
                                            res);
                           });

    return res;
  }
#endif

// Update auxiliary fields after solution of the system
//
template<std::size_t dim>
void BN_Solver<dim>::update_auxiliary_fields() {
  // Resize fields because of multiresolution
  rho.resize();
  p.resize();

  rho1.resize();
  p1.resize();
  c1.resize();
  vel1.resize();
  T1.resize();

  rho2.resize();
  p2.resize();
  c2.resize();
  vel2.resize();
  T2.resize();

  alpha2.resize();

  // Loop over all cells
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           // Compute the fields
                           alpha2[cell] = 1.0 - conserved_variables[cell][ALPHA1_INDEX];

                           rho[cell] = conserved_variables[cell][ALPHA1_RHO1_INDEX]
                                     + conserved_variables[cell][ALPHA2_RHO2_INDEX];

                           rho1[cell]    = conserved_variables[cell][ALPHA1_RHO1_INDEX]/
                                           conserved_variables[cell][ALPHA1_INDEX]; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           vel1[cell]    = conserved_variables[cell][ALPHA1_RHO1_U1_INDEX]/
                                           conserved_variables[cell][ALPHA1_RHO1_INDEX]; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           const auto e1 = conserved_variables[cell][ALPHA1_RHO1_E1_INDEX]/
                                           conserved_variables[cell][ALPHA1_RHO1_INDEX]
                                         - 0.5*vel1[cell]*vel1[cell]; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           p1[cell]      = EOS_phase1.pres_value_Rhoe(rho1[cell], e1);
                           c1[cell]      = EOS_phase1.c_value_RhoP(rho1[cell], p1[cell]);
                           T1[cell]      = EOS_phase1.T_value_RhoP(rho1[cell], p1[cell]);

                           rho2[cell]    = conserved_variables[cell][ALPHA2_RHO2_INDEX]/
                                           (1.0 - conserved_variables[cell][ALPHA1_INDEX]); /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           vel2[cell]    = conserved_variables[cell][ALPHA2_RHO2_U2_INDEX]/
                                           conserved_variables[cell][ALPHA2_RHO2_INDEX]; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           const auto e2 = conserved_variables[cell][ALPHA2_RHO2_E2_INDEX]/
                                           conserved_variables[cell][ALPHA2_RHO2_INDEX]
                                         - 0.5*vel2[cell]*vel2[cell]; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           p2[cell]      = EOS_phase2.pres_value_Rhoe(rho2[cell], e2);
                           c2[cell]      = EOS_phase2.c_value_RhoP(rho2[cell], p2[cell]);
                           T2[cell]      = EOS_phase2.T_value_RhoP(rho2[cell], p2[cell]);

                           p[cell]       = conserved_variables[cell][ALPHA1_INDEX]*p1[cell]
                                         + (1.0 - conserved_variables[cell][ALPHA1_INDEX])*p2[cell];
                         });
}

// Save desired fields and info
//
template<std::size_t dim>
template<class... Variables>
void BN_Solver<dim>::save(const fs::path& path,
                          const std::string& filename,
                          const std::string& suffix,
                          const Variables&... fields) {
  auto level_ = samurai::make_field<std::size_t, 1>("level", mesh);

  if(!fs::exists(path)) {
    fs::create_directory(path);
  }

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           level_[cell] = cell.level;
                         });

  samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, fields..., level_);
}

/*--- AUXILIARY ROUTINE FOR THE RELAXATION ---*/
template<std::size_t dim>
template<typename State>
void BN_Solver<dim>::compute_coefficients_source_relaxation(const State& q,
                                                            Matrix_Relaxation& A, Vector_Relaxation& S) {
  // Compute auxiliary variables for phase 1
  const auto rho1_loc = q[ALPHA1_RHO1_INDEX]/q[ALPHA1_INDEX]; /*--- TODO: Add treatment for vanishing volume fraction ---*/
  const auto vel1_loc = q[ALPHA1_RHO1_U1_INDEX]/q[ALPHA1_RHO1_INDEX]; /*--- TODO: Add treatment for vanishing volume fraction ---*/
  const auto e1_loc   = q[ALPHA1_RHO1_E1_INDEX]/q[ALPHA1_RHO1_INDEX]
                      - 0.5*vel1_loc*vel1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
  const auto p1_loc   = EOS_phase1.pres_value_Rhoe(rho1_loc, e1_loc);
  const auto c1_loc   = EOS_phase1.c_value_RhoP(rho1_loc, p1_loc);
  const auto kappa1   = EOS_phase1.de_dP_rho(p1_loc, rho1_loc);
  const auto T1_loc   = EOS_phase1.T_value_RhoP(rho1_loc, p1_loc);
  const auto cv1      = EOS_phase1.de_dT_rho(T1_loc, rho1_loc);
  const auto Gamma1   = EOS_phase1.de_drho_T(rho1_loc, T1_loc);

  // Compute auxiliary variables for phase 2
  const auto rho2_loc = q[ALPHA2_RHO2_INDEX]/(1.0 - q[ALPHA1_INDEX]); /*--- TODO: Add treatment for vanishing volume fraction ---*/
  const auto vel2_loc = q[ALPHA2_RHO2_U2_INDEX]/q[ALPHA2_RHO2_INDEX]; /*--- TODO: Add treatment for vanishing volume fraction ---*/
  const auto e2_loc   = q[ALPHA2_RHO2_E2_INDEX]/q[ALPHA2_RHO2_INDEX]
                      - 0.5*vel2_loc*vel2_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
  const auto p2_loc   = EOS_phase2.pres_value_Rhoe(rho2_loc, e2_loc);
  const auto c2_loc   = EOS_phase2.c_value_RhoP(rho2_loc, p2_loc);
  const auto kappa2   = EOS_phase2.de_dP_rho(p2_loc, rho2_loc);
  const auto T2_loc   = EOS_phase2.T_value_RhoP(rho2_loc, p2_loc);
  const auto cv2      = EOS_phase2.de_dT_rho(T2_loc, rho2_loc);
  const auto Gamma2   = EOS_phase2.de_drho_T(rho2_loc, T2_loc);

  /*--- uI = beta*u1 + (1 - beta)*u2
        pI = chi*p1 + (1 - chi)*p2 ---*/
  typename Field::value_type beta = 1.0;
  typename Field::value_type chi  = (1.0 - beta)/((1.0 - beta)*T2_loc + beta*T1_loc);
  auto pI_relax = chi*p1_loc + (1.0 - chi)*p2_loc;
  /*--- TODO: Possibly change, a priori this is not necessarily the same of the convective operator, even though
              substituting uI \cdot grad\alpha iwe det d\alpha/dt... ---*/

  // Compute the coefficients
  const auto a_pp = -(1.0/eps_p)*(rho1_loc*c1_loc*c1_loc/q[ALPHA1_INDEX] +
                                  rho2_loc*c2_loc*c2_loc/(1.0 - q[ALPHA1_INDEX]) +
                                  ((chi - 1.0)/(q[ALPHA1_RHO1_INDEX]*kappa1) +
                                   chi/(q[ALPHA2_RHO2_INDEX]*kappa2))*(p1_loc - p2_loc)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
  const auto a_pT = -(1.0/eps_T)*(1.0/(q[ALPHA1_RHO1_INDEX]*kappa1) +
                                  1.0/(q[ALPHA2_RHO2_INDEX]*kappa2)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
  const auto a_Tp = -(1.0/eps_p)*((pI_relax - rho1_loc*rho1_loc*Gamma1)/(q[ALPHA1_RHO1_INDEX]*cv1) +
                                  (pI_relax - rho2_loc*rho2_loc*Gamma2)/(q[ALPHA2_RHO2_INDEX]*cv2));
  const auto a_TT = -(1.0/eps_T)*(1.0/(q[ALPHA1_RHO1_INDEX]*cv1) +
                                  1.0/(q[ALPHA2_RHO2_INDEX]*cv2)); /*--- TODO: Add treatment for vanishing volume fraction ---*/

  A[0][0] = 1.0 - dt*a_pp;
  A[0][1] = -dt*a_pT;
  A[1][0] = -dt*a_Tp;
  A[1][1] = 1.0 - dt*a_TT;

  // Set source term
  const auto rho_0      = q[ALPHA1_RHO1_INDEX] + q[ALPHA2_RHO2_INDEX];
  const auto Y1_0       = q[ALPHA1_RHO1_INDEX]/rho_0;
  const auto a_tilde_pu = -1.0/eps_u*((beta - 1.0)/(kappa1*rho_0*Y1_0) + beta/(kappa2*rho_0*(1.0 - Y1_0)));
  const auto a_tilde_Tu = -1.0/eps_u*((beta - 1.0)/(cv1*rho_0*Y1_0) + beta/(cv2*rho_0*(1.0 - Y1_0)));
  S[0] = a_tilde_pu*((vel1_loc - vel2_loc)*(vel1_loc - vel2_loc));
  S[1] = a_tilde_Tu*((vel1_loc - vel2_loc)*(vel1_loc - vel2_loc));
}

// Finite rate relaxation (following Jomée 2023)
//
template<std::size_t dim>
void BN_Solver<dim>::perform_relaxation_finite_rate() {
  // Resize fields because of multiresolution
  rho1.resize();
  p1.resize();
  vel1.resize();
  T1.resize();

  rho2.resize();
  p2.resize();
  vel2.resize();
  T2.resize();

  // Loop over all cells
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           std::cout << "Starting relaxation cell " << cell << std::endl;
                           // Compute the fields
                           rho1[cell] = conserved_variables[cell][ALPHA1_RHO1_INDEX]/
                                        conserved_variables[cell][ALPHA1_INDEX]; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           vel1[cell] = conserved_variables[cell][ALPHA1_RHO1_U1_INDEX]/
                                        conserved_variables[cell][ALPHA1_RHO1_INDEX]; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           auto e1    = conserved_variables[cell][ALPHA1_RHO1_E1_INDEX]/
                                        conserved_variables[cell][ALPHA1_RHO1_INDEX]
                                      - 0.5*vel1[cell]*vel1[cell]; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           p1[cell]   = EOS_phase1.pres_value_Rhoe(rho1[cell], e1);

                           rho2[cell] = conserved_variables[cell][ALPHA2_RHO2_INDEX]/
                                        (1.0 - conserved_variables[cell][ALPHA1_INDEX]); /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           vel2[cell] = conserved_variables[cell][ALPHA2_RHO2_U2_INDEX]/
                                        conserved_variables[cell][ALPHA2_RHO2_INDEX]; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           auto e2    = conserved_variables[cell][ALPHA2_RHO2_E2_INDEX]/
                                        conserved_variables[cell][ALPHA2_RHO2_INDEX]
                                      - 0.5*vel2[cell]*vel2[cell]; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           p2[cell]   = EOS_phase2.pres_value_Rhoe(rho2[cell], e2);

                           // Compute updated delta_u (we have analytical formula)
                           typename Field::value_type delta_u;
                           const auto rho_0 = conserved_variables[cell][ALPHA1_RHO1_INDEX]
                                            + conserved_variables[cell][ALPHA2_RHO2_INDEX];
                           const auto Y1_0  = conserved_variables[cell][ALPHA1_RHO1_INDEX]/rho_0;
                           delta_u = (vel1[cell] - vel2[cell])*std::exp(-dt/eps_u*
                                                                         (1.0/conserved_variables[cell][ALPHA1_RHO1_INDEX] +
                                                                          1.0/conserved_variables[cell][ALPHA2_RHO2_INDEX]));

                           // Compute matrix relaxation coefficients
                           compute_coefficients_source_relaxation(conserved_variables[cell],
                                                                  A_relax, S_relax);

                           // Solve the system for the remaining variables
                           typename Field::value_type delta_p,
                                                      delta_T;
                           const auto delta_p0 = p1[cell] - p2[cell];
                           T1[cell] = EOS_phase1.T_value_RhoP(rho1[cell], p1[cell]);
                           T2[cell] = EOS_phase2.T_value_RhoP(rho2[cell], p2[cell]);
                           std::cout << "vel1 before relaxation = " << vel1[cell] << std::endl;
                           std::cout << "vel2 before relaxation = " << vel2[cell] << std::endl;
                           std::cout << "p1 before relaxation = "   << p1[cell] << std::endl;
                           std::cout << "p2 before relaxation = "   << p2[cell] << std::endl;
                           std::cout << "T1 before relaxation = "   << T1[cell] << std::endl;
                           std::cout << "T2 before relaxation = "   << T2[cell] << std::endl;
                           const auto delta_T0 = T1[cell] - T2[cell];
                           const auto det_A_pT = A_relax[0][0]*A_relax[1][1]
                                               - A_relax[0][1]*A_relax[1][0];
                           if(std::abs(det_A_pT) > 1e-10) {
                             delta_p = (1.0/det_A_pT)*(A_relax[1][1]*(delta_p0 + dt*S_relax[0]) -
                                                       A_relax[0][1]*(delta_T0 + dt*S_relax[1]));
                             delta_T = (1.0/det_A_pT)*(-A_relax[1][0]*(delta_p0 + dt*S_relax[0]) +
                                                        A_relax[0][0]*(delta_T0 + dt*S_relax[1]));
                           }
                           else {
                             std::cerr << "Singular matrix in the relaxation" << std::endl;
                             exit(1);
                           }
                           std::cout << "computed delta_p = " << delta_p << std::endl;
                           std::cout << "computed delta_T = " << delta_T << std::endl;

                           // Re-update conserved variables
                           /*--- Compute useful velocity norms ---*/
                           const auto um_d = Y1_0*vel1[cell]
                                           + (1.0 - Y1_0)*vel2[cell];
                           typename Field::value_type norm2_um = um_d*um_d;

                           vel1[cell] = um_d + (1.0 - Y1_0)*delta_u;
                           conserved_variables[cell][ALPHA1_RHO1_U1_INDEX] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*
                                                                             vel1[cell];
                           typename Field::value_type norm2_vel1 = vel1[cell]*vel1[cell];

                           vel2[cell] = um_d - Y1_0*delta_u;
                           conserved_variables[cell][ALPHA2_RHO2_U2_INDEX] = conserved_variables[cell][ALPHA2_RHO2_INDEX]*
                                                                             vel2[cell];
                           typename Field::value_type norm2_vel2 = vel2[cell]*vel2[cell];

                           typename Field::value_type norm2_deltau = delta_u*delta_u;

                           /*--- Newton method loop ---*/
                           const auto rhoe_0 = (conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] +
                                                conserved_variables[cell][ALPHA2_RHO2_E2_INDEX])
                                             - 0.5*rho_0*(norm2_um + Y1_0*(1.0 - Y1_0)*norm2_deltau);
                           std::cout << "p1 at the beginning of Newton (p2 + delta_p) = " << p2[cell] + delta_p << std::endl;
                           std::cout << "p2 at the beginning of Newton (p2) = " << p2[cell] << std::endl;
                           std::cout << "p1 at the beginning of Newton (p1) = " << p1[cell] << std::endl;
                           std::cout << "p2 at the beginning of Newton (p1 - delta_p) = " << p1[cell] - delta_p << std::endl;
                           for(unsigned int iter = 0; iter < 50; ++iter) {
                             // Compute Jacobian matrix
                             Jac_update[0][0] = -conserved_variables[cell][ALPHA1_RHO1_INDEX]/
                                                 (EOS_phase1.rho_value_PT(p2[cell] + delta_p, T2[cell] + delta_T)*
                                                  EOS_phase1.rho_value_PT(p2[cell] + delta_p, T2[cell] + delta_T))*
                                                 EOS_phase1.drho_dP_T(p2[cell] + delta_p, T1[cell])
                                                -conserved_variables[cell][ALPHA2_RHO2_INDEX]/
                                                 (EOS_phase2.rho_value_PT(p2[cell], T2[cell])*
                                                  EOS_phase2.rho_value_PT(p2[cell], T2[cell]))*
                                                 EOS_phase2.drho_dP_T(p2[cell], T2[cell]);
                             Jac_update[0][1] = -conserved_variables[cell][ALPHA1_RHO1_INDEX]/
                                                 (EOS_phase1.rho_value_PT(p2[cell] + delta_p, T2[cell] + delta_T)*
                                                  EOS_phase1.rho_value_PT(p2[cell] + delta_p, T2[cell] + delta_T))*
                                                 EOS_phase1.drho_dT_P(T2[cell] + delta_T, p2[cell] + delta_T)
                                                -conserved_variables[cell][ALPHA2_RHO2_INDEX]/
                                                 (EOS_phase2.rho_value_PT(p2[cell], T2[cell])*
                                                  EOS_phase2.rho_value_PT(p2[cell], T2[cell]))*
                                                 EOS_phase2.drho_dT_P(T2[cell], p2[cell]);
                             Jac_update[1][0] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*
                                                EOS_phase1.de_dP_T(p2[cell] + delta_p, T2[cell] + delta_T) +
                                                conserved_variables[cell][ALPHA2_RHO2_INDEX]*
                                                EOS_phase2.de_dP_T(p2[cell], T2[cell]);
                             Jac_update[1][1] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*
                                                EOS_phase1.de_dT_P(T2[cell] + delta_T, p2[cell] + delta_p) +
                                                conserved_variables[cell][ALPHA2_RHO2_INDEX]*
                                                EOS_phase2.de_dT_P(T2[cell], p2[cell]);

                             // Evaluate the functions for which we are looking for the zeros
                             const auto f1 = conserved_variables[cell][ALPHA1_RHO1_INDEX]/
                                             EOS_phase1.rho_value_PT(p2[cell] + delta_p, T2[cell] + delta_T)
                                           + conserved_variables[cell][ALPHA2_RHO2_INDEX]/
                                             EOS_phase2.rho_value_PT(p2[cell], T2[cell])
                                           - 1.0;
                             const auto f2 = conserved_variables[cell][ALPHA1_RHO1_INDEX]*
                                             EOS_phase1.e_value_PT(p2[cell] + delta_p, T2[cell] + delta_T)
                                           + conserved_variables[cell][ALPHA2_RHO2_INDEX]*
                                             EOS_phase2.e_value_PT(p2[cell], T2[cell])
                                           - rhoe_0;
                             if(std::abs(f1) > 1e-8 && std::abs(f2) > 1e-8) {
                               // Apply the Newton method
                               const auto det_Jac = Jac_update[0][0]*Jac_update[1][1]
                                                  - Jac_update[0][1]*Jac_update[1][0];
                               if(std::abs(det_Jac) > 1e-10) {
                                 const auto dp2 = (1.0/det_Jac)*(Jac_update[1][1]*f1 - Jac_update[0][1]*f2);
                                 const auto dT2 = (1.0/det_Jac)*(-Jac_update[1][0]*f1 + Jac_update[0][0]*f2);
                                 p2[cell] -= dp2;
                                 T2[cell] -= dT2;
                                 std::cout << "p1 after iteration " << iter << " of Newton = " << p2[cell] + delta_p << std::endl;
                                 std::cout << "p2 after iteration " << iter << " of Newton = " << p2[cell] << std::endl;

                                 if(std::abs(dp2) < 1e-8 && std::abs(dT2) < 1e-8) {
                                   break;
                                 }
                               }
                             }
                             else {
                               break;
                             }
                           }
                           /*--- Once pressure and temperature have been update, finalize the update ---*/
                           e2 = EOS_phase2.e_value_PT(p2[cell], T2[cell]);
                           conserved_variables[cell][ALPHA2_RHO2_E2_INDEX] = conserved_variables[cell][ALPHA2_RHO2_INDEX]*
                                                                             (e2 + 0.5*norm2_vel2);
                           rho2[cell] = EOS_phase2.rho_value_PT(p2[cell], T2[cell]);
                           conserved_variables[cell][ALPHA1_INDEX] = 1.0 - conserved_variables[cell][ALPHA2_RHO2_INDEX]/rho2[cell];

                           p1[cell] = p2[cell] + delta_p;
                           T1[cell] = T2[cell] + delta_T;
                           std::cout << "vel1 after relaxation = " << vel1[cell] << std::endl;
                           std::cout << "vel2 after relaxation = " << vel2[cell] << std::endl;
                           std::cout << "p1 after relaxation = "   << p1[cell] << std::endl;
                           std::cout << "p2 after relaxation = "   << p2[cell] << std::endl;
                           std::cout << "T1 after relaxation = "   << T1[cell] << std::endl;
                           std::cout << "T2 after relaxation = "   << T2[cell] << std::endl;
                           e1 = EOS_phase1.e_value_PT(p1[cell], T1[cell]);
                           conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*
                                                                             (e1 + 0.5*norm2_vel1);
                           rho1[cell] = EOS_phase1.rho_value_PT(p1[cell], T1[cell]);
                           std::cout << "Relaxation terminated cell " << cell << std::endl;
                         });
}

// Apply the instantaneous relaxation
//
template<std::size_t dim>
void BN_Solver<dim>::perform_instantaneous_relaxation() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           std::cout << "Starting relaxation cell " << cell << std::endl;

                           /*--- First focus on the velocity relaxation ---*/
                           // Compute mixture density and (specific) total energy for the updates
                           const auto rho_0 = conserved_variables[cell][ALPHA1_RHO1_INDEX]
                                            + conserved_variables[cell][ALPHA2_RHO2_INDEX];

                           const auto rhoE_0 = conserved_variables[cell][ALPHA1_RHO1_E1_INDEX]
                                             + conserved_variables[cell][ALPHA2_RHO2_E2_INDEX];

                           // Save initial specific internal energy of phase 1 for the total energy update
                           vel1[cell] = conserved_variables[cell][ALPHA1_RHO1_U1_INDEX]/
                                        conserved_variables[cell][ALPHA1_RHO1_INDEX]; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           auto e1_0  = conserved_variables[cell][ALPHA1_RHO1_E1_INDEX]/
                                        conserved_variables[cell][ALPHA1_RHO1_INDEX]
                                      - 0.5*vel1[cell]*vel1[cell];

                           // Save initial velocity of phase 2
                           vel2[cell] = conserved_variables[cell][ALPHA2_RHO2_U2_INDEX]/
                                        conserved_variables[cell][ALPHA2_RHO2_INDEX]; /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           std::cout << "vel1 before relaxation = " << vel1[cell] << std::endl;
                           std::cout << "vel2 before relaxation = " << vel2[cell] << std::endl;

                           // Update the momentum (and the kinetic energy of phase 1)
                           const auto vel_star = (conserved_variables[cell][ALPHA1_RHO1_U1_INDEX] +
                                                  conserved_variables[cell][ALPHA2_RHO2_U2_INDEX])/rho_0;

                           conserved_variables[cell][ALPHA1_RHO1_U1_INDEX] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*
                                                                             vel_star;

                           conserved_variables[cell][ALPHA2_RHO2_U2_INDEX] = conserved_variables[cell][ALPHA2_RHO2_INDEX]*
                                                                             vel_star;

                           std::cout << "vel1 after relaxation = " << vel_star << std::endl;
                           std::cout << "vel2 after relaxation = " << vel_star << std::endl;

                           // Update total energy of the two phases
                           conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] = 0.5*conserved_variables[cell][ALPHA1_RHO1_INDEX]*
                                                                             vel_star*vel_star;

                           const auto Y2_0    = conserved_variables[cell][ALPHA2_RHO2_INDEX]/rho_0;
                           const auto chi1    = 0.0; // uI = (1 - chi1)*u1 + chi1*u2;
                           const auto e1_star = e1_0 + 0.5*chi1*(vel1[cell] - vel2[cell])*(vel1[cell] - vel2[cell])*Y2_0;
                           conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] += conserved_variables[cell][ALPHA1_RHO1_INDEX]*e1_star;

                           conserved_variables[cell][ALPHA2_RHO2_E2_INDEX] = rhoE_0 - conserved_variables[cell][ALPHA1_RHO1_E1_INDEX];

                           /*--- Focus now on the pressure/temperature relaxation ---*/
                           // Compute the initial fileds for the pressure/temperature relaxation
                           e1_0            = conserved_variables[cell][ALPHA1_RHO1_E1_INDEX]/
                                             conserved_variables[cell][ALPHA1_RHO1_INDEX]
                                           - 0.5*vel1[cell]*vel1[cell]; /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           const auto e2_0 = conserved_variables[cell][ALPHA2_RHO2_E2_INDEX]/
                                             conserved_variables[cell][ALPHA2_RHO2_INDEX]
                                           - 0.5*vel2[cell]*vel2[cell]; /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           p1[cell] = EOS_phase1.pres_value_Rhoe(conserved_variables[cell][ALPHA1_RHO1_INDEX]/
                                                                 conserved_variables[cell][ALPHA1_INDEX],
                                                                 e1_0); /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           p2[cell] = EOS_phase2.pres_value_Rhoe(conserved_variables[cell][ALPHA2_RHO2_INDEX]/
                                                                 (1.0 - conserved_variables[cell][ALPHA1_INDEX]),
                                                                 e2_0); /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           rho1[cell] = conserved_variables[cell][ALPHA1_RHO1_INDEX]/
                                        conserved_variables[cell][ALPHA1_INDEX]; /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           rho2[cell] = conserved_variables[cell][ALPHA2_RHO2_INDEX]/
                                        (1.0 - conserved_variables[cell][ALPHA1_INDEX]); /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           c1[cell] = EOS_phase1.c_value_RhoP(rho1[cell], p1[cell]);

                           c2[cell] = EOS_phase2.c_value_RhoP(rho2[cell], p2[cell]);

                           std::cout << "p1 before relaxation = " << p1[cell] << std::endl;
                           std::cout << "p2 before relaxation = " << p2[cell] << std::endl;
                           std::cout << "T1 before relaxation = " << EOS_phase1.T_value_RhoP(rho1[cell], p1[cell]) << std::endl;
                           std::cout << "T2 before relaxation = " << EOS_phase2.T_value_RhoP(rho2[cell], p2[cell]) << std::endl;

                           // Compute the pressure equilibrium with the linearization method (Pelanti)
                           const auto a    = 1.0 + EOS_phase2.get_gamma()*conserved_variables[cell][ALPHA1_INDEX]
                                           + EOS_phase1.get_gamma()*(1.0 - conserved_variables[cell][ALPHA1_INDEX]);
                           const auto Z1   = rho1[cell]*c1[cell]*c1[cell];
                           const auto Z2   = rho2[cell]*c2[cell]*c2[cell];
                           const auto pI_0 = (Z2*p1[cell] + Z1*p2[cell])/(Z1 + Z2);
                           const auto C1   = 2.0*EOS_phase1.get_gamma()*EOS_phase1.get_pi_infty()
                                           + (EOS_phase1.get_gamma() - 1.0)*pI_0;
                           const auto C2   = 2.0*EOS_phase2.get_gamma()*EOS_phase2.get_pi_infty()
                                           + (EOS_phase2.get_gamma() - 1.0)*pI_0;
                           const auto b    = C1*(1.0 - conserved_variables[cell][ALPHA1_INDEX])
                                           + C2*conserved_variables[cell][ALPHA1_INDEX]
                                           - (1.0 + EOS_phase2.get_gamma())*conserved_variables[cell][ALPHA1_INDEX]*p1[cell]
                                           - (1.0 + EOS_phase1.get_gamma())*(1.0 - conserved_variables[cell][ALPHA1_INDEX])*p2[cell];
                           const auto d    = -(C2*conserved_variables[cell][ALPHA1_INDEX]*p1[cell] +
                                               C1*(1.0 - conserved_variables[cell][ALPHA1_INDEX])*p2[cell]);

                           auto p_star     = (-b + std::sqrt(b*b - 4.0*a*d))/(2.0*a);

                           std::cout << "p1 after relaxation = " << p_star << std::endl;
                           std::cout << "p2 after relaxation = " << p_star << std::endl;
                           std::cout << "T1 after relaxation = " << EOS_phase1.T_value_RhoP(conserved_variables[cell][ALPHA1_RHO1_INDEX]/
                                                                                            conserved_variables[cell][ALPHA1_INDEX],
                                                                                            p_star) << std::endl;
                           std::cout << "T2 after relaxation = " << EOS_phase2.T_value_RhoP(conserved_variables[cell][ALPHA2_RHO2_INDEX]/
                                                                                            (1.0 - conserved_variables[cell][ALPHA1_INDEX]),
                                                                                            p_star) << std::endl;

                           // Update the volume fraction using the computed pressure
                           conserved_variables[cell][ALPHA1_INDEX] *= ((EOS_phase1.get_gamma() - 1.0)*p_star + 2*p1[cell] + C1)/
                                                                      ((EOS_phase1.get_gamma() + 1.0)*p_star + C1);

                           // Update the total energy of both phases
                           auto E1 = EOS_phase1.e_value_RhoP(conserved_variables[cell][ALPHA1_RHO1_INDEX]/
                                                             conserved_variables[cell][ALPHA1_INDEX],
                                                             p_star)
                                   + 0.5*vel1[cell]*vel1[cell]; /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*E1;

                           conserved_variables[cell][ALPHA2_RHO2_E2_INDEX] = rhoE_0 - conserved_variables[cell][ALPHA1_RHO1_E1_INDEX];

                           std::cout << "Relaxation terminated cell " << cell << std::endl;
                         });
}

/*--- EXECUTE THE TEMPORAL LOOP ---*/

// Implement the function that effectively performs the temporal loop
//
template<std::size_t dim>
void BN_Solver<dim>::run() {
  /*--- Default output arguemnts ---*/
  fs::path path = fs::current_path();
  #ifdef SULICIU_RELAXATION
    std::string filename = "Relaxation_Suliciu";
  #elifdef RUSANOV_FLUX
    std::string filename = "Rusanov_Flux";
  #endif

  #ifdef ORDER_2
    filename = filename + "_order2";
  #else
    filename = filename + "_order1";
  #endif

  const double dt_save = Tf/static_cast<double>(nfiles);

  /*--- Auxiliary variables to save updated fields ---*/
  #ifdef ORDER_2
    auto conserved_variables_tmp   = samurai::make_field<double, EquationData::NVARS>("conserved_tmp", mesh);
    auto conserved_variables_tmp_2 = samurai::make_field<double, EquationData::NVARS>("conserved_tmp_2", mesh);
  #endif
  auto conserved_variables_np1 = samurai::make_field<double, EquationData::NVARS>("conserved_np1", mesh);

  /*--- Create the flux variables ---*/
  #ifdef SULICIU_RELAXATION
    double c = 0.0;
    auto Suliciu_flux = numerical_flux.make_flux(c);
  #elifdef RUSANOV_FLUX
    auto Rusanov_flux         = numerical_flux_cons.make_flux();
    auto NonConservative_flux = numerical_flux_non_cons.make_flux();
  #endif

  /*--- Save the initial condition ---*/
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(path, filename, suffix_init,
       conserved_variables, rho, p,
       vel1, rho1, p1, c1, T1,
       vel2, rho2, p2, c2, T2, alpha2);

  /*--- Set mesh size ---*/
  const double dx = mesh.cell_length(mesh.max_level());

  /*--- Start the loop ---*/
  std::size_t nsave = 0;
  std::size_t nt    = 0;
  double t          = 0.0;
  while(t != Tf) {
    // Apply mesh adaptation
    samurai::update_ghost_mr(conserved_variables);
    auto MRadaptation = samurai::make_MRAdapt(conserved_variables);
    MRadaptation(1e-2, 1);

    // Apply the numerical scheme
    samurai::update_ghost_mr(conserved_variables);
    #ifdef SULICIU_RELAXATION
      c = 0.0;
      std::cout << "Starting computation of Suliciu scheme iteration " << nt + 1 << std::endl;
      auto Relaxation_Flux = Suliciu_flux(conserved_variables);
      std::cout << "Ending computation of Suliciu scheme iteration " << nt + 1 << std::endl;
      dt = std::min(Tf - t, cfl*dx/c);
      t += dt;
      std::cout << fmt::format("Iteration {}: t = {}, dt = {}", ++nt, t, dt) << std::endl;

      #ifdef ORDER_2
        conserved_variables_tmp.resize();
        conserved_variables_tmp = conserved_variables - dt*Relaxation_Flux;
      #else
        conserved_variables_np1.resize();
        conserved_variables_np1 = conserved_variables - dt*Relaxation_Flux;
      #endif
    #elifdef RUSANOV_FLUX
      auto Cons_Flux    = Rusanov_flux(conserved_variables);
      auto NonCons_Flux = NonConservative_flux(conserved_variables);
      if(nt > 0 && !(t >= static_cast<double>(nsave + 1)*dt_save || t == Tf)) {
        update_auxiliary_fields();
      }
      dt = std::min(Tf - t, cfl*dx/get_max_lambda());
      t += dt;
      std::cout << fmt::format("Iteration {}: t = {}, dt = {}", ++nt, t, dt) << std::endl;

      #ifdef ORDER_2
        conserved_variables_tmp.resize();
        conserved_variables_tmp = conserved_variables - dt*Cons_Flux - dt*NonCons_Flux;
      #else
        conserved_variables_np1.resize();
        conserved_variables_np1 = conserved_variables - dt*Cons_Flux - dt*NonCons_Flux;
      #endif
    #endif

    #ifdef ORDER_2
      std::swap(conserved_variables.array(), conserved_variables_tmp.array());
    #else
      std::swap(conserved_variables.array(), conserved_variables_np1.array());
    #endif

    // Perform relaxation if desired
    if(apply_relaxation) {
      if(apply_finite_rate_relaxation) {
        perform_relaxation_finite_rate();
      }
      else {
        perform_instantaneous_relaxation();
      }
    }
    std::cout << "Finished relaxation" << std::endl;

    // Consider the second stage for the second order
    #ifdef ORDER_2
      samurai::update_ghost_mr(conserved_variables);
      conserved_variables_tmp_2.resize();
      #ifdef SULICIU_RELAXATION
        c = 0.0;
        Relaxation_Flux = Suliciu_flux(conserved_variables);

        conserved_variables_tmp_2 = conserved_variables - dt*Relaxation_Flux;
      #elifdef RUSANOV_FLUX
        Cons_Flux    = Rusanov_flux(conserved_variables);
        NonCons_Flux = NonConservative_flux(conserved_variables);

        conserved_variables_tmp_2 = conserved_variables - dt*Cons_Flux - dt*NonCons_Flux;
      #endif
      conserved_variables_np1 = 0.5*(conserved_variables_tmp + conserved_variables_tmp_2);
      std::swap(conserved_variables.array(), conserved_variables_np1.array());

      // Perform relaxation if desired
      if(apply_relaxation) {
        if(apply_finite_rate_relaxation) {
          perform_relaxation_finite_rate();
        }
        else {
          perform_instantaneous_relaxation();
        }
      }
    #endif

    // Save the results
    if(t >= static_cast<double>(nsave + 1)*dt_save || t == Tf) {
      update_auxiliary_fields();

      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";
      save(path, filename, suffix,
           conserved_variables, rho, p,
           vel1, rho1, p1, c1, T1,
           vel2, rho2, p2, c2, T2, alpha2);
    }
  }
}
