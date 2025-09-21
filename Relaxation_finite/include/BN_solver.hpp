// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#include <samurai/algorithm/update.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>

#include <filesystem>
namespace fs = std::filesystem;

/*--- Add header with auxiliary structs ---*/
#include "containers.hpp"

/*--- Add user implemented boundary condition ---*/
#include "user_bc.hpp"

/*--- Add header file for the multiresolution ---*/
#include <samurai/mr/adapt.hpp>

/*--- Add header for the restart ----*/
#include <samurai/io/restart.hpp>

/*--- Include the headers with the numerical fluxes ---*/
#define SULICIU_RELAXATION
//#define RUSANOV_FLUX

#ifdef SULICIU_RELAXATION
  #include "Suliciu_scheme.hpp"
#elifdef RUSANOV_FLUX
  #include "Rusanov_flux.hpp"
  #include "non_conservative_flux.hpp"
#endif

// This is the class for the simulation of a BN model
//
template<std::size_t dim>
class BN_Solver {
public:
  using Config  = samurai::MRConfig<dim, 2>;
  using Field   = samurai::VectorField<samurai::MRMesh<Config>, double, EquationData<dim>::NVARS, false>;
  using Number  = samurai::Flux<Field>::Number;  /*--- Define the shortcut for the arithmetic type ---*/
  using Indices = samurai::Flux<Field>::Indices; /*--- Shortcut for the indices storage ---*/

  BN_Solver() = default; /*--- Default constructor. This will do nothing
                               and basically will never be used ---*/

  BN_Solver(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
            const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
            const Simulation_Parameters<Number>& sim_param,
            const EOS_Parameters<Number>& eos_param,
            const Riemann_Parameters<Number>& Riemann_param); /*--- Class constrcutor with the arguments related
                                                                    to the grid and to the physics. ---*/

  void run(const unsigned nfiles = 10); /*--- Function which actually executes the temporal loop ---*/

  template<class... Variables>
  void save(const fs::path& path,
            const std::string& suffix,
            const Variables&... fields); /*--- Routine to save the results ---*/

private:
  std::ofstream output_data; /*--- Auxiliary output file to save relevant fields at final time ---*/

  /*--- Now we declare some relevant variables ---*/
  const samurai::Box<double, dim> box;

  samurai::MRMesh<Config> mesh; /*--- Variable to store the mesh ---*/

  using Field_Scalar = samurai::ScalarField<decltype(mesh), Number>;
  using Field_Vect   = samurai::VectorField<decltype(mesh), Number, dim, false>;

  const Number t0; /*--- Initial time of the simulation ---*/
  const Number Tf; /*--- Final time of the simulation ---*/

  bool apply_relaxation; /*--- Set whether to apply or not the relaxation ---*/

  Number cfl; /*--- Courant number of the simulation so as to compute the time step ---*/
  Number dt;  /*--- Time-step (in general modified according to CFL) ---*/

  bool   apply_finite_rate_relaxation; /*--- Set if finite rate relaxation is desired ---*/
  bool   relax_instantaneous_velocity; /*--- Set if instantaneous velocity relaxation is desired in the case of finite rate ---*/
  bool   relax_velocity;               /*--- If instantaneous relaxation, choose whether to relax the velocity ---*/
  bool   relax_pressure;               /*--- If instantaneous relaxation, choose whether to relax the pressure ---*/
  bool   relax_temperature;            /*--- If instantaneous relaxation, choose whether to relax the temperature
                                             (only possible with pressure) ---*/
  Number tau_u;                        /*--- Relaxation parameter for the velocity ---*/
  Number tau_p;                        /*--- Relaxation parameter for the pressure ---*/
  Number tau_T;                        /*--- Relaxation parameter for the temperature ---*/

  const Number      atol_Newton_relaxation; /*--- Absolute tolerance Newton method to recompute conserved variables form deltas ---*/
  const Number      rtol_Newton_relaxation; /*--- Relative tolerance Newton method to recompute conserved variables form deltas ---*/
  const std::size_t max_Newton_iters;       /*--- Maximum number of Newton iterations ---*/

  double MR_param;      /*--- Multiresolution parameter ---*/
  double MR_regularity; /*--- Multiresolution regularity ---*/

  const SG_EOS<Number> EOS_phase1; /*--- Equation of state of phase 1 ---*/
  const SG_EOS<Number> EOS_phase2; /*--- Equation of state of phase 2 ---*/

  #ifdef SULICIU_RELAXATION
    samurai::RelaxationFlux<Field> numerical_flux; /*--- function to compute the numerical flux
                                                         (this is necessary to call 'make_flux') ---*/
  #elifdef RUSANOV_FLUX
    samurai::RusanovFlux<Field> numerical_flux_cons; /*--- function to compute the numerical flux for the conservative part
                                                           (this is necessary to call 'make_flux') ---*/

    samurai::NonConservativeFlux<Field> numerical_flux_non_cons; /*--- function to compute the numerical flux for the non-conservative part
                                                                       (this is necessary to call 'make_flux') ---*/
  #endif

  std::string filename; /*--- Auxiliary variable to store the name of output ---*/

  Field conserved_variables; /*--- The variable which stores the conserved variables,
                                   namely the variables for which we solve a PDE system ---*/

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
               T2,
               Y1,
               Y2,
               delta_pres,
               delta_temp;

  Field_Vect vel1,
             vel2,
             vel,
             delta_vel;

  Field_Scalar entropy_after_flux_phase1,
               entropy_production_flux_phase1,
               entropy_after_relaxation_phase1,
               entropy_production_relaxation_phase1,
               entropy_after_flux_phase2,
               entropy_production_flux_phase2,
               entropy_after_relaxation_phase2,
               entropy_production_relaxation_phase2;

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  void create_fields(); /*--- Auxiliary routine to initialize the fileds to the mesh ---*/

  void init_variables(const Riemann_Parameters<Number>& Riemann_param); /*--- Routine to initialize the variables
                                                                              (both conserved and auxiliary, this is problem dependent) ---*/

  void apply_bcs(const Riemann_Parameters<Number>& Riemann_param); /*--- Auxiliary routine for the boundary conditions ---*/

  #ifdef RUSANOV_FLUX
    Number get_max_lambda() const; /*--- Compute the estimate of the maximum eigenvalue ---*/
  #endif

  void update_auxiliary_fields(); /*--- Routine to update auxiliary fields for output and time step update ---*/

  using Matrix_Relaxation = std::array<std::array<Number, 2>, 2>;
  using Vector_Relaxation = std::array<Number, 2>;
  Matrix_Relaxation A_relax; /*--- Matrix associated to the relaxation ---*/
  Vector_Relaxation S_relax; /*--- Vector associated to the source term ---*/

  Matrix_Relaxation Jac_update;     /*--- Matrix associated to the Jacobian of the Newton method to update consered variables ---*/
  Matrix_Relaxation inv_Jac_update; /*--- Inverse of the Jacobian of the Newton method to update consered variables ---*/

  void perform_relaxation_finite_rate(); /*--- Routine to perform the finite-rate relaxation (following Jomée 2023) ---*/

  void perform_relaxation_finite_rate_pT(); /*--- Routine to perform the finite-rate relaxation for pressure and temperature (following Jomée 2023) ---*/

  void perform_instantaneous_velocity_relaxation(); /*--- Routine to perform instantaneous velocity relaxation ---*/

  void perform_instantaneous_velocity_pressure_relaxation(); /*--- Routine to perform instantaneous pressure relaxation (velocity before) ---*/

  void perform_instantaneous_pressure_relaxation(); /*--- Routine to perform instantaneous pressure relaxation ---*/

  void perform_instantaneous_relaxation(); /*--- Routine to perform instantaneous relaxation (velocity and pressure-temperature) ---*/

  template<typename State>
  void compute_coefficients_source_relaxation(const State& q,
                                              const std::array<Number, dim>& delta_u_loc,
                                              Matrix_Relaxation& A, Vector_Relaxation& S); /*--- Compute the coefficients
                                                                                                 and the source term for the relaxation ---*/

  void compute_entropy_after_flux(); /*--- Keep track of entropy and entropy production after convective step ---*/

  void compute_entropy_after_relaxation(); /*--- Keep track of entropy and entropy production after relaxation step ---*/
};

//////////////////////////////////////////////////////////////
/*---- START WITH THE IMPLEMENTATION OF THE CONSTRUCTOR ---*/
//////////////////////////////////////////////////////////////

// Implement class constructor
//
template<std::size_t dim>
BN_Solver<dim>::BN_Solver(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                          const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                          const Simulation_Parameters<Number>& sim_param,
                          const EOS_Parameters<Number>& eos_param,
                          const Riemann_Parameters<Number>& Riemann_param):
  box(min_corner, max_corner),
  t0(sim_param.t0), Tf(sim_param.Tf),
  apply_relaxation(sim_param.apply_relaxation),
  cfl(sim_param.Courant),
  apply_finite_rate_relaxation(sim_param.apply_finite_rate_relaxation),
  relax_instantaneous_velocity(sim_param.relax_instantaneous_velocity),
  relax_velocity(sim_param.relax_velocity),
  relax_pressure(sim_param.relax_pressure),
  relax_temperature(sim_param.relax_temperature),
  tau_u(sim_param.tau_u), tau_p(sim_param.tau_p), tau_T(sim_param.tau_T),
  atol_Newton_relaxation(sim_param.atol_Newton_relaxation),
  rtol_Newton_relaxation(sim_param.rtol_Newton_relaxation),
  max_Newton_iters(sim_param.max_Newton_iters),
  MR_param(sim_param.MR_param), MR_regularity(sim_param.MR_regularity),
  EOS_phase1(eos_param.gamma_1, eos_param.pi_infty_1, eos_param.q_infty_1, eos_param.c_v_1),
  EOS_phase2(eos_param.gamma_2, eos_param.pi_infty_2, eos_param.q_infty_2, eos_param.c_v_2),
  #ifdef SULICIU_RELAXATION
    numerical_flux(EOS_phase1, EOS_phase2,
                   sim_param.atol_Newton_Suliciu, sim_param.rtol_Newton_Suliciu,
                   sim_param.max_Newton_iters)
  #elifdef RUSANOV_FLUX
    numerical_flux_cons(EOS_phase1, EOS_phase2),
    numerical_flux_non_cons(EOS_phase1, EOS_phase2)
  #endif
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) {
      std::cout << "Initializing variables " << std::endl;
      std::cout << std::endl;
    }

    /*--- Attach the fields to the mesh ---*/
    create_fields();

    /*--- Initialize the fields ---*/
    if(sim_param.restart_file.empty()) {
      mesh = {box, sim_param.min_level, sim_param.max_level, {{false}}};
      init_variables(Riemann_param);
    }
    else {
      samurai::load(sim_param.restart_file, mesh, conserved_variables,
                                                  rho, p, vel,
                                                  rho1, p1, c1, vel1, T1, Y1,
                                                  rho2, p2, c2, alpha2, vel2, T2, Y2,
                                                  delta_pres, delta_temp, delta_vel,
                                                  entropy_after_flux_phase1,
                                                  entropy_production_flux_phase1,
                                                  entropy_after_relaxation_phase1,
                                                  entropy_production_relaxation_phase1,
                                                  entropy_after_flux_phase2,
                                                  entropy_production_flux_phase2,
                                                  entropy_after_relaxation_phase2,
                                                  entropy_production_relaxation_phase2);
    }

    /*--- Apply boundary conditions ---*/
    apply_bcs(Riemann_param);
  }

// Auxiliary routine to create the fields
//
template<std::size_t dim>
void BN_Solver<dim>::create_fields() {
  /*--- Create conserved and auxiliary fields ---*/
  conserved_variables = samurai::make_vector_field<Number, Field::n_comp>("conserved", mesh);

  rho  = samurai::make_scalar_field<Number>("rho", mesh);
  p    = samurai::make_scalar_field<Number>("p", mesh);

  rho1 = samurai::make_scalar_field<Number>("rho1", mesh);
  p1   = samurai::make_scalar_field<Number>("p1", mesh);
  c1   = samurai::make_scalar_field<Number>("c1", mesh);

  rho2 = samurai::make_scalar_field<Number>("rho2", mesh);
  p2   = samurai::make_scalar_field<Number>("p2", mesh);
  c2   = samurai::make_scalar_field<Number>("c2", mesh);

  vel1 = samurai::make_vector_field<Number, dim>("vel1", mesh);
  vel2 = samurai::make_vector_field<Number, dim>("vel2", mesh);
  vel  = samurai::make_vector_field<Number, dim>("vel", mesh);

  alpha2 = samurai::make_scalar_field<Number>("alpha2", mesh);
  Y1     = samurai::make_scalar_field<Number>("Y1", mesh);
  Y2     = samurai::make_scalar_field<Number>("Y2", mesh);

  T1 = samurai::make_scalar_field<Number>("T1", mesh);
  T2 = samurai::make_scalar_field<Number>("T2", mesh);

  delta_pres = samurai::make_scalar_field<Number>("delta_pres", mesh);
  delta_temp = samurai::make_scalar_field<Number>("delta_temp", mesh);
  delta_vel  = samurai::make_vector_field<Number, dim>("delta_vel", mesh);

  /*--- Create auxiliary fields to keep track of the entropy ---*/
  entropy_after_flux_phase1            = samurai::make_scalar_field<Number>("entropy_after_flux_phase1", mesh);
  entropy_production_flux_phase1       = samurai::make_scalar_field<Number>("entropy_production_flux_phase1", mesh);
  entropy_after_relaxation_phase1      = samurai::make_scalar_field<Number>("entropy_after_relaxation_phase1", mesh);
  entropy_production_relaxation_phase1 = samurai::make_scalar_field<Number>("entropy_production_relaxation_phase1", mesh);

  entropy_after_flux_phase2            = samurai::make_scalar_field<Number>("entropy_after_flux_phase2", mesh);
  entropy_production_flux_phase2       = samurai::make_scalar_field<Number>("entropy_production_flux_phase2", mesh);
  entropy_after_relaxation_phase2      = samurai::make_scalar_field<Number>("entropy_after_relaxation_phase2", mesh);
  entropy_production_relaxation_phase2 = samurai::make_scalar_field<Number>("entropy_production_relaxation_phase2", mesh);
}

// Initialization of conserved and auxiliary variables
//
template<std::size_t dim>
void BN_Solver<dim>::init_variables(const Riemann_Parameters<Number>& Riemann_param) {
  /*--- Reisze fields since now mesh has been surely created ---*/
  conserved_variables.resize();
  rho.resize();
  p.resize();
  rho1.resize();
  p1.resize();
  c1.resize();
  rho2.resize();
  p2.resize();
  c2.resize();
  vel1.resize();
  vel2.resize();
  vel.resize();
  alpha2.resize();
  Y1.resize();
  Y2.resize();
  T1.resize();
  T2.resize();
  delta_pres.resize();
  delta_temp.resize();
  delta_vel.resize();
  entropy_after_flux_phase1.resize();
  entropy_production_flux_phase1.resize();
  entropy_after_relaxation_phase1.resize();
  entropy_production_relaxation_phase1.resize();
  entropy_after_flux_phase2.resize();
  entropy_production_flux_phase2.resize();
  entropy_after_relaxation_phase2.resize();
  entropy_production_relaxation_phase2.resize();

  /*--- Initialize the fields with a loop over all cells ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              conserved_variables[cell][Indices::ALPHA1_INDEX] = Riemann_param.alpha1L;

                              p1[cell]      = Riemann_param.p1L;
                              vel1[cell][0] = Riemann_param.u1L;
                              rho1[cell]    = Riemann_param.rho1L;

                              p2[cell]      = Riemann_param.p2L;
                              vel2[cell][0] = Riemann_param.u2L;
                              rho2[cell]    = Riemann_param.rho2L;

                              T1[cell] = EOS_phase1.T_value_RhoP(rho1[cell], p1[cell]);

                              conserved_variables[cell][Indices::ALPHA1_RHO1_INDEX] =
                              conserved_variables[cell][Indices::ALPHA1_INDEX]*rho1[cell];

                              const auto e1   = EOS_phase1.e_value_RhoP(rho1[cell], p1[cell]);
                              auto norm2_vel1 = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                conserved_variables[cell][Indices::ALPHA1_RHO1_U1_INDEX + d] =
                                conserved_variables[cell][Indices::ALPHA1_RHO1_INDEX]*vel1[cell][d];

                                norm2_vel1 += vel1[cell][d]*vel1[cell][d];
                              }

                              conserved_variables[cell][Indices::ALPHA1_RHO1_E1_INDEX] =
                              conserved_variables[cell][Indices::ALPHA1_RHO1_INDEX]*
                              (e1 + static_cast<Number>(0.5)*norm2_vel1);

                              T2[cell] = EOS_phase2.T_value_RhoP(rho2[cell], p2[cell]);

                              alpha2[cell] = static_cast<Number>(1.0)
                                           - conserved_variables[cell][Indices::ALPHA1_INDEX];
                              conserved_variables[cell][Indices::ALPHA2_RHO2_INDEX] = alpha2[cell]*rho2[cell];

                              const auto e2   = EOS_phase2.e_value_RhoP(rho2[cell], p2[cell]);
                              auto norm2_vel2 = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                conserved_variables[cell][Indices::ALPHA2_RHO2_U2_INDEX + d] =
                                conserved_variables[cell][Indices::ALPHA2_RHO2_INDEX]*vel2[cell][d];

                                norm2_vel2 += vel2[cell][d]*vel2[cell][d];
                              }
                              conserved_variables[cell][Indices::ALPHA2_RHO2_E2_INDEX] =
                              conserved_variables[cell][Indices::ALPHA2_RHO2_INDEX]*
                              (e2 + static_cast<Number>(0.5)*norm2_vel2);

                              c1[cell] = EOS_phase1.c_value_RhoP(rho1[cell], p1[cell]);

                              c2[cell] = EOS_phase2.c_value_RhoP(rho2[cell], p2[cell]);

                              rho[cell] = conserved_variables[cell][Indices::ALPHA1_RHO1_INDEX]
                                        + conserved_variables[cell][Indices::ALPHA2_RHO2_INDEX];

                              p[cell] = conserved_variables[cell][Indices::ALPHA1_INDEX]*p1[cell]
                                      + alpha2[cell]*p2[cell];

                              Y1[cell] = conserved_variables[cell][Indices::ALPHA1_RHO1_INDEX]/rho[cell];
                              Y2[cell] = conserved_variables[cell][Indices::ALPHA2_RHO2_INDEX]/rho[cell];

                              for(std::size_t d = 0; d < dim; ++d) {
                                vel[cell][d] = Y1[cell]*vel1[cell][d]
                                             + Y2[cell]*vel2[cell][d];
                              }

                              // Save deltas
                              delta_pres[cell] = p1[cell] - p2[cell];
                              delta_temp[cell] = T1[cell] - T2[cell];
                              for(std::size_t d = 0; d < dim; ++d) {
                                delta_vel[cell][d] = vel1[cell][d] - vel2[cell][d];
                              }

                              // Compute entropies
                              entropy_after_flux_phase1[cell]            = EOS_phase1.s_value_Rhoe(rho1[cell], e1);
                              entropy_after_relaxation_phase1[cell]      = EOS_phase1.s_value_Rhoe(rho1[cell], e1);
                              entropy_production_flux_phase1[cell]       = static_cast<Number>(0.0);
                              entropy_production_relaxation_phase1[cell] = static_cast<Number>(0.0);

                              entropy_after_flux_phase2[cell]            = EOS_phase2.s_value_Rhoe(rho2[cell], e2);
                              entropy_after_relaxation_phase2[cell]      = EOS_phase2.s_value_Rhoe(rho2[cell], e2);
                              entropy_production_flux_phase2[cell]       = static_cast<Number>(0.0);
                              entropy_production_relaxation_phase2[cell] = static_cast<Number>(0.0);
                            }
                        );
}

// Auxiliary routine to impose the boundary conditions
//
template<std::size_t dim>
void BN_Solver<dim>::apply_bcs(const Riemann_Parameters<Number>& Riemann_param) {
  const samurai::DirectionVector<dim> left  = {-1};
  const samurai::DirectionVector<dim> right = {1};

  samurai::make_bc<samurai::Dirichlet<1>>(conserved_variables,
                                          Riemann_param.alpha1L,
                                          Riemann_param.alpha1L*Riemann_param.rho1L,
                                          Riemann_param.alpha1L*Riemann_param.rho1L*Riemann_param.u1L,
                                          Riemann_param.alpha1L*Riemann_param.rho1L*
                                          (EOS_phase1.e_value_RhoP(Riemann_param.rho1L, Riemann_param.p1L) +
                                           static_cast<Number>(0.5)*Riemann_param.u1L*Riemann_param.u1L),
                                          (static_cast<Number>(1.0) - Riemann_param.alpha1L)*Riemann_param.rho2L,
                                          (static_cast<Number>(1.0) - Riemann_param.alpha1L)*Riemann_param.rho2L*Riemann_param.u2L,
                                          (static_cast<Number>(1.0) - Riemann_param.alpha1L)*Riemann_param.rho2L*
                                          (EOS_phase2.e_value_RhoP(Riemann_param.rho2L, Riemann_param.p2L) +
                                           static_cast<Number>(0.5)*Riemann_param.u2L*Riemann_param.u2L))->on(left);
  samurai::make_bc<Default>(conserved_variables,
                            Outflow(conserved_variables,
                                    Riemann_param.p1L, EOS_phase1,
                                    Riemann_param.p2L, EOS_phase2))->on(right);
}

//////////////////////////////////////////////////////////////
/*---- FOCUS NOW ON THE AUXILIARY FUNCTIONS ---*/
/////////////////////////////////////////////////////////////

// Compute the estimate of the maximum eigenvalue for CFL condition
//
#ifdef RUSANOV_FLUX
  template<std::size_t dim>
  typename BN_Solver<dim>::Number BN_Solver<dim>::get_max_lambda() const {
    auto local_res = static_cast<Number>(0.0);

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                              {
                                for(std::size_t d = 0; d < dim; ++d) {
                                  local_res = std::max(std::max(std::abs(vel1[cell][d]) + c1[cell],
                                                                std::abs(vel2[cell][d]) + c2[cell]),
                                                       local_res);
                                }
                              }
                          );

    double global_res;
    MPI_Allreduce(&local_res, &global_res, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    return global_res;
  }
#endif

// Update auxiliary fields after solution of the system
//
template<std::size_t dim>
void BN_Solver<dim>::update_auxiliary_fields() {
  /*--- Resize fields because of (possible) multiresolution ---*/
  rho.resize();
  p.resize();
  vel.resize();

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
  Y1.resize();
  Y2.resize();

  delta_pres.resize();
  delta_temp.resize();
  delta_vel.resize();

  /*--- Loop over all cells ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Pre-fetch varaibles used multiple times in order to exploit (possible) vectorization
                              // as well as to enhance readability
                              const auto alpha1_loc = conserved_variables[cell][Indices::ALPHA1_INDEX];
                              const auto m1_loc     = conserved_variables[cell][Indices::ALPHA1_RHO1_INDEX];
                              const auto m1E1_loc   = conserved_variables[cell][Indices::ALPHA1_RHO1_E1_INDEX];
                              const auto m2_loc     = conserved_variables[cell][Indices::ALPHA2_RHO2_INDEX];
                              const auto m2E2_loc   = conserved_variables[cell][Indices::ALPHA2_RHO2_E2_INDEX];

                              // Compute the fields
                              const auto rho_loc     = m1_loc + m2_loc;
                              const auto inv_rho_loc = static_cast<Number>(1.0)/rho_loc;
                              rho[cell]              = rho_loc;

                              const auto rho1_loc   = m1_loc/alpha1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              rho1[cell]            = rho1_loc;
                              const auto inv_m1_loc = static_cast<Number>(1.0)/m1_loc;
                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              auto e1_loc           = m1E1_loc*inv_m1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              for(std::size_t d = 0; d < dim; ++d) {
                                vel1[cell][d] = conserved_variables[cell][Indices::ALPHA1_RHO1_U1_INDEX + d]*inv_m1_loc;
                                                /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                e1_loc -= static_cast<Number>(0.5)*(vel1[cell][d]*vel1[cell][d]);
                              }
                              const auto p1_loc = EOS_phase1.pres_value_Rhoe(rho1_loc, e1_loc);
                              p1[cell]          = p1_loc;
                              c1[cell]          = EOS_phase1.c_value_RhoP(rho1_loc, p1_loc);
                              const auto T1_loc = EOS_phase1.T_value_RhoP(rho1_loc, p1_loc);
                              T1[cell]          = T1_loc;

                              const auto alpha2_loc = static_cast<Number>(1.0) - alpha1_loc;
                              const auto rho2_loc   = m2_loc/alpha2_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              rho2[cell]            = rho2_loc;
                              const auto inv_m2_loc = static_cast<Number>(1.0)/m2_loc;
                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              auto e2_loc           = m2E2_loc*inv_m2_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              for(std::size_t d = 0; d < dim; ++d) {
                                vel2[cell][d] = conserved_variables[cell][Indices::ALPHA2_RHO2_U2_INDEX + d]*inv_m2_loc;
                                                /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                e2_loc -= static_cast<Number>(0.5)*(vel2[cell][d]*vel2[cell][d]);
                              }
                              const auto p2_loc = EOS_phase2.pres_value_Rhoe(rho2_loc, e2_loc);
                              p2[cell]          = p2_loc;
                              c2[cell]          = EOS_phase2.c_value_RhoP(rho2_loc, p2_loc);
                              const auto T2_loc = EOS_phase2.T_value_RhoP(rho2_loc, p2_loc);
                              T2[cell]          = T2_loc;

                              // Compute remaining useful fields
                              alpha2[cell] = alpha2_loc;

                              const auto Y1_loc = m1_loc*inv_rho_loc;
                              Y1[cell]          = Y1_loc;
                              const auto Y2_loc = static_cast<Number>(1.0) - Y1_loc;
                              Y2[cell]          = Y2_loc;

                              for(std::size_t d = 0; d < dim; ++d) {
                                vel[cell][d] = Y1_loc*vel1[cell][d]
                                             + Y2_loc*vel2[cell][d];
                              }

                              p[cell] = alpha1_loc*p1_loc
                                      + alpha2_loc*p2_loc;

                              // Save deltas
                              delta_pres[cell] = p1_loc - p2_loc;
                              delta_temp[cell] = T1_loc - T2_loc;
                              for(std::size_t d = 0; d < dim; ++d) {
                                delta_vel[cell][d] = vel1[cell][d] - vel2[cell][d];
                              }
                            }
                        );
}

// Save desired fields and info
//
template<std::size_t dim>
template<class... Variables>
void BN_Solver<dim>::save(const fs::path& path,
                          const std::string& suffix,
                          const Variables&... fields) {
  auto level_ = samurai::make_scalar_field<std::size_t>("level", mesh);

  if(!fs::exists(path)) {
    fs::create_directory(path);
  }

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              level_[cell] = cell.level;
                            }
                        );

  samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, fields..., level_);
  samurai::dump(path, fmt::format("{}{}", filename, "_restart"), mesh, fields..., level_);
}

//////////////////////////////////////////////////////////////
/*---- FOCUS NOW ON THE RELAXATION FUNCTIONS ---*/
/////////////////////////////////////////////////////////////

// Routine to compute the source terms associated to the finite-rate relaxation
//
template<std::size_t dim>
template<typename State>
void BN_Solver<dim>::compute_coefficients_source_relaxation(const State& q,
                                                            const std::array<Number, dim>& delta_u_loc,
                                                            Matrix_Relaxation& A, Vector_Relaxation& S) {
  /*--- Pre-fetch variables that will be used several times so as to exploit (possible) vectorization
        as well as to enhance readability ---*/
  const auto alpha1_loc = q[Indices::ALPHA1_INDEX];
  const auto m1_loc     = q[Indices::ALPHA1_RHO1_INDEX];
  const auto m1E1_loc   = q[Indices::ALPHA1_RHO1_E1_INDEX];
  const auto m2_loc     = q[Indices::ALPHA2_RHO2_INDEX];
  const auto m2E2_loc   = q[Indices::ALPHA2_RHO2_E2_INDEX];

  /*--- Compute auxiliary variables for phase 1 ---*/
  const auto inv_alpha1_loc = static_cast<Number>(1.0)/alpha1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
  const auto rho1_loc       = m1_loc*inv_alpha1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
  std::array<Number, dim> vel1_loc;
  const auto inv_m1_loc = static_cast<Number>(1.0)/m1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
  auto e1_loc           = m1E1_loc*inv_m1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
  for(std::size_t d = 0; d < dim; ++d) {
    vel1_loc[d] = q[Indices::ALPHA1_RHO1_U1_INDEX + d]*inv_m1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    e1_loc -= static_cast<Number>(0.5)*vel1_loc[d]*vel1_loc[d];
  }
  const auto p1_loc = EOS_phase1.pres_value_Rhoe(rho1_loc, e1_loc);
  const auto c1_loc = EOS_phase1.c_value_RhoP(rho1_loc, p1_loc);
  const auto kappa1 = EOS_phase1.de_dP_rho(p1_loc, rho1_loc);
  const auto T1_loc = EOS_phase1.T_value_RhoP(rho1_loc, p1_loc);
  const auto cv1    = EOS_phase1.de_dT_rho(T1_loc, rho1_loc);
  const auto Gamma1 = EOS_phase1.de_drho_T(rho1_loc, T1_loc);

  /*--- Compute auxiliary variables for phase 2 ---*/
  const auto alpha2_loc     = static_cast<Number>(1.0) - alpha1_loc;
  const auto inv_alpha2_loc = static_cast<Number>(1.0)/alpha1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
  const auto rho2_loc       = m2_loc*inv_alpha2_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
  std::array<Number, dim> vel2_loc;
  const auto inv_m2_loc = static_cast<Number>(1.0)/m2_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
  auto e2_loc           = m2E2_loc*inv_m2_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
  for(std::size_t d = 0; d < dim; ++d) {
    vel2_loc[d] = q[Indices::ALPHA2_RHO2_U2_INDEX + d]*inv_m2_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    e2_loc -= static_cast<Number>(0.5)*vel2_loc[d]*vel2_loc[d];
  }
  const auto p2_loc = EOS_phase2.pres_value_Rhoe(rho2_loc, e2_loc);
  const auto c2_loc = EOS_phase2.c_value_RhoP(rho2_loc, p2_loc);
  const auto kappa2 = EOS_phase2.de_dP_rho(p2_loc, rho2_loc);
  const auto T2_loc = EOS_phase2.T_value_RhoP(rho2_loc, p2_loc);
  const auto cv2    = EOS_phase2.de_dT_rho(T2_loc, rho2_loc);
  const auto Gamma2 = EOS_phase2.de_drho_T(rho2_loc, T2_loc);

  /*--- uI = beta*u1 + (1 - beta)*u2
        pI = chi*p1 + (1 - chi)*p2 ---*/
  auto beta = static_cast<Number>(1.0);
  auto chi  = (static_cast<Number>(1.0) - beta)*T2_loc/
              ((static_cast<Number>(1.0) - beta)*T2_loc + beta*T1_loc);
  auto pI_relax = chi*p1_loc
                + (static_cast<Number>(1.0) - chi)*p2_loc;
  /*--- TODO: Possibly change, a priori this is not necessarily the same of the convective operator, even though
              substituting uI \cdot grad\alpha we get d\alpha/dt... ---*/

  /*--- Compute the coefficients ---*/
  const auto p_ref_loc     = rho1_loc*c1_loc*c1_loc*inv_alpha1_loc
                           + rho2_loc*c2_loc*c2_loc*inv_alpha2_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
  const auto p_relax_coeff = (alpha1_loc*alpha2_loc)/(tau_p*p_ref_loc);
  const auto T_relax_coeff = (m1_loc*cv1*m2_loc*cv2)/
                             (tau_T*(m1_loc*cv1 + m2_loc*cv2));
  const auto a_pp = -p_relax_coeff*(rho1_loc*c1_loc*c1_loc*inv_alpha1_loc +
                                    rho2_loc*c2_loc*c2_loc*inv_alpha2_loc +
                                    ((chi - static_cast<Number>(1.0))*inv_m1_loc/kappa1 +
                                     chi*inv_m2_loc/kappa2)*
                                    (p1_loc - p2_loc)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
  const auto a_pT = -T_relax_coeff*(inv_m1_loc/kappa1 + inv_m2_loc/kappa2); /*--- TODO: Add treatment for vanishing volume fraction ---*/
  const auto a_Tp = -p_relax_coeff*((pI_relax - rho1_loc*rho1_loc*Gamma1)*inv_m1_loc/cv1 +
                                    (pI_relax - rho2_loc*rho2_loc*Gamma2)*inv_m2_loc/cv2);
  const auto a_TT = -T_relax_coeff*(inv_m1_loc*cv1 + inv_m2_loc*cv2); /*--- TODO: Add treatment for vanishing volume fraction ---*/

  A[0][0] = static_cast<Number>(1.0) - dt*a_pp;
  A[0][1] = -dt*a_pT;
  A[1][0] = -dt*a_Tp;
  A[1][1] = static_cast<Number>(1.0) - dt*a_TT;

  /*--- Set source term ---*/
  const auto rho_0      = m1_loc + m2_loc;
  const auto Y1_0       = m1_loc/rho_0;
  const auto Y2_0       = static_cast<Number>(1.0) - Y1_0;
  const auto a_tilde_pu = -static_cast<Number>(1.0)/tau_u*
                          ((Y2_0*(beta - static_cast<Number>(1.0)))/kappa1 + (Y1_0*beta)/kappa2);
  const auto a_tilde_Tu = -static_cast<Number>(1.0)/tau_u*
                          ((Y2_0*(beta - static_cast<Number>(1.0)))/cv1 + (Y1_0*beta)/cv2);
  S[0] = static_cast<Number>(0.0);
  S[1] = static_cast<Number>(0.0);
  for(std::size_t d = 0; d < dim; ++d) {
    S[0] += (a_tilde_pu*(vel1_loc[d] - vel2_loc[d]))*delta_u_loc[d];
    S[1] += (a_tilde_Tu*(vel1_loc[d] - vel2_loc[d]))*delta_u_loc[d];
  }
}

// Finite rate relaxation (following Jomée 2023)
//
template<std::size_t dim>
void BN_Solver<dim>::perform_relaxation_finite_rate() {
  /*--- Resize fields because of (possible) multiresolution ---*/
  vel1.resize();
  vel2.resize();

  /*--- Loop over all cells ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Pre-fetch several variables that will be used several times so as to exploit (possible) vectorization
                              // as well as to enhance readability
                              auto alpha1_loc   = conserved_variables[cell][Indices::ALPHA1_INDEX];
                              const auto m1_loc = conserved_variables[cell][Indices::ALPHA1_RHO1_INDEX];
                              auto m1E1_loc     = conserved_variables[cell][Indices::ALPHA1_RHO1_E1_INDEX];
                              const auto m2_loc = conserved_variables[cell][Indices::ALPHA2_RHO2_INDEX];
                              auto m2E2_loc     = conserved_variables[cell][Indices::ALPHA2_RHO2_E2_INDEX];

                              // Compute updated delta_u (we have analytical formula)
                              std::array<Number, dim> delta_u;
                              const auto inv_m1_loc = static_cast<Number>(1.0)/m1_loc;
                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              const auto inv_m2_loc = static_cast<Number>(1.0)/m2_loc;
                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              for(std::size_t d = 0; d < dim; ++d) {
                                vel1[cell][d] = conserved_variables[cell][Indices::ALPHA1_RHO1_U1_INDEX + d]*inv_m1_loc;
                                                /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                vel2[cell][d] = conserved_variables[cell][Indices::ALPHA2_RHO2_U2_INDEX + d]*inv_m2_loc;
                                                /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                delta_u[d] = (vel1[cell][d] - vel2[cell][d])*std::exp(-dt/tau_u);
                              }

                              // Solve the system for delta_p and delta_T
                              /*--- Compute the auxiliary fields to initalize delta_p and delta_T ---*/
                              auto rho1_loc = m1_loc/alpha1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              auto e1_loc   = m1E1_loc*inv_m1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              for(std::size_t d = 0; d < dim; ++d) {
                                e1_loc -= static_cast<Number>(0.5)*vel1[cell][d]*vel1[cell][d];
                              }
                              auto p1_loc = EOS_phase1.pres_value_Rhoe(rho1_loc, e1_loc);

                              auto rho2_loc = m2_loc/(static_cast<Number>(1.0) - alpha1_loc);
                              auto e2_loc   = m2E2_loc*inv_m2_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              for(std::size_t d = 0; d < dim; ++d) {
                                e2_loc -= static_cast<Number>(0.5)*vel2[cell][d]*vel2[cell][d];
                                // Recall that vel1 and vel2 are the initial values!!!!
                              }
                              auto p2_loc = EOS_phase2.pres_value_Rhoe(rho2_loc, e2_loc);

                              /*--- Compute matrix relaxation coefficients ---*/
                              compute_coefficients_source_relaxation(conserved_variables[cell],
                                                                     delta_u, A_relax, S_relax);

                              /*--- Solve the linear system ---*/
                              Number delta_p,
                                     delta_T;
                              const auto delta_p0 = p1_loc - p2_loc;
                              auto T1_loc         = EOS_phase1.T_value_RhoP(rho1_loc, p1_loc);
                              auto T2_loc         = EOS_phase2.T_value_RhoP(rho2_loc, p2_loc);
                              const auto delta_T0 = T1_loc - T2_loc;
                              const auto det_A_pT = A_relax[0][0]*A_relax[1][1]
                                                  - A_relax[0][1]*A_relax[1][0];
                              if(std::abs(det_A_pT) > static_cast<Number>(1e-10)) {
                                delta_p = (static_cast<Number>(1.0)/det_A_pT)*
                                          (A_relax[1][1]*(delta_p0 + dt*S_relax[0]) -
                                           A_relax[0][1]*(delta_T0 + dt*S_relax[1]));
                                delta_T = (static_cast<Number>(1.0)/det_A_pT)*
                                          (-A_relax[1][0]*(delta_p0 + dt*S_relax[0]) +
                                            A_relax[0][0]*(delta_T0 + dt*S_relax[1]));
                              }
                              else {
                                int rank;
                                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                                if(rank == 0) {
                                  std::cerr << "Singular matrix in the relaxation" << std::endl;
                                }
                                exit(1);
                              }

                              // Re-update conserved variables
                              /*--- Start from phasic momentum (compute also useful norms) ---*/
                              const auto rho_0  = m1_loc + m2_loc;
                              const auto Y1_0   = m1_loc/rho_0;
                              const auto Y2_0   = static_cast<Number>(1.0) - Y1_0;
                              auto norm2_um     = static_cast<Number>(0.0);
                              auto norm2_deltau = static_cast<Number>(0.0);
                              auto norm2_vel1   = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                const auto um_d = Y1_0*vel1[cell][d]
                                                + Y2_0*vel2[cell][d];

                                norm2_um += um_d*um_d;

                                vel1[cell][d] = um_d + Y2_0*delta_u[d];
                                conserved_variables[cell][Indices::ALPHA1_RHO1_U1_INDEX + d] = m1_loc*vel1[cell][d];
                                norm2_vel1 += vel1[cell][d]*vel1[cell][d];

                                vel2[cell][d] = um_d - Y1_0*delta_u[d];
                                conserved_variables[cell][Indices::ALPHA2_RHO2_U2_INDEX + d] = m2_loc*vel2[cell][d];

                                norm2_deltau += delta_u[d]*delta_u[d];
                              }

                              /*--- Newton method loop to compute pressure and temperature (so far we have compute just delta_p and delta_T)
                                    Pay attention to the variable that we employ in the Newton: it is basically necessary to employ the most
                                    restricting from a thermodynamic point of view so as to compute always admissible state for both phases
                                    In this case, this means gas phase, i.e. phase 2 (p2, T2). Suppose on the contrary that we update p1.
                                    If p1 at the beginning is, e.g., -10^3 Pa (typically admissible for the liquid) and delta_p is close to zero
                                    because we are miming an instantaneous relaxation, then p2 is set initially to around -10^3 which is not admissible!!! ---*/
                              const auto rhoE_0 = m1E1_loc + m2E2_loc;
                              const auto rhoe_0 = rhoE_0
                                                - static_cast<Number>(0.5)*rho_0*(norm2_um + Y1_0*Y2_0*norm2_deltau);
                              auto dp2 = std::numeric_limits<Number>::max();
                              auto dT2 = std::numeric_limits<Number>::max();
                              unsigned iter;
                              auto f1 = m1_loc/EOS_phase1.rho_value_PT(p2_loc + delta_p, T2_loc + delta_T)
                                      + m2_loc/EOS_phase2.rho_value_PT(p2_loc, T2_loc)
                                      - static_cast<Number>(1.0);
                              auto f2 = m1_loc*EOS_phase1.e_value_PT(p2_loc + delta_p, T2_loc + delta_T)
                                      + m2_loc*EOS_phase2.e_value_PT(p2_loc, T2_loc)
                                      - rhoe_0;
                              for(iter = 0; iter < max_Newton_iters; ++iter) {
                                if((std::abs(dp2) > atol_Newton_relaxation + rtol_Newton_relaxation*std::abs(p2[cell]) ||
                                    std::abs(dT2) > atol_Newton_relaxation + rtol_Newton_relaxation*std::abs(T2[cell])) &&
                                   (std::abs(f1) > atol_Newton_relaxation ||
                                    std::abs(f2) > atol_Newton_relaxation + rtol_Newton_relaxation*std::abs(rhoe_0))) {
                                  // Compute Jacobian matrix
                                  Jac_update[0][0] = -m1_loc/
                                                     (EOS_phase1.rho_value_PT(p2_loc + delta_p, T2_loc + delta_T)*
                                                      EOS_phase1.rho_value_PT(p2_loc + delta_p, T2_loc + delta_T))*
                                                     EOS_phase1.drho_dP_T(p2_loc + delta_p, T2_loc + delta_T)
                                                     -m2_loc/
                                                     (EOS_phase2.rho_value_PT(p2_loc, T2_loc)*
                                                      EOS_phase2.rho_value_PT(p2_loc, T2_loc))*
                                                     EOS_phase2.drho_dP_T(p2_loc, T2_loc);
                                  Jac_update[0][1] = -m1_loc/
                                                     (EOS_phase1.rho_value_PT(p2_loc + delta_p, T2_loc + delta_T)*
                                                      EOS_phase1.rho_value_PT(p2_loc + delta_p, T2_loc + delta_T))*
                                                     EOS_phase1.drho_dT_P(T2_loc + delta_T, p2_loc + delta_T)
                                                     -m2_loc/
                                                     (EOS_phase2.rho_value_PT(p2_loc, T2_loc)*
                                                      EOS_phase2.rho_value_PT(p2_loc, T2_loc))*
                                                     EOS_phase2.drho_dT_P(T2[cell], p2[cell]);
                                  Jac_update[1][0] = m1_loc*
                                                     EOS_phase1.de_dP_T(p2_loc + delta_p, T2_loc + delta_T) +
                                                     m2_loc*
                                                     EOS_phase2.de_dP_T(p2_loc, T2_loc);
                                  Jac_update[1][1] = m1_loc*
                                                     EOS_phase1.de_dT_P(T2_loc + delta_T, p2_loc + delta_p) +
                                                     m2_loc*
                                                     EOS_phase2.de_dT_P(T2_loc, p2_loc);

                                  // Apply the Newton method
                                  const auto det_Jac = Jac_update[0][0]*Jac_update[1][1]
                                                     - Jac_update[0][1]*Jac_update[1][0];
                                  if(std::abs(det_Jac) > static_cast<Number>(1e-10)) {
                                    dp2 = (static_cast<Number>(1.0)/det_Jac)*
                                          (Jac_update[1][1]*f1 - Jac_update[0][1]*f2);
                                    dp2 = std::min(dp2, static_cast<Number>(0.9)*(p2_loc + EOS_phase2.get_pi_infty()));
                                    dp2 = std::min(dp2, static_cast<Number>(0.9)*(p2_loc + delta_p + EOS_phase1.get_pi_infty()));
                                    dT2 = (static_cast<Number>(1.0)/det_Jac)*
                                          (-Jac_update[1][0]*f1 + Jac_update[0][0]*f2);
                                    dT2 = std::min(dT2, static_cast<Number>(0.9)*T2_loc);
                                    dT2 = std::min(dT2, static_cast<Number>(0.9)*(T2_loc + delta_T));
                                    p2_loc -= dp2;
                                    T2_loc -= dT2;
                                  }
                                  else {
                                    int rank;
                                    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                                    if(rank == 0) {
                                      std::cerr << "Non-invertible Jacobian in the Newton method in the re-update of "
                                                << "conserved variables after relaxation"
                                                << std::endl;
                                    }
                                    exit(1);
                                  }

                                  // Recompute functions for which we look for zero for next iteration
                                  f1 = m1_loc/EOS_phase1.rho_value_PT(p2_loc + delta_p, T2_loc + delta_T)
                                     + m2_loc/EOS_phase2.rho_value_PT(p2_loc, T2_loc)
                                     - static_cast<Number>(1.0);
                                  f2 = m1_loc*EOS_phase1.e_value_PT(p2_loc + delta_p, T2_loc + delta_T)
                                     + m2_loc*EOS_phase2.e_value_PT(p2_loc, T2_loc)
                                     - rhoe_0;
                                }
                                else {
                                  break;
                                }
                              }
                              if(iter == max_Newton_iters &&
                                 (std::abs(dp2) > atol_Newton_relaxation + rtol_Newton_relaxation*std::abs(p2[cell]) ||
                                  std::abs(dT2) > atol_Newton_relaxation + rtol_Newton_relaxation*std::abs(T2[cell])) &&
                                 (std::abs(f1) > atol_Newton_relaxation ||
                                  std::abs(f2) > atol_Newton_relaxation + rtol_Newton_relaxation*std::abs(rhoe_0))) {
                                int rank;
                                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                                if(rank == 0) {
                                  std::cerr << "Newton method not converged in the re-update of "
                                            << "conserved variables after relaxation"
                                            << std::endl;
                                }
                                exit(1);
                              }

                              // Once pressure and temperature have been update, finalize the update
                              p1_loc = p2_loc + delta_p;
                              T1_loc = T2_loc + delta_T;

                              e1_loc = EOS_phase1.e_value_PT(p1_loc, T1_loc);
                              conserved_variables[cell][Indices::ALPHA1_RHO1_E1_INDEX] = m1_loc*(e1_loc + static_cast<Number>(0.5)*norm2_vel1);

                              rho1_loc = EOS_phase1.rho_value_PT(p1_loc, T1_loc);
                              conserved_variables[cell][Indices::ALPHA1_INDEX] = m1_loc/rho1_loc;

                              conserved_variables[cell][Indices::ALPHA2_RHO2_E2_INDEX] = rhoE_0
                                                                                       - conserved_variables[cell][Indices::ALPHA1_RHO1_E1_INDEX];
                            }
                        );
}

// Finite rate relaxation for pressure and temperature (following Jomée 2023) after instantaneous equilibrium velocity
//
template<std::size_t dim>
void BN_Solver<dim>::perform_relaxation_finite_rate_pT() {
  /*--- Resize fields because of (possible) multiresolution ---*/
  vel1.resize();
  vel2.resize();

  /*--- Loop over all cells ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Pre-fetch several variables that will be used several times so as to exploit (possible) vectorization
                              // as well as to enhance readability
                              auto alpha1_loc   = conserved_variables[cell][Indices::ALPHA1_INDEX];
                              const auto m1_loc = conserved_variables[cell][Indices::ALPHA1_RHO1_INDEX];
                              auto m1E1_loc     = conserved_variables[cell][Indices::ALPHA1_RHO1_E1_INDEX];
                              const auto m2_loc = conserved_variables[cell][Indices::ALPHA2_RHO2_INDEX];
                              auto m2E2_loc     = conserved_variables[cell][Indices::ALPHA2_RHO2_E2_INDEX];

                              // Instantaneous velocity update
                              /*--- Save phasic velocities and initial specific internal energy of phase 1 for the total energy update ---*/
                              const auto inv_m1_loc = static_cast<Number>(1.0)/m1_loc;
                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              const auto inv_m2_loc = static_cast<Number>(1.0)/m2_loc;
                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              auto e1_0             = m1E1_loc*inv_m1_loc;
                              for(std::size_t d = 0; d < dim; ++d) {
                                vel1[cell][d] = conserved_variables[cell][Indices::ALPHA1_RHO1_U1_INDEX + d]*inv_m1_loc;
                                                /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                vel2[cell][d] = conserved_variables[cell][Indices::ALPHA2_RHO2_U2_INDEX + d]*inv_m2_loc;
                                                /*--- TODO: Add treatment for vanishing volume fraction ---*/

                                e1_0 -= static_cast<Number>(0.5)*vel1[cell][d]*vel1[cell][d];
                              }

                              /*--- Compute constant quantities
                                    (mixture density, (specific) total energy, mass fraction, 'mixture' velocity) for the updates ---*/
                              const auto rho_0     = m1_loc + m2_loc;
                              const auto inv_rho_0 = static_cast<Number>(1.0)/rho_0;
                              const auto Y1_0      = m1_loc*inv_rho_0;
                              const auto Y2_0      = static_cast<Number>(1.0) - Y1_0;
                              const auto rhoE_0    = m1E1_loc + m2E2_loc;
                              auto norm2_vel       = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                norm2_vel += (Y1_0*vel1[cell][d] + Y2_0*vel2[cell][d])*
                                             (Y1_0*vel1[cell][d] + Y2_0*vel2[cell][d]);
                              }

                              /*--- Update the momentum (and the kinetic energy of phase 1) ---*/
                              std::array<Number, dim> vel_star;
                              m1E1_loc = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                vel_star[d] = (conserved_variables[cell][Indices::ALPHA1_RHO1_U1_INDEX + d] +
                                               conserved_variables[cell][Indices::ALPHA2_RHO2_U2_INDEX + d])*inv_rho_0;

                                conserved_variables[cell][Indices::ALPHA1_RHO1_U1_INDEX + d] = m1_loc*vel_star[d];

                                conserved_variables[cell][Indices::ALPHA2_RHO2_U2_INDEX + d] = m2_loc*vel_star[d];

                                m1E1_loc += static_cast<Number>(0.5)*
                                            m1_loc*vel_star[d]*vel_star[d];
                              }

                              /*--- Update total energy of the two phases ---*/
                              const auto chi1 = static_cast<Number>(0.0); // uI = (1 - chi1)*u1 + chi1*u2;
                              auto e1_star    = e1_0;
                              for(std::size_t d = 0; d < dim; ++d) {
                                e1_star += static_cast<Number>(0.5)*chi1*
                                           (vel1[cell][d] - vel2[cell][d])*(vel1[cell][d] - vel2[cell][d])*Y2_0;
                                           // Recall that vel1 and vel2 are the initial values!!!!
                              }
                              m1E1_loc += m1_loc*e1_star;

                              m2E2_loc = rhoE_0 - m1E1_loc;

                              /*--- Update the velocity of the two phases ---*/
                              std::array<Number, dim> delta_u;
                              for(std::size_t d = 0; d < dim; ++d) {
                                delta_u[0]    = static_cast<Number>(0.0);
                                vel1[cell][d] = vel_star[d];
                                vel2[cell][d] = vel_star[d];
                              }

                              // Solve the system for delta_p and delta_T
                              /*--- Compute the auxiliary fields to initalize delta_p and delta_T ---*/
                              auto rho1_loc = m1_loc/alpha1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              e1_0          = e1_star; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              auto p1_loc   = EOS_phase1.pres_value_Rhoe(rho1_loc, e1_0);

                              auto rho2_loc = m2_loc/(static_cast<Number>(1.0) - alpha1_loc);
                                              /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              auto e2_0     = m2E2_loc/m2_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              for(std::size_t d = 0; d < dim; ++d) {
                                e2_0 -= static_cast<Number>(0.5)*vel2[cell][d]*vel2[cell][d];
                              }
                              auto p2_loc = EOS_phase2.pres_value_Rhoe(rho2_loc, e2_0);

                              /*--- Compute matrix relaxation coefficients ---*/
                              compute_coefficients_source_relaxation(conserved_variables[cell],
                                                                     delta_u, A_relax, S_relax);

                              /*--- Solve the linear system ---*/
                              Number delta_p,
                                     delta_T;
                              const auto delta_p0 = p1_loc - p2_loc;
                              auto T1_loc         = EOS_phase1.T_value_RhoP(rho1_loc, p1_loc);
                              auto T2_loc         = EOS_phase2.T_value_RhoP(rho2_loc, p2_loc);
                              const auto delta_T0 = T1_loc - T2_loc;
                              const auto det_A_pT = A_relax[0][0]*A_relax[1][1]
                                                  - A_relax[0][1]*A_relax[1][0];
                              if(std::abs(det_A_pT) > static_cast<Number>(1e-10)) {
                                delta_p = (static_cast<Number>(1.0)/det_A_pT)*
                                          (A_relax[1][1]*(delta_p0 + dt*S_relax[0]) -
                                           A_relax[0][1]*(delta_T0 + dt*S_relax[1]));
                                delta_T = (static_cast<Number>(1.0)/det_A_pT)*
                                          (-A_relax[1][0]*(delta_p0 + dt*S_relax[0]) +
                                            A_relax[0][0]*(delta_T0 + dt*S_relax[1]));
                              }
                              else {
                                int rank;
                                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                                if(rank == 0) {
                                  std::cerr << "Singular matrix in the relaxation" << std::endl;
                                }
                                exit(1);
                              }

                              // Re-update conserved variables. Newton method loop
                              const auto rhoe_0 = rhoE_0
                                                - static_cast<Number>(0.5)*rho_0*norm2_vel;
                              auto dp2 = std::numeric_limits<Number>::max();
                              auto dT2 = std::numeric_limits<Number>::max();
                              unsigned iter;
                              auto f1 = m1_loc/EOS_phase1.rho_value_PT(p2_loc + delta_p, T2_loc + delta_T)
                                      + m2_loc/EOS_phase2.rho_value_PT(p2_loc, T2_loc)
                                      - static_cast<Number>(1.0);
                              auto f2 = m1_loc*EOS_phase1.e_value_PT(p2_loc + delta_p, T2_loc + delta_T)
                                      + m2_loc*EOS_phase2.e_value_PT(p2_loc, T2_loc)
                                      - rhoe_0;
                              for(iter = 0; iter < max_Newton_iters; ++iter) {
                                if((std::abs(dp2) > atol_Newton_relaxation + rtol_Newton_relaxation*std::abs(p2[cell]) ||
                                    std::abs(dT2) > atol_Newton_relaxation + rtol_Newton_relaxation*std::abs(T2[cell])) &&
                                   (std::abs(f1) > atol_Newton_relaxation ||
                                    std::abs(f2) > atol_Newton_relaxation + rtol_Newton_relaxation*std::abs(rhoe_0))) {
                                  // Compute Jacobian matrix
                                  Jac_update[0][0] = -m1_loc/
                                                     (EOS_phase1.rho_value_PT(p2_loc + delta_p, T2_loc + delta_T)*
                                                      EOS_phase1.rho_value_PT(p2_loc + delta_p, T2_loc + delta_T))*
                                                     EOS_phase1.drho_dP_T(p2_loc + delta_p, T2_loc + delta_T)
                                                     -m2_loc/
                                                     (EOS_phase2.rho_value_PT(p2_loc, T2_loc)*
                                                      EOS_phase2.rho_value_PT(p2_loc, T2_loc))*
                                                     EOS_phase2.drho_dP_T(p2_loc, T2_loc);
                                  Jac_update[0][1] = -m1_loc/
                                                     (EOS_phase1.rho_value_PT(p2_loc + delta_p, T2_loc + delta_T)*
                                                      EOS_phase1.rho_value_PT(p2_loc + delta_p, T2_loc + delta_T))*
                                                     EOS_phase1.drho_dT_P(T2_loc + delta_T, p2_loc + delta_T)
                                                     -m2_loc/
                                                     (EOS_phase2.rho_value_PT(p2_loc, T2_loc)*
                                                      EOS_phase2.rho_value_PT(p2_loc, T2_loc))*
                                                     EOS_phase2.drho_dT_P(T2[cell], p2[cell]);
                                  Jac_update[1][0] = m1_loc*
                                                     EOS_phase1.de_dP_T(p2_loc + delta_p, T2_loc + delta_T) +
                                                     m2_loc*
                                                     EOS_phase2.de_dP_T(p2_loc, T2_loc);
                                  Jac_update[1][1] = m1_loc*
                                                     EOS_phase1.de_dT_P(T2_loc + delta_T, p2_loc + delta_p) +
                                                     m2_loc*
                                                     EOS_phase2.de_dT_P(T2_loc, p2_loc);

                                  // Apply the Newton method
                                  const auto det_Jac = Jac_update[0][0]*Jac_update[1][1]
                                                     - Jac_update[0][1]*Jac_update[1][0];
                                  if(std::abs(det_Jac) > static_cast<Number>(1e-10)) {
                                    dp2 = (static_cast<Number>(1.0)/det_Jac)*
                                          (Jac_update[1][1]*f1 - Jac_update[0][1]*f2);
                                    dp2 = std::min(dp2, static_cast<Number>(0.9)*(p2_loc + EOS_phase2.get_pi_infty()));
                                    dp2 = std::min(dp2, static_cast<Number>(0.9)*(p2_loc + delta_p + EOS_phase1.get_pi_infty()));
                                    dT2 = (static_cast<Number>(1.0)/det_Jac)*
                                          (-Jac_update[1][0]*f1 + Jac_update[0][0]*f2);
                                    dT2 = std::min(dT2, static_cast<Number>(0.9)*T2_loc);
                                    dT2 = std::min(dT2, static_cast<Number>(0.9)*(T2_loc + delta_T));
                                    p2_loc -= dp2;
                                    T2_loc -= dT2;
                                  }
                                  else {
                                    int rank;
                                    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                                    if(rank == 0) {
                                      std::cerr << "Non-invertible Jacobian in the Newton method in the re-update of "
                                                << "conserved variables after relaxation"
                                                << std::endl;
                                    }
                                    exit(1);
                                  }

                                  // Recompute functions for which we look for zero for next iteration
                                  f1 = m1_loc/EOS_phase1.rho_value_PT(p2_loc + delta_p, T2_loc + delta_T)
                                     + m2_loc/EOS_phase2.rho_value_PT(p2_loc, T2_loc)
                                     - static_cast<Number>(1.0);
                                  f2 = m1_loc*EOS_phase1.e_value_PT(p2_loc + delta_p, T2_loc + delta_T)
                                     + m2_loc*EOS_phase2.e_value_PT(p2_loc, T2_loc)
                                     - rhoe_0;
                                }
                                else {
                                  break;
                                }
                              }
                              if(iter == max_Newton_iters &&
                                 (std::abs(dp2) > atol_Newton_relaxation + rtol_Newton_relaxation*std::abs(p2[cell]) ||
                                  std::abs(dT2) > atol_Newton_relaxation + rtol_Newton_relaxation*std::abs(T2[cell])) &&
                                 (std::abs(f1) > atol_Newton_relaxation ||
                                  std::abs(f2) > atol_Newton_relaxation + rtol_Newton_relaxation*std::abs(rhoe_0))) {
                                int rank;
                                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                                if(rank == 0) {
                                  std::cerr << "Newton method not converged in the re-update of "
                                            << "conserved variables after relaxation"
                                            << std::endl;
                                }
                                exit(1);
                              }

                              // Once pressure and temperature have been update, finalize the update
                              p1_loc = p2_loc + delta_p;
                              T1_loc = T2_loc + delta_T;

                              const auto e1_loc = EOS_phase1.e_value_PT(p1_loc, T1_loc);
                              conserved_variables[cell][Indices::ALPHA1_RHO1_E1_INDEX] = m1_loc*(e1_loc + static_cast<Number>(0.5)*norm2_vel);

                              rho1_loc = EOS_phase1.rho_value_PT(p1_loc, T1_loc);
                              conserved_variables[cell][Indices::ALPHA1_INDEX] = m1_loc/rho1_loc;

                              conserved_variables[cell][Indices::ALPHA2_RHO2_E2_INDEX] = rhoE_0
                                                                                       - conserved_variables[cell][Indices::ALPHA1_RHO1_E1_INDEX];
                            }
                        );
}

// Apply the instantaneous relaxation (velocity)
//
template<std::size_t dim>
void BN_Solver<dim>::perform_instantaneous_velocity_relaxation() {
  /*--- Resize fields because of multiresolution ---*/
  vel1.resize();
  vel2.resize();

  /*--- Loop over all cells ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Pre-fetch several variables that will be used several times so as to exploit (possible) vectorization
                              // as well as to enhance readability
                              const auto m1_loc = conserved_variables[cell][Indices::ALPHA1_RHO1_INDEX];
                              auto m1E1_loc     = conserved_variables[cell][Indices::ALPHA1_RHO1_E1_INDEX];
                              const auto m2_loc = conserved_variables[cell][Indices::ALPHA2_RHO2_INDEX];
                              auto m2E2_loc     = conserved_variables[cell][Indices::ALPHA2_RHO2_E2_INDEX];

                              // Save phasic velocities and initial specific internal energy of phase 1 for the total energy update ---*/
                              const auto inv_m1_loc = static_cast<Number>(1.0)/m1_loc;
                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              const auto inv_m2_loc = static_cast<Number>(1.0)/m2_loc;
                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              auto e1_0             = m1E1_loc*inv_m1_loc;
                              for(std::size_t d = 0; d < dim; ++d) {
                                vel1[cell][d] = conserved_variables[cell][Indices::ALPHA1_RHO1_U1_INDEX + d]*inv_m1_loc;
                                                /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                vel2[cell][d] = conserved_variables[cell][Indices::ALPHA2_RHO2_U2_INDEX + d]*inv_m2_loc;
                                                /*--- TODO: Add treatment for vanishing volume fraction ---*/

                                e1_0 -= static_cast<Number>(0.5)*vel1[cell][d]*vel1[cell][d];
                              }

                              // Compute mixture density and (specific) total energy for the updates
                              const auto rho_0     = m1_loc + m2_loc;
                              const auto inv_rho_0 = static_cast<Number>(1.0)/rho_0;
                              const auto rhoE_0    = m1E1_loc + m2E2_loc;

                              // Update the momentum (and the kinetic energy of phase 1)
                              m1E1_loc = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                const auto vel_star_d = (conserved_variables[cell][Indices::ALPHA1_RHO1_U1_INDEX + d] +
                                                         conserved_variables[cell][Indices::ALPHA2_RHO2_U2_INDEX + d])*inv_rho_0;

                                conserved_variables[cell][Indices::ALPHA1_RHO1_U1_INDEX + d] = m1_loc*vel_star_d;

                                conserved_variables[cell][Indices::ALPHA2_RHO2_U2_INDEX + d] = m2_loc*vel_star_d;

                                m1E1_loc += static_cast<Number>(0.5)*
                                            m1_loc*vel_star_d*vel_star_d;
                              }

                              // Update total energy of the two phases
                              const auto Y2_0 = m2_loc*inv_rho_0;
                              const auto chi1 = static_cast<Number>(0.0); // uI = (1 - chi1)*u1 + chi1*u2;
                              auto e1_star    = e1_0;
                              for(std::size_t d = 0; d < dim; ++d) {
                                e1_star += static_cast<Number>(0.5)*chi1*
                                           (vel1[cell][d] - vel2[cell][d])*(vel1[cell][d] - vel2[cell][d])*Y2_0;
                                           // Recall that vel1 and vel2 are the initial values!!!!
                              }
                              m1E1_loc += m1_loc*e1_star;

                              conserved_variables[cell][Indices::ALPHA1_RHO1_E1_INDEX] = m1E1_loc;
                              conserved_variables[cell][Indices::ALPHA2_RHO2_E2_INDEX] = rhoE_0 - m1E1_loc;
                            }
                        );
}

// Apply the instantaneous relaxation (velocity and pressure)
//
template<std::size_t dim>
void BN_Solver<dim>::perform_instantaneous_velocity_pressure_relaxation() {
  /*--- Resize fields because of (possible) multiresolution ---*/
  vel1.resize();
  vel2.resize();

  /*--- Loop over all cells ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Pre-fetch several variables that will be used several times so as to exploit (possible) vectorization
                              // as well as to enhance readability
                              auto alpha1_loc   = conserved_variables[cell][Indices::ALPHA1_INDEX];
                              const auto m1_loc = conserved_variables[cell][Indices::ALPHA1_RHO1_INDEX];
                              auto m1E1_loc     = conserved_variables[cell][Indices::ALPHA1_RHO1_E1_INDEX];
                              const auto m2_loc = conserved_variables[cell][Indices::ALPHA2_RHO2_INDEX];
                              auto m2E2_loc     = conserved_variables[cell][Indices::ALPHA2_RHO2_E2_INDEX];

                              // First focus on the velocity relaxation
                              /*--- Save phasic velocities and initial specific internal energy of phase 1 for the total energy update ---*/
                              const auto inv_m1_loc = static_cast<Number>(1.0)/m1_loc;
                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              const auto inv_m2_loc = static_cast<Number>(1.0)/m2_loc;
                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              auto e1_0             = m1E1_loc*inv_m1_loc;
                              for(std::size_t d = 0; d < dim; ++d) {
                                vel1[cell][d] = conserved_variables[cell][Indices::ALPHA1_RHO1_U1_INDEX + d]*inv_m1_loc;
                                                /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                vel2[cell][d] = conserved_variables[cell][Indices::ALPHA2_RHO2_U2_INDEX + d]*inv_m2_loc;
                                                /*--- TODO: Add treatment for vanishing volume fraction ---*/

                                e1_0 -= static_cast<Number>(0.5)*vel1[cell][d]*vel1[cell][d];
                              }

                              /*--- Compute mixture density and (specific) total energy for the updates ---*/
                              const auto rho_0     = m1_loc + m2_loc;
                              const auto inv_rho_0 = static_cast<Number>(1.0)/rho_0;
                              const auto rhoE_0    = m1E1_loc + m2E2_loc;

                              /*--- Update the momentum (and the kinetic energy of phase 1) ---*/
                              m1E1_loc            = static_cast<Number>(0.0);
                              auto norm2_vel_star = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                const auto vel_star_d = (conserved_variables[cell][Indices::ALPHA1_RHO1_U1_INDEX + d] +
                                                         conserved_variables[cell][Indices::ALPHA2_RHO2_U2_INDEX + d])*inv_rho_0;
                                norm2_vel_star += vel_star_d*vel_star_d;

                                conserved_variables[cell][Indices::ALPHA1_RHO1_U1_INDEX + d] = m1_loc*vel_star_d;

                                conserved_variables[cell][Indices::ALPHA2_RHO2_U2_INDEX + d] = m2_loc*vel_star_d;

                                m1E1_loc += static_cast<Number>(0.5)*
                                            m1_loc*vel_star_d*vel_star_d;
                              }

                              /*--- Update total energy of the two phases ---*/
                              const auto Y2_0 = m2_loc*inv_rho_0;
                              const auto chi1 = static_cast<Number>(0.0); // uI = (1 - chi1)*u1 + chi1*u2;
                              auto e1_star    = e1_0;
                              for(std::size_t d = 0; d < dim; ++d) {
                                e1_star += static_cast<Number>(0.5)*chi1*
                                           (vel1[cell][d] - vel2[cell][d])*(vel1[cell][d] - vel2[cell][d])*Y2_0;
                                           // Recall that vel1 and vel2 are the initial values!!!!
                              }
                              m1E1_loc += m1_loc*e1_star;

                              m2E2_loc = rhoE_0 - m1E1_loc;

                              // Focus now on the pressure relaxation
                              /*--- Compute the initial fields for the pressure relaxation ---*/
                              e1_0            = e1_star;
                              const auto e2_0 = m2E2_loc*inv_m2_loc /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                              - static_cast<Number>(0.5)*norm2_vel_star;

                              auto rho1_loc   = m1_loc/alpha1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              auto p1_loc     = EOS_phase1.pres_value_Rhoe(rho1_loc, e1_0);

                              auto alpha2_loc = static_cast<Number>(1.0) - alpha1_loc;
                              auto rho2_loc   = m2_loc/alpha2_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              auto p2_loc     = EOS_phase2.pres_value_Rhoe(rho2_loc, e2_0);

                              /*--- Compute the pressure equilibrium with the linearization method (Pelanti) ---*/
                              const auto a    = static_cast<Number>(1.0)
                                              + EOS_phase2.get_gamma()*alpha1_loc
                                              + EOS_phase1.get_gamma()*alpha2_loc;
                              const auto pI_0 = p2_loc;
                              const auto C1   = static_cast<Number>(2.0)*EOS_phase1.get_gamma()*EOS_phase1.get_pi_infty()
                                              + (EOS_phase1.get_gamma() - static_cast<Number>(1.0))*pI_0;
                              const auto C2   = static_cast<Number>(2.0)*EOS_phase2.get_gamma()*EOS_phase2.get_pi_infty()
                                              + (EOS_phase2.get_gamma() - static_cast<Number>(1.0))*pI_0;
                              const auto b    = C1*alpha2_loc + C2*alpha1_loc
                                              - (static_cast<Number>(1.0) + EOS_phase2.get_gamma())*alpha1_loc*p1_loc
                                              - (static_cast<Number>(1.0) + EOS_phase1.get_gamma())*alpha2_loc*p2_loc;
                              const auto d    = -(C2*alpha1_loc*p1_loc + C1*alpha2_loc*p2_loc);

                              const auto p_star = (-b + std::sqrt(b*b - static_cast<Number>(4.0)*a*d))/
                                                  (static_cast<Number>(2.0)*a);

                              /*--- Update the volume fraction using the computed pressure ---*/
                              alpha1_loc *= ((EOS_phase1.get_gamma() - static_cast<Number>(1.0))*p_star +
                                             static_cast<Number>(2.0)*p1_loc + C1)/
                                            ((EOS_phase1.get_gamma() + static_cast<Number>(1.0))*p_star + C1);
                              conserved_variables[cell][Indices::ALPHA1_INDEX] = alpha1_loc;

                              /*--- Update the total energy of both phases ---*/
                              const auto E1_loc = EOS_phase1.e_value_RhoP(m1_loc/alpha1_loc, p_star)
                                                + static_cast<Number>(0.5)*norm2_vel_star;
                                                /*--- TODO: Add treatment for vanishing volume fraction ---*/

                              m1E1_loc = m1_loc*E1_loc;
                              conserved_variables[cell][Indices::ALPHA1_RHO1_E1_INDEX] = m1E1_loc;

                              conserved_variables[cell][Indices::ALPHA2_RHO2_E2_INDEX] = rhoE_0 - m1E1_loc;
                            }
                        );
}

// Apply the instantaneous relaxation (only pressure)
//
template<std::size_t dim>
void BN_Solver<dim>::perform_instantaneous_pressure_relaxation() {
  /*--- Loop over all cells ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Pre-fetch several variables that will be used several times so as to exploit (possible) vectorization
                              // as well as to enhance readability
                              auto alpha1_loc   = conserved_variables[cell][Indices::ALPHA1_INDEX];
                              const auto m1_loc = conserved_variables[cell][Indices::ALPHA1_RHO1_INDEX];
                              auto m1E1_loc     = conserved_variables[cell][Indices::ALPHA1_RHO1_E1_INDEX];
                              const auto m2_loc = conserved_variables[cell][Indices::ALPHA2_RHO2_INDEX];
                              auto m2E2_loc     = conserved_variables[cell][Indices::ALPHA2_RHO2_E2_INDEX];

                              // Focus on the pressure relaxation
                              /*--- Compute the initial fields for the pressure relaxation ---*/
                              const auto inv_m1_loc = static_cast<Number>(1.0)/m1_loc;
                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              const auto inv_m2_loc = static_cast<Number>(1.0)/m2_loc;
                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              auto e1_0             = m1E1_loc*inv_m1_loc;
                              auto e2_0             = m2E2_loc*inv_m2_loc;
                              auto norm2_vel1       = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                const auto vel1_d = conserved_variables[cell][Indices::ALPHA1_RHO1_U1_INDEX + d]*inv_m1_loc;
                                                /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                const auto vel2_d = conserved_variables[cell][Indices::ALPHA2_RHO2_U2_INDEX + d]*inv_m2_loc;
                                                /*--- TODO: Add treatment for vanishing volume fraction ---*/

                                e1_0 -= static_cast<Number>(0.5)*vel1_d*vel1_d;
                                e2_0 -= static_cast<Number>(0.5)*vel2_d*vel2_d;

                                norm2_vel1 += vel1_d*vel1_d;
                              }

                              const auto rhoE_0 = m1E1_loc + m2E2_loc;

                              auto rho1_loc   = m1_loc/alpha1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              auto p1_loc     = EOS_phase1.pres_value_Rhoe(rho1_loc, e1_0);

                              auto alpha2_loc = static_cast<Number>(1.0) - alpha1_loc;
                              auto rho2_loc   = m2_loc/alpha2_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              auto p2_loc     = EOS_phase2.pres_value_Rhoe(rho2_loc, e2_0);

                              /*--- Compute the pressure equilibrium with the linearization method (Pelanti) ---*/
                              const auto a    = static_cast<Number>(1.0)
                                              + EOS_phase2.get_gamma()*alpha1_loc
                                              + EOS_phase1.get_gamma()*alpha2_loc;
                              const auto pI_0 = p2_loc;
                              const auto C1   = static_cast<Number>(2.0)*EOS_phase1.get_gamma()*EOS_phase1.get_pi_infty()
                                              + (EOS_phase1.get_gamma() - static_cast<Number>(1.0))*pI_0;
                              const auto C2   = static_cast<Number>(2.0)*EOS_phase2.get_gamma()*EOS_phase2.get_pi_infty()
                                              + (EOS_phase2.get_gamma() - static_cast<Number>(1.0))*pI_0;
                              const auto b    = C1*alpha2_loc + C2*alpha1_loc
                                              - (static_cast<Number>(1.0) + EOS_phase2.get_gamma())*alpha1_loc*p1_loc
                                              - (static_cast<Number>(1.0) + EOS_phase1.get_gamma())*alpha2_loc*p2_loc;
                              const auto d    = -(C2*alpha1_loc*p1_loc + C1*alpha2_loc*p2_loc);

                              const auto p_star = (-b + std::sqrt(b*b - static_cast<Number>(4.0)*a*d))/
                                                  (static_cast<Number>(2.0)*a);

                              /*--- Update the volume fraction using the computed pressure ---*/
                              alpha1_loc *= ((EOS_phase1.get_gamma() - static_cast<Number>(1.0))*p_star +
                                             static_cast<Number>(2.0)*p1_loc + C1)/
                                            ((EOS_phase1.get_gamma() + static_cast<Number>(1.0))*p_star + C1);
                              conserved_variables[cell][Indices::ALPHA1_INDEX] = alpha1_loc;

                              /*--- Update the total energy of both phases ---*/
                              const auto E1_loc = EOS_phase1.e_value_RhoP(m1_loc/alpha1_loc, p_star)
                                                + static_cast<Number>(0.5)*norm2_vel1;
                                                /*--- TODO: Add treatment for vanishing volume fraction ---*/

                              m1E1_loc = m1_loc*E1_loc;
                              conserved_variables[cell][Indices::ALPHA1_RHO1_E1_INDEX] = m1E1_loc;

                              conserved_variables[cell][Indices::ALPHA2_RHO2_E2_INDEX] = rhoE_0 - m1E1_loc;
                            }
                        );
}

// Apply the instantaneous relaxation (velocity and pressure+temperature)
//
template<std::size_t dim>
void BN_Solver<dim>::perform_instantaneous_relaxation() {
  /*--- Resize because of (possible) multiresolution ---*/
  vel1.resize();
  vel2.resize();

  /*--- Loop over all cells ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Pre-fetch several variables that will be used several times so as to exploit (possible) vectorization
                              // as well as to enhance readability
                              auto alpha1_loc   = conserved_variables[cell][Indices::ALPHA1_INDEX];
                              const auto m1_loc = conserved_variables[cell][Indices::ALPHA1_RHO1_INDEX];
                              auto m1E1_loc     = conserved_variables[cell][Indices::ALPHA1_RHO1_E1_INDEX];
                              const auto m2_loc = conserved_variables[cell][Indices::ALPHA2_RHO2_INDEX];
                              auto m2E2_loc     = conserved_variables[cell][Indices::ALPHA2_RHO2_E2_INDEX];

                              // First focus on the velocity relaxation
                              /*--- Save phasic velocities and initial specific internal energy of phase 1 for the total energy update ---*/
                              const auto inv_m1_loc = static_cast<Number>(1.0)/m1_loc;
                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              const auto inv_m2_loc = static_cast<Number>(1.0)/m2_loc;
                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              auto e1_0             = m1E1_loc*inv_m1_loc;
                              for(std::size_t d = 0; d < dim; ++d) {
                                vel1[cell][d] = conserved_variables[cell][Indices::ALPHA1_RHO1_U1_INDEX + d]*inv_m1_loc;
                                                /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                vel2[cell][d] = conserved_variables[cell][Indices::ALPHA2_RHO2_U2_INDEX + d]*inv_m2_loc;
                                                /*--- TODO: Add treatment for vanishing volume fraction ---*/

                                e1_0 -= static_cast<Number>(0.5)*vel1[cell][d]*vel1[cell][d];
                              }

                              /*--- Compute mixture density and (specific) total energy for the updates ---*/
                              const auto rho_0     = m1_loc + m2_loc;
                              const auto inv_rho_0 = static_cast<Number>(1.0)/rho_0;
                              const auto rhoE_0    = m1E1_loc + m2E2_loc;

                              /*--- Update the momentum (and the kinetic energy of phase 1) ---*/
                              m1E1_loc            = static_cast<Number>(0.0);
                              auto norm2_vel_star = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                const auto vel_star_d = (conserved_variables[cell][Indices::ALPHA1_RHO1_U1_INDEX + d] +
                                                         conserved_variables[cell][Indices::ALPHA2_RHO2_U2_INDEX + d])*inv_rho_0;
                                norm2_vel_star += vel_star_d*vel_star_d;

                                conserved_variables[cell][Indices::ALPHA1_RHO1_U1_INDEX + d] = m1_loc*vel_star_d;

                                conserved_variables[cell][Indices::ALPHA2_RHO2_U2_INDEX + d] = m2_loc*vel_star_d;

                                m1E1_loc += static_cast<Number>(0.5)*
                                            m1_loc*vel_star_d*vel_star_d;
                              }

                              /*--- Update total energy of the two phases ---*/
                              const auto Y2_0 = m2_loc*inv_rho_0;
                              const auto chi1 = static_cast<Number>(0.0); // uI = (1 - chi1)*u1 + chi1*u2;
                              auto e1_star    = e1_0;
                              for(std::size_t d = 0; d < dim; ++d) {
                                e1_star += static_cast<Number>(0.5)*chi1*
                                           (vel1[cell][d] - vel2[cell][d])*(vel1[cell][d] - vel2[cell][d])*Y2_0;
                                           // Recall that vel1 and vel2 are the initial values!!!!
                              }
                              m1E1_loc += m1_loc*e1_star;

                              m2E2_loc = rhoE_0 - m1E1_loc;

                              // Focus now on the pressure/temperature relaxation
                              const auto rhoe_0 = rhoE_0
                                                - static_cast<Number>(0.5)*rho_0*norm2_vel_star;

                              const auto a = EOS_phase1.get_cv()*m1_loc
                                           + EOS_phase2.get_cv()*m2_loc;
                              const auto b = EOS_phase1.get_q_infty()*EOS_phase1.get_cv()*
                                             (EOS_phase1.get_gamma() - static_cast<Number>(1.0))*
                                             m1_loc*m1_loc
                                           + EOS_phase2.get_q_infty()*EOS_phase2.get_cv()*
                                             (EOS_phase2.get_gamma() - static_cast<Number>(1.0))*
                                             m2_loc*m2_loc
                                           + m1_loc*
                                             EOS_phase1.get_cv()*(EOS_phase1.get_gamma()*EOS_phase1.get_pi_infty() + EOS_phase2.get_pi_infty())
                                           + m2_loc*
                                             EOS_phase2.get_cv()*(EOS_phase2.get_gamma()*EOS_phase2.get_pi_infty() + EOS_phase1.get_pi_infty())
                                           + m1_loc*m2_loc*
                                             (EOS_phase1.get_q_infty()*EOS_phase2.get_cv()*
                                              (EOS_phase2.get_gamma() - static_cast<Number>(1.0)) +
                                              EOS_phase2.get_q_infty()*EOS_phase1.get_cv()*
                                              (EOS_phase1.get_gamma() - static_cast<Number>(1.0)))
                                           - rhoe_0*
                                             (EOS_phase1.get_cv()*(EOS_phase1.get_gamma() - static_cast<Number>(1.0))*m1_loc +
                                              EOS_phase2.get_cv()*(EOS_phase2.get_gamma() - static_cast<Number>(1.0))*m2_loc);
                              const auto d = EOS_phase1.get_q_infty()*EOS_phase1.get_cv()*
                                             (EOS_phase1.get_gamma() - static_cast<Number>(1.0))*EOS_phase2.get_pi_infty()*
                                             m1_loc*m1_loc
                                           + EOS_phase2.get_q_infty()*EOS_phase2.get_cv()*
                                             (EOS_phase2.get_gamma() - static_cast<Number>(1.0))*EOS_phase1.get_pi_infty()*
                                             m2_loc*m2_loc
                                           + (m1_loc*EOS_phase1.get_cv()*EOS_phase1.get_gamma() +
                                              m2_loc*EOS_phase2.get_cv()*EOS_phase2.get_gamma())*
                                             EOS_phase1.get_pi_infty()*EOS_phase2.get_pi_infty()
                                           + m1_loc*m2_loc*
                                             (EOS_phase1.get_q_infty()*EOS_phase2.get_cv()*
                                              (EOS_phase2.get_gamma() - static_cast<Number>(1.0))*EOS_phase1.get_pi_infty() +
                                              EOS_phase2.get_q_infty()*EOS_phase1.get_cv()*
                                              (EOS_phase1.get_gamma() - static_cast<Number>(1.0))*EOS_phase2.get_pi_infty())
                                           - rhoe_0*
                                             (EOS_phase1.get_cv()*(EOS_phase1.get_gamma() - static_cast<Number>(1.0))*
                                              m1_loc*EOS_phase2.get_pi_infty() +
                                              EOS_phase2.get_cv()*(EOS_phase2.get_gamma() - static_cast<Number>(1.0))*
                                              m2_loc*EOS_phase1.get_pi_infty());

                              const auto p_star = (-b + std::sqrt(b*b - static_cast<Number>(4.0)*a*d))/
                                                  (static_cast<Number>(2.0)*a);

                              /*--- Update the volume fraction using the computed pressure ---*/
                              alpha1_loc = (EOS_phase1.get_cv()*(EOS_phase1.get_gamma() - static_cast<Number>(1.0))*
                                            (p_star + EOS_phase2.get_pi_infty())*m1_loc)/
                                           (EOS_phase1.get_cv()*(EOS_phase1.get_gamma() - static_cast<Number>(1.0))*
                                            (p_star + EOS_phase2.get_pi_infty())*m1_loc +
                                            EOS_phase2.get_cv()*(EOS_phase2.get_gamma() - static_cast<Number>(1.0))*
                                            (p_star + EOS_phase1.get_pi_infty())*m2_loc);
                              conserved_variables[cell][Indices::ALPHA1_INDEX] = alpha1_loc;

                              /*--- Update the total energy of both phases ---*/
                              const auto E1_loc = EOS_phase1.e_value_RhoP(m1_loc/alpha1_loc, p_star)
                                                + static_cast<Number>(0.5)*norm2_vel_star;
                                                /*--- TODO: Add treatment for vanishing volume fraction ---*/

                              m1E1_loc = m1_loc*E1_loc;
                              conserved_variables[cell][Indices::ALPHA1_RHO1_E1_INDEX] = m1E1_loc;

                              conserved_variables[cell][Indices::ALPHA2_RHO2_E2_INDEX] = rhoE_0 - m1E1_loc;
                            }
                        );
}

// Keep track of entropy and entropy production after convective step
//
template<std::size_t dim>
void BN_Solver<dim>::compute_entropy_after_flux() {
  /*--- Resize because of (possible) multiresolution ---*/
  entropy_after_flux_phase1.resize();
  entropy_production_flux_phase1.resize();

  entropy_after_flux_phase2.resize();
  entropy_production_flux_phase2.resize();

  /*--- Loop over all cells ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Pre-fetch variables used several times in order to exploit (possible) vectorization
                              // as well as to enhance readability
                              const auto alpha1_loc = conserved_variables[cell][Indices::ALPHA1_INDEX];
                              const auto m1_loc     = conserved_variables[cell][Indices::ALPHA1_RHO1_INDEX];
                              const auto m1E1_loc   = conserved_variables[cell][Indices::ALPHA1_RHO1_E1_INDEX];
                              const auto m2_loc     = conserved_variables[cell][Indices::ALPHA2_RHO2_INDEX];
                              const auto m2E2_loc   = conserved_variables[cell][Indices::ALPHA2_RHO2_E2_INDEX];

                              // Compute entropies
                              const auto rho1_loc   = m1_loc/alpha1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              const auto inv_m1_loc = static_cast<Number>(1.0)/m1_loc;
                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              auto e1_loc           = m1E1_loc*inv_m1_loc;
                              for(std::size_t d = 0; d < dim; ++d) {
                                const auto vel1_d = conserved_variables[cell][Indices::ALPHA1_RHO1_U1_INDEX + d]*inv_m1_loc;
                                                    /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                e1_loc -= static_cast<Number>(0.5)*vel1_d*vel1_d;
                              }
                              entropy_after_flux_phase1[cell]      = EOS_phase1.s_value_Rhoe(rho1_loc, e1_loc);
                              entropy_production_flux_phase1[cell] = entropy_after_flux_phase1[cell] - entropy_after_relaxation_phase1[cell];
                              /*--- TODO: To be corrected taking into account entropy flux ---*/

                              const auto rho2_loc   = m2_loc/(static_cast<Number>(1.0) - alpha1_loc);
                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              const auto inv_m2_loc = static_cast<Number>(1.0)/m2_loc;
                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              auto e2_loc           = m2E2_loc*inv_m2_loc;
                              for(std::size_t d = 0; d < dim; ++d) {
                                const auto vel2_d = conserved_variables[cell][Indices::ALPHA2_RHO2_U2_INDEX + d]*inv_m2_loc;
                                                    /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                e2_loc -= static_cast<Number>(0.5)*vel2_d*vel2_d;
                              }
                              entropy_after_flux_phase2[cell]      = EOS_phase2.s_value_Rhoe(rho2_loc, e2_loc);
                              entropy_production_flux_phase2[cell] = entropy_after_flux_phase2[cell] - entropy_after_relaxation_phase2[cell];
                              /*--- TODO: To be corrected taking into account entropy flux ---*/

                              /*if(entropy_production_flux_phase1[cell] < -1e-5) {
                                std::cerr << "Suspicious sign of entropy production phase 1 after convective step " << cell << std::endl;
                                exit(1);
                              }
                              if(entropy_production_flux_phase2[cell] < -1e-5) {
                                std::cerr << "Suspicious sign of entropy production phase 2 after convective step " << cell << std::endl;
                                exit(1);
                              }*/
                            }
                        );
}

// Keep track of entropy and entropy production after relaxation
//
template<std::size_t dim>
void BN_Solver<dim>::compute_entropy_after_relaxation() {
  /*--- Resize because of (possible) multiresolution ---*/
  entropy_after_relaxation_phase1.resize();
  entropy_production_relaxation_phase1.resize();

  entropy_after_relaxation_phase2.resize();
  entropy_production_relaxation_phase2.resize();

  /*--- Loop over all cells ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Pre-fetch variables used several times in order to exploit (possible) vectorization
                              // as well as to enhance readability
                              const auto alpha1_loc = conserved_variables[cell][Indices::ALPHA1_INDEX];
                              const auto m1_loc     = conserved_variables[cell][Indices::ALPHA1_RHO1_INDEX];
                              const auto m1E1_loc   = conserved_variables[cell][Indices::ALPHA1_RHO1_E1_INDEX];
                              const auto m2_loc     = conserved_variables[cell][Indices::ALPHA2_RHO2_INDEX];
                              const auto m2E2_loc   = conserved_variables[cell][Indices::ALPHA2_RHO2_E2_INDEX];

                              // Compute entropies
                              const auto rho1_loc   = m1_loc/alpha1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              const auto inv_m1_loc = static_cast<Number>(1.0)/m1_loc;
                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              auto e1_loc           = m1E1_loc*inv_m1_loc;
                              for(std::size_t d = 0; d < dim; ++d) {
                                const auto vel1_d = conserved_variables[cell][Indices::ALPHA1_RHO1_U1_INDEX + d]*inv_m1_loc;
                                                    /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                e1_loc -= static_cast<Number>(0.5)*vel1_d*vel1_d;
                              }
                              entropy_after_relaxation_phase1[cell]      = EOS_phase1.s_value_Rhoe(rho1_loc, e1_loc);
                              entropy_production_relaxation_phase1[cell] = entropy_after_relaxation_phase1[cell] - entropy_after_flux_phase1[cell];

                              const auto rho2_loc   = m2_loc/(static_cast<Number>(1.0) - alpha1_loc);
                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              const auto inv_m2_loc = static_cast<Number>(1.0)/m2_loc;
                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              auto e2_loc           = m2E2_loc*inv_m2_loc;
                              for(std::size_t d = 0; d < dim; ++d) {
                                const auto vel2_d = conserved_variables[cell][Indices::ALPHA2_RHO2_U2_INDEX + d]*inv_m2_loc;
                                                    /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                e2_loc -= static_cast<Number>(0.5)*vel2_d*vel2_d;
                              }
                              entropy_after_relaxation_phase2[cell]      = EOS_phase2.s_value_Rhoe(rho2_loc, e2_loc);
                              entropy_production_relaxation_phase2[cell] = entropy_after_relaxation_phase2[cell] - entropy_after_flux_phase2[cell];

                              /*if(entropy_production_relaxation_phase1[cell] < -1e-5) {
                                std::cerr << "Suspicious sign of entropy production phase 1 after relaxation " << cell << std::endl;
                                std::cerr << entropy_production_relaxation_phase1[cell] << std::endl;
                                std::cerr << entropy_production_relaxation_phase1[cell] +
                                          entropy_production_relaxation_phase2[cell] << std::endl;
                                exit(1);
                              }
                              if(entropy_production_relaxation_phase2[cell] < -1e-5) {
                                std::cerr << "Suspicious sign of entropy production phase 2 after relaxation " << cell << std::endl;
                                std::cerr << entropy_production_relaxation_phase2[cell] << std::endl;
                                std::cerr << entropy_production_relaxation_phase1[cell] +
                                             entropy_production_relaxation_phase2[cell] << std::endl;
                                exit(1);
                              }*/
                            }
                        );
}

//////////////////////////////////////////////////////////////
/*---- IMPLEMENT THE FUNCTION THAT EFFECTIVELY SOLVES THE PROBLEM ---*/
/////////////////////////////////////////////////////////////

// Implement the function that effectively performs the temporal loop
//
template<std::size_t dim>
void BN_Solver<dim>::run(const unsigned nfiles) {
  /*--- Default output arguemnts ---*/
  fs::path path = fs::current_path();
  #ifdef SULICIU_RELAXATION
    filename = "Relaxation_Suliciu";
  #elifdef RUSANOV_FLUX
    filename = "Rusanov_Flux";
  #endif

  #ifdef ORDER_2
    filename = filename + "_order2";
  #else
    filename = filename + "_order1";
  #endif

  const auto dt_save = Tf/static_cast<Number>(nfiles);

  /*--- Auxiliary variables to save updated fields ---*/
  #ifdef ORDER_2
    auto conserved_variables_tmp = samurai::make_vector_field<Number, Field::n_comp>("conserved_tmp", mesh);
    auto conserved_variables_old = samurai::make_vector_field<Number, Field::n_comp>("conserved_old", mesh);
  #endif
  auto conserved_variables_np1 = samurai::make_vector_field<Number, Field::n_comp>("conserved_np1", mesh);

  /*--- Create the flux variables ---*/
  #ifdef SULICIU_RELAXATION
    auto c = static_cast<Number>(0.0);
    auto Suliciu_flux = numerical_flux.make_flux(c);
  #elifdef RUSANOV_FLUX
    auto Rusanov_flux         = numerical_flux_cons.make_flux();
    auto NonConservative_flux = numerical_flux_non_cons.make_flux();
  #endif

  /*--- Add the gravity contribution ---*/
  using cfg = samurai::LocalCellSchemeConfig<samurai::SchemeType::LinearHomogeneous,
                                             Field::n_comp,
                                             decltype(conserved_variables)>;
  auto gravity = samurai::make_cell_based_scheme<cfg>();
  gravity.coefficients_func() = [](Number)
                                  {
                                    samurai::StencilCoeffs<cfg> coeffs;

                                    coeffs[0].fill(0.0);

                                    coeffs[0](Indices::ALPHA1_RHO1_U1_INDEX, Indices::ALPHA1_RHO1_INDEX) = 9.8;
                                    coeffs[0](Indices::ALPHA2_RHO2_U2_INDEX, Indices::ALPHA2_RHO2_INDEX) = 9.8;

                                    return coeffs;
                                  };

  /*--- Save the initial condition ---*/
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(path, suffix_init, conserved_variables,
                          rho, p, vel,
                          vel1, rho1, p1, c1, T1, Y1,
                          vel2, rho2, p2, c2, T2, alpha2, Y2,
                          delta_pres, delta_temp, delta_vel,
                          entropy_after_flux_phase1, entropy_production_flux_phase1,
                          entropy_after_relaxation_phase1, entropy_production_relaxation_phase1,
                          entropy_after_flux_phase2, entropy_production_flux_phase2,
                          entropy_after_relaxation_phase2, entropy_production_relaxation_phase2);

  /*--- Set mesh size ---*/
  const auto dx = static_cast<Number>(mesh.cell_length(mesh.max_level()));
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  using mesh_id_t = typename decltype(mesh)::mesh_id_t;
  const auto n_elements_per_subdomain = mesh[mesh_id_t::cells].nb_cells();
  unsigned n_elements;
  MPI_Allreduce(&n_elements_per_subdomain, &n_elements, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
  if(rank == 0) {
    std::cout << "Number of initial elements = " <<  n_elements << std::endl;
    std::cout << std::endl;
  }

  /*--- Start the loop ---*/
  std::size_t nsave = 0;
  std::size_t nt    = 0;
  auto t            = static_cast<Number>(t0);
  while(t != Tf) {
    // Apply mesh adaptation
    auto MRadaptation = samurai::make_MRAdapt(conserved_variables);
    auto mra_config   = samurai::mra_config();
    mra_config.epsilon(MR_param);
    mra_config.regularity(MR_regularity);
    MRadaptation(mra_config);

    // Save current state in case of order 2
    #ifdef ORDER_2
      conserved_variables_old.resize();
      conserved_variables_old = conserved_variables;
    #endif

    // Compute time step
    samurai::update_ghost_mr(conserved_variables);
    #ifdef SULICIU_RELAXATION
      c = 0.0;
      auto Relaxation_Flux = Suliciu_flux(conserved_variables);
      dt = std::min(Tf - t, cfl*dx/c);
    #elifdef RUSANOV_FLUX
      auto Cons_Flux    = Rusanov_flux(conserved_variables);
      auto NonCons_Flux = NonConservative_flux(conserved_variables);
      dt = std::min(Tf - t, cfl*dx/get_max_lambda());
    #endif
    t += dt;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) {
      std::cout << fmt::format("Iteration {}: t = {}, dt = {}", ++nt, t, dt) << std::endl;
    }

    // Apply the numerical scheme
    #ifdef SULICIU_RELAXATION
      #ifdef ORDER_2
        conserved_variables_tmp.resize();
        conserved_variables_tmp = conserved_variables - dt*Relaxation_Flux + dt*gravity(conserved_variables);
      #else
        conserved_variables_np1.resize();
        conserved_variables_np1 = conserved_variables - dt*Relaxation_Flux + dt*gravity(conserved_variables);
      #endif
    #elifdef RUSANOV_FLUX
      #ifdef ORDER_2
        conserved_variables_tmp.resize();
        conserved_variables_tmp = conserved_variables - dt*Cons_Flux - dt*NonCons_Flux + dt*gravity(conserved_variables);
      #else
        conserved_variables_np1.resize();
        conserved_variables_np1 = conserved_variables - dt*Cons_Flux - dt*NonCons_Flux + dt*gravity(conserved_variables);
      #endif
    #endif

    #ifdef ORDER_2
      std::swap(conserved_variables.array(), conserved_variables_tmp.array());
    #else
      std::swap(conserved_variables.array(), conserved_variables_np1.array());
    #endif
    compute_entropy_after_flux(); /*--- Check entropy production after convective step ---*/

    // Perform relaxation if desired
    if(apply_relaxation) {
      if(apply_finite_rate_relaxation) {
        if(relax_instantaneous_velocity) {
          perform_relaxation_finite_rate_pT();
        }
        else {
          perform_relaxation_finite_rate();
        }
      }
      else {
        if(relax_velocity) {
          if(relax_pressure) {
            if(relax_temperature) {
              perform_instantaneous_relaxation();
            }
            else {
              perform_instantaneous_velocity_pressure_relaxation();
            }
          }
          else {
            perform_instantaneous_velocity_relaxation();
          }
        }
        else {
          perform_instantaneous_pressure_relaxation();
        }
      }
    }
    compute_entropy_after_relaxation(); /*--- Check entropy production after (possible) relaxation ---*/

    /*--- Consider the second stage for the second order ---*/
    #ifdef ORDER_2
      samurai::update_ghost_mr(conserved_variables);
      #ifdef SULICIU_RELAXATION
        c = 0.0;
        Relaxation_Flux = Suliciu_flux(conserved_variables);

        conserved_variables_tmp = conserved_variables - dt*Relaxation_Flux + dt*gravity(conserved_variables);
      #elifdef RUSANOV_FLUX
        Cons_Flux    = Rusanov_flux(conserved_variables);
        NonCons_Flux = NonConservative_flux(conserved_variables);

        conserved_variables_tmp = conserved_variables - dt*Cons_Flux - dt*NonCons_Flux + dt*gravity(conserved_variables);
      #endif
      std::swap(conserved_variables.array(), conserved_variables_tmp.array());
      compute_entropy_after_flux(); /*--- Check entropy production after convective step ---*/

      // Perform relaxation if desired
      if(apply_relaxation) {
        if(apply_finite_rate_relaxation) {
          perform_relaxation_finite_rate();
        }
        else {
          if(relax_pressure) {
            if(relax_temperature) {
              perform_instantaneous_relaxation();
            }
            else {
              perform_instantaneous_pressure_relaxation();
            }
          }
          else {
            perform_instantaneous_velocity_relaxation();
          }
        }
      }
      compute_entropy_after_relaxation(); /*--- Check entropy production after (possible) relaxation ---*/

      // Complete the evaluation
      conserved_variables_np1.resize();
      conserved_variables_np1 = static_cast<Number>(0.5)*
                                (conserved_variables_tmp + conserved_variables_old);
      std::swap(conserved_variables.array(), conserved_variables_np1.array());
    #endif

    // Save the results
    update_auxiliary_fields();
    if(t >= static_cast<Number>(nsave + 1)*dt_save || t == Tf) {
      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";
      save(path, suffix, conserved_variables,
                         rho, p, vel,
                         vel1, rho1, p1, c1, T1, Y1,
                         vel2, rho2, p2, c2, T2, alpha2, Y2,
                         delta_pres, delta_temp, delta_vel,
                         entropy_after_flux_phase1, entropy_production_flux_phase1,
                         entropy_after_relaxation_phase1, entropy_production_relaxation_phase1,
                         entropy_after_flux_phase2, entropy_production_flux_phase2,
                         entropy_after_relaxation_phase2, entropy_production_relaxation_phase2);

      /*--- Save fields in a output file ---*/
      output_data.open("output_data.dat", std::ofstream::out);
      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                             {
                               output_data << std::setprecision(10)
                                           << std::setw(20) << std::left << cell.center()[0]
                                           << std::setw(20) << std::left << conserved_variables[cell][Indices::ALPHA1_INDEX]
                                           << std::setw(20) << std::left << rho1[cell]
                                           << std::setw(20) << std::left << vel1[cell][0]
                                           << std::setw(20) << std::left << p1[cell]
                                           << std::setw(20) << std::left << rho2[cell]
                                           << std::setw(20) << std::left << vel2[cell][0]
                                           << std::setw(20) << std::left << p2[cell]
                                           << std::endl;
                             });
      output_data.close();
    }
  }
}
