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

/*--- Include the headers with the numerical fluxes ---*/
#include "HLLC_conservative_6eqs_internal_energy_flux.hpp"
#include "non_conservative_6eqs_internal_energy_flux.hpp"

// Specify the use of this namespace where we just store the indices
// and some parameters related to the equations of state
using namespace EquationData;

#define assertm(exp, msg) assert(((void)msg, exp))

/** This is the class for the simulation of a 6-equation model with internal energy formulation
 */
template<std::size_t dim>
class Relaxation {
public:
  using Config = samurai::MRConfig<dim, 2, 2, 0>;

  Relaxation() = default; /*--- Default constructor. This will do nothing
                                and basically will never be used ---*/

  Relaxation(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
             const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
             const Simulation_Parameters<double>& sim_param,
             const EOS_Parameters<double>& eos_param,
             const Riemann_Parameters<double>& Riemann_param); /*--- Class constructor with the arguments related
                                                                     to the grid, to the physics, and to the relaxation ---*/

  void run(); /*--- Function which actually executes the temporal loop ---*/

  template<class... Variables>
  void save(const fs::path& path,
            const std::string& suffix,
            const Variables&... fields); /*--- Routine to save the results ---*/

private:
  /*--- Now we declare some relevant variables ---*/
  const samurai::Box<double, dim> box;

  samurai::MRMesh<Config> mesh; /*--- Variable to store the mesh ---*/

  using Field        = samurai::VectorField<decltype(mesh), double, EquationData::NVARS, false>;
  using Number       = typename Field::value_type; /*--- Define the shortcut for the arithmetic type ---*/
  using Field_Scalar = samurai::ScalarField<decltype(mesh), Number>;
  using Field_Vect   = samurai::VectorField<decltype(mesh), Number, dim, false>;

  const Number t0; /*--- Initial time of the simulation ---*/
  const Number Tf; /*--- Final time of the simulation ---*/

  bool apply_pressure_relax; /*--- Set whether to apply or not the pressure relaxation ---*/

  Number cfl; /*--- Courant number of the simulation so as to compute the time step ---*/

  bool   apply_finite_rate_relax; /*--- Set whether to perform a finite rate relaxation or an infinite rate ---*/
  Number tau_p;                   /*--- Finite rate parameter ---*/
  Number dt;                      /*--- Time step (to be declared here because of finite rate) ---*/

  const SG_EOS<Number> EOS_phase1; /*--- Equation of state of phase 1 ---*/
  const SG_EOS<Number> EOS_phase2; /*--- Equation of state of phase 2 ---*/

  samurai::HLLCFlux_Conservative<Field> numerical_flux_cons; /*--- variable to compute the numerical flux for the conservative part
                                                                   (this is necessary to call 'make_flux') ---*/
  samurai::NonConservativeFlux<Field> numerical_flux_non_cons; /*--- variable to compute the numerical flux for the non-conservative part
                                                                     (this is necessary to call 'make_flux') ---*/

  std::size_t nfiles; /*--- Number of files desired for output ---*/

  std::string filename; /*--- Auxiliary variable to store the name of output ---*/

  Field conserved_variables; /*--- The variable which stores the conserved variables,
                                   namely the variables for which we solve a PDE system ---*/

  /*--- Now we declare a bunch of fields which depend from the state,
        but it is useful to have it for the output ---*/
  Field_Scalar rho,
               p,
               rho1,
               p1,
               c1,
               T1,
               e1_0,
               e1,
               de1,
               rho2,
               p2,
               c2,
               T2,
               c,
               alpha2,
               Y2,
               e2_0,
               e2,
               de2;

  Field_Vect vel;

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  void create_fields(); /*--- Auxiliary routine to initialize the fileds to the mesh ---*/

  void init_variables(const Riemann_Parameters<double>& Riemann_param); /*--- Routine to initialize the variables
                                                                              (both conserved and auxiliary, this is problem dependent) ---*/

  void apply_bcs(const Riemann_Parameters<double>& Riemann_param); /*--- Auxiliary routine for the boundary conditions ---*/

  void update_auxiliary_fields(); /*--- Routine to update auxiliary fields for output and time step update ---*/

  Number get_max_lambda() const; /*--- Compute the estimate of the maximum eigenvalue ---*/

  void update_pressure_before_relaxation(); /*--- Update pressure fields before relaxation ---*/

  void apply_instantaneous_pressure_relaxation_linearization(); /*--- Apply an instantaneous pressure relaxation (linearization method Pelanti based) ---*/

  void apply_finite_rate_pressure_relaxation(); /*--- Apply a finite rate pressure relaxation (arbitrary-rate Pelanti based) ---*/
};

//////////////////////////////////////////////////////////////
/*---- START WITH THE IMPLEMENTATION OF THE CONSTRUCTOR ---*/
//////////////////////////////////////////////////////////////

// Implement class constructor
//
template<std::size_t dim>
Relaxation<dim>::Relaxation(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                            const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                            const Simulation_Parameters<double>& sim_param,
                            const EOS_Parameters<double>& eos_param,
                            const Riemann_Parameters<double>& Riemann_param):
  box(min_corner, max_corner), mesh(box, sim_param.min_level, sim_param.max_level, {{false}}),
  t0(sim_param.t0), Tf(sim_param.Tf),
  apply_pressure_relax(sim_param.apply_pressure_relax),
  cfl(sim_param.Courant),
  apply_finite_rate_relax(sim_param.apply_finite_rate_relax), tau_p(sim_param.tau_p),
  EOS_phase1(eos_param.gamma_1, eos_param.pi_infty_1, eos_param.q_infty_1, eos_param.c_v_1),
  EOS_phase2(eos_param.gamma_2, eos_param.pi_infty_2, eos_param.q_infty_2, eos_param.c_v_2),
  numerical_flux_cons(EOS_phase1, EOS_phase2),
  numerical_flux_non_cons(EOS_phase1, EOS_phase2),
  nfiles(sim_param.nfiles)
  {
    assertm(sim_param.min_level == sim_param.max_level,
            "The current implementation does not support multiresolution");

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) {
      std::cout << "Initializing variables " << std::endl;
      std::cout << std::endl;
    }

    /*--- Attach the fields to the mesh ---*/
    create_fields();

    /*--- Initialize the fields ---*/
    init_variables(Riemann_param);

    /*--- Apply boundary conditions ---*/
    apply_bcs(Riemann_param);
  }

// Auxiliary routine to create the fields
//
template<std::size_t dim>
void Relaxation<dim>::create_fields() {
  /*--- Create conserved and auxiliary fields ---*/
  conserved_variables = samurai::make_vector_field<Number, EquationData::NVARS>("conserved", mesh);

  rho    = samurai::make_scalar_field<Number>("rho", mesh);
  p      = samurai::make_scalar_field<Number>("p", mesh);

  rho1   = samurai::make_scalar_field<Number>("rho1", mesh);
  p1     = samurai::make_scalar_field<Number>("p1", mesh);
  c1     = samurai::make_scalar_field<Number>("c1", mesh);
  T1     = samurai::make_scalar_field<Number>("T1", mesh);

  rho2   = samurai::make_scalar_field<Number>("rho2", mesh);
  p2     = samurai::make_scalar_field<Number>("p2", mesh);
  c2     = samurai::make_scalar_field<Number>("c2", mesh);
  T2     = samurai::make_scalar_field<Number>("T2", mesh);

  c      = samurai::make_scalar_field<Number>("c", mesh);

  vel    = samurai::make_vector_field<Number, dim>("vel", mesh);

  alpha2 = samurai::make_scalar_field<Number>("alpha2", mesh);
  Y2     = samurai::make_scalar_field<Number>("Y2", mesh);

  e1     = samurai::make_scalar_field<Number>("e1", mesh);
  e1_0   = samurai::make_scalar_field<Number>("e1_0", mesh);
  de1    = samurai::make_scalar_field<Number>("de1", mesh);

  e2     = samurai::make_scalar_field<Number>("e2", mesh);
  e2_0   = samurai::make_scalar_field<Number>("e2_0", mesh);
  de2    = samurai::make_scalar_field<Number>("de2", mesh);
}

// Initialization of conserved and auxiliary variables
//
template<std::size_t dim>
void Relaxation<dim>::init_variables(const Riemann_Parameters<double>& Riemann_param) {
  /*--- Initialize the fields with a loop over all cells ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              const auto center = cell.center();
                              const auto x      = static_cast<Number>(center[0]);

                              // Left state (primitive variables)
                              if(x <= Riemann_param.xd) {
                                conserved_variables[cell][ALPHA1_INDEX] = Riemann_param.alpha1L;

                                vel[cell][0] = Riemann_param.uL;

                                p1[cell]     = Riemann_param.p1L;
                                rho1[cell]   = Riemann_param.rho1L;

                                p2[cell]     = Riemann_param.p2L;
                                rho2[cell]   = Riemann_param.rho2L;
                              }
                              // Right state (primitive variables)
                              else {
                                conserved_variables[cell][ALPHA1_INDEX] = Riemann_param.alpha1R;

                                vel[cell][0] = Riemann_param.uR;

                                p1[cell]     = Riemann_param.p1R;
                                rho1[cell]   = Riemann_param.rho1R;

                                p2[cell]     = Riemann_param.p2R;
                                rho2[cell]   = Riemann_param.rho2R;
                              }

                              // Complete the conserved variables (and some auxiliary fields for the sake of completeness)
                              conserved_variables[cell][ALPHA1_RHO1_INDEX] = conserved_variables[cell][ALPHA1_INDEX]*rho1[cell];

                              conserved_variables[cell][ALPHA2_RHO2_INDEX] = (static_cast<Number>(1.0) -
                                                                              conserved_variables[cell][ALPHA1_INDEX])*rho2[cell];

                              rho[cell] = conserved_variables[cell][ALPHA1_RHO1_INDEX]
                                        + conserved_variables[cell][ALPHA2_RHO2_INDEX];
                              for(std::size_t d = 0; d < Field::dim; ++d) {
                                conserved_variables[cell][RHO_U_INDEX + d] = rho[cell]*vel[cell][d];
                              }

                              e1[cell] = EOS_phase1.e_value(rho1[cell], p1[cell]);
                              conserved_variables[cell][ALPHA1_RHO1_e1_INDEX] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*e1[cell];

                              e2[cell] = EOS_phase2.e_value(rho2[cell], p2[cell]);
                              conserved_variables[cell][ALPHA2_RHO2_e2_INDEX] = conserved_variables[cell][ALPHA2_RHO2_INDEX]*e2[cell];

                              c1[cell] = EOS_phase1.c_value(rho1[cell], p1[cell]);

                              T1[cell] = EOS_phase1.T_value(rho1[cell], p1[cell]);

                              c2[cell] = EOS_phase2.c_value(rho2[cell], p2[cell]);

                              T2[cell] = EOS_phase2.T_value(rho2[cell], p2[cell]);

                              alpha2[cell] = static_cast<Number>(1.0) - conserved_variables[cell][ALPHA1_INDEX];
                              Y2[cell]     = conserved_variables[cell][ALPHA2_RHO2_INDEX]/rho[cell];

                              p[cell] = conserved_variables[cell][ALPHA1_INDEX]*p1[cell]
                                      + alpha2[cell]*p2[cell];

                              c[cell] = std::sqrt((static_cast<Number>(1.0) - Y2[cell])*c1[cell]*c1[cell] +
                                                  Y2[cell]*c2[cell]*c2[cell]);
                            }
                        );
}

// Auxiliary routine to impose the boundary conditions
//
template<std::size_t dim>
void Relaxation<dim>::apply_bcs(const Riemann_Parameters<double>& Riemann_param) {
  const xt::xtensor_fixed<int, xt::xshape<1>> left  = {-1};
  const xt::xtensor_fixed<int, xt::xshape<1>> right = {1};
  samurai::make_bc<samurai::Dirichlet<1>>(conserved_variables,
                                          Riemann_param.alpha1L,
                                          Riemann_param.alpha1L*Riemann_param.rho1L,
                                          (static_cast<Number>(1.0) - Riemann_param.alpha1L)*Riemann_param.rho2L,
                                          (Riemann_param.alpha1L*Riemann_param.rho1L +
                                           (static_cast<Number>(1.0) - Riemann_param.alpha1L)*Riemann_param.rho2L)*Riemann_param.uL,
                                          Riemann_param.alpha1L*Riemann_param.rho1L*
                                          (EOS_phase1.e_value(Riemann_param.rho1L, Riemann_param.p1L)),
                                          (static_cast<Number>(1.0) - Riemann_param.alpha1L)*Riemann_param.rho2L*
                                          (EOS_phase2.e_value(Riemann_param.rho2L, Riemann_param.p2L)))->on(left);
  samurai::make_bc<samurai::Dirichlet<1>>(conserved_variables,
                                          Riemann_param.alpha1R,
                                          Riemann_param.alpha1R*Riemann_param.rho1R,
                                          (static_cast<Number>(1.0) - Riemann_param.alpha1R)*Riemann_param.rho2R,
                                          (Riemann_param.alpha1R*Riemann_param.rho1R +
                                           (static_cast<Number>(1.0) - Riemann_param.alpha1R)*Riemann_param.rho2R)*Riemann_param.uR,
                                          Riemann_param.alpha1R*Riemann_param.rho1R*
                                          (EOS_phase1.e_value(Riemann_param.rho1R, Riemann_param.p1R)),
                                          (static_cast<Number>(1.0) - Riemann_param.alpha1R)*Riemann_param.rho2R*
                                          (EOS_phase2.e_value(Riemann_param.rho2R, Riemann_param.p2R)))->on(right);
}

//////////////////////////////////////////////////////////////
/*---- FOCUS NOW ON THE AUXILIARY FUNCTIONS ---*/
/////////////////////////////////////////////////////////////

// Compute the estimate of the maximum eigenvalue for CFL condition
//
template<std::size_t dim>
typename Relaxation<dim>::Field::value_type Relaxation<dim>::get_max_lambda() const {
  auto local_res = static_cast<Number>(0.0);

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              for(std::size_t d = 0; d < Field::dim; ++d) {
                                local_res = std::max(std::abs(vel[cell][d]) + c[cell],
                                                     local_res);
                              }
                            }
                        );

  Number global_res;
  MPI_Allreduce(&local_res, &global_res, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  return global_res;
}

// Update auxiliary fields after solution of the system
//
template<std::size_t dim>
void Relaxation<dim>::update_auxiliary_fields() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              /*--- Pre-fetch variables that will be used several times ---*/
                              const auto alpha1_loc = conserved_variables[cell][ALPHA1_INDEX];
                              const auto m1_loc     = conserved_variables[cell][ALPHA1_RHO1_INDEX];
                              const auto m2_loc     = conserved_variables[cell][ALPHA2_RHO2_INDEX];
                              const auto m1e1_loc   = conserved_variables[cell][ALPHA1_RHO1_e1_INDEX];
                              const auto m2e2_loc   = conserved_variables[cell][ALPHA2_RHO2_e2_INDEX];

                              /*--- Compute mixture density and velocity ---*/
                              const auto rho_loc     = m1_loc + m2_loc;
                              const auto inv_rho_loc = static_cast<Number>(1.0)/rho_loc;
                              rho[cell]              = rho_loc;
                              for(std::size_t d = 0; d < Field::dim; ++d) {
                                vel[cell][d] = conserved_variables[cell][RHO_U_INDEX + d]*inv_rho_loc;
                              }

                              /*--- Phase 1 ---*/
                              const auto rho1_loc = m1_loc/alpha1_loc;
                              rho1[cell]          = rho1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              const auto e1_loc   = m1e1_loc/m1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              e1[cell]            = e1_loc;
                              de1[cell]           = e1_loc - e1_0[cell];
                              const auto p1_loc   = EOS_phase1.pres_value(rho1_loc, e1_loc);
                              p1[cell]            = p1_loc;
                              c1[cell]            = EOS_phase1.c_value(rho1_loc, p1_loc);
                              T1[cell]            = EOS_phase1.T_value(rho1_loc, p1_loc);

                              /*--- Phase 2 ---*/
                              const auto rho2_loc = m2_loc/(static_cast<Number>(1.0) - alpha1_loc);
                                                    /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              rho2[cell]          = rho2_loc;
                              const auto e2_loc   = m2e2_loc/m2_loc;
                              e2[cell]            = e2_loc;
                              de2[cell]           = e2_loc - e2_0[cell];
                              const auto p2_loc   = EOS_phase2.pres_value(rho2_loc, e2_loc);
                              p2[cell]            = p2_loc;
                              c2[cell]            = EOS_phase2.c_value(rho2_loc, p2_loc);
                              T2[cell]            = EOS_phase2.T_value(rho2_loc, p2_loc);

                              const auto alpha2_loc = static_cast<Number>(1.0) - alpha1_loc;
                              alpha2[cell]          = alpha2_loc;
                              const auto Y2_loc     = m2_loc*inv_rho_loc;
                              Y2[cell]              = Y2_loc;

                              /*--- Remaining mixture variables ---*/
                              p[cell] = alpha1_loc*p1_loc
                                      + alpha2_loc*p2_loc;

                              c[cell] = std::sqrt((static_cast<Number>(1.0) - Y2_loc)*c1[cell]*c1[cell] +
                                                   Y2_loc*c2[cell]*c2[cell]);
                         });
}

// Save desired fields and info
//
template<std::size_t dim>
template<class... Variables>
void Relaxation<dim>::save(const fs::path& path,
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
}

//////////////////////////////////////////////////////////////
/*---- FOCUS NOW ON THE RELAXATION FUNCTIONS ---*/
/////////////////////////////////////////////////////////////

// Update pressure fields before relaxation
//
template<std::size_t dim>
void Relaxation<dim>::update_pressure_before_relaxation() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              /*--- Pre-fetch variables that will be used several times so as to exploit vectorization
                                    (as well as to enhance readability) ---*/
                              const auto alpha1_loc = conserved_variables[cell][ALPHA1_INDEX];
                              const auto m1_loc     = conserved_variables[cell][ALPHA1_RHO1_INDEX];
                              const auto m2_loc     = conserved_variables[cell][ALPHA2_RHO2_INDEX];
                              const auto m1e1_loc   = conserved_variables[cell][ALPHA1_RHO1_e1_INDEX];
                              const auto m2e2_loc   = conserved_variables[cell][ALPHA2_RHO2_e2_INDEX];

                              /*--- Compute the variables ----*/
                              const auto e1_loc   = m1e1_loc/m1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              e1_0[cell]          = e1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/

                              const auto e2_loc   = m2e2_loc/m2_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              e2_0[cell]          = e2_loc;

                              const auto rho1_loc = m1_loc/alpha1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              rho1[cell]          = rho1_loc;
                              const auto p1_loc   = EOS_phase1.pres_value(rho1_loc, e1_loc);
                              p1[cell]            = p1_loc;
                              c1[cell]            = EOS_phase1.c_value(rho1_loc, p1_loc);

                              const auto rho2_loc = m2_loc/(static_cast<Number>(1.0) - alpha1_loc);
                              /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              rho2[cell]          = rho2_loc;
                              const auto p2_loc   = EOS_phase2.pres_value(rho2_loc, e2_loc);
                              p2[cell]            = p2_loc;
                              c2[cell]            = EOS_phase2.c_value(rho2_loc, p2_loc);
                            }
                        );
}

// Apply the instantaneous relaxation for the pressure (polynomial method)
//
template<std::size_t dim>
void Relaxation<dim>::apply_instantaneous_pressure_relaxation_linearization() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           /*--- Pre-fetch variables that will be used several times so as to exploit possible vectorization
                                 (as well as to enhance readability) ---*/
                           auto alpha1_loc   = conserved_variables[cell][ALPHA1_INDEX];
                           auto alpha2_loc   = static_cast<Number>(1.0) - alpha1_loc;
                           const auto m1_loc = conserved_variables[cell][ALPHA1_RHO1_INDEX];
                           const auto m2_loc = conserved_variables[cell][ALPHA2_RHO2_INDEX];

                           auto rho1_loc     = rho1[cell];
                           auto p1_loc       = p1[cell];
                           auto c1_loc       = c1[cell];

                           auto rho2_loc     = rho2[cell];
                           auto p2_loc       = p2[cell];
                           auto c2_loc       = c2[cell];

                           /*--- Compute the pressure equilibrium with the linearization method (Pelanti) ---*/
                           const auto a    = static_cast<Number>(1.0)
                                           + EOS_phase2.get_gamma()*alpha1_loc
                                           + EOS_phase1.get_gamma()*alpha2_loc;
                           const auto Z1   = rho1_loc*c1_loc;
                           const auto Z2   = rho2_loc*c2_loc;
                           const auto pI_0 = (Z2*p1_loc + Z1*p2_loc)/(Z1 + Z2);
                           const auto C1   = static_cast<Number>(2.0)*EOS_phase1.get_gamma()*EOS_phase1.get_pi_infty()
                                           + (EOS_phase1.get_gamma() - static_cast<Number>(1.0))*pI_0;
                           const auto C2   = static_cast<Number>(2.0)*EOS_phase2.get_gamma()*EOS_phase2.get_pi_infty()
                                           + (EOS_phase2.get_gamma() - static_cast<Number>(1.0))*pI_0;
                           const auto b    = C1*alpha2_loc + C2*alpha1_loc
                                           - (static_cast<Number>(1.0) + EOS_phase2.get_gamma())*
                                             alpha1_loc*p1_loc
                                           - (static_cast<Number>(1.0) + EOS_phase1.get_gamma())*
                                             alpha2_loc*p2_loc;
                           const auto d    = -(C2*alpha1_loc*p1_loc + C1*alpha2_loc*p2_loc);

                           const auto p_star = (-b + std::sqrt(b*b - static_cast<Number>(4.0)*a*d))/
                                               (static_cast<Number>(2.0)*a);

                           /*--- Update the volume fraction using the computed pressure ---*/
                           alpha1_loc *= ((EOS_phase1.get_gamma() - static_cast<Number>(1.0))*p_star +
                                          static_cast<Number>(2.0)*p1_loc + C1)/
                                         ((EOS_phase1.get_gamma() + static_cast<Number>(1.0))*p_star + C1);
                           conserved_variables[cell][ALPHA1_INDEX] = alpha1_loc;

                           /*--- Update the internal energy of both phases ---*/
                           const auto e1_np1 = EOS_phase1.e_value(m1_loc/alpha1_loc, p_star);
                           /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           const auto e2_np1 = EOS_phase2.e_value(m2_loc/(static_cast<Number>(1.0) - alpha1_loc), p_star);
                           /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           #ifdef NDEBUG
                             const auto rhoe_0 = conserved_variables[cell][ALPHA1_RHO1_e1_INDEX]
                                               + conserved_variables[cell][ALPHA2_RHO2_e2_INDEX];
                             conserved_variables[cell][ALPHA1_RHO1_e1_INDEX] = m1_loc*e1_np1;
                             conserved_variables[cell][ALPHA2_RHO2_e2_INDEX] = m2_loc*e2_np1;
                             assertm(std::abs((conserved_variables[cell][ALPHA1_RHO1_e1_INDEX] +
                                               conserved_variables[cell][ALPHA2_RHO2_e2_INDEX]) -
                                              rhoe_0)/rhoe_0 < static_cast<Number>(1e-12),
                                     "No conservation of total energy in the relexation");
                           #else
                             conserved_variables[cell][ALPHA1_RHO1_e1_INDEX] = m1_loc*e1_np1;
                             conserved_variables[cell][ALPHA2_RHO2_e2_INDEX] = m2_loc*e2_np1;
                           #endif
                         });
}

// Apply the finite relaxation for the pressure (Pelanti)
//
template<std::size_t dim>
void Relaxation<dim>::apply_finite_rate_pressure_relaxation() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              /*--- Pre-fetch variables that will be used several times so as to exploit possible vectorization
                                   (as well as to enhance readability) ---*/
                              auto alpha1_loc   = conserved_variables[cell][ALPHA1_INDEX];
                              auto alpha2_loc   = static_cast<Number>(1.0) - alpha1_loc;
                              const auto m1_loc = conserved_variables[cell][ALPHA1_RHO1_INDEX];
                              const auto m2_loc = conserved_variables[cell][ALPHA2_RHO2_INDEX];

                              auto rho1_loc     = rho1[cell];
                              auto p1_loc       = p1[cell];
                              auto c1_loc       = c1[cell];

                              auto rho2_loc     = rho2[cell];
                              auto p2_loc       = p2[cell];
                              auto c2_loc       = c2[cell];

                              /*--- Compute constant fields that do not change by hypothesis in the relaxation ---*/
                              const auto Z1   = rho1_loc*c1_loc;
                              const auto Z2   = rho2_loc*c2_loc;
                              const auto pI_0 = (Z2*p1_loc + Z1*p2_loc)/(Z1 + Z2);

                              const auto xi1_m1_0 = static_cast<Number>(1.0)/alpha1_loc*
                                                    ((EOS_phase1.get_gamma() - static_cast<Number>(1.0))*pI_0 +
                                                     p1_loc + EOS_phase1.get_gamma()*EOS_phase1.get_pi_infty());
                                                    /*--- TODO: Add treatment for vanishing volume fraction ---*/
                              const auto xi2_m1_0 = static_cast<Number>(1.0)/alpha2_loc*
                                                    ((EOS_phase2.get_gamma() - static_cast<Number>(1.0))*pI_0 +
                                                     p2_loc + EOS_phase2.get_gamma()*EOS_phase2.get_pi_infty());
                                                    /*--- TODO: Add treatment for vanishing volume fraction ---*/

                              const auto rhoe_0 = conserved_variables[cell][ALPHA1_RHO1_e1_INDEX]
                                                + conserved_variables[cell][ALPHA2_RHO2_e2_INDEX];

                              /*--- Update the volume fraction ---*/
                              alpha1_loc += (p1_loc - p2_loc)/(xi1_m1_0 + xi2_m1_0)*
                                            (static_cast<Number>(1.0) - std::exp(-dt/tau_p));
                              conserved_variables[cell][ALPHA1_INDEX] = alpha1_loc;

                              /*--- Compute the pressure difference after relaxation ---*/
                              const auto Delta_p_star = (p1_loc - p2_loc)*std::exp(-dt/tau_p);

                              /*--- Compute phase 2 pressure after relaxation ---*/
                              alpha2_loc = static_cast<Number>(1.0) - alpha1_loc;
                              const auto p2_star = (rhoe_0 -
                                                    alpha1_loc*
                                                    (Delta_p_star/(EOS_phase1.get_gamma() - static_cast<Number>(1.0)) +
                                                     EOS_phase1.get_gamma()*EOS_phase1.get_pi_infty()/
                                                     (EOS_phase1.get_gamma() - static_cast<Number>(1.0)) +
                                                     EOS_phase1.get_q_infty()*m1_loc/alpha1_loc) -
                                                    alpha2_loc*
                                                    (EOS_phase2.get_gamma()*EOS_phase2.get_pi_infty()/
                                                     (EOS_phase2.get_gamma() - static_cast<Number>(1.0)) +
                                                     EOS_phase2.get_q_infty()*m2_loc/alpha2_loc))/
                                                   (alpha1_loc/(EOS_phase1.get_gamma() - static_cast<Number>(1.0)) +
                                                    alpha2_loc/(EOS_phase2.get_gamma() - static_cast<Number>(1.0)));
                                                   /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           /*--- Update the internal energy of both phases ---*/
                           const auto e2_np1 = EOS_phase2.e_value(m2_loc/alpha2_loc, p2_star); /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           conserved_variables[cell][ALPHA2_RHO2_e2_INDEX] = m2_loc*e2_np1;
                           conserved_variables[cell][ALPHA1_RHO1_e1_INDEX] = rhoe_0 - conserved_variables[cell][ALPHA2_RHO2_e2_INDEX];
                         });
}

//////////////////////////////////////////////////////////////
/*---- IMPLEMENT THE FUNCTION THAT EFFECTIVELY SOLVES THE PROBLEM ---*/
/////////////////////////////////////////////////////////////

// Implement the function that effectively performs the temporal loop
//
template<std::size_t dim>
void Relaxation<dim>::run() {
  /*--- Default output arguemnts ---*/
  fs::path path = fs::current_path();
  filename      = "Relaxation_HLLC_non_cons_6eqs_total_energy";

  #ifdef ORDER_2
    filename = filename + "_order2";
  #else
    filename = filename + "_order1";
  #endif

  const auto dt_save = Tf/static_cast<Number>(nfiles);

  /*--- Auxiliary variables to save updated fields ---*/
  #ifdef ORDER_2
    auto conserved_variables_tmp = samurai::make_vector_field<Number, EquationData::NVARS>("conserved_tmp", mesh);
    auto conserved_variables_old = samurai::make_vector_field<Number, EquationData::NVARS>("conserved_old", mesh);
  #endif
  auto conserved_variables_np1 = samurai::make_vector_field<Number, EquationData::NVARS>("conserved_np1", mesh);

  /*--- Create the flux variables ---*/
  auto HLLC_Conservative_flux = numerical_flux_cons.make_flux();
  auto NonConservative_flux   = numerical_flux_non_cons.make_flux();

  /*--- Save the initial condition ---*/
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(path, suffix_init, conserved_variables,
                          rho, p, vel, c,
                          rho1, p1, c1, T1, e1, rho2,
                          p2, c2, T2, alpha2, Y2, e2);

  /*--- Save mesh size ---*/
  const auto dx   = static_cast<Number>(mesh.cell_length(mesh.max_level()));
  using mesh_id_t = typename decltype(mesh)::mesh_id_t;
  const auto n_elements_per_subdomain = mesh[mesh_id_t::cells].nb_cells();
  unsigned n_elements;
  MPI_Allreduce(&n_elements_per_subdomain, &n_elements, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank == 0) {
    std::cout << "Number of initial elements = " <<  n_elements << std::endl;
    std::cout << std::endl;
  }

  /*--- Start the loop ---*/
  std::size_t nsave = 0;
  std::size_t nt    = 0;
  auto t            = static_cast<Number>(t0);
  dt                = std::min(Tf - t, cfl*dx/get_max_lambda());
  while(t != Tf) {
    t += dt;

    if(rank == 0) {
      std::cout << fmt::format("Iteration {}: t = {}, dt = {}", ++nt, t, dt) << std::endl;
    }

    // Save current state in case of order 2
    #ifdef ORDER_2
      conserved_variables_old = conserved_variables;
    #endif

    // Apply the numerical scheme
    samurai::update_ghost_mr(conserved_variables);
    auto Cons_Flux    = HLLC_Conservative_flux(conserved_variables);
    auto NonCons_Flux = NonConservative_flux(conserved_variables);
    #ifdef ORDER_2
      conserved_variables_tmp = conserved_variables - dt*Cons_Flux - dt*NonCons_Flux;
      std::swap(conserved_variables.array(), conserved_variables_tmp.array());
    #else
      conserved_variables_np1 = conserved_variables - dt*Cons_Flux - dt*NonCons_Flux;
      std::swap(conserved_variables.array(), conserved_variables_np1.array());
    #endif

    // Apply the relaxation for the pressure
    if(apply_pressure_relax) {
      update_pressure_before_relaxation();
      if(apply_finite_rate_relax) {
        apply_finite_rate_pressure_relaxation();
      }
      else {
        apply_instantaneous_pressure_relaxation_linearization();
      }
    }

    // Consider the second stage for the second order
    #ifdef ORDER_2
      // Apply the numerical scheme
      samurai::update_ghost_mr(conserved_variables);
      conserved_variables_tmp = conserved_variables - dt*Cons_Flux - dt*NonCons_Flux;
      conserved_variables_np1 = static_cast<Number>(0.5)*
                                (conserved_variables_tmp + conserved_variables_old);
      std::swap(conserved_variables.array(), conserved_variables_np1.array());

      // Apply the relaxation for the pressure
      if(apply_pressure_relax) {
        update_pressure_before_relaxation();
        if(apply_finite_rate_relax) {
          apply_finite_rate_pressure_relaxation();
        }
        else {
          apply_instantaneous_pressure_relaxation_linearization();
        }
      }
    #endif

    // Compute updated time step
    update_auxiliary_fields();
    dt = std::min(Tf - t, cfl*dx/get_max_lambda());

    // Save the results
    if(t >= static_cast<Number>(nsave + 1)*dt_save || t == Tf) {
      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";
      save(path, suffix, conserved_variables,
                         rho, p, vel, c,
                         rho1, p1, c1, T1, e1_0, e1, de1,
                         rho2, p2, c2, T2, alpha2, Y2, e2_0, e2, de2);
    }
  }
}
