// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#include <samurai/algorithm/update.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>

#include <filesystem>
namespace fs = std::filesystem;

//#define HLLC_FLUX
#define HLLC_NON_CONS_FLUX
//#define RUSANOV_FLUX

#ifdef HLLC_FLUX
  #include "HLLC_6eqs_internal_energy_flux.hpp"
#else
  #ifdef HLLC_NON_CONS_FLUX
    #include "HLLC_conservative_6eqs_internal_energy_flux.hpp"
  #elifdef RUSANOV_FLUX
    #include "Rusanov_6eqs_internal_energy_flux.hpp"
  #endif
  #include "non_conservative_6eqs_internal_energy_flux.hpp"
#endif

#include "containers.hpp"

#define assertm(exp, msg) assert(((void)msg, exp))

// Specify the use of this namespace where we just store the indices
// and some parameters related to the equations of state
using namespace EquationData;

// This is the class for the simulation of a 6-equation mixture-energy consistent model
//
template<std::size_t dim>
class Relaxation {
public:
  using Config = samurai::MRConfig<dim, 2>;

  Relaxation() = default; /*--- Default constructor. This will do nothing
                                and basically will never be used ---*/

  Relaxation(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
             const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
             const Simulation_Parameters& sim_param,
             const EOS_Parameters& eos_param,
             const Riemann_Parameters& Riemann_param); /*--- Class constructor with the arguments related
                                                             to the grid, to the physics and to the relaxation ---*/

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
  using Field_Scalar = samurai::ScalarField<decltype(mesh), typename Field::value_type>;
  using Field_Vect   = samurai::VectorField<decltype(mesh), typename Field::value_type, dim, false>;

  double Tf;  /*--- Final time of the simulation ---*/
  double cfl; /*--- Courant number of the simulation so as to compute the time step ---*/

  std::size_t nfiles; /*--- Number of files desired for output ---*/

  bool   apply_pressure_relax;    /*--- Set whether to apply or not the pressure relaxation ---*/
  bool   apply_finite_rate_relax; /*--- Set whether to perform a finite rate relaxation or an infinite rate ---*/
  double tau_p;                   /*--- Finite rate parameter ---*/
  double dt;                      /*--- Time step (to be declared here because of finite rate) ---*/

  Field conserved_variables; /*--- The variable which stores the conserved variables,
                                   namely the variables for which we solve a PDE system ---*/

  const SG_EOS<typename Field::value_type> EOS_phase1; /*--- Equation of state of phase 1 ---*/
  const SG_EOS<typename Field::value_type> EOS_phase2; /*--- Equation of state of phase 2 ---*/

  #ifdef RUSANOV_FLUX
    samurai::RusanovFlux<Field> numerical_flux_cons; /*--- variable to compute the numerical flux for the conservative part
                                                           (this is necessary to call 'make_flux') ---*/

    samurai::NonConservativeFlux<Field> numerical_flux_non_cons; /*--- variable to compute the numerical flux for the non-conservative part
                                                                      (this is necessary to call 'make_flux') ---*/
  #elifdef HLLC_FLUX
    samurai::HLLCFlux<Field> numerical_flux; /*--- variable to compute the numerical flux
                                                  (this is necessary to call 'make_flux') ---*/
  #elifdef HLLC_NON_CONS_FLUX
    samurai::HLLCFlux_Conservative<Field> numerical_flux_cons; /*--- variable to compute the numerical flux for the conservative part
                                                                    (this is necessary to call 'make_flux') ---*/

    samurai::NonConservativeFlux<Field> numerical_flux_non_cons; /*--- variable to compute the numerical flux for the non-conservative part
                                                                      (this is necessary to call 'make_flux') ---*/
  #endif

  std::string filename; /*--- Auxiliary variable to store the name of output ---*/

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
  void init_variables(const Riemann_Parameters& Riemann_param); /*--- Routine to initialize the variables
                                                                      (both conserved and auxiliary, this is problem dependent) ---*/

  void update_auxiliary_fields(); /*--- Routine to update auxiliary fields for output and time step update ---*/

  double get_max_lambda() const; /*--- Compute the estimate of the maximum eigenvalue ---*/

  void update_pressure_before_relaxation(); /*--- Update pressure fields before relaxation ---*/

  void apply_instantaneous_pressure_relaxation_linearization(); /*--- Apply an instantaneous pressure relaxation (linearization method Pelanti based) ---*/

  void apply_finite_rate_pressure_relaxation(); /*--- Apply a finite rate pressure relaxation (arbitrary rate Pelanti based) ---*/
};

// Implement class constructor
//
template<std::size_t dim>
Relaxation<dim>::Relaxation(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                            const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                            const Simulation_Parameters& sim_param,
                            const EOS_Parameters& eos_param,
                            const Riemann_Parameters& Riemann_param):
  box(min_corner, max_corner), mesh(box, sim_param.min_level, sim_param.max_level, {{false}}),
  Tf(sim_param.Tf), cfl(sim_param.Courant), nfiles(sim_param.nfiles),
  apply_pressure_relax(sim_param.apply_pressure_relax),
  apply_finite_rate_relax(sim_param.apply_finite_rate_relax), tau_p(sim_param.tau_p),
  EOS_phase1(eos_param.gamma_1, eos_param.pi_infty_1, eos_param.q_infty_1, eos_param.c_v_1),
  EOS_phase2(eos_param.gamma_2, eos_param.pi_infty_2, eos_param.q_infty_2, eos_param.c_v_2),
  #if defined RUSANOV_FLUX || defined HLLC_NON_CONS_FLUX
    numerical_flux_cons(EOS_phase1, EOS_phase2),
    numerical_flux_non_cons(EOS_phase1, EOS_phase2)
  #else
    numerical_flux(EOS_phase1, EOS_phase2)
  #endif
  {
    std::cout << "Initializing variables" << std::endl;
    std::cout << std::endl;
    init_variables(Riemann_param);
  }

// Initialization of conserved and auxiliary variables
//
template<std::size_t dim>
void Relaxation<dim>::init_variables(const Riemann_Parameters& Riemann_param) {
  /*--- Create conserved and auxiliary fields ---*/
  conserved_variables = samurai::make_vector_field<typename Field::value_type, EquationData::NVARS>("conserved", mesh);

  rho    = samurai::make_scalar_field<typename Field::value_type>("rho", mesh);
  p      = samurai::make_scalar_field<typename Field::value_type>("p", mesh);

  rho1   = samurai::make_scalar_field<typename Field::value_type>("rho1", mesh);
  p1     = samurai::make_scalar_field<typename Field::value_type>("p1", mesh);
  c1     = samurai::make_scalar_field<typename Field::value_type>("c1", mesh);
  T1     = samurai::make_scalar_field<typename Field::value_type>("T1", mesh);

  rho2   = samurai::make_scalar_field<typename Field::value_type>("rho2", mesh);
  p2     = samurai::make_scalar_field<typename Field::value_type>("p2", mesh);
  c2     = samurai::make_scalar_field<typename Field::value_type>("c2", mesh);
  T2     = samurai::make_scalar_field<typename Field::value_type>("T2", mesh);

  c      = samurai::make_scalar_field<typename Field::value_type>("c", mesh);

  vel    = samurai::make_vector_field<typename Field::value_type, dim>("vel", mesh);

  alpha2 = samurai::make_scalar_field<typename Field::value_type>("alpha2", mesh);
  Y2     = samurai::make_scalar_field<typename Field::value_type>("Y2", mesh);

  e1     = samurai::make_scalar_field<typename Field::value_type>("e1", mesh);
  e1_0   = samurai::make_scalar_field<typename Field::value_type>("e1_0", mesh);
  de1    = samurai::make_scalar_field<typename Field::value_type>("de1", mesh);

  e2     = samurai::make_scalar_field<typename Field::value_type>("e2", mesh);
  e2_0   = samurai::make_scalar_field<typename Field::value_type>("e2_0", mesh);
  de2    = samurai::make_scalar_field<typename Field::value_type>("de2", mesh);

  /*--- Initialize the fields with a loop over all cells ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           const auto center = cell.center();
                           const double x    = center[0];

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

                           conserved_variables[cell][ALPHA2_RHO2_INDEX] = (1.0 - conserved_variables[cell][ALPHA1_INDEX])*rho2[cell];

                           rho[cell] = conserved_variables[cell][ALPHA1_RHO1_INDEX]
                                     + conserved_variables[cell][ALPHA2_RHO2_INDEX];
                           for(std::size_t d = 0; d < Field::dim; ++d) {
                             conserved_variables[cell][RHO_U_INDEX + d] = rho[cell]*vel[cell][d];
                           }

                           e1[cell] = EOS_phase1.e_value(rho1[cell], p1[cell]);
                           conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*e1[cell];

                           e2[cell] = EOS_phase2.e_value(rho2[cell], p2[cell]);
                           conserved_variables[cell][ALPHA2_RHO2_E2_INDEX] = conserved_variables[cell][ALPHA2_RHO2_INDEX]*e2[cell];

                           c1[cell] = EOS_phase1.c_value(rho1[cell], p1[cell]);

                           T1[cell] = EOS_phase1.T_value(rho1[cell], p1[cell]);

                           c2[cell] = EOS_phase2.c_value(rho2[cell], p2[cell]);

                           T2[cell] = EOS_phase2.T_value(rho2[cell], p2[cell]);

                           alpha2[cell] = 1.0 - conserved_variables[cell][ALPHA1_INDEX];
                           Y2[cell]     = conserved_variables[cell][ALPHA2_RHO2_INDEX]/rho[cell];

                           p[cell] = conserved_variables[cell][ALPHA1_INDEX]*p1[cell]
                                   + alpha2[cell]*p2[cell];

                           c[cell] = std::sqrt((1.0 - Y2[cell])*c1[cell]*c1[cell] +
                                               Y2[cell]*c2[cell]*c2[cell]);
                         });


  const xt::xtensor_fixed<int, xt::xshape<1>> left{-1};
  const xt::xtensor_fixed<int, xt::xshape<1>> right{1};
  samurai::make_bc<samurai::Dirichlet<1>>(conserved_variables,
                                          Riemann_param.alpha1L,
                                          Riemann_param.alpha1L*Riemann_param.rho1L,
                                          (1.0 - Riemann_param.alpha1L)*Riemann_param.rho2L,
                                          (Riemann_param.alpha1L*Riemann_param.rho1L +
                                           (1.0 - Riemann_param.alpha1L)*Riemann_param.rho2L)*Riemann_param.uL,
                                          Riemann_param.alpha1L*Riemann_param.rho1L*
                                          (EOS_phase1.e_value(Riemann_param.rho1L, Riemann_param.p1L)),
                                          (1.0- Riemann_param.alpha1L)*Riemann_param.rho2L*
                                          (EOS_phase2.e_value(Riemann_param.rho2L, Riemann_param.p2L)))->on(left);
  samurai::make_bc<samurai::Dirichlet<1>>(conserved_variables,
                                          Riemann_param.alpha1R,
                                          Riemann_param.alpha1R*Riemann_param.rho1R,
                                          (1.0 - Riemann_param.alpha1R)*Riemann_param.rho2R,
                                          (Riemann_param.alpha1R*Riemann_param.rho1R +
                                           (1.0 - Riemann_param.alpha1R)*Riemann_param.rho2R)*Riemann_param.uR,
                                          Riemann_param.alpha1R*Riemann_param.rho1R*
                                          (EOS_phase1.e_value(Riemann_param.rho1R, Riemann_param.p1R)),
                                          (1.0- Riemann_param.alpha1R)*Riemann_param.rho2R*
                                          (EOS_phase2.e_value(Riemann_param.rho2R, Riemann_param.p2R)))->on(right);
}

// Update pressure fields before relaxation
//
template<std::size_t dim>
void Relaxation<dim>::update_pressure_before_relaxation() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           e1_0[cell] = conserved_variables[cell][ALPHA1_RHO1_E1_INDEX]/
                                        conserved_variables[cell][ALPHA1_RHO1_INDEX]; /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           e2_0[cell] = conserved_variables[cell][ALPHA2_RHO2_E2_INDEX]/
                                        conserved_variables[cell][ALPHA2_RHO2_INDEX]; /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           p1[cell] = EOS_phase1.pres_value(conserved_variables[cell][ALPHA1_RHO1_INDEX]/
                                                            conserved_variables[cell][ALPHA1_INDEX],
                                                            e1_0[cell]); /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           p2[cell] = EOS_phase2.pres_value(conserved_variables[cell][ALPHA2_RHO2_INDEX]/
                                                            (1.0 - conserved_variables[cell][ALPHA1_INDEX]),
                                                            e2_0[cell]); /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           rho1[cell] = conserved_variables[cell][ALPHA1_RHO1_INDEX]/
                                        conserved_variables[cell][ALPHA1_INDEX]; /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           rho2[cell] = conserved_variables[cell][ALPHA2_RHO2_INDEX]/
                                        (1.0 - conserved_variables[cell][ALPHA1_INDEX]); /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           c1[cell] = EOS_phase1.c_value(rho1[cell], p1[cell]);

                           c2[cell] = EOS_phase2.c_value(rho2[cell], p2[cell]);
                         });
}


// Apply the instantaneous relaxation for the pressure (polynomial method Saurel)
//
template<std::size_t dim>
void Relaxation<dim>::apply_instantaneous_pressure_relaxation_linearization() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           /*--- Compute the pressure equilibrium with the linearization method (Pelanti) ---*/
                           const auto a    = 1.0 + EOS_phase2.get_gamma()*conserved_variables[cell][ALPHA1_INDEX]
                                           + EOS_phase1.get_gamma()*(1.0 - conserved_variables[cell][ALPHA1_INDEX]);
                           const auto Z1   = rho1[cell]*c1[cell];
                           const auto Z2   = rho2[cell]*c2[cell];
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

                           const auto p_star = (-b + std::sqrt(b*b - 4.0*a*d))/(2.0*a);

                           /*--- Update the volume fraction using the computed pressure ---*/
                           conserved_variables[cell][ALPHA1_INDEX] *= ((EOS_phase1.get_gamma() - 1.0)*p_star + 2*p1[cell] + C1)/
                                                                      ((EOS_phase1.get_gamma() + 1.0)*p_star + C1);

                           /*--- Update the internal energy of both phases ---*/
                           const auto e1_np1 = EOS_phase1.e_value(conserved_variables[cell][ALPHA1_RHO1_INDEX]/
                                                                  conserved_variables[cell][ALPHA1_INDEX],
                                                                  p_star); /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           const auto e2_np1 = EOS_phase2.e_value(conserved_variables[cell][ALPHA2_RHO2_INDEX]/
                                                                  (1.0 - conserved_variables[cell][ALPHA1_INDEX]),
                                                                  p_star); /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           #ifdef NDEBUG
                             const auto rhoe_0 = conserved_variables[cell][ALPHA1_RHO1_E1_INDEX]
                                               + conserved_variables[cell][ALPHA2_RHO2_E2_INDEX];
                             conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*e1_np1;
                             conserved_variables[cell][ALPHA2_RHO2_E2_INDEX] = conserved_variables[cell][ALPHA2_RHO2_INDEX]*e2_np1;
                             assertm(std::abs((conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] +
                                               conserved_variables[cell][ALPHA2_RHO2_E2_INDEX]) -
                                              rhoe_0)/rhoe_0 < 1e-12,
                                     "No conservation of total energy in the relexation");
                           #else
                             conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*e1_np1;
                             conserved_variables[cell][ALPHA2_RHO2_E2_INDEX] = conserved_variables[cell][ALPHA2_RHO2_INDEX]*e2_np1;
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
                           /*--- Compute constant fields that do not change by hypothesis in the relaxation ---*/
                           const auto Z1   = rho1[cell]*c1[cell];
                           const auto Z2   = rho2[cell]*c2[cell];
                           const auto pI_0 = (Z2*p1[cell] + Z1*p2[cell])/(Z1 + Z2);

                           const auto xi1_m1_0 = 1.0/conserved_variables[cell][ALPHA1_INDEX]*
                                                 ((EOS_phase1.get_gamma() - 1.0)*pI_0 +
                                                  p1[cell] + EOS_phase1.get_gamma()*EOS_phase1.get_pi_infty());
                                                  /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           const auto xi2_m1_0 = 1.0/(1.0 - conserved_variables[cell][ALPHA1_INDEX])*
                                                 ((EOS_phase2.get_gamma() - 1.0)*pI_0 +
                                                  p2[cell] + EOS_phase2.get_gamma()*EOS_phase2.get_pi_infty());
                                                  /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           const auto rhoe_0 = conserved_variables[cell][ALPHA1_RHO1_E1_INDEX]
                                             + conserved_variables[cell][ALPHA2_RHO2_E2_INDEX];

                           // Update the volume fraction
                           conserved_variables[cell][ALPHA1_INDEX] += (p1[cell] - p2[cell])/(xi1_m1_0 + xi2_m1_0)*
                                                                      (1.0 - std::exp(-dt/tau_p));

                           // Compute the pressure difference after relaxation
                           const auto Delta_p_star = (p1[cell] - p2[cell])*std::exp(-dt/tau_p);

                           // Compute phase 2 pressure after relaxation
                           const auto p2_star = (rhoe_0 -
                                                 conserved_variables[cell][ALPHA1_INDEX]*
                                                 (Delta_p_star/(EOS_phase1.get_gamma() - 1.0) +
                                                  EOS_phase1.get_gamma()*EOS_phase1.get_pi_infty()/(EOS_phase1.get_gamma() - 1.0) +
                                                  EOS_phase1.get_q_infty()*
                                                  conserved_variables[cell][ALPHA1_RHO1_INDEX]/conserved_variables[cell][ALPHA1_INDEX]) -
                                                 (1.0 - conserved_variables[cell][ALPHA1_INDEX])*
                                                 (EOS_phase2.get_gamma()*EOS_phase2.get_pi_infty()/(EOS_phase2.get_gamma() - 1.0) +
                                                  EOS_phase2.get_q_infty()*
                                                  conserved_variables[cell][ALPHA2_RHO2_INDEX]/(1.0 - conserved_variables[cell][ALPHA1_INDEX])))/
                                                 (conserved_variables[cell][ALPHA1_INDEX]/(EOS_phase1.get_gamma() - 1.0) +
                                                  (1.0 - conserved_variables[cell][ALPHA1_INDEX])/(EOS_phase2.get_gamma() - 1.0));
                                                  /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           // Update the total energy of both phases
                           const auto e2_np1 = EOS_phase2.e_value(conserved_variables[cell][ALPHA2_RHO2_INDEX]/
                                                                  (1.0 - conserved_variables[cell][ALPHA1_INDEX]),
                                                                  p2_star); /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           conserved_variables[cell][ALPHA2_RHO2_E2_INDEX] = conserved_variables[cell][ALPHA2_RHO2_INDEX]*e2_np1;
                           conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] = rhoe_0 - conserved_variables[cell][ALPHA2_RHO2_E2_INDEX];
                         });
}

// Compute the estimate of the maximum eigenvalue for CFL condition
//
template<std::size_t dim>
double Relaxation<dim>::get_max_lambda() const {
  double local_res = 0.0;

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           for(std::size_t d = 0; d < Field::dim; ++d) {
                             local_res = std::max(std::abs(vel[cell][d]) + c[cell],
                                                  local_res);
                           }
                         });

  double global_res;
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
                           rho[cell] = conserved_variables[cell][ALPHA1_RHO1_INDEX]
                                     + conserved_variables[cell][ALPHA2_RHO2_INDEX];

                           for(std::size_t d = 0; d < Field::dim; ++d) {
                             vel[cell][d] = conserved_variables[cell][RHO_U_INDEX + d]/rho[cell];
                           }

                           /*--- Phase 1 ---*/
                           rho1[cell] = conserved_variables[cell][ALPHA1_RHO1_INDEX]/
                                        conserved_variables[cell][ALPHA1_INDEX]; /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           e1[cell]  = conserved_variables[cell][ALPHA1_RHO1_E1_INDEX]/
                                       conserved_variables[cell][ALPHA1_RHO1_INDEX]; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           de1[cell] = e1[cell] - e1_0[cell];
                           p1[cell]  = EOS_phase1.pres_value(rho1[cell], e1[cell]);
                           c1[cell]  = EOS_phase1.c_value(rho1[cell], p1[cell]);
                           T1[cell]  = EOS_phase1.T_value(rho1[cell], p1[cell]);

                           /*--- Phase 2 ---*/
                           rho2[cell] = conserved_variables[cell][ALPHA2_RHO2_INDEX]/
                                        (1.0 - conserved_variables[cell][ALPHA1_INDEX]); /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           e2[cell]  = conserved_variables[cell][ALPHA2_RHO2_E2_INDEX]/
                                       conserved_variables[cell][ALPHA2_RHO2_INDEX]; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           de2[cell] = e2[cell] - e2_0[cell];
                           p2[cell]  = EOS_phase2.pres_value(rho2[cell], e2[cell]);
                           c2[cell]  = EOS_phase2.c_value(rho2[cell], p2[cell]);
                           T2[cell]  = EOS_phase2.T_value(rho2[cell], p2[cell]);

                           alpha2[cell] = 1.0 - conserved_variables[cell][ALPHA1_INDEX];
                           Y2[cell]     = conserved_variables[cell][ALPHA2_RHO2_INDEX]/rho[cell];

                           /*--- Remaining mixture variables ---*/
                           p[cell] = conserved_variables[cell][ALPHA1_INDEX]*p1[cell]
                                   + (1.0 - conserved_variables[cell][ALPHA1_INDEX])*p2[cell];

                           c[cell] = std::sqrt((1.0 - Y2[cell])*c1[cell]*c1[cell] +
                                               Y2[cell]*c2[cell]*c2[cell]);
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
                         });

  samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, fields..., level_);
}

// Implement the function that effectively performs the temporal loop
//
template<std::size_t dim>
void Relaxation<dim>::run() {
  /*--- Default output arguemnts ---*/
  fs::path path = fs::current_path();
  #ifdef RUSANOV_FLUX
    filename = "Relaxation_Rusanov_6eqs_total_energy";
  #elifdef HLLC_FLUX
    filename = "Relaxation_HLLC_6eqs_total_energy";
  #elifdef HLLC_NON_CONS_FLUX
    filename = "Relaxation_HLLC_non_cons_6eqs_total_energy";
  #endif

  #ifdef ORDER_2
    filename = filename + "_order2";
  #else
    filename = filename + "_order1";
  #endif

  const double dt_save = Tf/static_cast<double>(nfiles);

  /*--- Auxiliary variables to save updated fields ---*/
  #ifdef ORDER_2
    auto conserved_variables_tmp = samurai::make_vector_field<typename Field::value_type, EquationData::NVARS>("conserved_tmp", mesh);
    auto conserved_variables_old = samurai::make_vector_field<typename Field::value_type, EquationData::NVARS>("conserved_old", mesh);
  #endif
  auto conserved_variables_np1 = samurai::make_vector_field<typename Field::value_type, EquationData::NVARS>("conserved_np1", mesh);

  /*--- Create the flux variables ---*/
  #ifdef RUSANOV_FLUX
    auto Rusanov_flux         = numerical_flux_cons.make_flux();
    auto NonConservative_flux = numerical_flux_non_cons.make_flux();
  #elifdef HLLC_FLUX
    auto HLLC_flux = numerical_flux.make_flux();
  #elifdef HLLC_NON_CONS_FLUX
    auto HLLC_Conservative_flux = numerical_flux_cons.make_flux();
    auto NonConservative_flux   = numerical_flux_non_cons.make_flux();
  #endif

  /*--- Save the initial condition ---*/
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(path, suffix_init,
       conserved_variables, rho, p, vel, c,
       rho1, p1, c1, T1, e1, rho2,
       p2, c2, T2, alpha2, Y2, e2);

  /*--- Save mesh size ---*/
  using mesh_id_t = typename decltype(mesh)::mesh_id_t;
  const double dx = mesh.cell_length(mesh.max_level());
  std::cout << "Number of elements = " << mesh[mesh_id_t::cells].nb_cells() << std::endl;
  std::cout << std::endl;

  /*--- Start the loop ---*/
  std::size_t nsave = 0;
  std::size_t nt    = 0;
  double t          = 0.0;
  dt                = std::min(Tf - t, cfl*dx/get_max_lambda());
  while(t != Tf) {
    t += dt;

    std::cout << fmt::format("Iteration {}: t = {}, dt = {}", ++nt, t, dt) << std::endl;

    // Save current state in case of order 2
    #ifdef ORDER_2
      conserved_variables_old = conserved_variables;
    #endif

    // Apply the numerical scheme
    samurai::update_ghost_mr(conserved_variables);
    #ifdef RUSANOV_FLUX
      auto Cons_Flux    = Rusanov_flux(conserved_variables);
      auto NonCons_Flux = NonConservative_flux(conserved_variables);

      #ifdef ORDER_2
        conserved_variables_tmp = conserved_variables - dt*Cons_Flux - dt*NonCons_Flux;
      #else
        conserved_variables_np1 = conserved_variables - dt*Cons_Flux - dt*NonCons_Flux;
      #endif
    #elifdef HLLC_FLUX
      auto Total_Flux = HLLC_flux(conserved_variables);

      #ifdef ORDER_2
        conserved_variables_tmp = conserved_variables - dt*Total_Flux;
      #else
        conserved_variables_np1 = conserved_variables - dt*Total_Flux;
      #endif
    #elifdef HLLC_NON_CONS_FLUX
      auto Cons_Flux    = HLLC_Conservative_flux(conserved_variables);
      auto NonCons_Flux = NonConservative_flux(conserved_variables);

      #ifdef ORDER_2
        conserved_variables_tmp = conserved_variables - dt*Cons_Flux - dt*NonCons_Flux;
      #else
        conserved_variables_np1 = conserved_variables - dt*Cons_Flux - dt*NonCons_Flux;
      #endif
    #endif

    #ifdef ORDER_2
      std::swap(conserved_variables.array(), conserved_variables_tmp.array());
    #else
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
      #ifdef RUSANOV_FLUX
        Cons_Flux    = Rusanov_flux(conserved_variables);
        NonCons_Flux = NonConservative_flux(conserved_variables);

        conserved_variables_tmp = conserved_variables - dt*Cons_Flux - dt*NonCons_Flux;
      #elifdef HLLC_FLUX
        Total_Flux = HLLC_flux(conserved_variables);

        conserved_variables_tmp = conserved_variables - dt*Total_Flux;
      #elifdef HLLC_NON_CONS_FLUX
        Cons_Flux    = HLLC_Conservative_flux(conserved_variables);
        NonCons_Flux = NonConservative_flux(conserved_variables);

        conserved_variables_tmp = conserved_variables - dt*Cons_Flux - dt*NonCons_Flux;
      #endif
      conserved_variables_np1 = 0.5*(conserved_variables_tmp + conserved_variables_old);
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
    if(t >= static_cast<double>(nsave + 1) * dt_save || t == Tf) {
      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";
      save(path, suffix,
           conserved_variables, rho, p, vel, c,
           rho1, p1, c1, T1, e1_0, e1, de1,
           rho2, p2, c2, T2, alpha2, Y2, e2_0, e2, de2);
    }
  }
}
