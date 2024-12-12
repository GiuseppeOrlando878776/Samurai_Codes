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
#include "Suliciu_scheme_interface.hpp"

#include "containers.hpp"

#define SULICIU_RELAXATION
//#define RUSANOV_FLUX

#ifdef SULICIU_RELAXATION
  #ifdef ORDER_2
    #undef ORDER_2
  #endif
#endif

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
            const Simulation_Paramaters& sim_param,
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

  std::size_t nfiles; // Number of files desired for output

  Field conserved_variables; // The variable which stores the conserved variables,
                             // namely the varialbes for which we solve a PDE system

  const SG_EOS<> EOS_phase1; // Equation of state of phase 1
  const SG_EOS<> EOS_phase2; // Equation of state of phase 2

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
               c2;

  Field_Vect vel1,
             vel2;

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  void init_variables(const Riemann_Parameters& Riemann_param); // Routine to initialize the variables (both conserved and auxiliary, this is problem dependent)

  void update_auxiliary_fields(); // Routine to update auxiliary fields for output and time step update

  #ifdef RUSANOV_FLUX
    double get_max_lambda() const; // Compute the estimate of the maximum eigenvalue
  #endif
};


// Implement class constructor
//
template<std::size_t dim>
BN_Solver<dim>::BN_Solver(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                          const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                          const Simulation_Paramaters& sim_param,
                          const EOS_Parameters& eos_param,
                          const Riemann_Parameters& Riemann_param):
  box(min_corner, max_corner), mesh(box, sim_param.min_level, sim_param.max_level, {false, true}),
  Tf(sim_param.Tf), cfl(sim_param.Courant), nfiles(sim_param.nfiles),
  EOS_phase1(eos_param.gamma_1, eos_param.pi_infty_1, eos_param.q_infty_1),
  EOS_phase2(eos_param.gamma_2, eos_param.pi_infty_2, eos_param.q_infty_2),
  #ifdef SULICIU_RELAXATION
    numerical_flux()
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
  conserved_variables = samurai::make_field<double, EquationData::NVARS>("conserved", mesh);

  rho  = samurai::make_field<double, 1>("rho", mesh);
  p    = samurai::make_field<double, 1>("p", mesh);

  rho1 = samurai::make_field<double, 1>("rho1", mesh);
  p1   = samurai::make_field<double, 1>("p1", mesh);
  c1   = samurai::make_field<double, 1>("c1", mesh);

  rho2 = samurai::make_field<double, 1>("rho2", mesh);
  p2   = samurai::make_field<double, 1>("p2", mesh);
  c2   = samurai::make_field<double, 1>("c2", mesh);

  vel1 = samurai::make_field<double, dim>("vel1", mesh);
  vel2 = samurai::make_field<double, dim>("vel2", mesh);

  // Initialize the fields with a loop over all cells
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           const auto center = cell.center();
                           const double x    = center[0];

                           if(x <= Riemann_param.xd) {
                             conserved_variables[cell][ALPHA1_INDEX] = Riemann_param.alpha1L;

                             rho1[cell]    = Riemann_param.rho1L;
                             vel1[cell][0] = Riemann_param.u1L;
                             vel1[cell][1] = Riemann_param.v1L;
                             p1[cell]      = Riemann_param.p1L;

                             rho2[cell]    = Riemann_param.rho2L;
                             vel2[cell][0] = Riemann_param.u2L;
                             vel2[cell][1] = Riemann_param.v2L;
                             p2[cell]      = Riemann_param.p2L;
                           }
                           else {
                             conserved_variables[cell][ALPHA1_INDEX] = Riemann_param.alpha1R;

                             rho1[cell]    = Riemann_param.rho1R;
                             vel1[cell][0] = Riemann_param.u1R;
                             vel1[cell][1] = Riemann_param.v1R;
                             p1[cell]      = Riemann_param.p1R;

                             rho2[cell]    = Riemann_param.rho2R;
                             vel2[cell][0] = Riemann_param.u2R;
                             vel2[cell][1] = Riemann_param.v2R;
                             p2[cell]      = Riemann_param.p2R;
                           }

                           conserved_variables[cell][ALPHA1_RHO1_INDEX] = conserved_variables[cell][ALPHA1_INDEX]*rho1[cell];
                           for(std::size_t d = 0; d < dim; ++d) {
                             conserved_variables[cell][ALPHA1_RHO1_U1_INDEX + d] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*vel1[cell][d];
                           }
                           const auto e1 = EOS_phase1.e_value_RhoP(rho1[cell], p1[cell]);
                           conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*e1;
                           for(std::size_t d = 0; d < dim; ++d) {
                             conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] += conserved_variables[cell][ALPHA1_RHO1_INDEX]*
                                                                                (0.5*vel1[cell][d]*vel1[cell][d]);
                           }

                           conserved_variables[cell][ALPHA2_RHO2_INDEX] = (1.0 - conserved_variables[cell][ALPHA1_INDEX])*rho2[cell];
                           for(std::size_t d = 0; d < dim; ++d) {
                             conserved_variables[cell][ALPHA2_RHO2_U2_INDEX + d] = conserved_variables[cell][ALPHA2_RHO2_INDEX]*vel2[cell][d];
                           }
                           const auto e2 = EOS_phase2.e_value_RhoP(rho2[cell], p2[cell]);
                           conserved_variables[cell][ALPHA2_RHO2_E2_INDEX] = conserved_variables[cell][ALPHA2_RHO2_INDEX]*e2;
                           for(std::size_t d = 0; d < dim; ++d) {
                             conserved_variables[cell][ALPHA2_RHO2_E2_INDEX] += conserved_variables[cell][ALPHA2_RHO2_INDEX]*
                                                                                (0.5*vel2[cell][d]*vel2[cell][d]);
                           }

                           c1[cell] = EOS_phase1.c_value_RhoP(rho1[cell], p1[cell]);

                           c2[cell] = EOS_phase2.c_value_RhoP(rho2[cell], p2[cell]);

                           rho[cell] = conserved_variables[cell][ALPHA1_RHO1_INDEX]
                                     + conserved_variables[cell][ALPHA2_RHO2_INDEX];

                           p[cell] = conserved_variables[cell][ALPHA1_INDEX]*p1[cell]
                                   + (1.0 - conserved_variables[cell][ALPHA1_INDEX])*p2[cell];
                         });

  samurai::make_bc<samurai::Neumann<1>>(conserved_variables, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
}

// Compute the estimate of the maximum eigenvalue for CFL condition
//
#ifdef RUSANOV_FLUX
  template<std::size_t dim>
  double BN_Solver<dim>::get_max_lambda() const {
    double res = 0.0;

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                             res = std::max(std::max(std::max(std::abs(vel1[cell][0]) + c1[cell],
                                                              std::abs(vel2[cell][0]) + c2[cell]),
                                                     std::max(std::abs(vel1[cell][1]) + c1[cell],
                                                              std::abs(vel2[cell][1]) + c2[cell])),
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

  rho2.resize();
  p2.resize();
  c2.resize();
  vel2.resize();

  // Loop over all cells
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           // Compute the fields
                           rho[cell] = conserved_variables[cell][ALPHA1_RHO1_INDEX]
                                     + conserved_variables[cell][ALPHA2_RHO2_INDEX];

                           rho1[cell] = conserved_variables[cell][ALPHA1_RHO1_INDEX]/
                                        conserved_variables[cell][ALPHA1_INDEX]; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           for(std::size_t d = 0; d < dim; ++d) {
                             vel1[cell][d] = conserved_variables[cell][ALPHA1_RHO1_U1_INDEX + d]/
                                             conserved_variables[cell][ALPHA1_RHO1_INDEX];
                             /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           }
                           auto e1 = conserved_variables[cell][ALPHA1_RHO1_E1_INDEX]/
                                     conserved_variables[cell][ALPHA1_RHO1_INDEX]; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           for(std::size_t d = 0; d < dim; ++d) {
                             e1 -= 0.5*vel1[cell][d]*vel1[cell][d];
                           }
                           p1[cell] = EOS_phase1.pres_value_Rhoe(rho1[cell], e1);
                           c1[cell] = EOS_phase1.c_value_RhoP(rho1[cell], p1[cell]);

                           rho2[cell] = conserved_variables[cell][ALPHA2_RHO2_INDEX]/
                                        (1.0 - conserved_variables[cell][ALPHA1_INDEX]); /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           for(std::size_t d = 0; d < dim; ++d) {
                             vel2[cell][d] = conserved_variables[cell][ALPHA2_RHO2_U2_INDEX + d]/
                                             conserved_variables[cell][ALPHA2_RHO2_INDEX];
                             /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           }
                           auto e2 = conserved_variables[cell][ALPHA2_RHO2_E2_INDEX]/
                                     conserved_variables[cell][ALPHA2_RHO2_INDEX]; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                           for(std::size_t d = 0; d < dim; ++d) {
                             e2 -= 0.5*vel2[cell][d]*vel2[cell][d];
                           }
                           p2[cell] = EOS_phase2.pres_value_Rhoe(rho2[cell], e2);
                           c2[cell] = EOS_phase2.c_value_RhoP(rho2[cell], p2[cell]);

                           p[cell] = conserved_variables[cell][ALPHA1_INDEX]*p1[cell]
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
  save(path, filename, suffix_init, conserved_variables, rho, p, vel1, rho1, p1, c1, vel2, rho2, p2, c2);

  /*--- Set mesh size ---*/
  using mesh_id_t = typename decltype(mesh)::mesh_id_t;
  const double dx = samurai::cell_length(mesh[mesh_id_t::cells].max_level());

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
    samurai::update_bc(conserved_variables);
    #ifdef SULICIU_RELAXATION
      c = 0.0;
      auto Relaxation_Flux = Suliciu_flux(conserved_variables);
      const double dt = std::min(Tf - t, cfl*dx/c);
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
      const double dt = std::min(Tf - t, cfl*dx/get_max_lambda());
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

    // Consider the second stage for the second order
    #ifdef ORDER_2
      samurai::update_ghost_mr(conserved_variables);
      samurai::update_bc(conserved_variables);
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
    #endif

    // Save the results
    if(t >= static_cast<double>(nsave + 1)*dt_save || t == Tf) {
      update_auxiliary_fields();

      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";
      save(path, filename, suffix, conserved_variables, rho, p, vel1, rho1, p1, c1, vel2, rho2, p2, c2);
    }
  }
}