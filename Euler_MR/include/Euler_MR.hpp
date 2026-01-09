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
#include <samurai/io/restart.hpp>
#include <samurai/io/hdf5.hpp>

#include <filesystem>
namespace fs = std::filesystem;

/*--- Add header files for the multiresolution ---*/
#include <samurai/mr/adapt.hpp>
#include "prediction.hpp"

/*--- Add header with auxiliary structs ---*/
#include "containers.hpp"

/*--- Include the header with the numerical fluxes ---*/
#include "schemes.hpp"

/*--- Define preprocessor to check whether to control data or not ---*/
#define VERBOSE

/*--- Specify the use of this namespace where we just store the indices ---*/
using namespace EquationData;

#define assertm(exp, msg) assert(((void)msg, exp))

/** This is the class for the simulation for the Euler equations
 */
template<std::size_t dim>
class Euler_MR {
public:
  using Config = samurai::MRConfig<dim, 2, 2, 1>;
  using Field  = samurai::VectorField<samurai::MRMesh<Config>,
                                      double,
                                      EquationData::NVARS,
                                      false>;
  using Number = typename Field::value_type;

  Euler_MR() = default; /*--- Default constructor. This will do nothing
                              and basically will never be used ---*/

  Euler_MR(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
           const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
           const Simulation_Paramaters<Number>& sim_param,
           const EOS_Parameters<Number>& eos_param,
           const Riemann_Parameters<Number>& Riemann_param); /*--- Class constrcutor with the arguments related
                                                                   to the grid and to the physics ---*/

  void run(const std::size_t nfiles = 10); /*--- Function which actually executes the temporal loop ---*/

  template<class... Variables>
  void save(const std::string& suffix,
            const Variables&... fields); /*--- Routine to save the results ---*/

private:
  /*--- Now we declare some relevant variables ---*/
  const samurai::Box<double, dim> box;

  samurai::MRMesh<Config> mesh; /*--- Variable to store the mesh ---*/

  using Field_Scalar = samurai::ScalarField<decltype(mesh), Number>;
  using Field_Vect   = samurai::VectorField<decltype(mesh), Number, dim, false>;

  const Number t0; /*--- Initial time of the simulation ---*/
  const Number Tf; /*--- Final time of the simulation ---*/

  Number cfl; /*--- Courant number of the simulation so as to compute the time step ---*/

  double MR_param;      /*--- Multiresolution parameter ---*/
  double MR_regularity; /*--- Multiresolution regularity ---*/

  const SG_EOS<Number> Euler_EOS; /*--- Equation of state ---*/

  std::unique_ptr<samurai::Flux<Field>> numerical_flux; /*--- variable to compute the numerical flux
                                                              (this is necessary to call 'make_flux') ---*/

  fs::path    path;     /*--- Auxiliary variable to store the output directory ---*/
  std::string filename; /*--- Auxiliary variable to store the name of output ---*/

  Field conserved_variables; /*--- The variable which stores the conserved variables,
                                   namely the varialbes for which we solve a PDE system ---*/

  /*--- Now we declare a bunch of fields which depend from the state,
        but it is useful to have it for the output ---*/
  Field_Scalar p,
               c;

  Field_Vect vel;

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  void create_fields(); /*--- Auxiliary routine to initialize the fileds to the mesh ---*/

  void init_variables(const Riemann_Parameters<Number>& Riemann_param); /*--- Routine to initialize the variables
                                                                              (both conserved and auxiliary, this is problem dependent) ---*/

  void apply_bcs(const Riemann_Parameters<Number>& Riemann_param); /*--- Auxiliary routine for the boundary conditions ---*/

  void update_auxiliary_fields(); /*--- Routine to update auxiliary fields for output and time step update ---*/

  Number get_max_lambda() const; /*--- Compute the estimate of the maximum eigenvalue ---*/

  void check_data(unsigned flag = 0); /*--- Auxiliary routine to check if (small) spurious negative values are present ---*/
};

//////////////////////////////////////////////////////////////
/*---- START WITH THE IMPLEMENTATION OF THE CONSTRUCTOR ---*/
//////////////////////////////////////////////////////////////

// Implement class constructor
//
template<std::size_t dim>
Euler_MR<dim>::Euler_MR(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                        const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                        const Simulation_Paramaters<Number>& sim_param,
                        const EOS_Parameters<Number>& eos_param,
                        const Riemann_Parameters<Number>& Riemann_param):
  box(min_corner, max_corner),
  t0(sim_param.t0), Tf(sim_param.Tf), cfl(sim_param.Courant),
  MR_param(sim_param.MR_param), MR_regularity(sim_param.MR_regularity),
  Euler_EOS(eos_param.gamma, eos_param.pi_infty, eos_param.q_infty),
  path(sim_param.save_dir)
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
                                                  vel, p, c);
    }

    /*--- Apply boundary conditions ---*/
    apply_bcs(Riemann_param);

    /*--- Initiliaze the numerical flux ---*/
    numerical_flux = get_numerical_flux<Field>(sim_param.flux_name, Euler_EOS);
    numerical_flux->set_flux_name(sim_param.flux_name);
  }

// Auxiliary routine to create the fields
//
template<std::size_t dim>
void Euler_MR<dim>::create_fields() {
  conserved_variables = samurai::make_vector_field<Number, Field::n_comp>("conserved", mesh);

  p   = samurai::make_scalar_field<Number>("p", mesh);
  c   = samurai::make_scalar_field<Number>("c", mesh);
  vel = samurai::make_vector_field<Number, dim>("vel", mesh);
}

// Initialization of conserved and auxiliary variables
//
template<std::size_t dim>
void Euler_MR<dim>::init_variables(const Riemann_Parameters<Number>& Riemann_param) {
  /*--- Resize the fields since now mesh has been created ---*/
  conserved_variables.resize();
  p.resize();
  c.resize();
  vel.resize();

  /*--- Set the initial state ---*/
  const auto xd = Riemann_param.xd;

  // Initialize the fields with a loop over all cells
  const auto velL = Riemann_param.uL;

  const auto pL   = Riemann_param.pL;
  const auto rhoL = Riemann_param.rhoL;

  const auto velR = Riemann_param.uR;

  const auto pR   = Riemann_param.pR;
  const auto rhoR = Riemann_param.rhoR;

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              const auto center = cell.center();
                              const auto x      = static_cast<Number>(center[0]);

                              // Left state (primitive variables)
                              if(x <= xd) {
                                conserved_variables[cell][RHO_INDEX] = rhoL;

                                vel[cell][0] = velL;
                                p[cell]      = pL;
                              }
                              // Right state (primitive variables)
                              else {
                                conserved_variables[cell][RHO_INDEX] = rhoR;

                                vel[cell][0] = velR;
                                p[cell]      = pR;
                              }

                              // Complete the conserved variables (and some auxiliary fields for the sake of completeness)
                              auto norm2_vel_loc = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                conserved_variables[cell][RHOU_INDEX + d] = conserved_variables[cell][RHO_INDEX]*vel[cell][d];
                                norm2_vel_loc += vel[cell][d]*vel[cell][d];
                              }

                              const auto e_loc = Euler_EOS.e_value(conserved_variables[cell][RHO_INDEX], p[cell]);
                              conserved_variables[cell][RHOE_INDEX] = conserved_variables[cell][RHO_INDEX]*
                                                                      (e_loc + static_cast<Number>(0.5)*norm2_vel_loc);

                              c[cell] = Euler_EOS.c_value(conserved_variables[cell][RHO_INDEX], p[cell]);
                            }
                        );
}

// Auxiliary routine to impose the boundary conditions
//
template<std::size_t dim>
void Euler_MR<dim>::apply_bcs(const Riemann_Parameters<Number>& Riemann_param) {
  const xt::xtensor_fixed<int, xt::xshape<1>> left  = {-1};
  const xt::xtensor_fixed<int, xt::xshape<1>> right = {1};

  samurai::make_bc<samurai::Dirichlet<1>>(conserved_variables,
                                          Riemann_param.rhoL,
                                          Riemann_param.rhoL*Riemann_param.uL,
                                          Riemann_param.rhoL*
                                          (Euler_EOS.e_value(Riemann_param.rhoL, Riemann_param.pL) +
                                           static_cast<Number>(0.5)*Riemann_param.uL*Riemann_param.uL))->on(left);
  samurai::make_bc<samurai::Dirichlet<1>>(conserved_variables,
                                          Riemann_param.rhoR,
                                          Riemann_param.rhoR*Riemann_param.uR,
                                          Riemann_param.rhoR*
                                          (Euler_EOS.e_value(Riemann_param.rhoR, Riemann_param.pR) +
                                           static_cast<Number>(0.5)*Riemann_param.uR*Riemann_param.uR))->on(right);
}

//////////////////////////////////////////////////////////////
/*---- FOCUS NOW ON THE AUXILIARY FUNCTIONS ---*/
/////////////////////////////////////////////////////////////

// Auxiliary routine to check if spurious negative values arise
//
template<std::size_t dim>
void Euler_MR<dim>::check_data(unsigned flag) {
  /*--- Recompute data so as to save the whole state in case of diverging solution ---*/
  update_auxiliary_fields();

  /*--- Check data ---*/
  std::string op;
  if(flag == 0) {
    op = "after hyperbolic operator";
  }
  else {
    op = "after mesh adaptation";
  }
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Sanity check for the density
                              if(conserved_variables[cell][RHO_INDEX] < static_cast<Number>(0.0)) {
                                std::cerr << "Negative density " + op << std::endl;
                                save("_diverged", conserved_variables,
                                                  vel, p, c);
                                exit(1);
                              }
                              else if(std::isnan(conserved_variables[cell][RHO_INDEX])) {
                                std::cerr << "NaN density " + op << std::endl;
                                save("_diverged", conserved_variables,
                                                  vel, p, c);
                                exit(1);
                              }

                              // Sanity check for the speed of sound
                              if(std::isnan(c[cell])) {
                                std::cerr << "NaN speed of sound " + op << std::endl;
                                save("_diverged", conserved_variables,
                                                  vel, p, c);
                                exit(1);
                              }
                            }
                        );
}

// Compute the estimate of the maximum eigenvalue for CFL condition
//
template<std::size_t dim>
typename Euler_MR<dim>::Number Euler_MR<dim>::get_max_lambda() const {
  auto local_res = static_cast<Number>(0.0);

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              for(std::size_t d = 0; d < dim; ++d) {
                                local_res = std::max(std::abs(vel[cell][d]) + c[cell], local_res);
                              }
                            });

  double local_res_d = static_cast<double>(local_res);
  double global_res;
  MPI_Allreduce(&local_res_d, &global_res, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  return static_cast<Number>(global_res);
}

// Update auxiliary fields after solution of the system
//
template<std::size_t dim>
void Euler_MR<dim>::update_auxiliary_fields() {
  vel.resize();
  p.resize();
  c.resize();

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              const auto rho_loc     = conserved_variables[cell][RHO_INDEX];
                              const auto inv_rho_loc = static_cast<Number>(1.0)/rho_loc;
                              auto norm2_vel_loc     = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                const auto vel_d_loc = conserved_variables[cell][RHOU_INDEX + d]*inv_rho_loc;
                                vel[cell][d]         = vel_d_loc;
                                norm2_vel_loc        += vel_d_loc*vel_d_loc;
                              }
                              auto e_loc       = conserved_variables[cell][RHOE_INDEX]*inv_rho_loc
                                               - static_cast<Number>(0.5)*norm2_vel_loc;
                              const auto p_loc = Euler_EOS.pres_value(rho_loc, e_loc);
                              p[cell]          = p_loc;
                              c[cell]          = Euler_EOS.c_value(rho_loc, p_loc);
                            }
                        );
}

// Save desired fields and info
//
template<std::size_t dim>
template<class... Variables>
void Euler_MR<dim>::save(const std::string& suffix,
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
/*---- IMPLEMENT THE FUNCTION THAT EFFECTIVELY SOLVES THE PROBLEM ---*/
/////////////////////////////////////////////////////////////

// Implement the function that effectively performs the temporal loop
//
template<std::size_t dim>
void Euler_MR<dim>::run(const std::size_t nfiles) {
  /*--- Default output arguemnts ---*/
  filename = "Euler_MR_" + numerical_flux->get_flux_name();

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

  /*--- Create the flux variable ---*/
  auto Discrete_flux = numerical_flux->make_flux();

  /*--- Save the initial condition ---*/
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(suffix_init, conserved_variables,
                    p, vel, c);

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

  /*--- Declare operators for multiresolution ---*/
  auto prediction_fn = [&](auto& new_field, const auto& old_field) {
    return make_field_operator_function<Euler_prediction_op>(new_field, old_field, Euler_EOS);
  };
  //update_ghost_mr(c);
  auto MRadaptation = samurai::make_MRAdapt(prediction_fn, c);
  auto mra_config   = samurai::mra_config();
  mra_config.epsilon(MR_param);
  mra_config.regularity(MR_regularity);

  /*--- Start the loop ---*/
  std::size_t nsave = 0;
  std::size_t nt    = 0;
  auto t            = static_cast<Number>(t0);
  auto dt           = std::min(Tf - t, cfl*dx/get_max_lambda());
  while(t != Tf) {
    t += dt;

    if(rank == 0) {
      std::cout << fmt::format("Iteration {}: t = {}, dt = {}", ++nt, t, dt) << std::endl;
    }

    // Apply mesh adaptation
    //update_ghost_mr(conserved_variables);
    MRadaptation(mra_config, conserved_variables);
    #ifdef VERBOSE
      check_data(1);
    #endif

    // Save current state in case of order 2
    #ifdef ORDER_2
      conserved_variables_old.resize();
      conserved_variables_old = conserved_variables;
    #endif

    // Apply the numerical scheme
    try {
      #ifdef ORDER_2
        conserved_variables_tmp.resize();
        conserved_variables_tmp = conserved_variables - dt*Discrete_flux(conserved_variables);
        samurai::swap(conserved_variables, conserved_variables_tmp);
      #else
        conserved_variables_np1.resize();
        conserved_variables_np1 = conserved_variables - dt*Discrete_flux(conserved_variables);
        samurai::swap(conserved_variables, conserved_variables_np1);
      #endif
    }
    catch(const std::exception& e) {
      std::cerr << e.what() << std::endl;
      save("_diverged", conserved_variables,
                        vel, p, c);
      exit(1);
    }
    #ifdef VERBOSE
      check_data();
    #endif

    // Consider the second stage for the second order
    #ifdef ORDER_2
      // Apply the numerical scheme
      try {
        conserved_variables_tmp = conserved_variables - dt*Discrete_flux(conserved_variables);
        conserved_variables_np1 = static_cast<Number>(0.5)*
                                  (conserved_variables_tmp + conserved_variables_old);
        samurai::swap(conserved_variables, conserved_variables_np1);
      }
      catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
        save("_diverged", conserved_variables,
                          vel, p, c);
        exit(1);
      }
      #ifdef VERBOSE
        check_data();
      #endif
    #endif

    // Compute updated time step
    #if !defined VERBOSE
      update_auxiliary_fields();
    #endif
    dt = std::min(Tf - t, cfl*dx/get_max_lambda());

    // Save the results
    if(t >= static_cast<Number>(nsave + 1)*dt_save || t == Tf) {
      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";
      save(suffix, conserved_variables,
                   p, vel, c);
    }
  }
}
