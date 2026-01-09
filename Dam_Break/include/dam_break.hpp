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

/*--- Add header with auxiliary structs ---*/
#include "containers.hpp"

/*--- Include the header with the custom bundary condition ---*/
#include "user_bc.hpp"

/*--- Include the header with the numerical fluxes ---*/
#include "schemes.hpp"

/*--- Define preprocessor to check whether to control data or not ---*/
#define VERBOSE

/*--- Specify the use of this namespace where we just store the indices ---*/
using namespace EquationData;

/*--- Define preprocessor to choose whether gravity has to be tretaed implicitly or not ---*/
//#define GRAVITY_IMPLICIT

/** This is the class for the simulation of a model
 *  for the dam-break problem
 **/
template<std::size_t dim>
class DamBreak {
public:
  using Config = samurai::MRConfig<dim, 2, 1, 0>;
  using Field  = samurai::VectorField<samurai::MRMesh<Config>,
                                      double,
                                      EquationData::NVARS,
                                      false>;
  using Number = typename Field::value_type;

  DamBreak() = default; /*--- Default constructor. This will do nothing
                              and basically will never be used ---*/

  DamBreak(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
           const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
           const Simulation_Paramaters<Number>& sim_param,
           const EOS_Parameters<Number>& eos_param); /*--- Class constrcutor with the arguments related
                                                           to the grid and to the physics ---*/

  void run(const std::size_t n_files = 10); /*--- Function which actually executes the temporal loop ---*/

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

  bool apply_relax; /*--- Choose whether to apply or not the relaxation ---*/

  Number cfl; /*--- Courant number of the simulation so as to compute the time step ---*/

  const Number      lambda;           /*--- Parameter for bound preserving strategy ---*/
  const Number      atol_Newton;      /*--- Absolute tolerance Newton method relaxation ---*/
  const Number      rtol_Newton;      /*--- Relative tolerance Newton method relaxation ---*/
  const std::size_t max_Newton_iters; /*--- Maximum number of Newton iterations ---*/

  double MR_param;      /*--- Multiresolution parameter ---*/
  double MR_regularity; /*--- Multiresolution regularity ---*/

  LinearizedBarotropicEOS<Number> EOS_phase1,
                                  EOS_phase2; /*--- The two variables which take care of the
                                                    barotropic EOS to compute the speed of sound ---*/

  std::unique_ptr<samurai::Flux<Field>> numerical_flux; /*--- variable to compute the numerical flux
                                                              (this is necessary to call 'make_flux') ---*/

  fs::path    path;     /*--- Auxiliary variable to store the output directory ---*/
  std::string filename; /*--- Auxiliary variable to store the name of output ---*/

  Field conserved_variables; /*--- The variable which stores the conserved variables,
                                   namely the varialbes for which we solve a PDE system ---*/

  /*--- Now we declare a bunch of fields which depend from the state, but it is useful
        to have it so as to avoid recomputation ---*/
  Field_Scalar alpha1,
               dalpha1,
               p1,
               p2,
               p,
               rho;

  Field_Vect   vel,
               grad_alpha1;

  using gradient_type = decltype(samurai::make_gradient_order2<decltype(alpha1)>());
  gradient_type gradient;

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  void create_fields(); /*--- Auxiliary routine to initialize the fileds to the mesh ---*/

  void init_variables(const Number L0,
                      const Number H0,
                      const Number W0); /*--- Routine to initialize the variables
                                              (both conserved and auxiliary, this is problem dependent) ---*/

  void apply_bcs(); /*--- Auxiliary routine for the boundary conditions ---*/

  Number get_max_lambda(); /*--- Compute the estimate of the maximum eigenvalue ---*/

  void check_data(unsigned flag = 0); /*--- Auxiliary routine to check if (small) spurious negative values are present ---*/

  void perform_mesh_adaptation(); /*--- Perform the mesh adaptation ---*/

  void apply_relaxation(); /*--- Apply the relaxation ---*/

  template<typename State>
  void perform_Newton_step_relaxation(State local_conserved_variables,
                                      Number& dalph1_loc,
                                      Number& alpha1_loc,
                                      bool& local_relaxation_applied); /*--- Single iteration of the Newton method for the relaxation ---*/
};

//////////////////////////////////////////////////////////////
/*---- START WITH THE IMPLEMENTATION OF THE CONSTRUCTOR ---*/
//////////////////////////////////////////////////////////////

// Implement class constructor
//
template<std::size_t dim>
DamBreak<dim>::DamBreak(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                        const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                        const Simulation_Paramaters<Number>& sim_param,
                        const EOS_Parameters<Number>& eos_param):
  box(min_corner, max_corner),
  t0(sim_param.t0), Tf(sim_param.Tf), apply_relax(sim_param.apply_relaxation),
  cfl(sim_param.Courant),
  lambda(sim_param.lambda), atol_Newton(sim_param.atol_Newton),
  rtol_Newton(sim_param.rtol_Newton), max_Newton_iters(sim_param.max_Newton_iters),
  MR_param(sim_param.MR_param), MR_regularity(sim_param.MR_regularity),
  EOS_phase1(eos_param.p0_phase1, eos_param.rho0_phase1, eos_param.c0_phase1),
  EOS_phase2(eos_param.p0_phase2, eos_param.rho0_phase2, eos_param.c0_phase2),
  path(sim_param.save_dir),
  gradient(samurai::make_gradient_order2<decltype(alpha1)>())
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
      mesh = {box, sim_param.min_level, sim_param.max_level, {{false, false, false}}};
      init_variables(sim_param.L0, sim_param.H0, sim_param.W0);
    }
    else {
      samurai::load(sim_param.restart_file, mesh, conserved_variables,
                                                  alpha1, rho, p1, p2, p, vel);
    }

    /*--- Apply boundary conditions ---*/
    apply_bcs();

    /*--- Initiliaze the numerical flux ---*/
    numerical_flux = get_numerical_flux<Field>(sim_param.flux_name,
                                               EOS_phase1, EOS_phase2,
                                               lambda, atol_Newton, rtol_Newton, max_Newton_iters);
    numerical_flux->set_flux_name(sim_param.flux_name);
  }

// Auxiliary routine to create the fields
//
template<std::size_t dim>
void DamBreak<dim>::create_fields() {
  conserved_variables = samurai::make_vector_field<Number, Field::n_comp>("conserved", mesh);

  alpha1  = samurai::make_scalar_field<Number>("alpha1", mesh);

  grad_alpha1 = samurai::make_vector_field<Number, dim>("grad_alpha1", mesh);

  dalpha1 = samurai::make_scalar_field<Number>("dalpha1", mesh);

  p1      = samurai::make_scalar_field<Number>("p1", mesh);
  p2      = samurai::make_scalar_field<Number>("p2", mesh);
  p       = samurai::make_scalar_field<Number>("p", mesh);
  rho     = samurai::make_scalar_field<Number>("rho", mesh);

  vel     = samurai::make_vector_field<Number, dim>("u", mesh);
}

// Initialization of conserved and auxiliary variables
//
template<std::size_t dim>
void DamBreak<dim>::init_variables(const Number L0,
                                   const Number H0,
                                   const Number W0) {
  /*--- Resize the fields since now mesh has been created ---*/
  conserved_variables.resize();
  alpha1.resize();
  grad_alpha1.resize();
  dalpha1.resize();
  p1.resize();
  p2.resize();
  p.resize();
  rho.resize();
  vel.resize();

  /*--- Initialize the fields with a loop over all cells ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              const auto center = cell.center();
                              const auto x      = static_cast<Number>(center[0]);
                              const auto y      = static_cast<Number>(center[1]);
                              const auto z      = static_cast<Number>(center[2]);

                              // Set volume fraction
                              if(x < L0 && y < H0 && z < W0) {
                                alpha1[cell] = static_cast<Number>(1.0)
                                             - static_cast<Number>(1e-6);
                              }
                              else {
                                alpha1[cell] = static_cast<Number>(1e-6);
                              }

                              // Set mass phase 1
                              p1[cell] = EOS_phase1.get_p0();
                              conserved_variables[cell][M1_INDEX] = alpha1[cell]*EOS_phase1.rho_value(p1[cell]);

                              // Set mass phase 2
                              p2[cell] = EOS_phase2.get_p0();
                              conserved_variables[cell][M2_INDEX] = (static_cast<Number>(1.0) - alpha1[cell])*
                                                                    EOS_phase2.rho_value(p2[cell]);

                              // Set conserved variable associated to volume fraction
                              rho[cell] = conserved_variables[cell][M1_INDEX]
                                        + conserved_variables[cell][M2_INDEX];

                              conserved_variables[cell][RHO_ALPHA1_INDEX] = rho[cell]*alpha1[cell];

                              // Set momentum
                              for(std::size_t d = 0; d < Field::dim; ++ d) {
                                vel[cell][d] = 0.0;
                                conserved_variables[cell][RHO_U_INDEX + d] = rho[cell]*vel[cell][d];
                              }

                              // Set mixture pressure for output
                              p[cell] = alpha1[cell]*p1[cell]
                                      + (static_cast<Number>(1.0) - alpha1[cell])*p2[cell];
                            }
                        );
}

// Auxiliary routine to impose the boundary conditions
//
template<std::size_t dim>
void DamBreak<dim>::apply_bcs() {
  /*--- Consider non-reflecting bcs ---*/
  samurai::make_bc<Default>(conserved_variables, NonReflecting(conserved_variables));
}

//////////////////////////////////////////////////////////////
/*---- FOCUS NOW ON THE AUXILIARY FUNCTIONS ---*/
/////////////////////////////////////////////////////////////

// Compute the estimate of the maximum eigenvalue for CFL condition
//
template<std::size_t dim>
typename DamBreak<dim>::Number DamBreak<dim>::get_max_lambda() {
  auto local_res = static_cast<Number>(0.0);

  std::array<Number, dim> local_vel;

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Pre-fetch some variables in order to exploit possible vectorization
                              const auto m1_loc = conserved_variables[cell][M1_INDEX];
                              const auto m2_loc = conserved_variables[cell][M2_INDEX];

                              // Compute the volume fraction and velocity along all the directions
                              const auto rho_loc     = m1_loc + m2_loc;
                              const auto inv_rho_loc = static_cast<Number>(1.0)/rho_loc;
                              const auto alpha1_loc  = conserved_variables[cell][RHO_ALPHA1_INDEX]*inv_rho_loc;
                              for(std::size_t d = 0; d < dim; ++d) {
                                local_vel[d] = conserved_variables[cell][RHO_U_INDEX + d]*inv_rho_loc;
                              }

                              // Compute frozen speed of sound
                              const auto rho1_loc = m1_loc/alpha1_loc;
                              /*--- TODO: Add a check in case of zero volume fraction ---*/
                              const auto rho2_loc = m2_loc/(static_cast<Number>(1.0) - alpha1_loc);
                              /*--- TODO: Add a check in case of zero volume fraction ---*/
                              const auto Y1_loc   = m1_loc*inv_rho_loc;
                              const auto c        = std::sqrt(Y1_loc*
                                                              EOS_phase1.c_value(rho1_loc)*
                                                              EOS_phase1.c_value(rho1_loc) +
                                                              (static_cast<Number>(1.0) - Y1_loc)*
                                                              EOS_phase2.c_value(rho2_loc)*
                                                              EOS_phase2.c_value(rho2_loc));

                              // Update eigenvalue estimate
                              for(std::size_t d = 0; d < dim; ++d) {
                                local_res = std::max(std::abs(local_vel[d]) + c, local_res);
                              }
                            }
                        );

  const double local_res_d = static_cast<double>(local_res);
  double global_res;
  MPI_Allreduce(&local_res_d, &global_res, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  return static_cast<Number>(global_res);
}

// Auxiliary routine to check if spurious negative values arise
//
template<std::size_t dim>
void DamBreak<dim>::check_data(unsigned flag) {
  std::string op;
  if(flag == 0) {
    op = "at the beginning of the relaxation";
  }
  else {
    op = "after mesh adaptation";
  }

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Start with the volume fraction
                              alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/
                                             (conserved_variables[cell][M1_INDEX] +
                                              conserved_variables[cell][M2_INDEX]);
                              if(alpha1[cell] < static_cast<Number>(0.0)) {
                                std::cerr << "Negative volume fraction for phase 1 " + op << std::endl;
                                save("_diverged", conserved_variables,
                                                  alpha1);
                                exit(1);
                              }
                              else if(alpha1[cell] > static_cast<Number>(1.0)) {
                                std::cerr << "Exceeding volume fraction for phase 1 " + op << std::endl;
                                save("_diverged", conserved_variables,
                                                  alpha1);
                                exit(1);
                              }
                              else if(std::isnan(alpha1[cell])) {
                                std::cerr << "NaN volume fraction for phase 1 " + op << std::endl;
                                save("_diverged", conserved_variables,
                                                  alpha1);
                                exit(1);
                              }

                              // Sanity check for m1
                              if(conserved_variables[cell][M1_INDEX] < static_cast<Number>(0.0)) {
                                std::cerr << "Negative mass for phase 1 " + op << std::endl;
                                save("_diverged", conserved_variables,
                                                  alpha1);
                                exit(1);
                              }
                              else if(std::isnan(conserved_variables[cell][M1_INDEX])) {
                                std::cerr << "NaN mass for phase 1 " + op << std::endl;
                                save("_diverged", conserved_variables,
                                                  alpha1);
                                exit(1);
                              }

                              // Sanity check for m2
                              if(conserved_variables[cell][M2_INDEX] < static_cast<Number>(0.0)) {
                                std::cerr << "Negative mass for phase 2 " + op << std::endl;
                                save("_diverged", conserved_variables,
                                                  alpha1);
                                exit(1);
                              }
                              else if(std::isnan(conserved_variables[cell][M2_INDEX])){
                                std::cerr << "NaN mass for phase 2 " + op << std::endl;
                                save("_diverged", conserved_variables,
                                                  alpha1);
                                exit(1);
                              }
                            }
                        );
}

//////////////////////////////////////////////////////////////
/*---- FOCUS NOW ON THE RELAXATION FUNCTIONS ---*/
/////////////////////////////////////////////////////////////

// Apply the relaxation. This procedure is valid for a generic EOS
//
template<std::size_t dim>
void DamBreak<dim>::apply_relaxation() {
  samurai::times::timers.start("apply_relaxation");

  /*--- Loop of Newton method. ---*/
  std::size_t Newton_iter = 0;
  bool global_relaxation_applied = true;
  while(global_relaxation_applied == true) {
    bool local_relaxation_applied = false;
    Newton_iter++;

    // Loop over all cells.
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                              {
                                try {
                                  perform_Newton_step_relaxation(conserved_variables[cell],
                                                                 dalpha1[cell], alpha1[cell],
                                                                 local_relaxation_applied);
                                }
                                catch(const std::exception& e) {
                                  std::cerr << e.what() << std::endl;
                                  save("_diverged", conserved_variables,
                                                    alpha1);
                                }
                              }
                          );

    mpi::communicator world;
    boost::mpi::all_reduce(world, local_relaxation_applied, global_relaxation_applied, std::logical_or<bool>());

    // Newton cycle diverged
    if(Newton_iter > max_Newton_iters && global_relaxation_applied == true) {
      std::cerr << "Netwon method not converged in the post-hyperbolic relaxation" << std::endl;
      save("_diverged", conserved_variables,
                        alpha1);
      exit(1);
    }

    samurai::times::timers.stop("apply_relaxation");
  }
}

// Implement a single step of the relaxation procedure (valid for a general EOS)
//
template<std::size_t dim>
template<typename State>
void DamBreak<dim>::perform_Newton_step_relaxation(State local_conserved_variables,
                                                   Number& dalpha1_loc,
                                                   Number& alpha1_loc,
                                                   bool& local_relaxation_applied) {
  /*--- Pre-fetch some variables used multiple times in order to exploit possible vectorization ---*/
  const auto m1_loc = local_conserved_variables(M1_INDEX);
  const auto m2_loc = local_conserved_variables(M2_INDEX);

  /*--- Update auxiliary values affected by the nonlinear function for which we seek a zero ---*/
  const auto inv_alpha1_loc = static_cast<Number>(1.0)/alpha1_loc; /*--- TODO: Add a check in case of zero volume fraction ---*/
  const auto rho1_loc       = m1_loc*inv_alpha1_loc; /*--- TODO: Add a check in case of zero volume fraction ---*/
  const auto p1_loc         = EOS_phase1.pres_value(rho1_loc);

  const auto alpha2_loc     = static_cast<Number>(1.0) - alpha1_loc;
  const auto inv_alpha2_loc = static_cast<Number>(1.0)/alpha2_loc; /*--- TODO: Add a check in case of zero volume fraction ---*/
  const auto rho2_loc       = m2_loc*inv_alpha2_loc; /*--- TODO: Add a check in case of zero volume fraction ---*/
  const auto p2_loc         = EOS_phase2.pres_value(rho2_loc);

  /*--- Compute the nonlinear function for which we seek the zero (basically the pressure equilibrium) ---*/
  const auto F = p1_loc - p2_loc;

  /*--- Perform the relaxation only where really needed ---*/
  if(!std::isnan(F) && std::abs(F) > atol_Newton + rtol_Newton*EOS_phase1.get_p0() &&
     std::abs(dalpha1_loc) > atol_Newton) {
    local_relaxation_applied = true;

    /*--- Compute the derivative w.r.t volume fraction recalling that for a barotropic EOS dp/drho = c^2 ---*/
    const auto dF_dalpha1_loc = -m1_loc*inv_alpha1_loc*inv_alpha1_loc*
                                EOS_phase1.c_value(rho1_loc)*EOS_phase1.c_value(rho1_loc)
                                -m2_loc*inv_alpha2_loc*inv_alpha2_loc*
                                EOS_phase2.c_value(rho2_loc)*EOS_phase2.c_value(rho2_loc);

    /*--- Compute the volume fraction update with a bound-preserving strategy ---*/
    dalpha1_loc = -F/dF_dalpha1_loc;
    if(dalpha1_loc > static_cast<Number>(0.0)) {
      dalpha1_loc = std::min(dalpha1_loc, lambda*alpha2_loc);
    }
    else if(dalpha1_loc < static_cast<Number>(0.0)) {
      dalpha1_loc = std::max(dalpha1_loc, -lambda*alpha1_loc);
    }

    #ifdef VERBOSE
      if(alpha1_loc + dalpha1_loc < static_cast<Number>(0.0) ||
         alpha1_loc + dalpha1_loc > static_cast<Number>(1.0)) {
        // We should never arrive here thanks to the bound-preserving strategy. Added only for the sake of safety
        throw std::runtime_error("Bounds exceeding value for the volume fraction inside Newton step");
      }
    #endif
    alpha1_loc += dalpha1_loc;
  }

  /*--- Update the vector of conserved variables ---*/
  local_conserved_variables(RHO_ALPHA1_INDEX) = (m1_loc + m2_loc)*alpha1_loc;
}

//////////////////////////////////////////////////////////////
/*---- FOCUS NOW ON THE POST-PROCESSING FUNCTIONS ---*/
/////////////////////////////////////////////////////////////

// Save desired fields and info
//
template<std::size_t dim>
template<class... Variables>
void DamBreak<dim>::save(const std::string& suffix,
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
void DamBreak<dim>::run(const std::size_t nfiles) {
  /*--- Default output arguemnts ---*/
  filename = "dam_break_" + numerical_flux->get_flux_name();

  #ifdef ORDER_2
    filename += "_order2";
    #ifdef RELAX_RECONSTRUCTION
      filename += "_relaxed_reconstruction";
    #endif
  #else
    filename += "_order1";
  #endif

  const auto dt_save = Tf/static_cast<Number>(nfiles);

  /*--- Auxiliary variable to save updated fields ---*/
  #ifdef ORDER_2
    auto conserved_variables_tmp = samurai::make_vector_field<Number, Field::n_comp>("conserved_tmp", mesh);
    auto conserved_variables_old = samurai::make_vector_field<Number, Field::n_comp>("conserved_old", mesh);
  #endif
  auto conserved_variables_np1 = samurai::make_vector_field<Number, Field::n_comp>("conserved_np1", mesh);

  /*--- Create the flux variable ---*/
  auto Discrete_flux = numerical_flux->make_flux();

  /*--- Create the gravity term ---*/
  #ifdef GRAVITY_IMPLICIT
    auto id = samurai::make_identity<decltype(conserved_variables)>(); // Identity matrix for the purpose of implicitation
  #endif
  using cfg = samurai::LocalCellSchemeConfig<samurai::SchemeType::LinearHomogeneous,
                                             Field,
                                             Field>;
  auto gravity = samurai::make_cell_based_scheme<cfg>();
  gravity.coefficients_func() = [](samurai::StencilCoeffs<cfg>& coeffs, Number)
                                  {
                                    coeffs.fill(static_cast<Number>(0.0));

                                    coeffs(RHO_U_INDEX + 1, M1_INDEX) = static_cast<Number>(-9.81);
                                    coeffs(RHO_U_INDEX + 1, M2_INDEX) = static_cast<Number>(-9.81);
                                  };

  /*--- Save the initial condition ---*/
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(suffix_init, conserved_variables,
                    alpha1, rho, p1, p2, p, vel);

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

  /*--- Delcare operators for multiresolution ---*/
  /*auto prediction_fn = [&](auto& new_field, const auto& old_field) {
    return make_field_operator_function<DamBreak_prediction_op>(new_field, old_field);
  };*/
  //auto MRadaptation = samurai::make_MRAdapt(grad_alpha1);
  auto MRadaptation = samurai::make_MRAdapt(/*prediction_fn,*/ conserved_variables);
  auto mra_config   = samurai::mra_config();
  mra_config.epsilon(MR_param);
  mra_config.regularity(MR_regularity);

  /*--- Start the loop ---*/
  std::size_t nsave = 0;
  std::size_t nt    = 0;
  auto t            = static_cast<Number>(t0);
  while(t != Tf) {
    // Apply mesh adaptation
    /*alpha1.resize();
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                              {
                                alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/
                                               (conserved_variables[cell][M1_INDEX] +
                                                conserved_variables[cell][M2_INDEX]);
                              }
                          );
    samurai::update_ghost_mr(alpha1);
    grad_alpha1.resize();
    grad_alpha1 = gradient(alpha1);*/
    MRadaptation(mra_config);
    //MRadaptation(mra_config, conserved_variables);
    #ifdef VERBOSE
      check_data(1);
    #endif

    // Compute the time step
    const auto dt = std::min(Tf - t, cfl*dx/get_max_lambda());
    t += dt;

    if(rank == 0) {
      std::cout << fmt::format("Iteration {}: t = {}, dt = {}", ++nt, t, dt) << std::endl;
    }

    // Save current state in case of order 2
    #ifdef ORDER_2
      conserved_variables_old.resize();
      conserved_variables_old = conserved_variables;
    #endif

    // Apply the numerical scheme without relaxation
    try {
      #ifdef ORDER_2
        conserved_variables_tmp.resize();
        #ifdef GRAVITY_IMPLICIT
          auto implicit_operator = id - dt*gravity;
          auto rhs = conserved_variables - dt*Discrete_flux(conserved_variables);
          samurai::petsc::solve(implicit_operator, conserved_variables_tmp, rhs);
        #else
          conserved_variables_tmp = conserved_variables
                                  - dt*Discrete_flux(conserved_variables)
                                  + dt*gravity(conserved_variables);
        #endif
        samurai::swap(conserved_variables, conserved_variables_tmp);
      #else
        conserved_variables_np1.resize();
        #ifdef GRAVITY_IMPLICIT
          auto implicit_operator = id - dt*gravity;
          auto rhs = conserved_variables - dt*Discrete_flux(conserved_variables);
          samurai::petsc::solve(implicit_operator, conserved_variables_np1, rhs);
        #else
          conserved_variables_np1 = conserved_variables
                                  - dt*Discrete_flux(conserved_variables)
                                  + dt*gravity(conserved_variables);
        #endif
        samurai::swap(conserved_variables, conserved_variables_np1);
      #endif
    }
    catch(const std::exception& e) {
      std::cerr << e.what() << std::endl;
      save("_diverged", conserved_variables);
      exit(1);
    }
    #ifdef VERBOSE
      check_data();
    #endif

    // Apply relaxation
    if(apply_relax) {
      // Apply relaxation if desired, which will modify alpha1 and, consequently, for what
      // concerns next time step or stage, rho_alpha1
      dalpha1.resize();
      rho.resize();
      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                             {
                               rho[cell]     = conserved_variables[cell][M1_INDEX]
                                             + conserved_variables[cell][M2_INDEX];
                               alpha1[cell]  = conserved_variables[cell][RHO_ALPHA1_INDEX]/rho[cell];
                               dalpha1[cell] = std::numeric_limits<Number>::infinity();
                             });
      apply_relaxation();
    }

    // Consider the second stage for the second order
    #ifdef ORDER_2
      // Apply the numerical scheme
      try {
        #ifdef GRAVITY_IMPLICIT
          auto implicit_operator = id - dt*gravity;
          auto rhs = conserved_variables - dt*Discrete_flux(conserved_variables);
          samurai::petsc::solve(implicit_operator, conserved_variables_tmp, rhs);
        #else
          conserved_variables_tmp = conserved_variables
                                  - dt*Discrete_flux(conserved_variables)
                                  + dt*gravity(conserved_variables);
        #endif
        conserved_variables_np1.resize();
        conserved_variables_np1 = static_cast<Number>(0.5)*
                                  (conserved_variables_tmp + conserved_variables_old);
        samurai::swap(conserved_variables, conserved_variables_np1);
      }
      catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
        save("_diverged", conserved_variables);
        exit(1);
      }
      #ifdef VERBOSE
        check_data();
      #endif

      // Apply the relaxation
      if(apply_relax) {
        // Apply relaxation if desired, which will modify alpha1 and, consequently, for what
        // concerns next time step, rho_alpha1
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                                  {
                                    rho[cell]     = conserved_variables[cell][M1_INDEX]
                                                  + conserved_variables[cell][M2_INDEX];
                                    alpha1[cell]  = conserved_variables[cell][RHO_ALPHA1_INDEX]/rho[cell];
                                    dalpha1[cell] = std::numeric_limits<Number>::infinity();
                                  }
                              );
        apply_relaxation();
      }
    #endif

    // Save the results
    if(t >= static_cast<Number>(nsave + 1)*dt_save || t == Tf) {
      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";

      /*--- Compute auxiliary fields for the output ---*/
      if(!apply_relax) {
        rho.resize();
        alpha1.resize();
      }
      p1.resize();
      p2.resize();
      p.resize();
      vel.resize();
      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                                {
                                  // Pre-fetch variables that will be used multiple times to explit possible vectorization
                                  const auto m1_loc = conserved_variables[cell][M1_INDEX];
                                  const auto m2_loc = conserved_variables[cell][M2_INDEX];

                                  // Recompute volume fraction in case relaxation is not applied
                                  // (and therefore they have not been updated)
                                  const auto rho_loc     = m1_loc + m2_loc;
                                  const auto inv_rho_loc = static_cast<Number>(1.0)/rho_loc;
                                  rho[cell] = rho_loc;
                                  if(!apply_relax) {
                                    alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]*inv_rho_loc;
                                  }

                                  // Update the velocity
                                  for(std::size_t d = 0; d < dim; ++d) {
                                    vel[cell][d] = conserved_variables[cell][RHO_U_INDEX + d]*inv_rho_loc;
                                  }

                                  // Compute pressure fields
                                  const auto alpha1_loc = alpha1[cell];
                                  const auto rho1_loc   = m1_loc/alpha1_loc;
                                  /*--- TODO: Add a check in case of zero volume fraction ---*/
                                  const auto p1_loc = EOS_phase1.pres_value(rho1_loc);
                                  p1[cell] = p1_loc;

                                  const auto alpha2_loc = static_cast<Number>(1.0) - alpha1_loc;
                                  const auto rho2_loc   = m2_loc/alpha2_loc;
                                  /*--- TODO: Add a check in case of zero volume fraction ---*/
                                  const auto p2_loc = EOS_phase2.pres_value(rho2_loc);
                                  p2[cell] = p2_loc;

                                  p[cell] = alpha1_loc*p1_loc
                                          + alpha2_loc*p2_loc;
                                }
                            );

      save(suffix, conserved_variables,
                   alpha1, rho, p1, p2, p, vel);
    }
  }
}
