// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#include <samurai/algorithm/update.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <numbers>

#include <filesystem>
namespace fs = std::filesystem;

/*--- Add header file for the multiresolution ---*/
#include <samurai/mr/adapt.hpp>

/*--- Add header with auxiliary structs ---*/
#include "containers.hpp"

/*--- Add user implemented boundary condition ---*/
#include "user_bc.hpp"

/*--- Include the headers with the numerical fluxes ---*/
//#define RUSANOV_FLUX
#define HLLC_FLUX

#ifdef RUSANOV_FLUX
  #include "Rusanov_flux.hpp"
#elifdef HLLC_FLUX
  #include "HLLC_flux.hpp"
#endif
#include "SurfaceTension_flux.hpp"

// Specify the use of this namespace where we just store the indices
// and, in this case, some parameters related to EOS
using namespace EquationData;

/*--- Define preprocessor to check whether to control data or not ---*/
#define VERBOSE

/** This is the class for the simulation for the two-scale capillarity model
 */
template<std::size_t dim>
class TwoScaleCapillarity {
public:
  using Config = samurai::MRConfig<dim, 2, 2, 0>;

  TwoScaleCapillarity() = default; /*--- Default constructor. This will do nothing
                                         and basically will never be used ---*/

  TwoScaleCapillarity(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                      const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                      const Simulation_Paramaters& sim_param,
                      const EOS_Parameters& eos_param); /*--- Class constrcutor with the arguments related
                                                              to the grid, to the physics and to the relaxation. ---*/

  void run(); /*--- Function which actually executes the temporal loop ---*/

  template<class... Variables>
  void save(const fs::path& path,
            const std::string& suffix,
            const Variables&... fields); /*--- Routine to save the results ---*/

private:
  /*--- Now we declare some relevant variables ---*/
  const samurai::Box<double, dim> box;

  samurai::MRMesh<Config> mesh; /*--- Variable to store the mesh ---*/

  using Field              = samurai::VectorField<decltype(mesh), double, EquationData::NVARS, false>;
  using Field_Scalar       = samurai::ScalarField<decltype(mesh), typename Field::value_type>;
  using Field_Vect         = samurai::VectorField<decltype(mesh), typename Field::value_type, dim, false>;
  using Field_ScalarVector = samurai::VectorField<decltype(mesh), typename Field::value_type, 1, false>;

  bool apply_relax; /*--- Choose whether to apply or not the relaxation ---*/

  double Tf;  /*--- Final time of the simulation ---*/
  double cfl; /*--- Courant number of the simulation so as to compute the time step ---*/

  std::size_t nfiles; /*--- Number of files desired for output ---*/

  Field conserved_variables; /*--- The variable which stores the conserved variables,
                                   namely the varialbes for which we solve a PDE system ---*/

  /*--- Now we declare a bunch of fields which depend from the state, but it is useful
        to have it so as to avoid recomputation ---*/
  Field_Scalar alpha1,
               dalpha1,
               p1,
               p2,
               p;

  Field_Vect normal,
             grad_alpha1;

  Field_Scalar alpha1_d,
               alpha1_bar,
               Sigma_d;

  Field_Vect grad_alpha1_d,
             vel,
             normal_bar,
             grad_alpha1_bar;

  Field_ScalarVector H,
                     H_bar,
                     div_vel;

  samurai::ScalarField<decltype(mesh), std::size_t> to_be_relaxed;
  samurai::ScalarField<decltype(mesh), std::size_t> Newton_iterations;

  using gradient_type = decltype(samurai::make_gradient_order2<decltype(alpha1)>());
  gradient_type gradient;

  using divergence_type = decltype(samurai::make_divergence_order2<decltype(normal)>());
  divergence_type divergence;

  const double sigma; /*--- Surface tension coefficient ---*/

  const double mod_grad_alpha1_min; /*--- Minimum threshold for which not computing anymore the unit normal ---*/

  const bool   mass_transfer; /*--- Choose wheter to apply or not the mass transfer ---*/
  const double Hmax;          /*--- Threshold length scale ---*/
  const double kappa;         /*--- Parameter related to the radius of small-scale droplets ---*/
  const double alpha1d_max;   /*--- Maximum threshold of small-scale volume fraction ---*/
  const double alpha1_min;    /*--- Minimum effective volume fraction to identify the mixture region ---*/
  const double alpha1_max;    /*--- Maximum effective volume fraction to identify the mixture region ---*/

  const double      lambda;           /*--- Parameter for bound preserving strategy ---*/
  const double      atol_Newton;      /*--- Absolute tolerance Newton method relaxation ---*/
  const double      rtol_Newton;      /*--- Relative tolerance Newton method relaxation ---*/
  const std::size_t max_Newton_iters; /*--- Maximum number of Newton iterations ---*/

  const double atol_Newton_small_scale; /*--- Absolute tolerance Newton method small-scale local Laplace law ---*/
  const double rtol_Newton_small_scale; /*--- Relative tolerance Newton method small-scale local Laplace law ---*/

  LinearizedBarotropicEOS<typename Field::value_type> EOS_phase1,
                                                      EOS_phase2; /*--- The two variables which take care of the
                                                                        barotropic EOS to compute the speed of sound ---*/

  #ifdef RUSANOV_FLUX
    samurai::RusanovFlux<Field> Rusanov_flux; /*--- Auxiliary variable to compute the flux for the hyperbolic operator ---*/
  #elifdef HLLC_FLUX
    samurai::HLLCFlux<Field> HLLC_flux; /*--- Auxiliary variable to compute the flux ---*/
  #endif
  samurai::SurfaceTensionFlux<Field> SurfaceTension_flux; /*--- Auxiliary variable to compute the contribution associated to surface tension ---*/

  std::string filename; /*--- Auxiliary variable to store the name of output ---*/

  double MR_param;      /*--- Multiresolution parameter ---*/
  double MR_regularity; /*--- Multiresolution regularity ---*/

  /*--- Auxiliary output streams for post-processing ---*/
  std::ofstream Hlig;
  std::ofstream m1_integral;
  std::ofstream m1_d_integral;
  std::ofstream alpha1_integral;
  std::ofstream grad_alpha1_integral;
  std::ofstream Sigma_d_integral;
  std::ofstream alpha1_d_integral;
  std::ofstream grad_alpha1_d_integral;
  std::ofstream grad_alpha1_tot_integral;
  std::ofstream alpha1_bar_integral;
  std::ofstream grad_alpha1_bar_integral;

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  void update_geometry(); /*--- Auxiliary routine to compute normals and curvature ---*/

  void init_variables(const double R, const double eps_over_R,
                      const double alpha_residual); /*--- Routine to initialize the variables
                                                          (both conserved and auxiliary, this is problem dependent) ---*/

  double get_max_lambda(); /*--- Compute the estimate of the maximum eigenvalue ---*/

  void check_data(unsigned int flag = 0); /*--- Auxiliary routine to check if spurious values are present ---*/

  void perform_mesh_adaptation(); /*--- Perform the mesh adaptation ---*/

  void apply_relaxation(); /*--- Apply the relaxation ---*/

  template<typename State, typename Gradient>
  void perform_Newton_step_relaxation(std::unique_ptr<State> local_conserved_variables,
                                      const typename Field::value_type H_loc,
                                      typename Field::value_type& dalpha1_loc,
                                      typename Field::value_type& alpha1_loc,
                                      std::size_t& to_be_relaxed_loc,
                                      std::size_t& Newton_iterations_loc,
                                      bool& local_relaxation_applied,
                                      const Gradient& grad_alpha1_loc,
                                      const bool mass_transfer_NR);

  void execute_postprocess(const double time); /*--- Execute the postprocess ---*/
};

/*---- START WITH THE IMPLEMENTATION OF THE CONSTRUCTOR ---*/

// Implement class constructor
//
template<std::size_t dim>
TwoScaleCapillarity<dim>::TwoScaleCapillarity(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                                              const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                                              const Simulation_Paramaters& sim_param,
                                              const EOS_Parameters& eos_param):
  box(min_corner, max_corner), mesh(box, sim_param.min_level, sim_param.max_level, {{false, true}}),
  apply_relax(sim_param.apply_relaxation), Tf(sim_param.Tf),
  cfl(sim_param.Courant), nfiles(sim_param.nfiles),
  gradient(samurai::make_gradient_order2<decltype(alpha1)>()),
  divergence(samurai::make_divergence_order2<decltype(normal)>()),
  sigma(sim_param.sigma), mod_grad_alpha1_min(sim_param.mod_grad_alpha1_min),
  mass_transfer(sim_param.mass_transfer), Hmax(sim_param.Hmax),
  kappa(sim_param.kappa), alpha1d_max(sim_param.alpha1d_max),
  alpha1_min(sim_param.alpha1_min), alpha1_max(sim_param.alpha1_max),
  lambda(sim_param.lambda), atol_Newton(sim_param.atol_Newton),
  rtol_Newton(sim_param.rtol_Newton), max_Newton_iters(sim_param.max_Newton_iters),
  atol_Newton_small_scale(sim_param.atol_Newton_small_scale),
  rtol_Newton_small_scale(sim_param.rtol_Newton_small_scale),
  EOS_phase1(eos_param.p0_phase1, eos_param.rho0_phase1, eos_param.c0_phase1),
  EOS_phase2(eos_param.p0_phase2, eos_param.rho0_phase2, eos_param.c0_phase2),
  #ifdef RUSANOV_FLUX
    Rusanov_flux(EOS_phase1, EOS_phase2,
                 sigma, mod_grad_alpha1_min,
                 lambda, atol_Newton, rtol_Newton, max_Newton_iters,
                 atol_Newton_small_scale, rtol_Newton_small_scale),
  #elifdef HLLC_FLUX
    HLLC_flux(EOS_phase1, EOS_phase2,
              sigma, mod_grad_alpha1_min,
              lambda, atol_Newton, rtol_Newton, max_Newton_iters,
              atol_Newton_small_scale, rtol_Newton_small_scale),
  #endif
  SurfaceTension_flux(EOS_phase1, EOS_phase2,
                      sigma, mod_grad_alpha1_min,
                      lambda, atol_Newton, rtol_Newton, max_Newton_iters,
                      atol_Newton_small_scale, rtol_Newton_small_scale),
  MR_param(sim_param.MR_param), MR_regularity(sim_param.MR_regularity)
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) {
      std::cout << "Initializing variables " << std::endl;
      std::cout << std::endl;
    }
    init_variables(sim_param.R, sim_param.eps_over_R, sim_param.alpha_residual);
    to_be_relaxed     = samurai::make_scalar_field<std::size_t>("to_be_relaxed", mesh);
    Newton_iterations = samurai::make_scalar_field<std::size_t>("Newton_iterations", mesh);
  }

// Auxiliary routine to compute normals and curvature
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::update_geometry() {
  samurai::update_ghost_mr(alpha1);

  grad_alpha1 = gradient(alpha1);

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           const auto mod_grad_alpha1 = std::sqrt(xt::sum(grad_alpha1[cell]*grad_alpha1[cell])());

                           if(mod_grad_alpha1 > mod_grad_alpha1_min) {
                             normal[cell] = grad_alpha1[cell]/mod_grad_alpha1;
                           }
                           else {
                             for(std::size_t d = 0; d < dim; ++d) {
                               normal[cell][d] = nan("");
                             }
                           }
                         });
  samurai::update_ghost_mr(normal);
  H = -divergence(normal);
}

// Initialization of conserved and auxiliary variables
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::init_variables(const double R, const double eps_over_R, const double alpha_residual) {
  /*--- Create conserved and auxiliary fields ---*/
  conserved_variables = samurai::make_vector_field<typename Field::value_type, EquationData::NVARS>("conserved", mesh);

  alpha1      = samurai::make_scalar_field<typename Field::value_type>("alpha1", mesh);
  grad_alpha1 = samurai::make_vector_field<typename Field::value_type, dim>("grad_alpha1", mesh);
  normal      = samurai::make_vector_field<typename Field::value_type, dim>("normal", mesh);
  H           = samurai::make_vector_field<typename Field::value_type, 1>("H", mesh);

  dalpha1     = samurai::make_scalar_field<typename Field::value_type>("dalpha1", mesh);

  p1          = samurai::make_scalar_field<typename Field::value_type>("p1", mesh);
  p2          = samurai::make_scalar_field<typename Field::value_type>("p2", mesh);
  p           = samurai::make_scalar_field<typename Field::value_type>("p", mesh);

  alpha1_d      = samurai::make_scalar_field<typename Field::value_type>("alpha1_d", mesh);
  grad_alpha1_d = samurai::make_vector_field<typename Field::value_type, dim>("grad_alpha1_d", mesh);
  Sigma_d       = samurai::make_scalar_field<typename Field::value_type>("Sigma_d", mesh);
  vel           = samurai::make_vector_field<typename Field::value_type, dim>("vel", mesh);
  div_vel       = samurai::make_vector_field<typename Field::value_type, 1>("div_vel", mesh);

  alpha1_bar      = samurai::make_scalar_field<typename Field::value_type>("alpha1_bar", mesh);
  grad_alpha1_bar = samurai::make_vector_field<typename Field::value_type, dim>("grad_alpha1_bar", mesh);
  normal_bar      = samurai::make_vector_field<typename Field::value_type, dim>("normal_bar", mesh);
  H_bar           = samurai::make_vector_field<typename Field::value_type, 1>("H_bar", mesh);

  /*--- Declare some constant parameters associated to the grid and to the
        initial state ---*/
  const double x0    = 1.0;
  const double y0    = 1.0;
  const double eps_R = eps_over_R*R;

  const double U_0 = 6.66;
  const double U_1 = 0.0;
  const double V   = 0.0;

  /*--- Initialize some fields to define the bubble with a loop over all cells ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           // Set large-scale volume fraction
                           const auto center = cell.center();
                           const double x    = center[0];
                           const double y    = center[1];

                           const double r = std::sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));

                           const double w = (r >= R && r < R + eps_R) ?
                                            std::max(std::exp(2.0*(r - R)*(r - R)/(eps_R*eps_R)*((r - R)*(r - R)/(eps_R*eps_R) - 3.0)/
                                                              (((r - R)*(r - R)/(eps_R*eps_R) - 1.0)*((r - R)*(r - R)/(eps_R*eps_R) - 1.0))), 0.0) :
                                            ((r < R) ? 1.0 : 0.0);

                           alpha1[cell] = std::min(std::max(alpha_residual, w), 1.0 - alpha_residual);
                         });

  /*--- Compute the geometrical quantities ---*/
  update_geometry();

  /*--- Loop over a cell to complete the remaining variables ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           // Set small-scale variables
                           alpha1_d[cell]                         = 0.0;
                           conserved_variables[cell][RHO_Z_INDEX] = 0.0;
                           Sigma_d[cell]                          = 0.0;
                           conserved_variables[cell][M1_D_INDEX]  = alpha1_d[cell]*EOS_phase1.get_rho0();

                           // Recompute geometric locations to set partial masses
                           const auto center = cell.center();
                           const double x    = center[0];
                           const double y    = center[1];

                           const double r = std::sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));

                           // Set mass large-scale phase 1
                           if(r >= R + eps_R) {
                             p1[cell] = EOS_phase1.get_p0();
                           }
                           else {
                             p1[cell] = EOS_phase2.get_p0();
                             if(r >= R && r < R + eps_R && !std::isnan(H[cell][0])) {
                               p1[cell] += sigma*H[cell][0];
                             }
                             else {
                               p1[cell] += sigma/R;
                             }
                           }
                           const auto rho1 = EOS_phase1.rho_value(p1[cell]);

                           alpha1_bar[cell] = alpha1[cell]/(1.0 - alpha1_d[cell]);
                           conserved_variables[cell][M1_INDEX] = alpha1[cell]*rho1;

                           // Set mass phase 2
                           p2[cell] = EOS_phase2.get_p0();
                           const auto rho2 = EOS_phase2.rho_value(p2[cell]);

                           const auto alpha2 = 1.0 - alpha1[cell] - alpha1_d[cell];
                           conserved_variables[cell][M2_INDEX] = alpha2*rho2;

                           // Set mixture pressure
                           p[cell] = alpha1[cell]*p1[cell]
                                   + (1.0 - alpha1[cell])*p2[cell];

                           // Set conserved variable associated to large-scale volume fraction
                           const auto rho = conserved_variables[cell][M1_INDEX]
                                          + conserved_variables[cell][M2_INDEX]
                                          + conserved_variables[cell][M1_D_INDEX];

                           conserved_variables[cell][RHO_ALPHA1_INDEX] = rho*alpha1[cell];

                           // Set momentum
                           conserved_variables[cell][RHO_U_INDEX]     = conserved_variables[cell][M1_INDEX]*U_1
                                                                      + conserved_variables[cell][M2_INDEX]*U_0;
                           conserved_variables[cell][RHO_U_INDEX + 1] = rho*V;

                           vel[cell][0] = conserved_variables[cell][RHO_U_INDEX]/rho;
                           vel[cell][1] = conserved_variables[cell][RHO_U_INDEX + 1]/rho;
                         });

  /*--- Set useful small-scale related fields ---*/
  samurai::update_ghost_mr(alpha1_d);
  grad_alpha1_d = gradient(alpha1_d);

  samurai::update_ghost_mr(vel);
  div_vel = divergence(vel);

  /*--- Set auxiliary gradient alpha1_bar volume fraction ---*/
  samurai::update_ghost_mr(alpha1_bar);
  grad_alpha1_bar = gradient(alpha1_bar);
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           const auto mod_grad_alpha1_bar = std::sqrt(xt::sum(grad_alpha1_bar[cell]*grad_alpha1_bar[cell])());

                           if(mod_grad_alpha1_bar > mod_grad_alpha1_min) {
                             normal_bar[cell] = grad_alpha1_bar[cell]/mod_grad_alpha1_bar;
                           }
                           else {
                             for(std::size_t d = 0; d < dim; ++d) {
                               normal_bar[cell][d] = nan("");
                             }
                           }
                         });
  samurai::update_ghost_mr(normal_bar);
  H_bar = -divergence(normal_bar);

  /*--- Apply bcs ---*/
  const samurai::DirectionVector<dim> left = {-1, 0};
  samurai::make_bc<Default>(conserved_variables,
                            Inlet(conserved_variables, U_0, 0.0, alpha_residual, 0.0, EOS_phase1.get_rho0(), 0.0,
                                  sigma, EOS_phase1, EOS_phase2, atol_Newton, rtol_Newton, max_Newton_iters, lambda))->on(left);
  /*samurai::make_bc<samurai::Dirichlet<1>>(conserved_variables, alpha_residual*EOS_phase1.get_rho0(),
                                                               (1.0 - alpha_residual)*EOS_phase2.get_rho0(),
                                                               0.0, 0.0, 0.0,
                                                               (alpha_residual*EOS_phase1.get_rho0() + (1.0 - alpha_residual)*EOS_phase2.get_rho0())*
                                                               alpha_residual,
                                                               (alpha_residual*EOS_phase1.get_rho0() +
                                                                (1.0 - alpha_residual)*EOS_phase2.get_rho0())*U_0, 0.0)->on(left);*/

  const samurai::DirectionVector<dim> right = {1, 0};
  samurai::make_bc<samurai::Neumann<1>>(conserved_variables, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)->on(right);

  /*const samurai::DirectionVector<dim> top = {0, 1};
  samurai::make_bc<samurai::Neumann<1>>(conserved_variables, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)->on(top);*/

  /*const samurai::DirectionVector<dim> bottom = {0, -1};
  samurai::make_bc<samurai::Neumann<1>>(conserved_variables, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)->on(bottom);*/
}

/*---- FOCUS NOW ON THE AUXILIARY FUNCTIONS ---*/

// Compute the estimate of the maximum eigenvalue for CFL condition
//
template<std::size_t dim>
double TwoScaleCapillarity<dim>::get_max_lambda() {
  double local_res = 0.0;

  alpha1.resize();
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           #ifndef RELAX_RECONSTRUCTION
                             alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/
                                            (conserved_variables[cell][M1_INDEX] +
                                             conserved_variables[cell][M2_INDEX] +
                                             conserved_variables[cell][M1_D_INDEX]);
                           #endif

                           /*--- Compute the velocity along both horizontal and vertical direction ---*/
                           const auto rho   = conserved_variables[cell][M1_INDEX]
                                            + conserved_variables[cell][M2_INDEX]
                                            + conserved_variables[cell][M1_D_INDEX];
                           const auto vel_x = conserved_variables[cell][RHO_U_INDEX]/rho;
                           const auto vel_y = conserved_variables[cell][RHO_U_INDEX + 1]/rho;

                           /*--- Compute frozen speed of sound ---*/
                           const auto rho1 = conserved_variables[cell][M1_INDEX]/alpha1[cell]; /*--- TODO: Add a check in case of zero volume fraction ---*/
                           typename Field::value_type rho1_d;
                           try {
                             rho1_d = compute_rho1_d_local_Laplace<Field>(conserved_variables[cell][M1_D_INDEX],
                                                                          conserved_variables[cell][M2_INDEX],
                                                                          conserved_variables[cell][M1_INDEX],
                                                                          alpha1[cell],
                                                                          conserved_variables[cell][RHO_Z_INDEX],
                                                                          sigma, EOS_phase1, EOS_phase2,
                                                                          atol_Newton_small_scale, rtol_Newton_small_scale, max_Newton_iters, lambda);
                           }
                           catch(const std::exception& e) {
                             std::cerr << "Small-scale error when computing max speed sound" << std::endl;
                             std::cerr << conserved_variables[cell] << std::endl;
                             std::cerr << e.what() << std::endl;
                             exit(1);
                           }
                           alpha1_d[cell]          = conserved_variables[cell][M1_D_INDEX]/rho1_d;
                           const auto alpha2       = 1.0 - alpha1[cell] - alpha1_d[cell];
                           const auto rho2         = conserved_variables[cell][M2_INDEX]/alpha2; /*--- TODO: Add a check in case of zero volume fraction ---*/
                           const auto rhoc_squared = conserved_variables[cell][M1_INDEX]*EOS_phase1.c_value(rho1)*EOS_phase1.c_value(rho1)
                                                   + ((1.0 - alpha1[cell])/(alpha2))*((1.0 - alpha1[cell])/(alpha2))*
                                                     conserved_variables[cell][M2_INDEX]*EOS_phase2.c_value(rho2)*EOS_phase2.c_value(rho2);
                           const auto c            = std::sqrt(rhoc_squared/rho);

                           /*--- Add term due to surface tension ---*/
                           const double r = sigma*std::sqrt(xt::sum(grad_alpha1[cell]*grad_alpha1[cell])())/(rho*c*c);

                           /*--- Update eigenvalue estimate ---*/
                           local_res = std::max(std::max(std::abs(vel_x) + c*(1.0 + 0.125*r),
                                                         std::abs(vel_y) + c*(1.0 + 0.125*r)),
                                                local_res);
                         });

  double global_res;
  MPI_Allreduce(&local_res, &global_res, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  return global_res;
}

// Perform the mesh adaptation strategy.
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::perform_mesh_adaptation() {
  samurai::update_ghost_mr(grad_alpha1);
  auto MRadaptation = samurai::make_MRAdapt(grad_alpha1);
  MRadaptation(MR_param, MR_regularity, conserved_variables);

  /*--- Sanity check after mesh adaptation ---*/
  alpha1.resize();
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                            alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/
                                           (conserved_variables[cell][M1_INDEX] +
                                            conserved_variables[cell][M2_INDEX] +
                                            conserved_variables[cell][M1_D_INDEX]);
                         });
  #ifdef VERBOSE
    check_data(1);
  #endif
}

// Auxiliary fuction to check if spurious values are present
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::check_data(unsigned int flag) {
  std::string op;
  if(flag == 0) {
    op = "after hyperbolic operator (i.e. at the beginning of the relaxation)";
  }
  else {
    op = "after mesh adptation";
  }

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           // Sanity check for alpha1
                           if(alpha1[cell] < 0.0) {
                             std::cerr << cell << std::endl;
                             std::cerr << "Negative volume fraction large-scale liquid " + op << std::endl;
                             save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                             exit(1);
                           }
                           else if(alpha1[cell] > 1.0) {
                             std::cerr << cell << std::endl;
                             std::cerr << "Exceeding volume fraction large-scale liquid " + op << std::endl;
                             save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                             exit(1);
                           }
                           else if(std::isnan(alpha1[cell])) {
                             std::cerr << cell << std::endl;
                             std::cerr << "NaN volume fraction large-scale liquid " + op << std::endl;
                             save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                             exit(1);
                           }

                           // Sanity check for m1
                           if(conserved_variables[cell][M1_INDEX] < 0.0) {
                             std::cerr << cell << std::endl;
                             std::cerr << "Negative mass large-scale liquid " + op << std::endl;
                             save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                             exit(1);
                           }
                           else if(std::isnan(conserved_variables[cell][M1_INDEX])) {
                             std::cerr << cell << std::endl;
                             std::cerr << "NaN mass large-scale liquid " + op << std::endl;
                             save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                             exit(1);
                           }

                           // Sanity check for m2
                           if(conserved_variables[cell][M2_INDEX] < 0.0) {
                             std::cerr << cell << std::endl;
                             std::cerr << "Negative mass gas phase " + op << std::endl;
                             save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                             exit(1);
                           }
                           else if(std::isnan(conserved_variables[cell][M2_INDEX])) {
                             std::cerr << cell << std::endl;
                             std::cerr << "NaN mass gas phase " + op << std::endl;
                             save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                             exit(1);
                           }

                           // Sanity check for m1_d
                           if(conserved_variables[cell][M1_D_INDEX] < -1e-15) {
                             std::cerr << cell << std::endl;
                             std::cerr << "Negative mass small-scale liquid " + op << std::endl;
                             save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                             exit(1);
                           }
                           else if(std::isnan(conserved_variables[cell][M1_D_INDEX])) {
                             std::cerr << cell << std::endl;
                             std::cerr << "NaN mass small-scale liquid " + op << std::endl;
                             save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                             exit(1);
                           }

                           // Sanity check for z (the transported variable related to small-scale IAD)
                           if(conserved_variables[cell][RHO_Z_INDEX] < -1e-15) {
                             std::cerr << cell << std::endl;
                             std::cerr << "Negative interface area small-scale liquid " + op << std::endl;
                             save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                             exit(1);
                           }
                           else if(std::isnan(conserved_variables[cell][RHO_Z_INDEX])) {
                             std::cerr << cell << std::endl;
                             std::cerr << "NaN interface area small-scale liquid " + op << std::endl;
                             save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                             exit(1);
                           }
                         });
}

// Apply the relaxation. This procedure is valid for a generic EOS
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::apply_relaxation() {
  /*--- Initialize the variables ---*/
  samurai::times::timers.start("apply_relaxation");

  std::size_t Newton_iter = 0;
  Newton_iterations.fill(0);
  dalpha1.fill(std::numeric_limits<typename Field::value_type>::infinity());
  bool global_relaxation_applied = true;
  bool mass_transfer_NR          = mass_transfer; // In principle we might think to disable it after a certain
                                                  // number of iterations (as in Arthur's code), not done here.

  samurai::times::timers.stop("apply_relaxation");

  /*--- Loop of Newton method. Conceptually, a loop over cells followed by a Newton loop
        over each cell would (could?) be more logic, but this would lead to issues to call 'update_geometry' ---*/
  while(global_relaxation_applied == true) {
    samurai::times::timers.start("apply_relaxation");

    bool local_relaxation_applied = false;
    Newton_iter++;

    // Loop over all cells.
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                             try {
                               perform_Newton_step_relaxation(std::make_unique<decltype(conserved_variables[cell])>(conserved_variables[cell]),
                                                              H[cell][0], dalpha1[cell], alpha1[cell],
                                                              to_be_relaxed[cell], Newton_iterations[cell], local_relaxation_applied,
                                                              grad_alpha1[cell], mass_transfer_NR);
                             }
                             catch(std::exception& e) {
                               std::cerr << e.what() << std::endl;
                               std::cerr << cell << std::endl;
                               save(fs::current_path(), "_diverged",
                                    conserved_variables, alpha1, dalpha1, grad_alpha1, normal, H,
                                    to_be_relaxed, Newton_iterations);
                               exit(1);
                             }
                           });

    mpi::communicator world;
    boost::mpi::all_reduce(world, local_relaxation_applied, global_relaxation_applied, std::logical_or<bool>());

    // Newton cycle diverged
    if(Newton_iter > max_Newton_iters && global_relaxation_applied == true) {
      std::cerr << "Netwon method not converged in the post-hyperbolic relaxation" << std::endl;
      save(fs::current_path(), "_diverged",
           conserved_variables, alpha1, dalpha1, grad_alpha1, normal, H,
           to_be_relaxed, Newton_iterations);
      exit(1);
    }

    samurai::times::timers.stop("apply_relaxation");

    // Recompute geometric quantities (curvature potentially changed in the Newton loop)
    update_geometry();
  }
}

// Implement a single step of the relaxation procedure (valid for a general EOS)
//
template<std::size_t dim>
template<typename State, typename Gradient>
void TwoScaleCapillarity<dim>::perform_Newton_step_relaxation(std::unique_ptr<State> local_conserved_variables,
                                                              const typename Field::value_type H_loc,
                                                              typename Field::value_type& dalpha1_loc,
                                                              typename Field::value_type& alpha1_loc,
                                                              std::size_t& to_be_relaxed_loc,
                                                              std::size_t& Newton_iterations_loc,
                                                              bool& local_relaxation_applied,
                                                              const Gradient& grad_alpha1_loc,
                                                              const bool mass_transfer_NR) {
  to_be_relaxed_loc = 0;

  if(!std::isnan(H_loc)) {
    /*--- Update auxiliary values affected by the nonlinear function for which we seek a zero ---*/
    const auto rho1_loc = (*local_conserved_variables)(M1_INDEX)/alpha1_loc; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto p1_loc   = EOS_phase1.pres_value(rho1_loc);

    typename Field::value_type rho1_d_loc;
    try {
      rho1_d_loc = compute_rho1_d_local_Laplace<Field>((*local_conserved_variables)(M1_D_INDEX),
                                                       (*local_conserved_variables)(M2_INDEX),
                                                       (*local_conserved_variables)(M1_INDEX),
                                                       alpha1_loc,
                                                       (*local_conserved_variables)(RHO_Z_INDEX),
                                                       sigma, EOS_phase1, EOS_phase2,
                                                       atol_Newton_small_scale, rtol_Newton_small_scale, max_Newton_iters, lambda);
    }
    catch(const std::exception& e) {
      std::cerr << "Small-scale error when computing mass transfer" << std::endl;
      std::cerr << (*local_conserved_variables) << std::endl;
      throw std::runtime_error("Checking data");
    }
    const auto alpha1_d_loc = (*local_conserved_variables)(M1_D_INDEX)/rho1_d_loc;
    const auto alpha2_loc   = 1.0 - alpha1_loc - alpha1_d_loc;
    const auto rho2_loc     = (*local_conserved_variables)(M2_INDEX)/alpha2_loc; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto p2_loc       = EOS_phase2.pres_value(rho2_loc);

    /*--- Prepare for mass transfer if desired ---*/
    const auto rho_loc = (*local_conserved_variables)(M1_INDEX)
                       + (*local_conserved_variables)(M2_INDEX)
                       + (*local_conserved_variables)(M1_D_INDEX);

    // Compute first ordrer integral reminder "specific enthalpy"
    typename Field::value_type p2_minus_p1_times_theta;
    try {
      p2_minus_p1_times_theta = rho1_loc*(EOS_phase1.e_value(rho1_d_loc) - EOS_phase1.e_value(rho1_loc))
                              + p2_loc*(rho1_loc/rho1_d_loc - 1.0);
    }
    catch(std::exception& e) {
      std::cerr << e.what() << std::endl;
      throw std::runtime_error("Error when computing the internal energy");
    }
    typename Field::value_type H_lim = std::min(H_loc, Hmax);
    const auto fac_Ru                = sigma*(3.0*H_lim/(kappa*rho1_d_loc))*rho1_loc -
                                       sigma*H_lim +
                                       p2_minus_p1_times_theta;
    if(mass_transfer_NR) {
      if(fac_Ru > 0.0 &&
         alpha1_loc > alpha1_min && alpha1_loc < alpha1_max &&
         -grad_alpha1_loc[0]*(*local_conserved_variables)(RHO_U_INDEX)
         -grad_alpha1_loc[1]*(*local_conserved_variables)(RHO_U_INDEX + 1) > 0.0 &&
         alpha1_d_loc < alpha1d_max) {
        ;
      }
      else {
        H_lim = H_loc;
      }
    }
    else {
      H_lim = H_loc;
    }

    const auto dH = H_loc - H_lim;

    // Compute the nonlinear function for which we seek the zero (basically the Laplace law)
    const auto F = (p1_loc - p2_loc) - sigma*H_lim;

    // Perform the relaxation only where really needed
    if(std::abs(F) > atol_Newton + rtol_Newton*std::min(EOS_phase1.get_p0(), sigma*std::abs(H_lim)) &&
       std::abs(dalpha1_loc) > atol_Newton) {
      to_be_relaxed_loc = 1;
      Newton_iterations_loc++;
      local_relaxation_applied = true;

      // Compute the derivative w.r.t large scale volume fraction recalling that for a barotropic EOS dp/drho = c^2
      const auto dF_dalpha1 = -(*local_conserved_variables)(M1_INDEX)/(alpha1_loc*alpha1_loc)*
                               EOS_phase1.c_value(rho1_loc)*EOS_phase1.c_value(rho1_loc)
                              -(*local_conserved_variables)(M2_INDEX)/(alpha2_loc*alpha2_loc)*
                               EOS_phase2.c_value(rho2_loc)*EOS_phase2.c_value(rho2_loc);

      // Compute the pseudo time step starting as initial guess from the ideal unmodified Newton method
      auto dtau_ov_epsilon = std::numeric_limits<typename Field::value_type>::infinity();

      // Bound preserving condition for m1, velocity and small-scale volume fraction
      if(dH > 0.0) {
        // Bound preserving condition for m1
        dtau_ov_epsilon = lambda*alpha1_loc/(sigma*dH);
        if(dtau_ov_epsilon < 0.0) {
          throw std::runtime_error("Negative time step found after relaxation of mass of large-scale phase 1");
        }

        // Bound preserving for the velocity
        const auto mom_dot_vel   = ((*local_conserved_variables)(RHO_U_INDEX)*(*local_conserved_variables)(RHO_U_INDEX) +
                                    (*local_conserved_variables)(RHO_U_INDEX + 1)*(*local_conserved_variables)(RHO_U_INDEX + 1))/rho_loc;
        auto dtau_ov_epsilon_tmp = lambda*mom_dot_vel/(dH*fac_Ru*sigma);
        dtau_ov_epsilon          = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
        if(dtau_ov_epsilon < 0.0) {
          throw std::runtime_error("Negative time step found after relaxation of velocity");
        }
      }

      // Bound preserving condition for large-scale volume fraction
      const auto dF_dalpha1d   = -EOS_phase2.c_value(rho2_loc)*EOS_phase2.c_value(rho2_loc)*rho2_loc/alpha2_loc;
      const auto dF_dm1        = EOS_phase1.c_value(rho1_loc)*EOS_phase1.c_value(rho1_loc)/alpha1_loc;
      const auto R             = dF_dalpha1d/rho1_d_loc - dF_dm1;
      const auto a             = rho1_loc*sigma*dH*R;
      // Upper bound
      auto b                   = F + lambda*(1.0 - alpha1_loc)*dF_dalpha1;
      auto D                   = b*b - 4.0*a*(-lambda*(1.0 - alpha1_loc));
      auto dtau_ov_epsilon_tmp = std::numeric_limits<typename Field::value_type>::infinity();
      if(D > 0.0 && (a > 0.0 || (a < 0.0 && b > 0.0))) {
        dtau_ov_epsilon_tmp = 0.5*(-b + std::sqrt(D))/a;
      }
      if(a == 0.0 && b > 0.0) {
        dtau_ov_epsilon_tmp = lambda*(1.0 - alpha1_loc)/b;
      }
      dtau_ov_epsilon = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
      // Lower bound
      dtau_ov_epsilon_tmp = std::numeric_limits<typename Field::value_type>::infinity();
      b                   = (F - lambda*alpha1_loc*dF_dalpha1);
      D                   = b*b - 4.0*a*(lambda*alpha1_loc);
      if(D > 0.0 && (a < 0.0 || (a > 0.0 && b < 0.0))) {
        dtau_ov_epsilon_tmp = 0.5*(-b - std::sqrt(D))/a;
      }
      if(a == 0.0 && b < 0.0) {
        dtau_ov_epsilon_tmp = -lambda*alpha1_loc/b;
      }
      dtau_ov_epsilon = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
      if(dtau_ov_epsilon < 0.0) {
        throw std::runtime_error("Negative time step found after relaxation of large-scale volume fraction");
      }

      // Compute the effective variation of the variables
      if(std::isinf(dtau_ov_epsilon)) {
        // If we are in this branch we do not have mass transfer
        // and we do not have other restrictions on the bounds of large scale volume fraction
        dalpha1_loc = -F/dF_dalpha1;

        /*if(dalpha1_loc > 0.0) {
          dalpha1 = std::min(-F/dF_dalpha1, lambda*(1.0 - alpha1_loc));
        }
        else if(dalpha1_loc < 0.0) {
          dalpha1 = std::max(-F/dF_dalpha1, -lambda*alpha1_loc);
        }*/
      }
      else {
        const auto dm1 = -dtau_ov_epsilon*
                          ((*local_conserved_variables)(M1_INDEX)/(alpha1_loc))*
                          sigma*dH;

        dalpha1_loc = (dtau_ov_epsilon/(1.0 - dtau_ov_epsilon*dF_dalpha1))*(F - dm1*R);

        if(dm1 > 0.0) {
          throw std::runtime_error("Negative sign of mass transfer inside Newton step");
        }
        else {
          (*local_conserved_variables)(M1_INDEX) += dm1;
          if((*local_conserved_variables)(M1_INDEX) < 0.0) {
            throw std::runtime_error("Negative mass of large-scale phase 1 inside Newton step");
          }

          (*local_conserved_variables)(M1_D_INDEX) -= dm1;
          if((*local_conserved_variables)(M1_D_INDEX) < 0.0) {
            throw std::runtime_error("Negative mass of small-scale phase 1 inside Newton step");
          }
        }

        const auto R_Sigma_D = -dm1*3.0*Hmax/(kappa*rho1_d_loc);
        (*local_conserved_variables)(RHO_Z_INDEX) += (std::pow(rho1_d_loc, 2.0/3.0))*R_Sigma_D;
      }

      if(alpha1_loc + dalpha1_loc < 0.0 || alpha1_loc + dalpha1_loc > 1.0) {
        throw std::runtime_error("Bounds exceeding value for large-scale volume fraction inside Newton step");
      }
      else {
        alpha1_loc += dalpha1_loc;
      }

      if(dH > 0.0) {
        double drho_fac_Ru = 0.0;
        const auto mom_squared = (*local_conserved_variables)(RHO_U_INDEX)*(*local_conserved_variables)(RHO_U_INDEX)
                               + (*local_conserved_variables)(RHO_U_INDEX + 1)*(*local_conserved_variables)(RHO_U_INDEX + 1);
        if(mom_squared > 0.0) {
          drho_fac_Ru = dtau_ov_epsilon*
                        sigma*dH*fac_Ru*rho_loc/mom_squared; /*--- u/u^{2} = rho*u/(rho*(u^{2})) = (rho/(rho*u)^{2})*(rho*u) ---*/
        }

        for(std::size_t d = 0; d < Field::dim; ++d) {
          (*local_conserved_variables)(RHO_U_INDEX + d) -= drho_fac_Ru*(*local_conserved_variables)(RHO_U_INDEX + d);
        }
      }
    }

    // Update "conservative counter part" of large-scale volume fraction.
    // Do it outside because this can change either because of relaxation or
    // alpha1.
    (*local_conserved_variables)(RHO_ALPHA1_INDEX) = rho_loc*alpha1_loc;
  }
}

// Save desired fields and info
//
template<std::size_t dim>
template<class... Variables>
void TwoScaleCapillarity<dim>::save(const fs::path& path,
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

// Execute postprocessing
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::execute_postprocess(const double time) {
  /*--- Initialize relevant integral quantities ---*/
  typename Field::value_type local_H_lig               = 0.0;
  typename Field::value_type local_m1_int              = 0.0;
  typename Field::value_type local_m1_d_int            = 0.0;
  typename Field::value_type local_alpha1_int          = 0.0;
  typename Field::value_type local_grad_alpha1_int     = 0.0;
  typename Field::value_type local_Sigma_d_int         = 0.0;
  typename Field::value_type local_alpha1_d_int        = 0.0;
  typename Field::value_type local_grad_alpha1_d_int   = 0.0;
  typename Field::value_type local_grad_alpha1_tot_int = 0.0;
  typename Field::value_type local_alpha1_bar_int      = 0.0;
  typename Field::value_type local_grad_alpha1_bar_int = 0.0;

  alpha1_bar.resize();
  alpha1_d.resize();
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           // Save alpha1_bar
                           alpha1_bar[cell] = alpha1[cell]/(1.0 - alpha1_d[cell]);
                         });
  samurai::update_ghost_mr(alpha1_bar);
  grad_alpha1_bar.resize();
  grad_alpha1_bar = gradient(alpha1_bar);
  samurai::update_ghost_mr(alpha1_d);
  grad_alpha1_d.resize();
  grad_alpha1_d = gradient(alpha1_d);

  p1.resize();
  p2.resize();
  p.resize();
  Sigma_d.resize();
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           // Compue H_lig
                           const auto rho1   = conserved_variables[cell][M1_INDEX]/alpha1[cell]; /*--- TODO: Add a check in case of zero volume fraction ---*/
                           const auto rho1_d = (alpha1_d[cell] > 0.0) ?
                                               conserved_variables[cell][M1_D_INDEX]/alpha1_d[cell] :
                                               EOS_phase1.get_rho0();
                           p1[cell]          = EOS_phase1.pres_value(rho1);
                           const auto rho2   = conserved_variables[cell][M2_INDEX]/
                                               (1.0 - alpha1[cell] - alpha1_d[cell]);
                           /*--- TODO: Add a check in case of zero volume fraction ---*/
                           p2[cell]         = EOS_phase2.pres_value(rho2);
                           p[cell]          = alpha1[cell]*p1[cell]
                                            + (1.0 - alpha1[cell])*p2[cell];
                           const auto H_lim = std::min(H[cell][0], Hmax);
                           typename Field::value_type p2_minus_p1_times_theta;
                           try {
                             p2_minus_p1_times_theta = rho1*(EOS_phase1.e_value(rho1_d) - EOS_phase1.e_value(rho1))
                                                     + p2[cell]*(rho1/rho1_d - 1.0);
                           }
                           catch(std::exception& e) {
                             std::cerr << e.what() << std::endl;
                             exit(1);
                           }
                           const auto fac_Ru = sigma*(3.0*H_lim/(kappa*rho1_d))*rho1 -
                                               sigma*H_lim +
                                               p2_minus_p1_times_theta;
                           if(fac_Ru > 0.0 &&
                              alpha1[cell] > alpha1_min && alpha1[cell] < alpha1_max &&
                              -grad_alpha1[cell][0]*conserved_variables[cell][RHO_U_INDEX]
                              -grad_alpha1[cell][1]*conserved_variables[cell][RHO_U_INDEX + 1] > 0.0 &&
                              alpha1_d[cell] < alpha1d_max) {
                             local_H_lig = std::max(H[cell][0], local_H_lig);
                           }

                           // Compute the integral quantities
                           local_m1_int += conserved_variables[cell][M1_INDEX]*std::pow(cell.length, dim);
                           local_m1_d_int += conserved_variables[cell][M1_D_INDEX]*std::pow(cell.length, dim);
                           local_alpha1_int += alpha1[cell]*std::pow(cell.length, dim);
                           local_grad_alpha1_int += std::sqrt(xt::sum(grad_alpha1[cell]*grad_alpha1[cell])())*std::pow(cell.length, dim);
                           Sigma_d[cell] = conserved_variables[cell][RHO_Z_INDEX]/std::pow(rho1_d, 2.0/3.0);
                           local_Sigma_d_int += Sigma_d[cell]*std::pow(cell.length, dim);
                           local_grad_alpha1_d_int += std::sqrt(xt::sum(grad_alpha1_d[cell]*grad_alpha1_d[cell])())*std::pow(cell.length, dim);
                           local_alpha1_d_int += alpha1_d[cell]*std::pow(cell.length, dim);
                           local_grad_alpha1_tot_int += std::sqrt(xt::sum((grad_alpha1[cell] + grad_alpha1_d[cell])*
                                                                          (grad_alpha1[cell] + grad_alpha1_d[cell]))())*std::pow(cell.length, dim);
                           local_alpha1_bar_int += alpha1_bar[cell]*std::pow(cell.length, dim);
                           local_grad_alpha1_bar_int += std::sqrt(xt::sum(grad_alpha1_bar[cell]*grad_alpha1_bar[cell])())*std::pow(cell.length, dim);
                         });

  /*--- Perform MPI collective operations ---*/
  typename Field::value_type global_H_lig;
  MPI_Allreduce(&local_H_lig, &global_H_lig, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  typename Field::value_type global_m1_int;
  MPI_Allreduce(&local_m1_int, &global_m1_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  typename Field::value_type global_m1_d_int;
  MPI_Allreduce(&local_m1_d_int, &global_m1_d_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  typename Field::value_type global_alpha1_int;
  MPI_Allreduce(&local_alpha1_int, &global_alpha1_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  typename Field::value_type global_grad_alpha1_int;
  MPI_Allreduce(&local_grad_alpha1_int, &global_grad_alpha1_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  typename Field::value_type global_Sigma_d_int;
  MPI_Allreduce(&local_Sigma_d_int, &global_Sigma_d_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  typename Field::value_type global_alpha1_d_int;
  MPI_Allreduce(&local_alpha1_d_int, &global_alpha1_d_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  typename Field::value_type global_grad_alpha1_d_int;
  MPI_Allreduce(&local_grad_alpha1_d_int, &global_grad_alpha1_d_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  typename Field::value_type global_grad_alpha1_tot_int;
  MPI_Allreduce(&local_grad_alpha1_tot_int, &global_grad_alpha1_tot_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  typename Field::value_type global_alpha1_bar_int;
  MPI_Allreduce(&local_alpha1_bar_int, &global_alpha1_bar_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  typename Field::value_type global_grad_alpha1_bar_int;
  MPI_Allreduce(&local_grad_alpha1_bar_int, &global_grad_alpha1_bar_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  /*--- Save the data ---*/
  Hlig                     << std::fixed << std::setprecision(12) << time << '\t' << global_H_lig               << std::endl;
  m1_integral              << std::fixed << std::setprecision(12) << time << '\t' << global_m1_int              << std::endl;
  m1_d_integral            << std::fixed << std::setprecision(12) << time << '\t' << global_m1_d_int            << std::endl;
  alpha1_integral          << std::fixed << std::setprecision(12) << time << '\t' << global_alpha1_int          << std::endl;
  grad_alpha1_integral     << std::fixed << std::setprecision(12) << time << '\t' << global_grad_alpha1_int     << std::endl;
  Sigma_d_integral         << std::fixed << std::setprecision(12) << time << '\t' << global_Sigma_d_int         << std::endl;
  alpha1_d_integral        << std::fixed << std::setprecision(12) << time << '\t' << global_alpha1_d_int        << std::endl;
  grad_alpha1_d_integral   << std::fixed << std::setprecision(12) << time << '\t' << global_grad_alpha1_d_int   << std::endl;
  grad_alpha1_tot_integral << std::fixed << std::setprecision(12) << time << '\t' << global_grad_alpha1_tot_int << std::endl;
  alpha1_bar_integral      << std::fixed << std::setprecision(12) << time << '\t' << global_alpha1_bar_int      << std::endl;
  grad_alpha1_bar_integral << std::fixed << std::setprecision(12) << time << '\t' << global_grad_alpha1_bar_int << std::endl;
}

/*---- IMPLEMENT THE FUNCTION THAT EFFECTIVELY SOLVES THE PROBLEM ---*/

// Implement the function that effectively performs the temporal loop
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::run() {
  /*--- Default output arguemnts ---*/
  fs::path path = fs::current_path();
  filename = "liquid_column";
  #ifdef RUSANOV_FLUX
    filename += "_Rusanov";
  #elifdef HLLC_FLUX
    filename += "_HLLC";
  #endif

  #ifdef ORDER_2
    filename += "_order2";
    #ifdef RELAX_RECONSTRUCTION
      filename += "_relaxed_reconstruction";
    #endif
  #else
    filename += "_order1";
  #endif

  if(mass_transfer)
    filename += "_mass_transfer";
  else
    filename += "_no_mass_transfer";

  const double dt_save = Tf/static_cast<double>(nfiles);

  /*--- Auxiliary variables to save updated fields ---*/
  #ifdef ORDER_2
    auto conserved_variables_tmp = samurai::make_vector_field<typename Field::value_type, EquationData::NVARS>("conserved_tmp", mesh);
    auto conserved_variables_old = samurai::make_vector_field<typename Field::value_type, EquationData::NVARS>("conserved_old", mesh);
  #endif
  auto conserved_variables_np1 = samurai::make_vector_field<typename Field::value_type, EquationData::NVARS>("conserved_np1", mesh);

  /*--- Create the flux variable ---*/
  #ifdef RUSANOV_FLUX
    #ifdef ORDER_2
      auto numerical_flux_hyp = Rusanov_flux.make_two_scale_capillarity(H);
    #else
      auto numerical_flux_hyp = Rusanov_flux.make_two_scale_capillarity();
    #endif
  #elifdef HLLC_FLUX
    #ifdef ORDER_2
      auto numerical_flux_hyp = HLLC_flux.make_two_scale_capillarity(H);
    #else
      auto numerical_flux_hyp = HLLC_flux.make_two_scale_capillarity();
    #endif
  #endif
  auto numerical_flux_st = SurfaceTension_flux.make_two_scale_capillarity(grad_alpha1);

  /*--- Save the initial condition ---*/
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(path, suffix_init, conserved_variables, alpha1, grad_alpha1, normal, H, p1, p2, p,
                          alpha1_d, Sigma_d, grad_alpha1_d, vel, div_vel,
                          alpha1_bar, grad_alpha1_bar, H_bar);
  Hlig.open("Hlig.dat", std::ofstream::out);
  m1_integral.open("m1_integral.dat", std::ofstream::out);
  m1_d_integral.open("m1_d_integral.dat", std::ofstream::out);
  alpha1_integral.open("alpha1_integral.dat", std::ofstream::out);
  grad_alpha1_integral.open("grad_alpha1_integral.dat", std::ofstream::out);
  Sigma_d_integral.open("Sigma_d_integral.dat", std::ofstream::out);
  alpha1_d_integral.open("alpha1_d_integral.dat", std::ofstream::out);
  grad_alpha1_d_integral.open("grad_alpha1_d_integral.dat", std::ofstream::out);
  grad_alpha1_tot_integral.open("grad_alpha1_tot_integral.dat", std::ofstream::out);
  alpha1_bar_integral.open("alpha1_bar_integral.dat", std::ofstream::out);
  grad_alpha1_bar_integral.open("grad_alpha1_bar_integral.dat", std::ofstream::out);
  double t = 0.0;
  execute_postprocess(t);

  /*--- Set initial time step ---*/
  const double dx = mesh.cell_length(mesh.max_level());
  double dt       = std::min(Tf - t, cfl*dx/get_max_lambda());
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  using mesh_id_t = typename decltype(mesh)::mesh_id_t;
  const auto n_elements_per_subdomain = mesh[mesh_id_t::cells].nb_cells();
  unsigned int n_elements;
  MPI_Allreduce(&n_elements_per_subdomain, &n_elements, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
  if(rank == 0) {
    std::cout << "Number of initial elements = " <<  n_elements << std::endl;
    std::cout << std::endl;
  }

  /*--- Start the loop ---*/
  std::size_t nsave = 0;
  std::size_t nt    = 0;
  while(t != Tf) {
    t += dt;
    if(t > Tf) {
      dt += Tf - t;
      t = Tf;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) {
      std::cout << fmt::format("Iteration {}: t = {}, dt = {}", ++nt, t, dt) << std::endl;
    }

    // Apply mesh adaptation
    perform_mesh_adaptation();

    // Save current state in case of order 2
    #ifdef ORDER_2
      conserved_variables_old.resize();
      conserved_variables_old = conserved_variables;
    #endif

    // Apply the numerical scheme without relaxation
    // Convective operator
    samurai::update_ghost_mr(conserved_variables);
    samurai::update_ghost_mr(H_bar);
    try {
      auto flux_hyp = numerical_flux_hyp(conserved_variables);
      #ifdef ORDER_2
        conserved_variables_tmp.resize();
        conserved_variables_tmp = conserved_variables - dt*flux_hyp;
        std::swap(conserved_variables.array(), conserved_variables_tmp.array());
      #else
        conserved_variables_np1.resize();
        conserved_variables_np1 = conserved_variables - dt*flux_hyp;
        std::swap(conserved_variables.array(), conserved_variables_np1.array());
      #endif
    }
    catch(std::exception& e) {
      std::cerr << e.what() << std::endl;
      save(fs::current_path(), "_diverged", conserved_variables, alpha1);
      exit(1);
    }

    // Update the geometry to recompute volume fraction gradient
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                              alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/
                                             (conserved_variables[cell][M1_INDEX] +
                                              conserved_variables[cell][M2_INDEX] +
                                              conserved_variables[cell][M1_D_INDEX]);
                           });
    #ifdef VERBOSE
      check_data();
    #endif
    normal.resize();
    H.resize();
    grad_alpha1.resize();
    update_geometry();

    // Capillarity contribution
    samurai::update_ghost_mr(conserved_variables);
    auto flux_st = numerical_flux_st(conserved_variables);
    #ifdef ORDER_2
      conserved_variables_tmp = conserved_variables - dt*flux_st;
      std::swap(conserved_variables.array(), conserved_variables_tmp.array());
    #else
      conserved_variables_np1 = conserved_variables - dt*flux_st;
      std::swap(conserved_variables.array(), conserved_variables_np1.array());
    #endif

    // Apply relaxation
    if(apply_relax) {
      // Apply relaxation if desired, which will modify alpha1 and, consequently, for what
      // concerns next time step, rho_alpha1 (as well as grad_alpha1).
      dalpha1.resize();
      to_be_relaxed.resize();
      Newton_iterations.resize();
      apply_relaxation();
      #ifdef RELAX_RECONSTRUCTION
        update_geometry();
      #endif
    }

    /*--- Consider the second stage for the second order ---*/
    #ifdef ORDER_2
      // Apply the numerical scheme
      // Convective operator
      samurai::update_ghost_mr(conserved_variables);
      samurai::update_ghost_mr(H_bar);
      try {
        auto flux_hyp = numerical_flux_hyp(conserved_variables);
        conserved_variables_tmp = conserved_variables - dt*flux_hyp;
        std::swap(conserved_variables.array(), conserved_variables_tmp.array());
      }
      catch(std::exception& e) {
        std::cerr << e.what() << std::endl;
        save(fs::current_path(), "_diverged", conserved_variables, alpha1);
        exit(1);
      }

      // Recompute geometrical quantities
      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                             {
                                alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/
                                               (conserved_variables[cell][M1_INDEX] +
                                                conserved_variables[cell][M2_INDEX] +
                                                conserved_variables[cell][M1_D_INDEX]);
                             });
      #ifdef VERBOSE
        check_data();
      #endif
      update_geometry();

      // Capillarity contribution
      samurai::update_ghost_mr(conserved_variables);
      flux_st = numerical_flux_st(conserved_variables);
      conserved_variables_tmp = conserved_variables - dt*flux_st;
      std::swap(conserved_variables.array(), conserved_variables_tmp.array());

      // Apply relaxation
      if(apply_relax) {
        // Apply relaxation if desired, which will modify alpha1 and, consequently, for what
        // concerns next time step, rho_alpha1 (as well as grad_alpha1).
        apply_relaxation();
      }

      // Complete evaluation
      conserved_variables_np1.resize();
      conserved_variables_np1 = 0.5*(conserved_variables_old + conserved_variables);
      std::swap(conserved_variables.array(), conserved_variables_np1.array());

      // Recompute volume fraction gradient and curvature for the next time step
      #ifdef RELAX_RECONSTRUCTION
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                  alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/
                                                 (conserved_variables[cell][M1_INDEX] +
                                                  conserved_variables[cell][M2_INDEX] +
                                                  conserved_variables[cell][M1_D_INDEX]);
                               });
        update_geometry();
      #endif
    #endif

    /*--- Compute updated time step ---*/
    dt = std::min(Tf - t, cfl*dx/get_max_lambda());

    /*--- Postprocess data ---*/
    execute_postprocess(t);

    /*--- Save the results ---*/
    if(t >= static_cast<double>(nsave + 1) * dt_save || t == Tf) {
      // Resize all the fields
      vel.resize();
      div_vel.resize();

      // Compute auxiliary variables for saving
      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                             {
                               // Save velocity field
                               const auto rho = conserved_variables[cell][M1_INDEX]
                                              + conserved_variables[cell][M2_INDEX]
                                              + conserved_variables[cell][M1_D_INDEX];
                               vel[cell][0]   = conserved_variables[cell][RHO_U_INDEX]/rho;
                               vel[cell][1]   = conserved_variables[cell][RHO_U_INDEX + 1]/rho;
                             });

      samurai::update_ghost_mr(vel);
      div_vel = divergence(vel);

      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                             {
                               const auto mod_grad_alpha1_bar = std::sqrt(xt::sum(grad_alpha1_bar[cell]*grad_alpha1_bar[cell])());

                               if(mod_grad_alpha1_bar > mod_grad_alpha1_min) {
                                 normal_bar[cell] = grad_alpha1_bar[cell]/mod_grad_alpha1_bar;
                               }
                               else {
                                 for(std::size_t d = 0; d < dim; ++d) {
                                   normal_bar[cell][d] = nan("");
                                 }
                               }
                             });

      samurai::update_ghost_mr(normal_bar);
      H_bar = -divergence(normal_bar);

      // Perform the saving
      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";
      save(path, suffix, conserved_variables, alpha1, grad_alpha1, normal, H, p1, p2, p,
                         alpha1_d, Sigma_d, grad_alpha1_d, vel, div_vel,
                         alpha1_bar, grad_alpha1_bar, H_bar, Newton_iterations);
    }
  }

  /*--- Close the files for post-proessing ---*/
  Hlig.close();
  m1_integral.close();
  m1_d_integral.close();
  alpha1_integral.close();
  grad_alpha1_integral.close();
  Sigma_d_integral.close();
  alpha1_d_integral.close();
  grad_alpha1_d_integral.close();
  grad_alpha1_tot_integral.close();
  alpha1_bar_integral.close();
  grad_alpha1_bar_integral.close();
}
