// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
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
                      const Simulation_Paramaters<double>& sim_param,
                      const EOS_Parameters<double>& eos_param); /*--- Class constrcutor with the arguments related
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

  const typename Field::value_type Tf; /*--- Final time of the simulation ---*/

  const typename Field::value_type sigma; /*--- Surface tension coefficient ---*/

  bool apply_relax; /*--- Choose whether to apply or not the relaxation ---*/

  const bool                       mass_transfer; /*--- Choose wheter to apply or not the mass transfer ---*/
  const typename Field::value_type Hmax;          /*--- Threshold length scale ---*/
  const typename Field::value_type kappa;         /*--- Parameter related to the radius of small-scale droplets ---*/
  const typename Field::value_type alpha_d_max;   /*--- Maximum threshold of small-scale volume fraction ---*/
  const typename Field::value_type alpha_l_min;   /*--- Minimum large-scale volume fraction to identify the mixture region ---*/
  const typename Field::value_type alpha_l_max;   /*--- Maximum large-scale volume fraction to identify the mixture region ---*/

  typename Field::value_type cfl; /*--- Courant number of the simulation so as to compute the time step ---*/

  const typename Field::value_type mod_grad_alpha_l_min; /*--- Minimum threshold for which not computing anymore the unit normal ---*/

  const typename Field::value_type lambda;           /*--- Parameter for bound preserving strategy ---*/
  const typename Field::value_type atol_Newton;      /*--- Absolute tolerance Newton method relaxation ---*/
  const typename Field::value_type rtol_Newton;      /*--- Relative tolerance Newton method relaxation ---*/
  const std::size_t                max_Newton_iters; /*--- Maximum number of Newton iterations ---*/

  double MR_param;      /*--- Multiresolution parameter ---*/
  double MR_regularity; /*--- Multiresolution regularity ---*/

  LinearizedBarotropicEOS<typename Field::value_type> EOS_phase_liq,
                                                      EOS_phase_gas; /*--- The two variables which take care of the
                                                                           barotropic EOS to compute the speed of sound ---*/

  #ifdef RUSANOV_FLUX
    samurai::RusanovFlux<Field> Rusanov_flux; /*--- Auxiliary variable to compute the flux for the hyperbolic operator ---*/
  #elifdef HLLC_FLUX
    samurai::HLLCFlux<Field> HLLC_flux; /*--- Auxiliary variable to compute the flux ---*/
  #endif
  samurai::SurfaceTensionFlux<Field> SurfaceTension_flux; /*--- Auxiliary variable to compute the contribution associated to surface tension ---*/

  std::size_t nfiles; /*--- Number of files desired for output ---*/

  std::string filename; /*--- Auxiliary variable to store the name of output ---*/

  Field conserved_variables; /*--- The variable which stores the conserved variables,
                                   namely the varialbes for which we solve a PDE system ---*/

  /*--- Now we declare a bunch of fields which depend from the state, but it is useful
        to have it so as to avoid recomputation ---*/
  Field_Scalar alpha_l,
               dalpha_l,
               p_liq,
               p_g,
               p;

  Field_Vect normal,
             grad_alpha_l;

  Field_Scalar alpha_d,
               alpha_l_bar,
               Sigma_d;

  Field_Vect grad_alpha_d,
             vel,
             normal_bar,
             grad_alpha_l_bar;

  Field_ScalarVector H,
                     H_bar,
                     div_vel;

  Field_Scalar Mach;

  samurai::ScalarField<decltype(mesh), std::size_t> to_be_relaxed;
  samurai::ScalarField<decltype(mesh), std::size_t> Newton_iterations;

  using gradient_type = decltype(samurai::make_gradient_order2<decltype(alpha_l)>());
  gradient_type gradient;

  using divergence_type = decltype(samurai::make_divergence_order2<decltype(normal)>());
  divergence_type divergence;

  /*--- Auxiliary output streams for post-processing ---*/
  std::ofstream Hlig;
  std::ofstream m_l_integral;
  std::ofstream m_d_integral;
  std::ofstream alpha_l_integral;
  std::ofstream grad_alpha_l_integral;
  std::ofstream Sigma_d_integral;
  std::ofstream alpha_d_integral;
  std::ofstream grad_alpha_d_integral;
  std::ofstream grad_alpha_l_tot_integral;
  std::ofstream alpha_l_bar_integral;
  std::ofstream grad_alpha_l_bar_integral;

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  void update_geometry(); /*--- Auxiliary routine to compute normals and curvature ---*/

  void init_variables(const typename Field::value_type x0, const typename Field::value_type y0,
                      const typename Field::value_type U0, const typename Field::value_type U1, const typename Field::value_type V0,
                      const typename Field::value_type R, const typename Field::value_type eps_over_R,
                      const typename Field::value_type alpha_residual); /*--- Routine to initialize the variables
                                                                              (both conserved and auxiliary, this is problem dependent) ---*/

  typename Field::value_type get_max_lambda(); /*--- Compute the estimate of the maximum eigenvalue ---*/

  void check_data(unsigned int flag = 0); /*--- Auxiliary routine to check if spurious values are present ---*/

  void perform_mesh_adaptation(); /*--- Perform the mesh adaptation ---*/

  void apply_relaxation(); /*--- Apply the relaxation ---*/

  template<typename State, typename Gradient>
  void perform_Newton_step_relaxation(State local_conserved_variables,
                                      const typename Field::value_type H_loc,
                                      typename Field::value_type& dalpha_l_loc,
                                      typename Field::value_type& alpha_l_loc,
                                      std::size_t& to_be_relaxed_loc,
                                      std::size_t& Newton_iterations_loc,
                                      bool& local_relaxation_applied,
                                      const Gradient& grad_alpha_l_loc,
                                      const bool mass_transfer_NR);

  void execute_postprocess(const typename Field::value_type time); /*--- Execute the postprocess ---*/
};

/*---- START WITH THE IMPLEMENTATION OF THE CONSTRUCTOR ---*/

// Implement class constructor
//
template<std::size_t dim>
TwoScaleCapillarity<dim>::TwoScaleCapillarity(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                                              const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                                              const Simulation_Paramaters<double>& sim_param,
                                              const EOS_Parameters<double>& eos_param):
  box(min_corner, max_corner), mesh(box, sim_param.min_level, sim_param.max_level, {{false, true}}),
  Tf(sim_param.Tf), sigma(sim_param.sigma),
  apply_relax(sim_param.apply_relaxation),
  mass_transfer(sim_param.mass_transfer), Hmax(sim_param.Hmax),
  kappa(sim_param.kappa), alpha_d_max(sim_param.alpha_d_max),
  alpha_l_min(sim_param.alpha_l_min), alpha_l_max(sim_param.alpha_l_max),
  cfl(sim_param.Courant), mod_grad_alpha_l_min(sim_param.mod_grad_alpha_l_min),
  lambda(sim_param.lambda), atol_Newton(sim_param.atol_Newton),
  rtol_Newton(sim_param.rtol_Newton), max_Newton_iters(sim_param.max_Newton_iters),
  MR_param(sim_param.MR_param), MR_regularity(sim_param.MR_regularity),
  EOS_phase_liq(eos_param.p0_phase1, eos_param.rho0_phase1, eos_param.c0_phase1),
  EOS_phase_gas(eos_param.p0_phase2, eos_param.rho0_phase2, eos_param.c0_phase2),
  #ifdef RUSANOV_FLUX
    Rusanov_flux(EOS_phase_liq, EOS_phase_gas,
                 sigma, mod_grad_alpha_l_min,
                 lambda, atol_Newton, rtol_Newton, max_Newton_iters),
  #elifdef HLLC_FLUX
    HLLC_flux(EOS_phase_liq, EOS_phase_gas,
              sigma, mod_grad_alpha_l_min,
              lambda, atol_Newton, rtol_Newton, max_Newton_iters),
  #endif
  SurfaceTension_flux(EOS_phase_liq, EOS_phase_gas,
                      sigma, mod_grad_alpha_l_min,
                      lambda, atol_Newton, rtol_Newton, max_Newton_iters),
  nfiles(sim_param.nfiles),
  gradient(samurai::make_gradient_order2<decltype(alpha_l)>()),
  divergence(samurai::make_divergence_order2<decltype(normal)>())
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) {
      std::cout << "Initializing variables " << std::endl;
      std::cout << std::endl;
    }
    init_variables(sim_param.x0, sim_param.y0,
                   sim_param.U0, sim_param.U1, sim_param.V0,
                   sim_param.R, sim_param.eps_over_R,
                   sim_param.alpha_residual);
    to_be_relaxed     = samurai::make_scalar_field<std::size_t>("to_be_relaxed", mesh);
    Newton_iterations = samurai::make_scalar_field<std::size_t>("Newton_iterations", mesh);
  }

// Auxiliary routine to compute normals and curvature
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::update_geometry() {
  samurai::update_ghost_mr(alpha_l);

  grad_alpha_l = gradient(alpha_l);

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              const auto mod_grad_alpha_l = std::sqrt(xt::sum(grad_alpha_l[cell]*grad_alpha_l[cell])());

                              if(mod_grad_alpha_l > mod_grad_alpha_l_min) {
                                normal[cell] = grad_alpha_l[cell]/mod_grad_alpha_l;
                              }
                              else {
                                for(std::size_t d = 0; d < dim; ++d) {
                                  normal[cell][d] = static_cast<typename Field::value_type>(nan(""));
                                }
                              }
                            }
                        );
  samurai::update_ghost_mr(normal);
  H = -divergence(normal);
}

// Initialization of conserved and auxiliary variables
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::init_variables(const typename Field::value_type x0, const typename Field::value_type y0,
                                              const typename Field::value_type U0, const typename Field::value_type U1, const typename Field::value_type V0,
                                              const typename Field::value_type R, const typename Field::value_type eps_over_R,
                                              const typename Field::value_type alpha_residual) {
  /*--- Create conserved and auxiliary fields ---*/
  conserved_variables = samurai::make_vector_field<typename Field::value_type, EquationData::NVARS>("conserved", mesh);

  alpha_l      = samurai::make_scalar_field<typename Field::value_type>("alpha_l", mesh);
  grad_alpha_l = samurai::make_vector_field<typename Field::value_type, dim>("grad_alpha_l", mesh);
  normal       = samurai::make_vector_field<typename Field::value_type, dim>("normal", mesh);
  H            = samurai::make_vector_field<typename Field::value_type, 1>("H", mesh);

  dalpha_l     = samurai::make_scalar_field<typename Field::value_type>("dalpha_l", mesh);

  p_liq        = samurai::make_scalar_field<typename Field::value_type>("p_liq", mesh);
  p_g          = samurai::make_scalar_field<typename Field::value_type>("p_g", mesh);
  p            = samurai::make_scalar_field<typename Field::value_type>("p", mesh);

  alpha_d      = samurai::make_scalar_field<typename Field::value_type>("alpha_d", mesh);
  grad_alpha_d = samurai::make_vector_field<typename Field::value_type, dim>("grad_alpha_d", mesh);
  Sigma_d      = samurai::make_scalar_field<typename Field::value_type>("Sigma_d", mesh);
  vel          = samurai::make_vector_field<typename Field::value_type, dim>("vel", mesh);
  div_vel      = samurai::make_vector_field<typename Field::value_type, 1>("div_vel", mesh);

  alpha_l_bar      = samurai::make_scalar_field<typename Field::value_type>("alpha_l_bar", mesh);
  grad_alpha_l_bar = samurai::make_vector_field<typename Field::value_type, dim>("grad_alpha_l_bar", mesh);
  normal_bar       = samurai::make_vector_field<typename Field::value_type, dim>("normal_bar", mesh);
  H_bar            = samurai::make_vector_field<typename Field::value_type, 1>("H_bar", mesh);

  Mach = samurai::make_scalar_field<typename Field::value_type>("Mach", mesh);

  /*--- Declare some constant parameters associated to the initial state ---*/
  const auto eps_R = eps_over_R*R;

  /*--- Initialize some fields to define the bubble with a loop over all cells ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Set large-scale volume fraction
                              const auto center = cell.center();
                              const auto x      = static_cast<typename Field::value_type>(center[0]);
                              const auto y      = static_cast<typename Field::value_type>(center[1]);

                              const auto r  = std::sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));

                              const auto w  = (r >= R && r < R + eps_R) ?
                                              std::max(std::exp(static_cast<typename Field::value_type>(2.0)*
                                                                (r - R)*(r - R)/(eps_R*eps_R)*
                                                                ((r - R)*(r - R)/(eps_R*eps_R) - static_cast<typename Field::value_type>(3.0))/
                                                                (((r - R)*(r - R)/(eps_R*eps_R) - static_cast<typename Field::value_type>(1.0))*
                                                                ((r - R)*(r - R)/(eps_R*eps_R) - static_cast<typename Field::value_type>(1.0)))),
                                                       static_cast<typename Field::value_type>(0.0)) :
                                              ((r < R) ? static_cast<typename Field::value_type>(1.0) :
                                                         static_cast<typename Field::value_type>(0.0));

                              alpha_l[cell] = std::min(std::max(alpha_residual, w),
                                                       static_cast<typename Field::value_type>(1.0) - alpha_residual);
                            }
                        );

  /*--- Compute the geometrical quantities ---*/
  update_geometry();

  /*--- Loop over a cell to complete the remaining variables ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Set small-scale variables
                              alpha_d[cell]                          = static_cast<typename Field::value_type>(0.0);
                              conserved_variables[cell][RHO_Z_INDEX] = static_cast<typename Field::value_type>(0.0);
                              Sigma_d[cell]                          = static_cast<typename Field::value_type>(0.0);
                              conserved_variables[cell][Md_INDEX]    = alpha_d[cell]*EOS_phase_liq.get_rho0();

                              // Recompute geometric locations to set partial masses
                              const auto center = cell.center();
                              const auto x      = static_cast<typename Field::value_type>(center[0]);
                              const auto y      = static_cast<typename Field::value_type>(center[1]);

                              const auto r = std::sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));

                              // Set mass large-scale phase 1
                              if(r >= R + eps_R) {
                                p_liq[cell] = EOS_phase_liq.get_p0();
                              }
                              else {
                                p_liq[cell] = EOS_phase_gas.get_p0();
                                if(r >= R && r < R + eps_R && !std::isnan(H[cell][0])) {
                                  p_liq[cell] += sigma*H[cell][0];
                                }
                                else {
                                  p_liq[cell] += sigma/R;
                                }
                              }
                              const auto rho_liq = EOS_phase_liq.rho_value(p_liq[cell]);

                              alpha_l_bar[cell] = alpha_l[cell]/(static_cast<typename Field::value_type>(1.0) - alpha_d[cell]);
                              conserved_variables[cell][Ml_INDEX] = alpha_l[cell]*rho_liq;

                              // Set mass phase 2
                              p_g[cell] = EOS_phase_gas.get_p0();
                              const auto rho_g = EOS_phase_gas.rho_value(p_g[cell]);

                              const auto alpha_g = static_cast<typename Field::value_type>(1.0) - alpha_l[cell] - alpha_d[cell];
                              conserved_variables[cell][Mg_INDEX] = alpha_g*rho_g;

                              // Set mixture pressure
                              p[cell] = (alpha_l[cell] + alpha_d[cell])*p_liq[cell]
                                      + alpha_g*p_g[cell] - static_cast<typename Field::value_type>(2.0/3.0)*sigma*Sigma_d[cell];

                              // Set conserved variable associated to large-scale volume fraction
                              const auto rho = conserved_variables[cell][Ml_INDEX]
                                             + conserved_variables[cell][Mg_INDEX]
                                             + conserved_variables[cell][Md_INDEX];

                              conserved_variables[cell][RHO_ALPHA_l_INDEX] = rho*alpha_l[cell];

                              // Set momentum
                              conserved_variables[cell][RHO_U_INDEX]     = conserved_variables[cell][Ml_INDEX]*U1
                                                                         + conserved_variables[cell][Mg_INDEX]*U0;
                              conserved_variables[cell][RHO_U_INDEX + 1] = rho*V0;

                              typename Field::value_type norm2_vel = 0.0;
                              for(std::size_t d = 0; d < dim; ++d) {
                                vel[cell][d] = conserved_variables[cell][RHO_U_INDEX + d]/rho;
                                norm2_vel += vel[cell][d]*vel[cell][d];
                              }

                              const auto Y_g = conserved_variables[cell](Mg_INDEX)/rho;
                              const auto cf  = std::sqrt((static_cast<typename Field::value_type>(1.0) - Y_g)*
                                                         EOS_phase_liq.c_value(rho_liq)*
                                                         EOS_phase_liq.c_value(rho_liq) +
                                                         Y_g*
                                                         EOS_phase_gas.c_value(rho_g)*
                                                         EOS_phase_gas.c_value(rho_g));
                              Mach[cell]     = std::sqrt(norm2_vel)/cf;
                            }
                        );

  /*--- Set useful small-scale related fields ---*/
  samurai::update_ghost_mr(alpha_d);
  grad_alpha_d = gradient(alpha_d);

  samurai::update_ghost_mr(vel);
  div_vel = divergence(vel);

  /*--- Set auxiliary gradient alpha_l_bar volume fraction ---*/
  samurai::update_ghost_mr(alpha_l_bar);
  grad_alpha_l_bar = gradient(alpha_l_bar);
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              const auto mod_grad_alpha_l_bar = std::sqrt(xt::sum(grad_alpha_l_bar[cell]*grad_alpha_l_bar[cell])());

                              if(mod_grad_alpha_l_bar > mod_grad_alpha_l_min) {
                                normal_bar[cell] = grad_alpha_l_bar[cell]/mod_grad_alpha_l_bar;
                              }
                              else {
                                for(std::size_t d = 0; d < dim; ++d) {
                                  normal_bar[cell][d] = static_cast<typename Field::value_type>(nan(""));
                                }
                              }
                            }
                        );
  samurai::update_ghost_mr(normal_bar);
  H_bar = -divergence(normal_bar);

  /*--- Apply bcs ---*/
  const samurai::DirectionVector<dim> left = {-1, 0};
  samurai::make_bc<Default>(conserved_variables,
                            Inlet(conserved_variables, U0, V0, alpha_residual,
                                  static_cast<typename Field::value_type>(0.0), static_cast<typename Field::value_type>(0.0)))->on(left);

  const samurai::DirectionVector<dim> right = {1, 0};
  samurai::make_bc<samurai::Neumann<1>>(conserved_variables,
                                        static_cast<typename Field::value_type>(0.0),
                                        static_cast<typename Field::value_type>(0.0),
                                        static_cast<typename Field::value_type>(0.0),
                                        static_cast<typename Field::value_type>(0.0),
                                        static_cast<typename Field::value_type>(0.0),
                                        static_cast<typename Field::value_type>(0.0),
                                        static_cast<typename Field::value_type>(0.0))->on(right);

  /*const samurai::DirectionVector<dim> top = {0, 1};
  samurai::make_bc<samurai::Neumann<1>>(conserved_variables,
                                        static_cast<typename Field::value_type>(0.0),
                                        static_cast<typename Field::value_type>(0.0),
                                        static_cast<typename Field::value_type>(0.0),
                                        static_cast<typename Field::value_type>(0.0),
                                        static_cast<typename Field::value_type>(0.0),
                                        static_cast<typename Field::value_type>(0.0),
                                        static_cast<typename Field::value_type>(0.0))->on(top);

  const samurai::DirectionVector<dim> bottom = {0, -1};
  samurai::make_bc<samurai::Neumann<1>>(conserved_variables,
                                        static_cast<typename Field::value_type>(0.0),
                                        static_cast<typename Field::value_type>(0.0),
                                        static_cast<typename Field::value_type>(0.0),
                                        static_cast<typename Field::value_type>(0.0),
                                        static_cast<typename Field::value_type>(0.0),
                                        static_cast<typename Field::value_type>(0.0),
                                        static_cast<typename Field::value_type>(0.0))->on(bottom);*/
}

/*---- FOCUS NOW ON THE AUXILIARY FUNCTIONS ---*/

// Compute the estimate of the maximum eigenvalue for CFL condition
//
template<std::size_t dim>
typename TwoScaleCapillarity<dim>::Field::value_type TwoScaleCapillarity<dim>::get_max_lambda() {
  auto local_res = static_cast<typename Field::value_type>(0.0);

  alpha_l.resize();
  alpha_d.resize();
  Sigma_d.resize();
  vel.resize();
  Mach.resize();
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              #ifndef RELAX_RECONSTRUCTION
                                alpha_l[cell] = conserved_variables[cell][RHO_ALPHA_l_INDEX]/
                                                (conserved_variables[cell][Ml_INDEX] +
                                                 conserved_variables[cell][Mg_INDEX] +
                                                 conserved_variables[cell][Md_INDEX]);
                              #endif

                              /*--- Compute the velocity along all the directions ---*/
                              const auto rho = conserved_variables[cell][Ml_INDEX]
                                             + conserved_variables[cell][Mg_INDEX]
                                             + conserved_variables[cell][Md_INDEX];
                              typename Field::value_type norm2_vel = 0.0;
                              for(std::size_t d = 0; d < dim; ++d) {
                                vel[cell][d] = conserved_variables[cell][RHO_U_INDEX + d]/rho;
                                norm2_vel += vel[cell][d]*vel[cell][d];
                              }

                              /*--- Compute frozen speed of sound ---*/
                              const auto rho_liq = (conserved_variables[cell][Ml_INDEX] + conserved_variables[cell][Md_INDEX])/
                                                   (alpha_l[cell] + alpha_d[cell]); /*--- TODO: Add a check in case of zero volume fraction ---*/
                              alpha_d[cell]      = alpha_l[cell]*conserved_variables[cell](Md_INDEX)/conserved_variables[cell](Ml_INDEX);
                              /*--- TODO: Add a check in case of zero volume fraction ---*/
                              const auto alpha_g = static_cast<typename Field::value_type>(1.0) - alpha_l[cell] - alpha_d[cell];
                              const auto rho_g   = conserved_variables[cell][Mg_INDEX]/alpha_g; /*--- TODO: Add a check in case of zero volume fraction ---*/
                              const auto Y_g     = conserved_variables[cell](Mg_INDEX)/rho;
                              Sigma_d[cell]      = conserved_variables[cell](RHO_Z_INDEX)/
                                                   std::pow(rho_liq, static_cast<typename Field::value_type>(2.0/3.0));
                              const auto c       = std::sqrt((static_cast<typename Field::value_type>(1.0) - Y_g)*
                                                             EOS_phase_liq.c_value(rho_liq)*
                                                             EOS_phase_liq.c_value(rho_liq) +
                                                             Y_g*
                                                             EOS_phase_gas.c_value(rho_g)*
                                                             EOS_phase_gas.c_value(rho_g) -
                                                             static_cast<typename Field::value_type>(2.0/9.0)*sigma*Sigma_d[cell]/rho);

                              const auto cf = std::sqrt((static_cast<typename Field::value_type>(1.0) - Y_g)*
                                                        EOS_phase_liq.c_value(rho_liq)*
                                                        EOS_phase_liq.c_value(rho_liq) +
                                                        Y_g*
                                                        EOS_phase_gas.c_value(rho_g)*
                                                        EOS_phase_gas.c_value(rho_g));
                              Mach[cell]    = std::sqrt(norm2_vel)/cf;

                              /*--- Add term due to surface tension ---*/
                              const auto r = sigma*std::sqrt(xt::sum(grad_alpha_l[cell]*grad_alpha_l[cell])())/(rho*c*c);

                              /*--- Update eigenvalue estimate ---*/
                              local_res = std::max(std::max(std::abs(vel[cell][0]) + c*(static_cast<typename Field::value_type>(1.0) +
                                                                                        static_cast<typename Field::value_type>(0.125)*r),
                                                            std::abs(vel[cell][1]) + c*(static_cast<typename Field::value_type>(1.0) +
                                                                                        static_cast<typename Field::value_type>(0.125)*r)),
                                                   local_res);
                            }
                        );

  typename Field::value_type global_res;
  MPI_Allreduce(&local_res, &global_res, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  return global_res;
}

// Perform the mesh adaptation strategy.
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::perform_mesh_adaptation() {
  samurai::update_ghost_mr(grad_alpha_l);
  auto MRadaptation = samurai::make_MRAdapt(grad_alpha_l);
  MRadaptation(MR_param, MR_regularity, conserved_variables);

  /*--- Sanity check after mesh adaptation ---*/
  alpha_l.resize();
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              alpha_l[cell] = conserved_variables[cell][RHO_ALPHA_l_INDEX]/
                                              (conserved_variables[cell][Ml_INDEX] +
                                               conserved_variables[cell][Mg_INDEX] +
                                               conserved_variables[cell][Md_INDEX]);
                            }
                        );
  #ifdef VERBOSE
    check_data(1);
  #endif
}

// Auxiliary function to check if spurious values are present
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
                              // Sanity check for alpha_l
                              if(alpha_l[cell] < static_cast<typename Field::value_type>(0.0)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "Negative volume fraction large-scale liquid " + op << std::endl;
                                save(fs::current_path(), "_diverged", conserved_variables, alpha_l);
                                exit(1);
                              }
                              else if(alpha_l[cell] > static_cast<typename Field::value_type>(1.0)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "Exceeding volume fraction large-scale liquid " + op << std::endl;
                                save(fs::current_path(), "_diverged", conserved_variables, alpha_l);
                                exit(1);
                              }
                              else if(std::isnan(alpha_l[cell])) {
                                std::cerr << cell << std::endl;
                                std::cerr << "NaN volume fraction large-scale liquid " + op << std::endl;
                                save(fs::current_path(), "_diverged", conserved_variables, alpha_l);
                                exit(1);
                              }

                              // Sanity check for m_l
                              if(conserved_variables[cell][Ml_INDEX] < static_cast<typename Field::value_type>(0.0)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "Negative mass large-scale liquid " + op << std::endl;
                                save(fs::current_path(), "_diverged", conserved_variables, alpha_l);
                                exit(1);
                              }
                              else if(std::isnan(conserved_variables[cell][Ml_INDEX])) {
                                std::cerr << cell << std::endl;
                                std::cerr << "NaN mass large-scale liquid " + op << std::endl;
                                save(fs::current_path(), "_diverged", conserved_variables, alpha_l);
                                exit(1);
                              }

                              // Sanity check for m_g
                              if(conserved_variables[cell][Mg_INDEX] < static_cast<typename Field::value_type>(0.0)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "Negative mass gas phase " + op << std::endl;
                                save(fs::current_path(), "_diverged", conserved_variables, alpha_l);
                                exit(1);
                              }
                              else if(std::isnan(conserved_variables[cell][Mg_INDEX])) {
                                std::cerr << cell << std::endl;
                                std::cerr << "NaN mass gas phase " + op << std::endl;
                                save(fs::current_path(), "_diverged", conserved_variables, alpha_l);
                                exit(1);
                              }

                              // Sanity check for m_d
                              if(conserved_variables[cell][Md_INDEX] < static_cast<typename Field::value_type>(-1e-15)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "Negative mass small-scale liquid " + op << std::endl;
                                save(fs::current_path(), "_diverged", conserved_variables, alpha_l);
                                exit(1);
                              }
                              else if(std::isnan(conserved_variables[cell][Md_INDEX])) {
                                std::cerr << cell << std::endl;
                                std::cerr << "NaN mass small-scale liquid " + op << std::endl;
                                save(fs::current_path(), "_diverged", conserved_variables, alpha_l);
                                exit(1);
                              }

                              // Sanity check for z (the transported variable related to small-scale IAD)
                              if(conserved_variables[cell][RHO_Z_INDEX] < static_cast<typename Field::value_type>(-1e-15)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "Negative interface area small-scale liquid " + op << std::endl;
                                save(fs::current_path(), "_diverged", conserved_variables, alpha_l);
                                exit(1);
                              }
                              else if(std::isnan(conserved_variables[cell][RHO_Z_INDEX])) {
                                std::cerr << cell << std::endl;
                                std::cerr << "NaN interface area small-scale liquid " + op << std::endl;
                                save(fs::current_path(), "_diverged", conserved_variables, alpha_l);
                                exit(1);
                              }
                            }
                        );
}

// Apply the relaxation. This procedure is valid for a generic EOS
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::apply_relaxation() {
  /*--- Initialize the variables ---*/
  samurai::times::timers.start("apply_relaxation");

  std::size_t Newton_iter = 0;
  Newton_iterations.fill(0);
  dalpha_l.fill(std::numeric_limits<typename Field::value_type>::infinity());
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
                                  perform_Newton_step_relaxation(conserved_variables[cell],
                                                                 H[cell][0], dalpha_l[cell], alpha_l[cell],
                                                                 to_be_relaxed[cell], Newton_iterations[cell], local_relaxation_applied,
                                                                 grad_alpha_l[cell], mass_transfer_NR);
                                }
                                catch(std::exception& e) {
                                  std::cerr << e.what() << std::endl;
                                  std::cerr << cell << std::endl;
                                  save(fs::current_path(), "_diverged",
                                       conserved_variables, alpha_l, dalpha_l, grad_alpha_l, normal, H,
                                       to_be_relaxed, Newton_iterations);
                                  exit(1);
                                }
                              }
                          );

    mpi::communicator world;
    boost::mpi::all_reduce(world, local_relaxation_applied, global_relaxation_applied, std::logical_or<bool>());

    // Newton cycle diverged
    if(Newton_iter > max_Newton_iters && global_relaxation_applied == true) {
      std::cerr << "Netwon method not converged in the post-hyperbolic relaxation" << std::endl;
      save(fs::current_path(), "_diverged",
           conserved_variables, alpha_l, dalpha_l, grad_alpha_l, normal, H,
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
void TwoScaleCapillarity<dim>::perform_Newton_step_relaxation(State local_conserved_variables,
                                                              const typename Field::value_type H_loc,
                                                              typename Field::value_type& dalpha_l_loc,
                                                              typename Field::value_type& alpha_l_loc,
                                                              std::size_t& to_be_relaxed_loc,
                                                              std::size_t& Newton_iterations_loc,
                                                              bool& local_relaxation_applied,
                                                              const Gradient& grad_alpha_l_loc,
                                                              const bool mass_transfer_NR) {
  to_be_relaxed_loc = 0;

  if(!std::isnan(H_loc)) {
    /*--- Update auxiliary values affected by the nonlinear function for which we seek a zero ---*/
    const auto alpha_d_loc = alpha_l_loc*
                             local_conserved_variables(Md_INDEX)/local_conserved_variables(Ml_INDEX);
                             /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto alpha_g_loc = static_cast<typename Field::value_type>(1.0) - alpha_l_loc - alpha_d_loc;

    const auto rho_liq_loc = (local_conserved_variables(Ml_INDEX) + local_conserved_variables(Md_INDEX))/
                             (alpha_l_loc + alpha_d_loc); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto p_liq_loc   = EOS_phase_liq.pres_value(rho_liq_loc);
    const auto rho_g_loc   = local_conserved_variables(Mg_INDEX)/alpha_g_loc; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto p_g_loc     = EOS_phase_gas.pres_value(rho_g_loc);

    /*--- Prepare for mass transfer if desired ---*/
    const auto rho_loc = local_conserved_variables(Ml_INDEX)
                       + local_conserved_variables(Mg_INDEX)
                       + local_conserved_variables(Md_INDEX);

    // Compute first ordrer integral reminder "specific enthalpy"
    auto H_lim        = std::min(H_loc, Hmax);
    const auto fac_Ru = sigma*H_lim*(static_cast<typename Field::value_type>(3.0)/kappa - static_cast<typename Field::value_type>(1.0));
    if(mass_transfer_NR) {
      if(fac_Ru > static_cast<typename Field::value_type>(0.0) &&
         alpha_l_loc > alpha_l_min && alpha_l_loc < alpha_l_max &&
         -grad_alpha_l_loc[0]*local_conserved_variables(RHO_U_INDEX)
         -grad_alpha_l_loc[1]*local_conserved_variables(RHO_U_INDEX + 1) > static_cast<typename Field::value_type>(0.0) &&
         alpha_d_loc < alpha_d_max) {
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
    const auto delta_p = p_liq_loc - p_g_loc;
    const auto F_LS    = local_conserved_variables(Ml_INDEX)*(delta_p - sigma*H_lim);
    const auto aux_SS  = static_cast<typename Field::value_type>(2.0/3.0)*sigma*
                         local_conserved_variables(RHO_Z_INDEX)*
                         std::pow(local_conserved_variables(Ml_INDEX), static_cast<typename Field::value_type>(1.0/3.0));
    const auto F_SS    = local_conserved_variables(Md_INDEX)*delta_p
                       - std::pow(alpha_l_loc, static_cast<typename Field::value_type>(-1.0/3.0))*aux_SS;
    const auto F       = F_LS + F_SS;

    // Perform the relaxation only where really needed
    if(std::abs(F) > atol_Newton + rtol_Newton*std::min(EOS_phase_liq.get_p0(), sigma*std::abs(H_lim)) &&
       std::abs(dalpha_l_loc) > atol_Newton) {
      to_be_relaxed_loc = 1;
      Newton_iterations_loc++;
      local_relaxation_applied = true;

      // Compute the derivative w.r.t large scale volume fraction recalling that for a barotropic EOS dp/drho = c^2
      const auto ddelta_p_dalpha_l = -local_conserved_variables(Ml_INDEX)/(alpha_l_loc*alpha_l_loc)*
                                     EOS_phase_liq.c_value(rho_liq_loc)*EOS_phase_liq.c_value(rho_liq_loc)
                                     -local_conserved_variables(Mg_INDEX)/(alpha_g_loc*alpha_g_loc)*
                                     EOS_phase_gas.c_value(rho_g_loc)*EOS_phase_gas.c_value(rho_g_loc)*
                                     (local_conserved_variables(Ml_INDEX) + local_conserved_variables(Md_INDEX))/
                                     local_conserved_variables(Ml_INDEX);
      const auto dF_LS_dalpha_l    = local_conserved_variables(Ml_INDEX)*ddelta_p_dalpha_l;
      const auto dF_SS_dalpha_l    = local_conserved_variables(Md_INDEX)*ddelta_p_dalpha_l
                                   + static_cast<typename Field::value_type>(1.0/3.0)*
                                     std::pow(alpha_l_loc, static_cast<typename Field::value_type>(-4.0/3.0))*aux_SS;
      const auto dF_dalpha_l       = dF_LS_dalpha_l + dF_SS_dalpha_l;

      // Compute the pseudo time step starting as initial guess from the ideal unmodified Newton method
      auto dtau_ov_epsilon = std::numeric_limits<typename Field::value_type>::infinity();

      // Bound preserving condition for m_l, velocity and small-scale volume fraction
      if(dH > static_cast<typename Field::value_type>(0.0)) {
        // Bound preserving condition for m_l
        dtau_ov_epsilon = lambda/(sigma*dH);
        if(dtau_ov_epsilon < static_cast<typename Field::value_type>(0.0)) {
          throw std::runtime_error("Negative time step found after relaxation of mass of large-scale phase 1");
        }

        // Bound preserving for the velocity
        const auto mom_dot_vel   = (local_conserved_variables(RHO_U_INDEX)*local_conserved_variables(RHO_U_INDEX) +
                                    local_conserved_variables(RHO_U_INDEX + 1)*local_conserved_variables(RHO_U_INDEX + 1))/rho_loc;
        auto dtau_ov_epsilon_tmp = lambda*mom_dot_vel/(alpha_l_loc*sigma*dH*fac_Ru);
        dtau_ov_epsilon          = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
        if(dtau_ov_epsilon < static_cast<typename Field::value_type>(0.0)) {
          throw std::runtime_error("Negative time step found after relaxation of velocity");
        }

        /*--- No specific condition to impose for the positivty of alpha_d since alpha_d = alpha_l*m_l/m_d and
              m_d is increasing, m_l has already been imposed positive and alpha_l is going to be set with proper bounds later.
              On the other hand, there is no a priori superior limit, apart from the alpha_d_max which deactivates the mass transfer.
              Hence, in the first iteration, one can potentially reach alpha_d > alpha_d_max (likely unphyisical...) ---*/
      }

      // Bound preserving condition for large-scale volume fraction
      const auto dF_drhoz     = static_cast<typename Field::value_type>(-2.0/3.0)*
                                std::pow(rho_liq_loc, static_cast<typename Field::value_type>(1.0/3.0));

      const auto ddelta_p_dmd = -local_conserved_variables(Mg_INDEX)/(alpha_g_loc*alpha_g_loc)*
                                EOS_phase_gas.c_value(rho_g_loc)*EOS_phase_gas.c_value(rho_g_loc)/rho_liq_loc;
      const auto dF_LS_dmd    = local_conserved_variables(Ml_INDEX)*ddelta_p_dmd;
      const auto dF_SS_dmd    = delta_p + local_conserved_variables(Md_INDEX)*ddelta_p_dmd;
      const auto dF_dmd       = dF_LS_dmd + dF_SS_dmd;

      const auto ddelta_p_dml = EOS_phase_liq.c_value(rho_liq_loc)*EOS_phase_liq.c_value(rho_liq_loc)/alpha_l_loc
                              + local_conserved_variables(Mg_INDEX)/(alpha_g_loc*alpha_g_loc)*
                                EOS_phase_gas.c_value(rho_g_loc)*EOS_phase_gas.c_value(rho_g_loc)*
                                (alpha_l_loc*local_conserved_variables(Md_INDEX))/
                                (local_conserved_variables(Ml_INDEX)*local_conserved_variables(Ml_INDEX));
      const auto dF_LS_dml    = (delta_p - sigma*H_lim) + local_conserved_variables(Ml_INDEX)*ddelta_p_dml;
      const auto dF_SS_dml    = local_conserved_variables(Md_INDEX)*ddelta_p_dml
                              - static_cast<typename Field::value_type>(1.0/3.0)*aux_SS/
                                (std::pow(alpha_l_loc, static_cast<typename Field::value_type>(1.0/3.0))*local_conserved_variables(Ml_INDEX));
      const auto dF_dml       = dF_LS_dml + dF_SS_dml;

      const auto R            = dF_dml - dF_dmd - dF_drhoz*(static_cast<typename Field::value_type>(3.0)*Hmax/
                                                            (kappa*std::pow(rho_liq_loc, static_cast<typename Field::value_type>(1.0/3.0))));
                                /*equivalent to dF_drhoz*(S_avg/m_avg)*((rho*z/Sigma))*/

      // Upper bound
      const auto R_ml          = -local_conserved_variables(Ml_INDEX)*sigma*dH;
      const auto a             = (R_ml/rho_liq_loc)*R;
      auto b                   = (F + lambda*(static_cast<typename Field::value_type>(1.0) - alpha_l_loc)*dF_dalpha_l)/rho_liq_loc;
      auto D                   = b*b - static_cast<typename Field::value_type>(4.0)*a*(-lambda*(static_cast<typename Field::value_type>(1.0) - alpha_l_loc));
      auto dtau_ov_epsilon_tmp = std::numeric_limits<typename Field::value_type>::infinity();
      if(D > static_cast<typename Field::value_type>(0.0) &&
        (a > static_cast<typename Field::value_type>(0.0) ||
         (a < static_cast<typename Field::value_type>(0.0) && b > static_cast<typename Field::value_type>(0.0)))) {
        dtau_ov_epsilon_tmp = static_cast<typename Field::value_type>(0.5)*(-b + std::sqrt(D))/a;
      }
      if(a == static_cast<typename Field::value_type>(0.0) &&
         b > static_cast<typename Field::value_type>(0.0)) {
        dtau_ov_epsilon_tmp = lambda*(static_cast<typename Field::value_type>(1.0) - alpha_l_loc)/b;
      }
      dtau_ov_epsilon = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
      // Lower bound
      dtau_ov_epsilon_tmp = std::numeric_limits<typename Field::value_type>::infinity();
      b                   = (F - lambda*alpha_l_loc*dF_dalpha_l)/rho_liq_loc;
      D                   = b*b - static_cast<typename Field::value_type>(4.0)*a*(lambda*alpha_l_loc);
      if(D > static_cast<typename Field::value_type>(0.0) &&
         (a < static_cast<typename Field::value_type>(0.0) ||
          (a > static_cast<typename Field::value_type>(0.0) && b < static_cast<typename Field::value_type>(0.0)))) {
        dtau_ov_epsilon_tmp = static_cast<typename Field::value_type>(0.5)*(-b - std::sqrt(D))/a;
      }
      if(a == static_cast<typename Field::value_type>(0.0) &&
         b < static_cast<typename Field::value_type>(0.0)) {
        dtau_ov_epsilon_tmp = -lambda*alpha_l_loc/b;
      }
      dtau_ov_epsilon = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
      if(dtau_ov_epsilon < static_cast<typename Field::value_type>(0.0)) {
        throw std::runtime_error("Negative time step found after relaxation of large-scale volume fraction");
      }

      // Compute the effective variation of the variables
      if(std::isinf(dtau_ov_epsilon)) {
        // If we are in this branch we do not have mass transfer
        // and we do not have other restrictions on the bounds of large scale volume fraction
        dalpha_l_loc = -F/dF_dalpha_l;

        if(dalpha_l_loc > static_cast<typename Field::value_type>(0.0)) {
          dalpha_l_loc = std::min(-F/dF_dalpha_l, lambda*(static_cast<typename Field::value_type>(1.0) - alpha_l_loc));
        }
        else if(dalpha_l_loc < static_cast<typename Field::value_type>(0.0)) {
          dalpha_l_loc = std::max(-F/dF_dalpha_l, -lambda*alpha_l_loc);
        }
      }
      else {
        const auto dm_l = dtau_ov_epsilon*R_ml;

        dalpha_l_loc = (dtau_ov_epsilon/rho_liq_loc)/((static_cast<typename Field::value_type>(1.0) - dtau_ov_epsilon/rho_liq_loc*dF_dalpha_l))*
                       (F + dm_l*R);

        if(dm_l > static_cast<typename Field::value_type>(0.0)) {
          throw std::runtime_error("Negative sign of mass transfer inside Newton step");
        }
        else {
          local_conserved_variables(Ml_INDEX) += dm_l;
          if(local_conserved_variables(Ml_INDEX) < static_cast<typename Field::value_type>(0.0)) {
            throw std::runtime_error("Negative mass of large-scale phase 1 inside Newton step");
          }

          local_conserved_variables(Md_INDEX) -= dm_l;
          if(local_conserved_variables(Md_INDEX) < static_cast<typename Field::value_type>(0.0)) {
            throw std::runtime_error("Negative mass of small-scale phase 1 inside Newton step");
          }
        }

        const auto R_Sigma_D = -dm_l*static_cast<typename Field::value_type>(3.0)*Hmax/(kappa*rho_liq_loc);
        local_conserved_variables(RHO_Z_INDEX) += (std::pow(rho_liq_loc, static_cast<typename Field::value_type>(2.0/3.0)))*R_Sigma_D;
      }

      if(alpha_l_loc + dalpha_l_loc < static_cast<typename Field::value_type>(0.0) ||
         alpha_l_loc + dalpha_l_loc > static_cast<typename Field::value_type>(1.0)) {
        throw std::runtime_error("Bounds exceeding value for large-scale volume fraction inside Newton step");
      }
      else {
        alpha_l_loc += dalpha_l_loc;
      }

      if(dH > static_cast<typename Field::value_type>(0.0)) {
        typename Field::value_type drho_fac_Ru = static_cast<typename Field::value_type>(0.0);
        const auto mom_squared = local_conserved_variables(RHO_U_INDEX)*local_conserved_variables(RHO_U_INDEX)
                               + local_conserved_variables(RHO_U_INDEX + 1)*local_conserved_variables(RHO_U_INDEX + 1);
        if(mom_squared > static_cast<typename Field::value_type>(0.0)) {
          drho_fac_Ru = dtau_ov_epsilon*
                        sigma*dH*fac_Ru*rho_loc/mom_squared; /*--- u/u^{2} = rho*u/(rho*(u^{2})) = (rho/(rho*u)^{2})*(rho*u) ---*/
        }

        for(std::size_t d = 0; d < Field::dim; ++d) {
          local_conserved_variables(RHO_U_INDEX + d) -= drho_fac_Ru*local_conserved_variables(RHO_U_INDEX + d);
        }
      }
    }

    // Update "conservative counter part" of large-scale volume fraction.
    // Do it outside because this can change either because of relaxation or
    // alpha_l.
    local_conserved_variables(RHO_ALPHA_l_INDEX) = rho_loc*alpha_l_loc;
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
                            }
                        );

  samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, fields..., level_);
}

// Execute postprocessing
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::execute_postprocess(const typename Field::value_type time) {
  /*--- Initialize relevant integral quantities ---*/
  auto local_H_lig                = static_cast<typename Field::value_type>(0.0);
  auto local_m_l_int              = static_cast<typename Field::value_type>(0.0);
  auto local_m_d_int              = static_cast<typename Field::value_type>(0.0);
  auto local_alpha_l_int          = static_cast<typename Field::value_type>(0.0);
  auto local_grad_alpha_l_int     = static_cast<typename Field::value_type>(0.0);
  auto local_Sigma_d_int          = static_cast<typename Field::value_type>(0.0);
  auto local_alpha_d_int          = static_cast<typename Field::value_type>(0.0);
  auto local_grad_alpha_d_int     = static_cast<typename Field::value_type>(0.0);
  auto local_grad_alpha_l_tot_int = static_cast<typename Field::value_type>(0.0);
  auto local_alpha_l_bar_int      = static_cast<typename Field::value_type>(0.0);
  auto local_grad_alpha_l_bar_int = static_cast<typename Field::value_type>(0.0);

  alpha_l_bar.resize();
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Save alpha_l_bar
                              alpha_l_bar[cell] = alpha_l[cell]/(static_cast<typename Field::value_type>(1.0) - alpha_d[cell]);
                            }
                        );
  samurai::update_ghost_mr(alpha_l_bar);
  grad_alpha_l_bar.resize();
  grad_alpha_l_bar = gradient(alpha_l_bar);
  samurai::update_ghost_mr(alpha_d);
  grad_alpha_d.resize();
  grad_alpha_d = gradient(alpha_d);

  p_liq.resize();
  p_g.resize();
  p.resize();
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Compue H_lig
                              const auto rho_liq = (conserved_variables[cell][Ml_INDEX] + conserved_variables[cell][Md_INDEX])/
                                                   (alpha_l[cell] + alpha_d[cell]); /*--- TODO: Add a check in case of zero volume fraction ---*/
                              p_liq[cell]        = EOS_phase_liq.pres_value(rho_liq);
                              const auto alpha_g = static_cast<typename Field::value_type>(1.0) - alpha_l[cell] - alpha_d[cell];
                              const auto rho_g   = conserved_variables[cell][Mg_INDEX]/alpha_g; /*--- TODO: Add a check in case of zero volume fraction ---*/
                              p_g[cell]          = EOS_phase_gas.pres_value(rho_g);
                              p[cell]            = (alpha_l[cell] + alpha_d[cell])*p_liq[cell]
                                                 + alpha_g*p_g[cell] - static_cast<typename Field::value_type>(2.0/3.0)*sigma*Sigma_d[cell];
                              const auto H_lim   = std::min(H[cell][0], Hmax);
                              const auto fac_Ru  = sigma*H_lim*
                                                   (static_cast<typename Field::value_type>(3.0)/kappa - static_cast<typename Field::value_type>(1.0));
                              if(fac_Ru > static_cast<typename Field::value_type>(0.0) &&
                                 alpha_l[cell] > alpha_l_min && alpha_l[cell] < alpha_l_max &&
                                 -grad_alpha_l[cell][0]*conserved_variables[cell][RHO_U_INDEX]
                                 -grad_alpha_l[cell][1]*conserved_variables[cell][RHO_U_INDEX + 1] > static_cast<typename Field::value_type>(0.0) &&
                                 alpha_d[cell] < alpha_d_max) {
                                local_H_lig = std::max(H[cell][0], local_H_lig);
                              }

                              // Compute the integral quantities
                              const auto cell_volume = std::pow(static_cast<typename Field::value_type>(cell.length),
                                                                static_cast<typename Field::value_type>(dim));
                              local_m_l_int += conserved_variables[cell][Ml_INDEX]*cell_volume;
                              local_m_d_int += conserved_variables[cell][Md_INDEX]*cell_volume;
                              local_alpha_l_int += alpha_l[cell]*cell_volume;
                              local_grad_alpha_l_int += std::sqrt(xt::sum(grad_alpha_l[cell]*grad_alpha_l[cell])())*cell_volume;
                              local_Sigma_d_int += Sigma_d[cell]*cell_volume;
                              local_grad_alpha_d_int += std::sqrt(xt::sum(grad_alpha_d[cell]*grad_alpha_d[cell])())*cell_volume;
                              local_alpha_d_int += alpha_d[cell]*cell_volume;
                              local_grad_alpha_l_tot_int += std::sqrt(xt::sum((grad_alpha_l[cell] + grad_alpha_d[cell])*
                                                                              (grad_alpha_l[cell] + grad_alpha_d[cell]))())*cell_volume;
                              local_alpha_l_bar_int += alpha_l_bar[cell]*cell_volume;
                              local_grad_alpha_l_bar_int += std::sqrt(xt::sum(grad_alpha_l_bar[cell]*grad_alpha_l_bar[cell])())*cell_volume;
                            }
                        );

  /*--- Perform MPI collective operations ---*/
  typename Field::value_type global_H_lig;
  MPI_Allreduce(&local_H_lig, &global_H_lig, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  typename Field::value_type global_m_l_int;
  MPI_Allreduce(&local_m_l_int, &global_m_l_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  typename Field::value_type global_m_d_int;
  MPI_Allreduce(&local_m_d_int, &global_m_d_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  typename Field::value_type global_alpha_l_int;
  MPI_Allreduce(&local_alpha_l_int, &global_alpha_l_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  typename Field::value_type global_grad_alpha_l_int;
  MPI_Allreduce(&local_grad_alpha_l_int, &global_grad_alpha_l_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  typename Field::value_type global_Sigma_d_int;
  MPI_Allreduce(&local_Sigma_d_int, &global_Sigma_d_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  typename Field::value_type global_grad_alpha_d_int;
  MPI_Allreduce(&local_grad_alpha_d_int, &global_grad_alpha_d_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  typename Field::value_type global_alpha_d_int;
  MPI_Allreduce(&local_alpha_d_int, &global_alpha_d_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  typename Field::value_type global_grad_alpha_l_tot_int;
  MPI_Allreduce(&local_grad_alpha_l_tot_int, &global_grad_alpha_l_tot_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  typename Field::value_type global_alpha_l_bar_int;
  MPI_Allreduce(&local_alpha_l_bar_int, &global_alpha_l_bar_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  typename Field::value_type global_grad_alpha_l_bar_int;
  MPI_Allreduce(&local_grad_alpha_l_bar_int, &global_grad_alpha_l_bar_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  /*--- Save the data ---*/
  Hlig                      << std::fixed << std::setprecision(12) << time << '\t' << global_H_lig                << std::endl;
  m_l_integral              << std::fixed << std::setprecision(12) << time << '\t' << global_m_l_int              << std::endl;
  m_d_integral              << std::fixed << std::setprecision(12) << time << '\t' << global_m_d_int              << std::endl;
  alpha_l_integral          << std::fixed << std::setprecision(12) << time << '\t' << global_alpha_l_int          << std::endl;
  grad_alpha_l_integral     << std::fixed << std::setprecision(12) << time << '\t' << global_grad_alpha_l_int     << std::endl;
  Sigma_d_integral          << std::fixed << std::setprecision(12) << time << '\t' << global_Sigma_d_int          << std::endl;
  alpha_d_integral          << std::fixed << std::setprecision(12) << time << '\t' << global_alpha_d_int          << std::endl;
  grad_alpha_d_integral     << std::fixed << std::setprecision(12) << time << '\t' << global_grad_alpha_d_int     << std::endl;
  grad_alpha_l_tot_integral << std::fixed << std::setprecision(12) << time << '\t' << global_grad_alpha_l_tot_int << std::endl;
  alpha_l_bar_integral      << std::fixed << std::setprecision(12) << time << '\t' << global_alpha_l_bar_int      << std::endl;
  grad_alpha_l_bar_integral << std::fixed << std::setprecision(12) << time << '\t' << global_grad_alpha_l_bar_int << std::endl;
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

  const auto dt_save = Tf/static_cast<typename Field::value_type>(nfiles);

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
  auto numerical_flux_st = SurfaceTension_flux.make_two_scale_capillarity(grad_alpha_l);

  /*--- Save the initial condition ---*/
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(path, suffix_init, conserved_variables, alpha_l, grad_alpha_l, normal, H, p_liq, p_g, p,
                          alpha_d, Sigma_d, grad_alpha_d, vel, div_vel,
                          alpha_l_bar, grad_alpha_l_bar, H_bar,
                          Mach);
  Hlig.open("Hlig.dat", std::ofstream::out);
  m_l_integral.open("m_l_integral.dat", std::ofstream::out);
  m_d_integral.open("m_d_integral.dat", std::ofstream::out);
  alpha_l_integral.open("alpha_l_integral.dat", std::ofstream::out);
  grad_alpha_l_integral.open("grad_alpha_l_integral.dat", std::ofstream::out);
  Sigma_d_integral.open("Sigma_d_integral.dat", std::ofstream::out);
  alpha_d_integral.open("alpha_d_integral.dat", std::ofstream::out);
  grad_alpha_d_integral.open("grad_alpha_d_integral.dat", std::ofstream::out);
  grad_alpha_l_tot_integral.open("grad_alpha_l_tot_integral.dat", std::ofstream::out);
  alpha_l_bar_integral.open("alpha_l_bar_integral.dat", std::ofstream::out);
  grad_alpha_l_bar_integral.open("grad_alpha_l_bar_integral.dat", std::ofstream::out);
  auto t = static_cast<typename Field::value_type>(0.0);
  execute_postprocess(t);

  /*--- Set initial time step ---*/
  const auto dx = static_cast<typename Field::value_type>(mesh.cell_length(mesh.max_level()));
  auto dt       = std::min(Tf - t, cfl*dx/get_max_lambda());
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
    #ifdef RELAX_RECONSTRUCTION
      normal.resize();
      H.resize();
      grad_alpha_l.resize();
      update_geometry();
      samurai::update_ghost_mr(H);
    #endif
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
      save(fs::current_path(), "_diverged", conserved_variables, alpha_l);
      exit(1);
    }

    // Update the geometry to recompute volume fraction gradient
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                              {
                                alpha_l[cell] = conserved_variables[cell][RHO_ALPHA_l_INDEX]/
                                                (conserved_variables[cell][Ml_INDEX] +
                                                 conserved_variables[cell][Mg_INDEX] +
                                                 conserved_variables[cell][Md_INDEX]);
                              }
                          );
    #ifdef VERBOSE
      check_data();
    #endif
    #ifndef RELAX_RECONSTRUCTION
      normal.resize();
      H.resize();
      grad_alpha_l.resize();
    #endif
    update_geometry();

    // Capillarity contribution
    samurai::update_ghost_mr(conserved_variables);
    samurai::update_ghost_mr(grad_alpha_l);
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
      // Apply relaxation if desired, which will modify alpha_l and, consequently, for what
      // concerns next time step, rho_alpha_l (as well as grad_alpha_l).
      dalpha_l.resize();
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
      #ifdef RELAX_RECONSTRUCTION
        samurai::update_ghost_mr(H);
      #endif
      try {
        auto flux_hyp = numerical_flux_hyp(conserved_variables);
        conserved_variables_tmp = conserved_variables - dt*flux_hyp;
        std::swap(conserved_variables.array(), conserved_variables_tmp.array());
      }
      catch(std::exception& e) {
        std::cerr << e.what() << std::endl;
        save(fs::current_path(), "_diverged", conserved_variables, alpha_l);
        exit(1);
      }

      // Recompute geometrical quantities
      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                                {
                                  alpha_l[cell] = conserved_variables[cell][RHO_ALPHA_l_INDEX]/
                                                  (conserved_variables[cell][Ml_INDEX] +
                                                   conserved_variables[cell][Mg_INDEX] +
                                                   conserved_variables[cell][Md_INDEX]);
                                }
                            );
      #ifdef VERBOSE
        check_data();
      #endif
      update_geometry();

      // Capillarity contribution
      samurai::update_ghost_mr(conserved_variables);
      samurai::update_ghost_mr(grad_alpha_l);
      flux_st = numerical_flux_st(conserved_variables);
      conserved_variables_tmp = conserved_variables - dt*flux_st;
      std::swap(conserved_variables.array(), conserved_variables_tmp.array());

      // Apply relaxation
      if(apply_relax) {
        // Apply relaxation if desired, which will modify alpha_l and, consequently, for what
        // concerns next time step, rho_alpha_l (as well as grad_alpha_l).
        apply_relaxation();
      }

      // Complete evaluation
      conserved_variables_np1.resize();
      conserved_variables_np1 = static_cast<typename Field::value_type>(0.5)*(conserved_variables_old + conserved_variables);
      std::swap(conserved_variables.array(), conserved_variables_np1.array());

      // Recompute volume fraction gradient and curvature for the next time step
      #ifdef RELAX_RECONSTRUCTION
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                                  {
                                    alpha_l[cell] = conserved_variables[cell][RHO_ALPHA_l_INDEX]/
                                                    (conserved_variables[cell][Ml_INDEX] +
                                                     conserved_variables[cell][Mg_INDEX] +
                                                     conserved_variables[cell][Md_INDEX]);
                                  }
                              );
        update_geometry();
      #endif
    #endif

    /*--- Compute updated time step ---*/
    #ifndef RELAX_RECONSTRUCTION
      update_geometry();
    #endif
    dt = std::min(Tf - t, cfl*dx/get_max_lambda());

    /*--- Postprocess data ---*/
    execute_postprocess(t);

    /*--- Save the results ---*/
    if(t >= static_cast<typename Field::value_type>(nsave + 1)*dt_save || t == Tf) {
      // Resize all the fields not resized yet
      div_vel.resize();
      samurai::update_ghost_mr(vel);
      div_vel = divergence(vel);

      normal_bar.resize();
      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                                {
                                  const auto mod_grad_alpha_l_bar = std::sqrt(xt::sum(grad_alpha_l_bar[cell]*grad_alpha_l_bar[cell])());

                                  if(mod_grad_alpha_l_bar > mod_grad_alpha_l_min) {
                                    normal_bar[cell] = grad_alpha_l_bar[cell]/mod_grad_alpha_l_bar;
                                  }
                                  else {
                                    for(std::size_t d = 0; d < dim; ++d) {
                                      normal_bar[cell][d] = static_cast<typename Field::value_type>(nan(""));
                                    }
                                  }
                                }
                            );

      samurai::update_ghost_mr(normal_bar);
      H_bar.resize();
      H_bar = -divergence(normal_bar);

      // Perform the saving
      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";
      save(path, suffix, conserved_variables, alpha_l, grad_alpha_l, normal, H, p_liq, p_g, p,
                         alpha_d, Sigma_d, grad_alpha_d, vel, div_vel,
                         alpha_l_bar, grad_alpha_l_bar, H_bar, Newton_iterations,
                         Mach);
    }
  }

  /*--- Close the files for post-proessing ---*/
  Hlig.close();
  m_l_integral.close();
  m_d_integral.close();
  alpha_l_integral.close();
  grad_alpha_l_integral.close();
  Sigma_d_integral.close();
  alpha_d_integral.close();
  grad_alpha_d_integral.close();
  grad_alpha_l_tot_integral.close();
  alpha_l_bar_integral.close();
  grad_alpha_l_bar_integral.close();
}
