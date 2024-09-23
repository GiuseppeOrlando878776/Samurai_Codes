#ifndef containers_hpp
#define containers_hpp

// Declare a struct with the simulation parameters
// (domain, levels, final time, and Courant number)
struct Simulation_Paramaters {
  double xL;
  double xR;
  std::size_t min_level;
  std::size_t max_level;
  double Tf;
  double Courant;
  std::size_t nfiles;
};

// Declare a struct with EOS parameters
struct EOS_Parameters {
  double gamma_1;
  double pi_infty_1;
  double q_infty_1;
  double gamma_2;
  double pi_infty_2;
  double q_infty_2;
};

#endif
