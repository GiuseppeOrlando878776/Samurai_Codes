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

#endif
