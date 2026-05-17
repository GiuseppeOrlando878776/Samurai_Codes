// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2026
//
#pragma once

/**
 * Generic interface for a barotropic EOS
 */
template<typename T = double>
class BarotropicEOS {
public:
  /**
   * Default constructor
   */
  BarotropicEOS() = default;

  /**
   * Default copy-constructor
   */
  BarotropicEOS(const BarotropicEOS&) = default;

  /**
   * Default destructor
   */
  virtual ~BarotropicEOS() = default;

  /**
   * Function to compute the pressure from the density
   * @param rho density value
   */
  virtual T pres_value(const T& rho) const = 0; /*--- Function to compute the pressure from the density ---*/

  /**
   * Function to compute the speed of sound from the density
   * @param rho density value
   */
  virtual T c_value(const T& rho) const = 0;

  /**
   * Function to compute the density from the pressure
   * @param pres pressure value
   */
  virtual T rho_value(const T& pres) const = 0;

  /**
   * Function to compute the internal energy from the density
   * @param rho density value
   */
  virtual T e_value(const T& rho) const = 0;
};


/**
 * Implementation of a linearized barotropic EOS
 */
template<typename T = double>
class LinearizedBarotropicEOS: public BarotropicEOS<T> {
public:
  /**
   * Default constructor
   */
  LinearizedBarotropicEOS() = default;

  /**
   * Default copy-constructor
   */
  LinearizedBarotropicEOS(const LinearizedBarotropicEOS&) = default;

  /**
   * Class constructor for the linearized barotropic EOS
   * @param p0 reference pressure
   * @param rho0 reference density
   * @param c0 speed of sound
   */
  LinearizedBarotropicEOS(const T p0_, const T rho0_, const T c0_);

  /**
   * Function to compute the pressure from the density
   * @param rho density value
   */
  virtual T pres_value(const T& rho) const override;

  /**
   * Function to compute the speed of sound from the density
   * @param rho density value
   */
  virtual T c_value(const T& rho) const override;

  /**
   * Function to compute the density from the pressure
   * @param pres pressure value
   */
  virtual T rho_value(const T& pres) const override;

  /**
   * Function to compute the internal energy from the density
   * @param rho density value
   */
  virtual T e_value(const T& rho) const override;

  /**
   * Get the speed of sound
   */
  inline T get_c0() const;

  /**
   * Get the reference pressure
   */
  inline T get_p0() const;

  /**
   * Get the reference density
   */
  inline T get_rho0() const;

  /**
   * Set the speed of sound
   * @param c0_ speed of sound to be set
   */
  inline void set_c0(const T c0_);

  /**
   * Set the reference pressure
   * @param p0_ reference pressure to be set
   */
  inline void set_p0(const T p0_);

  /**
   * Set the reference density
   * @param rho0_ reference density to be set
   */
  inline void set_rho0(const T rho0_);

private:
  T p0;   /*!< Reference pressure */
  T rho0; /*!< Reference density */
  T c0;   /*!< Speed of sound */
};

// Implement the constructor
//
template<typename T>
LinearizedBarotropicEOS<T>::LinearizedBarotropicEOS(const T p0_, const T rho0_, const T c0_):
  BarotropicEOS<T>(), p0(p0_), rho0(rho0_), c0(c0_) {}

// Implement the pressure value from the density
//
template<typename T>
T LinearizedBarotropicEOS<T>::pres_value(const T& rho) const {
  if(std::isnan(rho)) {
    return static_cast<T>(nan(""));
  }

  return p0 + c0*c0*(rho - rho0);
}

// Implement the speed of sound from the density
//
template<typename T>
T LinearizedBarotropicEOS<T>::c_value(const T& rho) const {
  (void) rho;

  return c0;
}

// Implement the density from the pressure
//
template<typename T>
T LinearizedBarotropicEOS<T>::rho_value(const T& pres) const {
  if(std::isnan(pres)) {
    return static_cast<T>(nan(""));
  }

  return (pres - p0)/(c0*c0) + rho0;
}

// Implement the internal energy from the density
//
template<typename T>
T LinearizedBarotropicEOS<T>::e_value(const T& rho) const {
  if(std::isnan(rho)) {
    return static_cast<T>(nan(""));
  }

  if(rho > static_cast<T>(1e-10)) {
    return c0*c0*std::log(rho) - p0/rho;
  }
  else {
    throw std::runtime_error("Zero density when computing internal energy");
  }
}

// Implement the getter of the speed of sound
//
template<typename T>
inline T LinearizedBarotropicEOS<T>::get_c0() const {
  return c0;
}

// Implement the getter of the reference pressure
//
template<typename T>
inline T LinearizedBarotropicEOS<T>::get_p0() const {
  return p0;
}

// Implement the getter of the reference density
//
template<typename T>
inline T LinearizedBarotropicEOS<T>::get_rho0() const {
  return rho0;
}

// Implement the setter for the speed of sound
//
template<typename T>
inline void LinearizedBarotropicEOS<T>::set_c0(const T c0_) {
  c0 = c0_;
}

// Implement the setter of the reference pressure
//
template<typename T>
inline void LinearizedBarotropicEOS<T>::set_p0(const T p0_) {
  p0 = p0_;
}

// Implement the setter of the reference density
//
template<typename T>
inline void LinearizedBarotropicEOS<T>::set_rho0(const T rho0_) {
  rho0 = rho0_;
}
