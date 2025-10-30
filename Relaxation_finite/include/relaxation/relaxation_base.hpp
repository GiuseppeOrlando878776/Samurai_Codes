// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#pragma once

#include "../utilities.hpp"

namespace samurai {
  /**
    * Implementation of a source term (finite-rate or instantaneous relaxation)
    */
  template<class Field>
  class Source {
  public:
    /*--- Useful definitions ---*/
    using Indices = EquationData<Field::dim>;
    using Number  = typename Field::value_type; /*--- Define the shortcut for the arithmetic type ---*/

    using cfg = LocalCellSchemeConfig<SchemeType::NonLinear, Field, Field>;

    Source() = default; /*--- Default constructor (for minimal instantaneous relaxations this is sufficient) ---*/

    virtual ~Source() {} /*--- Virtual destructor (because pure virtual class) ---*/

    inline void set_source_name(const std::string& source_name_); /*--- Set the name of the source term ---*/

    inline std::string get_source_name() const; /*--- Get the name of the source term ---*/

    inline void set_dt(const Number dt_); /*--- Set the actual time step (needed for finite-rate relaxation) ---*/

    inline Number get_dt() const; /*--- Get the actual time step (needed for finite-rate relaxation) ---*/

    virtual decltype(make_cell_based_scheme<cfg>()) make_relaxation() = 0; /*--- Compute the relaxation ---*/

  private:
    std::string source_name; /*--- Name of the source term ---*/

    Number dt; /*--- Needed for finite rate relaxation ---*/
  };

  // Set the name of the source term
  //
  template<class Field>
  inline void Source<Field>::set_source_name(const std::string& source_name_) {
    source_name = source_name_;
  }

  // Get the name of the source term
  //
  template<class Field>
  inline std::string Source<Field>::get_source_name() const {
    return source_name;
  }

  // Set the name of the source term
  //
  template<class Field>
  inline void Source<Field>::set_dt(const Number dt_) {
    dt = dt_;
  }

  // Get the name of the source term
  //
  template<class Field>
  inline typename Source<Field>::Number
  Source<Field>::get_dt() const {
    return dt;
  }

} // end of namespace
