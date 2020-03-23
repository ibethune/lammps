/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef INTEGRATE_CLASS

IntegrateStyle(mist,Mist)

#else

#ifndef LMP_MIST_RUN_H
#define LMP_MIST_RUN_H

#include "integrate.h"
#include "mist.h"

namespace LAMMPS_NS {

typedef struct
{
  Domain *d;
  Force *f;
} cell_data_t;

class Mist : public Integrate {
 public:
  Mist(class LAMMPS *, int, char **);
  virtual ~Mist() {}
  virtual void init();
  virtual void setup(int flag);
  virtual void setup_minimal(int);
  virtual void run(int);
  void cleanup();

 protected:

  int extraflag;

  virtual void force_clear();

  // Everything above here by analogy to verlet.h

  bool energy_required, pressure_required;

  // Helper function to check for errors from MIST
  void MIST_chkerr(int misterr, const char* file,int line);

  // Setup function to avoid code duplication
  void mist_setup();

  // Callback that MIST uses to update forces
  static void lammps_force_wrapper(void *data);

  // Do the actual force update
  void update_forces();

  class Compute *pe_compute, *press_compute;

  double *masses;

  cell_data_t cell_data;

};

}
#endif
#endif

/* ERROR/WARNING messages:

W: Fixes have been defined, but MIST will ignore them!

The MIST library is solely responsible for updating the system state.  Any
fixes thar are defined are ignored

*/
