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

IntegrateStyle(mist_run,Mist_run)

#else

#ifndef LMP_MIST_RUN_H
#define LMP_MIST_RUN_H

#include "integrate.h"
#include "mist.h"

namespace LAMMPS_NS {

class Mist_run : public Integrate {
 public:
  Mist_run(class LAMMPS *, int, char **);
  virtual ~Mist_run() {}
  virtual void init();
  virtual void setup(int flag);
  virtual void setup_minimal(int);
  virtual void run(int);
  void cleanup();

 protected:

  int extraflag;

  virtual void force_clear();

  // Everything above here by analogy to verlet.h

  Pointers *pointerToData;

  void MIST_chkerr(int misterr, const char* file,int line);
  void mist_setup();
  static void step_force_wrapper(void *s);
  static void update_forces_step(Mist_run *lp);

  class Compute *pe;
  double *potEnergyPtr;

};


/*
// Mist_run pointer -> bad solution...
Mist_run* mist_force_wrapper;

void Mist_run_memberFunctionWrapper(void *s){

mist_force_wrapper->update_forces_step(Mist_run *s);

}
*/

}
#endif
#endif



/* ERROR/WARNING messages:

W: No fixes defined, atoms won't move

If you are not using a fix like nve, nvt, npt then atom velocities and
coordinates will not be updated during timestepping.

E: KOKKOS package requires run_style verlet/kk

The KOKKOS package requires the Kokkos version of run_style verlet; the
regular version cannot be used.

*/
