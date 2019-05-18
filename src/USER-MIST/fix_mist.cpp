/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_mist.h"
#include "atom.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

FixMIST::FixMIST(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 3) error->all(FLERR,"Illegal fix mist command");

  // register with Atom class
  atom->add_callback(0);
}

/* ---------------------------------------------------------------------- */

FixMIST::~FixMIST()
{
  // unregister callbacks to this fix from Atom class
  atom->delete_callback(id,0);
}

/* ---------------------------------------------------------------------- */

int FixMIST::setmask()
{
  int mask = 0;
  return mask;
}

/* ----------------------------------------------------------------------
   copy values within local atom-based array
------------------------------------------------------------------------- */

void FixMIST::copy_arrays(int i, int j, int /*delflag*/)
{
  atom->f[j][0] = atom->f[i][0];
  atom->f[j][1] = atom->f[i][1];
  atom->f[j][2] = atom->f[i][2];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixMIST::pack_exchange(int i, double *buf)
{
  int n = 0;
  buf[n++] = atom->f[i][0];
  buf[n++] = atom->f[i][1];
  buf[n++] = atom->f[i][2];
  return n;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixMIST::unpack_exchange(int nlocal, double *buf)
{
  int n = 0;
  atom->f[nlocal][0] = buf[n++];
  atom->f[nlocal][1] = buf[n++];
  atom->f[nlocal][2] = buf[n++];
  return n;
}
