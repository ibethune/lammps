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

#include <string.h>
#include "mist_run.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "atom.h"
#include "atom_vec.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "output.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "fix.h"
#include "timer.h"
#include "memory.h"
#include "error.h"
#include "math_extra.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

Mist::Mist(LAMMPS *lmp, int narg, char **arg) :
  Integrate(lmp, narg, arg) {

  energy_required = false;
  pe_compute = NULL;
  masses = NULL;
}

/* ----------------------------------------------------------------------
   initialization before run
------------------------------------------------------------------------- */

void Mist::init()
{
  Integrate::init(); 

  // warn if any fixes - mist does everything

  if (modify->nfix != 0 && comm->me == 0)
    error->warning(FLERR,"Fixes have been defined, but MIST will ignore them!");

  // virial_style:
  // 1 if computed explicitly by pair->compute via sum over pair interactions
  // 2 if computed implicitly by pair->virial_fdotr_compute via sum over ghosts

  if (force->newton_pair) virial_style = 2;
  else  virial_style = 1;

  // setup lists of computes for global and per-atom PE and pressure

  ev_setup();

  // detect if fix omp is present for clearing force arrays

  int ifix = modify->find_fix("package_omp");
  if (ifix >= 0) external_force_clear = 1;

  // set flags for arrays to clear in force_clear()

  extraflag = 0;
  if (atom->avec->forceclearflag) extraflag = 1;

  if (domain->triclinic) error->all(FLERR, "MIST only supports orthorhombic cells");
  if (domain->xperiodic != domain->yperiodic || domain->xperiodic != domain->zperiodic)
    error->all(FLERR, "MIST supports xyz-periodic or non-periodic systems.  1D or 2D periodicity is not supported");

}

/* ----------------------------------------------------------------------
   setup before run
------------------------------------------------------------------------- */

void Mist::setup(int flag)
{
  if (comm->me == 0 && screen) {
    fprintf(screen,"Setting up MIST run ...\n");
    if (flag) {
      fprintf(screen,"  Unit style    : %s\n", update->unit_style);
      fprintf(screen,"  Current step  : " BIGINT_FORMAT "\n", update->ntimestep);
      fprintf(screen,"  Time step     : %g\n", update->dt);
      timer->print_timeout(screen);
    }
  }

  update->setupflag = 1;

  // setup domain, communication and neighboring
  // acquire ghosts
  // build neighbor lists

  atom->setup();
  modify->setup_pre_exchange();
  domain->pbc();
  domain->reset_box();
  comm->setup();
  if (neighbor->style) neighbor->setup_bins();
  comm->exchange();
  if (atom->sortfreq > 0) atom->sort();
  comm->borders();
  domain->image_check();
  domain->box_too_small_check();
  modify->setup_pre_neighbor();
  neighbor->build(1);
  modify->setup_post_neighbor();
  neighbor->ncalls = 0;

  // Initialise MIST

  mist_setup();

  // compute all forces

  force->setup();
  ev_set(update->ntimestep);
  if (energy_required) eflag |= 1;
  force_clear();
  modify->setup_pre_force(vflag);

  if (pair_compute_flag) force->pair->compute(eflag,vflag);
  else if (force->pair) force->pair->compute_dummy(eflag,vflag);

  if (atom->molecular) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
  }

  if (force->kspace) {
    force->kspace->setup();
    if (kspace_compute_flag) force->kspace->compute(eflag,vflag);
    else force->kspace->compute_dummy(eflag,vflag);
  }

  modify->pre_reverse(eflag,vflag);
  if (force->newton) comm->reverse_comm();

  modify->setup(vflag);
  output->setup(flag);
  update->setupflag = 0;

}

/* ----------------------------------------------------------------------
   setup without output
   flag = 0 = just force calculation
   flag = 1 = reneighbor and force calculation
------------------------------------------------------------------------- */

void Mist::setup_minimal(int flag)
{
  update->setupflag = 1;

  // setup domain, communication and neighboring
  // acquire ghosts
  // build neighbor lists

  if (flag) {
    modify->setup_pre_exchange();
    domain->pbc();
    domain->reset_box();
    comm->setup();
    if (neighbor->style) neighbor->setup_bins();
    comm->exchange();
    comm->borders();
    domain->image_check();
    domain->box_too_small_check();
    modify->setup_pre_neighbor();
    neighbor->build(1);
    modify->setup_post_neighbor();
    neighbor->ncalls = 0;
  }

  // Initialise MIST

  mist_setup();

  // compute all forces

  ev_set(update->ntimestep);
  if (energy_required) eflag |= 1;
  force_clear();
  modify->setup_pre_force(vflag);

  if (pair_compute_flag) force->pair->compute(eflag,vflag);
  else if (force->pair) force->pair->compute_dummy(eflag,vflag);

  if (atom->molecular) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
  }

  if (force->kspace) {
    force->kspace->setup();
    if (kspace_compute_flag) force->kspace->compute(eflag,vflag);
    else force->kspace->compute_dummy(eflag,vflag);
  }

  modify->pre_reverse(eflag,vflag);
  if (force->newton) comm->reverse_comm();

  modify->setup(vflag);
  update->setupflag = 0;

}

/* ----------------------------------------------------------------------
   run for N steps
------------------------------------------------------------------------- */

void Mist::run(int n)
{
  bigint ntimestep;
  int nflag,sortflag;

  int n_post_integrate = modify->n_post_integrate;
  int n_pre_exchange = modify->n_pre_exchange;
  int n_pre_neighbor = modify->n_pre_neighbor;
  int n_post_neighbor = modify->n_post_neighbor;
  int n_pre_force = modify->n_pre_force;
  int n_pre_reverse = modify->n_pre_reverse;
  int n_post_force = modify->n_post_force;
  int n_end_of_step = modify->n_end_of_step;

  if (atom->sortfreq > 0) sortflag = 1;
  else sortflag = 0;

  for (int i = 0; i < n; i++) {
    if (timer->check_timeout(i)) {
      update->nsteps = i;
      break;
    }

    ntimestep = ++update->ntimestep;
    ev_set(ntimestep);
    if (energy_required) eflag |= 1;


    MIST_chkerr(MIST_Step(update->dt),__FILE__,__LINE__);




    // force modifications, final time integration, diagnostics

    if (n_end_of_step) modify->end_of_step();

    // all output

    if (ntimestep == output->next) {
      timer->stamp();
      output->write(ntimestep);
      timer->stamp(Timer::OUTPUT);
    }
  }
}

/* ---------------------------------------------------------------------- */

void Mist::cleanup()
{
  // Finalise MIST library
  MIST_chkerr(MIST_Cleanup(),__FILE__,__LINE__);

  delete[] masses;
  masses = NULL;

  modify->post_run();
  domain->box_too_small_check();
  update->update_time();
}

/* ----------------------------------------------------------------------
   clear force on own & ghost atoms
   clear other arrays as needed
------------------------------------------------------------------------- */

void Mist::force_clear()
{
  size_t nbytes;

  if (external_force_clear) return;

  // clear force on all particles
  // if either newton flag is set, also include ghosts
  // when using threads always clear all forces.

  int nlocal = atom->nlocal;

  if (neighbor->includegroup == 0) {
    nbytes = sizeof(double) * nlocal;
    if (force->newton) nbytes += sizeof(double) * atom->nghost;

    if (nbytes) {
      memset(&atom->f[0][0],0,3*nbytes);
      if (extraflag) atom->avec->force_clear(0,nbytes);
    }

  // neighbor includegroup flag is set
  // clear force only on initial nfirst particles
  // if either newton flag is set, also include ghosts

  }else {
    nbytes = sizeof(double) * atom->nfirst;

    if (nbytes) {
      memset(&atom->f[0][0],0,3*nbytes);
      if (extraflag) atom->avec->force_clear(0,nbytes);
    }

    if (force->newton) {
      nbytes = sizeof(double) * atom->nghost;

      if (nbytes) {
        memset(&atom->f[nlocal][0],0,3*nbytes);
        if (extraflag) atom->avec->force_clear(nlocal,nbytes);
      }
    }
  }
}

void Mist::MIST_chkerr(int misterr, const char* file,int line){

  char errmsg[100];

  if (misterr == MIST_NO_SETUP_FILE){
    strcpy(errmsg,"MIST reports it cannot find a MIST setup file.");
  } else if(misterr == MIST_INVALID_SYSTEM){
    strcpy(errmsg,"MIST reports it does not have a system defined. Has MIST_Init() been called?");
  }  else if(misterr == MIST_INVALID_POINTER_IN){
    strcpy(errmsg,"MIST reports LAMMPS is passing a null pointer to it.");
  }  else if(misterr == MIST_INVALID_POINTER_OUT){
    strcpy(errmsg,"MIST is attempting to output a null pointer to LAMMPS.");
  }  else if(misterr == MIST_INVALID_VALUE_IN){
    strcpy(errmsg,"MIST reports LAMMPS is passing an invalid value to it.");
  }  else if(misterr == MIST_INVALID_OPTION){
    strcpy(errmsg,"MIST reports LAMMPS is passing an invalid option to it.");
  }  else if(misterr == MIST_INVALID_INTEGRATOR){
    strcpy(errmsg,"MIST reports it doesn't have an integrator defined. Has MIST_Init() been called?");
  } else {
    strcpy(errmsg,"Unknown MIST error.");
  }

  if (misterr != MIST_OK){
    char errstr[120];
    sprintf(errstr,"Mist error: %s",errmsg);
    error->all(FLERR,errstr);
  }
}



void Mist::lammps_force_wrapper(void *data)
{

  Mist *m = (Mist *)data;
  m->update_forces();
}

void Mist::update_forces()
{
  timer->stamp();
  if (modify->n_post_integrate) modify->post_integrate();
  timer->stamp(Timer::MODIFY);

  // regular communication vs neighbor list rebuild

  int nflag = neighbor->decide();

  if (nflag == 0) {
    timer->stamp();
    comm->forward_comm();
    timer->stamp(Timer::COMM);
  } else {
    if (modify->n_pre_exchange) {
      timer->stamp();
      modify->pre_exchange();
      timer->stamp(Timer::MODIFY);
    }
    domain->pbc();
    if (domain->box_change) {
      domain->reset_box();
      comm->setup();
      if (neighbor->style) neighbor->setup_bins();
    }
    timer->stamp();
    comm->exchange();
    if (atom->sortfreq > 0 && update->ntimestep >= atom->nextsort) atom->sort();
    comm->borders();
    timer->stamp(Timer::COMM);
    if (modify->n_pre_neighbor) {
      modify->pre_neighbor();
      timer->stamp(Timer::MODIFY);
    }
    neighbor->build(1);
    timer->stamp(Timer::NEIGH);
    if (modify->n_post_neighbor) {
      modify->post_neighbor();
      timer->stamp(Timer::MODIFY);
    }
  }

  // force computations
  // important for pair to come before bonded contributions
  // since some bonded potentials tally pairwise energy/virial
  // and Pair:ev_tally() needs to be called before any tallying

  force_clear();

  timer->stamp();

  if (modify->n_pre_force) {
    modify->pre_force(vflag);
    timer->stamp(Timer::MODIFY);
  }

  if (pair_compute_flag) {
    force->pair->compute(eflag,vflag);
    timer->stamp(Timer::PAIR);
  }

  if (atom->molecular) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
    timer->stamp(Timer::BOND);
  }

  if (kspace_compute_flag) {
    force->kspace->compute(eflag,vflag);
    timer->stamp(Timer::KSPACE);
  }

  if (modify->n_pre_reverse) {
    modify->pre_reverse(eflag,vflag);
    timer->stamp(Timer::MODIFY);
  }

  // reverse communication of forces

  if (force->newton) {
    comm->reverse_comm();
    timer->stamp(Timer::COMM);
  }

  // force modification

  if (modify->n_post_force) modify->post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void Mist::mist_setup(){

  // Initialise MIST library

  MIST_chkerr(MIST_Init(),__FILE__,__LINE__);

  double amu = 1.0; // for lj, real, metal, and electron units

  if (strcmp(update->unit_style,"si") == 0) {
    amu = 1.660539040e-27;
  } else if (strcmp(update->unit_style,"cgs") == 0) {
    amu = 1.660539040e-24;
  } else if (strcmp(update->unit_style,"micro") == 0) {
    amu = 1.660539040e-12;
  } else if (strcmp(update->unit_style,"nano") == 0) {
    amu = 1.660539040e-6;
  }

  // masses are stored scaled by a factor 1/ftm2v
  MIST_chkerr(MIST_SetUnitSystem(force->boltz, force->femtosecond*1000.0, amu*force->ftm2v, force->angstrom ),__FILE__,__LINE__);


  int features;
  MIST_chkerr(MIST_GetFeatures(&features),__FILE__,__LINE__);

  if (features & MIST_FEATURE_FORCE_COMPONENTS) error->all(FLERR, "Access to individual components of the force-field is not implemented in LAMMPS");
  if (features & MIST_FEATURE_POS_VEL_OFFSET_PLUS_HALF_DT) error->all(FLERR, "Time-offset positions and velocities is not implemented in LAMMPS");
  if (features & MIST_FEATURE_REQUIRES_KINDS) error->all(FLERR, "Access to particle kinds is not implemented in LAMMPS");
  if (features & MIST_FEATURE_REQUIRES_ENERGY_WITH_FORCES) {
    energy_required = true;

    // compute for potential energy
    int id = modify->find_compute("thermo_pe");
    if (id < 0) error->all(FLERR,"MIST could not find thermo_pe compute");
    pe_compute = modify->compute[id];
    MIST_chkerr(MIST_SetPotentialEnergy(&(pe_compute->scalar)),__FILE__,__LINE__);
  }

  if (domain->xperiodic) {
    MIST_chkerr(MIST_SetCell(domain->boxhi[0]-domain->boxlo[0], 0.0, 0.0,
                             0.0, domain->boxhi[1]-domain->boxlo[1], 0.0,
                             0.0, 0.0, domain->boxhi[2]-domain->boxlo[2]),__FILE__,__LINE__);
  }

  // XXX only works in serial case
  int natoms = atom->nlocal;//atom->natoms;
  MIST_chkerr(MIST_SetNumParticles(natoms),__FILE__,__LINE__);
  MIST_chkerr(MIST_SetKinds(atom->type),__FILE__,__LINE__);
  MIST_chkerr(MIST_SetPositions(*atom->x),__FILE__,__LINE__);
  MIST_chkerr(MIST_SetVelocities(*atom->v),__FILE__,__LINE__);
  MIST_chkerr(MIST_SetForces(*atom->f),__FILE__,__LINE__);

  MIST_chkerr(MIST_SetForceCallback(lammps_force_wrapper, (void *)this),__FILE__,__LINE__);

  // XXX serial only
  masses = new double [sizeof(double)*atom->nlocal];
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;

  if (rmass) {
    for (int i = 0; i < natoms; i++)
      masses[i]= rmass[i] / force->ftm2v;
  } else {
    for (int i = 0; i < natoms; i++)
      masses[i]= mass[type[i]] / force->ftm2v;
  }

  MIST_chkerr(MIST_SetMasses(masses),__FILE__,__LINE__);

  int nbonds=atom->nbonds;
  int counter=0;

  MIST_chkerr(MIST_SetNumBonds(nbonds),__FILE__,__LINE__);

  if (atom->molecular == 1) {
    for (int i = 0; i < natoms; i++) {
      for (int b=0; b<atom->num_bond[i]; b++){
        int j = atom->map(atom->bond_atom[i][b]); // local atom index
        double *y=atom->x[j];
        double tmp[3];
        MathExtra::sub3(atom->x[i],y, tmp);
        domain->minimum_image(tmp);
        double dist = MathExtra::len3(tmp);

        bool hydrogen = lround(masses[i] * force->ftm2v) == 1 || lround(masses[j] * force->ftm2v) == 1;

        MIST_chkerr(MIST_SetBond(counter, // bond index
                    i, // atom a
                    j, // atom b
                    dist, // bond length
                    hydrogen // if either atom is a hydrogen
                    ),__FILE__,__LINE__);

                    counter++;
      }
    }
  } else if (atom->molecular == 2) {
    error->all(FLERR, "Molecule template system not implemented with MIST");
  }

}
