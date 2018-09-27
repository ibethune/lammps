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

Mist_run::Mist_run(LAMMPS *lmp, int narg, char **arg) :
  Integrate(lmp, narg, arg) {

pointerToData=new Pointers(lmp);
//potEnergyPtr=NULL;

//step_force_wrapper_pointer=NULL;

}

/* ----------------------------------------------------------------------
   initialization before run
------------------------------------------------------------------------- */

void Mist_run::init()
{
  pointerToData=this;

  Integrate::init(); 

  // warn if any fixes - mist does everything

  if (modify->nfix != 0 && comm->me == 0)
    error->warning(FLERR,"Fixes have been defined, but MIST may ignore them!");

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


  // Initialise MIST library

  MIST_chkerr(MIST_Init(),__FILE__,__LINE__);


}

/* ----------------------------------------------------------------------
   setup before run
------------------------------------------------------------------------- */

void Mist_run::setup(int flag)
{
  if (comm->me == 0 && screen) {
    fprintf(screen,"Setting up MIST run ...\n");
    fprintf(screen,"  Unit style    : %s\n", update->unit_style);
    if (strcmp(update->unit_style,"lj") != 0) {
      error->all(FLERR,"MIST only works with lj units");
    }
    fprintf(screen,"  Current step  : " BIGINT_FORMAT "\n", update->ntimestep);
    fprintf(screen,"  Time step     : %g\n", update->dt);
    timer->print_timeout(screen);
  }

  if (lmp->kokkos)
    error->all(FLERR,"KOKKOS package not supported by run_style mist");

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

  // compute all forces

  force->setup();
  ev_set(update->ntimestep);
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

  // Set up MIST library
  mist_setup();

}

/* ----------------------------------------------------------------------
   setup without output
   flag = 0 = just force calculation
   flag = 1 = reneighbor and force calculation
------------------------------------------------------------------------- */

void Mist_run::setup_minimal(int flag)
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

  // compute all forces

  ev_set(update->ntimestep);
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

  // Set up MIST library
  mist_setup();
}

/* ----------------------------------------------------------------------
   run for N steps
------------------------------------------------------------------------- */

void Mist_run::run(int n)
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


 // TODO: make sure the timers get updated
 // TODO: put pre and post-force in force callback
 // TODO: check if any of the modify calls are required

    // regular communication vs neighbor list rebuild

    nflag = neighbor->decide();

    if (nflag == 0) {
      timer->stamp();
      comm->forward_comm();
      timer->stamp(Timer::COMM);
    } else {
      if (n_pre_exchange) {
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
      if (sortflag && ntimestep >= atom->nextsort) atom->sort();
      comm->borders();
      timer->stamp(Timer::COMM);
      if (n_pre_neighbor) {
        modify->pre_neighbor();
        timer->stamp(Timer::MODIFY);
      }
      neighbor->build(1);
      timer->stamp(Timer::NEIGH);
      if (n_post_neighbor) {
        modify->post_neighbor();
        timer->stamp(Timer::MODIFY);
      }
    }


     //potEnergyPtr[0]= pe->compute_scalar();
      MIST_chkerr(MIST_Step(update->dt),__FILE__,__LINE__);


    // force modifications, final time integration, diagnostics

    if (n_post_force) modify->post_force(vflag);
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

void Mist_run::cleanup()
{
  // Finalise MIST library
  MIST_chkerr(MIST_Cleanup(),__FILE__,__LINE__);

  modify->post_run();
  domain->box_too_small_check();
  update->update_time();
}

/* ----------------------------------------------------------------------
   clear force on own & ghost atoms
   clear other arrays as needed
------------------------------------------------------------------------- */

void Mist_run::force_clear()
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

void Mist_run::MIST_chkerr(int misterr, const char* file,int line){

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



void Mist_run::step_force_wrapper(void * s)
{

  update_forces_step((Mist_run *) s);

}

void Mist_run::update_forces_step(Mist_run* m){


  LAMMPS *lp = m->lmp;

  int eeflag=lp->update->eflag_global;
  int vvflag=lp->update->vflag_global;

    // force computations
    // important for pair to come before bonded contributions
    // since some bonded potentials tally pairwise energy/virial
    // and Pair:ev_tally() needs to be called before any tallying


    m->force_clear();


        lp->timer->stamp();

        // no preforce in mist
        //if (n_pre_force) {
        //  modify->pre_force(vflag);
        //  timer->stamp(Timer::MODIFY);
        //}

        if (m->pair_compute_flag) {
          lp->force->pair->compute(eeflag,vvflag);
          lp->timer->stamp(Timer::PAIR);
        }

        if (lp->atom->molecular) {
          if (lp->force->bond) lp->force->bond->compute(eeflag,vvflag);
          if (lp->force->angle) lp->force->angle->compute(eeflag,vvflag);
          if (lp->force->dihedral) lp->force->dihedral->compute(eeflag,vvflag);
          if (lp->force->improper) lp->force->improper->compute(eeflag,vvflag);
          lp->timer->stamp(Timer::BOND);
        }

        if (m->kspace_compute_flag) {
          lp->force->kspace->compute(eeflag,vvflag);
          lp->timer->stamp(Timer::KSPACE);
        }

        int n_pre_reverse = lp->modify->n_pre_reverse;
        if (n_pre_reverse) {
          lp->modify->pre_reverse(eeflag,vvflag);
          lp->timer->stamp(Timer::MODIFY);
        }


        // reverse communication of forces

        if (lp->force->newton) {
          lp->comm->reverse_comm();
          lp->timer->stamp(Timer::COMM);
        }

}



void Mist_run::mist_setup(){

  // MIST setup
      int feature;
      MIST_chkerr(MIST_GetFeatures(&feature),__FILE__,__LINE__);
      fprintf(screen, "TO BE DONE: Mist feature = %d\n", feature);

      // Set pointers for MIST to access particle data
      //in atom- positions as double **x

//number of atoms is natoms or nlocal- parallel..???

    MIST_chkerr(MIST_SetNumParticles(atom->nlocal),__FILE__,__LINE__);
    MIST_chkerr(MIST_SetKinds(atom->type),__FILE__,__LINE__);
    MIST_chkerr(MIST_SetPositions(*atom->x),__FILE__,__LINE__);
    MIST_chkerr(MIST_SetVelocities(*atom->v),__FILE__,__LINE__);
    MIST_chkerr(MIST_SetForces(*atom->f),__FILE__,__LINE__);

      fprintf(screen, "To be done: set potential energy\n");
  //  MIST_chkerr(MIST_SetPotentialEnergy(*atom->f),__FILE__,__LINE__);

    MIST_chkerr(MIST_SetForceCallback(step_force_wrapper, (void *)this),__FILE__,__LINE__);

    double *masses = new double [sizeof(double)*atom->nlocal];//atom->natoms];
    double *mass = atom->mass;
    double *rmass = atom->rmass;
    int *type = atom->type;
    int *mask = atom->mask;
    int nratoms = atom->nlocal;//atom->natoms;

      if (rmass) {
        for (int i = 0; i < nratoms; i++)
          if (mask[i] ) {
              masses[i]= rmass[i] / force->ftm2v;

        }

      } else {
        for (int i = 0; i < nratoms; i++)
          if (mask[i] ) {
            masses[i]= mass[type[i]] / force->ftm2v;

            }
        }



    MIST_chkerr(MIST_SetMasses(masses),__FILE__,__LINE__);


 int nbonds= atom->nbonds;
 int counter=0;


//if (atom->molecular) {}

    MIST_chkerr(MIST_SetNumBonds(nbonds),__FILE__,__LINE__);


if (atom->molecular) {
    for (int i = 0; i < nratoms; i++)
    {
      for (int j=0; j<atom->num_bond[i]; j++){



        double *y=atom->x[atom->bond_atom[i][j]];
        double *tmp=new double[3];
//fprintf(screen, "Atom position y = %f %f %f \n", y[0], y[1], y[2]);

        MathExtra::sub3(atom->x[i],y, tmp);
        double dist = MathExtra::len3(tmp);

        //  fprintf(screen,"dist =%f\n", dist);

       MIST_chkerr(MIST_SetBond(counter, // bond index
                    i, // atom a
                    atom->bond_atom[i][j], // atom b
                    dist, // bond length
                  false // if either atom is a hydrogen
                    ),__FILE__,__LINE__);

                    //double tmp2=  atom->bond->eatom[counter];
                    //->single(atom->bond_type[i][atom->bond_atom[i][j]],dist,i,  atom->bond_atom[i][j],tmp2);
            //        fprintf(screen, "\n bond force component %f\n", tmp2);
                      //while(1);

                    counter++;
              }
    }
  }


}
