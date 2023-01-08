#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
                                                                        
    return(0); // exit succcesfully
} // end of the mover



/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));

        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}


// move particle function for GPU version. 
// same functionality as mover_PC but split up into the GPU preparations and the function
__global__ void move_particles_GPU(particles* part_GPU, EMfield* field_GPU, grid* grd_GPU, parameters* param_GPU) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= part_GPU->nop) return;
    
    // error "calling a host function from a __global__ function is not allowed" -> I've put it inside the mover_PC_GPU code
    // print species and subcycling
    //std::cout << "***  MOVER with SUBCYCLYING "<< param_GPU->n_sub_cycles << " - species " << part_GPU->species_ID << " ***" << std::endl;

    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param_GPU->dt/((double) part_GPU->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part_GPU->qom*dto2/param_GPU->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub < part_GPU->n_sub_cycles; i_sub++){
        // move each particle with new fields
        //for (int i=0; i < part_GPU->nop; i++){
            xptilde = part_GPU->x[idx];
            yptilde = part_GPU->y[idx];
            zptilde = part_GPU->z[idx];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part_GPU->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part_GPU->x[idx] - grd_GPU->xStart)*grd_GPU->invdx);
                iy = 2 +  int((part_GPU->y[idx] - grd_GPU->yStart)*grd_GPU->invdy);
                iz = 2 +  int((part_GPU->z[idx] - grd_GPU->zStart)*grd_GPU->invdz);
                
                // calculate weights
                xi[0]   = part_GPU->x[idx] - grd_GPU->XN_flat[get_idx(ix - 1, iy, iz, grd_GPU->nyn, grd_GPU->nzn)];
                eta[0]  = part_GPU->y[idx] - grd_GPU->YN_flat[get_idx(ix, iy - 1, iz, grd_GPU->nyn, grd_GPU->nzn)];
                zeta[0] = part_GPU->z[idx] - grd_GPU->ZN_flat[get_idx(ix, iy, iz - 1, grd_GPU->nyn, grd_GPU->nzn)];
                xi[1]   = grd_GPU->XN_flat[get_idx(ix, iy, iz, grd_GPU->nyn, grd_GPU->nzn)] - part_GPU->x[idx];
                eta[1]  = grd_GPU->YN_flat[get_idx(ix, iy, iz, grd_GPU->nyn, grd_GPU->nzn)] - part_GPU->y[idx];
                zeta[1] = grd_GPU->ZN_flat[get_idx(ix, iy, iz, grd_GPU->nyn, grd_GPU->nzn)] - part_GPU->z[idx];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd_GPU->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field_GPU->Ex_flat[get_idx(ix-ii, iy-jj, iz-kk, grd_GPU->nyn, grd_GPU->nzn)];
                            Eyl += weight[ii][jj][kk]*field_GPU->Ey_flat[get_idx(ix-ii, iy-jj, iz-kk, grd_GPU->nyn, grd_GPU->nzn)];
                            Ezl += weight[ii][jj][kk]*field_GPU->Ez_flat[get_idx(ix-ii, iy-jj, iz-kk, grd_GPU->nyn, grd_GPU->nzn)];
                            Bxl += weight[ii][jj][kk]*field_GPU->Bxn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd_GPU->nyn, grd_GPU->nzn)];
                            Byl += weight[ii][jj][kk]*field_GPU->Byn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd_GPU->nyn, grd_GPU->nzn)];
                            Bzl += weight[ii][jj][kk]*field_GPU->Bzn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd_GPU->nyn, grd_GPU->nzn)];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part_GPU->u[idx] + qomdt2*Exl;
                vt= part_GPU->v[idx] + qomdt2*Eyl;
                wt= part_GPU->w[idx] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part_GPU->x[idx] = xptilde + uptilde*dto2;
                part_GPU->y[idx] = yptilde + vptilde*dto2;
                part_GPU->z[idx] = zptilde + wptilde*dto2;            
                
            } // end of iteration
            // update the final position and velocity
            part_GPU->u[idx]= 2.0*uptilde - part_GPU->u[idx];
            part_GPU->v[idx]= 2.0*vptilde - part_GPU->v[idx];
            part_GPU->w[idx]= 2.0*wptilde - part_GPU->w[idx];
            part_GPU->x[idx] = xptilde + uptilde*dt_sub_cycling;
            part_GPU->y[idx] = yptilde + vptilde*dt_sub_cycling;
            part_GPU->z[idx] = zptilde + wptilde*dt_sub_cycling;
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part_GPU->x[idx] > grd_GPU->Lx){
                if (param_GPU->PERIODICX==true){ // PERIODIC
                    part_GPU->x[idx] = part_GPU->x[idx] - grd_GPU->Lx;
                } else { // REFLECTING BC
                    part_GPU->u[idx] = -part_GPU->u[idx];
                    part_GPU->x[idx] = 2*grd_GPU->Lx - part_GPU->x[idx];
                }
            }
                                                                        
            if (part_GPU->x[idx] < 0){
                if (param_GPU->PERIODICX==true){ // PERIODIC
                   part_GPU->x[idx] = part_GPU->x[idx] + grd_GPU->Lx;
                } else { // REFLECTING BC
                    part_GPU->u[idx] = -part_GPU->u[idx];
                    part_GPU->x[idx] = -part_GPU->x[idx];
                }
            }
            
            // Y-DIRECTION: BC particles
            if (part_GPU->y[idx] > grd_GPU->Ly){
                if (param_GPU->PERIODICY==true){ // PERIODIC
                    part_GPU->y[idx] = part_GPU->y[idx] - grd_GPU->Ly;
                } else { // REFLECTING BC
                    part_GPU->v[idx] = -part_GPU->v[idx];
                    part_GPU->y[idx] = 2*grd_GPU->Ly - part_GPU->y[idx];
                }
            }
                                                                        
            if (part_GPU->y[idx] < 0){
                if (param_GPU->PERIODICY==true){ // PERIODIC
                    part_GPU->y[idx] = part_GPU->y[idx] + grd_GPU->Ly;
                } else { // REFLECTING BC
                    part_GPU->v[idx] = -part_GPU->v[idx];
                    part_GPU->y[idx] = -part_GPU->y[idx];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part_GPU->z[idx] > grd_GPU->Lz){
                if (param_GPU->PERIODICZ==true){ // PERIODIC
                    part_GPU->z[idx] = part_GPU->z[idx] - grd_GPU->Lz;
                } else { // REFLECTING BC
                    part_GPU->w[idx] = -part_GPU->w[idx];
                    part_GPU->z[idx] = 2*grd_GPU->Lz - part_GPU->z[idx];
                }
            }
                                                                        
            if (part_GPU->z[idx] < 0){
                if (param_GPU->PERIODICZ==true){ // PERIODIC
                    part_GPU->z[idx] = part_GPU->z[idx] + grd_GPU->Lz;
                } else { // REFLECTING BC
                    part_GPU->w[idx] = -part_GPU->w[idx];
                    part_GPU->z[idx] = -part_GPU->z[idx];
                }
            }
                                                                        
        //}  // end of subcycling
    } // end of one particle
}


// gpu particle mover
int mover_PC_GPU(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    //variables for gpu
    particles* part_GPU;
    EMfield* field_GPU;
    grid* grd_GPU;
    parameters* param_GPU;

    // arrays of structs (see include/Particles.h, EMfield.h, Grid.h)
    FPpart* part_GPU_x; FPpart*  part_GPU_y; FPpart* part_GPU_z; FPpart* part_GPU_u; FPpart* part_GPU_v; FPpart* part_GPU_w;
    FPfield* field_GPU_Ex_flat; FPfield* field_GPU_Ey_flat; FPfield* field_GPU_Ez_flat; FPfield* field_GPU_Bxn_flat; FPfield* field_GPU_Byn_flat; FPfield* field_GPU_Bzn_flat;
    FPfield* grd_GPU_XN_flat; FPfield* grd_GPU_YN_flat; FPfield* grd_GPU_ZN_flat;

    //allocate memory
    cudaMalloc(&part_GPU, sizeof(particles));
    cudaMalloc(&field_GPU, sizeof(EMfield));
    cudaMalloc(&grd_GPU, sizeof(grid));
    cudaMalloc(&param_GPU, sizeof(parameters));

    cudaMalloc(&part_GPU_x, part->npmax * sizeof(FPpart)); // part arrays; allocating memory needs to be done separately for the arrays inside structs
    cudaMalloc(&part_GPU_y, part->npmax * sizeof(FPpart));
    cudaMalloc(&part_GPU_z, part->npmax * sizeof(FPpart));
    cudaMalloc(&part_GPU_u, part->npmax * sizeof(FPpart));
    cudaMalloc(&part_GPU_v, part->npmax * sizeof(FPpart));
    cudaMalloc(&part_GPU_w, part->npmax * sizeof(FPpart));

    cudaMalloc(&field_GPU_Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield)); //field arrays; 3D -> x*y*z needs to be allocated
    cudaMalloc(&field_GPU_Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&field_GPU_Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&field_GPU_Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&field_GPU_Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&field_GPU_Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));

    cudaMalloc(&grd_GPU_XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield)); //grid arrays; 3D -> x*y*z needs to be allocated
    cudaMalloc(&grd_GPU_YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&grd_GPU_ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));



    //copy memory from cpu to gpu
    cudaMemcpy(part_GPU, part, sizeof(particles), cudaMemcpyHostToDevice);
    cudaMemcpy(field_GPU, field, sizeof(EMfield), cudaMemcpyHostToDevice);
    cudaMemcpy(grd_GPU, grd, sizeof(grid), cudaMemcpyHostToDevice);
    cudaMemcpy(param_GPU, param, sizeof(parameters), cudaMemcpyHostToDevice);

    cudaMemcpy(part_GPU_x, part->x, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice); // part array; again separately for the arrays inside structs
    cudaMemcpy(part_GPU_y, part->y, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(part_GPU_z, part->z, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(part_GPU_u, part->u, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(part_GPU_v, part->v, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(part_GPU_w, part->w, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMemcpy(field_GPU_Ex_flat, field->Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice); // field array
    cudaMemcpy(field_GPU_Ey_flat, field->Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(field_GPU_Ez_flat, field->Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(field_GPU_Bxn_flat, field->Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(field_GPU_Byn_flat, field->Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(field_GPU_Bzn_flat, field->Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMemcpy(grd_GPU_XN_flat, grd->XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice); //grid array
    cudaMemcpy(grd_GPU_YN_flat, grd->YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(grd_GPU_ZN_flat, grd->ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    // All the array pointers need to be binded to the struct again. 
    // See https://www.appsloveworld.com/cplus/100/134/cuda-cudamemcpy-struct-of-arrays
    cudaMemcpy(&(part_GPU->x), &part_GPU_x, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(part_GPU->y), &part_GPU_y, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(part_GPU->z), &part_GPU_z, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(part_GPU->u), &part_GPU_u, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(part_GPU->v), &part_GPU_v, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(part_GPU->w), &part_GPU_w, sizeof(FPpart*), cudaMemcpyHostToDevice);

    cudaMemcpy(&(field_GPU->Ex_flat), &field_GPU_Ex_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(field_GPU->Ey_flat), &field_GPU_Ey_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(field_GPU->Ez_flat), &field_GPU_Ez_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(field_GPU->Bxn_flat), &field_GPU_Bxn_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(field_GPU->Byn_flat), &field_GPU_Byn_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(field_GPU->Bzn_flat), &field_GPU_Bzn_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);

    cudaMemcpy(&(grd_GPU->XN_flat), &grd_GPU_XN_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(grd_GPU->YN_flat), &grd_GPU_YN_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(grd_GPU->ZN_flat), &grd_GPU_ZN_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);

    //Init block and grid dims
    int Db_d = 64;
    int Dg_d = (part->nop + Db_d - 1) / Db_d;


    // when I used this inside the "mover_particles_GPU" function, I've got the error 
    //"calling a host function from a __global__ function is not allowed" -> I've put it inside the mover_PC_GPU code
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    //Launch GPU kernel
    move_particles_GPU<<<Dg_d, Db_d>>>(part_GPU, field_GPU, grd_GPU, param_GPU);


    // additional variables for cpu for the arrays inside the structs
    FPpart* part_CPU_x; FPpart* part_CPU_y; FPpart* part_CPU_z; FPpart* part_CPU_u; FPpart* part_CPU_v; FPpart* part_CPU_w;


    // allocate memory on cpu
    part_CPU_x = (FPpart*) malloc(part->npmax * sizeof(FPpart));
    part_CPU_y = (FPpart*) malloc(part->npmax * sizeof(FPpart));
    part_CPU_z = (FPpart*) malloc(part->npmax * sizeof(FPpart));
    part_CPU_u = (FPpart*) malloc(part->npmax * sizeof(FPpart));
    part_CPU_v = (FPpart*) malloc(part->npmax * sizeof(FPpart));
    part_CPU_w = (FPpart*) malloc(part->npmax * sizeof(FPpart));

    //copy memory from gpu to cpu
    cudaMemcpy(part, part_GPU, sizeof(particles), cudaMemcpyDeviceToHost);

    cudaMemcpy(part_CPU_x, part_GPU_x, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost); // arrays need to be copied separately again
    cudaMemcpy(part_CPU_y, part_GPU_y, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part_CPU_z, part_GPU_z, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part_CPU_u, part_GPU_u, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part_CPU_v, part_GPU_v, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part_CPU_w, part_GPU_w, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);

    // All the array pointers need to be binded to the struct again. 
    // See https://www.appsloveworld.com/cplus/100/134/cuda-cudamemcpy-struct-of-arrays
    part->x = part_CPU_x;
    part->y = part_CPU_y;
    part->z = part_CPU_z;
    part->u = part_CPU_u;
    part->v = part_CPU_v;
    part->w = part_CPU_w;

    //free gpu mem
    cudaFree(part_GPU);
    cudaFree(field_GPU);
    cudaFree(grd_GPU);
    cudaFree(param_GPU);                                                               

    return(0); // "return;"" is not working because "error: it should return a value"
} 


