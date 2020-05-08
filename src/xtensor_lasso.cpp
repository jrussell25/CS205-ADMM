#include <mpi.h>
#include <istream>
#include <ostream>
#include <fstream>
#include <iostream>
#include <string>

#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xcsv.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xnorm.hpp"
#include "xtensor-blas/xlinalg.hpp"



std::string DATA_DIR("./data/");

int main(int argc, char *argv[])
{
    //initialize MPI
    int ierr = MPI_Init(&argc, &argv);
    int size;  //number of processes
    int rank;  //MPI process id
    if (ierr != 0) {
        std::cout << "\n";
        std::cout << "MPI - Fatal error!\n";
        std::cout << "MPI_Init returned nonzero ierr.\n";
        exit(1);
    }

    ierr = MPI_Comm_size(MPI_COMM_WORLD, &size);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double N = (double) size;  // Number of subsystems/slaves for ADMM

    std::ifstream Afile, bfile;
    std::ifstream true_sol_file;
    std::string Afile_name(DATA_DIR + "A" + std::to_string(rank) + ".csv");
    std::string bfile_name(DATA_DIR + "b" + std::to_string(rank) + ".csv");
    std::string true_solution_name(DATA_DIR+"xtrue.csv");
    Afile.open(Afile_name);
    bfile.open(bfile_name);
    true_sol_file.open(true_solution_name);
    auto Ain = xt::load_csv<double>(Afile);
    auto bin = xt::load_csv<double>(bfile);
    auto true_sol_in=xt::load_csv<double>(true_sol_file);

    xt::xtensor<double, 2> A = Ain;
    xt::xtensor<double, 1> b = xt::squeeze(bin);
    xt::xtensor<double, 1> true_sol = xt::squeeze(true_sol_in);

    auto sA = xt::adapt(A.shape());
    if (rank==0){
        std::cout << "Using " << N << " MPI processes" << std::endl;
        std::cout << "Size of per processor A matrix: " << sA << std::endl;
    }

    auto sb  = xt::adapt(b.shape());
    //std::cout << "Size of b vector" << sb << std::endl;

    // The above was mostly just checking and testing. Below is the implementation
    // Im trying to follow boyd as closely as possible and not worrying about MPI for now
    
    void soft_threshold(xt::xtensor<double, 1> &v, const double a);
    void xtensor2array(const xt::xtensor<double, 1> &x, double* ptr); //copy x to ptr for MPI pessage passing
    void array2xtensor(xt::xtensor<double, 1> &x, double* ptr); //copy x to ptr for MPI pessage passing
    void ComputeError(xt::xtensor<double, 1> &z, xt::xtensor<double,1> &true_sol);

    int m = sA(0);
    int n = sA(1);
    bool skinny = sA(1)>sA(0);
    const int MAX_ITER  = 100;
    const double RELTOL = 1e-3;
    const double ABSTOL = 1e-5;
    
    /*
     * The lasso regularization parameter here is just hardcoded
     * to 0.5 for simplicity. Using the lambda_max heuristic would require 
     * network communication, since it requires looking at the *global* A^T b.
     */
    double send[3]; // an array used to aggregate 3 scalars at once
    double recv[3]; // used to receive the results of these aggregations

    double lambda;// use global lambda max heuristic = 0.5;	
    double rho = 1.0;
    double prires = 0;
    double dualres = 0;
    double eps_pri = 0;
    double eps_dual = 0;
    double nxstack = 0;
    double nystack = 0;

    //We may not actually need many of these in memory given 
    //lazy evaluation. i.e. many could just be autos
    xt::xtensor<double, 1> x = xt::zeros<double>({n});
    xt::xtensor<double, 1> u = xt::zeros<double>({n});
    xt::xtensor<double, 1> z = xt::zeros<double>({n});
    xt::xtensor<double, 1> r = xt::zeros<double>({n});

    double* mpi_w_ptr = new double[n];   //double array for holding vector sending with MPI
    double* mpi_z_ptr = new double[n];


    xt::xtensor<double,1> Atb = xt::linalg::dot(xt::transpose(A), b);

    double lambda_max_local = xt::linalg::norm(Atb,xt::linalg::normorder::inf);	

    MPI_Allreduce(&lambda_max_local, &lambda, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    lambda = 0.1*lambda;

    // xtensors inv uses cholesky factorization under the hood
    // In boyd/matlab notation LUinv = U \ L \
    // precompute this instead of cholesky factor
    xt::xtensor<double, 2> LUinv; 
    if (skinny){
        //L = chol(AtA+rho*I)
        auto eye_n = xt::eye(n);
        auto AtA = xt::linalg::dot(xt::transpose(A), A);
        LUinv = xt::linalg::inv(AtA+rho*eye_n);
    }
    else{
        /* L = chol(I + 1/rho*AAt) */
        auto eye_m = xt::eye(m);	
        auto AAt = xt::linalg::dot(A,xt::transpose(A));
        LUinv = xt::linalg::inv(eye_m + (1./rho)*AAt);
    }

    int iter = 0;
    if (rank == 0) {
        std::cout << "Using 0.1 x Lambda max heuristic: lambda = " << lambda << std::endl;
        std::printf("%3s %10s %10s %10s %10s %10s\n", "#", "r norm", "eps_pri", "s norm", "eps_dual", "objective");
    }
    
    while (iter < MAX_ITER) {
        // u update	
        u += x - z;
        //std::cout << "||u||_2 " << xt::linalg::norm(u,2) << std::endl;

        //x update
        auto q = Atb + rho*(z-u);
        //std::cout << "||q||_2 " << xt::linalg::norm(q,2) << std::endl;
	
	if (skinny){
	    x = xt::linalg::dot(LUinv,q);
	}
	else{
            //use matrix inversion lemma
            auto Aq = xt::linalg::dot(A,q);
            auto p = xt::linalg::dot(LUinv,Aq);
            auto xtemp = xt::linalg::dot(xt::transpose(A) ,p);
            //std::cout << "||A^Tp||_2 " << xt::linalg::norm(xtemp,2) << std::endl;

            x = (1./rho)*q - (1./(rho*rho))*xtemp;
            //std::cout <<"||x||_2 " << xt::linalg::norm(x,2) << std::endl;
        }
        xt::xtensor<double, 1> w = x+u;
        //Message passing should go here

        send[0] = xt::linalg::vdot(r, r);
        send[1] = xt::linalg::vdot(x, x);
        send[2] = xt::linalg::vdot(u, u) / pow(rho, 2);

        auto zprev = z;
        //copy xtensor to double array
        xtensor2array(w, mpi_w_ptr);
        xtensor2array(z, mpi_z_ptr);

        MPI_Allreduce(mpi_w_ptr, mpi_z_ptr,  n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(send,    recv,     3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        array2xtensor(w, mpi_w_ptr);
        array2xtensor(z, mpi_z_ptr);
        prires = sqrt(recv[0]);
        nxstack = sqrt(recv[1]);
        nystack = sqrt(recv[2]);

        z = z / N;
        soft_threshold(z, lambda/(N*rho));
        auto zdiff = z - zprev;
        //std::cout << "||z||_2 " << xt::linalg::norm(z,2) << std::endl; 
        dualres = sqrt(N) * rho * xt::linalg::norm(zdiff, 2);

        eps_pri = sqrt(n * N)*ABSTOL + RELTOL*fmax(nxstack, xt::linalg::norm(z, 2)*sqrt(N));
        eps_dual = sqrt(n * N)*ABSTOL + RELTOL*nystack;

        double Azb_nrm = xt::linalg::norm(xt::linalg::dot(A,z)-b, 2);
        double obj = 0.5*Azb_nrm*Azb_nrm + lambda*xt::linalg::norm(z,1);

        if (rank == 0){
            std::printf("%3d %10.4f %10.4f %10.4f %10.4f %10.4f\n", iter, prires, eps_pri, dualres, eps_dual, obj);
        }

        if((prires <= eps_pri) && (dualres <= eps_dual)){
            break;
        }

        r = x-z;
        //std::cout << "||r||_2 " << xt::linalg::norm(r,2) << std::endl;
        iter++;
    }
    std::ofstream sol_file;
    //std::string sol_file_name(DATA_DIR + "xt_solution" + std::to_string(rank) + ".csv");
    std::string sol_file_name(DATA_DIR + "xt_solution" + ".csv");
    sol_file.open(sol_file_name);
    xt::dump_csv(sol_file, xt::expand_dims(z,1));

    MPI_Finalize();

    delete[] mpi_w_ptr;
    delete[] mpi_z_ptr;
    if (rank==0){
        computeError(z,true_sol);
    }

    return 0;
}

void soft_threshold(xt::xtensor<double,1> &v, const double a){
	xt::filtration(v, xt::abs(v)<=a) = 0;
	xt::filtration(v, v>a) -= a;
	xt::filtration(v, v<-1*a) += a;
}

void xtensor2array(const xt::xtensor<double, 1> &x, double* ptr){
    auto s = xt::adapt(x.shape());
    int n = s(0);
    for (int i = 0; i < n; i++){
        ptr[i] = x(i);
    }
}

void array2xtensor(xt::xtensor<double, 1> &x, double* ptr){
    auto s = xt::adapt(x.shape());
    int n = s(0);
    for (int i = 0; i < n; i++){
        x(i) = ptr[i];
    }
}

void ComputeError(xt::xtensor<double,1> &z, xt::xtensor<double, 1> &true_sol){
    auto err=true_sol-z;
    //double eps=0.01;
    //int n=z.size();
    //xt::xarray<double> z_abs=xt::abs(z);
    double err_abs = xt::linalg::norm(err,1);
    //xt::xarray<double> ts_abs =xt::abs(true_sol);
    //xt::xarray<double> err_abs_sum = xt::sum(err_abs);
    //xt::xarray<double> ts_abs_sum  = xt::sum(ts_abs);
    double nnz_result = xt::linalg::norm(z,0);
    double nnz_true = xt::linalg::norm(true_sol, 0);
    /*
    for(int i=0; i<n;i+=1){
        if(z[i]<eps && ts_abs[i]<eps){
            num_zero_result+=1;
        }
        if(ts_abs[i]<eps){
            num_zero_true+=1;
        }
    }
    */
    std::cout<<"Absolute Error ||x - x_true||_1 = " << err_abs <<std::endl;
    //std::cout<<"|x| norm 1 sum of true solution:"<<ts_abs_sum<<std::endl;
    std::cout<<"Number of nonzero entries in calculated solution: "<< nnz_result <<std::endl;
    std::cout<<"Number of nonzero entries in true solution: "<< nnz_true <<std::endl;

    return;
}

