#include <istream>
#include <ostream>
#include <fstream>
#include <iostream>

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

int main()
{
	std::ifstream Afile, bfile;
	Afile.open("A.csv");
	bfile.open("b.csv");
	auto Ain = xt::load_csv<double>(Afile);
	auto bin = xt::load_csv<double>(bfile);
	
	xt::xtensor<double, 2> A = Ain;
	xt::xtensor<double, 1> b = xt::squeeze(bin);

	auto sA = xt::adapt(A.shape());
	std::cout << "Size of A matrix:" << sA << std::endl;

	auto sb  = xt::adapt(b.shape());
	std::cout << "Size of b vector" << sb << std::endl;

	// The above was mostly just checking and testing. Below is the implementation
	// Im trying to follow boyd as closely as possible and not worrying about MPI for now
	
	void soft_threshold(xt::xtensor<double, 1> &v, double a);

	int m = sA(0);
	int n = sA(1);
	bool skinny = sA(1)>sA(0);
	const int MAX_ITER  = 50;
 	const double RELTOL = 1e-2;
	const double ABSTOL = 1e-4;
	
	/*
  	 * The lasso regularization parameter here is just hardcoded
  	 * to 0.5 for simplicity. Using the lambda_max heuristic would require 
  	 * network communication, since it requires looking at the *global* A^T b.
  	 */
	double lambda = 0.5;	
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
	//xt::xtensor<double, 1> y = xt::zeros<double>({n});
	xt::xtensor<double, 1> r = xt::zeros<double>({n});
	//xt::xtensor<double, 1> zdiff = xt::zeros<double>({n});


	xt::xtensor<double,1> Atb = xt::linalg::dot(xt::transpose(A), b);
	//skip the skinny check for now, assume the matrix is fat
	/* L = chol(I + 1/rho*AAt) */
	auto eye_m = xt::eye(m);	
	auto AAt = xt::linalg::dot(A,xt::transpose(A));
	// xtensor doesnt natively have cholesky solve so this is unneccessary for now
	//xt::xtensor<double, 2> L = xt::linalg::cholesky(eye_m +(1./rho)*AtA);
	xt::xtensor<double, 2> LUinv = xt::linalg::inv(eye_m + (1./rho)*AAt);
	// In boyd/matlab notation LUinv = U \ L \
	// precompute this instead of cholesky factor
	
	int iter = 0;
	std::printf("%3s %10s %10s %10s %10s %10s\n", "#", "r norm", "eps_pri", "s norm", "eps_dual", "objective");

	while (iter < MAX_ITER) {
		// u update	
		u += x - z;
		//std::cout << "||u||_2 " << xt::linalg::norm(u,2) << std::endl;

		//x update
		auto q = Atb + rho*(z-u);
		//std::cout << "||q||_2 " << xt::linalg::norm(q,2) << std::endl;

		auto Aq = xt::linalg::dot(A,q);
		auto p = xt::linalg::dot(LUinv,Aq);//Just precompute the entire inverse rather than chol factor
		auto xtemp = xt::linalg::dot(xt::transpose(A) ,p);
		//std::cout << "||A^Tp||_2 " << xt::linalg::norm(xtemp,2) << std::endl;

		x = (1./rho)*q - (1./(rho*rho))*xtemp;
		//std::cout <<"||x||_2 " << xt::linalg::norm(x,2) << std::endl;

		auto w = x+u;
		//Message passing should go here
		
		prires = xt::linalg::norm(r, 2);
		nxstack = xt::linalg::norm(x, 2);
		nystack = xt::linalg::norm(u,2)/rho;

		auto zprev = z;
		z = w;
		soft_threshold(z, lambda/rho);
		auto zdiff = z - zprev;
		//std::cout << "||z||_2 " << xt::linalg::norm(z,2) << std::endl; 
		dualres = rho*xt::linalg::norm(zdiff, 2);	

		eps_pri = sqrt(n)*ABSTOL + RELTOL*fmax(nxstack, xt::linalg::norm(z, 2));
		eps_dual = sqrt(n)*ABSTOL + RELTOL*nystack;

		/*
		 * double obj = 0.5*xt::eval(xt::norm_sq(xt::linalg::dot(A,z) - b, {0}))(0);
		 * obj += lambda*xt::eval(xt::norm_l1(z, {0}))(0);
		 */
		double Azb_nrm = xt::linalg::norm(xt::linalg::dot(A,z)-b, 2);
		double obj = 0.5*Azb_nrm*Azb_nrm + lambda*xt::linalg::norm(z,1);

		std::printf("%3d %10.4f %10.4f %10.4f %10.4f %10.4f\n", iter, prires, eps_pri, dualres, eps_dual, obj);

		if((prires <= eps_pri) && (dualres <= eps_dual)){
			break;
		}

		r = x-z;
	 	//std::cout << "||r||_2 " << xt::linalg::norm(r,2) << std::endl;
		iter++;
	}
	std::ofstream sol_file;
	sol_file.open("xt_solution.csv");
	xt::dump_csv(sol_file, xt::expand_dims(z,1));
	return 0;
}

void soft_threshold(xt::xtensor<double,1> &v, double a){
	xt::filtration(v, xt::abs(v)<=a) = 0;
	xt::filtration(v, v>a) -= a;
	xt::filtration(v, v<-1*a) += a;
	
}

