#include <istream>
#include <fstream>
#include <iostream>

#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xcsv.hpp"
#include "xtensor-blas/xlinalg.hpp"

int main()
{
	std::ifstream Afile, bfile;
	Afile.open("A.csv");
	bfile.open("b.csv");
	auto A = xt::load_csv<double>(Afile);
	auto b = xt::load_csv<double>(bfile);
	
	xt::xtensor<double, 2> Atensor = A;

	std::cout << "Size of A matrix:" << std::endl;

	auto sA = xt::adapt(A.shape());

	bool skinny = sA(1)>sA(0);

	std::cout << sA << std::endl;
	
	std::cout << "Size of b vector" << std::endl;

	auto sb  = xt::adapt(b.shape());

	std::cout << sb << std::endl;

	// The above was mostly just checking and testing. Below is the implementation
	// Im trying to follow boyd as closely as possible and not worrying about MPI for now

	double rho = 1;
	const int MAX_ITER  = 50;
 	const double RELTOL = 1e-2;
	const double ABSTOL = 1e-4;
	
	xt::xtensor<double,2> Atb = xt::linalg::dot(xt::transpose(A), b);

	//skip the skinny check for now, assume the matrix is fat
	/* L = chol(I + 1/rho*AAt) */


	auto eye = xt::eye(sA(1));	
	
	auto AtA = xt::linalg::dot(xt::transpose(A), A);

	xt::xtensor<double, 2> L = xt::linalg::cholesky(eye+(1./rho)*AtA);

	

	return 0;

}
