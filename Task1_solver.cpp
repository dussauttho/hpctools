#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <math.h>
//#include <boost/algorithm/string.hpp>
#include <time.h>
//#include "mkl_lapacke.h"

using namespace std;


#define Nsizea 1024

// class CSVReader
// {
// 	string fileName;
// 	string delimeter;
 
// public:
// 	CSVReader(std::string filename, std::string delm = ",") :
// 			fileName(filename), delimeter(delm)
// 	{ }
 
// 	vector<vector<double> > getData(){
//         std::ifstream file(fileName);
        
//         vector<vector<double>> dataList;
        
//         std::string line = "";
//            while (getline(file, line))
//            {
//                vector<string> vec;
//                boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
//                vector<double> doublevec(vec.size());
//                transform(vec.begin(), vec.end(), doublevec.begin(), [](const std::string& val)
//                {
//                    return stod(val);
//                });
//                dataList.push_back(doublevec);
//            }
//            // Close the File
//            file.close();
        
//            return dataList;

//     }
// };

void print_matrix(const string name, vector<vector<double> > matrix) {
    cout << "matrix : " << name << endl;
    for ( vector<vector<double> >::size_type i = 0; i < matrix.size(); i++ )
    {
       for ( vector<double>::size_type j = 0; j < matrix[i].size(); j++ )
       {
          cout << matrix[i][j] << ' ';
       }
       cout << endl;
    }
}

double norm(const vector<double> &v){
  double norm = 0;
  for(const auto &value: v){
    norm=norm+value*value;
  }
  norm=sqrt(norm);
  return norm;
}

void trans(vector<vector<double>> &A){
  unsigned const n = A.size(); // if we don't already know the size of our vector, we calculate it
  double temp;
  for(unsigned i=0;i<n;++i){
    for(unsigned j=0;j<i;++j){
      temp=A[i][j];
      A[i][j]=A[j][i];
      A[j][i]=temp;
    }
  }
}

vector<vector<double>> mult(const vector<vector<double>> &A, const vector<vector<double>> &B){
  const unsigned columnsA = A[0].size();
  const unsigned rowsA = A.size();
  const unsigned columnsB = B[0].size();
  const unsigned rowsB = B.size();

  // We multiply our matrix and return it as a vector<vector<double>>, or we throw a "Dimension error" exception
  if (columnsA!=rowsB){
    throw "Dimension error";
  } else {
    vector<vector<double>> C(rowsA,vector<double>(columnsB,0));
    for(unsigned i=0;i<rowsA;++i){
      for(unsigned j=0;j<columnsB;++j){
        C[i][j]=0;
        for(unsigned k=0;k<columnsA;++k){
          C[i][j]=C[i][j]+A[i][k]*B[k][j];
        }
      }
    }
    return C;
  }
}

void givens_rotation(vector<vector<double>> &M, const unsigned &n, const unsigned &i, const unsigned &j, const double &c, const double &s){
  for(unsigned k=0;k<n;++k){
    M[k][k]=1;
  }
  M[i][i]=c;
  M[j][j]=c;
  M[i][j]=-s;
  M[j][i]=s;
}

void QR_decomposition_givens(vector<vector<double>> &Q, vector<vector<double>> &R, const unsigned &n){
  for(unsigned j=0;j<n-1;++j){
    for(unsigned i=n-1;i>j;--i){
      unsigned pos1 = i;
      unsigned pos2 = i-1;
      double r = sqrt(R[pos1][j]*R[pos1][j]+R[pos2][j]*R[pos2][j]);
      vector<vector<double>> G(n, vector<double>(n,0));
      givens_rotation(G,n,pos1,pos2,R[pos2][j]/r,R[pos1][j]/r);
      R = mult(G,R);
      trans(G);
      Q = mult(Q,G);
      trans(G);
    }
  }
  if(R[n-1][n-1]<0){
    R[n-1][n-1]=-R[n-1][n-1];
    for(unsigned i=0;i<n;++i){
      Q[i][2]=-Q[i][2];
    }
  }
}

vector<vector<double>> resolve(const vector<vector<double>> &R,const vector<vector<double>> &B){
  const unsigned n = B.size();
  vector<vector<double>> X(n,vector<double>(n,0));
  for(int i=n-1;i>=0;--i){
    for(unsigned j=0;j<n;++j){
      double sum = 0;
      for(int k=n-1;k>=0;--k){
        sum=sum+R[i][k]*X[k][i];
      }
      if ((unsigned)i==n-1){
        if (R[i][i]!=0){
          X[i][j]=B[i][j]/R[i][i];
        } else {
          cerr<<"AX=B doesn't have a unique solution"<<endl;
          cerr<<"There is no solution or an infinity of solutions"<<endl;
          throw "Infinite solutions error";
        }
      }
      if (R[i][i]!=0){
        X[i][j]=(B[i][j]-sum)/R[i][i];
      } else {
        cerr<<"AX=B doesn't have a unique solution"<<endl;
        cerr<<"There is no solution or an infinity of solutions"<<endl;
        throw "Infinite solutions error";
    }
    }
  }
  return X;
}

void initQR(vector<vector<double>> &Q, vector<vector<double>> &R, const vector<vector<double>> &A, const unsigned &n){
  for(unsigned i=0;i<n;++i){
    Q[i][i]=1;
    for(unsigned j=0;j<n;++j){
      R[i][j]=A[i][j];
    }
  }
}

vector<double> columnToVector(const vector<vector<double>> &col){
  const unsigned n = col.size();
  vector<double> res(n,0);
  for(unsigned i=0;i<n;++i){
    res[i]=col[i][0];
  }
  return res;
}

vector<vector<double>> vectorToColumn(const vector<double>& vec){
  const unsigned n = vec.size();
  vector<vector<double>> res(n,vector<double>(1,0));
  for(unsigned i=0;i<n;++i){
    res[i][0]=vec[i];
  }
  return res;
}


int main(int, char**) {
  
  // const string filea = argv[1];
  // const unsigned sizea = strtoul(argv[2], NULL, 0);
  // const string fileb = argv[3];
  // const unsigned sizeb = strtoul(argv[4], NULL, 0);
  
  // CSVReader readera(filea);
  // CSVReader readerb(fileb);

  // vector<vector<double> > A = readera.getData();
  // vector<vector<double> > b = readerb.getData();
  // vector<double> B = columnToVector(b,sizeb);

  vector<vector<double>> A = {
    {1,2},
    {3,-0.2}
  };
  vector<vector<double>> B = {
    {7, 2},
    {0, 0}
  };

  unsigned n = A.size();
  print_matrix("A",A);
  print_matrix("B",B);
  cout<<endl;

  vector<vector<double>> R(n,vector<double>(n,0));
  vector<vector<double>> Q(n,vector<double>(n,0));

  initQR(Q,R,A,n);
  QR_decomposition_givens(Q,R,n);
  trans(Q);

  print_matrix("Q", Q);
  B=mult(Q,B);
  print_matrix("R",R);
  print_matrix("Q*R",mult(Q,R));

  vector<vector<double>> X = resolve(R,B);


  print_matrix("AX =", mult(A,X));

  cout<<endl<<"The answer of the linear system of equation AX=B is"<<endl;
  print_matrix("X = ",X);




//   print_matrix("A",a);
//   print_matrix("B",b);

  // Using MKL to solve the system

  // MKL_INT n = sizea, nrhs = sizeb, lda = sizea, ldb = sizeb, info;
  // MKL_INT *ipiv = (MKL_INT *)malloc(sizeof(MKL_INT));
  // //MKL_INT ipiv[1024];

  // clock_t tStart = clock();
  // info = LAPACKE_dgesv( LAPACK_ROW_MAJOR, n, nrhs, a[0].data(), lda, ipiv,b[0].data(), ldb );    
  // printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);


  //print_matrix("X",b);

  return 0;

}
