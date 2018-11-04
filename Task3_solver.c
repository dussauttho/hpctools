#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
//#include "mkl_lapacke.h"

//Function to get a column of a matrix
//Takes a 2d array (matrix), the dimension (size), and the number of the column to get (col)
//Return an array which contain matrix[:,col]

// double *get_column_matrix(double **matrix, int size, int col){
double* get_column_matrix(double* v, double **matrix, int size, int col){
    // double *vector = malloc( size * sizeof(double));
    int i;
    #pragma omp parallel private(i) shared(size, v, matrix,col)
    {
        #pragma omp for
        for (i = 0; i < size; ++i)
        {
            v[i] = matrix[i][col];
        }
    }
    return v;
}

//Funchion which computes the scalar product of two vectors
//Takes two arrays (v1 and v2) and the size of the arrays (size)
//Returns a double which corresponds to the scalar product of v1 and v2
double dot_product_vectors(double *v1, double *v2, int size){
    double res = 0;
    int i;
    #pragma omp parallel private(i) shared(v1,v2,size) reduction(+:res) 
    {
        #pragma omp for 
        for (i = 0; i < size; ++i)
        {
            res += v1[i]*v2[i];
        }
    }
    return res;
}

double *multiply_vector_scalar(double *v, double scalar, int size){
    int i;
    for (i = 0; i < size; ++i)
    {
        v[i] *= scalar;
    }
    return v;
}

//Function which divide a vector by its norm
//Takes an array v, and the size of the vector (size)
//Returns an anrray which containes each element of v divided by norm(v)
double *norm_vector(double *v, int size){
    double norm = sqrt(dot_product_vectors(v,v,size));
    int i;
    for (i = 0; i < size; ++i)
    {
        v[i]/=norm;
    }
    return v;
}


double** create_matrix(double** new_matrix, double *matrix, int size){
/*    for(int i = 0; i < size; i++)
    {
        new_matrix[i] = malloc( size * sizeof(double));
    }*/
    int i,j;
    #pragma omp parallel private(i,j) shared(size,new_matrix,matrix)
    {
        #pragma omp for
        for (i = 0; i < size; ++i)
        {
            for (j = 0; j < size; ++j)
            {
                new_matrix[i][j] = matrix[i*size+j];
            }
        }
    }
    return new_matrix;
}

double** multiply_2matrix(double **res, double **A, double **B, int size){
    int i,j,k;
    #pragma omp parallel private(i,j,k) shared(size, res, A, B)
    {
        #pragma omp for
        for(i = 0; i < size; ++i){
            for(j = 0; j < size; ++j){
                res[i][j]=0;
                for(k = 0; k < size; ++k){
                    res[i][j] +=  A[i][k] * B[k][j];
                }
            }
        }
    }
    return res;
}

double** transpose_matrix(double** res, double **mat, int size){
    int i,j;
    #pragma omp parallel private(i,j) shared(res,mat,size)
    {
        #pragma omp for
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                res[i][j] = mat[j][i];
            }
        }
    }
    return res;
}


void print_2Dmatrix(const char *name, double **matrix, int size)
{
    int i, j;
    printf("matrix: %c \n", matrix);
    for (i = 0; i < size; i++)
    {
            for (j = 0; j < size; j++)
            {
                printf("%f ", matrix[i][j]);
            }
            printf("\n");
    }
}


double ** QR_GramSchmidt(double** X, double **A, double **B, int size){
    //Initialisation of Q, R, X and QTB = Q^T*B matrix
    double **Q = malloc( size * sizeof(double *));
    double **R = malloc( size * sizeof(double *));
    double **Q_transp = malloc( size * sizeof(double *));
    double **QTB = malloc( size * sizeof(double *));
    int i,j,k;
    #pragma omp parallel private(i) shared(Q,R,size,Q_transp,QTB)
    {
        #pragma omp for 
        for(i = 0; i < size; ++i){
            Q[i] = malloc( size * sizeof(double));
            R[i] = malloc( size * sizeof(double));
            Q_transp[i] = malloc( size * sizeof(double));
            QTB[i] = malloc( size * sizeof(double));
        }
    }
    #pragma omp parallel private(i,j) shared(Q,R,X,size)
    {
        #pragma omp for
        for (i = 0; i < size; ++i)
        {
            for (j = 0; j < size; ++j)
            {
                Q[i][j] = 0;
                R[i][j] = 0;
                X[i][j] = 0;
            }
        }
    }

    //Calculation of Q and R matrix with Gram Schmidt method
    double* v = malloc( size * sizeof(double *));
    v=get_column_matrix(v, A,size,0);
    R[0][0] = sqrt(dot_product_vectors(v,v,size));
    double *temp_q = norm_vector(v,size);
    #pragma omp parallel private(i) shared(Q,temp_q,size)
    {
        #pragma omp for
        for (i = 0; i < size; ++i)
        {
            Q[i][0] = temp_q[i];
        }
    }

    double norm;
    double *q = malloc(sizeof(double)*size);
    double *v_temp;
    for (i = 0; i < size-1; ++i)
    {
        v=get_column_matrix(v, A,size,i+1);
        for (k = 0; k < i+1; ++k)
        {
            v=get_column_matrix(q, Q, size, k);
            R[k][i+1] = dot_product_vectors(v, q, size);
            v_temp = multiply_vector_scalar(q, R[k][i+1], size);
            for (j = 0; j < size; ++j)
            {
                v[j] -= v_temp[j];
            }
        }
        norm = sqrt(dot_product_vectors(v,v,size));
        R[i+1][i+1] = norm;
        temp_q = norm_vector(v,size);
        for (int j = 0; j < size; ++j)
        {
            Q[j][i+1] = temp_q[j];
        }

    }
    free(q);
    //print_2Dmatrix("Q", Q, size);
    //print_2Dmatrix('R', R, size);

    //Solve of AX = B system :
    Q_transp = transpose_matrix(Q_transp, Q, size);
    QTB = multiply_2matrix(QTB, Q_transp, B, size);
    for (int j = size - 1; j >= 0; --j){
        for(int i = size - 1; i >= 0; --i){
            double sum = 0;
            for(int k = size - 1; k >= 0; --k){
                sum += R[i][k] * X[k][j];
            }
            if (R[i][i] != 0){
                X[i][j] = (QTB[i][j] - sum)/R[i][i];
            }
        }
    }
    #pragma omp parallel private(i) shared(size,R,Q,QTB,Q_transp)
    {
        #pragma omp for
        for(i = 0; i<size; ++i){
            free(R[i]);
            free(Q[i]);
            free(QTB[i]);
            free(Q_transp[i]);
        }
    }
    free(Q);
    free(R);
    free(QTB);
    free(Q_transp);
     //print_2Dmatrix('X', X, size);
}

int check_result_matrix(double **bref, double **b, int size) {
    int i;
    for(i=0;i<size;i++) {
        for (int j = 0; j < size; ++j)
        {
            if (bref[i][j]!=b[i][j]) return 0;
        }
    }
    return 1;
}


double *generate_matrix(int size)
{
    int i;
    double *matrix = (double *)malloc(sizeof(double) * size * size);
    srand(1);
    for (i = 0; i < size * size; i++)
    {
        matrix[i] = rand() % 100;
    }

    return matrix;
}

void print_matrix(const char *name, double *matrix, int size)
{
    int i, j;
    printf("matrix: %s \n", matrix);
    for (i = 0; i < size; i++)
    {
            for (j = 0; j < size; j++)
            {
                printf("%f ", matrix[i * size + j]);
            }
            printf("\n");
    }
}

int check_result(double *bref, double *b, int size) {
    int i;
    for(i=0;i<size*size;i++) {
        if (bref[i]!=b[i]) return 0;
    }
    return 1;
}

//int my_dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb) {

    //Replace with your implementation
    //LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, a, lda, ipiv, b, ldb);

//}


    void main(int argc, char *argv[])
    {

        int size = atoi(argv[1]);

        double *a;
        double *b;

        int i;

        a = generate_matrix(size);
        b = generate_matrix(size);
        double **a_new =  malloc( size * sizeof(double *));
        double **b_new =  malloc( size * sizeof(double *));
        double **X =  malloc( size * sizeof(double *));
        #pragma omp parallel private(i) shared(size,a_new,b_new,X)
        {
            #pragma omp for
            for(i = 0; i < size; i++)
            {
                a_new[i] = malloc( size * sizeof(double));
                b_new[i] = malloc( size * sizeof(double));
                X[i] = malloc( size * sizeof(double));
            }
        }
        a_new=create_matrix(a_new, a, size);
        b_new=create_matrix(b_new, b, size);
        free(a);
        free(b);
        clock_t tStart = clock();
        QR_GramSchmidt(X, a_new, b_new, size);
        #pragma omp parallel private(i) shared(size,a_new,b_new,X)
        {
            #pragma omp for
            for(i = 0; i < size; i++)
            {
               free(a_new[i]);
               free(b_new[i]);
               free(X[i]);
            }
        }
        free(a_new);
        free(b_new);
        free(X);

        //print_matrix("A", a, size);
        //print_matrix("B", b, size);

        // Using MKL to solve the system
        //MKL_INT n = size, nrhs = size, lda = size, ldb = size, info;
        //MKL_INT *ipiv = (MKL_INT *)malloc(sizeof(MKL_INT)*size);


        //info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, aref, lda, ipiv, bref, ldb);
        //printf("Time taken by MKL: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);


        //MKL_INT *ipiv2 = (MKL_INT *)malloc(sizeof(MKL_INT)*size);
        //my_dgesv(n, nrhs, a, lda, ipiv2, b, ldb);
        printf("Time taken by my implementation: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
        //bref = create_matrix(bref, size);
        /*if (check_result_matrix(bref,X,size)==1)
            printf("Result is ok!\n");
        else
            printf("Result is wrong!\n");
        */
        //print_matrix("X", b, size);
        //print_matrix("Xref", bref, size);
    }
