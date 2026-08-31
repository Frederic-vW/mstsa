#include <stdlib.h>
#include <math.h>
//#include <stdio.h> // printf

/* compilation:
gcc -shared se.c -o libse.so -lm -fPIC
*/

double sample_entropy_cont(double* x, size_t n, size_t m, size_t tau, double r) {
    size_t i, j, k;
    double A, B, d, se;
    //printf("\nn, m, tau, r: %zu, %zu, %zu, %.3f", n, m, tau, r);
    B = 0.0;
    A = 0.0;
    size_t tmax = n - m*tau;
    for (i=0; i<tmax; i++) {
        for (j=i+1; j<tmax; j++) {
            //printf("\ni,j: %zu, %zu", i, j);
            // distance norm computation: max_k |x_k - y_k|, start at k=0
            k = 0;
            d = fabs( x[i+k*tau] - x[j+k*tau] );
            while ((d < r) && (k < m)) {
                k++;
                d = fabs(x[i+k*tau]-x[j+k*tau]);
            }
            if (k == m) {
                B++; // |x_k - y_k| < r for all k=0..m-1
                if (fabs(x[i+m*tau]-x[j+m*tau]) < r) {
                    A++; // |x_k - y_k| < r for k=m too
                }
            }
        }
    }
    if (A == 0.0) {
        se = 0.0;
    } else {
        se = -log(A/B);
    }
    //printf("\nA, B: %.2f, %.2f", A, B);
    //printf("\nse: %.2f\n", se);
    return se;
}

double sample_entropy_disc(long long* x, size_t n, size_t m, size_t tau) {
    size_t i, j, k;
    double A, B, se;
    B = 0.0;
    A = 0.0;
    size_t tmax = n - m*tau;
    for (i=0; i<tmax; i++) {
        for (j=i+1; j<tmax; j++) {
            k = 0;
            while ((x[i+k*tau] == x[j+k*tau]) && (k < m)) {
                k++;
            }
            if (k == m) {
                B++;
                if (x[i+m*tau] == x[j+m*tau]) {
                    A++;
                }
            }
        }
    }
    if (A == 0.0) {
        se = 0.0;
    } else {
        se = -log(A/B);
    }
    return se;
}

int sample_entropy_cont_fast(double* se, double* x, size_t n, size_t m, double r) {
    size_t i, j, k;
    size_t nj, jj, m1;
    double x1;
    // n*(n-1)/2 overflows a 32-bit long (Windows) for realistic sequence
    // lengths (e.g. n > ~65536); long long is 64-bit on every platform.
    long long N = (long long) (n*(n-1)/2);
    double* A = (double *) calloc(m, sizeof(double));
    double* B = (double *) calloc(m, sizeof(double));
    double* p = (double *) calloc(m, sizeof(double));
    long* run     = (long *) calloc(n, sizeof(long));
    long* lastrun = (long *) calloc(n, sizeof(long));
    for (i=0; i<n-1; i++) {
        nj = n - i - 1;
        x1 = x[i];
        for (jj=0; jj<nj; jj++) {
            j = i + 1 + jj;
            if (fabs(x[j] - x1) < r) {
                run[jj] = lastrun[jj] + 1;
                m1 = m < (size_t)run[jj] ? m : (size_t)run[jj];
                for (k=0; k<m1; k++) {
                    A[k]++;
                    if (j < n-1) {
                        B[k]++;
                    }
                }
            } else {
                run[jj] = 0;
            }
        }
        for (j=0; j<nj; j++) {
            lastrun[j] = run[j];
        }
    }
    for (k=0; k<m; k++) {
        if (k==0) {
            p[k] = A[k]/N;
        } else {
            p[k] = A[k]/B[k-1];
        }
        if (p[k] == 0) {
            se[k] = 0.0;
        } else {
            se[k] = -log(p[k]);
        }
    }
    free(A);
    free(B);
    free(p);
    free(run);
    free(lastrun);
    return 0;
}

int sample_entropy_disc_fast(double* se, long long* x, size_t n, size_t m) {
    size_t i, j, k;
    size_t nj, jj, m1;
    long long x1;
    // n*(n-1)/2 overflows a 32-bit long (Windows) for realistic sequence
    // lengths (e.g. n > ~65536); long long is 64-bit on every platform.
    long long N = (long long) (n*(n-1)/2);
    double* A = (double *) calloc(m, sizeof(double));
    double* B = (double *) calloc(m, sizeof(double));
    double* p = (double *) calloc(m, sizeof(double));
    long* run     = (long *) calloc(n, sizeof(long));
    long* lastrun = (long *) calloc(n, sizeof(long));
    for (i=0; i<n-1; i++) {
        nj = n - i - 1;
        x1 = x[i];
        for (jj=0; jj<nj; jj++) {
            j = i + 1 + jj;
            if (x[j] == x1) {
                run[jj] = lastrun[jj] + 1;
                m1 = m < (size_t)run[jj] ? m : (size_t)run[jj];
                for (k=0; k<m1; k++) {
                    A[k]++;
                    if (j < n-1) {
                        B[k]++;
                    }
                }
            } else {
                run[jj] = 0;
            }
        } // jj
        for (j=0; j<nj; j++) {
            lastrun[j] = run[j];
        }
    } // i
    
    // compute SE values
    for (k=0; k<m; k++) {
        if (k==0) {
            p[k] = A[k]/N;
        } else {
            p[k] = A[k]/B[k-1];
        }
	    if (p[k] == 0) {
            se[k] = 0.0;
        } else {
            se[k] = -log(p[k]);
        }
    }
    free(A);
    free(B);
    free(p);
    free(run);
    free(lastrun);
    return 0;
}
