/* compilation:
gcc -shared c_entropy.c -o libcentropy.so -lm -fPIC
*/

#include <stdlib.h>
#include <math.h>
//#include <stdio.h> // printf

double c_entropy(int *x, size_t n, size_t m, size_t k)
{
    size_t i, j, l;
    int n_hist = (int)(pow(m,k));
    double h;
    double hist[n_hist];
    
    // powers-of-m to convert k-dim to 1-dim array indices
    size_t ns[k-1]; // [1, m, m^2, ..., m^(k-1)]
    for (i=0; i<k; i++) {
        if (i==0) {
            ns[i] = 1;
        } else {
            ns[i] = (size_t)(pow(m,i));
        }
    }

    // initialize empty histogram
    for (i=0; i<n_hist; i++) {
        hist[i] = 0.0;
    }

    // compute histogram
    for (i=0; i<(n-k); i++) {
        // compute 1-D index of k-dim indices defined by x[i,i+1,...,i+k-1]
        l=0; // 1-D histogram index
        for (j=0; j<k; j++) {
            l += ( x[i+j] * ns[j] );
        }
        hist[l]++;
    }

    // normalize histogram
    for (i=0; i<n_hist; i++) {
        hist[i] /= (double)(n-k);
    }

    // compute entropy ! uses NATURAL logarithm, returns nats !
    h = 0.0;
    for (i=0; i<n_hist; i++) {
        if (hist[i] > 0) {
            h -= (hist[i]*log(hist[i]));
        }
    }

    return h;
}
