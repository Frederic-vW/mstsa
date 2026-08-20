// compile: cc -fPIC -shared -o liblzc.so lzc76.c
int lz76(int *x, int n) {
    /*
    int i;
    int sum = 0;
    for (i = 0; i < num_numbers; i++) {
        sum += numbers[i];
    }
    return sum;
    */
    int c = 1.0;
    int l = 1;
    int i = 0;
    int k = 1;
    int k_max = 1;
    int stop = 0;

    while (stop == 0) {
        if (x[i+k] != x[l+k]) {
            if (k > k_max) {
                k_max = k;
            }
            i += 1;
            if (i == l) {
                c += 1;
                l += k_max;
                if (l+1 > n-1) {
                    stop = 1;
                } else {
                    i = 0;
                    k = 1;
                    k_max = 1;
                }
            } else {
                k = 1;
            }
        } else {
            k += 1;
            if (l+k > n-1) {
                c += 1;
                stop = 1;
            }
        }
    }
    return c;
}
