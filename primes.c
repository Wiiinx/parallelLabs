#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

typedef char *String;

// According to the PDF I think this is the algorithm
void SieveOfEratosthenes(long n, bool *prime) {
    memset(prime, true, (n + 1) * sizeof(bool));
    for (long p = 2; p * p <= n; p++) {
        if (prime[p]) {
            for (long i = p * p; i <= n; i += p)
                prime[i] = false;
        }
    }
}

int main(int argc, char *argv[]) {
    int my_rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (argc != 2) {
        if (my_rank == 0) {
            printf("Type in this format: mpiexec -n <procs> %s <n>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    long n = atol(argv[1]);
    if (n < 2 || n > 10000000) {
        if (my_rank == 0) {
            printf("<n> must be a positive number between 2 and 10000000\n");
        }
        MPI_Finalize();
        return 1;
    }

    long sqrt_n = (long)sqrt(n) + 1;
    bool *base_primes = (bool *)malloc((sqrt_n + 1) * sizeof(bool));
    SieveOfEratosthenes(sqrt_n, base_primes);

    long local_n = (n - 1) / num_procs + 1;
    long local_start = my_rank * local_n + 2;
    long local_end = (my_rank + 1) * local_n + 1;
    if (local_end > n) local_end = n + 1;

    bool *local_non_primes = (bool *)calloc(local_n, sizeof(bool));
    if (!local_non_primes) {
        fprintf(stderr, "Out of memory\n");
        MPI_Finalize();
        return 1;
    }

    // Wait for all processes to reach this point
    MPI_Barrier(MPI_COMM_WORLD);

    // Start timing here!
    double start = MPI_Wtime();

    for (long i = 2; i <= sqrt_n; i++) {
        if (base_primes[i]) {
            long min_multiple = (local_start + i - 1) / i * i;
            if (min_multiple < i * i) min_multiple = i * i;
            for (long j = min_multiple; j < local_end; j += i) {
                local_non_primes[j - local_start] = true;
            }
        }
    }

    // Wait for all processes to finish computation
    MPI_Barrier(MPI_COMM_WORLD);

    // Stop timing
    double finish = MPI_Wtime();

    bool *global_non_primes = NULL;
    if (my_rank == 0) {
        global_non_primes = (bool *)calloc(n - 1, sizeof(bool));
    }

    MPI_Gather(local_non_primes, local_n, MPI_C_BOOL, global_non_primes, local_n, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // Only rank 0 will output the results and the total time
    if (my_rank == 0) {
        printf("Total time elapsed: %lf seconds\n", finish - start);

        char filename[256];
        sprintf(filename, "%ld.txt", n);
        FILE *file = fopen(filename, "w");
        if (!file) {
            fprintf(stderr, "Error: Cannot open file %s.\n", filename);
            free(global_non_primes);
            MPI_Finalize();
            return 1;
        }

        for (long i = 2; i <= n; i++) {
            if (!global_non_primes[i - 2]) {
                fprintf(file, "%ld ", i);
            }
        }

        fclose(file);
        free(global_non_primes);
    }

    free(local_non_primes);
    free(base_primes);

    MPI_Finalize();
    return 0;
}
