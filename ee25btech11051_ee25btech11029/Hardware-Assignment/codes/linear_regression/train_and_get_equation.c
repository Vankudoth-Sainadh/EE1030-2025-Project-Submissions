#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void solve_quadratic_ls(const char *filename);
int count_lines(const char *filename);
void read_data(const char *filename, double *T_data, double *V_data, int N);
void invert_3x3(double A[3][3], double A_inv[3][3]);
void print_matrix(const char *title, double A[3][3]);

int main() {
    solve_quadratic_ls("training_data.txt");
    return 0;
}

void solve_quadratic_ls(const char *filename) {
    int N = count_lines(filename);
    if (N <= 0) {
        fprintf(stderr, "Error: No data in file %s\n", filename);
        exit(1);
    }

    double *T_data = malloc(N * sizeof(double));
    double *V_data = malloc(N * sizeof(double));
    if (T_data == NULL || V_data == NULL) {
        fprintf(stderr, "Error: Could not allocate memory.\n");
        exit(1);
    }

    read_data(filename, T_data, V_data, N);

    double A_norm[3][3] = {0};
    double b_norm[3] = {0};

    for (int i = 0; i < N; i++) {
        double v = V_data[i];
        double t = T_data[i];
        double v2 = v * v;
        double v3 = v * v2;
        double v4 = v2 * v2;

        b_norm[0] += t;
        b_norm[1] += v * t;
        b_norm[2] += v2 * t;

        A_norm[0][0] += 1.0;
        A_norm[0][1] += v;
        A_norm[0][2] += v2;
        A_norm[1][1] += v2;
        A_norm[1][2] += v3;
        A_norm[2][2] += v4;
    }

    A_norm[1][0] = A_norm[0][1];
    A_norm[2][0] = A_norm[0][2];
    A_norm[2][1] = A_norm[1][2];

    double A_inv[3][3];
    invert_3x3(A_norm, A_inv);

    double w[3];
    w[0] = A_inv[0][0] * b_norm[0] + A_inv[0][1] * b_norm[1] + A_inv[0][2] * b_norm[2];
    w[1] = A_inv[1][0] * b_norm[0] + A_inv[1][1] * b_norm[1] + A_inv[1][2] * b_norm[2];
    w[2] = A_inv[2][0] * b_norm[0] + A_inv[2][1] * b_norm[1] + A_inv[2][2] * b_norm[2];

    printf("\nFINAL MODEL EQUATION\n");
    printf("T = Temperature, V = Voltage\n\n");
    printf("Equation: T = %.6f + (%.6f * V) + (%.6f * V^2)\n\n", w[0], w[1], w[2]);

    free(T_data);
    free(V_data);
}

void invert_3x3(double A[3][3], double A_inv[3][3]) {
    double det;
    double A_T[3][3]; 

    det = A[0][0] * (A[1][1] * A[2][2] - A[2][1] * A[1][2]) -
          A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
          A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

    if (fabs(det) < 1e-10) {
        fprintf(stderr, "Error: Matrix is singular (determinant is zero).\n");
        exit(1);
    }
    
    double inv_det = 1.0 / det;

    A_T[0][0] = (A[1][1] * A[2][2] - A[2][1] * A[1][2]);
    A_T[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]);
    A_T[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]);
    A_T[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]);
    A_T[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]);
    A_T[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]);
    A_T[2][0] = (A[1][0] * A[2][1] - A[2][0] * A[1][1]);
    A_T[2][1] = (A[2][0] * A[0][1] - A[0][0] * A[2][1]);
    A_T[2][2] = (A[0][0] * A[1][1] - A[1][0] * A[0][1]);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A_inv[i][j] = A_T[i][j] * inv_det;
        }
    }
}

int count_lines(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (f == NULL) {
        perror("Error opening file");
        exit(1);
    }

    int count = 0;
    int first_char = 1;
    char c;

    while ((c = fgetc(f)) != EOF) {
        if (first_char && c == '#') {
            while ((c = fgetc(f)) != EOF && c != '\n');
            first_char = 1;
        } else if (c == '\n') {
            count++;
            first_char = 1;
        } else if (first_char) {
            first_char = 0;
        }
    }

    if (!first_char) {
        count++;
    }

    fclose(f);
    return count;
}

void read_data(const char *filename, double *T_data, double *V_data, int N) {
    FILE *f = fopen(filename, "r");
    if (f == NULL) {
        perror("Error opening file");
        exit(1);
    }

    char buffer[256];
    int i = 0;
    
    while (i < N && fgets(buffer, sizeof(buffer), f)) {
        if (buffer[0] == '#') {
            continue; 
        }
        
        int result = sscanf(buffer, "%lf %lf", &T_data[i], &V_data[i]);
        if (result == 2) {
            i++;
        }
    }
    
    fclose(f);
}

void print_matrix(const char *title, double A[3][3]) {
    printf("%s:\n", title);
    for (int i = 0; i < 3; i++) {
        printf("[ ");
        for (int j = 0; j < 3; j++) {
            printf("%9.3f ", A[i][j]);
        }
        printf("]\n");
    }
    printf("\n");
}
