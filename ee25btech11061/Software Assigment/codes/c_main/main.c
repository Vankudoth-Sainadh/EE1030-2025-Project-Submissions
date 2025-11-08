#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double norm(double *x, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    return sqrt(s);
}

void matvec(double *A, int m, int n, double *x, double *y) {
    for (int i = 0; i < m; i++) {
        double s = 0;
        for (int j = 0; j < n; j++) s += A[i * n + j] * x[j];
        y[i] = s;
    }
}

void matvecT(double *A, int m, int n, double *x, double *y) {
    for (int j = 0; j < n; j++) {
        double s = 0;
        for (int i = 0; i < m; i++) s += A[i * n + j] * x[i];
        y[j] = s;
    }
}

double top_svd(double *A, int m, int n, double *u, double *v) {
    for (int j = 0; j < n; j++) v[j] = 1.0 / n;
    double *tmp = malloc(sizeof(double) * m);
    for (int it = 0; it < 20; it++) {
        matvec(A, m, n, v, tmp);
        double nu = norm(tmp, m);
        for (int i = 0; i < m; i++) u[i] = tmp[i] / nu;
        matvecT(A, m, n, u, v);
        double nv = norm(v, n);
        for (int j = 0; j < n; j++) v[j] /= nv;
    }
    matvec(A, m, n, v, tmp);
    double s = norm(tmp, m);
    for (int i = 0; i < m; i++) u[i] = tmp[i] / s;
    free(tmp);
    return s;
}

void deflate(double *A, int m, int n, double *u, double *v, double s) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            A[i * n + j] -= s * u[i] * v[j];
}

void reconstruct(double *R, int m, int n, double *U, double *S, double *V, int k) {
    for (int i = 0; i < m * n; i++) R[i] = 0;
    for (int r = 0; r < k; r++)
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                R[i * n + j] += S[r] * U[r * m + i] * V[r * n + j];
}

double frob_error(double *A, double *B, int m, int n) {
    double s = 0;
    for (int i = 0; i < m * n; i++) {
        double d = A[i] - B[i];
        s += d * d;
    }
    return sqrt(s);
}

double frob_norm(double *A, int m, int n) {
    double s = 0;
    for (int i = 0; i < m * n; i++) s += A[i] * A[i];
    return sqrt(s);
}

int main() {
    char inname[128];
    int k;

    printf("Enter image file name : ");
    scanf("%127s", inname);

    printf("Enter k value: ");
    scanf("%d", &k);

    FILE *fp = fopen(inname, "rb");
    if (!fp) {
        printf("cannot open %s\n", inname);
        return 1;
    }

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    unsigned char *buffer = malloc(size);
    fread(buffer, 1, size, fp);
    fclose(fp);

    int w, h, c;
    unsigned char *img = stbi_load_from_memory(buffer, size, &w, &h, &c, 0);
    free(buffer);
    if (!img) {
        printf("decode failed\n");
        return 1;
    }

    int m = h, n = w;
    if (k > m) k = m;
    if (k > n) k = n;

    double *A = malloc(sizeof(double) * m * n);
    if (c == 1) {
        for (int i = 0; i < m * n; i++)
            A[i] = img[i];
    } else {
        for (int i = 0; i < m * n; i++) {
            unsigned char r = img[i * c + 0];
            A[i] = r;
        }
    }

    double *Awork = malloc(sizeof(double) * m * n);
    for (int i = 0; i < m * n; i++) Awork[i] = A[i];

    double *U = malloc(sizeof(double) * k * m);
    double *S = malloc(sizeof(double) * k);
    double *V = malloc(sizeof(double) * k * n);
    double *u = malloc(sizeof(double) * m);
    double *v = malloc(sizeof(double) * n);

    for (int r = 0; r < k; r++) {
        double s = top_svd(Awork, m, n, u, v);
        S[r] = s;
        for (int i = 0; i < m; i++) U[r * m + i] = u[i];
        for (int j = 0; j < n; j++) V[r * n + j] = v[j];
        deflate(Awork, m, n, u, v, s);
    }

    double *R = malloc(sizeof(double) * m * n);
    reconstruct(R, m, n, U, S, V, k);

    double abs_err = frob_error(A, R, m, n);
    double A_norm = frob_norm(A, m, n);
    double rel_err = abs_err / A_norm;
    double perc_err = rel_err * 100.0;

    printf("Absolute Frobenius error: %f\n", abs_err);
    printf("Relative Frobenius error: %f\n", rel_err);
    printf("Percentage Frobenius error: %f%%\n", perc_err);

    unsigned char *out = malloc(m * n);
    for (int i = 0; i < m * n; i++) {
        double val = R[i];
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        out[i] = (unsigned char)(val + 0.5);
    }

    char outname[160];
    sprintf(outname, "output_k%d.jpg", k);
    stbi_write_jpg(outname, n, m, 1, out, 90);
    printf("Wrote %s\n", outname);

    stbi_image_free(img);
    free(A); free(Awork);
    free(U); free(S); free(V);
    free(u); free(v);
    free(R); free(out);

    return 0;
}
