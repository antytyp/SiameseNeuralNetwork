#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "utils.h"

void red() {
  printf("\033[1;31m");
}

void green() {
  printf("\033[0;32m");
}

void reset() {
  printf("\033[0m");
}

void print_result(char title[], int passed) {
    printf("%s: ", title);
    if (passed) {
        green();
        printf("passed\n");
    } else {
        red();
        printf("failed\n");
    }
    reset();
}
// Vector -> Vector, dla kazdego elementu x <- W ReLU(x) = max{0, x}
void relu(int size, double* entries) {
    int i;
    for (i = 0; i < size; i++) {
        if (entries[i] < 0) {
            entries[i] = 0;
        }
    }
}

void relu_derivative(int size, double* entries) {
    int i;
    for (i = 0; i < size; i++) {
        if (entries[i] < 0) {
            entries[i] = 0;
        } else if (entries[i] > 0) {
            entries[i] = 1;
        }
    }
}

// ConvolutionalBox -> ConvolutionalBox, dla kazdego elementu tablicy x <- HxWxD ReLU(x) = max{0, x}
void convbox_relu(ConvolutionalBox* convBox) {
    int i, j, k;
    for (i = 0; i < convBox->height; i++) {
        for (j = 0; j < convBox->width; j++) {
            for (k = 0; k < convBox->depth; k++) {
                if (convBox->entries[i][j][k] < 0) {
                    convBox->entries[i][j][k] = 0;
                }
            }
        }
    }
}

/*
    funkcja sigmoidalna
    Vector -> Vector, dla kazdego elementu x <- W sigmoid(x) = 1/(1+e^(-x))
*/
void sigmoid(int size, double* entries) {
    int i;
    for (i = 0; i < size; i++) {
        entries[i] = 1.0 / (1.0 + exp(entries[i] * (-1.0)));
    }
}

void sigmoid_derivative(int size, double* entries) {
    int i;
    for (i = 0; i < size; i++) {
        double tmp = 1.0 / (1.0 + exp(entries[i] * (-1.0)));
        entries[i] = tmp * (1 - tmp);
    }
}

/* Operacja konwolucji */
// ConvolutionalBox -> ConvolutionalBox
void conv2D(ConvolutionalBox* convBox, ConvolutionalBox* resultConvBox, Filter* filter, int stride) {
    if (stride > filter->size
      || (convBox->width - filter->size) % stride != 0
      || (convBox->height - filter->size) % stride != 0) {
        return;
    }
    resultConvBox->width = (convBox->width - filter->size) / stride + 1;
    resultConvBox->height = (convBox->height - filter->size) / stride + 1;
    resultConvBox->depth = convBox->depth * filter->number_of_filters;

    int w, h, d;
    for (d = 0; d < resultConvBox->depth; d++) {
        int curr_filter = d % filter->number_of_filters;
        int curr_depth = d / filter->number_of_filters;

        for (h = 0; h < resultConvBox->height; h += stride) {
            for (w = 0; w < resultConvBox->width; w += stride) {
                double result = 0.0;

                int i, j;
                for (i = 0; i < filter->size; i++) {
                    for (j = 0; j < filter->size; j++) {
                        result += convBox->entries[h + i][w + j][curr_depth] * filter->entries[i][j][curr_filter];
                    }
                }
                resultConvBox->entries[h][w][d] = result;
            }
        }
    }
}

/* max pooling */
// ConvolutionalBox -> ConvolutionalBox
void max_pooling(ConvolutionalBox* convBox, ConvolutionalBox* resultConvBox, int stride) {
    if (stride > convBox->width || stride > convBox->height
      || (convBox->width - POOL_SIZE) % stride!= 0
      || (convBox->height - POOL_SIZE) % stride != 0) {
        return;
    }
    resultConvBox->width = (convBox->width - POOL_SIZE) / stride + 1;
    resultConvBox->height = (convBox->height - POOL_SIZE) / stride + 1;
    resultConvBox->depth = convBox->depth;


    int w, h, d;
    for (d = 0; d < resultConvBox->depth; d++) {
        for (w = 0; w < resultConvBox->width; w += stride) {
            for (h = 0; h < resultConvBox->height; h += stride) {
                double result = convBox->entries[w][h][d];

                int i, j;
                for (i = 0; i < POOL_SIZE; i++) {
                    for (j = 0; j < POOL_SIZE; j++) {
                        if (result < convBox->entries[w + i][h + j][d]) {
                            result = convBox->entries[w + i][h + j][d];
                        }
                    }
                }
                resultConvBox->entries[w][h][d] = result;
            }
        }
    }
}

// ConvolutionalBox -> Vector
void flatten(ConvolutionalBox* convBox, Vector *vector) {
    vector->size = convBox->width * convBox->height * convBox->depth;
    if (vector->size < MAX_VECTOR_SIZE) {
        int i, j, k;
        int index = 0;
        for (k = 0; k < convBox->depth; k++) {
            for (i = 0; i < convBox->width; i++) {
                for (j = 0; j < convBox->height; j++) {
                    vector->entries[index++] = convBox->entries[i][j][k];
                }
            }
        }
    }
}

double triplet_loss(Vector* anchor, Vector* positive, Vector* negative) {
    // anchor, positive, negative - encodings
    // TRIPLET_ALPHA - hyperparameter, > 0

    double positive_dist = 0.0;
    double negative_dist = 0.0;
    int i;
    for (i = 0; i < anchor->size; i++) {
        positive_dist += (anchor->entries[i] - positive->entries[i]) * (anchor->entries[i] - positive->entries[i]);
        negative_dist += (anchor->entries[i] - negative->entries[i]) * (anchor->entries[i] - negative->entries[i]);
    }
    double loss = positive_dist - negative_dist + TRIPLET_ALPHA;
    if (loss > 0.0) {
        return loss;
    }
    return 0.0;
}

void dense(Vector* vector_in, FullyConnectedLayer* fcl,
           Vector* vector_out, ForwardPropData* fpd) {
    if (vector_in->size != fcl->width) {
        printf("przyps");
        return;
    }
    vector_out->size = vector_in->size;

    int w, h, d, i;

    // potrzebne do backprop
    for (i = 0; i < vector_in->size; i++) {
        fpd->results[0][i] = vector_in->entries[i];
        fpd->activations[0][i] = vector_in->entries[i];
    }
    sigmoid(fpd->height, fpd->activations[0]);

    for (d = 0; d < fcl->depth; d++) {
        for (h = 0; h < fcl->height; h++) {
            fpd->results[d + 1][h] = 0.0;
            for (w = 0; w < fcl->width; w++) {
                fpd->results[d + 1][h] += vector_in->entries[w] * fcl->weights[h][w][d] + fcl->biases[h][d];
            }
        }
        for (i = 0; i < vector_out->size; i++) {
            vector_out->entries[i] = fpd->results[d + 1][i];
        }

        sigmoid(vector_out->size, vector_out->entries);

        for (i = 0; i < vector_in->size; i++) {
            vector_in->entries[i] = vector_out->entries[i];
            fpd->activations[d + 1][i] = vector_out->entries[i];
        }
    }
}

void backpropagation(double m, ForwardPropData* fpd, BackPropData* bpd,
                     FullyConnectedLayer* fcl, Vector* prediction) {
    int l, h, w;

    double d_results[fpd->height];
    double d_activations[fpd->height];

    for (l = fpd->depth - 1; l >= 0; l--) {
        sigmoid_derivative(fpd->height, fpd->results[l + 1]);
        for (h = 0; h < fpd->height; h++) {
            d_results[h] = prediction->entries[h] * fpd->results[l + 1][h];
        }

        for (h = 0; h < fpd->height; h++) {
            for (w = 0; w < fpd->height; w++) {
                bpd->d_weights[h][w][l] = 1 / m * d_results[h] * fpd->activations[l][w];
            }
            bpd->d_biases[h][l] = d_results[h];
        }

        for (h = 0; h < fpd->height; h++) {
            d_activations[h] = 0.0;
            for (w = 0; w < fpd->height; w++) {
                d_activations[h] += fcl->weights[w][h][l] * d_results[w];
            }
            prediction->entries[h] = d_activations[h]; // to zostanie wykorzystane w nastepnej warstwie
        }
    }
}

void matrix_times_constant(double** matrix, int m, int n, int constant) {
    int i, j;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            matrix[i][j] *= constant;
        }
    }
}

void matrix_squared_elementwise(double** matrix, int m, int n) {
    int i, j;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            matrix[i][j] *= matrix[i][j];
        }
    }
}

void get_backprop_data(BackPropData* bpd) {
    return;
}

void adam_optimizer(FullyConnectedLayer* fcl) {
    int t;
    double VdW[fcl->height][fcl->width][fcl->depth];
    double SdW[fcl->height][fcl->width][fcl->depth];
    double Vdb[fcl->width][fcl->depth];
    double Sdb[fcl->width][fcl->depth];

    double VdWcorr[fcl->height][fcl->width][fcl->depth];
    double SdWcorr[fcl->height][fcl->width][fcl->depth];
    double Vdbcorr[fcl->width][fcl->depth];
    double Sdbcorr[fcl->width][fcl->depth];

    int h, w, d;
    for (d = 0; d < fcl->depth; d++) {
        for (w = 0; w < fcl->width; w++) {
            for (h = 0; h < fcl->height; h++) {
                VdW[h][w][d] = 0.0;
                SdW[h][w][d] = 0.0;
            }
            Vdb[w][d] = 0.0;
            Sdb[w][d] = 0.0;
        }
    }

    for (t = 0; t < NUMBER_OF_ITERATIONS; t++) {
        BackPropData bpd;
        get_backprop_data(&bpd);

        for (d = 0; d < fcl->depth; d++) {
            for (w = 0; w < fcl->width; w++) {
                for (h = 0; h < fcl->height; h++) {
                    VdW[h][w][d] = BETA_1 * VdW[h][w][d] + (1 - BETA_1) * bpd.d_weights[h][w][d];
                    SdW[h][w][d] = BETA_2 * SdW[h][w][d] + (1 - BETA_2) * bpd.d_weights[h][w][d] * bpd.d_weights[h][w][d];
                }
                Vdb[w][d] = BETA_1 * Vdb[w][d] + (1 - BETA_1) * bpd.d_biases[w][d];
                Sdb[w][d] = BETA_2 * Sdb[w][d] + (1 - BETA_2) * bpd.d_biases[w][d] * bpd.d_biases[w][d];
            }
        }

        // "correction"
        double beta1_corr = 1.0 / (1 - pow(BETA_1, t));
        double beta2_corr = 1.0 / (1 - pow(BETA_2, t));
        for (d = 0; d < fcl->depth; d++) {
            for (w = 0; w < fcl->width; w++) {
                for (h = 0; h < fcl->height; h++) {
                    VdWcorr[h][w][d] = VdW[h][w][d] * beta1_corr;
                    SdWcorr[h][w][d] = SdW[h][w][d] * beta2_corr;
                }
                Vdbcorr[w][d] = Vdb[w][d] * beta1_corr;
                Sdbcorr[w][d] = Sdb[w][d] * beta2_corr;
            }
        }

        // aktualizacja sieci neuronowej
        for (d = 0; d < fcl->depth; d++) {
            for (w = 0; w < fcl->width; w++) {
                for (h = 0; h < fcl->height; h++) {
                    fcl->weights[h][w][d] = fcl->weights[h][w][d] - ALPHA * VdWcorr[h][w][d] / (sqrt(SdWcorr[h][w][d]) + EPSILON);
                }
                fcl->biases[w][d] = fcl->biases[w][d] - ALPHA * Vdbcorr[w][d] / (sqrt(Sdbcorr[w][d]) + EPSILON);
            }
        }
    }
}

void print(ConvolutionalBox* convBox) {
    int i, j, k;
    for (k = 0; k < convBox->depth; k++) {
        for (i = 0; i < convBox->width; i++) {
            for (j = 0; j < convBox->height; j++) {
                printf("%f ", convBox->entries[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
