#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "utils.h"

int global = 0;

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

double simple_loss(Vector* anchor, Vector* prediction, Vector* costs) {
    int i;
    double loss = 0.0;
    for (i = 0; i < anchor->size; i++) {
        costs->entries[i] = prediction->entries[i] - anchor->entries[i];
        loss += costs->entries[i] * costs->entries[i];
    }
    global += 1;
    if(global%100 == 1) {
	printf("Current loss: %f\n", loss);
	/*for (i = 0; i < costs->size; i++) {
 	    printf("%f, ", costs->entries[i]);
	}
	printf("\n");*/
    }
    return loss;
}

// TODO
double triplet_loss(Vector* anchor, Vector* positive, Vector* negative, Vector* costs) {
    // anchor, positive, negative - encodings
    // TRIPLET_ALPHA - hyperparameter, > 0

    double positive_dist = 0.0;
    double negative_dist = 0.0;
    int i;
    double loss = 0.0;
    for (i = 0; i < anchor->size; i++) {
        positive_dist += (anchor->entries[i] - positive->entries[i]) * (anchor->entries[i] - positive->entries[i]);
        negative_dist += (anchor->entries[i] - negative->entries[i]) * (anchor->entries[i] - negative->entries[i]);
        loss += positive_dist;
        loss -= negative_dist;
        costs->entries[i] = positive_dist - negative_dist + TRIPLET_ALPHA;
        loss += costs->entries[i] * costs->entries[i];
        if (costs->entries[i] < 0) costs->entries[i] = 0.0;
    }
    loss += TRIPLET_ALPHA;
    if (loss > 0) {
       loss = 0.0;
    }
    // printf("Current loss: %f\n", loss);
    return loss;
}

void dense(Vector* vector_in, FullyConnectedLayer* fcl, Vector* vector_out, ForwardPropData* fpd) {
    if (vector_in->size != fcl->width) {
	printf("%d, %d", vector_in->size, fcl->width);        
	printf("CONV and FCL cannot be connected");
        return;
    }
    vector_out->size = vector_in->size;

    int w, h, d, i;

    // needed by backprop
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
	if (global == 8000 && d == fcl->depth - 1) {
	    int k;
	    for (k = 0; k < vector_out->size; k++)	    
		printf("%f, ", vector_out->entries[k]);
	    printf("\n");
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
            prediction->entries[h] = d_activations[h]; // this is going to be used in next layer
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

void train(Model* model) {
    conv2D(&model->convBox1, &model->convBox2, &model->filter1, model->stride1);
    max_pooling(&model->convBox2, &model->convBox3, model->stride2);
    flatten(&model->convBox3, &model->vector);
    dense(&model->vector, &model->fcl, &model->encoding, &model->fpd);
    simple_loss(&model->anchor, &model->encoding, &model->costs);
    int m = 1; // 1 training sample
    backpropagation(m, &model->fpd, &model->bpd, &model->fcl, &model->costs);
}

void adam_optimizer(Model* model) {
    int t;
   
    double VdW[model->fcl.height][model->fcl.width][model->fcl.depth];
    double SdW[model->fcl.height][model->fcl.width][model->fcl.depth];
    double Vdb[model->fcl.width][model->fcl.depth];
    double Sdb[model->fcl.width][model->fcl.depth];

    double VdWcorr[model->fcl.height][model->fcl.width][model->fcl.depth];
    double SdWcorr[model->fcl.height][model->fcl.width][model->fcl.depth];
    double Vdbcorr[model->fcl.width][model->fcl.depth];
    double Sdbcorr[model->fcl.width][model->fcl.depth];

    int h, w, d;
    for (d = 0; d < model->fcl.depth; d++) {
        for (w = 0; w < model->fcl.width; w++) {
            for (h = 0; h < model->fcl.height; h++) {
                VdW[h][w][d] = 0.0;
                SdW[h][w][d] = 0.0;
            }
            Vdb[w][d] = 0.0;
            Sdb[w][d] = 0.0;
        }
    }

    for (t = 1; t < NUMBER_OF_ITERATIONS; t++) {
        train(model);
    	/*if(t%100 == 1) {
      	   printf("[%f, %f, %f], \n", model->encoding.entries[0], model->encoding.entries[1], model->encoding.entries[2]);
    	}*/

        for (d = 0; d < model->fcl.depth; d++) {
            for (w = 0; w < model->fcl.width; w++) {
                for (h = 0; h < model->fcl.height; h++) {
                    VdW[h][w][d] = BETA_1 * VdW[h][w][d] + (1 - BETA_1) * model->bpd.d_weights[h][w][d];
                    SdW[h][w][d] = BETA_2 * SdW[h][w][d] + (1 - BETA_2) * model->bpd.d_weights[h][w][d] * model->bpd.d_weights[h][w][d];               
                }
                Vdb[w][d] = BETA_1 * Vdb[w][d] + (1 - BETA_1) * model->bpd.d_biases[w][d];
                Sdb[w][d] = BETA_2 * Sdb[w][d] + (1 - BETA_2) * model->bpd.d_biases[w][d] * model->bpd.d_biases[w][d];
            }
        }

        // "correction"
        double beta1_corr = 1.0 / (1 - pow(BETA_1, t));
        double beta2_corr = 1.0 / (1 - pow(BETA_2, t));
        for (d = 0; d < model->fcl.depth; d++) {
            for (w = 0; w < model->fcl.width; w++) {
                for (h = 0; h < model->fcl.height; h++) {
                    VdWcorr[h][w][d] = VdW[h][w][d] * beta1_corr;
                    SdWcorr[h][w][d] = SdW[h][w][d] * beta2_corr;
                }
                Vdbcorr[w][d] = Vdb[w][d] * beta1_corr;
                Sdbcorr[w][d] = Sdb[w][d] * beta2_corr;
            }
        }

        // neural net update
        for (d = 0; d < model->fcl.depth; d++) {
            for (w = 0; w < model->fcl.width; w++) {
                for (h = 0; h < model->fcl.height; h++) {
                    model->fcl.weights[h][w][d] = model->fcl.weights[h][w][d] - ALPHA * (VdWcorr[h][w][d] / (sqrt(SdWcorr[h][w][d]) + EPSILON));
                }
                model->fcl.biases[w][d] = model->fcl.biases[w][d] - ALPHA * (Vdbcorr[w][d] / (sqrt(Sdbcorr[w][d]) + EPSILON));
            }
        }
    }
    printf("\nEncoding\n");
    int i;
    for (i = 0; i < model->encoding.size; i++) {
	printf("%f, ", model->encoding.entries[i]);
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

// reading images stored in csv file
void read_csv(ConvolutionalBox* conv_boxes, int num_of_images, char* filepath) {
    FILE* file;
    file = fopen(filepath, "r");
    if (file == NULL) printf("Failed to open this file");
    
    int i, j, k, l, pixel;
    size_t len = 0;
    char* buff = NULL;
    for (i = 0; i < num_of_images; i++) {
        for (j = 0; j < conv_boxes->height; j++) {
            for (k = 0; k < conv_boxes->width; k++) {
                for (l = 0; l < conv_boxes->depth; l++) {
                    if (getline(&buff, &len, file) != -1) {
	                // parsing
		        int p = 0;
		        while (buff[p] != ',') {
		            p += 1;
		        }
		        pixel = atoi(buff + (p+1));
                    }
                    // normalization
              	    conv_boxes->entries[j][k][l] = (pixel - 128) / 256.0;
	    	}
            }
        }
    }
    
    fclose(file);
}