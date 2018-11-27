#ifndef UTILS_H
#define UTILS_H

// podstawowe stałe
#define MAX_VECTOR_SIZE        20
#define MAX_FILTER_SIZE        10
#define MAX_NUMBER_OF_FILTERS  10
#define MAX_CONVBOX_HEIGHT     40
#define MAX_CONVBOX_WIDTH      40
#define MAX_CONVBOX_DEPTH      40
#define POOL_SIZE              2
#define MAX_FCL_HEIGHT         40
#define MAX_FCL_WIDTH          40
#define MAX_FCL_DEPTH          40


// Struktury danych

typedef struct Vector {
    int size;
    double entries[MAX_VECTOR_SIZE];
} Vector;

typedef struct Filter {
    int size;
    int number_of_filters;

    double entries[MAX_FILTER_SIZE][MAX_FILTER_SIZE][MAX_NUMBER_OF_FILTERS];
} Filter;

typedef struct ConvolutionalBox {
    int height;
    int width;
    int depth;

    double entries[MAX_CONVBOX_WIDTH][MAX_CONVBOX_HEIGHT][MAX_CONVBOX_DEPTH];
} ConvolutionalBox;

typedef struct FullyConnectedLayer {
    int width;
    int height;
    int depth;

    double weights[MAX_FCL_HEIGHT][MAX_FCL_WIDTH][MAX_FCL_DEPTH];
    double biases[MAX_FCL_HEIGHT][MAX_FCL_DEPTH];
} FullyConnectedLayer;

typedef struct ForwardPropData {
    int height;
    int depth;

    // tutaj najpierw podaje "glebokosc"
    double results[MAX_FCL_DEPTH + 1][MAX_FCL_HEIGHT];
    // act = sigmoid(results)
    double activations[MAX_FCL_DEPTH + 1][MAX_FCL_HEIGHT];
} ForwardPropData;

typedef struct BackPropData {
    int width;
    int height;
    int depth;

    double d_weights[MAX_FCL_WIDTH][MAX_FCL_HEIGHT][MAX_FCL_DEPTH];
    double d_biases[MAX_FCL_HEIGHT][MAX_FCL_DEPTH];
} BackPropData;

void red();
void green();
void reset();
void print_result(char title[], int passed);
void relu(int size, double* entries);
void relu_derivative(int size, double* entries);
void convbox_relu(ConvolutionalBox* convBox);
void sigmoid(int size, double* entries);
void sigmoid_derivative(int size, double* entries);
void conv2D(ConvolutionalBox* convBox, ConvolutionalBox* resultConvBox, Filter* filter, int stride);
void max_pooling(ConvolutionalBox* convBox, ConvolutionalBox* resultConvBox, int stride);
void flatten(ConvolutionalBox* convBox, Vector *vector);
void dense(Vector* vector_in, FullyConnectedLayer* fcl, Vector* vector_out, ForwardPropData* fpd);
void backpropagation(double m, ForwardPropData* fpd, BackPropData* bpd, FullyConnectedLayer* fcl, Vector* prediction);
double triplet_loss(Vector* anchor, Vector* positive, Vector* negative, double alpha);
void print(ConvolutionalBox* convBox);

#endif // UTILS_H
