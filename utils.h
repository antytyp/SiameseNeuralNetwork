#ifndef UTILS_H
#define UTILS_H

// podstawowe stałe
#define MAX_VECTOR_SIZE        243
#define MAX_FILTER_SIZE        5
#define MAX_NUMBER_OF_FILTERS  5
#define MAX_CONVBOX_HEIGHT     125
#define MAX_CONVBOX_WIDTH      125
#define MAX_CONVBOX_DEPTH      27
#define POOL_SIZE              3
#define MAX_FCL_HEIGHT         243
#define MAX_FCL_WIDTH          243
#define MAX_FCL_DEPTH          8

// Adam Optimizer hyperparameters
#define BETA_1                 0.9
#define BETA_2                 0.999
#define EPSILON                1e-8
#define ALPHA                  0.001
#define NUMBER_OF_ITERATIONS   50

// Triplet function hyperparameter
#define TRIPLET_ALPHA          0.3

// Struktury danych

typedef struct {
    int size;
    double entries[MAX_VECTOR_SIZE];
} Vector;

typedef struct {
    int size;
    int number_of_filters;

    double entries[MAX_FILTER_SIZE][MAX_FILTER_SIZE][MAX_NUMBER_OF_FILTERS];
} Filter;

typedef struct {
    int height;
    int width;
    int depth;

    double entries[MAX_CONVBOX_WIDTH][MAX_CONVBOX_HEIGHT][MAX_CONVBOX_DEPTH];
} ConvolutionalBox;

typedef struct {
    int height;
    int width;
    int depth;

    double weights[MAX_FCL_HEIGHT][MAX_FCL_WIDTH][MAX_FCL_DEPTH];
    double biases[MAX_FCL_HEIGHT][MAX_FCL_DEPTH];
} FullyConnectedLayer;

typedef struct {
    int height;
    int depth;

    double results[MAX_FCL_DEPTH + 1][MAX_FCL_HEIGHT];
    // act = sigmoid(results)
    double activations[MAX_FCL_DEPTH + 1][MAX_FCL_HEIGHT];
} ForwardPropData;

typedef struct {
    int height;
    int width;
    int depth;

    double d_weights[MAX_FCL_WIDTH][MAX_FCL_HEIGHT][MAX_FCL_DEPTH];
    double d_biases[MAX_FCL_HEIGHT][MAX_FCL_DEPTH];
} BackPropData;

typedef struct {
    ConvolutionalBox convBox1; // input
    Filter filter1;
    int stride1;
    ConvolutionalBox convBox2; // Conv2D output
    int stride2;
    ConvolutionalBox convBox3; // max pool output
    Filter filter2;
    int stride3;
    ConvolutionalBox convBox4; // Conv2D output
    int stride4;
    ConvolutionalBox convBox5; // max pool output
    Vector vector;             // flatten output of Conv layers
    FullyConnectedLayer fcl;   // trzeba wygenerowac wagi poczatkowe
    Vector encoding;           // fcl output, przyjmijmy ze to przyklad pozytywny
    // triplet stuff
    Vector anchor;             // do I need that?
    Vector negative;           // do I need that?
    Vector costs;              // cost function applied on encoding
    ForwardPropData fpd;       // needed by backprop
    BackPropData bpd;          // outputs of backprop, needed by Adam opt
} Model;

typedef struct {
    ConvolutionalBox convBox1; // input: 125x125x3
    Filter filter1;            // 3x3x3
    int stride1;               // str = 1
    ConvolutionalBox convBox2; // Conv2D output,123x123x9
    int stride2;               // max pool, str = 3
    ConvolutionalBox convBox3; // max pool output
    Filter filter2;            // 3x3x3
    int stride3;               // str = 1
    ConvolutionalBox convBox4; // Conv2D output, 39x39x27
    int stride4;               // 3x3x3
    ConvolutionalBox convBox5; // max pool output, 13x13x27
    Filter filter3;            // 5x5x1
    int stride5;               // str = 1
    ConvolutionalBox convBox6; // 9x9x27
    int stride6;               // str 
    ConvolutionalBox convBox7; // 3x3x27, max pool output
    Filter filter4;            // 3x3x1
    int stride7;               // str=3, conv2d output  
    ConvolutionalBox convBox8; // 1x1x27, max pool output    
    int stride8; 
    
    Vector vector;             // flatten output of Conv layers, 27x1
    FullyConnectedLayer fcl;   
    Vector encoding;           // fcl output
    
    Vector costs;              // cost function applied on encoding
    ForwardPropData fpd;       // needed by backprop
    BackPropData bpd;          // outputs of backprop, needed by Adam opt
} SNN;

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
double simple_loss(Vector* anchor, Vector* prediction, Vector* costs);
double triplet_loss(Vector* anchor, Vector* positive, Vector* negative, Vector* costs);
void train(Model* model);
void adam_optimizer(Model* model);
void matrix_times_constant(double** matrix, int m, int n, int constant);
void matrix_squared_elementwise(double** matrix, int m, int n);
void print(ConvolutionalBox* convBox);
void read_csv(ConvolutionalBox* conv_boxes, char* filepath);

#endif // UTILS_H
