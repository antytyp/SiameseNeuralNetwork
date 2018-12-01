#include <stdio.h>
#include "utils.c"

void test_relu() {
    Vector v1;
    v1.size = 5;
    v1.entries[0] = 2.0;
    v1.entries[1] = -1.0;
    v1.entries[2] = 0.0;
    v1.entries[3] = 12.333;
    v1.entries[4] = -5.333;

    relu(v1.size, v1.entries);

    Vector expected;
    expected.size = 5;
    expected.entries[0] = 2.0;
    expected.entries[1] = 0.0;
    expected.entries[2] = 0.0;
    expected.entries[3] = 12.333;
    expected.entries[4] = 0.0;

    int i, passed = 1;
    for(i = 0; i < 5; i++) {
        if (v1.entries[i] != expected.entries[i]) {
            passed = 0;
        }
    }
    print_result("Relu test", passed);
}

void test_convbox_relu() {
    ConvolutionalBox convBox;
    convBox.width = 2;
    convBox.height = 2;
    convBox.depth = 2;
    convBox.entries[0][0][0] = 1.0;
    convBox.entries[1][0][0] = 2.0;
    convBox.entries[0][1][0] = -1.0;
    convBox.entries[1][1][0] = 0.0;
    convBox.entries[0][0][1] = -11.0;
    convBox.entries[1][0][1] = -0.03;
    convBox.entries[0][1][1] = 1.0;
    convBox.entries[1][1][1] = 12.0;

    ConvolutionalBox expectedConvBox;
    expectedConvBox.width = 2;
    expectedConvBox.height = 2;
    expectedConvBox.depth = 2;
    expectedConvBox.entries[0][0][0] = 1.0;
    expectedConvBox.entries[1][0][0] = 2.0;
    expectedConvBox.entries[0][1][0] = 0.0;
    expectedConvBox.entries[1][1][0] = 0.0;
    expectedConvBox.entries[0][0][1] = 0.0;
    expectedConvBox.entries[1][0][1] = 0.0;
    expectedConvBox.entries[0][1][1] = 1.0;
    expectedConvBox.entries[1][1][1] = 12.0;

    convbox_relu(&convBox);

    int i, j, k, passed = 1;
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            for (k = 0; k < 2; k++) {
                if (convBox.entries[i][j][k] != expectedConvBox.entries[i][j][k]) {
                    passed = 0;
                }
            }
        }
    }
    print_result("Convbox relu test", passed);
}

void test_conv2D() {
    ConvolutionalBox convBox;
    convBox.width = 3;
    convBox.height = 3;
    convBox.depth = 2;
    convBox.entries[0][0][0] = 1;
    convBox.entries[0][1][0] = 1;
    convBox.entries[0][2][0] = 1;
    convBox.entries[1][0][0] = 0;
    convBox.entries[1][1][0] = 1;
    convBox.entries[1][2][0] = 1;
    convBox.entries[2][0][0] = 1;
    convBox.entries[2][1][0] = 1;
    convBox.entries[2][2][0] = 0;
    convBox.entries[0][0][1] = 1;
    convBox.entries[0][1][1] = 0;
    convBox.entries[0][2][1] = -1;
    convBox.entries[1][0][1] = 0;
    convBox.entries[1][1][1] = 0;
    convBox.entries[1][2][1] = 0;
    convBox.entries[2][0][1] = -1;
    convBox.entries[2][1][1] = 0;
    convBox.entries[2][2][1] = 1;

    Filter filter;
    filter.size = 2;
    filter.number_of_filters = 2;

    filter.entries[0][0][0] = 1;
    filter.entries[0][1][0] = 3;
    filter.entries[1][0][0] = 2;
    filter.entries[1][1][0] = 4;
    filter.entries[0][0][1] = 1;
    filter.entries[0][1][1] = 0;
    filter.entries[1][0][1] = 0;
    filter.entries[1][1][1] = 1;

    ConvolutionalBox expected;
    expected.width = 2;
    expected.height = 2;
    expected.depth = 4;
    expected.entries[0][0][0] = 8;
    expected.entries[0][1][0] = 10;
    expected.entries[1][0][0] = 9;
    expected.entries[1][1][0] = 6;
    expected.entries[0][0][1] = 2;
    expected.entries[0][1][1] = 2;
    expected.entries[1][0][1] = 1;
    expected.entries[1][1][1] = 1;
    expected.entries[0][0][2] = 1;
    expected.entries[0][1][2] = -3;
    expected.entries[1][0][2] = -2;
    expected.entries[1][1][2] = 4;
    expected.entries[0][0][3] = 1;
    expected.entries[0][1][3] = 0;
    expected.entries[1][0][3] = 0;
    expected.entries[1][1][3] = 1;

    ConvolutionalBox resultConvBox;
    conv2D(&convBox, &resultConvBox, &filter, 1);

    /* expected = {{8,20},{9,6}}, {{2,2},{1,1}}, {{1,-3},{-2,4}}, {{1,0},{0,1}}*/
    int i, j, k, passed = 1;
    for (k = 0; k < resultConvBox.depth; k++) {
        for (i = 0; i < resultConvBox.width; i++) {
            for (j = 0; j < resultConvBox.height; j++) {
                if (resultConvBox.entries[i][j][k] != expected.entries[i][j][k]) {
                    passed = 0;
                }
            }
        }
    }
    print_result("Conv2D test", passed);
}

void test_max_pooling() {
    ConvolutionalBox convBox;
    convBox.width = 3;
    convBox.height = 3;
    convBox.depth = 2;
    convBox.entries[0][0][0] = 5;
    convBox.entries[0][1][0] = 0;
    convBox.entries[0][2][0] = 6;
    convBox.entries[1][0][0] = 1;
    convBox.entries[1][1][0] = 1;
    convBox.entries[1][2][0] = 2;
    convBox.entries[2][0][0] = 7;
    convBox.entries[2][1][0] = 1;
    convBox.entries[2][2][0] = 8;
    convBox.entries[0][0][1] = 1;
    convBox.entries[0][1][1] = 0;
    convBox.entries[0][2][1] = -1;
    convBox.entries[1][0][1] = 0;
    convBox.entries[1][1][1] = 0;
    convBox.entries[1][2][1] = 0;
    convBox.entries[2][0][1] = -1;
    convBox.entries[2][1][1] = 0;
    convBox.entries[2][2][1] = 1;

    ConvolutionalBox expected;
    expected.width = 2;
    expected.height = 2;
    expected.depth = 2;
    expected.entries[0][0][0] = 5;
    expected.entries[0][1][0] = 6;
    expected.entries[1][0][0] = 7;
    expected.entries[1][1][0] = 8;
    expected.entries[0][0][1] = 1;
    expected.entries[0][1][1] = 0;
    expected.entries[1][0][1] = 0;
    expected.entries[1][1][1] = 1;

    ConvolutionalBox resultConvBox;
    max_pooling(&convBox, &resultConvBox, 1);

    /* expected = {{5,6},{7,8}}, {{1,0},{0,1}}*/
    int i, j, k, passed = 1;
    for (k = 0; k < resultConvBox.depth; k++) {
        for (i = 0; i < resultConvBox.width; i++) {
            for (j = 0; j < resultConvBox.height; j++) {
                if (resultConvBox.entries[i][j][k] != expected.entries[i][j][k]) {
                    passed = 0;
                }
            }
        }
    }
    print_result("Max pooling test", passed);
}

void test_sigmoid() {
    Vector v;
    v.size = 5;
    v.entries[0] = 2.0;
    v.entries[1] = -1.0;
    v.entries[2] = 0.0;
    v.entries[3] = 0.5;
    v.entries[4] = -0.333;

    Vector expected;
    expected.size = 5;
    expected.entries[0] = 0.880797;
    expected.entries[1] = 0.268941;
    expected.entries[2] = 0.500000;
    expected.entries[3] = 0.622459;
    expected.entries[4] = 0.417511;

    sigmoid(v.size, v.entries);

    int i, passed = 1;
    for(i = 0; i < 5; i++) {
        if ((v.entries[i] - expected.entries[i]) > 0.000001) {
            passed = 0;
        }
    }
    print_result("Sigmoid test", passed);
}

void test_flatten() {
    ConvolutionalBox convBox;
    convBox.width = 3;
    convBox.height = 3;
    convBox.depth = 2;
    convBox.entries[0][0][0] = 5;
    convBox.entries[0][1][0] = 0;
    convBox.entries[0][2][0] = 6;
    convBox.entries[1][0][0] = 1;
    convBox.entries[1][1][0] = 1;
    convBox.entries[1][2][0] = 2;
    convBox.entries[2][0][0] = 7;
    convBox.entries[2][1][0] = 1;
    convBox.entries[2][2][0] = 8;
    convBox.entries[0][0][1] = 1;
    convBox.entries[0][1][1] = 0;
    convBox.entries[0][2][1] = -1;
    convBox.entries[1][0][1] = 0;
    convBox.entries[1][1][1] = 0;
    convBox.entries[1][2][1] = 0;
    convBox.entries[2][0][1] = -1;
    convBox.entries[2][1][1] = 0;
    convBox.entries[2][2][1] = 1;

    Vector result;
    flatten(&convBox, &result);

    /* expected = {5,0,6,1,1,2,7,1,8,1,0,-1,0,0,0,-1,0,1} */
    Vector expected;
    expected.size = 18;
    expected.entries[0] = 5;
    expected.entries[1] = 0;
    expected.entries[2] = 6;
    expected.entries[3] = 1;
    expected.entries[4] = 1;
    expected.entries[5] = 2;
    expected.entries[6] = 7;
    expected.entries[7] = 1;
    expected.entries[8] = 8;
    expected.entries[9] = 1;
    expected.entries[10] = 0;
    expected.entries[11] = -1;
    expected.entries[12] = 0;
    expected.entries[13] = 0;
    expected.entries[14] = 0;
    expected.entries[15] = -1;
    expected.entries[16] = 0;
    expected.entries[17] = 1;

    int i, passed = 1;
    for(i = 0; i < result.size; i++) {
        if (result.entries[i] != expected.entries[i]) {
            passed = 0;
        }
    }
    print_result("Flatten test", passed);
}

void test_dense() {
    Vector vector_in;
    vector_in.size = 2;
    vector_in.entries[0] = 1;
    vector_in.entries[1] = 1;

    FullyConnectedLayer fcl;
    fcl.width = 2;
    fcl.height = 2;
    fcl.depth = 2;

    fcl.weights[0][0][0] = 1.0;
    fcl.weights[0][1][0] = 0.0;
    fcl.weights[1][0][0] = 1.0;
    fcl.weights[1][1][0] = 1.0;
    fcl.weights[0][0][1] = 1.0;
    fcl.weights[0][1][1] = 2.0;
    fcl.weights[1][0][1] = 3.0;
    fcl.weights[1][1][1] = 0.0;

    fcl.biases[0][0] = 0.0;
    fcl.biases[1][0] = 0.0;
    fcl.biases[0][1] = 0.0;
    fcl.biases[1][1] = 0.0;

    Vector vector_out;
    ForwardPropData fpd; fpd.height = 2; fpd.depth = 3;

    dense(&vector_in, &fcl, &vector_out, &fpd);

    double expected[2] = {0.923625, 0.899635};

    int i, passed = 1;
    for(i = 0; i < vector_out.size; i++) {
        if ((vector_out.entries[i] - expected[i]) > 0.000001) {
            passed = 0;
        }
    }
    print_result("Dense test", passed);
}

void test_backprop() {
    int m = 1;
    ForwardPropData fpd;
    fpd.height = 2;
    fpd.depth = 2;
    fpd.results[0][0] = 1;
    fpd.results[0][1] = 1;
    fpd.results[1][0] = 1;
    fpd.results[1][1] = 2;
    fpd.results[2][0] = 2.491;
    fpd.results[2][1] = 2.193;

    fpd.activations[0][0] = 0.731059;
    fpd.activations[0][1] = 0.731059;
    fpd.activations[1][0] = 0.731059;
    fpd.activations[1][1] = 0.880797;
    fpd.activations[2][0] = 0.923625;
    fpd.activations[2][1] = 0.899635;

    BackPropData bpd; bpd.width = bpd.height = 2; bpd.depth = 2;
    FullyConnectedLayer fcl;
    fcl.width = 2;
    fcl.height = 2;
    fcl.depth = 2;

    fcl.weights[0][0][0] = 1.0;
    fcl.weights[0][1][0] = 0.0;
    fcl.weights[1][0][0] = 1.0;
    fcl.weights[1][1][0] = 1.0;
    fcl.weights[0][0][1] = 1.0;
    fcl.weights[0][1][1] = 2.0;
    fcl.weights[1][0][1] = 3.0;
    fcl.weights[1][1][1] = 0.0;

    fcl.biases[0][0] = 0.0;
    fcl.biases[1][0] = 0.0;
    fcl.biases[0][1] = 0.0;
    fcl.biases[1][1] = 0.0;

    Vector prediction;
    prediction.size = 2;
    prediction.entries[0] = 1;
    prediction.entries[1] = 1;

    backpropagation(m, &fpd, &bpd, &fcl, &prediction);

    // expected W2 = {0.051642 0.062220 0.066018 0.079540}
    // expected W1 = {0.049093 0.049093 0.010844 0.010844}
    BackPropData expected_bpd;
    expected_bpd.width = expected_bpd.height = expected_bpd.depth = 2;
    expected_bpd.d_weights[0][0][0] = 0.049093;
    expected_bpd.d_weights[0][1][0] = 0.049093;
    expected_bpd.d_weights[1][0][0] = 0.010844;
    expected_bpd.d_weights[1][1][0] = 0.010844;
    expected_bpd.d_weights[0][0][1] = 0.051642;
    expected_bpd.d_weights[0][1][1] = 0.062220;
    expected_bpd.d_weights[1][0][1] = 0.066018;
    expected_bpd.d_weights[1][1][1] = 0.079540;

    int i, j, k, passed = 1;
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            for (k = 0; k < 2; k++) {
                if ((expected_bpd.d_weights[j][k][i] - bpd.d_weights[j][k][i]) > 0.000001) {
                    passed = 0;
                }
            }
        }
    }
    print_result("Backprop test", passed);
}

int main() {

    test_relu();
    test_convbox_relu();
    test_conv2D();
    test_max_pooling();
    test_sigmoid();
    test_flatten();
    test_dense();
    test_backprop();

    return 0;
}
