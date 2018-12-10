#include <stdio.h>
#include "utils.c"

Model model;

void init_model(Model* model) {
    // set filter1
    model->filter1.size = 2;
    model->filter1.number_of_filters = 1;

    model->filter1.entries[0][0][0] = 1.0;
    model->filter1.entries[0][1][0] = 1.0;
    model->filter1.entries[1][0][0] = 1.0;
    model->filter1.entries[1][1][0] = 1.0;
    model->stride1 = 1;
    model->stride2 = 2;

    // init weights    
    model->fcl.height = 3;
    model->fcl.width = 3;
    model->fcl.depth = 4;
    
    model->fcl.weights[0][0][0] = 0.4;
    model->fcl.weights[1][0][0] = 0.34;
    model->fcl.weights[2][0][0] = 0.2;
    model->fcl.weights[0][1][0] = 0.4;
    model->fcl.weights[1][1][0] = 1.04;
    model->fcl.weights[2][1][0] = 0.4;
    model->fcl.weights[0][2][0] = 3.14;
    model->fcl.weights[1][2][0] = 0.2;
    model->fcl.weights[2][2][0] = 0.1;
    
    model->fcl.weights[0][0][1] = 0.31;
    model->fcl.weights[1][0][1] = 0.02;
    model->fcl.weights[2][0][1] = 0.1;
    model->fcl.weights[0][1][1] = 0.2;
    model->fcl.weights[1][1][1] = 1.01;
    model->fcl.weights[2][1][1] = 2.01;
    model->fcl.weights[0][2][1] = 0.02;
    model->fcl.weights[1][2][1] = 0.43;
    model->fcl.weights[2][2][1] = 0.05;

    model->fcl.weights[0][0][2] = 0.21;
    model->fcl.weights[1][0][2] = 1.2;
    model->fcl.weights[2][0][2] = 0.1;
    model->fcl.weights[0][1][2] = 0.22;
    model->fcl.weights[1][1][2] = 1.01;
    model->fcl.weights[2][1][2] = 0.01;
    model->fcl.weights[0][2][2] = 0.12;
    model->fcl.weights[1][2][2] = 0.43;
    model->fcl.weights[2][2][2] = 0.35;

    model->fcl.weights[0][0][3] = 0.31;
    model->fcl.weights[1][0][3] = 0.52;
    model->fcl.weights[2][0][3] = 1.1;
    model->fcl.weights[0][1][3] = 0.2;
    model->fcl.weights[1][1][3] = 1.41;
    model->fcl.weights[2][1][3] = 2.01;
    model->fcl.weights[0][2][3] = 0.12;
    model->fcl.weights[1][2][3] = 0.43;
    model->fcl.weights[2][2][3] = 0.05;


    model->fcl.biases[0][0] = 0.05;
    model->fcl.biases[1][0] = 0.0132;
    model->fcl.biases[2][0] = 0.03;
    
    model->fcl.biases[0][1] = 0.15;
    model->fcl.biases[1][1] = 0.45;
    model->fcl.biases[2][1] = 1.05;

    model->fcl.biases[0][2] = 0.25;
    model->fcl.biases[1][2] = 0.01;
    model->fcl.biases[2][2] = 0.31;
    
    model->fcl.biases[0][3] = 2.05;
    model->fcl.biases[1][3] = 0.05;
    model->fcl.biases[2][3] = 1.05;

    model->anchor.size = 3;
    model->anchor.entries[0] = 0.01;
    model->anchor.entries[1] = 0.01;
    model->anchor.entries[2] = 0.03;
    /*
    model->negative.size = 3;
    model->negative.entries[0] = 2;
    model->negative.entries[1] = 2;
    model->negative.entries[2] = 2; */

    model->costs.size = 3;

    // fpd
    model->fpd.height = 3;
    model->fpd.depth = 4;

    // bpd
    model->bpd.height = 3;
    model->bpd.width = 3;
    model->bpd.depth = 4;

    // set input
    model->convBox1.height = 3;
    model->convBox1.width = 3;
    model->convBox1.depth = 3;

    model->convBox1.entries[0][0][0] = 3.01;
    model->convBox1.entries[1][0][0] = 2.0;
    model->convBox1.entries[2][0][0] = 1.0;
    model->convBox1.entries[0][1][0] = 3.0;
    model->convBox1.entries[1][1][0] = 0.44;
    model->convBox1.entries[2][1][0] = 0.33;
    model->convBox1.entries[0][2][0] = 3.1;
    model->convBox1.entries[1][2][0] = 0.01;
    model->convBox1.entries[2][2][0] = 2.0; 

    model->convBox1.entries[0][0][1] = 12.0;
    model->convBox1.entries[1][0][1] = 2.3;
    model->convBox1.entries[2][0][1] = 1.1;
    model->convBox1.entries[0][1][1] = 1.0;
    model->convBox1.entries[1][1][1] = 12.0;
    model->convBox1.entries[2][1][1] = 3.1;
    model->convBox1.entries[0][2][1] = 5.2;
    model->convBox1.entries[1][2][1] = 3.2;
    model->convBox1.entries[2][2][1] = 1.2;

    model->convBox1.entries[0][0][2] = 0.03;
    model->convBox1.entries[1][0][2] = 2.0;
    model->convBox1.entries[2][0][2] = 0.30;
    model->convBox1.entries[0][1][2] = 1.0;
    model->convBox1.entries[1][1][2] = 2.0;
    model->convBox1.entries[2][1][2] = 2.3;
    model->convBox1.entries[0][2][2] = 1.0;
    model->convBox1.entries[1][2][2] = 0.02;
    model->convBox1.entries[2][2][2] = 0.04;
}


int main() {
    init_model(&model);
    printf("\n");
    adam_optimizer(&model);
    return 0;
}
