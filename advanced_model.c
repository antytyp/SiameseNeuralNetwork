#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "utils.c"

Model model;

void init_model(Model* model) {
    // set input
    model->convBox1.height = 100;
    model->convBox1.width = 100;
    model->convBox1.depth = 3;

    int i, j, k;
    // random normalized image
    for (i = 0; i < model->convBox1.height; i++) {
        for (j = 0; j < model->convBox1.width; j++) {
            for (k = 0; k < model->convBox1.depth; k++) {
                model->convBox1.entries[i][j][k] = (rand() % 255 - 128) / 255.0;
            }
        }
    }
    
    // set filter1
    model->filter1.size = 5;
    model->filter1.number_of_filters = 1;
    model->stride1 = 5;
    
    for (i = 0; i < model->filter1.size; i++) {
        for (j = 0; j < model->filter1.size; j++) {
            model->filter1.entries[i][j][0] = rand() % 10;
        }
    }
    
    model->stride2 = 5;

    // init weights, they should be small!    
    model->fcl.height = 48;
    model->fcl.width = 48;
    model->fcl.depth = 6;

    for (k = 0; k < model->fcl.depth; k++) {
        for (i = 0; i < model->fcl.height; i++) {
            for (j = 0; j < model->fcl.width; j++) {
                model->fcl.weights[i][j][k] = (rand() % 100  - 50)/ 1000.0;
            }
            model->fcl.biases[i][k] = rand() % (100 - 50)/ 100.0;
        }
    }
    
    // anchor to learn
    model->anchor.size = 48;
    for (i = 0; i < model->anchor.size; i++) {
        model->anchor.entries[i] = 0.01 * i;
    }
    
    model->costs.size = 48;

    // fpd
    model->fpd.height = 48;
    model->fpd.depth = 6;

    // bpd
    model->bpd.height = 48;
    model->bpd.width = 48;
    model->bpd.depth = 6;
}


int main() {
    srand(time(NULL));
    
    init_model(&model);
    printf("\n");
    adam_optimizer(&model);
    return 0;
}
