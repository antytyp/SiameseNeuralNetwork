#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.c"

typedef struct {
    int anchor;
    int positive;
    int negative;
} Package;

int main() {
    char pawel_paths[][] = {
        "./csvs/pawels/pawel0.csv",
        "./csvs/pawels/pawel1.csv",
        "./csvs/pawels/pawel2.csv",
        "./csvs/pawels/pawel3.csv",
        "./csvs/pawels/pawel4.csv",
        "./csvs/pawels/pawel5.csv",
        "./csvs/pawels/pawel6.csv",
        "./csvs/pawels/pawel7.csv",
        "./csvs/pawels/pawel8.csv",
        "./csvs/pawels/pawel9.csv",
        "./csvs/pawels/pawel10.csv",
        "./csvs/pawels/pawel11.csv",
        "./csvs/pawels/pawel12.csv",
        "./csvs/pawels/pawel13.csv",
        "./csvs/pawels/pawel14.csv",
        "./csvs/pawels/pawel15.csv",
        "./csvs/pawels/pawel16.csv",
        "./csvs/pawels/pawel17.csv",
        "./csvs/pawels/pawel18.csv",
        "./csvs/pawels/pawel19.csv"
    };
    
    char obama_paths[][] = {
        "./csvs/obamas/obama0.csv",
        "./csvs/obamas/obama1.csv",
        "./csvs/obamas/obama2.csv",
        "./csvs/obamas/obama3.csv",
        "./csvs/obamas/obama4.csv",
        "./csvs/obamas/obama5.csv",
        "./csvs/obamas/obama6.csv",
        "./csvs/obamas/obama7.csv",
        "./csvs/obamas/obama8.csv",
        "./csvs/obamas/obama9.csv",
        "./csvs/obamas/obama10.csv",
        "./csvs/obamas/obama11.csv",
        "./csvs/obamas/obama12.csv",
        "./csvs/obamas/obama13.csv",
        "./csvs/obamas/obama14.csv",
        "./csvs/obamas/obama15.csv",
        "./csvs/obamas/obama16.csv",
        "./csvs/obamas/obama17.csv",
        "./csvs/obamas/obama18.csv",
        "./csvs/obamas/obama19.csv"
    };
    
    char madonna_paths[][] = {
        "./csvs/madonnas/madonna0.csv",
        "./csvs/madonnas/madonna1.csv",
        "./csvs/madonnas/madonna2.csv",
        "./csvs/madonnas/madonna3.csv",
        "./csvs/madonnas/madonna4.csv",
        "./csvs/madonnas/madonna5.csv",
        "./csvs/madonnas/madonna6.csv",
        "./csvs/madonnas/madonna7.csv",
        "./csvs/madonnas/madonna8.csv",
        "./csvs/madonnas/madonna9.csv",
        "./csvs/madonnas/madonna10.csv",
        "./csvs/madonnas/madonna11.csv",
        "./csvs/madonnas/madonna12.csv",
        "./csvs/madonnas/madonna13.csv",
        "./csvs/madonnas/madonna14.csv",
        "./csvs/madonnas/madonna15.csv",
        "./csvs/madonnas/madonna16.csv",
        "./csvs/madonnas/madonna17.csv",
        "./csvs/madonnas/madonna18.csv",
        "./csvs/madonnas/madonna19.csv"
    };
    
    // DATASET
    int number_of_images = 60;
    int labels[number_of_images]; // 0 - madonna, 1 - obama, 2 - pawel
    ConvolutionalBox conv_boxes[number_of_images];
    
    int i;
    for (i = 0; i < number_of_images; i++) {
        if (i < 20) {
            read_csv(&conv_boxes[i], madonna_paths[i]);
            labels[i] = 0;
        } else if (i < 40) {
            read_csv(&conv_boxes[i], obama_paths[i]);
            labels[i] = 1;            
        } else {
            read_csv(&conv_boxes[i], pawel_paths[i]);
            labels[i] = 2;        
        }
    }  
    
    // data has been saved to conv boxes
    // now we mix images to group them in the "packages"
    // that is triplets consisting of anchor, pos, neg

    // package - index of anchor, anchors = convb[0], convb[20], convb[40]
    // number of packages = number of images * 25
    Package packages[NUMBER_OF_IMAGES * 25];
    
    int j;
    srand(time(NULL));
    for (i = 0; i < NUMBER_OF_IMAGES; i++) {
        for (j = 0; j < 25; j++) {
            if (i < 20) {
                packages[i * NUMBER_OF_IMAGES + j].anchor = 0;
                packages[i * NUMBER_OF_IMAGES + j].positive = rand() % 20;
                packages[i * NUMBER_OF_IMAGES + j].negative = rand() % 40 + 20;
            } else if (i < 40) {
                packages[i * NUMBER_OF_IMAGES + j].anchor = 20;
                packages[i * NUMBER_OF_IMAGES + j].positive = rand() % 20 + 20;
                packages[i * NUMBER_OF_IMAGES + j].negative = ((rand() % 40) + 40) % 60;
            } else {
                packages[i * NUMBER_OF_IMAGES + j].anchor = 40;
                packages[i * NUMBER_OF_IMAGES + j].positive = rand() % 20 + 20;
                packages[i * NUMBER_OF_IMAGES + j].negative = rand() % 40;
            }
        }   
    }
    
    // snn construction
    SNN snn;
    
    // model initialization
    snn.convBox1.height = 125;
    snn.convBox1.width = 125;
    snn.convBox1.depth = 3;
    
    snn.filter1.size = 5;
    snn.filter1.number_of_filters = 1;
    snn.stride1 = 1;
    // TODO - znalezc sensowny filtr
    // snn.filter1.entries[][][] = ;
    
    snn.stride2 = 2;
    
    snn.filter2.size = 4;
    snn.filter2.number_of_filters = 1;
    snn.stride3 = 1;
    // jw
    
    snn.stride4 = 2;
    
    int OUTPUT_SIZE = 2352;
    
    snn.fcl.height = OUTPUT_SIZE; 
    snn.fcl.width = OUTPUT_SIZE;
    snn.fcl.depth = 5;
    
    for (k = 0; k < snn.fcl.depth; k++) {
        for (i = 0; i < snn.fcl.height; i++) {
            for (j = 0; j < snn.fcl.width; j++) {
                snn.fcl.weights[i][j][k] = (rand() % 100  - 50)/ 1000.0;
            }
            snn.fcl.biases[i][k] = rand() % (100 - 50)/ 100.0;
        }
    }
    
    snn.costs.size = OUTPUT_SIZE
    
    snn.fpd.height = OUTPUT_SIZE;
    snn.fpd.depth = 5;
    
    snn.bpd.height = OUTPUT_SIZE;
    snn.bpd.width = OUTPUT_SIZE;
    snn.bpd.depth = 5;
    
    int t;
    for (t = 0; t < NUMBER_OF_ITERATIONS; t++) {
        for (i = 0; i < NUMBER_OF_IMAGES * 25; i++) {
            // kazda paczka przechodzi cykl forwardprop, backprop, ocena
            // get encoding of anchor
            Vector anchor_encoding;
            get_encoding(&snn, &anchor_encoding, &conv_boxes[packages[i].anchor]);
            
            // get encoding of positive
            Vector positive_encoding;
            get_encoding(&snn, &positive_encoding, &conv_boxes[packages[i].positive]);
            
            // get encoding of negative
            Vector negative_encoding;
            get_encoding(&snn, &negative_encoding, &conv_boxes[packages[i].negative]);
            
            // triplet cost
            triplet_loss(&anchor_encoding, &positive_encoding, &negative_encoding, &snn.costs);
            
            // backprop, potrzebuje fpd anchora, 2 x backprop dla pos i neg
            // tbc
        }
    }
    
}