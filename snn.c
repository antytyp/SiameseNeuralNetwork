#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.c"

typedef struct {
    int anchor;
    int positive;
    int negative;
} Package;

void get_encoding(SNN* snn, Vector* encoding, ConvolutionalBox* conv_box) {    
    conv2D(conv_box, &snn->convBox2, &snn->filter1, 1);
    convbox_relu(&snn->convBox2);
    max_pooling(&snn->convBox2, &snn->convBox3, 3);
    conv2D(&snn->convBox3, &snn->convBox4, &snn->filter2, 1);
    convbox_relu(&snn->convBox4);
    max_pooling(&snn->convBox4, &snn->convBox5, 3);
    conv2D(&snn->convBox5, &snn->convBox6, &snn->filter3, 1);
    convbox_relu(&snn->convBox6);
    max_pooling(&snn->convBox6, &snn->convBox7, 3);
    // conv2D(&snn->convBox7, &snn->convBox8, &snn->filter4, 1);
    // convbox_relu(&snn->convBox8);
    flatten(&snn->convBox7, &snn->vector);
    sigmoid(snn->vector.size, snn->vector.entries);
    int i;

    dense(&snn->vector, &snn->fcl, &snn->encoding, &snn->fpd);
    
    for (i = 0; i < encoding->size; i++) {
        encoding->entries[i] = snn->encoding.entries[i];
    }
}

int main() {
    srand(time(NULL));
    char pawel_paths[20][40] = {
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

    char obama_paths[20][40] = {
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
    
    char madonna_paths[20][40] = {
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
    int NUMBER_OF_IMAGES = 60;

    int labels[NUMBER_OF_IMAGES]; // 0 - madonna, 1 - obama, 2 - pawel

    ConvolutionalBox* conv_boxes = (ConvolutionalBox*)malloc(sizeof(ConvolutionalBox) * NUMBER_OF_IMAGES);
    if (conv_boxes == NULL) printf("Malloc failed\n");    

    int i;
    for (i = 0; i < NUMBER_OF_IMAGES; i++) {
        conv_boxes[i].height = 125;
        conv_boxes[i].width = 125;
        conv_boxes[i].depth = 3;
    }

    for (i = 0; i < NUMBER_OF_IMAGES; i++) {
        if (i < 20) {
            read_csv(conv_boxes + i, madonna_paths[i]);
            labels[i] = 0;
        } else if (i < 40) {
            read_csv(conv_boxes + i, obama_paths[i]);
            labels[i] = 1;            
        } else {
            read_csv(conv_boxes + i, pawel_paths[i]);
            labels[i] = 2;        
        }
    }

    printf("Images loaded succesfully\n");

    // data has been saved to conv boxes
    // now we mix images to group them into "packages"
    // that is triplets consisting of anchor, pos, neg

    // package - index of anchor, anchors = convb[0], convb[20], convb[40]
    // number of packages = number of images * 25
    
    int TRIPLETS_PER_SAMPLE = 25;
    Package packages[NUMBER_OF_IMAGES * TRIPLETS_PER_SAMPLE];

    int j;

    for (i = 0; i < NUMBER_OF_IMAGES; i++) {
        for (j = 0; j < TRIPLETS_PER_SAMPLE; j++) {
            if (i < 20) {
                packages[i * TRIPLETS_PER_SAMPLE + j].anchor = 0;
                packages[i * TRIPLETS_PER_SAMPLE + j].positive = rand() % 20;
                packages[i * TRIPLETS_PER_SAMPLE + j].negative = rand() % 40 + 20;
            } else if (i < 40) {
                packages[i * TRIPLETS_PER_SAMPLE + j].anchor = 20;
                packages[i * TRIPLETS_PER_SAMPLE + j].positive = rand() % 20 + 20;
                packages[i * TRIPLETS_PER_SAMPLE + j].negative = ((rand() % 40) + 40) % 60;
            } else {
                packages[i * TRIPLETS_PER_SAMPLE + j].anchor = 40;
                packages[i * TRIPLETS_PER_SAMPLE + j].positive = rand() % 20 + 20;
                packages[i * TRIPLETS_PER_SAMPLE + j].negative = rand() % 40;
            }
        }
    }
    printf("Triplet data set prepared\n");

    // snn construction
    SNN* snn = (SNN*)malloc(sizeof(SNN));

    // model initialization
    snn->convBox1.height = 125;
    snn->convBox1.width = 125;
    snn->convBox1.depth = 3;

    snn->filter1.size = 3;
    snn->filter1.number_of_filters = 3;
    snn->stride1 = 1;
    
    // sharpen
    snn->filter1.entries[0][0][0] = 0;
    snn->filter1.entries[1][0][0] = -1;    
    snn->filter1.entries[2][0][0] = 0;
    snn->filter1.entries[0][1][0] = -1;
    snn->filter1.entries[1][1][0] = 5;
    snn->filter1.entries[2][1][0] = -1;
    snn->filter1.entries[0][2][0] = 0;
    snn->filter1.entries[1][2][0] = -1;
    snn->filter1.entries[2][2][0] = 0;
    // edge detection
    snn->filter1.entries[0][0][1] = -1;
    snn->filter1.entries[1][0][1] = -1;
    snn->filter1.entries[2][0][1] = -1;
    snn->filter1.entries[0][1][1] = -1;
    snn->filter1.entries[1][1][1] = 8;
    snn->filter1.entries[2][1][1] = -1;
    snn->filter1.entries[0][2][1] = -1;
    snn->filter1.entries[1][2][1] = -1;
    snn->filter1.entries[2][2][1] = -1;
    // gaussian blur
    snn->filter1.entries[0][0][2] = 1.0/16;
    snn->filter1.entries[1][0][2] = 2.0/16;
    snn->filter1.entries[2][0][2] = 1.0/16;
    snn->filter1.entries[0][1][2] = 2.0/16;
    snn->filter1.entries[1][1][2] = 4.0/16;
    snn->filter1.entries[2][1][2] = 2.0/16;
    snn->filter1.entries[0][2][2] = 1.0/16;
    snn->filter1.entries[1][2][2] = 2.0/16;
    snn->filter1.entries[2][2][2] = 1.0/16;
    
    snn->stride2 = 2;

    snn->filter2.size = 3;
    snn->filter2.number_of_filters = 3;
    snn->stride3 = 1;
    
    // box blur
    snn->filter2.entries[0][0][0] = 1.0/9;
    snn->filter2.entries[1][0][0] = 1.0/9;    
    snn->filter2.entries[2][0][0] = 1.0/9;
    snn->filter2.entries[0][1][0] = 1.0/9;
    snn->filter2.entries[1][1][0] = 1.0/9;
    snn->filter2.entries[2][1][0] = 1.0/9;
    snn->filter2.entries[0][2][0] = 1.0/9;
    snn->filter2.entries[1][2][0] = 1.0/9;
    snn->filter2.entries[2][2][0] = 1.0/9;
    // edge detection
    snn->filter2.entries[0][0][1] = 0;
    snn->filter2.entries[1][0][1] = 1;
    snn->filter2.entries[2][0][1] = 0;
    snn->filter2.entries[0][1][1] = 1;
    snn->filter2.entries[1][1][1] = -4;
    snn->filter2.entries[2][1][1] = 1;
    snn->filter2.entries[0][2][1] = 0;
    snn->filter2.entries[1][2][1] = 1;
    snn->filter2.entries[2][2][1] = 0;
    // Sobel filter
    snn->filter2.entries[0][0][2] = -1;
    snn->filter2.entries[1][0][2] = 0;
    snn->filter2.entries[2][0][2] = 1;
    snn->filter2.entries[0][1][2] = -2;
    snn->filter2.entries[1][1][2] = 0;
    snn->filter2.entries[2][1][2] = 2;
    snn->filter2.entries[0][2][2] = -1;
    snn->filter2.entries[1][2][2] = 0;
    snn->filter2.entries[2][2][2] = 1;
    
    snn->stride4 = 3;
    
    snn->filter3.size = 5;
    snn->filter3.number_of_filters = 1;
    snn->stride5 = 1;
    // Kang & Park filter
    snn->filter3.entries[0][0][0] = -1;
    snn->filter3.entries[1][0][0] = -1;
    snn->filter3.entries[2][0][0] = -1;
    snn->filter3.entries[3][0][0] = -1;
    snn->filter3.entries[4][0][0] = -1;
    snn->filter3.entries[0][1][0] = -1;
    snn->filter3.entries[1][1][0] = -1;
    snn->filter3.entries[2][1][0] = 4;
    snn->filter3.entries[3][1][0] = -1;
    snn->filter3.entries[4][1][0] = -1;
    snn->filter3.entries[0][2][0] = -1;
    snn->filter3.entries[1][2][0] = 4;
    snn->filter3.entries[2][2][0] = 4;
    snn->filter3.entries[3][2][0] = 4;
    snn->filter3.entries[4][2][0] = -1;
    snn->filter3.entries[0][3][0] = -1;
    snn->filter3.entries[1][3][0] = -1;
    snn->filter3.entries[2][3][0] = 4;
    snn->filter3.entries[3][3][0] = -1;
    snn->filter3.entries[4][3][0] = -1;
    snn->filter3.entries[0][4][0] = -1;
    snn->filter3.entries[1][4][0] = -1;
    snn->filter3.entries[2][4][0] = -1;
    snn->filter3.entries[3][4][0] = -1;
    snn->filter3.entries[4][4][0] = -1;
    
    snn->stride6 = 3;
    
    snn->filter4.size = 3;
    snn->filter4.number_of_filters = 1;
    snn->stride7 = 1;
    
    // identity
    snn->filter4.entries[0][0][0] = 0;
    snn->filter4.entries[0][1][0] = 0;
    snn->filter4.entries[0][2][0] = 0;
    snn->filter4.entries[1][0][0] = 0;
    snn->filter4.entries[1][1][0] = 0.1;
    snn->filter4.entries[1][2][0] = 0;
    snn->filter4.entries[2][0][0] = 0;
    snn->filter4.entries[2][1][0] = 0;
    snn->filter4.entries[2][2][0] = 0;
    
    snn->stride8 = 3;

    int OUTPUT_SIZE = 243;

    snn->fcl.height = OUTPUT_SIZE; 
    snn->fcl.width = OUTPUT_SIZE;
    snn->fcl.depth = 2;
    
    int k;
    for (k = 0; k < snn->fcl.depth; k++) {
        for (i = 0; i < snn->fcl.height; i++) {
            for (j = 0; j < snn->fcl.width; j++) {
                snn->fcl.weights[i][j][k] = (rand() % 100  - 50)/ 10000.0;
            }
            snn->fcl.biases[i][k] = (rand() % 100 - 50)/ 10000.0;
        }
    }
    printf("Filters loaded succesfully\n");

    snn->vector.size = OUTPUT_SIZE;
    snn->costs.size = OUTPUT_SIZE;

    snn->fpd.height = OUTPUT_SIZE;
    snn->fpd.depth = 2;

    snn->bpd.height = OUTPUT_SIZE;
    snn->bpd.width = OUTPUT_SIZE;
    snn->bpd.depth = 2;

    // remember to free memory
    double*** VdW = (double***) malloc(sizeof(double**) * snn->fcl.height);
    for (i = 0; i < snn->fcl.height; i++) {
        VdW[i] = (double**) malloc(sizeof(double*) * snn->fcl.width);
        for (j = 0; j < snn->fcl.width; j++) {
            VdW[i][j] = (double*) malloc(sizeof(double) * snn->fcl.depth);
        }
    }

    double*** SdW = (double***) malloc(sizeof(double**) * snn->fcl.height);
    for (i = 0; i < snn->fcl.height; i++) {
        SdW[i] = (double**) malloc(sizeof(double*) * snn->fcl.width);
        for (j = 0; j < snn->fcl.width; j++) {
            SdW[i][j] = (double*) malloc(sizeof(double) * snn->fcl.depth);
        }
    }

    double** Vdb = (double**) malloc(sizeof(double*) * snn->fcl.width);
    for (i = 0; i < snn->fcl.width; i++) {
        Vdb[i] = (double*) malloc(sizeof(double*) * snn->fcl.depth);
    }

    double** Sdb = (double**) malloc(sizeof(double*) * snn->fcl.width);
    for (i = 0; i < snn->fcl.width; i++) {
        Sdb[i] = (double*) malloc(sizeof(double*) * snn->fcl.depth);
    }

    double*** VdWcorr = (double***) malloc(sizeof(double**) * snn->fcl.height);
    for (i = 0; i < snn->fcl.height; i++) {
        VdWcorr[i] = (double**) malloc(sizeof(double*) * snn->fcl.width);
        for (j = 0; j < snn->fcl.width; j++) {
            VdWcorr[i][j] = (double*) malloc(sizeof(double) * snn->fcl.depth);
        }
    }

    double*** SdWcorr = (double***) malloc(sizeof(double**) * snn->fcl.height);
    for (i = 0; i < snn->fcl.height; i++) {
        SdWcorr[i] = (double**) malloc(sizeof(double*) * snn->fcl.width);
        for (j = 0; j < snn->fcl.width; j++) {
            SdWcorr[i][j] = (double*) malloc(sizeof(double) * snn->fcl.depth);
        }
    }

    double** Vdbcorr = (double**) malloc(sizeof(double*) * snn->fcl.width);
    for (i = 0; i < snn->fcl.width; i++) {
        Vdbcorr[i] = (double*) malloc(sizeof(double*) * snn->fcl.depth);
    }

    double** Sdbcorr = (double**) malloc(sizeof(double*) * snn->fcl.width);
    for (i = 0; i < snn->fcl.width; i++) {
        Sdbcorr[i] = (double*) malloc(sizeof(double*) * snn->fcl.depth);
    }
    printf("Ready to compute\n");
    int h, w, d;
    for (d = 0; d < snn->fcl.depth; d++) {
        for (w = 0; w < snn->fcl.width; w++) {
            for (h = 0; h < snn->fcl.height; h++) {
                VdW[h][w][d] = 0.0;
                SdW[h][w][d] = 0.0;
            }
            Vdb[w][d] = 0.0;
            Sdb[w][d] = 0.0;
        }
    }

    Vector positive_encoding;
    Vector negative_encoding;
    Vector anchor_encoding;

    Vector avg_costs[40];
    for (i = 0; i < 40; i++) {
        avg_costs[i].size = 243;
    }

    positive_encoding.size = OUTPUT_SIZE;
    negative_encoding.size = OUTPUT_SIZE;
    anchor_encoding.size = OUTPUT_SIZE;
    int t;
    for (t = 1; t < NUMBER_OF_ITERATIONS; t++) { // t > 0 !!!
	printf("Iteration: %d \n", t);
        for (i = 0; i < 40; i++) { // two classes by now
            // get encoding of positive
            get_encoding(snn, &positive_encoding, &conv_boxes[packages[i].positive]);

            // get encoding of negative
            get_encoding(snn, &negative_encoding, &conv_boxes[packages[i].negative]);

            // get encoding of anchor
            get_encoding(snn, &anchor_encoding, &conv_boxes[i]);

            // triplet cost
            triplet_loss(&anchor_encoding, &positive_encoding, &negative_encoding, &snn->costs);

	    // simple_loss(&anchor_encoding, &positive_encoding, &negative_encoding, &snn->costs);
            
	    // simple_loss2(labels[i], &anchor_encoding, &snn->costs);
	    int j;
	    for (j = 0; j < 243; j++) avg_costs[i].entries[j] = snn->costs.entries[j];

            backpropagation(40, &snn->fpd, &snn->bpd, &snn->fcl, &snn->costs);

            for (d = 0; d < snn->fcl.depth; d++) {
                for (w = 0; w < snn->fcl.width; w++) {
                    for (h = 0; h < snn->fcl.height; h++) {
                        VdW[h][w][d] = BETA_1 * VdW[h][w][d] + (1 - BETA_1) * snn->bpd.d_weights[h][w][d];
                        SdW[h][w][d] = BETA_2 * SdW[h][w][d] + (1 - BETA_2) * snn->bpd.d_weights[h][w][d] * snn->bpd.d_weights[h][w][d]; 
                    }
                    Vdb[w][d] = BETA_1 * Vdb[w][d] + (1 - BETA_1) * snn->bpd.d_biases[w][d];
                    Sdb[w][d] = BETA_2 * Sdb[w][d] + (1 - BETA_2) * snn->bpd.d_biases[w][d] * snn->bpd.d_biases[w][d];
                }
            }

            // "correction"
            double beta1_corr = 1.0 / (1 - pow(BETA_1, t));
            double beta2_corr = 1.0 / (1 - pow(BETA_2, t));
            for (d = 0; d < snn->fcl.depth; d++) {
                for (w = 0; w < snn->fcl.width; w++) {
                    for (h = 0; h < snn->fcl.height; h++) {
                        VdWcorr[h][w][d] = VdW[h][w][d] * beta1_corr;
                        SdWcorr[h][w][d] = SdW[h][w][d] * beta2_corr;
                    }
                    Vdbcorr[w][d] = Vdb[w][d] * beta1_corr;
                    Sdbcorr[w][d] = Sdb[w][d] * beta2_corr;
                }
            }

            // neural net update
            for (d = 0; d < snn->fcl.depth; d++) {
                for (w = 0; w < snn->fcl.width; w++) {
                    for (h = 0; h < snn->fcl.height; h++) {
                        snn->fcl.weights[h][w][d] = snn->fcl.weights[h][w][d] - ALPHA * (VdWcorr[h][w][d] / (sqrt(SdWcorr[h][w][d]) + EPSILON));
                    }
                    snn->fcl.biases[w][d] = snn->fcl.biases[w][d] - ALPHA * (Vdbcorr[w][d] / (sqrt(Sdbcorr[w][d]) + EPSILON));
                }
            }
        }
    // average cost computed
    double suma = 0.0;
	int j, k;
	for (j = 0; j < 40; j++) {
	    for (k = 0; k < 243; k++) {
		suma += avg_costs[j].entries[k] * avg_costs[j].entries[k];
	    }
	}
	suma = suma / (40 * 243.0);
	printf("Average cost: %f\n", suma);
    }
    for (i = 0; i < 40; i++) {
        get_encoding(snn, &anchor_encoding, &conv_boxes[i]);
	int j;
	for (j = 0; j < anchor_encoding.size; j++) printf("%f, ", anchor_encoding.entries[j]);
	printf("\n");
    }
    free(conv_boxes);
    free(snn);
    return 0;
}