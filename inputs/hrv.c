int main()
{
int N= 50;
int M= 16;
int P= 24;
int F=5;
int E=8;
int ensemble2value[49][15][23][23], ensemble2inputs[49][7][27][27],  ensemble2weights[15][7][4][4], ensemble3bias[15][0], ensemble3value[49][15][23][23] , ensemble4value[49][15][23][23];

#pragma scop 
for (int _neuron_index_0 = 0; _neuron_index_0 <  N; _neuron_index_0 += 1) {
        for (int _neuron_index_1 = 0; _neuron_index_1 <  M; _neuron_index_1 += 1) {
            for (int _neuron_index_2 = 0; _neuron_index_2 <  P; _neuron_index_2 += 1) {
                for (int _neuron_index_3 = 0; _neuron_index_3 <  P; _neuron_index_3 += 1) {
                    for (int j = 0; j < F; j += 1) {
                        for (int k = 0; k < F; k += 1) {
                            for (int i = 0; i < E; i += 1) {
                                ensemble2value[_neuron_index_0][_neuron_index_1][_neuron_index_2][_neuron_index_3] += ensemble2inputs[_neuron_index_0][(i * 1)][(j * 1 + _neuron_index_2)][(k * 1 + _neuron_index_3)] * ensemble2weights[_neuron_index_1][i][j][k];
                            }
                        }
                    }
                }
            }
        }
    }
    for (int _neuron_index_0 = 0; _neuron_index_0 <  N; _neuron_index_0 += 1) {
        for (int _neuron_index_1 = 0; _neuron_index_1 <  M; _neuron_index_1 += 1) {
            for (int _neuron_index_2 = 0; _neuron_index_2 <  P; _neuron_index_2 += 1) {
                for (int _neuron_index_3 = 0; _neuron_index_3 <  P; _neuron_index_3 += 1) {
                    ensemble3value[_neuron_index_0][_neuron_index_1][_neuron_index_2][_neuron_index_3] = ensemble2value[_neuron_index_0][_neuron_index_1][_neuron_index_2][_neuron_index_3] + ensemble3bias[_neuron_index_1][0];
                }
            }
        }
    }
    for (int _neuron_index_0 = 0; _neuron_index_0 <  N; _neuron_index_0 += 1) {
        for (int _neuron_index_1 = 0; _neuron_index_1 <  M; _neuron_index_1 += 1) {
            for (int _neuron_index_2 = 0; _neuron_index_2 <  P; _neuron_index_2 += 1) {
                for (int _neuron_index_3 = 0; _neuron_index_3 <  P; _neuron_index_3 += 1) {
                    ensemble4value[_neuron_index_0][_neuron_index_1][_neuron_index_2][_neuron_index_3] = (ensemble3value[_neuron_index_0][_neuron_index_1][_neuron_index_2][_neuron_index_3]);
                }
            }
        }
    }
#pragma endscop
}
