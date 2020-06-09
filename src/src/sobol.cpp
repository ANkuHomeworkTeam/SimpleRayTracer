#include "sobol.hpp"
#include <cmath>
#include <stdlib.h>

namespace Renderer
{
    Vec3 sobolSequence[SOBOL_SEQUENCE_CYCLE] = {};
    void initSobolSequence() {
        int index = 0;
        int cycle = SOBOL_SEQUENCE_CYCLE;
        unsigned L = (unsigned)std::ceil(std::log(float(cycle))/std::log(2.f));
        unsigned *C = new unsigned[cycle];
        C[0] = 1;
        for (unsigned i=1; i<cycle; i++) {
            C[i] = 1;
            unsigned value = i;
            while (value & 1) {
                value >>= 1;
                C[i]++;
            }
        }

        unsigned *V = new unsigned[L+1];

        for(unsigned i = 1; i<=L; i++)
            V[i] = 1 << (32 - i);
        
        unsigned *X = new unsigned [cycle];
        X[0] = 0;
        for (unsigned i=1; i<=cycle; i++) {
            X[i] = X[i-1]^V[C[i-1]];
            sobolSequence[i].x = (float)X[i]/std::pow(2.f, 32);
        }

        delete[] X;
        delete[] V;

        unsigned D = 3;

        unsigned ss[2] = {1, 2};
        unsigned aa[2] = {0, 1};

        for (unsigned j=1; j<=D-1; j++) {
            unsigned s = ss[j-1];
            unsigned a = aa[j-1];
            unsigned m[3] = {0, 1, 3};
            
            unsigned *V = new unsigned [L+1];

            if(L<=s) {
                for (unsigned i=1; i<=L; i++)
                    V[i] = m[i] << (32-i);
            }
            else {
                for (unsigned i=1; i<=s; i++)
                    V[i] = m[i] << (32 - i);
                for (unsigned i=s+1; i<=L; i++) {
                    V[i] = V[i-s]^(V[i-s] >> s);
                    for (unsigned k=1; k<=s-1; k++)
                        V[i] ^=(((a>>(s-1-k)) & 1) * V[i-k]);
                }
            }

            unsigned *X = new unsigned[cycle];
            X[0] = 0;
            for (unsigned i=1; i<=cycle-1; i++) {
                X[i] = X[i-1] ^ V[C[i-1]];
                sobolSequence[i].d[j] = float(X[i])/std::pow(2.f, 32);
            }
            delete[] V;
            delete[] X;
        }
        delete[] C;      
    }
};