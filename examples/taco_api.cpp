#include <iostream>
#include <time.h>
#include "taco.h"

using namespace taco;
int n = 512;
extern "C" {
     struct ScheduleEnv{
        Tensor<double> computed_tensor;
        Tensor<double> static_tensors[2];
        IndexVar original_axes[3];
        IndexStmt stmt;
    };

    ScheduleEnv* create_schedule(){
        std::cout << "Creating schedule..." << std::endl;
        
        try {
            ScheduleEnv* env = new ScheduleEnv();
            env->static_tensors[0]= Tensor<double>({n,n}, Format({Dense, Dense}));
            env->static_tensors[1] = Tensor<double>({n,n}, Format({Dense, Dense}));
            env->computed_tensor = Tensor<double>({n,n}, Format({Dense, Dense}));
            
            IndexVar i, j, k;
            env->computed_tensor(i,j) = env->static_tensors[0](i,k) * env->static_tensors[1](k,j);
            env->stmt = env->computed_tensor.getAssignment().concretize();
            env->original_axes[0] = i;
            env->original_axes[1] = j;
            env->original_axes[2] = k;
            std::cout << "Schedule created successfully" << std::endl;
            return env;
        } catch (const std::exception& e) {
            std::cerr << "Error in create_schedule: " << e.what() << std::endl;
            return nullptr;
        }
    }

    void finish_schedule(ScheduleEnv* env){
        try {
            std::cout << "Starting finish_schedule..." << std::endl;
            
            // Set random seed once
            srand(time(NULL));
            
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    env->static_tensors[0].insert({i,j}, (double)rand() / RAND_MAX);
                    env->static_tensors[1].insert({i,j}, (double)rand() / RAND_MAX);
                }
            }
            
            std::cout << "Inserted values, now packing..." << std::endl;
            
            env->static_tensors[0].pack();
            env->static_tensors[1].pack();
            
            std::cout << "Tensors packed, now compiling..." << std::endl;
            
            env->computed_tensor.compile();
            env->computed_tensor.assemble();
            
            std::cout << "Finish_schedule completed" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error in finish_schedule: " << e.what() << std::endl;
        }
    }

    double cost_function(ScheduleEnv* env){
        try {
            std::cout << "Starting computation..." << std::endl;
            
            clock_t begin = clock();
            env->computed_tensor.compute();
            clock_t end = clock();
            double error = 0.0;
            for(int i = 0; i < n; i++) {
                for(int j = 0; j < n; j++) {
                    double expected = 0.0;
                    for(int k = 0; k < n; k++) {
                        double a_val = env->static_tensors[0].at({i,k}); // TACO returns 0 if value not present
                        double b_val = env->static_tensors[1].at({k,j});
                        expected += a_val * b_val;
                    }
                    
                    double actual = env->computed_tensor.at({i,j});
                    error += std::abs(expected - actual);
                }
            }
            if(error > 1e-6) {
                std::cout << "Warning: Large error in computation: " << error << std::endl;
            }
            double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
            std::cout << "Time spent: " << time_spent << std::endl;
            return time_spent;
        } catch (const std::exception& e) {
            std::cerr << "Error in cost_function: " << e.what() << std::endl;
            return -1;
        }
    }


    Tensor<double>* get_computed_tensor(ScheduleEnv* env){
        return &env->computed_tensor;
    }

    Tensor<double>* get_static_tensor(ScheduleEnv* env, int index){
        return &env->static_tensors[index];
    }

    IndexVar* get_original_axes(ScheduleEnv* env, int index){
        return &env->original_axes[index];
    }

    IndexStmt* get_stmt(ScheduleEnv* env){
        return &env->stmt;
    }
}