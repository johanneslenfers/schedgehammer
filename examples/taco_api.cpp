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

    void finish_schedule(Tensor<double>* A, Tensor<double>* B, Tensor<double>* C){
        try {
            std::cout << "Starting finish_schedule..." << std::endl;
            
            // Set random seed once
            srand(time(NULL));
            
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    A->insert({i,j}, (double)rand() / RAND_MAX);
                    B->insert({i,j}, (double)rand() / RAND_MAX);
                }
            }
            
            std::cout << "Inserted values, now packing..." << std::endl;
            
            A->pack();
            B->pack();
            
            std::cout << "Tensors packed, now compiling..." << std::endl;
            
            C->compile();
            C->assemble();
            
            std::cout << "Finish_schedule completed" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error in finish_schedule: " << e.what() << std::endl;
        }
    }

    double cost_function(Tensor<double>* A, Tensor<double>* B, Tensor<double>* C){
        try {
            std::cout << "Starting computation..." << std::endl;
            
            clock_t begin = clock();
            C->compute();
            clock_t end = clock();
            double error = 0.0;
            for(int i = 0; i < n; i++) {
                for(int j = 0; j < n; j++) {
                    double expected = 0.0;
                    for(int k = 0; k < n; k++) {
                        double a_val = A->at({i,k}); 
                        double b_val = B->at({k,j});
                        expected += a_val * b_val;
                    }
                    
                    double actual = C->at({i,j});
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
    IndexVar* generate_new_axis(){
        std::string random_str;
        const std::string chars = "abcdefghijklmnopqrstuvwxyz";
        for(int j = 0; j < 10; j++) {
            random_str += chars[rand() % chars.length()];
        }
        return new IndexVar(random_str);
    }
    void reorder(IndexStmt* stmt, IndexVar* axes, int count_axes){
        std::vector<IndexVar> vars(axes, axes + count_axes);
        *stmt = stmt->reorder(vars);
    }
    void split(IndexStmt* stmt, IndexVar* parent, IndexVar* outer, IndexVar* inner, int split_factor) {
        *stmt = stmt->split(*parent, *outer, *inner, split_factor); 
    }
}   
int main(){
    ScheduleEnv* env = create_schedule();
    IndexVar* i = generate_new_axis();
    IndexVar* j = generate_new_axis();
    split(&env->stmt, &env->original_axes[0], i, j, 2);
    finish_schedule(&env->static_tensors[0], &env->static_tensors[1], &env->computed_tensor);
    std::cout << "Cost: " << cost_function(&env->static_tensors[0], &env->static_tensors[1], &env->computed_tensor) << std::endl;
    return 0;
}