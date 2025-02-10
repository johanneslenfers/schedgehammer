g++ -c -fPIC -g taco_api.cpp -o taco_api.o -ltaco
g++ -shared -g -Wl,-soname,taco_api.so -o taco_api.so taco_api.o -ltaco
python taco_bindings.py