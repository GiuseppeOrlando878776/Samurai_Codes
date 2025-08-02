cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-mtune=native -march=native -O3 -g"
cmake --build build --target all
