cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSAMURAI_WITH_MPI=ON
cmake --build build --target all
