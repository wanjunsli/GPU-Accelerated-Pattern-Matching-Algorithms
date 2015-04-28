# GPU-Accelerated-Rabin-Karp
Rabin-Karp Pattern Matching Accelerated on the GPU using PyCuda 

Specify architecture option, with -arch=sm_30, else it will try to run with -arch=sm_35, which won't work.

Testing files are pulled from Python's NLTK Corpora, specifically under nltk_data/corpora/gutenberg/

Add this to your .bashrc for PyCuda:

export CUDA_HOME=/usr/local/cuda-7.0/

export LD_LIBRARY_PATH=${CUDA_HOME}/lib64/

PATH=${CUDA_HOME}/bin:${PATH}
