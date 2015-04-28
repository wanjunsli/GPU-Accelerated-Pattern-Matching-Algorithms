import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy
import math 
import nltk

THREADS_PER_BLOCK = 512 

def RabinKarp(text, pattern, d, q): 
    mod = SourceModule("""
    #define THREADS_PER_BLOCK 512 
    __global__ void search(char *text, char *pattern, int h, int q, int d, int *output, int M, int N, int NUM_BLOCKS)
    {
        int index = blockDim.x * blockIdx.x + threadIdx.x;

        int chunk_len = 512000 / THREADS_PER_BLOCK;
        int start = index * chunk_len;
        int end = ( index + 1 ) * chunk_len + ( M - 1 );

        end = min( end, N );
        if (end < start)
        {
            return; 
        }

        if (end - start < M)
        {
            return;
        }

        int i, j = 0 ; 
        int hashPattern = 0; 
        int hashText = 0; 

        for (i = 0; i < M; i++)
        {
            hashPattern = (d * hashPattern + pattern[i]) % q;
            hashText = (d * hashText + text[i + start]) % q; 
        }

        for (i = start; i <= end - M; i++)
        {
            if (hashPattern == hashText)
            {
                for (j = 0; j < M; j++)
                {
                    if (text[i + j] != pattern[j])
                    {
                        break;
                    }
                }
                if (j == M)
                {
                    output[i] = 1; 
                }
            }
            if (i < end - M)                          
            {
                hashText = (d * (hashText - text[i] * h) + text[i + M]) % q;                                                        
                if (hashText < 0)   
                {
                    hashText += q; 
                } 
            }
        }
    }
    """)

    dText = gpuarray.to_gpu(numpy.array([text], dtype=str))         # set device memory for text + pattern
    dPattern = gpuarray.to_gpu(numpy.array([pattern], dtype=str))
    dResults = gpuarray.zeros(len(text), dtype=numpy.int32)         # hold the resulting indices
                                                                    # at each index, 0/1 if there's a match or not 
    dD = numpy.int32(d)         # gpu d
    dQ = numpy.int32(q)         # gpu q 

    hM = len(pattern)           # length of the input pattern
    hN = len(text)              # length of the input text 

    dM = numpy.int32(hM)        # set for the gpu
    dN = numpy.int32(hN)        # set for the gpu 

    start = cuda.Event()
    end = cuda.Event()

    start.record()

    h = 1                       # precompute hash of the pattern[1...m] using Horner's rule
    for i in range(0, hM-1):    # faster to mod by q after each of the m-2 multiplications
        h = (h * d) % q         # then to do h = pow(d, m-1) % q 

    print "h = %d" % h 

    dHash = numpy.int32(h)      # set for the gpu 
                                                                            # each block has 512 threads      
    NUM_BLOCKS = numpy.int(math.ceil(len(text) / (1.0 * 512000)))           # where each thread is evenly distributed over 
                                                                            # a chunk size of 512000 characters of the text 

    function = mod.get_function("search")           # prepare the function call 
    function(dText, dPattern, dHash, dQ, dD, dResults, dM, dN, numpy.int32(NUM_BLOCKS), block = (THREADS_PER_BLOCK, 1, 1), grid = (NUM_BLOCKS, 1, 1))
    
    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3

    print "EXITED KERNEL"
    hResults = dResults.get()            # get the results back from the GPU
    print "Is Shakespeare? "
    print dPattern.get()
    
    results = []
    for i in range(len(hResults)):
        if hResults[i] == 1:
            results.append(i)


    print "Time to Run: %fs" % secs
    print "Matchest found at index: %s" % results

if __name__ == "__main__":

    print "Test Case 1:"
    RabinKarp("adebc defde", "de", 256, 11)
    
    print "Test Case 2:"
    RabinKarp("3141592653589793", "26", 256, 11)

    print "Test Case 3:"
    RabinKarp("a", "a", 256, 11)

    print "Test Case 4:"
    with open("/home/wl2411/nltk_data/corpora/gutenberg/shakespeare-macbeth.txt", "r") as file2:
        data = file2.read().replace('\n', '')
    data = data * 6
    RabinKarp(data, "Shake", 256, 3355439)

    files = nltk.corpus.gutenberg.fileids()
    raw_text = ""  
    for file in files:
        filename = "/home/wl2411/nltk_data/corpora/gutenberg/" + str(file)        
        f = open(filename, 'r')
        raw_text += f.read().replace('\n', '')

    print "Test Case 5:"
    RabinKarp(raw_text, "Shakespeare", 256, 3355439)

