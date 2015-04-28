import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy
import math 
import nltk 

CHUNK_SIZE = 500000                                
THREADS_PER_BLOCK = 500

def RabinKarp(text, pattern, d, q): 
    mod = SourceModule("""
    #define THREADS_PER_BLOCK 500
    __global__ void search(char *text, char *pattern, int h, int q, int d, int *output, int M, int N, int globalStreamLocation)
    {
        int index = blockDim.x * blockIdx.x + threadIdx.x;
        
        int chunk_len = 500000 / THREADS_PER_BLOCK;
        int start = index * chunk_len;
        int end = ( index + 1 ) * chunk_len;
        
        end = min( end, N );

        if (end < start)
        {
            return; 
        }

        if (end - start < M)
        {
            return;
        }

        int i, j; 
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
                    output[i + globalStreamLocation] = 1; 
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

    dPattern = gpuarray.to_gpu(numpy.array([pattern], dtype=str))   # set device memory for the pattern 
    dResults = gpuarray.zeros(len(text), dtype=numpy.int32)         # hold the resulting indices
                                                                    # at each index, 0/1 if there's a match or not 
    dD = numpy.int32(d)                                             # gpu d
    dQ = numpy.int32(q)                                             # gpu q 

    hM = len(pattern)                                               # length of the input pattern
    dM = numpy.int32(hM)                                            # set for the gpu

    start = cuda.Event()                                            # create Cuda Events
    end = cuda.Event()
    start.record()                                                  # start Recording

    h = 1                                                           # precompute hash of the pattern[1...m] using Horner's rule
    for i in range(0, hM-1):                                        # faster to mod by q after each of the m-2 multiplications
        h = (h * d) % q                                             # then to do h = pow(d, m-1) % q 
                                
    dHash = numpy.int32(h)                                          # set for the gpu 

    stream = []                                                     # create 2 streams 
    for i in range(2):
        stream.append(cuda.Stream())

    function = mod.get_function("search")                           # prepare the function call

    for i in range(0, len(text), CHUNK_SIZE * 2):                   # call the streams, each at a text size of CHUNK_SIZE
        textSubStr1 = text[i : (i + CHUNK_SIZE + len(pattern) - 1)]                           # set the first stream's input text 
        NUM_BLOCKS = numpy.int(math.ceil(len(textSubStr1) / (1.0 * 5000)))

        if NUM_BLOCKS == 0:
           NUM_BLOCKS = 1

        text_chunk1 = gpuarray.to_gpu(numpy.array( [textSubStr1], dtype=str ) )
        function(text_chunk1, dPattern, dHash, dQ, dD, dResults, dM, numpy.int32(len(textSubStr1)), numpy.int32(i), block = (THREADS_PER_BLOCK, 1, 1), grid = (NUM_BLOCKS, 1, 1), stream = stream[0])
        
        textSubStr2 = text[i + CHUNK_SIZE : i + (2 * CHUNK_SIZE) + len(pattern) - 1]            # set the second stream's input text         
        NUM_BLOCKS = numpy.int(math.ceil(len(textSubStr2) / (1.0 * 5000)))
        
        if NUM_BLOCKS == 0:
            NUM_BLOCKS = 1

        text_chunk2 = gpuarray.to_gpu(numpy.array( [textSubStr2], dtype=str) )
        function(text_chunk2, dPattern, dHash, dQ, dD, dResults, dM, numpy.int32(len(textSubStr2)), numpy.int32(i + CHUNK_SIZE), block = (THREADS_PER_BLOCK, 1, 1), grid = (NUM_BLOCKS, 1, 1), stream = stream[1])
    
    end.record()                                                    # call cuda Event
    end.synchronize()
    secs = start.time_till(end)*1e-3                                # get the time 

    hResults = dResults.get()                                       # get the results back from the GPU
    results = []
    for i in range(len(hResults)):
        if hResults[i] == 1:
            results.append(i)

    print "Time to Run: %fs" % secs                                 # print time and results 
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
    RabinKarp(data, "JEN", 256, 3355439)

    files = nltk.corpus.gutenberg.fileids()
    raw_text = ""  
    for file in files:
        filename = "/home/wl2411/nltk_data/corpora/gutenberg/" + str(file)        
        f = open(filename, 'r')
        raw_text += f.read().replace('\n', '')

    print "Test Case 5:"
    RabinKarp(raw_text, "Shakespeare", 256, 3355439)



