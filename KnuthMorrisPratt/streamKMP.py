import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy
import math 
import nltk

THREADS_PER_BLOCK = 512
CHUNK_SIZE =  512000

def getLPS(pattern, M): 
	partialMatchTable = [0] * M 			# init the partial match table to be the size of the pattern
	length = 0								# length of the longest prefix suffix
											# same implementation as in the serial version 
	i = 1
	while i < M: 
		if (pattern[i] == pattern[length]):
			length += 1
			partialMatchTable[i] = length
			i += 1
		else: 
			if length != 0: 
				length = partialMatchTable[length - 1]
			else: 
				partialMatchTable[i] = 0
				i += 1
	return partialMatchTable

def KMP(pattern, text):
	# the kernel operates on chunks of data of size CHUNK_SIZE = 512000 where each block has 512 threads
	# so each thread operates on 1000 characters in the data set 
	mod = SourceModule(""" 
	# define THREADS_PER_BLOCK 512
	# define CHUNK_SIZE 512000
	__global__ void search(char *pattern, char *text, int *partialMatchTable, int M, int N, int *output, int chunkNum)
    {
    	int index = blockDim.x * blockIdx.x + threadIdx.x;

    	int chunk_len = CHUNK_SIZE / THREADS_PER_BLOCK;
        int start = index * chunk_len;
        int end = ( index + 1 ) * chunk_len;				// we do not offset by (M - 1) this time, like we did in the GPU version
        													// this is because when we allocated the chunk to send on the stream, M - 1 additional characters were already appended 
        end = min ( end, N );
        if ( end < start )
        {
            return; 
        }

        if ( end - start < M )
        {
            return;
        }

        int i = start; 
        int j = 0; 
        while ( i < end )
        {
        	if ( text[i] == pattern[j] )
        	{
        		i++;
        		j++; 
        	}
        	if ( j == M ) 
        	{
        		output[ (chunkNum * CHUNK_SIZE) + (i - j) ] = 1; 	// must offset the global location, we do this by taking what chunkNum it is and multiply it by the size of a chunk, CHUNK_SIZE
        		j = partialMatchTable[ j - 1 ]; 
        	}
        	else if ( ( i < end ) && ( text[ i ] != pattern[ j ] ) )
        	{
        		if ( j != 0 )
        		{
        			j = partialMatchTable[ j - 1 ]; 
        		}
        		else 
        		{
        			i++; 
        		}
        	}
        }

	}
	""")

	start = cuda.Event()		# create Cuda Events for timing
	end = cuda.Event()

	start.record()				# start the timer

	device_Text = gpuarray.to_gpu(numpy.array([text], dtype=str))         # set device memory for text + pattern
	device_Pattern = gpuarray.to_gpu(numpy.array([pattern], dtype=str))

	host_M = len(pattern)		# length of the pattern and text
	host_N = len(text)

	device_M = numpy.int32(host_M)        # set for the gpu
	device_N = numpy.int32(host_N)        # set for the gpu 

	host_partialMatchTable = getLPS(pattern, host_M)					# get the partial match table
	device_partialMatchTable = gpuarray.to_gpu(numpy.asarray(host_partialMatchTable, numpy.int32))

	host_indicesList = [0] * host_M 	  # create the output array 
	device_indicesList = gpuarray.zeros(host_N, dtype=numpy.int32) 

	streams = []						# create 2 streams 
	streams.append(cuda.Stream())
	streams.append(cuda.Stream())

	function = mod.get_function("search")

	chunkNum = 0 
	for i in range( 0, host_N, CHUNK_SIZE * 2 ): 			# iterate through the input text at CHUNK_SIZE pieces
		chunk1 = text[ i : i + CHUNK_SIZE + host_M - 1 ]	# must create an offset in the global index to track location, send to the kernel what chunk it is
		device_Chunk1 = gpuarray.to_gpu( numpy.array( [chunk1] , dtype=str ) )
		function(device_Pattern, device_Chunk1, device_partialMatchTable, device_M, numpy.int32(len(chunk1)), device_indicesList, numpy.int32( chunkNum ), block = (THREADS_PER_BLOCK, 1, 1), grid = (1, 1, 1), stream = streams[0])
		chunkNum += 1 

		chunk2 = text[ i + CHUNK_SIZE : i + (CHUNK_SIZE * 2) + host_M - 1 ]
		if ( len(chunk2) != 0 ):							# chunk2 can be zero, so if it's not 0 then call the kernel
			device_Chunk2 = gpuarray.to_gpu( numpy.array( [chunk2] , dtype=str ) )
			function(device_Pattern, device_Chunk2, device_partialMatchTable, device_M, numpy.int32(len(chunk2)), device_indicesList, numpy.int32( chunkNum ), block = (THREADS_PER_BLOCK, 1, 1), grid = (1, 1, 1), stream = streams[1])
			chunkNum += 1

	end.record()										# stop recording and get the time diff
	end.synchronize()
	secs = start.time_till(end) * 1e-3

	host_indicesList = device_indicesList.get()			# output results
	results = []
	for i in range(len(host_indicesList)):
		if host_indicesList[i] == 1:
			results.append(i)

	print "Matches Found At: %s" % results
	print "Time to Run: %fs" % secs


if __name__ == "__main__":
	print "Test Case 1: "
	text = "ABABDABACDABABCABAB"
	pattern = "ABABCABAB"
	KMP(pattern, text)

	print "Test Case 2: "
	text = "JENJEN"
	pattern = "EN"
	KMP(pattern, text)

	print "Test Case 3:"
	with open("/home/wl2411/nltk_data/corpora/gutenberg/shakespeare-macbeth.txt", "r") as file2:
		data = file2.read().replace('\n', '')
	data = data * 6
	KMP("Shakespeare ", data)

	print "Test Case 4: "
	files = nltk.corpus.gutenberg.fileids()
	raw_text = ""  
	for file in files:
		filename = "/home/wl2411/nltk_data/corpora/gutenberg/" + str(file)        
		f = open(filename, 'r')
		raw_text += f.read().replace('\n', '')
	KMP("The genius", raw_text)

	print "Test Case 5: "
	KMP("JEN", raw_text)

