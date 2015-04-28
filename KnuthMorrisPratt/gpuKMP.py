import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy
import math 
import nltk

THREADS_PER_BLOCK = 512

def getLPS(pattern, M): 
	partialMatchTable = [0] * M 			# init the partial match table to be the size of the pattern
	length = 0								# length of the longest prefix suffix
											# same implementation as in CPU Version 
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
	mod = SourceModule(""" 
	# define THREADS_PER_BLOCK 512
	__global__ void search(char *pattern, char *text, int *partialMatchTable, int M, int N, int *output)
    {
    	int index = blockDim.x * blockIdx.x + threadIdx.x;			// get the global index

    	int chunk_len = 51200 / THREADS_PER_BLOCK;					// each thread operates over 51200/512 characters
        int start = index * chunk_len;								// offset the starting point in the global text string
        int end = ( index + 1 ) * chunk_len + ( M - 1 );			// offset the ending point, must add (M - 1) to the end for overlap

        end = min ( end, N );										// if the end is beyond the actual length of the input text, return
        if ( end < start )
        {
            return; 
        }

        if ( end - start < M )										// if there are not enough characters remaining to match the length of the pattern
        {															// also return
            return;														
        }

        int i = start;
        int j = 0; 
        while ( i < end )											// iterate over the entire string from start to end
        {
        	if ( text[i] == pattern[j] )							// found a match, increment both the counter in the partialTable and in the string text
        	{
        		i++;
        		j++; 
        	}
        	if ( j == M ) 											// if it's a complete match, return the global location index 
        	{
        		output[ i - j ] = 1; 
        		j = partialMatchTable[ j - 1 ]; 					// reset the partial table index j
        	}
        	else if ( ( i < end ) && ( text[ i ] != pattern[ j ] ) )	// if it wasn't a match
        	{
        		if ( j != 0 )										// get the size j jump we can make because we have a  partial match 
        		{
        			j = partialMatchTable[ j - 1 ]; 
        		}
        		else 
        		{
        			i++; 											// if there was no partial match, just increment by 1 in the text 
        		}
        	}
        }

	}
	""")

	start = cuda.Event()			# create cuda Events for timer
	end = cuda.Event()				

	start.record()

	device_Text = gpuarray.to_gpu(numpy.array([text], dtype=str))         # set device memory for text + pattern
	device_Pattern = gpuarray.to_gpu(numpy.array([pattern], dtype=str))

	host_M = len(pattern)			# set the host length of the pattern and text
	host_N = len(text)

	device_M = numpy.int32(host_M)	# set for the gpu
	device_N = numpy.int32(host_N)	# set for the gpu 

	host_partialMatchTable = getLPS(pattern, host_M)	# get the partialMatchTable
	device_partialMatchTable = gpuarray.to_gpu(numpy.asarray(host_partialMatchTable, numpy.int32))

	host_indicesList = [0] * host_M 					# create the output list
	device_indicesList = gpuarray.zeros(host_N, dtype=numpy.int32) 

	NUM_BLOCKS = numpy.int(math.ceil(host_N / (1.0 * 51200)))	# define the number of blocks to be of 51200 characters each
																# operated on by 512 threads each
	function = mod.get_function("search")
	function(device_Pattern, device_Text, device_partialMatchTable, device_M, device_N, device_indicesList, block = (THREADS_PER_BLOCK, 1, 1), grid = (NUM_BLOCKS, 1, 1))

	end.record()						# stop recording and get the time difference 
	end.synchronize()
	secs = start.time_till(end) * 1e-3

	host_indicesList = device_indicesList.get()			# get the matching indices 
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



