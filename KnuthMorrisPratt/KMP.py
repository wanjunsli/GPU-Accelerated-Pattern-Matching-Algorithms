import sys
import time
import nltk 

def getLPS(pattern, M): 
	partialMatchTable = [0] * M 			# init the partial match table to be the size of the pattern
	length = 0								# length of the longest prefix suffix

	i = 1
	while i < M: 							# for the entire length of the pattern, calculate the maximum proper prefix/suffix size 
		if (pattern[i] == pattern[length]): # found a match between prefix and suffix 
			length += 1
			partialMatchTable[i] = length 	# set the length 
			i += 1
		else: 
			if length != 0: 				# if there isn't a match, we take the previous max match found 
				length = partialMatchTable[length - 1]
			else: 
				partialMatchTable[i] = 0	# otherwise it's zero b/c there was no match between prefix and suffix 
				i += 1
	return partialMatchTable

def KMP(pattern, text):
	start = time.time()

	M = len(pattern)
	N = len(text)

	indicesList = []
	partialMatchTable = getLPS(pattern, M)

	i = 0		# tracks where we are in the text 
	j = 0		# tracks where we are in the pattern (i.e. how many matches have we seen so far)

	while i < N: 
		if text[i] == pattern[j]:						# match found, increment both where we are in the text and in the pattern
			j += 1
			i += 1
		if j == M: 										# if we found a complete match in the text with the pattern (i.e. j is the same length as our pattern)
			indicesList.append(i-j)						# then output the index where it occured (remember to subtract j out of the index i) -> since we incremented i as well
			j = partialMatchTable[j-1]					# we now need to move it back j steps to be at the correct index position 
														# and reset j using the partialMatchTable
		elif (i < N) and (text[i] != pattern[j]):		# we got a mismatch, get the partial_match_length
			if j != 0: 									# if partialMatchTable[partial_match_length] > 1
				j = partialMatchTable[j-1]				# then skip ahead our index by partial_match_length - partialMatchTable[partial_match_length - 1] 
			else:										# on our next iteration characters
				i += 1 								# we didn't see a match and j is 0, i.e. we're starting at the first character in the pattern
	
	end = time.time()

	print "Matches Found At: %s" % indicesList
	print "Time to Run: %fs" % (end - start)

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


