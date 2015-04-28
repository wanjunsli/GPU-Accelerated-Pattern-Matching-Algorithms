import sys
import time
import nltk 

def RabinKarp(text, pattern, d, q):
	results = []			# find all indices of pattern matches

	M = len(pattern)		# length of the pattern string
	N = len(text)			# length of the text 

	hashPattern = 0			# hash of the pattern
	hashText = 0			# hash of the text 

	start = time.time()

	h = 1 					# precompute hash of the pattern[1...m] using Horner's rule
	for i in range(0, M-1):
		h = (h * d) % q 	# faster to mod by q after each of the m-2 multiplications
							# then to do h = pow(d, m-1) % q 
	for i in range(0, M): 
		hashPattern = (d * hashPattern + ord(pattern[i])) % q 	# calculate the hash of the pattern
		hashText = (d * hashText + ord(text[i])) % q 			# calculate the hash of the current sliding window 
	
	for i in range(0, N - M + 1): 				# sliding window; move pattern over the text 
		if hashPattern == hashText:				# if the hashes match
			foundMatch = True
			for j in range(0, M):				# compare character by character in cases of hash collision
				if (text[i + j] != pattern[j]):	# false positive 
					foundMatch = False
					break 
			if foundMatch: 						# not a collision, add to list of indices 
				results.append(i)

		if i < N - M:	# calculate the new hash for the next set of M characters in the text 
			hashText = (d * (hashText - ord(text[i]) * h) + ord(text[i + M])) % q  	# remove the high order digit of the previous hash
																					# insert the low order digit of the next hash 
			if hashText < 0: 	# make sure the hash is within the range of [0, q) 
				hashText += q
	end = time.time()
	secs = end - start

	print "Time to Run: %fs" % secs
	print "Matches found at index: %s" % results



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
    RabinKarp(raw_text, "JEN", 256, 3355439)



	