# numc

Here's what I did in project 4:
Task1 Matrix functions in C:
	Just implemented it natrually.Implementing isn't hard, debug is. It is really not easy to fully consider all possible input. And the lesson i learnt is that we really need to read the spec thoroughly. Because of misreading the spec, I used to allocate space for the return matrix "result" and found out that I actually don't need to. I promise I will carefully read the spec next project.

Task2 Writing the setup file:
	This is relatively an easy one to implement, just two lines of code. However, it really need us to read online official documents and extract useful information from them. This would be very useful for future working environment. I'm grateful for being given the chance to practice this general skill.

Task3 Writing the Python-C interface:
	This is a long part. I need to read tons of documents online and learn a lot of things. Again, implementing is not that hard, tackling all the exception and errors is the hardest thing. By the time of my last submission, I still got two unpassed test. "Slice after matrix" and "Non-negative dimension", However, I've checked all the function that init a matrix in numc.c and allocate_matrix.c and allocate_matrix_ref.c but found no error.

Task4 Speeding up matrix operations:
	This is relatively the most interesting part of this project, which also required a lot of thinking. What I did is basically the following:
	1.using OpenMP: "#pragma omp parallel for" before several for loops.
	2.using SIMD operation for add, sub, mul, neg, abs.
	3.For mul, I wrote an extra transpose function to transpose the second operand, which will make the multiplication easier to simulate(in my brain) and easier to apply SIMD operation on it, also speeding up the performance.
	4.For abs, I thought of two ways to make it faster, one is using bit mask and the other is taknig max(x,-x),considering that the former one is harder to implement in SIMD, I used the latter one.
	5.Fro pow, I first try OpenMP, but ran into a lot of troubles, I then recalled the "repeated square algorithm" and the fact that matrix calculations obey the union law. I then changed the naive solution completely and turned it into a recursive method, which gives a speedup of 1840x(wow)! It is truly surprising and amazing.
	6.I've also done a lot of basic code adjustions like preload the value outside of the for loop ,so that it won'tkeep accessing the pointer every iteration.
	7.There is still a lot of things I unfortunately don't have enough time to implement, such as using OpenMP and SIMD in pow and blocking and unrolling. If I got too bored because of the corona virus I might try to implement it .(LOL)
	
	At last, big thanks to the GSIs for their efforts and fast response on piazza!
