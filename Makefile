CXX := g++
CPPFLAGS := -g --std=c++14 -O3 -march=native -Wfatal-errors

all: benchmarks example float_destruction

benchmarks.x: benchmarks/naive_monte.cpp
	$(CXX) $(CPPFLAGS) -I./include $< -o $@ -lgsl -lgslcblas -lbenchmark

examples.x: example/naive_monte_carlo_example.cpp
	$(CXX) $(CPPFLAGS) -I./include $< -o $@

float_destruction.x: example/float_not_destroyed.cpp
	$(CXX) $(CPPFLAGS) -I./include $< -o $@

test.x: test/naive_monte_carlo_tests.cpp
	$(CXX) $(CPPFLAGS) -fsanitize=address -I./include $< -o $@

clean:
	rm -f a.out *.aux *.log *.gz *.out *.pdf *.x
