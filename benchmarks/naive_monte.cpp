#include <cmath>
#include <iostream>
#include <random>
#include <benchmark/benchmark.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_monte_miser.h>
#include <gsl/gsl_monte_vegas.h>
#include <naive_monte_carlo_single_thread.hpp>

using std::abs;

static double s_exact = 1.3932039296856768591842462603255;

double
g (double *k, size_t dim, void *params)
{
  (void)(dim); /* avoid unused parameter warnings */
  (void)(params);
  double A = 1.0 / (M_PI * M_PI * M_PI);
  return A / (1.0 - cos (k[0]) * cos (k[1]) * cos (k[2]));
}

static void BM_FunctionCall(benchmark::State& state)
{
    double y;
    std::vector<double> x{3.2, 7.8, 9.6};
    while(state.KeepRunning())
    {
      benchmark::DoNotOptimize(y = g(x.data(), 3, nullptr));
    }
    std::ostream cnull(0);
    cnull << y;
}

static void BM_GSLRNG(benchmark::State& state)
{
    const gsl_rng_type *T;
    gsl_rng *r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    double y;
    while(state.KeepRunning())
    {
        benchmark::DoNotOptimize(y = gsl_rng_uniform_pos (r));
    }
    gsl_rng_free(r);
}

static void BM_GSLMontePlain(benchmark::State& state)
{
    double res, err;
    double xl[3] = { 0, 0, 0 };
    double xu[3] = { M_PI, M_PI, M_PI };
    const gsl_rng_type *T;
    gsl_rng *r;
    gsl_monte_function G = { &g, 3, 0 };
    size_t calls = 5000000;

    gsl_rng_env_setup();

    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    gsl_monte_plain_state *s = gsl_monte_plain_alloc (3);
    double y;
    while(state.KeepRunning())
    {
        gsl_monte_plain_integrate(&G, xl, xu, 3, calls, r, s,
                                   &res, &err);
        benchmark::DoNotOptimize(y = res);
    }
    gsl_monte_plain_free (s);
    gsl_rng_free(r);
    std::cout << "Final estimate: " << res << std::endl;
    std::cout << "Exact         : " << s_exact << std::endl;
    std::cout << "Error estimate: " << err << std::endl;
    std::cout << "Actual error  : " << abs(y - s_exact) << std::endl;
    std::cout << "Function calls: " << calls << std::endl;
}

static void BM_SingleThreadMontePlain(benchmark::State& state)
{
    auto g1 = [&](std::vector<double> const & x) {
      return g(const_cast<double*>(x.data()), x.size(), nullptr);
    };
    std::vector<std::pair<double, double>> bounds{{0, M_PI}, {0, M_PI}, {0, M_PI}};

    double y;
    while(state.KeepRunning())
    {
        // We shouldn't count the constructor in here, but otherwise the calls to integrate refine.
        naive_monte_carlo_single_thread<double, decltype(g1)> mc(g1, bounds, (double) 0.0031);
        auto task = mc.integrate();
        benchmark::DoNotOptimize(y = task.get());
          std::cout << "Function calls      : " << mc.calls() << std::endl;
    }
    /*std::cout << "Final value: " << y << std::endl;
    std::cout << "Exact      : " << s_exact << std::endl;
    std::cout << "Final error estimate: " << mc.current_error_estimate() << std::endl;
    std::cout << "Actual error        : " << abs(y - s_exact) << std::endl;
    std::cout << "Function calls      : " << mc.calls() << std::endl;*/
}

static void BM_DefaultRand(benchmark::State& state)
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    auto y =  gen();
    while(state.KeepRunning())
    {
        benchmark::DoNotOptimize(y = gen());
    }
    std::cout << "Min: " << gen.min() << " max " << gen.max() << std::endl;
    std::ostream cnull(0);
    cnull << y;
}

static void BM_DefaultRandScaled(benchmark::State& state)
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    double y;
    double inv_denom = (double) 1/((double) gen.max() + (double)1 );
    while(state.KeepRunning())
    {
        benchmark::DoNotOptimize(y = gen()*inv_denom);
    }
    std::cout << "Min: " << gen.min() << " max " << gen.max() << std::endl;
    std::ostream cnull(0);
    cnull << y;
}

static void BM_MersenneTwister(benchmark::State& state)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    auto y = gen();
    while(state.KeepRunning())
    {
        benchmark::DoNotOptimize(y = gen());
    }
    std::cout << "Min: " << gen.min() << " max " << gen.max() << std::endl;
    std::ostream cnull(0);
    cnull << y;
}

static void BM_MersenneTwister64(benchmark::State& state)
{
    std::random_device rd;
    std::mt19937_64 gen(rd());
    auto y = gen();
    while(state.KeepRunning())
    {
        benchmark::DoNotOptimize(y = gen());
    }
    std::cout << "Min: " << gen.min() << " max " << gen.max() << std::endl;
    std::ostream cnull(0);
    cnull << y;
}

static void BM_MinStdRand(benchmark::State& state)
{
    std::random_device rd;
    std::minstd_rand gen(rd());
    auto y = gen();
    while(state.KeepRunning())
    {
        benchmark::DoNotOptimize(y = gen());
    }
    std::cout << "Min: " << gen.min() << " max " << gen.max() << std::endl;
    std::ostream cnull(0);
    cnull << y;
}

static void BM_MinStdRand0(benchmark::State& state)
{
    std::random_device rd;
    std::minstd_rand0 gen(rd());
    auto y = gen();
    while(state.KeepRunning())
    {
        benchmark::DoNotOptimize(y = gen());
    }
    std::cout << "Min: " << gen.min() << " max " << gen.max() << std::endl;
    std::ostream cnull(0);
    cnull << y;
}

static void BM_Ranlux24(benchmark::State& state)
{
    std::random_device rd;
    std::ranlux24 gen(rd());
    auto y = gen();
    while(state.KeepRunning())
    {
        benchmark::DoNotOptimize(y = gen());
    }
    std::cout << "Min: " << gen.min() << " max " << gen.max() << std::endl;
    std::ostream cnull(0);
    cnull << y;
}

static void BM_Ranlux24Base(benchmark::State& state)
{
    std::random_device rd;
    std::ranlux24_base gen(rd());
    auto y = gen();
    while(state.KeepRunning())
    {
        benchmark::DoNotOptimize(y = gen());
    }
    std::cout << "Min: " << gen.min() << " max " << gen.max() << std::endl;
    std::ostream cnull(0);
    cnull << y;
}

static void BM_Ranlux48Base(benchmark::State& state)
{
    std::random_device rd;
    std::ranlux48_base gen(rd());
    auto y = gen();
    while(state.KeepRunning())
    {
        benchmark::DoNotOptimize(y = gen());
    }
    std::cout << "Min: " << gen.min() << " max " << gen.max() << std::endl;
    std::ostream cnull(0);
    cnull << y;
}

static void BM_Ranlux48(benchmark::State& state)
{
    std::random_device rd;
    std::ranlux48 gen(rd());
    auto y = gen();
    while(state.KeepRunning())
    {
        benchmark::DoNotOptimize(y = gen());
    }
    std::cout << "Min: " << gen.min() << " max " << gen.max() << std::endl;
    std::ostream cnull(0);
    cnull << y;
}

static void BM_KnuthB(benchmark::State& state)
{
    std::random_device rd;
    std::knuth_b gen(rd());
    auto y = gen();
    while(state.KeepRunning())
    {
        benchmark::DoNotOptimize(y = gen());
    }
    std::cout << "Min: " << gen.min() << " max " << gen.max() << std::endl;
    std::ostream cnull(0);
    cnull << y;
}

static void BM_UniformReal(benchmark::State& state)
{
    std::random_device rd;
    std::knuth_b gen(rd());
    std::uniform_real_distribution<double> dis(std::nextafter(0, std::numeric_limits<double>::max()), 1.0);
    double y;
    while(state.KeepRunning())
    {
        benchmark::DoNotOptimize(y = dis(gen));
    }
    std::ostream cnull(0);
    cnull << y;
}




//BENCHMARK(BM_GSLMontePlain);
//BENCHMARK(BM_SingleThreadMontePlain);
//BENCHMARK(BM_FunctionCall);
//BENCHMARK(BM_GSLRNG);
BENCHMARK(BM_DefaultRand);
BENCHMARK(BM_MersenneTwister);
BENCHMARK(BM_MersenneTwister64);
BENCHMARK(BM_MinStdRand);
BENCHMARK(BM_MinStdRand0);
BENCHMARK(BM_Ranlux24);
BENCHMARK(BM_Ranlux24Base);
BENCHMARK(BM_Ranlux48);
BENCHMARK(BM_Ranlux48Base);
BENCHMARK(BM_KnuthB);
BENCHMARK(BM_UniformReal);
BENCHMARK(BM_DefaultRandScaled);

BENCHMARK_MAIN();
