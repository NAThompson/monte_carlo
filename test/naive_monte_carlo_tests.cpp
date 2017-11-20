/*
 * Copyright Nick Thompson, 2017
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */
#define BOOST_TEST_MODULE naive_monte_carlo_test
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/math/quadrature/naive_monte_carlo.hpp>

using boost::math::quadrature::naive_monte_carlo;

template<class Real>
void test_pi()
{

    auto g = [](std::vector<Real> const & x)->Real
    {
        Real r = x[0]*x[0]+x[1]*x[1];
        if (r <= 1)
        {
            return 4;
        }
        return 0;
    };

    std::vector<std::pair<Real, Real>> bounds{{0, 1}, {0, 1}};
    naive_monte_carlo<Real, decltype(g)> mc(g, bounds, (Real) 0.0001);

    auto task = mc.integrate();
    Real pi_estimated = task.get();
    if (abs(pi_estimated - M_PI)/M_PI > 0.005)
    {
        std::cout << "Error in estimation of pi too high, function calls: " << mc.calls() << "\n";
        BOOST_CHECK_CLOSE_FRACTION(pi_estimated, M_PI, 0.005);
    }

}

template<class Real>
void test_constant()
{
    auto g = [](std::vector<Real> const & x)->Real
    {
      return 1;
    };

    std::vector<std::pair<Real, Real>> bounds{{0, 1}, {0, 1}};
    naive_monte_carlo<Real, decltype(g)> mc(g, bounds, (Real) 0.0001);

    auto task = mc.integrate();
    Real one = task.get();
    BOOST_CHECK_CLOSE_FRACTION(one, 1, 0.001);
    BOOST_CHECK_SMALL(mc.current_error_estimate(), std::numeric_limits<Real>::epsilon());
    BOOST_CHECK(mc.calls() > 1000);
}


BOOST_AUTO_TEST_CASE(naive_monte_carlo_test)
{
    test_pi<float>();
    test_pi<double>();
    test_pi<long double>();

    test_constant<float>();
    test_constant<double>();
    test_constant<long double>();
}
