#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <thread>
#include <future>
#include <string>
#include <chrono>
#include <boost/math/quadrature/naive_monte_carlo.hpp>

using boost::math::quadrature::naive_monte_carlo;

void display_progress(double progress,
                      double error_estimate,
                      double current_estimate,
                      std::chrono::duration<double> estimated_time_to_completion)
{
    int barWidth = 70;

    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] "
              << int(progress * 100.0)
              << "%, E = "
              << std::setprecision(3)
              << error_estimate
              << ", time to completion: "
              << estimated_time_to_completion.count()
              << " seconds, estimate: "
              << std::setprecision(5)
              << current_estimate
              <<"\r";

    std::cout.flush();
}

int main()
{
    auto g = [](std::vector<float> const & x)->float
    {
      float r = x[0]*x[0]+x[1]*x[1];
      if (r <= 1) {
        return 4.0f;
      }
      return 0.0f;
    };

    std::vector<std::pair<float, float>> bounds{{0, 1.0f}, {0, 1.0f}};
    naive_monte_carlo<float, decltype(g)> mc(g, bounds, 0.0001);

    auto task = mc.integrate();
    std::cout << "Hit ctrl-c to cancel.\n";
    while (task.wait_for(std::chrono::seconds(5)) != std::future_status::ready)
    {
        display_progress(mc.progress(), mc.current_error_estimate(), mc.current_estimate(), mc.estimated_time_to_completion());
    }
    float y = task.get();
    display_progress(mc.progress(), mc.current_error_estimate(), mc.current_estimate(), mc.estimated_time_to_completion());
    std::cout << std::setprecision(std::numeric_limits<float>::digits10);
    std::cout << "\nFinal value: " << y << std::endl;
    std::cout << "Exact      : " << M_PI << std::endl;
    std::cout << "Final error estimate: " << mc.current_error_estimate() << std::endl;
    std::cout << "Actual error        : " << abs(y - M_PI) << std::endl;
    std::cout << "Function calls: " << mc.calls() << std::endl;

    std::cout << "If you turn off Kahan summation, we'll now go on a random walk: \n";
    mc.update_target_error(0.00001);
    task = mc.integrate();
    std::cout << "Hit ctrl-c to cancel.\n";
    while (task.wait_for(std::chrono::seconds(5)) != std::future_status::ready)
    {
        display_progress(mc.progress(),
                         mc.current_error_estimate(),
                         mc.current_estimate(),
                         mc.estimated_time_to_completion());
    }
    y = task.get();
    display_progress(mc.progress(),
                     mc.current_error_estimate(),
                     mc.current_estimate(),
                     mc.estimated_time_to_completion());
    std::cout << std::setprecision(std::numeric_limits<float>::digits10);
    std::cout << "\nFinal value: " << y << std::endl;
}
