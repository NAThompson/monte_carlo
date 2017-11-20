#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <thread>
#include <future>
#include <string>
#include <chrono>
#include <naive_monte_carlo_single_thread.hpp>

void display_progress(double progress, double error_estimate, double current_estimate, std::chrono::duration<double> estimated_time_to_completion)
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
              << "     \r";

    std::cout.flush();
}

int main()
{
    double exact = 1.3932039296856768591842462603255;
    double A = 1.0 / (M_PI * M_PI * M_PI);
    auto g = [&](std::vector<double> const & x)
    {
      return A / (1.0 - cos(x[0])*cos(x[1])*cos(x[2]));
    };
    std::vector<std::pair<double, double>> bounds{{0, M_PI}, {0, M_PI}, {0, M_PI}};
    naive_monte_carlo_single_thread<double, decltype(g)> mc(g, bounds, 0.0001);

    auto task = mc.integrate();

    //int s = 0;
    std::cout << "Hit ctrl-c to cancel.\n";
    while (task.wait_for(std::chrono::seconds(5)) != std::future_status::ready)
    {
        //std::cout << "\rError = " << mc.current_error_estimate() << ", Eta: " << mc.eta().count() << " seconds, progress = " << mc.progress()*100 << "%, current estimate: " << mc.current_estimate() << " calls: " << mc.calls() << std::flush;
        display_progress(mc.progress(), mc.current_error_estimate(), mc.current_estimate(), mc.estimated_time_to_completion());
        //++s;
        // TODO: The following shows that cancellation works, but it would be nice to show how it works with a ctrl-c signal handler.
        //if (s > 25){
        //  mc.cancel();
        //  std::cout << "\nCancelling because this is too slow!\n";
        //}
    }
    double y = task.get();
    std::cout << std::setprecision(std::numeric_limits<double>::digits10);
    std::cout << "\nFinal value: " << y << std::endl;
    std::cout << "Exact      : " << exact << std::endl;
    std::cout << "Final error estimate: " << mc.current_error_estimate() << std::endl;
    std::cout << "Actual error        : " << abs(y - exact) << std::endl;
    std::cout << "Function calls: " << mc.calls() << std::endl;
    std::cout << "Is this good enough? [y/N] ";
    bool goodenough = true;
    std::string line;
    std::getline(std::cin, line);
    if (line[0] != 'y')
    {
         goodenough = false;
    }
    double new_error = -1;
    if (!goodenough)
    {
        std::cout << "What is the new target error? ";
        std::getline(std::cin, line);
        new_error = atof(line.c_str());
    }
    if (new_error > 0)
    {
        mc.update_target_error(new_error);
        auto task = mc.integrate();
        std::cout << "Hit ctrl-c to cancel.\n";
        while (task.wait_for(std::chrono::seconds(5)) != std::future_status::ready)
        {
            display_progress(mc.progress(), mc.current_error_estimate(), mc.current_estimate(), mc.estimated_time_to_completion());
        }
        double y = task.get();
        std::cout << std::setprecision(std::numeric_limits<double>::digits10);
        std::cout << "\nFinal value: " << y << std::endl;
        std::cout << "Exact      : " << exact << std::endl;
        std::cout << "Final error estimate: " << mc.current_error_estimate() << std::endl;
        std::cout << "Actual error        : " << abs(y - exact) << std::endl;
        std::cout << "Function calls: " << mc.calls() << std::endl;
    }
}
