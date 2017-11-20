/*
 * Copyright Nick Thompson, 2017
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */
#ifndef NAIVE_MONTE_CARLO_HPP
#define NAIVE_MONTE_CARLO_HPP
#include <vector>
#include <atomic>
#include <future>
#include <initializer_list>
#include <utility>
#include <random>
#include <chrono>

namespace boost { namespace math { namespace quadrature {

template<class Real, class F>
class naive_monte_carlo
{
public:
    naive_monte_carlo(const F& f, std::vector<std::pair<Real, Real>> const & bounds, Real error_goal): m_f{f}
    {
        size_t n = bounds.size();
        m_lbs.resize(n);
        m_dxs.resize(n);
        m_volume = 1;
        for (size_t i = 0; i < n; ++i)
        {
            if (!(std::isfinite(bounds[i].first) && std::isfinite(bounds[i].second)))
            {
                throw std::domain_error("The routine only support bounded domains. Rescaling infinite domains must be done by the user.\n");
            }
            if (bounds[i].second <= bounds[i].first)
            {
                throw std::domain_error("The upper bound is <= the lower bound.\n");
            }
            m_lbs[i] = bounds[i].first;
            m_dxs[i] = bounds[i].second - m_lbs[i];
            m_volume *= m_dxs[i];
        }
        m_error_goal = error_goal;
        m_start = std::chrono::system_clock::now();
        std::vector<Real> x(m_lbs.size());
        for (size_t i = 0; i < x.size(); ++i)
        {
            x[i] = m_lbs[i] + 0.5*m_dxs[i];
        }
        m_avg = f(x);
        m_k = 2;
        m_S = 0;
    }

    std::future<Real> integrate()
    {
        return std::async(std::launch::async, &naive_monte_carlo::m_integrate, this);
    }

    void cancel()
    {
        m_cancel = true;
    }

    Real current_error_estimate() const
    {
        size_t k = (Real) m_k.load();
        Real sigma = m_S.load()/(k*(k-1));
        return sqrt(sigma)*m_volume;
    }

    std::chrono::duration<Real> estimated_time_to_completion() const
    {
        auto now = std::chrono::system_clock::now();
        std::chrono::duration<Real> elapsed_seconds = now - m_start;
        Real r = this->current_error_estimate()/m_error_goal.load();
        return (r*r - 1)*elapsed_seconds;
    }

    void update_target_error(Real new_target_error)
    {
        m_error_goal = new_target_error;
    }

    Real progress() const
    {
        Real r = m_error_goal.load()/this->current_error_estimate();
        return r*r;
    }

    Real current_estimate() const
    {
        return m_avg.load()*m_volume;
    }

    size_t calls() const
    {
        return m_k.load();
    }

private:

    Real m_integrate()
    {
        m_start = std::chrono::system_clock::now();
        std::vector<Real> x(m_lbs.size());
        std::random_device rd;
        // mt19937_64 benchmarks at 9 ns/call.
        // default_random_engine benchmarks at 7 ns/call, but the period is much lower.
        // NR recommends a period of at least 2^64.
        std::mt19937_64 gen(rd());
        // I thought this was great, but in fact calling 'dis(gen)' is slow!
        //std::uniform_real_distribution<Real> dis(std::nextafter(0, std::numeric_limits<Real>::max()), 1.0);
        bool ok = gen.min() == 0 || gen.min() == 1;
        if (!ok)
        {
            throw std::logic_error("Generator does not obey gen.min() = 0 or 1.\n");
        }

        Real inv_denom = (Real) 1/( (Real) gen.max() + (Real) 2);
        Real M1 = m_avg.load();
        Real S = m_S.load();
        size_t k = m_k.load();
        Real compensator = 0;
        while(k < 2048 || this->current_error_estimate() > m_error_goal)
        {
            int j = 0;
            while (j++ < 1024)
            {
                for (size_t i = 0; i < m_lbs.size(); ++i)
                {
                    x[i] = m_lbs[i] + (gen()+1)*inv_denom*m_dxs[i];
                }
                Real f = m_f(x);
                Real term = (f-M1)/k;
                Real y1 = term - compensator;
                Real M2 = M1 + y1;
                compensator = (M2 - M1) - y1;
                ++k;
                S += (f - M1)*(f - M2);
                M1 = M2;
            }
            // Don't update the error estimate every call;
            // because atomic writes are expensive
            m_avg = M1;
            m_S = S;
            m_k = k;
            if (m_cancel)
            {
                return m_avg*m_volume;
            }
        }
        return m_avg*m_volume;
    }

    void thread_monte()
    {
        std::vector<Real> x(m_lbs.size());
        std::random_device rd;
        std::mt19937_64 gen(rd());
        Real inv_denom = (Real) 1/( (Real) gen.max() + (Real) 2);

        for (size_t i = 0; i < m_lbs.size(); ++i)
        {
            x[i] = m_lbs[i] + (gen()+1)*inv_denom*m_dxs[i];
        }
        Real M1 = f(x);
        Real S = 0;
        Real compensator = 0;
        int j = 0;
        size_t k = 2;
        while (j++ < 1024)
        {
            for (size_t i = 0; i < m_lbs.size(); ++i)
            {
                x[i] = m_lbs[i] + (gen()+1)*inv_denom*m_dxs[i];
            }
            Real f = m_f(x);
            Real term = (f-M1)/k;
            Real y1 = term - compensator;
            Real M2 = M1 + y1;
            compensator = (M2 - M1) - y1;
            ++k;
            S += (f - M1)*(f - M2);
            M1 = M2;
        }
    }

    std::vector<Real> m_lbs;
    std::vector<Real> m_dxs;
    std::atomic<size_t> m_k;
    Real m_volume;
    std::atomic<Real> m_error_goal;
    std::atomic<Real> m_S; // See Knuth section 4.2.2 for defition of S
    std::atomic<Real> m_avg;
    std::atomic<bool> m_cancel;
    std::chrono::time_point<std::chrono::system_clock> m_start;
    const F& m_f;

};
}}}
#endif
