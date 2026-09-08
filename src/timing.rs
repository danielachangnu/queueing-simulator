use crate::{simulate, Config, Dist, Policy};
use std::time::Instant;

/// Runs a fixed set of (arrival rate, policy) benchmark cases, each over 10 seeds,
/// and prints a 95%-ish confidence interval for wall-clock run time per case.
///
/// Intended for comparing how simulation cost scales across policies and
/// utilization levels; it does not report simulation results themselves, only
/// timing.
pub fn timing_vec() {
    let config = Config {
            debug: false,
            response_time_histogram: None,
        };

    let num_jobs = 10_000_000;
    let dist = Dist::Hyperexponential(0.5, 3.0, 0.8);
    let n_of_seeds = 10;

let params = [
    (0.1, Policy::FCFS), (0.99, Policy::FCFS), (0.4, Policy::FCFS),
    (0.1, Policy::LCFS), (0.99, Policy::LCFS), (0.4, Policy::LCFS),
    (0.1, Policy::Nudge(0.2, false)), (0.99, Policy::Nudge(0.2, false)), (0.4, Policy::Nudge(0.2, false)),
    (0.1, Policy::SRPT), (0.99, Policy::SRPT), (0.4, Policy::SRPT),
    (0.1, Policy::LAS), (0.99, Policy::LAS), (0.4, Policy::LAS)
];

    for param in params {
        let mut total_sum = 0.0;
        let mut total_sum_squared = 0.0;
        for seed in 0..n_of_seeds {
            let now = Instant::now();
            let _output = simulate(param.0, dist, param.1, num_jobs, seed, &config);
            let elapsed_val = now.elapsed();
            let elapsed_secs = elapsed_val.as_secs_f64();
            total_sum+=elapsed_secs;
            total_sum_squared+=elapsed_secs.powi(2);
        }

        let mean = total_sum / n_of_seeds as f64;
        let var = (total_sum_squared - (total_sum.powi(2)/n_of_seeds as f64))/((n_of_seeds-1) as f64);

        let ci_lower_bound = mean - 2.0*var.sqrt();
        let ci_upper_bound = mean + 2.0*var.sqrt();

        println!("{:?}, {}, {}", param, ci_lower_bound, ci_upper_bound);
    }
}