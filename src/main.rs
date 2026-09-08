use queuing_sim::{simulate, Config, Dist, Policy};

#[cfg(test)]
const EPSILON: f64 = 1e-8;

fn main() {
    let num_jobs = 10_000_000;
    let rho = 0.4;
    let seed = 10;
    let dist = Dist::Hyperexponential(0.5, 3.0, 0.8);
    let policies = vec![Policy::Nudge(0.2, false)];

    let choice = std::env::args()
        .nth(1)
        .unwrap_or("0".to_string())
        .parse()
        .unwrap_or(0);

    let config = match choice {
        1 => Config {
            debug: false,
            response_time_histogram: Some(0.0001),
        },
        2 => Config {
            debug: true,
            response_time_histogram: None,
        },
        3 => Config {
            debug: true,
            response_time_histogram: Some(0.0001),
        },
        _ => Config {
            debug: false,
            response_time_histogram: None,
        },
    };

    for policy in policies {
        let results = simulate(rho, dist, policy, num_jobs, seed, &config);

        if let Some(step) = config.response_time_histogram {
            assert_eq!(results.response_times.iter().sum::<usize>(), num_jobs);
            let cumulant: Vec<usize> = results
                .response_times
                .iter()
                .scan(num_jobs, |state, &count| {
                    *state -= count;
                    Some(*state)
                })
                .collect();

            let log_frequencies = cumulant
                .iter()
                .take((100.0 / step) as usize)
                .map(|&freq| (freq as f64 / num_jobs as f64).log10());

            println!(
                "{:?};{}",
                policy,
                log_frequencies
                    .map(|f| format!("{}", f))
                    .collect::<Vec<String>>()
                    .join(";")
            );
        }
        println!("{:?}, {}", policy, results.mean_response_time);
    }
}

#[cfg(test)]
mod mean_tests;
