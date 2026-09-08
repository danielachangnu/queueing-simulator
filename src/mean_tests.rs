use super::*;

#[test]
fn test_dist_mean() {
    let uniform = Dist::Uniform(2.0, 4.0);
    assert!((uniform.mean() - 3.0).abs() < EPSILON);

    let exp = Dist::Exponential(5.0);
    assert!((exp.mean() - 5.0).abs() < EPSILON);

    let hyper = Dist::Hyperexponential(1.0, 10.0, 0.8);
    assert!((hyper.mean() - 2.8).abs() < EPSILON);
}

#[test]
fn test_system_littles_law() {
    let lambda = 0.6;
    let config = Config {
        debug: false,
        response_time_histogram: None,
    };

    let mut results = simulate(
        lambda,
        Dist::Exponential(1.0),
        Policy::FCFS,
        10_000,
        123,
        &config,
    );

    // Call the methods to update the fields
    results.mean_response_time();
    results.mean_number_of_jobs(lambda);

    // Access the fields directly
    let mean_response = results.mean_response_time;
    let mean_jobs = results.mean_number_of_jobs;

    let expected_jobs = lambda * mean_response;

    assert!((mean_jobs - expected_jobs).abs() < EPSILON);
}

#[test]
fn test_system_littles_law2() {
    let theoretical_lambda = 0.6;
    let config = Config {
        debug: false,
        response_time_histogram: None,
    };

    let mut results = simulate(
        theoretical_lambda,
        Dist::Exponential(1.0),
        Policy::FCFS,
        50_000,
        123,
        &config,
    );

    // Call method to update, then access field
    results.mean_number_of_jobs(theoretical_lambda);
    let expected_l = results.mean_number_of_jobs;

    let actual_l = results.total_response_time / results.total_time;

    let margin = actual_l * 0.02;

    assert!((actual_l - expected_l).abs() < margin);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: u64 = 42;
    const JOB_SIZE: usize = 150_000;
    const TOLERANCE: f64 = 0.05;

    fn test_config() -> Config {
        Config {
            debug: false,
            response_time_histogram: None,
        }
    }

    fn run_sim(lambda: f64, dist: Dist, policy: Policy) -> f64 {
        let mut results = simulate(lambda, dist, policy, JOB_SIZE, SEED, &test_config());
        results.mean_response_time(); // Update the field
        results.mean_response_time // Return the field's value
    }

    #[test]
    fn test_mm1_exponential_c2_equals_1() {
        let lambda = 0.5; // rho = 0.5
        let dist = Dist::Exponential(1.0);

        let fcfs = run_sim(lambda, dist, Policy::FCFS);
        let lcfs = run_sim(lambda, dist, Policy::LCFS);
        let plcfs = run_sim(lambda, dist, Policy::PLCFS);
        let ps = run_sim(lambda, dist, Policy::PS);

        //wait time
        assert!((fcfs - 2.0).abs() < TOLERANCE);
        assert!((lcfs - fcfs).abs() < TOLERANCE);

        //response time
        assert!((ps - 2.0).abs() < TOLERANCE);
        assert!((plcfs - ps).abs() < TOLERANCE);
    }

    #[test]
    fn test_mg1_hyperexponential() {
        let lambda = 0.5;
        let dist = Dist::Hyperexponential(0.5, 3.0, 0.8);

        let fcfs = run_sim(lambda, dist, Policy::FCFS);
        let lcfs = run_sim(lambda, dist, Policy::LCFS);
        let plcfs = run_sim(lambda, dist, Policy::PLCFS);
        let ps = run_sim(lambda, dist, Policy::PS);

        //wait time
        assert!((fcfs - 3.0).abs() < TOLERANCE);
        assert!((lcfs - fcfs).abs() < TOLERANCE);

        //response time
        assert!((ps - 2.0).abs() < TOLERANCE);
        assert!((plcfs - ps).abs() < TOLERANCE);
    }

    #[test]
    fn test_mg1_uniform_low_variance() {
        let lambda = 0.5;
        let dist = Dist::Uniform(0.0, 2.0);

        let fcfs = run_sim(lambda, dist, Policy::FCFS);
        let lcfs = run_sim(lambda, dist, Policy::LCFS);

        //wait time
        assert!((fcfs - 1.666).abs() < TOLERANCE);
        assert!((lcfs - fcfs).abs() < TOLERANCE);
    }

    #[test]
    fn test_ps_slowdown() {
        let lambda = 0.5;
        let e_s = 1.0;
        let rho = lambda * e_s;

        let dist = Dist::Hyperexponential(0.5, 3.0, 0.8);
        let mut results = simulate(lambda, dist, Policy::PS, JOB_SIZE, SEED, &test_config());

        // Update then access field
        results.mean_slowdown();
        let sim_slowdown = results.mean_slowdown;

        let expected_slowdown = 1.0 / (1.0 - rho);

        assert!((sim_slowdown - expected_slowdown).abs() < TOLERANCE);
    }

    #[test]
    fn test_plcfs_slowdown() {
        let lambda = 0.5;
        let e_s = 1.0;
        let rho = lambda * e_s;

        let dist = Dist::Exponential(e_s);
        let mut results = simulate(lambda, dist, Policy::PLCFS, JOB_SIZE, SEED, &test_config());

        // Update then access field
        results.mean_slowdown();
        let sim_slowdown = results.mean_slowdown;

        let expected_slowdown = 1.0 / (1.0 - rho);

        assert!((sim_slowdown - expected_slowdown).abs() < TOLERANCE);
    }
}
