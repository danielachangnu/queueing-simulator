//! This is a queing simulation for a variety of M/G/1 scheduling policies, from simpler 
//! introductory ones such as First Come First Served, to research-level policies such as 
//! Nudge. 
//! 
//! It can be used as a standalone program, as well as a library. 
//! 
//! 
//! # Implemented Policies
//! - First Come, First Serve (FCFS)
//! - Last Come, First Serve (LCFS)
//! - Preemptive Last Come, First Serve (PLCFS)
//! - Processor sharing (PS)
//! - Shortest Remaining Processing Time (SRPT)
//! - Preemptive Shortest Job First (PSJF)
//! - Least Attained Service (LAS)
//! - Longest Remaining Processing Time (LRPT)
//! - [Nudge (Nudge(Threshold, false))](https://isaacg1.github.io/publications/#nudge)
//! - [GammaBoost (GammaB(gamma))](https://ziv.codes/yu_strongly_2024/)
//! - [AccumulatingPriority](https://link.springer.com/article/10.1007/s11134-013-9382-6) 
//! 
//! 
//! # Usage
//! The simulation parameters live at the top of src/main.rs and are edited in place.
//! 
//! ```rust
//! let num_jobs = 10_000_000;
//! let rho      = 0.4;
//! let seed     = 10;
//! let dist     = Dist::Hyperexponential(0.5, 3.0, 0.8);
//! let policies = vec![Policy::Nudge(0.2, false)];
//! ```
//! 
//! Once you set the parameters, you need to choose an output mode with an argument via the command line. 
//! 
//! | Argument | Debug stepping | Response time histogram |
//! |---|---|---|
//! | *(none)* or `0` | off | off |
//! | `1` | off | on |
//! | `2` | on | off |
//! | `3` | on | on |
//! 
//! For example,
//! 
//! ```rust
//! cargo run --release 1
//! ```
//! 
//! would run the simulation with the debug configuration off and the response time histogram on.
//! 
//! Additionally, you can also call the simulate function from the library on its own. You must create a Config object to represent whether or not you would like the debug configuration and the response time histogram, and pass the parameters into the function. 
//! 
//! ```rust
//! let config = Config {
//!     response_time_histogram: None,
//!     debug: false,
//! };
//! 
//! let results = simulate(
//!     0.4,                                    
//!     Dist::Hyperexponential(0.5, 3.0, 0.8),  
//!     Policy::SRPT,                           
//!     10_000_000,                             
//!     10,                                     
//!     &config,
//! );
//! ```
//! 
//! # Results
//! 
//! The Results struct output by the simulate function stores the total response time, queuing time, service time, and slowdown. Additionally, it also calculates the mean response time, the mean queue time, the mean slowdown, and the mean number of jobs. 
//! 
//! # Extending the package
//! 
//! You can add your own index-based policy, where there is no mid-job preemption, by using IndexPolicy implementation:
//! 
//! ```rust
//! pub struct FirstComeFirstServe;
//! 
//! impl IndexPolicy for FirstComeFirstServe {
//!     fn index(&self, job: &Job) -> f64 {
//!         job.arrival_time()
//!     }
//! }
//! 
//! let results = simulate(
//!     0.4,                                    
//!     Dist::Hyperexponential(0.5, 3.0, 0.8),  
//!     FirstComeFirstServe,                           
//!     10_000_000,                             
//!     10,                                     
//!     &config,
//! );
//! ```

/// A discrete-event simulator for M/G/1 queueing systems.
///
/// `queuing-sim` simulates jobs arriving at a single server according to a Poisson
/// arrival process and being served under a chosen scheduling policy. Running a
/// simulation returns a Result with response time, queueing
/// time, and slowdown statistics gathered over the run.

/// Extending with a custom policy
///
/// Index-ordered policies (jobs run in order of a priority value you define, with
/// preemption only at arrival instants, never mid-service) can be added with IndexPolicy

/// Timing/benchmarking helpers for comparing policy performance across seeds.
pub mod timing;

use noisy_float::prelude::*;
use rand::prelude::*;
use rand_distr::Exp;
use std::collections::VecDeque;

const EPSILON: f64 = 1e-8;

/// A single job moving through the simulated system.
///
/// The Job struct keeps track of a Job's original size, arrival time, and remaining size after it has been worked on. 
pub struct Job {
    rem_size: f64,
    arrival_time: f64,
    original_size: f64,
}

/// Configuration for a single call to simulate. 
pub struct Config {
    /// When Some(step), buckets each completed job's response time into a
    /// histogram with bucket width step, accumulated in
    /// Results::response_times. None disables this feature.
    pub response_time_histogram: Option<f64>,
    /// When true, before the next event, the engine prints the current time and the next arrival time. 
    /// Must press enter on stdin to continue the simulation process. Leave false for normal simulation
    /// without the step through. 
    pub debug: bool,
}

/// Built in M/G/1 scheduling policy.
///
/// Pass a Policy to simulate to select how jobs are ordered and preempted. To
/// define a custom index-based priority policy instead, see IndexPolicy.
#[derive(Debug, Copy, Clone)]
pub enum Policy {
    /// First Come, First Served: jobs run in arrival order.
    FCFS,
    /// Preemptive Last Come, First Served: each arriving job immediately preempts
    /// whichever job is currently in service.
    PLCFS,
    /// Last Come, First Served: each new arrival jumps to the
    /// front of the waiting queue, ahead of jobs that arrived earlier and are still
    /// waiting.
    LCFS,
    /// Shortest Remaining Processing Time: always serves the job with the least
    /// remaining work, interrupting the running job whenever a shorter one arrives.
    SRPT,
    /// Preemptive Shortest Job First: always serves the job with the smallest original size,
    /// interrupting the running job whenever a shorter one arrives.
    PSJF,
    /// Processor Sharing: every job present in the system receives an equal share
    /// of service capacity.
    PS,
    /// Least Attained Service: prioritizes whichever job has received the least
    /// total service so far.
    LAS,
    /// Longest Remaining Processing Time: always serves the job with the most remaining work.
    LRPT,
    /// If a job arrives at the end of the queue and it is smaller than the threshold, 
    /// first check whether or not the  job right before it is greater than or equal to a threshold. If it is,
    /// check whether or not the job has been nudged before. If it hasn't switch the ultimate and penultimate jobs.
    /// 
    /// Otherwise, acts like a normal FCFS queue. 
    /// 
    /// The first field is the size threshold; the second is internal state the
    /// policy uses to track whether it just nudged a job, and should always start
    /// out `false` (e.g. Policy::Nudge(0.2, false), if the threshold is 0.2).
    Nudge(f64, bool),
    /// Use the Boost function, which calculates a job's index by calculating a
    /// virtual arrival time and sorting by it. The virtual arrival time is based on a job's arrival time
    /// as well as a Boost value based on a job's original size and a value Gamma. 
    /// Jobs that are smaller and arrived earlier are prioritized under this policy. 
    GammaB(f64),
    /// Calculate priority continuously based on (wait time/original size), and sort on that.
    /// Similarly to GammaB, jobs that arived earlier are prioritized. The greater the priority, 
    /// the earlier it gets served. 
    AccumulatingPriority,
}

/// Policy implements the GenericPolicy trait for all the built-in policies
pub trait GenericPolicy {
    /// Returns the fraction of server capacity each running job receives right now
    /// (For example, for FCFS, since we run one job at a time, it is 1.0)
    fn work(&mut self, running: &mut Vec<Job>) -> f64;

    /// Called when a new job arrives in the simulation. This function is in charge of 
    /// placing the new_job into either the running or waiting queue depending on policy specific
    /// guidelines and re-sorting and organizing queues.
    fn arrival(&mut self, waiting: &mut VecDeque<Job>, running: &mut Vec<Job>, new_job: Job);

    /// Called when a job has finished and needs to be removed from the running queue at a given time.  
    /// It also moves job(s) from waiting to running according to the policy.
    fn completion(&mut self, waiting: &mut VecDeque<Job>, running: &mut Vec<Job>, time: f64);

    /// For policies with preemption, the function calculates how much time must elapse 
    /// for preemption (e.g. a job in waiting overtakes a job in running) to occur from the given 
    /// time at the given work rate. If a policy isn't preemptive, it returns infinity. 
    fn time_to_preemption(
        &mut self,
        waiting: &VecDeque<Job>,
        running: &[Job],
        work_rate: f64,
        time: f64,
    ) -> f64;

    /// If the time to preemption is smaller than the time until a job completes, we must handle
    /// preemption. It switches the job that is running with the job that preempts it in the 
    /// running/waiting queues.
    fn handle_preemption(&mut self, waiting: &mut VecDeque<Job>, running: &mut Vec<Job>, time: f64);
}


/// IndexPolicy allows the implementation of any policy that can be defined by assigning
/// jobs priority values and don't preempt mid service. 
/// 
/// Upon the arrival of a new job, if the new job has a lower priority value, it preempts the 
/// currently running job. Otherwise, it joins the waiting queue, which is sorted according
/// to priority values. 
pub trait IndexPolicy {
    /// Returns job's priority. A lower value means higher priority.
    fn index(&self, job: &Job) -> f64;
}

impl<I: IndexPolicy> GenericPolicy for I {
    fn work(&mut self, running: &mut Vec<Job>) -> f64 {
        if running.is_empty() {
            0.0
        } else {
            1.0
        }
    }
    fn arrival(&mut self, waiting: &mut VecDeque<Job>, running: &mut Vec<Job>, new_job: Job) {
        if running.is_empty() {
            running.push(new_job);
        } else {
            let current_idx = self.index(&running[0]);
            let new_idx = self.index(&new_job);

            if new_idx < current_idx {
                waiting.push_back(running.remove(0));
                running.push(new_job);
            } else {
                waiting.push_back(new_job);
            }

            waiting.make_contiguous().sort_by_key(|job| n64(self.index(job)));
        }
    }
    fn completion(&mut self, waiting: &mut VecDeque<Job>, running: &mut Vec<Job>, _time: f64) {
        if let Some(job) = waiting.pop_front() {
            running.push(job);
        }
    }

    fn time_to_preemption(
        &mut self,
        _waiting: &VecDeque<Job>,
        _running: &[Job],
        _work_rate: f64,
        _time: f64,
    ) -> f64 {
        f64::INFINITY
    }
    fn handle_preemption(&mut self, _waiting: &mut VecDeque<Job>, _running: &mut Vec<Job>, _time: f64) {}
}

impl GenericPolicy for Policy {
    fn work(&mut self, running: &mut Vec<Job>) -> f64 {
        match self {
            Policy::PS | Policy::LRPT | Policy::LAS => {
                if running.is_empty() {
                    0.0
                } else {
                    1.0 / running.len() as f64
                }
            }

            _ => {
                if running.is_empty() {
                    0.0
                } else {
                    1.0
                }
            }
        }
    }

    fn arrival(&mut self, waiting: &mut VecDeque<Job>, running: &mut Vec<Job>, new_job: Job) {
        match self {
            Policy::FCFS => {
                if running.is_empty() {
                    running.push(new_job);
                } else {
                    waiting.push_back(new_job);
                }
            }

            Policy::PLCFS => {
                if !running.is_empty() {
                    waiting.push_front(running.remove(0));
                }
                running.push(new_job);
            }

            Policy::LCFS => {
                if running.is_empty() {
                    running.push(new_job);
                } else {
                    waiting.push_front(new_job);
                }
            }

            Policy::SRPT => {
                if running.is_empty() {
                    running.push(new_job);
                } else if new_job.rem_size < running[0].rem_size {
                    waiting.push_back(running.remove(0));
                    waiting.make_contiguous().sort_by_key(|job| n64(job.rem_size));
                    running.push(new_job);
                } else {
                    waiting.push_back(new_job);
                    waiting.make_contiguous().sort_by_key(|job| n64(job.rem_size));
                }
            }

            Policy::PSJF => {
                if running.is_empty() {
                    running.push(new_job);
                } else if new_job.original_size < running[0].original_size {
                    waiting.push_back(running.remove(0));
                    waiting.make_contiguous().sort_by_key(|job| n64(job.original_size));
                    running.push(new_job);
                } else {
                    waiting.push_back(new_job);
                    waiting.make_contiguous().sort_by_key(|job| n64(job.original_size));
                }
            }

            Policy::PS => {
                running.push(new_job);
            }

            Policy::LAS => {
                if running.is_empty() {
                    running.push(new_job);
                } else {
                    let current_attained = running[0].original_size - running[0].rem_size;
                    let new_attained = 0.0;
                    if new_attained < current_attained {
                        waiting.extend(running.drain(..));
                        running.push(new_job);
                        waiting.make_contiguous().sort_by_key(|job| n64(job.original_size - job.rem_size));
                    } else {
                        running.push(new_job);
                    }
                }
            }

            Policy::LRPT => {
                if running.is_empty() {
                    running.push(new_job);
                } else {
                    let current_rem = running[0].rem_size;
                    if new_job.rem_size > current_rem {
                        waiting.extend(running.drain(..));
                        running.push(new_job);
                        waiting.make_contiguous().sort_by_key(|job| -n64(job.rem_size)); 
                    } else if new_job.rem_size == current_rem {
                        running.push(new_job);
                    } else {
                        waiting.push_back(new_job);
                        waiting.make_contiguous().sort_by_key(|job| -n64(job.rem_size));
                    }
                }
            }

            Policy::Nudge(threshold, just_swapped) => {
                if running.is_empty() {
                    running.push(new_job);
                } else {
                    if !*just_swapped {
                        if new_job.original_size < *threshold {
                            if let Some(back_job) = waiting.back() {
                                if back_job.original_size >= *threshold {
                                    *just_swapped = true;
                                    waiting.insert(waiting.len() - 1, new_job);
                                } else {
                                    waiting.push_back(new_job);
                                }
                            } else {
                                waiting.push_back(new_job);
                            }
                        } else {
                            waiting.push_back(new_job);
                        }
                    } else {
                        *just_swapped = false;
                        waiting.push_back(new_job);
                    }
                }
            }

            Policy::GammaB(gamma) => {
                if running.is_empty() {
                    running.push(new_job);
                } else {
                    waiting.push_back(new_job);
                    waiting.make_contiguous().sort_by_key(|job| {
                        let s = job.original_size;
                        let boost = (1.0 / *gamma) * (1.0 / (1.0 - (-*gamma * s).exp())).ln();
                        let virtual_arrival_time = job.arrival_time - boost;
                        n64(virtual_arrival_time)
                    });
                }
            }

            Policy::AccumulatingPriority => {
                if running.is_empty() {
                    running.push(new_job);
                } else {
                    waiting.push_back(new_job);
                }
            }
        }
    }

    fn completion(&mut self, waiting: &mut VecDeque<Job>, running: &mut Vec<Job>, time: f64) {
        match self {
            Policy::PS => (),
            Policy::LAS => {
                if running.is_empty() && !waiting.is_empty() {
                    let first = waiting.pop_front().unwrap();
                    let target_attained = first.original_size - first.rem_size;
                    running.push(first);

                    while !waiting.is_empty() {
                        let next_attained = waiting[0].original_size - waiting[0].rem_size;
                        if (next_attained - target_attained).abs() <= EPSILON {
                            running.push(waiting.pop_front().unwrap());
                        } else {
                            break;
                        }
                    }
                }
            }

            Policy::LRPT => {
                if running.is_empty() && !waiting.is_empty() {
                    let first = waiting.pop_front().unwrap();
                    let target_rem = first.rem_size;
                    running.push(first);

                    while !waiting.is_empty() {
                        if (waiting[0].rem_size - target_rem).abs() <= EPSILON {
                            running.push(waiting.pop_front().unwrap());
                        } else {
                            break;
                        }
                    }
                }
            }

            Policy::AccumulatingPriority => {
                if running.is_empty() && !waiting.is_empty() {
                    let mut best_priority_index = 0;
                    let mut best_priority_val = 0.0;
                    let mut i = 0;

                    while i < waiting.len() {
                        let curr_job = &waiting[i];
                        let curr_priority = (time - curr_job.arrival_time) / curr_job.original_size;

                        if curr_priority > best_priority_val {
                            best_priority_val = curr_priority;
                            best_priority_index = i;
                        }
                        i += 1;
                    }
                    
                    running.push(waiting.remove(best_priority_index).unwrap());
                }
            }

            Policy::Nudge(_, just_swapped) => {
                if !waiting.is_empty() {
                    running.push(waiting.pop_front().unwrap());
                }

                if waiting.is_empty() {
                    *just_swapped = false;
                }
            }

            _ => {
                if !waiting.is_empty() {
                    running.push(waiting.pop_front().unwrap());
                }
            }
        }
    }

    fn time_to_preemption(
        &mut self,
        waiting: &VecDeque<Job>,
        running: &[Job],
        work_rate: f64,
        time: f64,
    ) -> f64 {
        match self {
            Policy::LAS if !running.is_empty() && !waiting.is_empty() => {
                let current_attained = running[0].original_size - running[0].rem_size;
                let next_attained = waiting[0].original_size - waiting[0].rem_size;

                ((next_attained - current_attained) / work_rate).max(0.0)
            }
            Policy::LRPT if !running.is_empty() && !waiting.is_empty() => {
                let current_rem = running[0].rem_size;
                let next_rem = waiting[0].rem_size;

                ((current_rem - next_rem) / work_rate).max(0.0)
            }

            Policy::AccumulatingPriority if !running.is_empty() && !waiting.is_empty() => {
                let curr_job = &running[0];
                let curr_priority = (time - curr_job.arrival_time) / curr_job.original_size;

                let mut min_overtake_time = f64::INFINITY;

                for job in waiting {
                    if job.original_size < curr_job.original_size {
                        let job_priority = (time - job.arrival_time) / job.original_size;
                        let net_closing_speed =
                            1.0 / job.original_size - 1.0 / curr_job.original_size;
                        let behind_in_ratio = curr_priority - job_priority;
                        let overtake_time = behind_in_ratio / net_closing_speed;

                        if overtake_time < min_overtake_time {
                            min_overtake_time = overtake_time;
                        }
                    }
                }
                min_overtake_time
            }
            _ => f64::INFINITY,
        }
    }

    fn handle_preemption(&mut self, waiting: &mut VecDeque<Job>, running: &mut Vec<Job>, time: f64) {
        if waiting.is_empty() || running.is_empty() {
            return;
        }

        match self {
            Policy::LAS => {
                let current_attained = running[0].original_size - running[0].rem_size;

                while !waiting.is_empty() {
                    let next_attained = waiting[0].original_size - waiting[0].rem_size;
                    if (next_attained - current_attained).abs() <= EPSILON {
                        running.push(waiting.pop_front().unwrap());
                    } else {
                        break;
                    }
                }
            }
            Policy::LRPT => {
                let current_rem = running[0].rem_size;

                while !waiting.is_empty() {
                    if (waiting[0].rem_size - current_rem).abs() <= EPSILON {
                        running.push(waiting.pop_front().unwrap());
                    } else {
                        break;
                    }
                }
            }

            Policy::AccumulatingPriority => {
                let curr_job = &running[0];
                let curr_priority = (time - curr_job.arrival_time) / curr_job.original_size;
                let mut best_priority_index = 0;
                let mut best_priority_val = 0.0;
                let mut i = 0;

                while i < waiting.len() {
                    let job = &waiting[i];
                    let job_priority = (time - job.arrival_time) / job.original_size;

                    if job_priority > best_priority_val {
                        best_priority_val = job_priority;
                        best_priority_index = i;
                    }
                    i += 1;
                }

                if best_priority_val > curr_priority {
                    waiting.push_back(running.remove(0));
                    running.push(waiting.remove(best_priority_index).unwrap());
                }
            }
            _ => (),
        }
    }
}

/// The Results struct stores statistics from a full simulation.
///
/// total_response_time, total_queue time, total_service_time, total_slowdown, total_time, and num_jobs 
/// are stored during the simulation run itself. all other fields are calculated 
/// before returning. 
pub struct Results {
    /// If the configuration for the response time histogram is not None, the Histogram of completed jobs' 
    /// response times is stored.
    pub response_times: Vec<usize>,
    step: Option<f64>,
    /// Sum of (completion time − arrival time) over all completed jobs.
    pub total_response_time: f64,
    /// Sum of (current time - job arrival time) over all completed jobs. 
    pub total_queue_time: f64,
    /// Number of jobs the simulation ran (the `num_jobs` argument to [`simulate`]).
    pub num_jobs: usize,
    /// Sum of job sizes over all completed jobs.
    pub total_service_time: f64,
    /// Sum of (response time / service time) over all completed jobs.
    pub total_slowdown: f64,
    /// total_response_time / num_jobs
    pub mean_response_time: f64,
    /// ttal_queue_time / num_jobs
    pub mean_queue_time: f64,
    /// total_service_time / num_jobs
    pub mean_service_time: f64,
    /// Mean number of jobs in the system, by Little's Law (lambda *
    /// mean_response_time)
    pub mean_number_of_jobs: f64,
    /// total_slowdown / num_jobs
    pub mean_slowdown: f64,
    /// total simulated time elapsed over the run.
    pub total_time: f64,
}

impl Results {
    /// Fills in mean_response_time
    pub fn mean_response_time(&mut self) {
        self.mean_response_time = self.total_response_time / (self.num_jobs as f64);
    }

    /// Fills in mean_queue_time
    pub fn mean_queue_time(&mut self) {
        self.mean_queue_time = self.total_queue_time / (self.num_jobs as f64);
    }

    /// Fills in mean_service_time
    pub fn mean_service_time(&mut self) {
        self.mean_service_time = self.total_service_time / (self.num_jobs as f64);
    }

    /// Fills in mean_number_of_jobs via Little's law
    pub fn mean_number_of_jobs(&mut self, lambda: f64) {
        self.mean_number_of_jobs = lambda * self.mean_response_time;
    }

    /// Fills in mean_slowdown
    pub fn mean_slowdown(&mut self) {
        self.mean_slowdown = self.total_slowdown / (self.num_jobs as f64);
    }

    /// Records one completed job's response time into Results::response_times
    /// for each completion when histogram collection is enabled
    pub fn update_response_time_histogram(&mut self, response: f64) {
        if let Some(step) = self.step {
            let response_index = (response / step) as usize;
            while self.response_times.len() <= response_index {
                self.response_times.push(0);
            }
            self.response_times[response_index] += 1;
        }
    }
}

/// Runs an M/G/1 queueing simulation until num_jobs jobs have completed.
///
/// Jobs arrive as a Poisson process with rate lambda, and job sizes are drawn
/// from the distributions in dist. 
///
/// policy selects the specific scheduling policies that the simulation will be ran for. 
/// simulate can be ran with one policy, or with a variety policies at once.
///
/// seed seeds the random number generator, so a given simulatiion can be reproduceable across different runs
pub fn simulate<P: GenericPolicy>(
    lambda: f64,
    dist: Dist,
    mut policy: P,
    num_jobs: usize,
    seed: u64,
    config: &Config,
) -> Results {
    assert!((dist.mean() - 1.0).abs() < EPSILON); 
    let mut rng = StdRng::seed_from_u64(seed);
    let mut time = 0.0; 
    let arrival_dist = Exp::new(lambda).unwrap();
    let mut next_arrival = rng.sample(arrival_dist);
    let mut num_completions = 0;
    let mut num_arrivals = 0;
    let mut jobs_on_arrival = 0;
    
    let mut jobs_waiting: VecDeque<Job> = VecDeque::new();
    let mut jobs_in_progress: Vec<Job> = vec![];
    
    let mut results = Results {
        step: config.response_time_histogram,
        response_times: vec![],
        total_response_time: 0.0,
        total_queue_time: 0.0,
        num_jobs,
        total_service_time: 0.0,
        mean_response_time: 0.0,
        total_slowdown: 0.0,
        mean_queue_time: 0.0,
        mean_service_time: 0.0,
        mean_number_of_jobs: 0.0,
        mean_slowdown: 0.0,
        total_time: 0.0,
    };

    if config.debug {
        println!("lambda: {lambda}");
    }

    while num_completions < num_jobs {
        let work_rate = policy.work(&mut jobs_in_progress);
        if config.debug {
            println!("time: {time} ");
            println!("next arrival time: {next_arrival}");
            std::io::stdin()
                .read_line(&mut String::new())
                .expect("continued");
        }

        let mut time_to_completion = f64::INFINITY;
        for job in &jobs_in_progress {
            let time_for_this_job = job.rem_size / work_rate;

            if time_for_this_job < time_to_completion {
                time_to_completion = time_for_this_job;
            }
        }
        let time_to_preemption =
            policy.time_to_preemption(&jobs_waiting, &jobs_in_progress, work_rate, time);

        let next_event_diff = (next_arrival - time)
            .min(time_to_completion)
            .min(time_to_preemption);
        let was_arrival = next_event_diff == (next_arrival - time);

        let was_preemption = next_event_diff == time_to_preemption
            && !was_arrival
            && next_event_diff < time_to_completion;

        time += next_event_diff;
        for job in &mut jobs_in_progress {
            job.rem_size -= next_event_diff * work_rate;
            assert!(job.rem_size >= -EPSILON);
        }

        let mut i = 0;
        while i < jobs_in_progress.len() {
            if jobs_in_progress[i].rem_size <= EPSILON {
                let job = jobs_in_progress.remove(i);

                let response = time - job.arrival_time;
                let service = job.original_size;
                let wait_time = response - service;
                let slowdown = response / service;

                results.total_response_time += response;
                results.total_service_time += service;
                results.total_queue_time += wait_time;
                results.total_slowdown += slowdown;

                if config.response_time_histogram.is_some() {
                    results.update_response_time_histogram(response);
                }
                num_completions += 1;

                policy.completion(&mut jobs_waiting, &mut jobs_in_progress, time);
            } else {
                i += 1;
            }
        }

        if was_preemption {
            policy.handle_preemption(&mut jobs_waiting, &mut jobs_in_progress, time);
        }

        if was_arrival {
            let size = dist.sample(&mut rng);
            let new_job = Job {
                rem_size: size,
                arrival_time: time,
                original_size: size,
            };
            next_arrival = time + arrival_dist.sample(&mut rng);
            num_arrivals += 1;
            jobs_on_arrival += jobs_waiting.len() + jobs_in_progress.len();
            policy.arrival(&mut jobs_waiting, &mut jobs_in_progress, new_job);
        }
    }

    assert!(results.total_queue_time >= -EPSILON);
    assert!(results.total_response_time >= results.total_service_time - EPSILON);
    results.total_time = time;
    let _mean_number_of_jobs_on_arrival = jobs_on_arrival as f64 / num_arrivals as f64;
    results.mean_response_time();
    results.mean_number_of_jobs(lambda);
    results.mean_service_time();
    results.mean_slowdown();
    results.mean_queue_time();
    results.mean_response_time();
    results
}

/// A job size distribution to draw from during a simulation run
#[derive(Clone, Copy, Debug)]
pub enum Dist {
    /// Continuous uniform distribution on [low, high).
    Uniform(f64, f64),
    /// Exponential distribution with the given mean.
    Exponential(f64),
    /// Hyperexponential distribution on (low_mean, high_mean, prob_low).
    Hyperexponential(f64, f64, f64),
}

impl Dist {
    fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        let sample = match self {
            Dist::Uniform(low, high) => rng.random_range(*low..*high),
            Dist::Exponential(mean) => rng.sample(Exp::new(1.0 / mean).unwrap()),
            Dist::Hyperexponential(low_mean, high_mean, prob_low) => {
                let mean = if rng.random::<f64>() < *prob_low {
                    low_mean
                } else {
                    high_mean
                };
                rng.sample(Exp::new(1.0 / mean).unwrap())
            }
        };
        assert!(sample >= 0.0);
        sample
    }

    /// This distribution's mean, which has to be 1.0 for simulate to run.
    pub fn mean(&self) -> f64 {
        match self {
            Dist::Uniform(low, high) => (low + high) / 2.0,
            Dist::Exponential(mean) => *mean,
            Dist::Hyperexponential(low_mean, high_mean, prob_low) => {
                low_mean * prob_low + high_mean * (1.0 - prob_low)
            }
        }
    }
}