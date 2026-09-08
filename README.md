This is a queueing simulation for a variety of M/G/1 scheduling policies, from simpler 
introductory ones such as First Come First Served, to research-level policies such as 
Nudge. It can be used as a standalone program, as well as a library.  

# Implemented Policies
- First Come, First Serve (FCFS)
- Last Come, First Serve (LCFS)
- Preemptive Last Come, First Serve (PLCFS)
- Processor sharing (PS)
- Shortest Remaining Processing Time (SRPT)
- Preemptive Shortest Job First (PSJF)
- Least Attained Service (LAS)
- Longest Remaining Processing Time (LRPT)
- [Nudge (Nudge(Threshold, false))](https://isaacg1.github.io/publications/#nudge)
- [GammaBoost (GammaB(gamma))](https://ziv.codes/yu_strongly_2024/)
- [AccumulatingPriority](https://link.springer.com/article/10.1007/s11134-013-9382-6)


# Usage
The simulation parameters live at the top of src/main.rs and are edited in place.

```rust
let num_jobs = 10_000_000;
let rho      = 0.4;
let seed     = 10;
let dist     = Dist::Hyperexponential(0.5, 3.0, 0.8);
let policies = vec![Policy::Nudge(0.2, false)];
```

Once you set the parameters, you need to choose an output mode with an argument via the command line. 

| Argument | Debug stepping | Response time histogram |
|---|---|---|
| *(none)* or `0` | off | off |
| `1` | off | on |
| `2` | on | off |
| `3` | on | on |

For example,

```rust
cargo run --release 1
```

would run the simulation with the debug configuration off and the response time histogram on.

Additionally, you can also call the simulate function from the library on its own. You must create a Config object to represent whether or not you would like the debug configuration and the response time histogram, and pass the parameters into the function. 

```rust
let config = Config {
    response_time_histogram: None,
    debug: false,
};

let results = simulate(
    0.4,                                    
    Dist::Hyperexponential(0.5, 3.0, 0.8),  
    Policy::SRPT,                           
    10_000_000,                             
    10,                                     
    &config,
);
```

# Results

The Results struct output by the simulate function stores the total response time, queuing time, service time, and slowdown. Additionally, it also calculates the mean response time, the mean queue time, the mean slowdown, and the mean number of jobs. 

# Extending the package

You can add your own index-based policy, where there is no mid-job preemption, by using IndexPolicy implementation:

```rust
pub struct FirstComeFirstServe;

impl IndexPolicy for FirstComeFirstServe {
    fn index(&self, job: &Job) -> f64 {
        job.arrival_time()
    }
}

let results = simulate(
    0.4,                                    
    Dist::Hyperexponential(0.5, 3.0, 0.8),  
    FirstComeFirstServe,                           
    10_000_000,                             
    10,                                     
    &config,
);
```
