# Optimus: an efficient dynamic resource scheduler for deep learning clusters

[Optimus: an efficient dynamic resource scheduler for deep learning clusters](https://dl.acm.org/citation.cfm?id=3190517) Peng et al., *EuroSys’18*

run on top of Kubernetes, and the authors claim about a 1.6x reduction in makespan compared to the mostly widely used schedulers today.

### Deep learning clusters

We’re using ever larger models, with ever increasing amounts of data (at least, whenever we can get our hands on it). improves the learning accuracy, but also increases the training time. Typically a model is partitioned among multiple *parameter servers*, and training data is spread across multiple *workers*. 

Training may be synchronous (a barrier at the end of each training step), or asynchronous.

> …different from experimental models, production models are mature and can typically converge to the global/local optimum very well since all hyper-parameters (e.g., learning rate – how quickly a DNN adjusts itself, mini-batch size) have been well-tuned during the experimental phase. In this work, **we focus on such production models**, and **leverage their convergence property** to estimate a training job’s progress towards convergence.

Optimus **uses the convergence of training loss.**

Existing schedulers such as **Borg** and **YARN** allocate a fixed amount of resource to each job upon its submission, and *do not vary resource* while a job is running. 

The number of allocated workers and parameter servers influence training speed. The following chart shows that for a ResNet-50 model, the maximal training speed is achieved with 8 workers and 12 parameter servers:

![img](https://adriancolyer.files.wordpress.com/2018/06/optimus-fig-4a.jpeg?w=480)

Alternatively, if we fix the ratio of parameter servers to workers at 1:1, and increase the number of containers available while maintaining that ratio, we get a chart like this:

![img](https://adriancolyer.files.wordpress.com/2018/06/optimus-fig-4b.jpeg?w=480)

Notice that **beyond a certain point, adding more resources starts to slow down training!** 

### Optimus high-level design

* **Configuring a fixed number of workers/parameter servers** upon job submission is hence unfavorable. 

> in Optimus, we maximally exploit **varying runtime resource availability** by **adjusting numbers and placement of workers and parameter servers**, aiming to pursue the best resource efficiency and training speed at each time.

To make good decisions, Optimus needs 

* **to understand the relationship** between **resource configuration** and **the time a training job takes to achieve convergence.** 

* builds and fits **performance models** to estimate 

  * how many steps/epochs a job will need to reach convergence
  * how different combinations of resource and parameter servers impact training speed. 

  takes about 10 samples to learn a good enough initial approximation.

> … we run a job for a few steps with different resource configurations, learn the **training speed as a function of resource configurations** using data collected from these steps, and then **keep tuning our model on the go.**

Using the predictions obtained from these models, Optimus allocates resources to workers and parameter servers using a **greedy algorithm based on estimated marginal gains.** 

Tasks are then placed using the smallest number of servers possible, subject to the following constraint: 

* each server runs *p* parameter servers, and *w* workers (i.e., the values of *p* and *w* are the same across all servers).

### Modeling deep learning jobs

**To estimate the number of steps/epochs a job needs to reach completion:**

SGD converges at a rate of O(1/k) given number of steps *k*. So we can **approximate the training loss curve** using the following model:

$ l = \frac{1}{\beta_0 \cdot k + \beta_1} + \beta_2 $​

Where *l* is the training loss and $\beta_0, \beta_1, \beta_2$ are training coefficients. Here are some examples of real training loss curves for different deep learning jobs:

![img](https://adriancolyer.files.wordpress.com/2018/06/optimus-fig-5.jpeg?w=480)

As we get more data points after each step, the model fit (**prediction error**) improves, as shown below:

![img](https://adriancolyer.files.wordpress.com/2018/06/optimus-fig-6.jpeg?w=480)

Here’s an example of model fitting when training a Seq2Seq model:

![img](https://adriancolyer.files.wordpress.com/2018/06/optimus-fig-7.jpeg?w=480)

**To estimate training speed based on the computation and communication patterns of different parameter server and worker configurations**. 

* **asynchronous training**, where workers process mini-batches at their own pace. Assuming *w* workers and *p* parameter servers the training speed function *f(p,w)* can be approximated as follows:  

  $f(p,w) = w \cdot (\theta_0 + \theta_1 \cdot \frac{w}{p} + \theta_2 \cdot w + \theta_3 \cdot p)^{-1}$

* **synchronous training** the progress is determined by the dataset size *M*, meaning that each worker is allocated $m = M/w$ mini-batches. The training speed function can be approximated by: 

  $f(p,w) = (\theta_0 \cdot \frac{M}{w} + \theta_1 \cdot \frac{w}{p} + \theta_2 \cdot w + \theta_3 \cdot w + \theta_4 \cdot p)^{-1}$

**To learn the values of the $\theta$ parameters**: 

each model is trained on a small sample set of training data for several steps, with multiple combinations of *p* and *w*. Each run takes tens of seconds. A **non-negative least-squares (NNLS)** solver is used to find the **best fitting parameters**. Here are some examples of model fitting:

![img](https://adriancolyer.files.wordpress.com/2018/06/optimus-fig-9.jpeg?w=520)

### Dynamic scheduling

Jobs arrive in an **online manner**, at each time step Optimus 

* allocates resources to newly submitted jobs,
* adjusts the resource allocations of existing jobs. 

The scheduler **aims to minimize the average completion time of these jobs**. With knowledge of the *estimated number of training steps remaining*, and *a model of how training speed is impacted by resources*, this **becomes a constraint solving problem.** 

#### Introduce a notion of *marginal gain*: a heuristic leading to an efficient approximation. 

***dominant resource*** in the cluster: the resource type that has maximal share in the overall capacity of the cluster. 

***marginal gain***: is the **estimated reduction in job completion time** when **one worker (or parameter server) is added to a job**, **divided by the amount of the dominant resource that a worker (parameter server) occupies**.



**Initially** each job is **allocated one worker and one parameter server.** Then we **iterate**, greedily **adding a worker (parameter server) to the job with the largest marginal gain in each iteration**. Iteration stops when cluster resource is exhausted, or the marginal gains are all negative.

#### To place jobs across servers.

> Given the numbers of workers and parameter servers in a synchronous training job, **the optimal worker/parameter server placement principle** to achieve the maximal training speed for the job, in a cluster of homogeneous servers, **is to use the smallest number of servers to host the job**, such that the same number of parameter servers and the same number of workers are deployed on each of these servers.

It’s also important to **detect stragglers** and **replace** them **by launching a new worker**.

#### A **good workload balance** among the parameter servers through careful division of the model parameters.

To minimize

* the maximal difference of parameter sizes between two parameter servers, 
* the total number of parameter update requests between parameter servers and workers during one training step (each request from a worker asks for one updated parameter block), 
* the maximal difference of the number of parameter update requests between two parameter servers

design a **parameter assignment algorithm (PAA)**

* sort parameter blocks in decreasing order of size 
* calculate the average parameter size avg_size, i.e., the overall parameter size divided by the number of parameter servers. 
* For each block, 
  * if its size is very small (e.g., less than 1% of avg_size),  
    * assign it to the parameter server with the least number of update requests. 
  * If the block size is between 1% of avg_size and avg_size,
    * assign the block to the parameter server with the smallest remaining capacity (avg_size minus the size of parameters assigned), that can accommodate it (a best-fit approach). 
  * If the block size is larger than avg_size, 
    * further slice it into partitions with size avg_size and assign the sliced partitions to the parameter server with the smallest size of parameters assigned. 
* Once a parameter block (or partition) is assigned to a parameter server, we add the number of parameter update requests on the server by 1.

### Evaluation

Optimus is compared against a fairness-based scheduler using [Dominant Resource Fairness](https://cs.stanford.edu/~matei/papers/2011/nsdi_drf.pdf) as used in many existing systems, as well as *Tetris*, which preferentially allocates resources to jobs with low duration or small resource consumption. Optimus reduces the average completion time and makespan by 2.39x and 1.63x respectively as compared to the DRF based scheduler.

![img](https://adriancolyer.files.wordpress.com/2018/06/optimus-fig-11.jpeg?w=480)

The resource allocation algorithm is the biggest contributing factor to the improvements. Optimus can schedule 100,000 tasks on 16,000 nodes in about 5 seconds.

> Our experiments on a Kubernetes cluster show that Optimus outperforms representative cluster schedulers significantly.