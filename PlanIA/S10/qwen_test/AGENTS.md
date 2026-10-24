# Working on Khipu

Use Slurm to run computational workloads. Use the login node for job management,
file transfers, environment setup, and small preparation tasks only.

Official guide: https://docs.khipu.utec.edu.pe/guia-de-usuario/enviar-jobs/

## Verified account limits

Verified from live Slurm configuration on 2026-10-23:

| Setting | Value |
| --- | --- |
| Account | `pregrado` |
| Account QoS | `a-pregrado` |
| RAM per user across running jobs | `98G` (98 GiB) |
| CPUs per user across running jobs | 32 |
| GPUs per user across running jobs | 1 |
| GPU shards per user across running jobs | 8 |
| Concurrent running jobs per user | 2 |
| Maximum wall time per job | `08:00:00` |

These are aggregate user limits, not separate allowances for each job. For
example, two jobs requesting `64G` each exceed the user's RAM allowance.
GPU and shard limits apply to the respective resource requests; they are not
interchangeable measures of GPU memory.

Account QoS also reported shared group limits of 480 CPUs, `228G` RAM, and
16 GPU shards. Other users' allocations can therefore delay a job even when it
fits the individual user's limits.

Partition QoS can impose additional limits. At verification time, `p-debug` and
`p-debug-gpu` each limited jobs to one hour. The guide lists 30 minutes for debug
jobs, so use a 30-minute request for initial tests and recheck live configuration
when a longer test is necessary. A partition's `MaxTime=UNLIMITED` does not remove
account or QoS wall-time limits.

Node RAM capacity and `free -h` output do not define the user's allocation.
Request job RAM explicitly with `--mem`, which specifies RAM **per node**.
GPU VRAM is separate from this RAM request. Do not assume a GPU shard reserves
an equivalent amount of isolated VRAM.

Limits and availability can change. Recheck Slurm before sizing a new workload.

## Inspect resources and limits

```bash
# Partitions, time limits, CPUs, node RAM (MiB), and GPU resources
sinfo -o '%P %a %l %D %c %m %G'

# Node state and reported free RAM (MiB); free RAM is not schedulable capacity
sinfo -N -o '%N %P %m %e %t'
scontrol show node <node-name>
scontrol show partition <partition-name>

# Account association and allowed QoS
sacctmgr -nP show assoc where user="$USER" format=Account,Partition,QOS

# User, job, group, and wall-time limits for all QoS entries
sacctmgr -nP show qos format=Name,MaxWall,MaxTRESPU,MaxTRESPJ,GrpTRES,MaxJobsPU
```

Use `standard` for CPU jobs, `big-mem` for memory-heavy CPU jobs, `gpu` for GPU
jobs, and `debug` / `debug-gpu` for short tests. Larger nodes do not override the
98 GiB user limit. Check current node membership with Slurm because it differs
from some tables in the guide.

## Submit, inspect, and cancel jobs

```bash
sbatch job.sbatch                         # Submit a batch script
squeue -u "$USER"                        # View your jobs and pending reasons
scontrol show job <job-id>                # Inspect allocation and job details
tail -f slurm-<job-id>.out                # Follow the default output log
sstat -j <job-id>.batch --format=JobID,MaxRSS,AveRSS,MaxVMSize
sacct -j <job-id> --format=JobID,JobName,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,ExitCode
scancel <job-id>                          # Cancel a specific job when requested
```

`MaxRSS` is a reported peak resident-memory metric for a task/step, not
necessarily total job memory. Check job-step rows and workload parallelism when
using it to size future requests. Accounting metrics may be delayed or unavailable
while a job is running.

## CPU batch example

Save as `job.sbatch`, replace the Python command with the actual workload, and
activate the required environment before execution:

```bash
#!/bin/bash
#SBATCH --job-name=python-job
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Load modules / activate the project's environment here.
srun python -u script.py
```

Match CPU and RAM requests to the workload and available aggregate allowance.
Use one of `--mem`, `--mem-per-cpu`, or `--mem-per-gpu` for a submission, rather
than combining these memory request modes.

## Interactive and GPU jobs

```bash
# Short interactive CPU session on a compute node
srun --partition=debug --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=4G --time=00:30:00 --pty bash
```

For GPU jobs, inspect `sinfo` GRES output before selecting resources. The guide
supports shared GPU requests such as `--gres=shard:1`. For `ag001`, the node guide
recommends a specific GPU/MIG instance instead of shards; for example, add these
directives to an otherwise appropriately sized batch script:

```bash
#SBATCH --partition=gpu
#SBATCH --nodelist=ag001
#SBATCH --gres=gpu:a100_1g.5gb:1
```

Select a GPU instance with enough VRAM for the actual workload. Confirm the type
exists and is usable under the current account/QoS before submitting. Prepare
dependencies and downloaded assets before running on compute nodes; verify node
internet access rather than assuming it is available.

## Agent execution notes

If a Slurm query fails with `Error creating slurm stream socket: Operation not
permitted` in the tool sandbox, request execution outside the sandbox using the
tool's approval mechanism. This was required for live controller queries in this
workspace. Do not interpret that sandbox error as evidence that Slurm is down.

Do not submit placeholder workloads or cancel unrelated jobs. When asked to run
a workload, use the project's actual command, choose explicit resource requests,
and report the submitted job ID and log location.

## References

- Submission guide: https://docs.khipu.utec.edu.pe/guia-de-usuario/enviar-jobs/
- Partitions: https://docs.khipu.utec.edu.pe/guia-de-usuario/enviar-jobs/particiones/
- Nodes and GPU instances: https://docs.khipu.utec.edu.pe/guia-de-usuario/enviar-jobs/nodos/
- Slurm options: https://docs.khipu.utec.edu.pe/guia-de-usuario/enviar-jobs/opciones-de-slurm/
