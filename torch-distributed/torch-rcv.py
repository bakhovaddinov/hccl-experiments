#!/usr/bin/env python3
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def run_benchmark(rank: int, world_size: int, master_ip: str, master_port: int):
    torch_npu.npu.set_device(rank)
    init_method = f"tcp://{master_ip}:{master_port}"
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    
    root_rank = 0
    client_ranks = list(range(1, world_size))
    
    # Create a sub-group for clients to handle sequencing without host overhead
    client_group = dist.new_group(ranks=client_ranks)
    
    # ~400 MB payload (100M elements * 4 bytes for float32)
    tensor_size = 100 * 1024 * 1024         
    
    if rank == root_rank:
        tensor = torch.randn(tensor_size, dtype=torch.float32, device="npu")
    else:
        tensor = torch.zeros(tensor_size, dtype=torch.float32, device="npu")
        
    # Warmup communication pipelines
    if rank == root_rank:
        h = dist.isend(tensor, dst=1)
        h.wait()
    elif rank == 1:
        dist.recv(tensor, src=0)
    dist.barrier()

    # ----------------------------------------------------
    # SCENARIO 1: Sequential Client Reads
    # ----------------------------------------------------
    dist.barrier()
    t_start_seq = time.perf_counter()
    
    if rank == root_rank:
        # Host dumps all allocations asynchronously to the HCCL runtime queue at once
        handles = [dist.isend(tensor, dst=i) for i in client_ranks]
        for h in handles:
            h.wait()
    else:
        # Clients strictly loop and pull one by one
        for target_rank in client_ranks:
            if rank == target_rank:
                dist.recv(tensor, src=root_rank)
            # Synchronize only among clients to step-lock execution sequence
            dist.barrier(group=client_group)
            
    dist.barrier()
    seq_time = (time.perf_counter() - t_start_seq) * 1000

    # Reset buffers for accuracy
    if rank != root_rank:
        tensor.zero_()
    dist.barrier()

    # ----------------------------------------------------
    # SCENARIO 2: Parallel Client Reads
    # ----------------------------------------------------
    dist.barrier()
    t_start_par = time.perf_counter()
    
    if rank == root_rank:
        # Host maps all tensors concurrently to the HCCL runtime
        handles = [dist.isend(tensor, dst=i) for i in client_ranks]
        for h in handles:
            h.wait()
    else:
        # Clients read concurrently using non-blocking primitives
        handle = dist.irecv(tensor, src=root_rank)
        handle.wait()
        
    dist.barrier()
    par_time = (time.perf_counter() - t_start_par) * 1000

    # Report results from root
    if rank == root_rank:
        print("\n--- HCCL Distributed P2P Results (400MB) ---")
        print(f"Scenario 1 (Sequential Pulls): {seq_time:.2f} ms")
        print(f"Scenario 2 (Parallel Pulls):   {par_time:.2f} ms")
        print(f"Parallel Acceleration Factor:  {seq_time / par_time:.2f}x")
        print("----------------------------------------------")

def main():
    ip, port = "127.0.0.1", 50003
    rank_size = torch_npu.npu.device_count()
    if rank_size < 2:
        print("This benchmark requires at least 2 NPUs.")
        return
    mp.spawn(run_benchmark, args=(rank_size, ip, port), nprocs=rank_size, join=True)

if __name__ == "__main__":
    main()