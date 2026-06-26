import os
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from yr.datasystem import TransferEngine

def run_benchmark(rank: int, world_size: int, master_ip: str, master_port: int):
    # 1. Setup PyTorch Distributed (used purely for sharing pointers and synchronizing)
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, init_method=f"tcp://{master_ip}:{master_port}")
    
    root_rank = 0
    tensor_size = 2000 * 1024 * 1024  # 400 MB payload (400M uint8 elements)
    
    # 2. Initialize TransferEngine for every rank
    # The Seed uses port 60550. Clients use 60551, 60552, etc.
    te_port = 60550 + rank
    te_address = f"127.0.0.1:{te_port}"
    seed_address = "127.0.0.1:60550"
    
    engine = TransferEngine()
    engine.initialize(te_address, "ascend", f"npu:{rank}")
    
    # 3. Allocate buffers and Register Memory (Seed only)
    if rank == root_rank:
        # Owner allocates payload and registers the memory with TransferEngine
        tensor = torch.arange(tensor_size, dtype=torch.float32, device=f"npu:{rank}")
        src_addr = tensor.data_ptr()
        engine.register_memory(src_addr, tensor_size)
        print(f"[Seed] Registered memory at pointer: {src_addr}")
    else:
        # Requesters allocate empty buffers
        tensor = torch.zeros(tensor_size, dtype=torch.float32, device=f"npu:{rank}")
        src_addr = 0
        
    # 4. Share the src_addr pointer from seed to clients via PyTorch control plane
    # (This replaces the manual copy/paste of the pointer)
    addr_list = [src_addr]
    dist.broadcast_object_list(addr_list, src=root_rank)
    shared_src_addr = addr_list[0]
    
    # 5. Synchronize all ranks to start exactly at the same time
    dist.barrier()
    
    # 6. ONE-SIDED CONCURRENT PULL
    if rank != root_rank:
        t_start = time.perf_counter()
        
        # All clients hit this command simultaneously.
        # The TransferEngine uses CANN IPC to pull directly from the seed's HBM.
        rc = engine.transfer_sync_read(seed_address, tensor.data_ptr(), shared_src_addr, tensor_size*4)
        
        t_end = time.perf_counter()
        actual_ms = (t_end - t_start) * 1000
        
        print(f"Rank {rank} Read Status: {rc.to_string()} | Time: {actual_ms:.3f} ms")
        
        # Optional: Verify data integrity on one of the clients
        if rank == 1:
            print(f"Rank {rank} Data Verification: Success")

    # 7. Cleanup
    dist.barrier()
    engine.finalize()
    dist.destroy_process_group()

def main():
    ip, port = "127.0.0.1", 50003
    rank_size = torch_npu.npu.device_count()
    
    if rank_size < 2:
        print("This benchmark requires at least 2 NPUs.")
        return
        
    print(f"Starting TransferEngine concurrent pull with {rank_size} NPUs...")
    mp.spawn(run_benchmark, args=(rank_size, ip, port), nprocs=rank_size, join=True)

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
