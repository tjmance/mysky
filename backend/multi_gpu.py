import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import subprocess
import traceback
import psutil
import threading
import time

class MultiGPUManager:
    """Multi-GPU management for distributed video generation"""
    
    def __init__(self):
        self.is_distributed = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.device_ids = []
        self.setup_complete = False
        
        self._setup_multi_gpu()
    
    def _setup_multi_gpu(self):
        """Setup multi-GPU environment"""
        try:
            print("🔧 Setting up multi-GPU environment...")
            
            # Check available GPUs
            if not torch.cuda.is_available():
                print("⚠️ CUDA not available, using single CPU")
                self.setup_complete = True
                return
            
            gpu_count = torch.cuda.device_count()
            print(f"🔍 Found {gpu_count} GPU(s)")
            
            if gpu_count <= 1:
                print("ℹ️ Single GPU mode")
                self.device_ids = [0] if gpu_count == 1 else []
                self.setup_complete = True
                return
            
            # Multi-GPU setup
            self.device_ids = list(range(gpu_count))
            self._print_gpu_info()
            
            # Check if running in distributed mode
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                self._setup_distributed()
            else:
                print("ℹ️ Multi-GPU available but not in distributed mode")
                print("💡 Use: torchrun --nproc_per_node=N script.py for distributed training")
            
            self.setup_complete = True
            
        except Exception as e:
            print(f"❌ Multi-GPU setup failed: {e}")
            traceback.print_exc()
            self.setup_complete = False
    
    def _print_gpu_info(self):
        """Print information about available GPUs"""
        print("\n🔍 GPU Information:")
        for i, device_id in enumerate(self.device_ids):
            props = torch.cuda.get_device_properties(device_id)
            memory_gb = props.total_memory / 1024**3
            print(f"  GPU {device_id}: {props.name} ({memory_gb:.1f} GB)")
            
            # Check current memory usage
            allocated = torch.cuda.memory_allocated(device_id) / 1024**3
            cached = torch.cuda.memory_reserved(device_id) / 1024**3
            print(f"    Memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached")
    
    def _setup_distributed(self):
        """Setup distributed training environment"""
        try:
            print("🚀 Setting up distributed training...")
            
            # Get distributed environment variables
            self.rank = int(os.environ['RANK'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            
            print(f"🌍 Rank: {self.rank}/{self.world_size}, Local Rank: {self.local_rank}")
            
            # Initialize process group
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=self.world_size,
                rank=self.rank
            )
            
            # Set device for this process
            torch.cuda.set_device(self.local_rank)
            
            self.is_distributed = True
            print("✅ Distributed training initialized")
            
        except Exception as e:
            print(f"❌ Distributed setup failed: {e}")
            self.is_distributed = False
    
    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap model for multi-GPU training"""
        try:
            if not self.setup_complete:
                return model
            
            # Move model to appropriate device
            if torch.cuda.is_available():
                if self.is_distributed:
                    model = model.cuda(self.local_rank)
                    model = DDP(model, device_ids=[self.local_rank])
                    print(f"✅ Model wrapped with DDP on GPU {self.local_rank}")
                elif len(self.device_ids) > 1:
                    model = torch.nn.DataParallel(model, device_ids=self.device_ids)
                    model = model.cuda()
                    print(f"✅ Model wrapped with DataParallel on GPUs {self.device_ids}")
                else:
                    model = model.cuda()
                    print("✅ Model moved to single GPU")
            
            return model
            
        except Exception as e:
            print(f"❌ Model wrapping failed: {e}")
            return model
    
    def distribute_batch(self, batch_data: Any, batch_size: int) -> List[Any]:
        """Distribute batch across multiple GPUs"""
        if not self.is_distributed or self.world_size <= 1:
            return [batch_data]
        
        try:
            # Calculate batch size per GPU
            per_gpu_batch_size = batch_size // self.world_size
            remainder = batch_size % self.world_size
            
            distributed_batches = []
            start_idx = 0
            
            for i in range(self.world_size):
                # Add remainder to first few processes
                current_batch_size = per_gpu_batch_size + (1 if i < remainder else 0)
                end_idx = start_idx + current_batch_size
                
                # Extract batch slice
                if hasattr(batch_data, '__getitem__'):
                    batch_slice = batch_data[start_idx:end_idx]
                else:
                    batch_slice = batch_data  # If can't slice, replicate
                
                distributed_batches.append(batch_slice)
                start_idx = end_idx
            
            return distributed_batches
            
        except Exception as e:
            print(f"⚠️ Batch distribution failed: {e}")
            return [batch_data]
    
    def synchronize(self):
        """Synchronize all processes"""
        if self.is_distributed:
            dist.barrier()
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """All-reduce operation across GPUs"""
        if self.is_distributed:
            dist.all_reduce(tensor, op=op)
        return tensor
    
    def gather_results(self, local_result: Any) -> List[Any]:
        """Gather results from all processes"""
        if not self.is_distributed:
            return [local_result]
        
        try:
            gathered_results = [None] * self.world_size
            dist.all_gather_object(gathered_results, local_result)
            return gathered_results
            
        except Exception as e:
            print(f"⚠️ Result gathering failed: {e}")
            return [local_result]
    
    def is_main_process(self) -> bool:
        """Check if current process is the main process"""
        return not self.is_distributed or self.rank == 0
    
    def get_device(self) -> torch.device:
        """Get appropriate device for current process"""
        if torch.cuda.is_available():
            if self.is_distributed:
                return torch.device(f"cuda:{self.local_rank}")
            elif self.device_ids:
                return torch.device(f"cuda:{self.device_ids[0]}")
        
        return torch.device("cpu")
    
    def optimize_memory(self):
        """Optimize GPU memory usage"""
        try:
            if torch.cuda.is_available():
                for device_id in self.device_ids:
                    torch.cuda.empty_cache()
                    
                    # Set memory management
                    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                        # Reserve some memory for other processes
                        memory_fraction = 0.9 if len(self.device_ids) == 1 else 0.8
                        torch.cuda.set_per_process_memory_fraction(
                            memory_fraction, device_id
                        )
                
                print("✅ GPU memory optimized")
                
        except Exception as e:
            print(f"⚠️ Memory optimization failed: {e}")
    
    def monitor_gpu_memory(self) -> Dict[int, Dict[str, float]]:
        """Monitor GPU memory usage across all devices"""
        memory_info = {}
        
        if torch.cuda.is_available():
            for device_id in self.device_ids:
                try:
                    allocated = torch.cuda.memory_allocated(device_id) / 1024**3
                    reserved = torch.cuda.memory_reserved(device_id) / 1024**3
                    total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
                    
                    memory_info[device_id] = {
                        "allocated_gb": allocated,
                        "reserved_gb": reserved,
                        "total_gb": total,
                        "free_gb": total - reserved,
                        "utilization_percent": (reserved / total) * 100
                    }
                    
                except Exception as e:
                    print(f"⚠️ Failed to get memory info for GPU {device_id}: {e}")
        
        return memory_info
    
    def create_distributed_launcher(
        self,
        script_path: str,
        num_gpus: Optional[int] = None,
        additional_args: List[str] = None
    ) -> str:
        """Create torchrun command for distributed execution"""
        
        if num_gpus is None:
            num_gpus = len(self.device_ids)
        
        if additional_args is None:
            additional_args = []
        
        # Build torchrun command
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "--nnodes=1",
            "--node_rank=0",
            "--master_addr=localhost",
            "--master_port=29500",
            script_path
        ] + additional_args
        
        return " ".join(cmd)
    
    def launch_distributed(
        self,
        script_path: str,
        num_gpus: Optional[int] = None,
        additional_args: List[str] = None,
        background: bool = False
    ) -> subprocess.Popen:
        """Launch distributed training/inference"""
        
        cmd = self.create_distributed_launcher(script_path, num_gpus, additional_args)
        
        print(f"🚀 Launching distributed process: {cmd}")
        
        try:
            if background:
                process = subprocess.Popen(
                    cmd.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            else:
                process = subprocess.run(cmd.split(), check=True)
            
            return process
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Distributed launch failed: {e}")
            raise
    
    def cleanup(self):
        """Cleanup distributed environment"""
        try:
            if self.is_distributed:
                dist.destroy_process_group()
                print("✅ Distributed environment cleaned up")
                
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get multi-GPU system information"""
        info = {
            "setup_complete": self.setup_complete,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": len(self.device_ids),
            "device_ids": self.device_ids,
            "is_distributed": self.is_distributed,
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank
        }
        
        if torch.cuda.is_available():
            info["gpu_memory"] = self.monitor_gpu_memory()
        
        return info
    
    def estimate_optimal_batch_size(self, model_size_mb: float = 1000) -> int:
        """Estimate optimal batch size based on available GPU memory"""
        if not torch.cuda.is_available() or not self.device_ids:
            return 1
        
        try:
            # Get available memory on primary GPU
            device_id = self.device_ids[0]
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            available_memory = total_memory * 0.8  # Reserve 20% for safety
            
            # Estimate memory per sample (rough approximation)
            estimated_memory_per_sample = model_size_mb * 1024 * 1024 * 2  # 2x model size
            
            # Calculate batch size
            estimated_batch_size = max(1, int(available_memory // estimated_memory_per_sample))
            
            # Scale by number of GPUs
            if len(self.device_ids) > 1:
                estimated_batch_size *= len(self.device_ids)
            
            return min(estimated_batch_size, 32)  # Cap at reasonable maximum
            
        except Exception as e:
            print(f"⚠️ Batch size estimation failed: {e}")
            return 1

# Global instance
multi_gpu_manager = MultiGPUManager()

# Utility functions for easy access
def is_distributed() -> bool:
    """Check if running in distributed mode"""
    return multi_gpu_manager.is_distributed

def get_world_size() -> int:
    """Get world size"""
    return multi_gpu_manager.world_size

def get_rank() -> int:
    """Get current rank"""
    return multi_gpu_manager.rank

def is_main_process() -> bool:
    """Check if main process"""
    return multi_gpu_manager.is_main_process()

def get_device() -> torch.device:
    """Get current device"""
    return multi_gpu_manager.get_device()

def wrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Wrap model for multi-GPU"""
    return multi_gpu_manager.wrap_model(model)

def synchronize():
    """Synchronize all processes"""
    multi_gpu_manager.synchronize()

def monitor_gpus() -> Dict[int, Dict[str, float]]:
    """Monitor GPU memory"""
    return multi_gpu_manager.monitor_gpu_memory()

def cleanup_distributed():
    """Cleanup distributed environment"""
    multi_gpu_manager.cleanup()