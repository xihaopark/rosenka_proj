"""
GPU资源管理器
优化GPU内存使用和性能监控
"""

import torch
import psutil
import logging
from typing import Dict, Optional
import time

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

class GPUManager:
    """GPU资源管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化NVIDIA管理库
        if NVML_AVAILABLE and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self.nvml_enabled = True
                self.logger.info("NVIDIA管理库初始化成功")
            except Exception as e:
                self.logger.warning(f"NVIDIA管理库初始化失败: {e}")
                self.nvml_enabled = False
        else:
            self.nvml_enabled = False
        
        # 性能监控
        self.memory_usage_history = []
        self.performance_stats = {
            'peak_memory_usage': 0,
            'current_memory_usage': 0,
            'memory_utilization': 0.0
        }
    
    def optimize_memory_usage(self):
        """优化GPU内存使用"""
        if not torch.cuda.is_available():
            self.logger.info("CUDA不可用，跳过GPU内存优化")
            return
        
        try:
            # 清理未使用的缓存
            torch.cuda.empty_cache()
            
            # 设置内存分配策略
            torch.cuda.set_per_process_memory_fraction(0.8)
            
            # 启用内存池
            if hasattr(torch.cuda, 'memory'):
                torch.cuda.memory.set_per_process_memory_fraction(0.8)
            
            self.logger.info("GPU内存优化完成")
            
        except Exception as e:
            self.logger.error(f"GPU内存优化失败: {e}")
    
    def get_gpu_info(self) -> Dict:
        """获取GPU信息"""
        if not torch.cuda.is_available():
            return {'available': False, 'message': 'CUDA不可用'}
        
        try:
            gpu_info = {
                'available': True,
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(),
                'memory_info': self._get_memory_info(),
                'compute_capability': torch.cuda.get_device_capability(),
            }
            
            if self.nvml_enabled:
                gpu_info.update(self._get_nvml_info())
            
            return gpu_info
            
        except Exception as e:
            self.logger.error(f"获取GPU信息失败: {e}")
            return {'available': False, 'error': str(e)}
    
    def _get_memory_info(self) -> Dict:
        """获取内存信息"""
        try:
            # PyTorch内存信息
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            max_memory_allocated = torch.cuda.max_memory_allocated()
            max_memory_reserved = torch.cuda.max_memory_reserved()
            
            return {
                'allocated': memory_allocated,
                'reserved': memory_reserved,
                'max_allocated': max_memory_allocated,
                'max_reserved': max_memory_reserved,
                'allocated_mb': memory_allocated / (1024**2),
                'reserved_mb': memory_reserved / (1024**2),
                'max_allocated_mb': max_memory_allocated / (1024**2),
                'max_reserved_mb': max_memory_reserved / (1024**2)
            }
        except Exception as e:
            self.logger.error(f"获取内存信息失败: {e}")
            return {}
    
    def _get_nvml_info(self) -> Dict:
        """获取NVML信息"""
        if not self.nvml_enabled:
            return {}
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # 内存信息
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # 利用率信息
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # 温度信息
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # 功率信息
            power_usage = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
            
            return {
                'nvml_memory_total': memory_info.total,
                'nvml_memory_used': memory_info.used,
                'nvml_memory_free': memory_info.free,
                'nvml_memory_total_mb': memory_info.total / (1024**2),
                'nvml_memory_used_mb': memory_info.used / (1024**2),
                'nvml_memory_free_mb': memory_info.free / (1024**2),
                'gpu_utilization': utilization.gpu,
                'memory_utilization': utilization.memory,
                'temperature': temperature,
                'power_usage': power_usage / 1000,  # 转换为瓦特
                'power_limit_min': power_limit[0] / 1000,
                'power_limit_max': power_limit[1] / 1000
            }
            
        except Exception as e:
            self.logger.error(f"获取NVML信息失败: {e}")
            return {}
    
    def monitor_memory_usage(self):
        """监控内存使用情况"""
        if not torch.cuda.is_available():
            return
        
        try:
            current_usage = torch.cuda.memory_allocated()
            current_usage_mb = current_usage / (1024**2)
            
            # 更新统计信息
            self.performance_stats['current_memory_usage'] = current_usage_mb
            if current_usage_mb > self.performance_stats['peak_memory_usage']:
                self.performance_stats['peak_memory_usage'] = current_usage_mb
            
            # 记录历史
            self.memory_usage_history.append({
                'timestamp': time.time(),
                'memory_usage': current_usage_mb
            })
            
            # 保持历史记录在合理范围内
            if len(self.memory_usage_history) > 1000:
                self.memory_usage_history = self.memory_usage_history[-500:]
            
            # 计算利用率
            if self.nvml_enabled:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.performance_stats['memory_utilization'] = memory_info.used / memory_info.total
            
        except Exception as e:
            self.logger.error(f"内存监控失败: {e}")
    
    def batch_process_optimization(self, batch_size: int = 4) -> int:
        """批处理优化"""
        if not torch.cuda.is_available():
            return batch_size
        
        try:
            # 获取可用内存
            if self.nvml_enabled:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                available_memory = memory_info.free
            else:
                # 使用PyTorch的内存信息估算
                total_memory = torch.cuda.get_device_properties(0).total_memory
                used_memory = torch.cuda.memory_allocated()
                available_memory = total_memory - used_memory
            
            # 根据可用内存调整批处理大小
            # 假设每个样本需要约1GB内存
            memory_per_sample = 1024**3  # 1GB
            max_batch_size = max(1, int(available_memory * 0.8 / memory_per_sample))
            
            optimal_batch_size = min(batch_size, max_batch_size)
            
            self.logger.info(f"优化批处理大小: {batch_size} -> {optimal_batch_size}")
            
            return optimal_batch_size
            
        except Exception as e:
            self.logger.error(f"批处理优化失败: {e}")
            return batch_size
    
    def clear_memory_cache(self):
        """清理内存缓存"""
        if not torch.cuda.is_available():
            return
        
        try:
            # 清理PyTorch缓存
            torch.cuda.empty_cache()
            
            # 重置内存统计
            torch.cuda.reset_peak_memory_stats()
            
            self.logger.info("GPU内存缓存已清理")
            
        except Exception as e:
            self.logger.error(f"清理内存缓存失败: {e}")
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计信息"""
        # 更新当前内存使用
        self.monitor_memory_usage()
        
        stats = self.performance_stats.copy()
        
        # 添加系统信息
        stats.update({
            'cpu_usage': psutil.cpu_percent(),
            'system_memory_usage': psutil.virtual_memory().percent,
            'gpu_available': torch.cuda.is_available(),
            'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A'
        })
        
        return stats
    
    def get_memory_history(self) -> list:
        """获取内存使用历史"""
        return self.memory_usage_history.copy()
    
    def check_memory_leak(self, threshold_mb: float = 100.0) -> bool:
        """检查内存泄漏"""
        if len(self.memory_usage_history) < 10:
            return False
        
        # 检查最近10个记录的内存使用趋势
        recent_usage = [record['memory_usage'] for record in self.memory_usage_history[-10:]]
        
        # 如果内存使用持续增长超过阈值，可能存在内存泄漏
        if len(recent_usage) >= 2:
            memory_increase = recent_usage[-1] - recent_usage[0]
            if memory_increase > threshold_mb:
                self.logger.warning(f"检测到可能的内存泄漏: 内存增长 {memory_increase:.2f}MB")
                return True
        
        return False
    
    def optimize_for_inference(self):
        """为推理优化GPU设置"""
        if not torch.cuda.is_available():
            return
        
        try:
            # 设置为推理模式
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # 禁用梯度计算
            torch.set_grad_enabled(False)
            
            # 优化内存使用
            self.optimize_memory_usage()
            
            self.logger.info("GPU推理优化完成")
            
        except Exception as e:
            self.logger.error(f"GPU推理优化失败: {e}")
    
    def reset_optimization(self):
        """重置优化设置"""
        if not torch.cuda.is_available():
            return
        
        try:
            # 重新启用梯度计算
            torch.set_grad_enabled(True)
            
            # 重置cudnn设置
            torch.backends.cudnn.benchmark = False
            
            self.logger.info("GPU优化设置已重置")
            
        except Exception as e:
            self.logger.error(f"重置GPU优化失败: {e}")
    
    def __del__(self):
        """析构函数"""
        if self.nvml_enabled:
            try:
                pynvml.nvmlShutdown()
            except:
                pass 