"""
性能监控
"""
import time
import psutil
import GPUtil
from functools import wraps

def monitor_performance(func):
    """性能监控装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 开始前的状态
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # GPU状态
        gpus = GPUtil.getGPUs()
        start_gpu_memory = gpus[0].memoryUsed if gpus else 0
        
        # 执行函数
        result = func(*args, **kwargs)
        
        # 执行后的状态
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_gpu_memory = gpus[0].memoryUsed if gpus else 0
        
        # 打印性能信息
        print(f"\n性能报告 - {func.__name__}")
        print(f"执行时间: {end_time - start_time:.2f} 秒")
        print(f"内存使用: {end_memory - start_memory:.2f} MB")
        if gpus:
            print(f"GPU内存: {end_gpu_memory - start_gpu_memory:.2f} MB")
        
        return result
    return wrapper