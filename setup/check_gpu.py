import tensorflow as tf 
import subprocess 
 
print("="*50) 
print("ENVIRONMENT CHECK") 
print("="*50) 
 
# TF version 
print(f"TensorFlow: {tf.__version__}") 
 
# CUDA and cuDNN versions (if available)
print(f"CUDA Available: {tf.config.list_physical_devices('GPU')}") 
if tf.test.is_built_with_cuda():
    print(f"CUDA Version: {tf.sysconfig.get_build_info()['cuda_version']}")
    print(f"cuDNN Version: {tf.sysconfig.get_build_info()['cudnn_version']}")

# GPU check 
gpus = tf.config.list_physical_devices('GPU') 
if gpus: 
    for gpu in gpus: 
        print(f"GPU Found: {gpu.name}") 
    # Get GPU details 
    result = subprocess.run( 
        ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', 
         '--format=csv,noheader'], 
        capture_output=True, text=True 
    ) 
    print(f"GPU Details: {result.stdout.strip()}") 
    
    # Enable memory growth 
    for gpu in gpus: 
        tf.config.experimental.set_memory_growth(gpu, True) 
    print("\n✅ GPU Ready for Training") 
else: 
    print("\n⚠️  No GPU found - Falling back to CPU") 
 
# Test operation 
with tf.device('/GPU:0' if gpus else '/CPU:0'): 
    a = tf.random.normal([1000, 1000]) 
    b = tf.random.normal([1000, 1000]) 
    c = tf.matmul(a, b) 
    print(f"Test operation result shape: {c.shape}") 
    print("✅ Device working correctly") 
