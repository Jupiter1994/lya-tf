import tensorflow as tf

print('Available devices:')
print(tf.config.list_physical_devices())

device_name = tf.test.gpu_device_name()
print('device_name:', device_name)