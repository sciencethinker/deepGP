import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # 将第一个GPU设置为不可见
  try:
    tf.config.set_visible_devices(gpus[1:], 'GPU')  # 假设gpus[0]是第一个GPU
  except RuntimeError as e:
    # 可见设备必须在GPU初始化之前设置
    print(e)