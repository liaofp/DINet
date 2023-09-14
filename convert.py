import argparse
import tensorflow as tf2
tf = tf2.compat.v1


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("src")
    parse.add_argument("dst")
    args = parse.parse_args()
    if args.src:
        raise Exception("lack of source model")
    if args.dst:
        raise Exception("lack of destination to save model")
    tf.disable_v2_behavior()
    
    # 加载 TensorFlow 1.x 的模型
    model_v1 = tf.keras.models.load_model(args.src)
    # 将模型保存为 TensorFlow 2.x 的格式
    tf.keras.models.save_model(model_v1, args.dst, save_format='tf')