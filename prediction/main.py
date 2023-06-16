import os
import pathlib
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd

from flask import Flask, request, jsonify
import json

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

ABSOLUTE_PATH = os.path.abspath(os.path.dirname(__file__))

def load_image_into_numpy_array(path):
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# List checkpoints in a file and sort from oldest to newest
checkpoint_dir = ABSOLUTE_PATH + '/content/fine_tuned_model/checkpoint' # Ganti path ini jadi lokasi file "ckpt-x.index"
filenames = list(pathlib.Path(checkpoint_dir).glob('*.index'))
filenames.sort()

# Get config and use the newest checkpoint
pipeline_config = ABSOLUTE_PATH +'/content/fine_tuned_model/pipeline.config' # Ganti path ini jadi lokasi fine_tuned_model/pipeline.config
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
model_dir = str(filenames[-1]).replace('.index','')
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(model_dir))

# Inference function
def get_model_detection_function(model):
  
  @tf.function
  def detect_fn(image):
    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)
    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn

detect_fn = get_model_detection_function(detection_model)
label_map_path = ABSOLUTE_PATH  + '/content/food-ingredients_label_map.pbtxt' # Ganti path ini jadi lokasi label_map.pbtxt
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)




# route()
def predict_image(image_path):
  # image_path = './kentang.jpg' # Ganti path ini jadi path gambar yang ingin diprediksi
  image_np = load_image_into_numpy_array(image_path)
  input_tensor = tf.convert_to_tensor(
      np.expand_dims(image_np, 0), dtype=tf.float32)
  detections, predictions_dict, shapes = detect_fn(input_tensor)
  # print(detections['detection_boxes'][0].numpy())
  predictScores= detections['detection_scores'][0].numpy()
  label_id_offset = 1
  prediction_result = (detections['detection_classes'][0].numpy() + label_id_offset).astype(int)

  result = [category_index[prediction_result[i]]['name'] for i, score in enumerate(predictScores) if score >= 0.9]
  result_clear_duplicate = sorted(set(result))
  return result_clear_duplicate


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
  file = request.files['image']
  if file:
      print(file)
      image = ABSOLUTE_PATH + "/tmp/" + file.filename # Temp dir in app engine
      file.save(image)
      prediction = predict_image(image)
      response_body = {"objects": prediction}
      os.remove(image)
      return jsonify(response_body)



recipes_df = pd.read_csv(ABSOLUTE_PATH + '/content/recipes_data.csv')
sim_df = pd.read_csv(ABSOLUTE_PATH + '/content/recipe_similarity.csv')
sim_df = pd.DataFrame(sim_df.values[:, 1:], index=recipes_df['id'], columns=recipes_df['id'])

def recipe_recommendations(recipe_id, similarity_data, k=5):
    index = similarity_data.loc[:,recipe_id].to_numpy().argpartition(range(-1, -k, -1))
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    closest = closest.drop(recipe_id, errors='ignore')
    return pd.DataFrame(closest)

@app.route('/recomendation', methods=['GET'])
def recomend():
  recipe_id = request.args.get('recipe')
  similiar_recipes=recipe_recommendations(recipe_id, sim_df).values.reshape(-1).tolist()

  response_body = {
    "recipes" : similiar_recipes
  }
  return jsonify(response_body)

  
if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080, debug=True)
  print("Listening to http://localhost:8080")