const { loadTFLiteModel } = require('tfjs-tflite-node');
const tf = require('@tensorflow/tfjs');
const tfnode = require("@tensorflow/tfjs-node")
const { Image } = require('canvas');
const fetch = require('node-fetch');
const fs = require('fs');
const cv2 = require('opencv4nodejs');
IMAGE_FILE = "./hand.jpg";

let model;

const loadModel = async () => {
    console.log('Loading Model...');
    model = await loadTFLiteModel('./model.tflite');
    console.log('Model loaded!');
};

async function predict() {
  console.log('Processing image...');
  console.log('Model:', model);
  console.log('Model:', model);

  const img = fs.readFileSync(IMAGE_FILE);
  const decode = tfnode.node.decodeImage(new Uint8Array(img), 1);
  
  const resizeImg = tf.image
    .resizeNearestNeighbor(decode, [32, 100])
    .toFloat();
  console.log('Resized Image:', resizeImg.shape);
  
  const permutedImg = resizeImg.transpose([2, 0, 1]); // Transpose dimensions
  console.log('Permuted Image:', permutedImg.shape);
  
  const offset = tf.scalar(127.5);
  const normalizedImg = permutedImg.sub(offset).div(offset);
  console.log('Normalized Image:', normalizedImg.shape);
  
  const inputImage = normalizedImg.expandDims(0); // Add batch dimension
  console.log('inputImage:', inputImage.shape);
  
  // Run the inference
  const outputTensor = model.predict(inputImage);
  console.log('outputTensor:', outputTensor);
}

async function run() {
  try {
    await loadModel();
    await predict();
  } catch (error) {
    console.error('Error:', error);
  }
}

run();
