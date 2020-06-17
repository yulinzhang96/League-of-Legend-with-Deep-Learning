const tf = require('@tensorflow/tfjs');
const tfn = require("@tensorflow/tfjs-node");
const fs = require('fs');
const BiLSTM_URL = tfn.io.fileSystem('./models/BiLSTM/model.json');
const vocab = require('./models/BiLSTM/vocab.json')



async function runBiLSTM(data, next_words_length) {
  const model = await tf.loadLayersModel(BiLSTM_URL);
  var seed_text = data;
  var length = next_words_length;
  print(model.summary());
  //console.log(vocab['also']);
}

module.exports = runBiLSTM;
