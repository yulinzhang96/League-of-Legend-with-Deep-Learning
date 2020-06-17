const tf = require('@tensorflow/tfjs');
const tfn = require("@tensorflow/tfjs-node");
const DNN_URL = tfn.io.fileSystem('./models/DNN/model.json');

async function runDNN(data) {
  const model = await tf.loadLayersModel(DNN_URL);
  var firstBlood = (data[0] - 0.0)/(1.0 - 0.0);
  var firstTower = (data[1] - 0.0)/(1.0 - 0.0);
  var firstBaron = (data[2] - 0.0)/(1.0 - 0.0);
  var firstDragon = (data[3] - 0.0)/(1.0 - 0.0);
  if(data[4] > 7) {
    data[4] = 7;
  }
  var dragonKills = (data[4] - 0.0)/(7.0 - 0.0);
  if(data[5] > 5) {
    data[5] = 5;
  }
  var baronKills = (data[5] - 0.0)/(5.0 - 0.0);
  if(data[6] > 248) {
    data[6] = 248;
  }
  var wardPlaced = (data[6] - 0.0)/(248.0 - 0.0);
  if(data[7] > 118) {
    data[7] = 118;
  }
  var wardkills = (data[7] - 0.0)/(118.0 - 0.0);
  if(data[8] > 116) {
    data[8] = 116;
  }
  var kills = (data[8] - 0.0)/(116.0 - 0.0);
  if(data[9] > 117) {
    data[9] = 117;
  }
  var death = (data[9] - 0.0)/(117.0 - 0.0);
  if(data[10] > 256) {
    data[10] = 256;
  }
  var assist = (data[10] - 0.0)/(256.0 - 0.0);
  if(data[11] > 381484) {
    data[11] = 381484;
  }
  var championDamageDealt = (data[11] - 0.0)/(381484.0 - 0.0);
  if(data[12] > 141692) {
    data[12] = 141692;
  }
  var totalGold = (data[12] - 0.0)/(141692.0 - 0.0);
  if(data[13] > 1514) {
    data[13] = 1514;
  }
  var totalMinionKills = (data[13] - 0.0)/(1514.0 - 0.0);
  if(data[14] > 145) {
    data[14] = 145;
  }
  var totalLevel = (data[14] - 0.0)/(145.0 - 0.0);
  if(data[15] > 29) {
    data[15] = 29;
  }
  var avgLevel = (data[15] - 0.0)/(29.0 - 0.0);
  if(data[16] > 488) {
    data[16] = 488;
  }
  var jungleMinionKills = (data[16] - 0.0)/(488.0 - 0.0);
  if(data[17] > 31) {
    data[17] = 31
  }
  var killingSpree = (data[17] - 0.0)/(31.0 - 0.0);
  if(data[18] > 261707) {
    data[18] = 261707;
  }
  var totalHeal = (data[18] - 0.0)/(261707.0 - 0.0);

  data = tf.tensor2d([
    firstBlood,
    firstTower,
    firstBaron,
    firstDragon,
    dragonKills,
    baronKills,
    wardPlaced,
    wardkills,
    kills,
    death,
    assist,
    championDamageDealt,
    totalGold,
    totalMinionKills,
    totalLevel,
    avgLevel,
    jungleMinionKills,
    killingSpree,
    totalHeal
  ], [1,19]);
  result = model.predict(data);
  return result;
}

module.exports = runDNN;
