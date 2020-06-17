const express = require("express");
const path = require("path");
const ejs = require("ejs");
const bodyParser = require("body-parser");
const tf = require('@tensorflow/tfjs-node');
const { PythonShell } = require("python-shell");
const viewPath = path.join(__dirname, "/views");
const app = express();
const multer = require("multer");
const fs = require("fs");

const runDNN = require("./DNN");

app.use(express.static("public"));
app.set("view engine", "ejs");
app.set("views", viewPath);
app.use(bodyParser.urlencoded({
  extended: true
}));

app.get("/", function(req, res) {
  res.render("home");
});

app.get("/predict", function(req, res) {
  res.render("predict");
});

app.post("/predict", async function(req, res) {
  const firstBlood = req.body.firstBlood;
  const firstTower = req.body.firstTower;
  const firstBaron = req.body.firstBaron;
  const firstDragon = req.body.firstDragon;
  const dragonKills = req.body.dragonKills;
  const baronKills = req.body.baronKills;
  const wardPlaced = req.body.wardPlaced;
  const wardKills = req.body.wardKills;
  const kills = req.body.kills;
  const death = req.body.death;
  const assist = req.body.assist;
  const championDamageDealt = req.body.championDamageDealt;
  const totalGold = req.body.totalGold;
  const totalMinionKills = req.body.totalMinionKills;
  const totalLevel = req.body.totalLevel;
  const avgLevel = totalLevel / 5.0;
  const jungleMinionKills = req.body.jungleMinionKills;
  const killingSpree = req.body.killingSpree;
  const totalHeal = req.body.totalHeal;

  const data = [
    firstBlood,
    firstTower,
    firstBaron,
    firstDragon,
    dragonKills,
    baronKills,
    wardPlaced,
    wardKills,
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
  ];
  var prediction = await runDNN(data);
  var result = prediction.dataSync();
  res.render("prediction_result", {
    data: data,
    prediction: Number(result * 100).toPrecision(4)
  });
});

app.get("/generate", function(req, res) {
  res.render("generate");
});

app.post("/generate", async function(req, res) {
  const inputText = req.body.inputText;
  const length = req.body.length;
  const model = req.body.model;
  if(model == "LSTM") {
    let options = {
      args: [inputText, length]
    };
    PythonShell.run('LSTM.py', options, function(err, results) {
      if (err) throw err;
      // results is an array consisting of messages collected during execution
      var result = results[0].replace("generate.py", inputText);
      //console.log(result);
      res.send(result);
    });
  } else {
    let options = {
      args: [inputText, length]
    };
    PythonShell.run('GRU.py', options, function(err, results) {
      if (err) throw err;
      // results is an array consisting of messages collected during execution
      var result = results[0].replace("generate.py", inputText);
      //console.log(result);
      res.send(result);
    });
  }
});

app.get("/demo", function(req, res) {
  res.render("team_predict", {id: 1})
});

app.post("/demo", function(req, res) {
  var id = req.body.id;
  res.render("team_predict", {id: id});
});

app.post("/team_predict", async function(req, res) {
  const inputOption = req.body.inputOption;
  let options = {
    args: [inputOption]
  };
  PythonShell.run('team_predict.py', options, function(err, results) {
    if (err) throw err;
    var result = results;
    res.send(result);
  });
});

app.get("/transfer", function(req, res) {
  res.render("transfer", {id: 1});
});

app.post("/transfer", function(req, res) {
  const id = req.body.id;
  res.render("transfer", {id: id});
});

app.get("/transfer_demo", function(req, res) {
  PythonShell.run('VGG19_2.py', null, async function(err, results) {
    res.send("Finished");
  });
});

const storage = multer.diskStorage({
  destination: function(req, file, callback){
    var dir = "./public/uploads";
    if(!fs.existsSync(dir)) {
      fs.mkdirSync(dir);
    }
    callback(null, dir); // set the destination
  },
  filename: function(req, file, callback){
    callback(null, file.fieldname + ".png"); // set the file name and extension
  }
});

const upload = multer({storage: storage});

var cpUpload = upload.fields([{ name: 'content', maxCount: 1 }, { name: 'style', maxCount: 1 }]);
app.post("/create", cpUpload, function(req, res, next) {
  PythonShell.run('VGG19.py', null, async function(err, results) {
    if(!err) {
      res.send("Finished");
    }
  });
});

////////////////////////////////// Port Connection Section //////////////////////////////////

let port = process.env.PORT;
if (port == null || port == "") {
  port = 3000;
}

app.listen(port, function() {
  console.log("Server started on port " + port + " successfully!");
});
