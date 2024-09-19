const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
let mobilenet = null;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let CLASS_NAMES = null;
let model = null;
const classLabelsMap = {};
let classIndex = 0;
let imageTensors = [];
let labels = [];
let classImages = {};
let xs, ys; // Global variables to access image tensors and labels
const dwnBtn = document.getElementById('downloadBtn');

async function loadMobileNetFeatureModel() {
  const URL =
    "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1";

  mobilenet = await tf.loadGraphModel(URL, { fromTFHub: true });
  console.log("MobileNet v3 loaded successfully!");

  tf.tidy(function () {
    let answer = mobilenet.predict(
      tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3])
    );
    console.log(answer.shape); // Should output [1, 1024]
  });
}

loadMobileNetFeatureModel();

function load_model() {
  model = tf.sequential();

  // Dense layers for classification after MobileNet feature extraction
  model.add(
    tf.layers.dense({ inputShape: [1024], units: 128, activation: "relu" })
  );
  model.add(
    tf.layers.dense({ units: CLASS_NAMES.length, activation: "softmax" })
  );

  model.summary();

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });
}

async function prepareData() {
  const inputElement = document.getElementById("imageUpload");
  const files = inputElement.files;

  if (files.length === 0) {
    alert("Please upload some images");
    return;
  }

  //   console.log("Files selected:", files);

  for (let file of files) {
    const imageTensor = await readImage(file);
    imageTensors.push(imageTensor);

    const label = getLabel(file);
    labels.push(label);

    // no need to log
    // console.log("File:", file.name, "Label:", label);

    const className = getClassName(label);

    if (!classImages[className]) {
      classImages[className] = [];
    }
    classImages[className].push(file);
  }

  CLASS_NAMES = Object.keys(classImages);
  // console.log("class names : ",CLASS_NAMES);

  // log the values to check the dimensions , tensor of the image labels , classwise images ordered
  //   console.log("Image Tensors:", imageTensors.length);
  //   console.log("Labels:", labels);
  //   console.log("Class Images:", classImages);

  // Convert image tensors and labels to proper tensors
  xs = tf.stack(imageTensors); // Stack all image feature vectors into a single tensor
  ys = tf.oneHot(tf.tensor1d(labels, "int32"), CLASS_NAMES.length); // One-hot encode the labels

  //   log the prepare data (x (features), y(target))
  //   console.log("XS Tensor Shape:", xs.shape); // Should be [batch_size, 1024]
  //   console.log("YS Tensor Shape:", ys.shape); // Should be [batch_size, num_classes]

  // Display the images by class on the website
  displayImagesByClass(classImages);
  // load the model after getting target variables length from CLASS_NAMES variable

  load_model();
}

// Extract the class label from the file name
function getLabel(file) {
  const className = file.name.match(/^[a-zA-Z]+/)[0];

  if (!(className in classLabelsMap)) {
    classLabelsMap[className] = classIndex;
    classIndex++;
  }

  //   console.log("Class Name:", className, "Label:", classLabelsMap[className]);
  return classLabelsMap[className];
}

function getClassName(label) {
  // Retrieve the class name from the label map
  const className = Object.keys(classLabelsMap).find(
    (key) => classLabelsMap[key] === label
  );
  //   print numbers according to the class names
  //   console.log("Label:", label, "Class Name:", className);

  return className;
}

// Display uploaded images by class on the webpage
function displayImagesByClass(classImages) {
  const container = document.getElementById("imageContainers");
  container.innerHTML = ""; // Clear the previous content

  // Loop through each class and create a section for it
  for (let className in classImages) {
    const classSection = document.createElement("div");
    classSection.className = "class-section";

    const title = document.createElement("h3");
    title.innerText = className;
    classSection.appendChild(title);

    const classContainer = document.createElement("div");
    classContainer.className = "class-container";

    // Display all images in this class
    classImages[className].forEach((file) => {
      const img = document.createElement("img");
      const url = URL.createObjectURL(file);
      img.src = url;
      classContainer.appendChild(img);
    });

    classSection.appendChild(classContainer);
    container.appendChild(classSection);
  }
}

async function readImage(file) {
  const img = document.createElement("img");
  const url = URL.createObjectURL(file);

  return new Promise((resolve) => {
    img.onload = () => {
      let imageFeatures = tf.tidy(function () {
        let videoFrameAsTensor = tf.browser.fromPixels(img);
        let resizedTensorFrame = tf.image.resizeBilinear(
          videoFrameAsTensor,
          [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
          true
        );
        let normalizedTensorFrame = resizedTensorFrame.div(255);
        return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
      });
      //   console.log(imageFeatures.shape); // Should output [1024]
      resolve(imageFeatures);
    };
    img.src = url;
  });
}

// Train the model with the prepared data
async function trainModel() {
  let results = await model.fit(xs, ys, {
    shuffle: true,
    batchSize: 30,
    epochs: 2,
    callbacks: { onEpochEnd: logProgress },
  });
  alert("model trained sucessfully");
  dwnBtn.attributes.removeNamedItem("disabled");
  xs.dispose();
  ys.dispose();
}

// Log training progress
function logProgress(epoch, logs) {
  var container = document.getElementById("trainResultContainer");
  var p = document.createElement("p");
  p.innerText = `Data for epoch ${epoch} - loss :  ${logs.loss.toFixed(
    3
  )} accuracy : ${logs.acc.toFixed(3)}`;
  container.appendChild(p);
  console.log("Data for epoch " + epoch, logs);
}

function downloadModel() {
    const formContainer = document.getElementById("formContainer");
  
    // Clear the previous form if it exists
    formContainer.innerHTML = "";
  
    // Create a form dynamically
    const form = document.createElement("form");
  
    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = "your-model"; // Placeholder text for model name
    input.id = "modelInput"; // Set an ID for easy reference
  
    const button = document.createElement("button");
    button.type = "button"; // Button type is 'button' to prevent form submission
    button.innerText = "Download Model";
  
    // Append input and button to the form
    form.appendChild(input);
    form.appendChild(button);
    formContainer.appendChild(form);
  
    // Add event listener to the button
    button.addEventListener("click", async function () {
      const inputValue = document.getElementById("modelInput").value.trim();
      
      if (inputValue) {
        try {
          // Save the model as a file with the provided name
          await model.save(`downloads://${inputValue}`);
          alert("Model downloaded successfully!");
        } catch (error) {
          console.error("Error downloading model:", error);
          alert("Error downloading model.");
        }
      } else {
        alert("Please provide a valid model name.");
      }
    });
  }
  