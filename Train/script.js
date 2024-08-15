const classLabelsMap = {};
let classIndex = 0;
let imageTensors = [];
let labels = [];
let classImages = {};

let xs, ys; // Global variables to access image tensors and labels outside of training

async function prepareData() {
  const inputElement = document.getElementById("imageUpload");
  const files = inputElement.files;

  if (files.length === 0) {
    alert("Please upload some images");
    return;
  }

  console.log("Files selected:", files);

  // Loop through each file
  for (let file of files) {
    const imageTensor = await readImage(file);
    imageTensors.push(imageTensor);

    // Extract label and group the images
    const label = getLabel(file);
    labels.push(label);

    console.log("File:", file.name, "Label:", label);

    const className = getClassName(label);

    if (!classImages[className]) {
      classImages[className] = [];
    }
    classImages[className].push(file);
  }

  console.log("Image Tensors:", imageTensors.length);
  console.log("Labels:", labels);
  console.log("Class Images:", classImages);

  // Convert to Tensors for the entire dataset, outside of training
  xs = tf.stack(imageTensors).toFloat();
  ys = tf.tensor1d(labels, 'int32');

  console.log("XS Tensor Shape:", xs.shape);
  console.log("YS Tensor Shape:", ys.shape);

  // Display the images by class on the website
  displayImagesByClass(classImages);
}

function getLabel(file) {
  const className = file.name.match(/^[a-zA-Z]+/)[0];

  if (!(className in classLabelsMap)) {
    classLabelsMap[className] = classIndex;
    classIndex++;
  }

  console.log("Class Name:", className, "Label:", classLabelsMap[className]);
  return classLabelsMap[className];
}

function getClassName(label) {
  // Retrieve the class name from the label map
  const className = Object.keys(classLabelsMap).find(
    (key) => classLabelsMap[key] === label
  );
  console.log("Label:", label, "Class Name:", className);
  return className;
}

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
      const tensor = tf.browser
        .fromPixels(img)
        .resizeNearestNeighbor([150, 150])
        .toFloat()
        .div(255.0); // Normalize
      console.log(
        "Image loaded and converted to tensor with shape:",
        tensor.shape
      );
      resolve(tensor);
    };
    img.src = url;
  });
}

async function trainModel() {
  try {
    const model = createModel();

    // Split the data into training and test sets (80/20 split)
    const totalSize = xs.shape[0];
    const trainSize = Math.floor(totalSize * 0.8);
    const testSize = totalSize - trainSize;

    console.log("Train Size:", trainSize, "Test Size:", testSize);

    // Manually slice the tensors
    let xTrain = xs.slice([0, 0, 0, 0], [trainSize, xs.shape[1], xs.shape[2], xs.shape[3]]);
    let xTest = xs.slice([trainSize, 0, 0, 0], [testSize, xs.shape[1], xs.shape[2], xs.shape[3]]);
    let yTrain = ys.slice([0], [trainSize]);
    let yTest = ys.slice([trainSize], [testSize]);

    // Ensure that xTrain and xTest are float32 tensors
    xTrain = xTrain.toFloat();
    xTest = xTest.toFloat();
    yTrain = yTrain.toFloat();
    yTest = yTest.toFloat();

    console.log("xTrain Shape:", xTrain.shape);
    console.log("yTrain Shape:", yTrain.shape);
    console.log("xTest Shape:", xTest.shape);
    console.log("yTest Shape:", yTest.shape);

    // Custom callback for progress logging
    const logProgress = new tf.CustomCallback({
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch ${epoch + 1} ended`);
        console.log(`Loss: ${(logs.loss).toFixed(4)}`);
        console.log(`Accuracy: ${(logs.acc || 0).toFixed(4)}`);
        console.log(`Validation Loss: ${(logs.val_loss || 0).toFixed(4)}`);
        console.log(`Validation Accuracy: ${(logs.val_acc || 0).toFixed(4)}`);
      }
    });

    console.log("Starting model training...");

    // Train the model
    const history = await model.fit(xTrain, yTrain, {
      epochs: 10,
      batchSize: 16,
      validationData: [xTest, yTest],
      callbacks: [logProgress],
    });

    console.log("Training complete");

    // Save the model
    await model.save('downloads://mymodel');
  } catch (error) {
    console.error("Error during model training:", error);
  }
}


function createModel() {
  const model = tf.sequential();

  // Convolutional layer
  model.add(tf.layers.conv2d({
    inputShape: [150, 150, 3],
    filters: 16,
    kernelSize: 3,
    activation: 'relu'
  }));

  // MaxPooling layer
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

  // Flatten the output before the Dense layer
  model.add(tf.layers.flatten());

  // Dense (Fully Connected) layers
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));

  // Final output layer with softmax activation for multi-class classification
  model.add(tf.layers.dense({
    units: Object.keys(classLabelsMap).length,  // Number of classes
    activation: 'softmax'
  }));

  // Compile the model with Adam optimizer and sparse categorical cross-entropy loss
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy']
  });

  console.log("Model created and compiled");
  model.summary();

  return model;
}
