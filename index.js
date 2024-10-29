import * as jsfeat from "jsfeat";
// Example usage of jsfeat
console.log(jsfeat);

const loadModelButton = document.getElementById('loadModelButton');
const predictImageButton = document.getElementById('predictImageButton');
const predictVideoButton = document.getElementById('predictVideoButton');
const pauseVideoButton = document.getElementById('pauseVideoButton'); // Pause Button
const predictWebcamButton = document.getElementById('predictWebcamButton');
const pauseWebcamButton = document.getElementById('pauseWebcamButton'); // Pause Webcam Button
const backendSelect = document.getElementById('backendSelect');
const statusMessage = document.getElementById('statusMessage');
const displayCanvas = document.getElementById('outputCanvas');
const displayCtx = displayCanvas.getContext('2d');
const inputCanvas = document.getElementById('inputCanvas'); // Hidden Canvas for Inference
const inputCtx = inputCanvas.getContext('2d');
const imagePredictionOutput = document.getElementById('imagePredictionOutput');
const videoPredictionOutput = document.getElementById('videoPredictionOutput');
const webcamPredictionOutput = document.getElementById('webcamPredictionOutput');
const backendDisplay = document.getElementById('backendDisplay');
const videoContainer = document.getElementById('videoContainer');
const inputVideo = document.getElementById('inputVideo'); // Hidden Video Element
const overlayCanvas = document.getElementById('overlayCanvas');
const overlayCtx = overlayCanvas.getContext('2d');

// Webcam Elements
const webcamContainer = document.getElementById('webcamContainer');
const webcamDisplayCanvas = document.getElementById('webcamDisplayCanvas');
const webcamDisplayCtx = webcamDisplayCanvas.getContext('2d');
const webcamVideo = document.getElementById('webcamVideo'); // Hidden Webcam Video Element
const webcamOverlayCanvas = document.getElementById('webcamOverlayCanvas');
const webcamOverlayCtx = webcamOverlayCanvas.getContext('2d');

let model = null;
let isVideoPaused = false; // To track video pause state
let isWebcamPaused = false; // To track webcam pause state
let isProcessingVideo = false;  // To prevent overlapping inferences for video
let isProcessingWebcam = false; // To prevent overlapping inferences for webcam

// **FPS Tracking Variables**
let fpsVideo = 0;
let frameCountVideo = 0;
let lastFpsUpdateTimeVideo = performance.now();

let fpsWebcam = 0;
let frameCountWebcam = 0;
let lastFpsUpdateTimeWebcam = performance.now();

// **Image Paths Array**
const imagePaths = ['sample1.jpeg', 'sample2.jpg', 'sample3.jpg','temp.webp'];  // Add your image paths here
// ====================== Temporal Smoothing ======================
// Exponential Moving Average smoothing factor (alpha)
const SMOOTHING_ALPHA = 0.5;

// Array to hold smoothed keypoints (2 * point_count)
let smoothed_xy = new Float32Array(20 * 2).fill(0); // Initialized to zeros

// Define the custom L2 Regularizer
class L2 extends tf.serialization.Serializable {
    constructor(config) {
        super();
        this.l2 = config.l2 || 1e-5;
    }
    apply(x) {
        return tf.tidy(() => {
            const square = tf.square(x);
            const sum = tf.sum(square);
            return tf.mul(this.l2 / 2, sum);
        });
    }
    getConfig() {
        return { l2: this.l2 };
    }
    static get className() {
        return 'L2';  // Must match the regularizer name in Keras
    }
    static fromConfig(cls, config) {
        return new cls(config);
    }
}
tf.serialization.registerClass(L2);

// ====================== Point Tracking Variables Initialization ======================
const tracking = {
    thresholdFloor: 25.0,
    thresholdSeil: 50.0,
    skipCall: 1, // Number of consecutive tracked frames before forcing model inference
    win_size: 20,
    epsilon: 0.01,
    min_eigen: 0.001,
    max_iterations: 30,
    totalDifference: 0,

    curr_img_pyr: new jsfeat.pyramid_t(3),
    prev_img_pyr: new jsfeat.pyramid_t(3),
    curr_img: new jsfeat.matrix_t(640, 640, jsfeat.U8_t | jsfeat.C1_t),
    prev_img: new jsfeat.matrix_t(640, 640, jsfeat.U8_t | jsfeat.C1_t),

    point_count: 20, // Total keypoints
    point_status: new Uint8Array(20),
    prev_xy: new Float32Array(20 * 2),
    curr_xy: new Float32Array(20 * 2),

    threshCount: 0,
    consecutiveTrackedFrames: 0
};

// Allocate image pyramids
tracking.curr_img_pyr.allocate(640, 640, jsfeat.U8_t | jsfeat.C1_t);
tracking.prev_img_pyr.allocate(640, 640, jsfeat.U8_t | jsfeat.C1_t);

// ====================== Point Tracking Helper Functions ======================

// Function to convert image to grayscale and copy to jsfeat matrix
function prepareGrayImage(imageElement, jsfeatMatrix, ctx) {
    // Draw the image to a temporary canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 640;
    tempCanvas.height = 640;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(imageElement, 0, 0, 640, 640);
    const imageData = tempCtx.getImageData(0, 0, 640, 640);
    
    // Convert to grayscale using jsfeat
    jsfeat.imgproc.grayscale(imageData.data, 640, 640, jsfeatMatrix);
}

// Function to initialize keypoints from model prediction
// function initializeKeypoints(finalLandmarks, finalBoxes) {
//     // Assuming finalLandmarks is an array of landmarks per detection
//     // and finalBoxes is an array of bounding boxes per detection
//     // Here, we take the first detection for simplicity
    
    
//     if (finalLandmarks.length > 0) {
//         // console.log("initializeKeypoints - Number of Classes Detected:", finalLandmarks.length);
        
//         let index = 0; // To keep track of the current position in tracking.prev_xy

//         // Iterate through each class's landmarks
//         for (let classIndex = 0; classIndex < finalLandmarks.length; classIndex++) {
//             const classKeypoints = finalLandmarks[classIndex]; // Array of 8 keypoints for this class

//             // Iterate through each keypoint in the class
//             for (let kpIndex = 0; kpIndex < classKeypoints.length; kpIndex++) {
//                 if (index >= tracking.prev_xy.length / 2) {
//                     console.warn('initializeKeypoints - Exceeded the maximum number of keypoints.');
//                     break; // Prevent writing beyond the allocated array
//                 }

//                 // Assign x and y coordinates to the tracking.prev_xy array
//                 tracking.prev_xy[index * 2] = classKeypoints[kpIndex].x;
//                 tracking.prev_xy[index * 2 + 1] = classKeypoints[kpIndex].y;
//                 index++;
//             }
//         }

//         // Reset threshold count after initializing keypoints
//         tracking.threshCount = 0;

//         // console.log("initializeKeypoints - Keypoints Initialized:", tracking.prev_xy);
//     } else {
//         console.log("initializeKeypoints - No landmarks to initialize.");
//     }
// }
/**
 * Initializes the tracking.prev_xy array with bounding box points and keypoints for each class.
 * Ensures that the sequence is maintained as:
 * [Class 0 Bounding Box (x1, y1, x2, y2), Class 0 Keypoints (8x [x, y]),
 *  Class 1 Bounding Box (x1, y1, x2, y2), Class 1 Keypoints (8x [x, y])]
 * 
 * If a class is not present in classArray, its bounding box and keypoints are filled with zeros.
 *
 * @param {Array<Array<{x: number, y: number}>>} finalLandmarks - Array of landmarks per detection.
 * @param {Array<{x1: number, y1: number, x2: number, y2: number}>} finalBoxes - Array of bounding boxes per detection.
 * @param {Array<number>} classArray - Array indicating which classes are present, e.g., [0, 1], [0], [1].
 */
function initializeKeypoints(finalLandmarks, finalBoxes, classArray) {
    // Define the list of desired classes in order: Class 0 (Left Shoe), Class 1 (Right Shoe)
    const desiredClasses = [0, 1]; // Adjust as needed for more classes

    // Check if there are any detections
    if (finalLandmarks.length > 0) {
        console.log("initializeKeypoints - Number of Classes Detected:", finalLandmarks.length);

        // Initialize all points to zero to handle missing classes
        tracking.prev_xy.fill(0);

        // Iterate through each desired class in order
        for (let cls of desiredClasses) {
            if (classArray.includes(cls)) {
                // Find the index of the current class in classArray
                const detectionIndex = classArray.indexOf(cls);

                // Retrieve the corresponding landmarks and bounding box
                const classKeypoints = finalLandmarks[detectionIndex]; // Array of 8 keypoints for this class
                const classBox = finalBoxes[detectionIndex]; // Corresponding bounding box for this class

                console.log(`initializeKeypoints - Processing Class ${cls}:`, classBox, classKeypoints);

                // Calculate the base index in tracking.prev_xy for this class
                // Each class has 2 bounding box points + 8 keypoints = 10 points
                const basePointIndex = cls * 10; // Class 0: 0, Class 1: 10

                // Assign Bounding Box Points
                tracking.prev_xy[basePointIndex * 2] = classBox.x1;
                tracking.prev_xy[basePointIndex * 2 + 1] = classBox.y1;

                tracking.prev_xy[(basePointIndex + 1) * 2] = classBox.x2;
                tracking.prev_xy[(basePointIndex + 1) * 2 + 1] = classBox.y2;

                // Assign Keypoints
                for (let kp = 0; kp < 8; kp++) {
                    tracking.prev_xy[(basePointIndex + 2 + kp) * 2] = classKeypoints[kp].x;
                    tracking.prev_xy[(basePointIndex + 2 + kp) * 2 + 1] = classKeypoints[kp].y;
                }
            } else {
                // If the class is not present, its points remain as zeros
                const basePointIndex = cls * 10; // Class 0: 0, Class 1: 10

                console.log(`initializeKeypoints - Class ${cls} not detected. Filling with zeros.`);

                // Optional: Explicitly set to zero (already done by fill)
                tracking.prev_xy[basePointIndex * 2] = 0;
                tracking.prev_xy[basePointIndex * 2 + 1] = 0;

                tracking.prev_xy[(basePointIndex + 1) * 2] = 0;
                tracking.prev_xy[(basePointIndex + 1) * 2 + 1] = 0;

                for (let kp = 0; kp < 8; kp++) {
                    tracking.prev_xy[(basePointIndex + 2 + kp) * 2] = 0;
                    tracking.prev_xy[(basePointIndex + 2 + kp) * 2 + 1] = 0;
                }
            }
        }

        // Reset threshold count after initializing keypoints
        tracking.threshCount = 0;

        console.log("initializeKeypoints - Keypoints Initialized:", tracking.prev_xy);
    } else {
        console.log("initializeKeypoints - No landmarks to initialize.");
    }
}


// Function to update keypoints after model prediction
function updateKeypoints(finalLandmarks, finalBoxes) {
    if (finalLandmarks.length > 0) {
        const landmarks = finalLandmarks[0]; // [16] array for 8 keypoints (x, y)
        for (let i = 0; i < 16; i++) {
            tracking.prev_xy[i * 2] = landmarks[i].x;
            tracking.prev_xy[i * 2 + 1] = landmarks[i].y;
        }
        tracking.threshCount = 0;
    }
}

// Function to track keypoints using jsfeat
function trackKeypoints() {
    // Perform optical flow tracking
    jsfeat.optical_flow_lk.track(
        tracking.prev_img_pyr,
        tracking.curr_img_pyr,
        tracking.prev_xy,
        tracking.curr_xy,
        tracking.point_count,
        tracking.win_size | 0,
        tracking.max_iterations | 0,
        tracking.point_status,
        tracking.epsilon,
        tracking.min_eigen
    );
    console.log('Tracking:: tracking.prev_xy',tracking.prev_xy);
    console.log('Tracking:: tracking.curr_xy',tracking.curr_xy);
    console.log('Tracking:: tracking.point_count',tracking.point_count);
    console.log('Tracking:: tracking.curpoint_statusr_xy',tracking.point_status);
    // Validate tracked points
    tracking.totalDifference = 0;
    let valid = true;
    for (let i = 0; i < tracking.point_count; i++) {
        // if (tracking.point_status[i] === 1) {
            const prevX = tracking.prev_xy[i * 2];
            const prevY = tracking.prev_xy[i * 2 + 1];
            const currX = tracking.curr_xy[i * 2];
            const currY = tracking.curr_xy[i * 2 + 1];

            if (isNaN(currX) || isNaN(currY)) {
                valid = false;
                break;
            }

            tracking.totalDifference += Math.abs(prevX - currX) + Math.abs(prevY - currY);
        // } else {
        //     valid = false;
        //     break;
        // }
    }
    console.log("tracking.totalDifference ",tracking.totalDifference,)
    return valid && tracking.totalDifference < tracking.thresholdSeil;
}

// ====================== Model Loading Function ======================
async function loadModel() {
    try {
        const backend = backendSelect.value;
        console.log(`Setting backend to ${backend}...`);
        statusMessage.innerText = 'Setting backend...';
        await tf.setBackend(backend);
        await tf.ready();
        backendDisplay.innerText = `Backend: ${backend}`;
        console.log(`Backend set to ${backend}.`);

        statusMessage.innerText = 'Loading model...';
        console.log('Loading model...');
        // Replace 'last_web_model/model.json' with the path to your model best_v10_160_shoe_web_model best_web_model_81_224
        model = await tf.loadGraphModel('best_web_model_81_224/model.json', {
            customObjects: { 'L2': L2 }
        });

        // Log model outputs for verification
        console.log('Model loaded successfully.');
        console.log('Model Outputs:', model.outputs);

        statusMessage.innerText = 'Model loaded successfully.';
        // Enable inference buttons
        predictImageButton.disabled = false;
        predictVideoButton.disabled = false;
        predictWebcamButton.disabled = false;
    } catch (error) {
        console.error('Error loading model:', error);
        statusMessage.innerText = `Error loading model: ${error.message}`;
    }
}

// ====================== Detection Processing Functions ======================

// Shared Function: Process Detections (Image and Video)
async function processDetections(boxesData, scoresData, landmarksData, scaleFactor) {
    const t0 = performance.now(); // Start timing

    // Define confidence threshold
    const confidenceThreshold = 0.5;

    // Prepare combined scores, class labels, and landmarks
    const detectionsByClass = {};

    for (let i = 0; i < scoresData.length; i++) {
        for (let c = 0; c < scoresData[i].length; c++) {
            if (scoresData[i][c] >= confidenceThreshold) {
                if (!detectionsByClass[c]) {
                    detectionsByClass[c] = {
                        boxes: [],
                        scores: [],
                        landmarks: []
                    };
                }
                detectionsByClass[c].boxes.push(boxesData[i]);
                detectionsByClass[c].scores.push(scoresData[i][c]);
                detectionsByClass[c].landmarks.push(landmarksData[i]);
            }
        }
    }

    const finalBoxes = [];
    const finalScores = [];
    const finalClassLabels = [];
    const finalLandmarks = [];

    const nmsPromises = [];

    for (const cls in detectionsByClass) {
        const clsDetections = detectionsByClass[cls];
        if (clsDetections.boxes.length === 0) continue;

        const clsBoxesTensor = tf.tensor2d(clsDetections.boxes);
        const clsScoresTensor = tf.tensor1d(clsDetections.scores);

        const maxOutputSize = 1; // Only one detection per class
        const iouThreshold = 0.45;
        const scoreThresholdNMS = 0.5;

        const nmsPromise = tf.image.nonMaxSuppressionAsync(
            clsBoxesTensor,
            clsScoresTensor,
            maxOutputSize,
            iouThreshold,
            scoreThresholdNMS
        ).then(clsNmsIndices => {
            return clsNmsIndices.array().then(clsNmsIndicesData => {
                clsNmsIndicesData.forEach(idx => {
                    finalBoxes.push(clsDetections.boxes[idx]);
                    finalScores.push(clsDetections.scores[idx]);
                    finalClassLabels.push(parseInt(cls));
                    finalLandmarks.push(clsDetections.landmarks[idx]);
                });
            }).finally(() => {
                tf.dispose([clsBoxesTensor, clsScoresTensor]);
            });
        });

        nmsPromises.push(nmsPromise);
    }

    await Promise.all(nmsPromises);

    const processTime = performance.now() - t0; // End timing

    console.log(`Process Detections Time: ${processTime.toFixed(1)} ms`);

    return {
        finalBoxes: finalBoxes,
        finalScores: finalScores,
        finalClassLabels: finalClassLabels,
        finalLandmarks: finalLandmarks,
        processTime: processTime // Return the time taken
    };
}

// Shared Function: Scale Detections
function scaleDetections(finalBoxes, finalLandmarks, scaleFactor) {
    // Adjust boxes to original image size (160 -> display size)
    const t0 = performance.now(); // Start timing
    const adjustedBoxes = finalBoxes.map(box => {
        const [center_x, center_y, width, height] = box;
        // Convert from normalized center coordinates to corner coordinates
        const x1 = (center_x - width / 2) * scaleFactor;
        const y1 = (center_y - height / 2) * scaleFactor;
        const x2 = (center_x + width / 2)  * scaleFactor;
        const y2 = (center_y + height / 2)  * scaleFactor;
        return { x1: x1, y1: y1, x2: x2, y2: y2 };
    });

    // Adjust keypoints to original image size (160 -> display size)
    const adjustedKeypoints = finalLandmarks.map(landmark => {
        const keypoints = [];
        for (let j = 0; j < 8; j++) { // Assuming 8 keypoints, each with x and y
            const x = landmark[j * 2]  * scaleFactor;
            const y = landmark[j * 2 + 1] * scaleFactor;
            keypoints.push({ x: x, y: y });
        }
        return keypoints;
    });
    const scaleTime = performance.now() - t0; // End timing
    console.log(`Scale Detections Time: ${scaleTime.toFixed(1)} ms`);

    return { adjustedBoxes, adjustedKeypoints, scaleTime };
}

// Shared Function: Format Output Text
function formatOutputText(inferenceTime, finalBoxes, finalScores, finalClassLabels, adjustedBoxes, adjustedKeypoints, fps = null) {
    let outputText = `Inference Time: ${inferenceTime.toFixed(1)} ms\nTotal Detections after NMS: ${finalBoxes.length}\n\n`;
    const t0 = performance.now(); // Start timing
    for (let index = 0; index < finalBoxes.length; index++) {
        const box = finalBoxes[index];
        const score = finalScores[index];
        const classLabel = finalClassLabels[index];
        const keypoints = adjustedKeypoints[index];

        // Check if variables are defined and are numbers
        if (typeof inferenceTime !== 'number') {
            console.error('Inference time is not a number:', inferenceTime);
            continue;
        }
        if (typeof score !== 'number') {
            console.error(`Score for detection ${index + 1} is not a number:`, score);
            continue;
        }
        if (typeof box.x1 !== 'number' || typeof box.y1 !== 'number' || typeof box.x2 !== 'number' || typeof box.y2 !== 'number') {
            continue;
        }
        // Check keypoints
        let keypointsDefined = true;
        for (let kp = 0; kp < keypoints.length; kp++) {
            if (typeof keypoints[kp].x !== 'number' || typeof keypoints[kp].y !== 'number') {
                console.error(`Keypoint ${kp + 1} for detection ${index + 1} is undefined:`, keypoints[kp]);
                keypointsDefined = false;
                break;
            }
        }
        if (!keypointsDefined) {
            continue;
        }

        // Safely call toFixed
        outputText += `Detection ${index + 1}:\n`;
        outputText += `  Class: ${classLabel === 0 ? 'Left Shoe' : 'Right Shoe'}\n`;
        outputText += `  Score: ${(score * 100).toFixed(1)}%\n`;
        outputText += `  Bounding Box: [x1: ${box.x1.toFixed(1)}, y1: ${box.y1.toFixed(1)}, x2: ${box.x2.toFixed(1)}, y2: ${box.y2.toFixed(1)}]\n`;
        outputText += `  Keypoints:\n`;
        keypoints.forEach((kp, kpIndex) => {
            outputText += `    ${kpIndex + 1}: (x: ${kp.x.toFixed(1)}, y: ${kp.y.toFixed(1)})\n`;
        });
        outputText += `\n`;
    }
    const totalFormatTime = performance.now() - t0; // End timing
    console.log(`Format Output Text Time: ${totalFormatTime.toFixed(1)} ms`);

    // **FPS Calculation and Appending**
    if (fps !== null) {
        outputText += `FPS: ${fps.toFixed(1)}\n`;
    }

    return outputText;
}

function reshapeFlatArray(flatArray, numRows, numCols) {
    const reshapedArray = new Array(numRows);
    for (let i = 0; i < numRows; i++) {
        const start = i * numCols;
        const end = start + numCols;
        reshapedArray[i] = flatArray.slice(start, end);
    }
    return reshapedArray;
}


/**
 * Splits the tracking.curr_xy array into adjustedBoxes and adjustedKeypoints.
 * Assumes that for each class:
 * - The first 2 points are the bounding box coordinates (x1, y1, x2, y2).
 * - The next 8 points are the keypoints (x, y) for each keypoint.
 *
 * @param {Float32Array} curr_xy - The flat array containing all tracked points.
 * @param {number} point_count - The total number of points being tracked.
 * @returns {Object} An object containing adjustedBoxes and adjustedKeypoints arrays.
 */
function splitTrackedPoints(curr_xy, point_count) {
    const adjustedBoxes = [];
    const adjustedKeypoints = [];
    const pointsPerClass = 10; // 2 box points + 8 keypoints per class

    // Calculate the number of classes based on point_count
    const numClasses = Math.floor(point_count / pointsPerClass);

    for (let classIndex = 0; classIndex < numClasses; classIndex++) {
        const baseIndex = classIndex * pointsPerClass;

        // Extract bounding box coordinates
        const box = {
            x1: curr_xy[baseIndex * 2],
            y1: curr_xy[baseIndex * 2 + 1],
            x2: curr_xy[(baseIndex + 1) * 2],
            y2: curr_xy[(baseIndex + 1) * 2 + 1]
        };
        adjustedBoxes.push(box);

        // Extract keypoints
        const keypoints = [];
        for (let kp = 0; kp < 8; kp++) {
            const kpIndex = baseIndex + 2 + kp;
            keypoints.push({
                x: curr_xy[kpIndex * 2],
                y: curr_xy[kpIndex * 2 + 1]
            });
        }
        adjustedKeypoints.push(keypoints);
    }

    return { adjustedBoxes, adjustedKeypoints };
}

// ====================== Perform Inference Function with Point Tracking Integration ======================
async function performInference(imageElement, displayCtx, overlayCtx, predictionOutput, isVideo = false, fps = null) {
    try {
        const totalT0 = performance.now(); // Start total timing
        console.log(`Performing inference${isVideo ? ' on video/webcam frame' : ' on image'}...`);
        statusMessage.innerText = isVideo ? 'Performing inference on video/webcam frame...' : 'Performing inference on image...';

        // Draw the image/frame on the display canvas
        displayCtx.drawImage(imageElement, 0, 0, displayCanvas.width, displayCanvas.height);
        console.log(`${isVideo ? 'Video/webcam frame' : 'Image'} drawn on display canvas.`);

        // Draw the image/frame on the hidden input canvas for inference
        inputCtx.drawImage(imageElement, 0, 0, inputCanvas.width, inputCanvas.height);
        console.log(`${isVideo ? 'Video/webcam frame' : 'Image'} drawn on input canvas for inference.`);

        // **Point Tracking Integration Starts Here**
        if (isVideo) { // Change to `isVideo || isWebcam` if handling both similarly
            if (tracking.consecutiveTrackedFrames >= tracking.skipCall) {
                // Force model inference after skipCall frames
                console.log('Skipping tracking due to consecutive tracked frames. Forcing model inference.',tracking.consecutiveTrackedFrames);
                tracking.consecutiveTrackedFrames = 0;
                // Proceed to perform model inference below
            } else if (tracking.prev_xy[0] !== 0 || tracking.prev_xy[1] !== 0) {
                // Perform tracking
                console.log('Performing point tracking...tracking.consecutiveTrackedFrames ',tracking.consecutiveTrackedFrames);
                prepareGrayImage(imageElement, tracking.curr_img, inputCtx);
                tracking.curr_img_pyr.build(tracking.curr_img,1);

                // Track keypoints from previous frame to current frame
                
                const trackingValid = trackKeypoints();

                if (trackingValid && tracking.consecutiveTrackedFrames < tracking.skipCall) {
                    console.log('Tracking valid. Using tracked keypoints.',tracking.consecutiveTrackedFrames);
                    tracking.consecutiveTrackedFrames += 1;

                    //Split back keypoints
                    const { adjustedBoxes, adjustedKeypoints } = splitTrackedPoints(tracking.curr_xy, tracking.point_count);
                    const finalClassLabels = [0, 1]; // Example class labels
                    const finalScores = [1.0, 1.0]; // Example scores

                    // **Temporal Smoothing Integration**
                    // Update smoothed_xy using EMA
                    for (let i = 0; i < tracking.point_count * 2; i++) {
                        smoothed_xy[i] = SMOOTHING_ALPHA * tracking.curr_xy[i] + (1 - SMOOTHING_ALPHA) * smoothed_xy[i];
                    }
                    drawDetections(adjustedBoxes, adjustedKeypoints, finalClassLabels, finalScores, overlayCtx);
                    // Render tracked points
                    // renderTrackedPoints(overlayCtx);

                    // Update previous image and keypoints for next tracking
                    const temp = tracking.prev_img;
                    tracking.prev_img = tracking.curr_img;
                    tracking.curr_img = temp;

                    // Swap pyramids
                    const temp_pyr = tracking.prev_img_pyr;
                    tracking.prev_img_pyr = tracking.curr_img_pyr;
                    tracking.curr_img_pyr = temp_pyr;

                    // Update previous keypoints
                    tracking.prev_xy.set(tracking.curr_xy);

                    // Update prediction output
                    predictionOutput.innerText = `Using tracked keypoints. \nConsecutive Tracked Frames: ${tracking.consecutiveTrackedFrames} \n\nFPS: ${fps.toFixed(1)}\n`;
                    
                    return; // Exit performInference without model inference
                } else {
                    console.log('Tracking invalid or threshold exceeded. Performing model inference.');
                    tracking.consecutiveTrackedFrames = 0;
                    // Proceed to perform model inference below
                }
            }

            // If no previous keypoints or tracking failed, perform model inference
        }
        // **Point Tracking Integration Ends Here**

        // Define scale factor for scaling predictions from 160 to display size
        const scaleFactor = isVideo ? (displayCanvas.width / inputCanvas.width) : (displayCanvas.width / inputCanvas.width); // Adjust as needed
        console.log('displayCanvas.width ',displayCanvas.width," inputCanvas.width ",inputCanvas.width);
        // Create a tensor from the input canvas
        const imageTensor = tf.browser.fromPixels(inputCanvas)
            .expandDims(0) // [1, 160, 160, 3]
            .toFloat()
            .div(tf.scalar(255));
        console.log('Image tensor created for inference.');

        // Prepare input dictionary
        const inputName = model.inputs[0].name;
        const inputs = {};
        inputs[inputName] = imageTensor;
        const preprocessTime = performance.now() - totalT0;
        console.log(`Preprocess completed in ${preprocessTime.toFixed(1)} ms.`);
        // Run inference using model.execute
        const t0 = performance.now();

        const predictions = model.execute(inputs); // Single output tensor [1,22,1029]
        const inferenceTime = performance.now() - t0;
        console.log(`Inference completed in ${inferenceTime.toFixed(1)} ms.`);

        // Handle the single output tensor
        if (!(predictions instanceof tf.Tensor)) {
            throw new Error('Model.execute did not return a tensor.');
        }

        const transRes = predictions.transpose([0, 2, 1]); // [1, 1029, 22]
        console.log(`Transpose completed.`);

        const squeezed = transRes.squeeze(); // [1029, 22]
        console.log(`Squeezed completed.`);

        // Slice into boxes, scores, and landmarks
        const boxes = squeezed.slice([0, 0], [-1, 4]); // [1029, 4]
        const scores = squeezed.slice([0, 4], [-1, 2]); // [1029, 2]
        const landmarks = squeezed.slice([0, 6], [-1, 16]); // [1029, 16]

        console.log(`Sliced tensors into boxes, scores, and landmarks.`);

        // Convert tensors to arrays
        console.log('Converting tensors to JavaScript arrays.');
        const { boxesData, scoresData, landmarksData } = await processTensors(boxes, scores, landmarks);
        console.log('Converted tensors to JavaScript arrays.');

        // Dispose intermediate tensors to free memory
        tf.dispose([imageTensor, predictions, transRes, squeezed, boxes, scores, landmarks]);

        
        // Process detections
        const { finalBoxes, finalScores, finalClassLabels, finalLandmarks } = await processDetections(boxesData, scoresData, landmarksData, scaleFactor);
        console.log(`Process detections completed. Total detections: ${finalBoxes.length}`);
        console.log("finalClassLabels ",finalClassLabels)
        console.log("finalScores ",finalScores)
        // Always clear the overlay canvas first
        overlayCtx.clearRect(0, 0, overlayCtx.canvas.width, overlayCtx.canvas.height);
        console.log('Overlay canvas cleared.');
        if (finalBoxes.length === 0) {
            console.log('No detections above the confidence threshold.');
            
            predictionOutput.innerText = isVideo ? 'No detections above the confidence threshold.' : 'No detections above the confidence threshold.';
            return { inferenceTime, finalBoxes, finalScores, finalClassLabels, finalLandmarks };
        }

        // Scale detections
        const { adjustedBoxes, adjustedKeypoints } = scaleDetections(finalBoxes, finalLandmarks, scaleFactor);
        console.log('Detections scaled to display canvas size.');
        console.log('finalLandmarks ',finalLandmarks);
        console.log('adjustedKeypoints ',adjustedKeypoints);
        console.log('finalBoxes ',finalBoxes);
        console.log('adjustedBoxes ',adjustedBoxes);

        // Format output text
        let outputText = formatOutputText(inferenceTime, finalBoxes, finalScores, finalClassLabels, adjustedBoxes, adjustedKeypoints, fps);
        console.log(outputText);

        // Update the prediction output
        predictionOutput.innerText = outputText;

        // Draw bounding boxes and keypoints on the overlay canvas
        drawDetections(adjustedBoxes, adjustedKeypoints, finalClassLabels, finalScores, overlayCtx);
        console.log('Detections drawn on overlay canvas.');

        // **Point Tracking Integration: Update Keypoints After Inference**
        if (isVideo) { // Change to `isVideo || isWebcam` if handling both similarly
            initializeKeypoints(adjustedKeypoints, adjustedBoxes,finalClassLabels);
            // console.log('tracking.prev_xy 2',tracking.prev_xy);
            prepareGrayImage(imageElement, tracking.prev_img, inputCtx);
            tracking.prev_img_pyr.build(tracking.prev_img,1);
        }
        // **Point Tracking Integration Ends Here**

        const totalTime = performance.now() - totalT0; // End total timing
        console.log(`Total Perform Inference Time: ${totalTime.toFixed(1)} ms`);

        return { inferenceTime, finalBoxes, finalScores, finalClassLabels, finalLandmarks };
    } catch (error) {
        console.error('Error :', error);
        statusMessage.innerText = `Error : ${error.message}`;
    }
}

// ====================== Render Tracked Points Function ======================
function renderTrackedPoints(ctx) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.strokeStyle = 'rgba(0, 255, 255, 0.8)'; // Cyan for lines
    ctx.lineWidth = 2;
    const classColors = {
        0: 'rgba(255, 0, 0, 0.7)', // Left Shoe - Red
        1: 'rgba(0, 255, 0, 0.7)'  // Right Shoe - Green
    };
    const keypointColor = 'rgba(0, 0, 255, 0.9)'; // Blue

    // Assuming all keypoints belong to a single detection for simplicity
    const keypoints = [];
    for (let i = 0; i < tracking.point_count; i++) {
        keypoints.push({ x: tracking.curr_xy[i * 2], y: tracking.curr_xy[i * 2 + 1] });
    }

    // Draw keypoints
    keypoints.forEach((kp, kpIndex) => {
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, 4, 0, 2 * Math.PI);
        ctx.fillStyle = keypointColor;
        ctx.fill();
        ctx.closePath();
    });

    // Optionally, draw bounding box if available
    // This example assumes a single bounding box; modify as needed
    // if (tracking.currentBoundingBox) {
    //     const box = tracking.currentBoundingBox;
    //     ctx.strokeStyle = classColors[0] || 'rgba(255, 255, 0, 0.7)';
    //     ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
    // }
}

// ====================== Draw Detections Function ======================
// Function to draw bounding boxes, keypoints, and connections on the overlay canvas
function drawDetections(adjustedBoxes, adjustedKeypoints, finalClassLabels, finalScores, ctx) {
    // Clear the overlay canvas before drawing
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // Define colors for different classes
    const classColors = {
        0: 'rgba(255, 0, 0, 0.7)', // Left Shoe - Red
        1: 'rgba(0, 255, 0, 0.7)'  // Right Shoe - Green
    };

    // Define keypoint color
    const keypointColor = 'rgba(0, 0, 255, 0.9)'; // Blue

    // Define line color for connections between keypoints
    ctx.strokeStyle = 'rgba(0, 255, 255, 0.8)'; // Cyan for lines
    ctx.lineWidth = 2; // Set line width for connections

    // Define font for labels
    ctx.font = '18px Arial';
    ctx.textBaseline = 'top';

    adjustedBoxes.forEach((box, index) => {
        const color = classColors[finalClassLabels[index]] || 'rgba(255, 255, 0, 0.7)'; // Default to Yellow

        // Draw bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
        // Draw label with score
        const label = finalClassLabels[index] === 0 ? 'Left Shoe' : 'Right Shoe';
        const score = (finalScores[index] * 100).toFixed(1) + '%';
        ctx.fillStyle = color;
        ctx.fillText(`${label} (${score})`, box.x1, box.y1 - 25);

        // Draw keypoints and connections
        const keypoints = adjustedKeypoints[index];
        keypoints.forEach((kp, kpIndex) => {
            // Draw keypoint
            ctx.beginPath();
            ctx.arc(kp.x, kp.y, 4, 0, 2 * Math.PI); // Radius of 4 for better visibility
            ctx.fillStyle = keypointColor;
            ctx.fill();
            ctx.closePath();
        });
        
        // Draw connections between keypoints
        ctx.strokeStyle = "yellow";
        drawLineBetweenPoints(ctx, keypoints, 0, 1);
        ctx.strokeStyle = "pink";
        drawLineBetweenPoints(ctx, keypoints, 0, 2);
        ctx.strokeStyle = "magenta";
        drawLineBetweenPoints(ctx, keypoints, 0, 3);
        ctx.strokeStyle = "indigo";
        drawLineBetweenPoints(ctx, keypoints, 4, 5);
        ctx.strokeStyle = "silver";
        drawLineBetweenPoints(ctx, keypoints, 5, 3);
        ctx.strokeStyle = "brown";
        drawLineBetweenPoints(ctx, keypoints, 6, 3);
        ctx.strokeStyle = "black";
        drawLineBetweenPoints(ctx, keypoints, 7, 3);
    });
}

function drawLineBetweenPoints(ctx, keypoints, index1, index2) {
    if (index1 < keypoints.length && index2 < keypoints.length) {
        const x1 = keypoints[index1].x;
        const y1 = keypoints[index1].y;
        const x2 = keypoints[index2].x;
        const y2 = keypoints[index2].y;

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.closePath();
    }
}

// ====================== Run Inference on Image ======================
async function runInferenceOnImage() {
    if (!model) {
        statusMessage.innerText = 'Model not loaded. Please load the model first.';
        return;
    }

    if (imagePaths.length === 0) {
        alert('No images available for inference. Please add image paths to the imagePaths array.');
        return;
    }
    
    for (let i = 0; i < 1; i++) {
        const randomIndex = Math.floor(Math.random() * imagePaths.length);
        const selectedImagePath = imagePaths[randomIndex];
        console.log(`Selected Image: ${selectedImagePath}`);

        // Create a new Image object
        const image = new Image();
        image.crossOrigin = "anonymous"; // To avoid CORS issues
        image.src = selectedImagePath;
        // Handle image loading
        image.onload = async () => {
            const iterationStartTime = performance.now();
            console.log('Random image loaded successfully.');
            await performInference(image, displayCtx, overlayCtx, imagePredictionOutput, false);
        };
        image.onerror = (error) => {
            console.error(`Error loading image ${selectedImagePath}:`, error);
            statusMessage.innerText = `Error loading image ${selectedImagePath}.`;
        };

        statusMessage.innerText = `Running inference on random image: ${selectedImagePath}`;
        console.log(`User initiated inference on random image: ${selectedImagePath}`);
    }
}

// ====================== Run Inference on Video ======================
async function runInferenceOnVideo() {
    if (!model) {
        statusMessage.innerText = 'Model not loaded. Please load the model first.';
        return;
    }

    statusMessage.innerText = 'Running inference on video...';
    console.log('Starting video inference.');

    // Show the video container
    videoContainer.style.display = 'block';

    // Set video source to local video 'shoevid.mp4'
    inputVideo.src = 'shoevid.mp4'; // Ensure 'shoevid.mp4' is in the same directory or provide the correct path
    inputVideo.width = 224;
    inputVideo.height = 224;
    inputVideo.autoplay = true;
    inputVideo.playsInline = true;
    inputVideo.muted = true;  // Mute the video if needed

    // Enable the pause button
    pauseVideoButton.disabled = false;
    pauseVideoButton.innerText = 'Pause Video';
    console.log('Pause button enabled.');

    // Start playing the video
    inputVideo.play().then(() => {
        console.log('Video playback started.');
    }).catch((error) => {
        console.error('Error playing video:', error);
        statusMessage.innerText = `Error playing video: ${error.message}`;
    });

    // Once the video can play, start processing frames
    inputVideo.oncanplay = () => {
        console.log('Video can play. Starting frame processing.');
        processVideoFrame();
    };

    // Handle video end
    inputVideo.onended = () => {
        console.log('Video ended.');
        statusMessage.innerText = 'Video inference completed.';
        pauseVideoButton.disabled = true;
    };

    // Handle video loading errors
    inputVideo.onerror = (error) => {
        console.error('Error loading video:', error);
        statusMessage.innerText = `Error loading video: ${error.message}`;
    };
}

// ====================== Toggle Pause Video ======================
function togglePauseVideo() {
    if (inputVideo.paused) {
        inputVideo.play();
        pauseVideoButton.innerText = 'Pause Video';
        isVideoPaused = false;
        console.log('Video resumed.');
    } else {
        inputVideo.pause();
        pauseVideoButton.innerText = 'Resume Video';
        isVideoPaused = true;
        console.log('Video paused.');
    }
}

// ====================== Process Video Frame with Point Tracking ======================
async function processVideoFrame() {
    if (inputVideo.paused || inputVideo.ended) {
        console.log('Video is paused or ended. Stopping frame processing.');
        statusMessage.innerText = 'Video inference completed.';
        pauseVideoButton.disabled = true;
        return;
    }

    if (isProcessingVideo) {
        // Skip processing this frame if an inference is already running
        console.log('Inference already in progress. Skipping frame.');
        requestAnimationFrame(processVideoFrame);
        return;
    }

    isProcessingVideo = true;
    console.log('Processing new video frame.');

    // Perform inference on the current frame
    await performInference(inputVideo, displayCtx, overlayCtx, videoPredictionOutput, true, fpsVideo);

    isProcessingVideo = false;
    console.log('Inference completed for current frame.');

    // FPS Calculation
    frameCountVideo++;
    const currentTime = performance.now();
    const elapsed = currentTime - lastFpsUpdateTimeVideo;
    if (elapsed >= 1000) {
        fpsVideo = frameCountVideo / (elapsed / 1000);
        lastFpsUpdateTimeVideo = currentTime;
        frameCountVideo = 0;
    }

    // Continue processing the next frame
    requestAnimationFrame(processVideoFrame);
}

// ====================== Run Inference on Webcam ======================
async function runInferenceOnWebcam() {
    if (!model) {
        statusMessage.innerText = 'Model not loaded. Please load the model first.';
        return;
    }

    statusMessage.innerText = 'Running inference on webcam...';
    console.log('Starting webcam inference.');

    // Show the webcam container
    webcamContainer.style.display = 'block';

    // Request access to the webcam with back camera
    try {
        const constraints = {
            video: { 
                width: { ideal: 360 }, 
                height: { ideal: 480 },
                frameRate: { ideal: 15, max: 30 }, 
                facingMode: { exact: "environment" } // Request the back camera
            },
            audio: false
        };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        const settings = stream.getVideoTracks()[0].getSettings();
        console.log(`Webcam Frame Rate: ${settings.frameRate}`);
        webcamVideo.srcObject = stream;
        await webcamVideo.play();
        console.log('Webcam stream started.');

        // Log actual video dimensions
        console.log(`Webcam Video Width: ${webcamVideo.videoWidth}, Height: ${webcamVideo.videoHeight}`);

        // Enable the pause button
        pauseWebcamButton.disabled = false;
        pauseWebcamButton.innerText = 'Pause Webcam';
    } catch (error) {
        console.error('Error accessing webcam:', error);
        statusMessage.innerText = `Error accessing webcam: ${error.message}`;
        return;
    }

    // Once the webcam can play, start processing frames
    webcamVideo.oncanplay = () => {
        console.log('Webcam can play. Starting frame processing.');
        processWebcamFrame();
    };

    // Handle webcam end
    webcamVideo.onended = () => {
        console.log('Webcam stream ended.');
        statusMessage.innerText = 'Webcam inference completed.';
        pauseWebcamButton.disabled = true;
    };

    // Handle webcam loading errors
    webcamVideo.onerror = (error) => {
        console.error('Error loading webcam video:', error);
        statusMessage.innerText = `Error loading webcam video: ${error.message}`;
    };
}

// ====================== Toggle Pause Webcam ======================
function togglePauseWebcam() {
    if (webcamVideo.paused) {
        webcamVideo.play();
        pauseWebcamButton.innerText = 'Pause Webcam';
        isWebcamPaused = false;
        console.log('Webcam resumed.');
    } else {
        webcamVideo.pause();
        pauseWebcamButton.innerText = 'Resume Webcam';
        isWebcamPaused = true;
        console.log('Webcam paused.');
    }
}

// ====================== Process Webcam Frame with Point Tracking ======================
async function processWebcamFrame() {
    if (webcamVideo.paused || webcamVideo.ended) {
        console.log('Webcam is paused or ended. Stopping frame processing.');
        statusMessage.innerText = 'Webcam inference completed.';
        pauseWebcamButton.disabled = true;
        return;
    }

    if (isProcessingWebcam) {
        // Skip processing this frame if an inference is already running
        console.log('Inference already in progress for webcam. Skipping frame.');
        // requestAnimationFrame(processWebcamFrame);
        setTimeout(() => {
            requestAnimationFrame(processWebcamFrame);
        }, 1000 / 15); // 15 FPS
        return;
    }

    isProcessingWebcam = true;
    console.log('Processing new webcam frame.');

    // Perform inference on the current frame
    await performInference(webcamVideo, webcamDisplayCtx, webcamOverlayCtx, webcamPredictionOutput, true, fpsWebcam);

    isProcessingWebcam = false;
    console.log('Inference completed for current webcam frame.');

    // FPS Calculation
    frameCountWebcam++;
    const currentTime = performance.now();
    const elapsed = currentTime - lastFpsUpdateTimeWebcam;
    if (elapsed >= 1000) {
        fpsWebcam = frameCountWebcam / (elapsed / 1000);
        lastFpsUpdateTimeWebcam = currentTime;
        frameCountWebcam = 0;
    }

    // Continue processing the next frame
    requestAnimationFrame(processWebcamFrame);
}

// ====================== Process Tensors Function ======================
async function processTensors(boxes, scores, landmarks) {
    // Start timing
    const t0 = performance.now();

    // Initiate all data retrievals concurrently using asynchronous methods
    const boxesDataPromise = boxes.data();
    const scoresDataPromise = scores.data();
    const landmarksDataPromise = landmarks.data();

    // Await all conversions simultaneously
    const [boxesDataFlat, scoresDataFlat, landmarksDataFlat] = await Promise.all([
        boxesDataPromise,
        scoresDataPromise,
        landmarksDataPromise
    ]);

    // Reshape flat arrays into the desired shapes
    const boxesData = reshapeFlatArray(Array.from(boxesDataFlat), boxes.shape[0], boxes.shape[1]);          // [1029, 4]
    const scoresData = reshapeFlatArray(Array.from(scoresDataFlat), scores.shape[0], scores.shape[1]);        // [1029, 2]
    const landmarksData = reshapeFlatArray(Array.from(landmarksDataFlat), landmarks.shape[0], landmarks.shape[1]); // [1029, 16]

    const processTime = performance.now() - t0;
    console.log(`Process Tensors Time: ${processTime.toFixed(1)} ms`);

    return { boxesData, scoresData, landmarksData };
}

// ====================== Render Tracked Points Function ======================
// This function was defined earlier in the tracking integration section

// ====================== Event Listeners ======================
loadModelButton.addEventListener('click', loadModel);
predictImageButton.addEventListener('click', runInferenceOnImage);
predictVideoButton.addEventListener('click', runInferenceOnVideo);
pauseVideoButton.addEventListener('click', togglePauseVideo); // Pause Button Event Listener
predictWebcamButton.addEventListener('click', runInferenceOnWebcam);
pauseWebcamButton.addEventListener('click', togglePauseWebcam); // Pause Webcam Button Event Listener

// ====================== Cleanup on Page Unload ======================
window.addEventListener('beforeunload', () => {
    // Stop video inference
    if (inputVideo.srcObject) {
        inputVideo.pause();
        inputVideo.srcObject.getTracks().forEach(track => track.stop());
    }

    // Stop webcam inference
    if (webcamVideo.srcObject) {
        webcamVideo.pause();
        webcamVideo.srcObject.getTracks().forEach(track => track.stop());
    }
});
