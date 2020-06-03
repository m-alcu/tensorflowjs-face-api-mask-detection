const video = document.getElementById('video')

const RELATIVE_MODEL_URL = './model/model.json';

const MASK_CLASSES = ['Mask', 'No Mask'];

let model

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
  loadfacemaskModel('./model/model.json'),
]).then(startVideo)

async function loadfacemaskModel(urlModel) {
  model = await tf.loadLayersModel(urlModel);
}

function startVideo() {
  navigator.getUserMedia(
    { video: {} },
    stream => video.srcObject = stream,
    err => console.error(err)
  )
}

video.addEventListener('play', () => {
  const canvas = faceapi.createCanvasFromMedia(video)
  document.body.append(canvas)
  const displaySize = { width: video.width, height: video.height }
  faceapi.matchDimensions(canvas, displaySize)
  setInterval(async () => {


    const detections = await faceapi.detectAllFaces(video).withFaceLandmarks().withFaceDescriptors()
    const faceImages = await faceapi.extractFaces(video, detections.map(item => { return item.detection }))
    const results = faceImages.map(canvas => tf.browser.fromPixels(canvas)
    .resizeNearestNeighbor([224, 224])
    .toFloat()
    .reverse(2)
    .expandDims()
    .div(127.5)
    .sub(1))

    const resizedDetections = faceapi.resizeResults(detections, displaySize)

    results.forEach((result, i) => {
      const prediction = model.predict(result)
      const winner = MASK_CLASSES[prediction.argMax(-1).dataSync()[0]];
      const box = resizedDetections[i].detection.box
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
      const drawBox = new faceapi.draw.DrawBox(box, { label: winner.toString() , boxColor: winner == MASK_CLASSES[0] ? 'blue' : 'red'})
      drawBox.draw(canvas)
    })    

  }, 100)
})