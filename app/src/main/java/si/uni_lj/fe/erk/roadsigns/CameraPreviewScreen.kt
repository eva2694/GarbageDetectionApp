package si.uni_lj.fe.erk.roadsigns

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.BatteryManager
import android.os.Debug
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Popup
import androidx.compose.ui.window.PopupProperties
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.task.vision.detector.Detection
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileWriter
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService

@Composable
fun CameraPreviewScreen(cameraExecutor: ExecutorService) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    var detectionResults by remember { mutableStateOf<List<YoloModelLoader.BoundingBox>>(listOf()) }

    val coroutineScope = rememberCoroutineScope()

    var selectedModelIndex by rememberSaveable { mutableStateOf(0) }
    val modelOptions = listOf(
        "YOLOv8s-float32.tflite", "YOLOv8s-float16.tflite", "YOLOv8n-float32.tflite",
        "YOLOv8n-float16.tflite", "EfficientDet-Lite0.tflite", "EfficientDet-Lite1.tflite"
    )
    val modelSizes = mapOf(
        "YOLOv8s-float32.tflite" to "44MB", "YOLOv8s-float16.tflite" to "22MB",
        "YOLOv8n-float32.tflite" to "12MB", "YOLOv8n-float16.tflite" to "6MB",
        "EfficientDet-Lite0.tflite" to "4.4MB", "EfficientDet-Lite1.tflite" to "5.8MB"
    )

    val modelDatatypes = mapOf(
        "YOLOv8s-float32.tflite" to "float32", "YOLOv8s-float16.tflite" to "float16",
        "YOLOv8n-float32.tflite" to "float32", "YOLOv8n-float16.tflite" to "float16",
        "EfficientDet-Lite0.tflite" to "int8", "EfficientDet-Lite1.tflite" to "int8"
    )

    val currentModel = remember {
        mutableStateOf(Triple(modelOptions[selectedModelIndex], modelDatatypes[modelOptions[selectedModelIndex]] ?: "Unknown", modelSizes[modelOptions[selectedModelIndex]] ?: "Unknown"))
    }


    val tfliteModelLoader = remember { mutableStateOf<YoloModelLoader?>(null) }
    val efficientDetModelLoader = remember { mutableStateOf<EfficientDetModelLoader?>(null) }

    LaunchedEffect(currentModel.value) {

        if (currentModel.value.first.contains("EfficientDet")) {
            efficientDetModelLoader.value = EfficientDetModelLoader(context = context, objectDetectorListener = object : EfficientDetModelLoader.DetectorListener {
                override fun onError(error: String) {
                    Log.e("EfficientDetModelLoader", error)
                }

                override fun onResults(results: MutableList<Detection>?, inferenceTime: Long, imageHeight: Int, imageWidth: Int) {
                    detectionResults = results?.map {
                        YoloModelLoader.BoundingBox(
                            x1 = it.boundingBox.left / imageWidth,
                            y1 = it.boundingBox.top / imageHeight,
                            x2 = it.boundingBox.right / imageWidth,
                            y2 = it.boundingBox.bottom / imageHeight,
                            cx = (it.boundingBox.left + it.boundingBox.right) / 2 / imageWidth,
                            cy = (it.boundingBox.top + it.boundingBox.bottom) / 2 / imageHeight,
                            w = (it.boundingBox.right - it.boundingBox.left) / imageWidth,
                            h = (it.boundingBox.bottom - it.boundingBox.top) / imageHeight,
                            cnf = it.categories.firstOrNull()?.score ?: 0f,
                            cls = it.categories.firstOrNull()?.index ?: 0,
                            clsName = it.categories.firstOrNull()?.label ?: "unknown"
                        )
                    } ?: listOf()
                }
            }, currentModel = if (currentModel.value.first == "EfficientDet-Lite0.tflite") 1 else 2)
            tfliteModelLoader.value = null
        } else {
            tfliteModelLoader.value = YoloModelLoader(context, currentModel.value.first)
            efficientDetModelLoader.value = null
        }
    }

    val startTime = remember { mutableStateOf(0L) }
    val endTime = remember { mutableStateOf(0L) }
    var positiveDetectionTime = remember { mutableStateOf(0) }
    val cpuUsage = remember { mutableStateOf(0L) }
    val memoryInfo = remember { mutableStateOf<ActivityManager.MemoryInfo?>(null) }
    val frameLatency = remember { mutableStateOf(0L) }
    val previousDetectionTime = remember { mutableStateOf(SystemClock.elapsedRealtimeNanos()) }
    val totalInferenceTime = remember { mutableStateOf(0L) }
    val inferenceCount = remember { mutableStateOf(0) }
    val averageConfidence = remember { mutableStateOf(0.0) }
    val fps = remember { mutableStateOf(0.0) }

    Column(modifier = Modifier.fillMaxSize()) {
        DropdownList(
            itemList = modelOptions,
            selectedIndex = selectedModelIndex,
            modifier = Modifier
                .width(200.dp)
                .padding(16.dp),
            onItemClick = { index ->
                selectedModelIndex = index
                currentModel.value = Triple(modelOptions[index], modelDatatypes[modelOptions[index]] ?: "Unknown", modelSizes[modelOptions[index]] ?: "Unknown")
            }
        )

        Box(modifier = Modifier.fillMaxSize()) {
            AndroidView(factory = { ctx ->
                val previewView = PreviewView(ctx)
                val preview = Preview.Builder().build().also { preview ->
                    preview.surfaceProvider = previewView.surfaceProvider
                }

                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                    .build()

                imageAnalysis.setAnalyzer(cameraExecutor) { image ->
                    coroutineScope.launch {
                        val bitmap = withContext(Dispatchers.IO) {
                            image.toBitmapCustom()
                        }
                        if (bitmap != null) {
                            withContext(Dispatchers.Default) {
                                startTime.value = SystemClock.elapsedRealtimeNanos()
                                val cpuStartTime = Debug.threadCpuTimeNanos()

                                if (tfliteModelLoader.value != null) {
                                    detectObjects(bitmap, tfliteModelLoader.value!!, currentModel.value.first, currentModel.value.second) { results ->
                                        detectionResults = results
                                        endTime.value = SystemClock.elapsedRealtimeNanos()
                                        val cpuEndTime = Debug.threadCpuTimeNanos()
                                        cpuUsage.value = (cpuEndTime - cpuStartTime) / 1_000_000
                                        if (endTime.value > startTime.value) {
                                            processResults(cpuEndTime, cpuStartTime, context, currentModel.value.first, currentModel.value.second, endTime, startTime, memoryInfo, frameLatency, previousDetectionTime, totalInferenceTime, inferenceCount, averageConfidence, fps, results, modelSizes)
                                        }
                                    }
                                } else if (efficientDetModelLoader.value != null) {
                                    efficientDetModelLoader.value!!.detect(bitmap)
                                    endTime.value = SystemClock.elapsedRealtimeNanos()
                                    val cpuEndTime = Debug.threadCpuTimeNanos()
                                    cpuUsage.value = (cpuEndTime - cpuStartTime) / 1_000_000
                                    if (endTime.value > startTime.value) {
                                        processResults(cpuEndTime, cpuStartTime, context, currentModel.value.first, currentModel.value.second, endTime, startTime, memoryInfo, frameLatency, previousDetectionTime, totalInferenceTime, inferenceCount, averageConfidence, fps, detectionResults, modelSizes)
                                    }
                                }
                            }
                        } else {
                            Log.e("ImageAnalysis", "Bitmap conversion failed")
                        }
                        image.close()
                    }
                }

                val cameraProvider = cameraProviderFuture.get()
                try {
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        imageAnalysis
                    )
                } catch (exc: Exception) {
                    Log.e("CameraPreviewScreen", "Camera binding failed", exc)
                }

                previewView
            }, modifier = Modifier.fillMaxSize())

            Box(modifier = Modifier.fillMaxSize()) {
                detectionResults.forEach { result ->
                    Canvas(modifier = Modifier.fillMaxSize()) {
                        val scaleX = size.width
                        val scaleY = size.height

                        val x1 = 1 - result.y2
                        val y1 = result.x1
                        val x2 = 1 - result.y1
                        val y2 = result.x2

                        val left = x1 * scaleX / 1.15
                        val top = y1 * scaleY
                        val right = x2 * scaleX * 1.15
                        val bottom = y2 * scaleY

                        drawRect(
                            color = Color.Red,
                            topLeft = androidx.compose.ui.geometry.Offset(left.toFloat(), top),
                            size = androidx.compose.ui.geometry.Size((right - left).toFloat(), (bottom - top)),
                            style = androidx.compose.ui.graphics.drawscope.Stroke(width = 2f)
                        )
                        drawContext.canvas.nativeCanvas.apply {
                            drawText(
                                "${result.clsName} ${"%.2f".format(result.cnf)}",
                                left.toFloat(),
                                (top - 10),
                                Paint().apply {
                                    color = android.graphics.Color.RED
                                    textSize = 30f
                                }
                            )
                        }
                    }
                }

                Column(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .padding(16.dp)
                ) {


                    val dT = (endTime.value - startTime.value) / 1_000_000


                    if (dT < 1000 && dT > 0) positiveDetectionTime.value = dT.toInt()

                    val usedMemory = memoryInfo.value?.availMem?.div(1024 * 1024)
                    val totalMemory = memoryInfo.value?.totalMem?.div(1024 * 1024)

                    Text("Model: ${currentModel.value.first} (${currentModel.value.third})", fontWeight = FontWeight.Bold, color = Color.Green)
                    Text("Detection Time: ${positiveDetectionTime.value} ms", fontWeight = FontWeight.Bold, color = Color.Green)
                    Text("CPU Time: ${cpuUsage.value} ms", fontWeight = FontWeight.Bold, color = Color.Green)
                    Text("Frame Latency: ${frameLatency.value} ms", fontWeight = FontWeight.Bold, color = Color.Green)
                    Text("FPS: ${"%.2f".format(fps.value)}", fontWeight = FontWeight.Bold, color = Color.Green)
                    Text("Average Confidence: ${"%.2f".format(averageConfidence.value)}", fontWeight = FontWeight.Bold, color = Color.Green)
                    Text("Available Memory: $usedMemory MB", fontWeight = FontWeight.Bold, color = Color.Green)
                    Text("Total Memory: $totalMemory MB", fontWeight = FontWeight.Bold, color = Color.Green)
                }
            }
        }
    }
}

fun detectObjects(bitmap: Bitmap, modelLoader: YoloModelLoader, modelName: String, datatype: String, onResults: (List<YoloModelLoader.BoundingBox>) -> Unit) {
    val results = modelLoader.detect(bitmap)
    onResults(results)
}

fun processResults(
    cpuEndTime: Long,
    cpuStartTime: Long,
    context: Context,
    selectedModel: String,
    selectedDatatype: String,
    endTime: MutableState<Long>,
    startTime: MutableState<Long>,
    memoryInfo: MutableState<ActivityManager.MemoryInfo?>,
    frameLatency: MutableState<Long>,
    previousDetectionTime: MutableState<Long>,
    totalInferenceTime: MutableState<Long>,
    inferenceCount: MutableState<Int>,
    averageConfidence: MutableState<Double>,
    fps: MutableState<Double>,
    results: List<YoloModelLoader.BoundingBox>,
    modelSizes: Map<String, String>
) {
    val cpuUsage = (cpuEndTime - cpuStartTime) / 1_000_000
    val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    val memInfo = ActivityManager.MemoryInfo()
    activityManager.getMemoryInfo(memInfo)
    memoryInfo.value = memInfo
    frameLatency.value = (SystemClock.elapsedRealtimeNanos() - previousDetectionTime.value) / 1_000_000
    previousDetectionTime.value = SystemClock.elapsedRealtimeNanos()

    totalInferenceTime.value += (endTime.value - startTime.value)
    inferenceCount.value += 1
    averageConfidence.value = results.map { it.cnf }.average()

    val averageInferenceTime = if (inferenceCount.value > 0) totalInferenceTime.value / inferenceCount.value / 1_000_000 else 0L
    val framesPerSecond = if (averageInferenceTime > 0) 1000F / averageInferenceTime.toFloat() else 0F
    fps.value = framesPerSecond.toDouble()

    val modelSize = modelSizes[selectedModel] ?: "Unknown"

    val batteryIntent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
    val level = batteryIntent?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
    val scale = batteryIntent?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
    val batteryPct = level / scale.toFloat() * 100

    val detectionTime = (endTime.value - startTime.value) / 1_000_000

    if (detectionTime > 0) {
        saveDataToCsv(
            model = selectedModel,
            datatype = selectedDatatype,
            modelSize = modelSize,
            detectionTime = detectionTime,
            cpuTime = cpuUsage,
            frameLatency = frameLatency.value,
            fps = framesPerSecond,
            avgConfidence = averageConfidence.value,
            usedMemory = memInfo.availMem / (1024 * 1024),
            totalMemory = memInfo.totalMem / (1024 * 1024),
            batteryLevel = batteryPct,
            context = context
        )
    }
}

fun ImageProxy.toBitmapCustom(): Bitmap? {
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)

    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
    val imageBytes = out.toByteArray()
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
}

// csv path: /storage/emulated/0/Android/data/si.uni_lj.fe.erk.roadsigns/files
fun saveDataToCsv(
    model: String,
    datatype: String,
    modelSize: String,
    detectionTime: Long,
    cpuTime: Long,
    frameLatency: Long,
    fps: Float,
    avgConfidence: Double,
    usedMemory: Long,
    totalMemory: Long,
    batteryLevel: Float,
    context: Context
) {
    val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())
    val fileName = "detection_data.csv"
    val file = File(context.getExternalFilesDir(null), fileName)

    try {
        val writer = FileWriter(file, true)
        writer.append("$timestamp,$model,$datatype,$modelSize,$detectionTime,$cpuTime,$frameLatency,${"%.2f".format(fps)},${"%.2f".format(avgConfidence)},$usedMemory,$totalMemory,$batteryLevel\n")
        writer.flush()
        writer.close()
        Log.d("saveDataToCsv", "Data saved: model=$model, datatype=$datatype, size=$modelSize, det time = $detectionTime, cpu time = $cpuTime, frame lat = $frameLatency, fps = ${"%.2f".format(fps)}, confidence = ${"%.2f".format(avgConfidence)}, used mem = $usedMemory, total mem = $totalMemory, bat = $batteryLevel")
    } catch (e: IOException) {
        e.printStackTrace()
    }
}

@Composable
fun DropdownList(itemList: List<String>, selectedIndex: Int, modifier: Modifier, onItemClick: (Int) -> Unit) {
    var showDropdown by rememberSaveable { mutableStateOf(false) }
    val scrollState = rememberScrollState()

    Column(
        modifier = Modifier,
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Box(
            modifier = modifier
                .background(Color.Red)
                .clickable { showDropdown = true },
            contentAlignment = Alignment.Center
        ) {
            Text(text = itemList[selectedIndex], modifier = Modifier.padding(3.dp))
        }

        Box {
            if (showDropdown) {
                Popup(
                    alignment = Alignment.TopCenter,
                    properties = PopupProperties(
                        excludeFromSystemGesture = true,
                    ),
                    onDismissRequest = { showDropdown = false }
                ) {
                    Column(
                        modifier = modifier
                            .heightIn(max = 200.dp)
                            .verticalScroll(state = scrollState)
                            .border(width = 1.dp, color = Color.Gray)
                            .background(Color.White),
                        horizontalAlignment = Alignment.CenterHorizontally,
                    ) {
                        itemList.forEachIndexed { index, item ->
                            if (index != 0) {
                                HorizontalDivider(thickness = 1.dp, color = Color.LightGray)
                            }
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clickable {
                                        onItemClick(index)
                                        showDropdown = false
                                    }
                                    .background(if (selectedIndex == index) Color.LightGray else Color.White),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(text = item, modifier = Modifier.padding(8.dp))
                            }
                        }
                    }
                }
            }
        }
    }
}
