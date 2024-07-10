package si.uni_lj.fe.erk.roadsigns

import android.Manifest
import android.app.ActivityManager
import android.content.Context
import android.graphics.*
import android.os.Bundle
import android.os.Debug
import android.os.SystemClock
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import si.uni_lj.fe.erk.roadsigns.ui.theme.RoadSignsTheme
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()
        setContent {
            RoadSignsTheme {
                RequestCameraPermission {
                    CameraPreviewScreen(cameraExecutor)
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}

@Composable
fun RequestCameraPermission(content: @Composable () -> Unit) {
    val context = LocalContext.current
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA
            ) == android.content.pm.PackageManager.PERMISSION_GRANTED
        )
    }
    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasCameraPermission = isGranted
    }

    LaunchedEffect(key1 = true) {
        launcher.launch(Manifest.permission.CAMERA)
    }

    if (hasCameraPermission) {
        content()
    } else {
        Text("Camera permission is required for this app to work.")
    }
}

@Composable
fun CameraPreviewScreen(cameraExecutor: ExecutorService) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val tfliteModelLoader = remember { TFLiteModelLoader(context) }
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    var detectionResults by remember { mutableStateOf<List<TFLiteModelLoader.BoundingBox>>(listOf()) }
    val coroutineScope = rememberCoroutineScope()
    val startTime = remember { mutableStateOf(0L) }
    val endTime = remember { mutableStateOf(0L) }
    val cpuUsage = remember { mutableStateOf(0L) }
    val memoryInfo = remember { mutableStateOf<ActivityManager.MemoryInfo?>(null) }
    val frameLatency = remember { mutableStateOf(0L) }
    val previousDetectionTime = remember { mutableStateOf(SystemClock.elapsedRealtimeNanos()) }
    val totalInferenceTime = remember { mutableStateOf(0L) }
    val inferenceCount = remember { mutableStateOf(0) }
    val averageConfidence = remember { mutableStateOf(0.0) }

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
                        Log.d("CameraPreviewScreen", "Bitmap size: ${bitmap.width}x${bitmap.height}")
                        withContext(Dispatchers.Default) {
                            startTime.value = SystemClock.elapsedRealtimeNanos()
                            val cpuStartTime = Debug.threadCpuTimeNanos()
                            detectObjects(bitmap, tfliteModelLoader) { results ->
                                detectionResults = results
                                endTime.value = SystemClock.elapsedRealtimeNanos()
                                val cpuEndTime = Debug.threadCpuTimeNanos()


                                cpuUsage.value = (cpuEndTime - cpuStartTime) / 1_000_000
                                val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                                val memInfo = ActivityManager.MemoryInfo()
                                activityManager.getMemoryInfo(memInfo)
                                memoryInfo.value = memInfo
                                frameLatency.value = (SystemClock.elapsedRealtimeNanos() - previousDetectionTime.value) / 1_000_000
                                previousDetectionTime.value = SystemClock.elapsedRealtimeNanos()

                                totalInferenceTime.value += (endTime.value - startTime.value)
                                inferenceCount.value += 1
                                averageConfidence.value = results.map { it.cnf }.average()
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
                    val scaleX = size.width * 1.7
                    val scaleY = size.height * 0.7

                    val left = result.x1 * scaleX
                    val top = result.y1 * scaleY
                    val right = result.x2 * scaleX
                    val bottom = result.y2 * scaleY

                    val previewAspectRatio = scaleX / scaleY
                    val modelAspectRatio = 1.0f
                    Log.d("CameraPreviewScreen","Preview aspect ratio = ${previewAspectRatio}, Model aspect ratio = $modelAspectRatio")

                    drawRect(
                        color = Color.Red,
                        topLeft = androidx.compose.ui.geometry.Offset(left.toFloat(), top.toFloat()),
                        size = androidx.compose.ui.geometry.Size((right - left).toFloat(),
                            (bottom - top).toFloat()
                        ),
                        style = androidx.compose.ui.graphics.drawscope.Stroke(width = 2f)
                    )
                    drawContext.canvas.nativeCanvas.apply {
                        drawText(
                            "${result.clsName} ${"%.2f".format(result.cnf)}",
                            left.toFloat(),
                            (top - 10).toFloat(),
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
                val detectionTime = (endTime.value - startTime.value) / -1_000_000 // in ms
                val usedMemory = memoryInfo.value?.availMem?.div(1024 * 1024)
                val totalMemory = memoryInfo.value?.totalMem?.div(1024 * 1024)
                val averageInferenceTime = if (inferenceCount.value > 0) totalInferenceTime.value / inferenceCount.value / 1_000_000 else 0L
                val fps = if (inferenceCount.value > 0) 100000 / averageInferenceTime else 0

                Text("Model: YOLOv8s (float32, 44MB)", fontWeight = FontWeight.Bold, color = Color.Green)
                Text("Detection Time: ${detectionTime} ms", fontWeight = FontWeight.Bold, color = Color.Green)
                Text("CPU Time: ${cpuUsage.value} ms", fontWeight = FontWeight.Bold, color = Color.Green)
                Text("Frame Latency: ${frameLatency.value} ms", fontWeight = FontWeight.Bold, color = Color.Green)
                Text("FPS: $fps", fontWeight = FontWeight.Bold, color = Color.Green)
                Text("Average Confidence: ${"%.2f".format(averageConfidence.value)}", fontWeight = FontWeight.Bold, color = Color.Green)
                Text("Available Memory: ${usedMemory} MB", fontWeight = FontWeight.Bold, color = Color.Green)
                Text("Total Memory: ${totalMemory} MB", fontWeight = FontWeight.Bold, color = Color.Green)
            }
        }
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

fun detectObjects(bitmap: Bitmap, modelLoader: TFLiteModelLoader, onResults: (List<TFLiteModelLoader.BoundingBox>) -> Unit) {
    val results = modelLoader.detect(bitmap)
    Log.d("detectObjects", "Detection Results: $results") // Debug log for detection results
    onResults(results)
}

