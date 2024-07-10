package si.uni_lj.fe.erk.roadsigns

import android.Manifest
import android.graphics.*
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.support.image.TensorImage
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
                            detectObjects(bitmap, tfliteModelLoader) { results ->
                                detectionResults = results
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

        detectionResults.forEach { result ->
            Canvas(modifier = Modifier.fillMaxSize()) {
                drawRect(
                    color = Color.Red,
                    topLeft = androidx.compose.ui.geometry.Offset(result.x1 * size.width, result.y1 * size.height),
                    size = androidx.compose.ui.geometry.Size((result.x2 - result.x1) * size.width, (result.y2 - result.y1) * size.height),
                    style = androidx.compose.ui.graphics.drawscope.Stroke(width = 2f)
                )
                drawContext.canvas.nativeCanvas.apply {
                    drawText(
                        "${result.clsName} ${"%.2f".format(result.cnf)}",
                        result.x1 * size.width,
                        result.y1 * size.height - 10,
                        android.graphics.Paint().apply {
                            color = android.graphics.Color.RED
                            textSize = 30f
                        }
                    )
                }
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
    onResults(results)
}

