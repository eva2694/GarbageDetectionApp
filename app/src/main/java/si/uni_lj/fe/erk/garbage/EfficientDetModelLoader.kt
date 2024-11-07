package si.uni_lj.fe.erk.garbage

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector

/**
 * EfficientDetModelLoader class to handle object detection using TensorFlow Lite models.
 *
 * @property threshold Detection score threshold.
 * @property numThreads Number of threads to use for inference.
 * @property maxResults Maximum number of detection results to return.
 * @property currentDelegate Delegate to use for inference (0: CPU, 1: GPU, 2: NNAPI).
 * @property currentModel Model to use for detection (1: EfficientDet-Lite0, 2: EfficientDet-Lite1).
 * @property context Android context.
 * @property objectDetectorListener Listener to handle detection results and errors.
 */
class EfficientDetModelLoader(
    private var threshold: Float = 0.5f,
    private var numThreads: Int = 2,
    private var maxResults: Int = 3,
    private var currentDelegate: Int = 0,
    var currentModel: Int = 0,
    val context: Context,
    val objectDetectorListener: DetectorListener?
) {

    private var objectDetector: ObjectDetector? = null

    init {
        setupObjectDetector()
    }

    /**
     * Sets up the object detector with the specified options.
     */
    private fun setupObjectDetector() {
        val optionsBuilder = ObjectDetector.ObjectDetectorOptions.builder()
            .setScoreThreshold(threshold)
            .setMaxResults(maxResults)

        val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)
        val compatList = CompatibilityList()
        // delegate configuration
        when (currentDelegate) {
            0 -> Log.d("EfficientDetModelLoader", "Using CPU for inference")
            1 -> {
                if (compatList.isDelegateSupportedOnThisDevice) {
                    Log.d("EfficientDetModelLoader", "Using GPU for inference")
                    baseOptionsBuilder.useGpu()
                } else {
                    val errorMsg = "GPU is not supported on this device"
                    Log.e("EfficientDetModelLoader", errorMsg)
                    objectDetectorListener?.onError(errorMsg)
                }
            }
            2 -> {
                Log.d("EfficientDetModelLoader", "Using NNAPI for inference")
                baseOptionsBuilder.useNnapi()
            }
        }

        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        val modelName = when (currentModel) {
            1 -> "EfficientDet-Lite0.tflite"
            2 -> "EfficientDet-Lite2.tflite"
            3 -> "EfficientDet-Lite4.tflite"
            else -> "EfficientDet-Lite0.tflite"
        }

        // load chosen model
        try {
            objectDetector = ObjectDetector.createFromFileAndOptions(context, modelName, optionsBuilder.build())
            Log.d("EfficientDetModelLoader", "Model $modelName loaded successfully")
        } catch (e: IllegalStateException) {
            objectDetectorListener?.onError("Object detector failed to initialize. See error logs for details")
            Log.e("EfficientDetModelLoader", "TFLite failed to load model with error: ${e.message}")
        }
    }

    /**
     * Detects objects in the given bitmap image.
     *
     * @param image Bitmap image to perform detection on.
     */
    fun detect(image: Bitmap) {
        if (objectDetector == null) {
            Log.w("EfficientDetModelLoader", "Object detector is not initialized, re-initializing")
            setupObjectDetector()
        }

        var inferenceTime = SystemClock.uptimeMillis()

        // preprocess
        val imageProcessor = ImageProcessor.Builder().build()
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))

        // detect
        val results = objectDetector?.detect(tensorImage)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        objectDetectorListener?.onResults(results, inferenceTime, tensorImage.height, tensorImage.width)
    }

    /**
     * Listener interface for handling detection results and errors.
     */
    interface DetectorListener {
        /**
         * Called when an error occurs.
         *
         * @param error Error message.
         */
        fun onError(error: String)

        /**
         * Called when detection results are available.
         *
         * @param results List of detection results.
         * @param inferenceTime Time taken for inference.
         * @param imageHeight Height of the processed image.
         * @param imageWidth Width of the processed image.
         */
        fun onResults(
            results: MutableList<Detection>?,
            inferenceTime: Long,
            imageHeight: Int,
            imageWidth: Int
        )
    }
}
