package si.uni_lj.fe.erk.garbage

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader

/**
 * YoloModelLoader class for loading and running YOLO models using TensorFlow Lite.
 *
 * @property context Android context.
 * @param modelPath Path to the model file.
 */
class YoloModelLoader(private val context: Context, modelPath: String) {
    private val labelPath = "labels.txt"
    private var interpreter: Interpreter? = null
    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0
    private var labels = mutableListOf<String>()
    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.3F
        private const val IOU_THRESHOLD = 0.5F
    }

    init {
        initializeInterpreter(modelPath)
        loadLabels()
    }

    /**
     * Initializes the TensorFlow Lite interpreter with the specified model path.
     *
     * @param modelPath Path to the yolo model file.
     */
    private fun initializeInterpreter(modelPath: String) {
        val compatList = CompatibilityList()
        try {
            val options = Interpreter.Options().apply {
                if (compatList.isDelegateSupportedOnThisDevice) {
                    val delegateOptions = compatList.bestOptionsForThisDevice
                    this.addDelegate(GpuDelegate(delegateOptions))
                } else {
                    this.setNumThreads(4)
                }
            }
            val model = FileUtil.loadMappedFile(context, modelPath)

            interpreter = Interpreter(model, options)

            val inputShape = interpreter?.getInputTensor(0)?.shape()
            val outputShape = interpreter?.getOutputTensor(0)?.shape()

            tensorWidth = inputShape?.get(1) ?: 0
            tensorHeight = inputShape?.get(2) ?: 0
            numChannel = outputShape?.get(1) ?: 0
            numElements = outputShape?.get(2) ?: 0

            Log.d("YoloModelLoader", "Interpreter initialized successfully with model: $modelPath")

        } catch (e: Exception) {
            Log.e("YoloModelLoader", "Error initializing interpreter", e)
        }
    }

    /**
     * Loads the labels from the asset file.
     */
    private fun loadLabels() {
        try {
            val inputStream: InputStream = context.assets.open(labelPath)
            val reader = BufferedReader(InputStreamReader(inputStream))
            var line: String? = reader.readLine()
            while (!line.isNullOrEmpty()) {
                labels.add(line)
                line = reader.readLine()
            }
            reader.close()
            inputStream.close()
            Log.d("YoloModelLoader", "Labels loaded successfully from $labelPath")
        } catch (e: IOException) {
            Log.e("YoloModelLoader", "Error loading labels", e)
            e.printStackTrace()
        }
    }
    /**
     * Detects objects in the given bitmap using the chosen YOLO model.
     *
     * @param bitmap Bitmap image to perform detection on.
     * @return List of detected bounding boxes.
     */
    fun detect(bitmap: Bitmap): List<BoundingBox> {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, tensorWidth, tensorHeight, false)

        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

        val output = TensorBuffer.createFixedSize(intArrayOf(1, numChannel, numElements), OUTPUT_IMAGE_TYPE)
        interpreter?.run(imageBuffer, output.buffer)

        return bestBox(output.floatArray) ?: listOf()
    }

    /**
     * Saves the processed bitmap for debugging purposes. It is not used
     *
     * @param bitmap Bitmap image to save.
     */
    fun saveProcessedBitmap(bitmap: Bitmap) {
        val filename = "processed_image_${System.currentTimeMillis()}.jpg"
        val file = File(context.getExternalFilesDir(null), filename)
        try {
            val outputStream = FileOutputStream(file)
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream)
            outputStream.flush()
            outputStream.close()
            Log.d("YoloModelLoader", "Processed bitmap saved as $filename")
        } catch (e: IOException) {
            Log.e("YoloModelLoader", "Failed to save processed image", e)
        }
    }

    /**
     * Finds the best bounding boxes based on confidence threshold.
     *
     * @param array Array of float values representing detected objects.
     * @return List of bounding boxes.
     */
    private fun bestBox(array: FloatArray): List<BoundingBox>? {
        val boundingBoxes = mutableListOf<BoundingBox>()

        for (c in 0 until numElements) {
            var maxConf = -1.0f
            var maxIdx = -1
            var j = 4
            var arrayIdx = c + numElements * j
            while (j < numChannel) {
                if (array[arrayIdx] > maxConf) {
                    maxConf = array[arrayIdx]
                    maxIdx = j - 4
                }
                j++
                arrayIdx += numElements
            }

            if (maxConf > CONFIDENCE_THRESHOLD) {
                val clsName = labels[maxIdx]
                val cx = array[c]
                val cy = array[c + numElements]
                val w = array[c + numElements * 2]
                val h = array[c + numElements * 3]
                val x1 = cx - (w / 2F)
                val y1 = cy - (h / 2F)
                val x2 = cx + (w / 2F)
                val y2 = cy + (h / 2F)
                if (x1 < 0F || x1 > 1F) continue
                if (y1 < 0F || y1 > 1F) continue
                if (x2 < 0F || x2 > 1F) continue
                if (y2 < 0F || y2 > 1F) continue

                boundingBoxes.add(
                    BoundingBox(
                        x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                        cx = cx, cy = cy, w = w, h = h,
                        cnf = maxConf, cls = maxIdx, clsName = clsName
                    )
                )
            }
        }

        if (boundingBoxes.isEmpty()) return null

        val nmsBoxes = applyNMS(boundingBoxes)
        Log.d("YoloModelLoader", "Non-max suppression applied, ${nmsBoxes.size} boxes selected")
        return nmsBoxes
    }
    /**
     * Applies non-maximum suppression to filter bounding boxes.
     *
     * @param boxes List of bounding boxes.
     * @return List of filtered bounding boxes.
     */
    private fun applyNMS(boxes: List<BoundingBox>): MutableList<BoundingBox> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<BoundingBox>()

        while (sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.first()
            selectedBoxes.add(first)
            sortedBoxes.remove(first)

            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val nextBox = iterator.next()
                val iou = calculateIoU(first, nextBox)
                if (iou >= IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }

        return selectedBoxes
    }
    /**
     * Calculates the Intersection over Union (IoU) of two bounding boxes.
     *
     * @param box1 First bounding box.
     * @param box2 Second bounding box.
     * @return IoU value.
     */
    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)
        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    /**
     * Data class representing a bounding box for detected objects.
     *
     * @property x1 Top-left x-coordinate.
     * @property y1 Top-left y-coordinate.
     * @property x2 Bottom-right x-coordinate.
     * @property y2 Bottom-right y-coordinate.
     * @property cx Center x-coordinate.
     * @property cy Center y-coordinate.
     * @property w Width of the bounding box.
     * @property h Height of the bounding box.
     * @property cnf Confidence score.
     * @property cls Class index.
     * @property clsName Class name.
     */
    data class BoundingBox(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val cx: Float,
        val cy: Float,
        val w: Float,
        val h: Float,
        val cnf: Float,
        val cls: Int,
        val clsName: String
    )
}
