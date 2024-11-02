package si.uni_lj.fe.erk.garbage

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import com.google.android.gms.tflite.java.TfLite
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await
import si.uni_lj.fe.erk.garbage.ui.theme.GarbageTheme

/**
 * Main activity of the application.
 */
class MainActivity : ComponentActivity() {
    private lateinit var cameraExecutor: ExecutorService
    /**
     * Called when the activity is initializing
     *
     * @param savedInstanceState If the activity is being re-initialized after previously being shut down then this Bundle contains the data it most recently supplied in [onSaveInstanceState].
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()
        Log.d("MainActivity", "Camera executor initialized")

        CoroutineScope(Dispatchers.Main).launch {
            try {
                TfLite.initialize(this@MainActivity).await()
                Log.d("MainActivity", "TensorFlow Lite initialized successfully")
                setContent {
                    GarbageTheme {
                        Surface(color = MaterialTheme.colorScheme.background) {
                            RequestCameraPermission {
                                CameraPreviewScreen(cameraExecutor)
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "Error initializing TensorFlow Lite", e)
                e.printStackTrace()
            }
        }
    }

    /**
     * Perform any final cleanup before an activity is destroyed.
     */
    override fun onDestroy() {
        super.onDestroy()
        Log.d("MainActivity", "Shutting down camera executor")
        cameraExecutor.shutdown()
        Log.d("MainActivity", "Camera executor shut down")
    }
}
