package si.uni_lj.fe.erk.roadsigns

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import si.uni_lj.fe.erk.roadsigns.ui.theme.RoadSignsTheme
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import com.google.android.gms.tflite.java.TfLite
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await

class MainActivity : ComponentActivity() {
    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()

        CoroutineScope(Dispatchers.Main).launch {
            try {
                TfLite.initialize(this@MainActivity).await()
                setContent {
                    RoadSignsTheme {
                        // A surface container using the 'background' color from the theme
                        Surface(color = MaterialTheme.colorScheme.background) {
                            RequestCameraPermission {
                                CameraPreviewScreen(cameraExecutor)
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}
