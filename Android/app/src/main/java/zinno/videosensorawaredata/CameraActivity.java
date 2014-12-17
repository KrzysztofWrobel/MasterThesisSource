package zinno.videosensorawaredata;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.Camera;
import android.hardware.Camera.PictureCallback;
import android.hardware.Camera.ShutterCallback;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.opengl.Matrix;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.Surface;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.google.gson.Gson;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.LinkedList;

public class CameraActivity extends Activity implements SensorEventListener {

    public static final String INTERNAT_EMULATED_SDCARD_DIR = "sdcard/";
    public static final double WALKING_MAX_VELOCITY = 1.5;
    static final float ALPHA = 0.95f; // if ALPHA = 1 OR 0, no filter applies.
    private Camera mCamera;
    private CameraSurface mPreview;
    private ViewGroup previewView;
    private boolean continousMode;
    private CameraState mPreviewState = CameraState.K_STATE_PREVIEW;
    private SensorManager mSensorManager;
    private Sensor mOrientation;
    private Sensor mLinearAcceleration;
    private TextView yawTextView;
    private TextView pitchTextView;
    private TextView rollTextView;
    private int mCameraId;
    private String currentDateandTime;
    private int currentPhotoIndex;
    private float[] currentRotationMatrix;
    private float roll_angle;
    private float currentRoll;
    private float currentAzimuth;
    private float currentPitch;
    private LinkedList<SensorPhotoCaptur> photosSensorData;
    private TextView accelGlobalXTextView;
    private TextView accelGlobalYTextView;
    private TextView accelGlobalZTextView;
    private long lastLinearAccelerationEvent = -1;
    private float[] currentGlobalVelocity;
    private float[] currentRelativePosition;
    private LinkedList<GlobalCoordinate> globalCoordinateArraylist;
    private float[] currentGlobalAcceleration;
    private TextView velociGlobalZTextView;
    private TextView velociGlobalXTextView;
    private TextView velociGlobalYTextView;
    private TextView posGlobalXTextView;
    private TextView posGlobalYTextView;
    private TextView posGlobalZTextView;
    private State deviceState;
    private long movingStartTime;
    private long movingEndTime;
    private Handler mHandler;
    private long counter = 0;
    private float[] invertedActualMatrix;
    private Button resetButton;
    private LinkedList<float[]> accelerationEvents;
    private Sensor mStepCounter;
    private int stepCount;
    private TextView stepsTextView;

    public static byte[] rawDataToJpgData(byte[] inputData) {
        Bitmap image = BitmapFactory.decodeByteArray(inputData, 0, inputData.length);
        if (image != null) {
            ByteArrayOutputStream stream = new ByteArrayOutputStream();
            image.compress(Bitmap.CompressFormat.JPEG, 100, stream);
            return stream.toByteArray();
        }
        return null;
    }

    public static void setCameraDisplayOrientation(Activity activity,
                                                   int cameraId, android.hardware.Camera camera) {
        android.hardware.Camera.CameraInfo info =
                new android.hardware.Camera.CameraInfo();
        android.hardware.Camera.getCameraInfo(cameraId, info);
        int rotation = activity.getWindowManager().getDefaultDisplay()
                .getRotation();
        int degrees = 0;
        switch (rotation) {
            case Surface.ROTATION_0:
                degrees = 0;
                break;
            case Surface.ROTATION_90:
                degrees = 90;
                break;
            case Surface.ROTATION_180:
                degrees = 180;
                break;
            case Surface.ROTATION_270:
                degrees = 270;
                break;
        }

        int result;
        if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
            result = (info.orientation + degrees) % 360;
            result = (360 - result) % 360;  // compensate the mirror
        } else {  // back-facing
            result = (info.orientation - degrees + 360) % 360;
        }
        camera.setDisplayOrientation(result);
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        initCameraView();

        continousMode = true;

        SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_HHmmss");
        currentDateandTime = sdf.format(new Date());

        String folderName = INTERNAT_EMULATED_SDCARD_DIR + currentDateandTime;
        File theDir = new File(folderName);

        // if the directory does not exist, create it
        if (!theDir.exists()) {
            System.out.println("creating directory: " + folderName);
            boolean result = false;

            try {
                theDir.mkdir();
                result = true;
            } catch (SecurityException se) {
                //handle it
            }
            if (result) {
                System.out.println("DIR created");
            }
        }

        currentPhotoIndex = 0;
        photosSensorData = new LinkedList<SensorPhotoCaptur>();

        //Acceleration stuff
        currentGlobalAcceleration = new float[4];
        currentGlobalVelocity = new float[3];
        currentRelativePosition = new float[3];
        lastLinearAccelerationEvent = -1;
        globalCoordinateArraylist = new LinkedList<GlobalCoordinate>();
        globalCoordinateArraylist.add(new GlobalCoordinate(0, 0, 0));
        deviceState = State.IDLE;
        accelerationEvents = new LinkedList<float[]>();

        //Orientation stuff
        invertedActualMatrix = new float[16];

        yawTextView = (TextView) findViewById(R.id.tv_yaw);
        pitchTextView = (TextView) findViewById(R.id.tv_pitch);
        rollTextView = (TextView) findViewById(R.id.tv_roll);

        accelGlobalXTextView = (TextView) findViewById(R.id.tv_accel_globalX);
        accelGlobalYTextView = (TextView) findViewById(R.id.tv_accel_globalY);
        accelGlobalZTextView = (TextView) findViewById(R.id.tv_accel_globalZ);

        velociGlobalXTextView = (TextView) findViewById(R.id.tv_veloci_globalX);
        velociGlobalYTextView = (TextView) findViewById(R.id.tv_veloci_globalY);
        velociGlobalZTextView = (TextView) findViewById(R.id.tv_veloci_globalZ);

        posGlobalXTextView = (TextView) findViewById(R.id.tv_pos_globalX);
        posGlobalYTextView = (TextView) findViewById(R.id.tv_pos_globalY);
        posGlobalZTextView = (TextView) findViewById(R.id.tv_pos_globalZ);

        stepsTextView = (TextView) findViewById(R.id.tv_steps);

        resetButton = (Button) findViewById(R.id.b_reset);
        resetButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                resetRelativePosition();
            }
        });

        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        mOrientation = mSensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);
        mLinearAcceleration = mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        mStepCounter = mSensorManager.getDefaultSensor(Sensor.TYPE_STEP_DETECTOR);

        mHandler = new Handler();
    }

    private void resetRelativePosition() {
        currentRelativePosition[0] = 0;
        currentRelativePosition[1] = 0;
        currentRelativePosition[2] = 0;
    }

    @Override
    protected void onResume() {
        super.onResume();
        mSensorManager.registerListener(this, mOrientation, SensorManager.SENSOR_DELAY_GAME);
        mSensorManager.registerListener(this, mLinearAcceleration, SensorManager.SENSOR_DELAY_GAME);
        mSensorManager.registerListener(this, mStepCounter, SensorManager.SENSOR_DELAY_FASTEST);
    }

    @Override
    protected void onPause() {
        super.onPause();
        mSensorManager.unregisterListener(this);
        final String path = currentDateandTime + "/sensor.txt";
        String jsonFile = new Gson().toJson(photosSensorData);
        new SaveDataAsyncTask(path).execute(jsonFile.getBytes());
    }

    @Override
    protected void onStop() {
        super.onStop();

    }

    private void initCameraView() {
        previewView = (ViewGroup) findViewById(R.id.content);
        previewView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                switch (mPreviewState) {
                    case K_STATE_FROZEN:
                        mCamera.startPreview();
                        mPreviewState = CameraState.K_STATE_PREVIEW;
                        break;

                    default:
                        final String path = currentDateandTime + String.format("/%d.jpg", currentPhotoIndex);
                        currentPhotoIndex++;
                        mCamera.takePicture(new ShutterCallback() {

                            @Override
                            public void onShutter() {
                                GlobalCoordinate last = globalCoordinateArraylist.getLast();
                                GlobalCoordinate newPos = last.add(currentRelativePosition[0], currentRelativePosition[1], currentRelativePosition[2]);
                                globalCoordinateArraylist.add(newPos);

                                SensorPhotoCaptur sensorPhotoCaptur = new SensorPhotoCaptur(currentPhotoIndex, path, currentAzimuth, currentPitch, currentRoll, newPos.X, newPos.Y, newPos.Z, currentRotationMatrix.clone());
                                photosSensorData.add(sensorPhotoCaptur);

                                resetRelativePosition();
                            }
                        }, null, new PictureCallback() {

                            @Override
                            public void onPictureTaken(byte[] data, android.hardware.Camera camera) {
                                new SaveDataAsyncTask(path).execute(data);
                                Toast.makeText(getBaseContext(), "Picture taken", Toast.LENGTH_SHORT).show();
                                if (!continousMode) {
                                    mPreviewState = CameraState.K_STATE_FROZEN;
                                } else {
                                    mCamera.startPreview();
                                    mPreviewState = CameraState.K_STATE_PREVIEW;
                                }
                            }
                        });
                        mPreviewState = CameraState.K_STATE_BUSY;
                }
            }
        });
        mPreview = (CameraSurface) findViewById(R.id.sv_camera_surface);

        if (mCamera != null) {
            mCamera.release();
        }
//        mCamera = Camera.open();

        mCameraId = -1;
        int numberOfCameras = Camera.getNumberOfCameras();
        Camera.CameraInfo cameraInfo = new Camera.CameraInfo();
        for (int i = 0; i < numberOfCameras; i++) {
            Camera.getCameraInfo(i, cameraInfo);
            if (cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_BACK) {
                mCamera = Camera.open(i);
                mCameraId = i;
            }
        }
        if (mCamera != null) {
            mPreview.setCamera(mCamera);
            setCameraDisplayOrientation(this, mCameraId, mCamera);
        }
    }

    @Override
    public void onBackPressed() {
        Toast.makeText(getApplicationContext(), "Go Back", Toast.LENGTH_SHORT).show();
        super.onBackPressed();
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor == mOrientation) {
            float[] orientation = rotationVectorAction(event.values);

            float azimuth_angle = (float) Math.toDegrees(orientation[0]);
            float pitch_angle = (float) Math.toDegrees(orientation[1]);
            float roll_angle = (float) Math.toDegrees(orientation[2]);

            yawTextView.setText(String.format("Azimuth: %f", azimuth_angle));
            pitchTextView.setText(String.format("Pitch: %f", pitch_angle));
            rollTextView.setText(String.format("Roll: %f", roll_angle));
            currentAzimuth = azimuth_angle;
            currentPitch = pitch_angle;
            currentRoll = roll_angle;
        } else if (event.sensor == mLinearAcceleration && currentRotationMatrix != null) { //TODO currentRotation can be null
            //TODO we should have accelInit, orientationInit
            accelerationEvents.add(event.values.clone());
            if (accelerationEvents.size() > 3) {
                accelerationEvents.removeFirst();
            } else if (accelerationEvents.size() < 3) {
                return;
            }

            if (lastLinearAccelerationEvent == -1) {
                lastLinearAccelerationEvent = event.timestamp;
                return;
            }
            double dT = (double) (event.timestamp - lastLinearAccelerationEvent) / 1000000000d; //timestamp in nanoseconds
            lastLinearAccelerationEvent = event.timestamp;

            float[] newGlobalAcceleration = new float[4];
            for (int i = 0; i < 3; i++) {
                newGlobalAcceleration[i] = (accelerationEvents.get(0)[i] + accelerationEvents.get(1)[i] + accelerationEvents.get(2)[i]) / 3;
            }
            //TODO synchronize invertedActualMatrix
            Matrix.multiplyMV(newGlobalAcceleration, 0, invertedActualMatrix.clone(), 0, newGlobalAcceleration, 0);

            //TODO we should check idle variance, when user is staying idle and device is lying on the ground

//            newGlobalAcceleration = lowPass(newGlobalAcceleration, currentGlobalAcceleration);
            currentGlobalAcceleration = newGlobalAcceleration;

            double distance = getLength(currentGlobalAcceleration);
            long currentTimeMillis = System.currentTimeMillis();
            if (distance > 0.5 && deviceState == State.IDLE) {
                if (currentTimeMillis - movingEndTime > 300) {
                    deviceState = State.MOVING;
                    movingStartTime = currentTimeMillis;
                    currentGlobalVelocity[0] = 0;
                    currentGlobalVelocity[1] = 0;
                    currentGlobalVelocity[2] = 0;
//                    currentRelativePosition[0] = 0;
//                    currentRelativePosition[1] = 0;
//                    currentRelativePosition[2] = 0;
//                    Toast.makeText(CameraActivity.this, "Device moving", Toast.LENGTH_SHORT).show();
                }
            } else if (distance < 0.1 && deviceState == State.MOVING) {
                if (currentTimeMillis - movingStartTime > 350) {
                    deviceState = State.IDLE;
                    movingEndTime = currentTimeMillis;
//                    currentGlobalVelocity[0] = 0;
//                    currentGlobalVelocity[1] = 0;
//                    currentGlobalVelocity[2] = 0;
//                    Toast.makeText(CameraActivity.this, "Device stopping", Toast.LENGTH_SHORT).show();
                }
            }

            if (deviceState == State.MOVING) {
                currentGlobalVelocity[0] += currentGlobalAcceleration[0] * dT;
                currentGlobalVelocity[1] += currentGlobalAcceleration[2] * dT;
                currentGlobalVelocity[2] += currentGlobalAcceleration[1] * dT;
                double velocity = getLength(currentGlobalVelocity);
                if (velocity > WALKING_MAX_VELOCITY) {
                    currentGlobalVelocity[0] /= velocity / WALKING_MAX_VELOCITY;
                    currentGlobalVelocity[1] /= velocity / WALKING_MAX_VELOCITY;
                    currentGlobalVelocity[2] /= velocity / WALKING_MAX_VELOCITY;
                }
                currentRelativePosition[0] += currentGlobalVelocity[0] * dT + currentGlobalAcceleration[0] * dT * dT/2; // s = V0 * t + a * t^2/2a
                currentRelativePosition[1] += currentGlobalVelocity[1] * dT + currentGlobalAcceleration[2] * dT * dT/2;
                currentRelativePosition[2] += currentGlobalVelocity[2] * dT + + currentGlobalAcceleration[1] * dT * dT/2;
            }


            if (counter++ % 20 == 0)
                mHandler.post(new Runnable() {
                    @Override
                    public void run() {
                        accelGlobalXTextView.setText(String.format("AccelX: %f" + (deviceState == State.MOVING ? "m" : ""), currentGlobalAcceleration[0]));
                        accelGlobalYTextView.setText(String.format("AccelY: %f", currentGlobalAcceleration[1]));
                        accelGlobalZTextView.setText(String.format("AccelZ: %f", currentGlobalAcceleration[2]));

                        velociGlobalXTextView.setText(String.format("VelociX: %f", currentGlobalVelocity[0]));
                        velociGlobalYTextView.setText(String.format("VelociY: %f", currentGlobalVelocity[1]));
                        velociGlobalZTextView.setText(String.format("VelociZ: %f", currentGlobalVelocity[2]));

                        posGlobalXTextView.setText(String.format("PosX: %f", currentRelativePosition[0]));
                        posGlobalYTextView.setText(String.format("PosY: %f", currentRelativePosition[1]));
                        posGlobalZTextView.setText(String.format("PosZ: %f", currentRelativePosition[2]));
                    }
                });

        } else if (event.sensor == mStepCounter) {
            stepCount++;
            stepsTextView.setText("St: " + stepCount);
        }
    }

    public double getDistance(float[] vector1, float[] vector2) {
        //TODO check if vectors are the same size
        float sum = 0;
        for (int i = 0; i < vector1.length; i++) {
            sum += (vector2[i] - vector1[i]) * (vector2[i] - vector1[i]);
        }
        return Math.sqrt(sum);
    }

    public double getLength(float[] vector) {
        //TODO check if vectors are the same size
        float sum = 0;
        for (int i = 0; i < vector.length; i++) {
            sum += (vector[i] * vector[i]);
        }
        return Math.sqrt(sum);
    }

    protected float[] lowPass(float[] input, float[] output) {
        if (output == null) return input;
        for (int i = 0; i < input.length; i++) {
            output[i] = output[i] + ALPHA * (input[i] - output[i]);
        }
        return output;
    }

    private float[] rotationVectorAction(float[] values) {
        float[] orientation = new float[3];
        float[] rotMat = new float[16];
        SensorManager.getRotationMatrixFromVector(rotMat, values);
        Matrix.invertM(invertedActualMatrix, 0, rotMat, 0);
        SensorManager.remapCoordinateSystem(rotMat, SensorManager.AXIS_MINUS_Z, SensorManager.AXIS_MINUS_X, rotMat);
        SensorManager.getOrientation(rotMat, orientation);
        currentRotationMatrix = rotMat;
        return orientation;
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    private enum State {
        MOVING,
        IDLE
    }

    enum CameraState {
        K_STATE_PREVIEW,
        K_STATE_BUSY,
        K_STATE_FROZEN
    }

    private static class GlobalCoordinate {
        private double X;
        private double Y;
        private double Z;

        private GlobalCoordinate(double x, double y, double z) {
            X = x;
            Y = y;
            Z = z;
        }

        public GlobalCoordinate add(double x, double y, double z) {
            return new GlobalCoordinate(X + x, Y + y, Z + z);
        }
    }

    class SaveDataAsyncTask extends AsyncTask<byte[], String, String> {
        private final String relativePath;

        SaveDataAsyncTask(String relativePath) {
            this.relativePath = relativePath;
        }

        @Override
        protected String doInBackground(byte[]... data) {

            File newFile = new File(INTERNAT_EMULATED_SDCARD_DIR + relativePath);
            if (newFile.exists()) {
                newFile.delete();
            }
            try {
                newFile.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }


            try {
                FileOutputStream fos = new FileOutputStream(newFile);

                fos.write(data[0]);
                fos.close();
            } catch (java.io.IOException e) {
                Log.e("PictureDemo", "Exception in photoCallback", e);
            }

            return (null);
        }
    }

    class SensorPhotoCaptur {
        private int photoId;
        private String photoPath;
        private float azimuth;
        private float pitch;
        private float roll;
        private double posX;
        private double posY;
        private double posZ;
        private float[] rotationMatrix;

        SensorPhotoCaptur(int photoId, String photoPath, float azimuth, float pitch, float roll, double posX, double posY, double posZ, float[] rotationMatrix) {
            this.photoId = photoId;
            this.photoPath = photoPath;
            this.azimuth = azimuth;
            this.pitch = pitch;
            this.roll = roll;
            this.posX = posX;
            this.posY = posY;
            this.posZ = posZ;
            this.rotationMatrix = rotationMatrix;
        }
    }

}