package cn.yege.dshcam

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlin.math.PI

/**
 * 传感器姿态检测器
 * 负责获取设备的横滚角和俯仰角
 */
class SensorPose(context: Context) : SensorEventListener {

    private val sensorManager: SensorManager =
        context.applicationContext.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    // 优先使用旋转矢量传感器，若不可用则退而求其次使用游戏旋转矢量传感器
    private val sensor: Sensor? =
        sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
            ?: sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)

    // 缓存角度值，使用 @Volatile 保证多线程可见性
    @Volatile
    private var roll: Float = 0f
    @Volatile
    private var pitch: Float = 0f

    /**
     * 启动传感器监听
     */
    fun start() {
        sensor?.let {
            // 使用 SENSOR_DELAY_UI 级别即可满足 UI 更新需求，且更省电
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_UI)
        }
    }

    /**
     * 停止传感器监听
     */
    fun stop() {
        sensorManager.unregisterListener(this)
    }

    /**
     * 获取横滚角（度），正值表示右倾
     */
    fun rollDeg(): Float = roll

    /**
     * 获取俯仰角（度）
     */
    fun pitchDeg(): Float = pitch

    override fun onSensorChanged(event: SensorEvent) {
        // 获取旋转矩阵
        val rMatrix = FloatArray(9)
        SensorManager.getRotationMatrixFromVector(rMatrix, event.values)

        // 处理竖屏情况，重映射坐标系
        // 竖屏下，设备的 Y 轴朝上，需要将其映射为世界的 Z 轴朝上
        val outR = FloatArray(9)
        SensorManager.remapCoordinateSystem(
            rMatrix,
            SensorManager.AXIS_X,
            SensorManager.AXIS_Z,
            outR
        )

        // 获取方位角、俯仰角和横滚角（弧度）
        val orientation = FloatArray(3)
        SensorManager.getOrientation(outR, orientation)

        // 转换为角度
        // orientation[1] 为 pitch，正值表示顶部抬起（仰）
        // orientation[2] 为 roll，正值表示左侧抬起（左倾），所以右倾需要取反
        pitch = Math.toDegrees(orientation[1].toDouble()).toFloat()
        roll = -Math.toDegrees(orientation[2].toDouble()).toFloat()
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // 精度变化时无需特殊处理
    }
}
